use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{Mmcs, OpenedValues, Pcs};
use p3_field::extension::{Complex, ComplexExtendable, HasFrobenius};
use p3_field::{batch_multiplicative_inverse, AbstractField, ExtensionField, Field};
use p3_fri::{FriConfig, FriProof, PowersReducer};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::permuted::PermutedMatrix;
use p3_matrix::routines::columnwise_dot_product;
use p3_matrix::{Matrix, MatrixRows};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::cfft::Cfft;
use crate::deep_quotient::{extract_lambda, is_low_degree};
use crate::domain::CircleDomain;
use crate::folding::{
    circle_bitrev_permute, fold_bivariate, CircleBitrevInvPermutation, CircleBitrevPermutation,
    CircleFriFolder,
};
use crate::util::{univariate_to_point, v_n};

pub struct CirclePcs<Val, InputMmcs, FriMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
    pub fri_config: FriConfig<FriMmcs>,
}

pub struct ProverData<Val, MmcsData> {
    committed_domains: Vec<CircleDomain<Val>>,
    mmcs_data: MmcsData,
}

impl<Val, InputMmcs, FriMmcs, Challenge, Challenger> Pcs<Challenge, Challenger>
    for CirclePcs<Val, InputMmcs, FriMmcs>
where
    Val: ComplexExtendable,
    InputMmcs: 'static + Mmcs<Val>,
    FriMmcs: Mmcs<Challenge>,
    Challenge: ExtensionField<Val> + HasFrobenius<Val>,
    Challenger: GrindingChallenger + CanSample<Challenge> + CanObserve<FriMmcs::Commitment>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = ProverData<Val, InputMmcs::ProverData<RowMajorMatrix<Val>>>;

    // TEMP: pass through reduced query openings
    type Proof = (
        FriProof<Challenge, FriMmcs, Challenger::Witness>,
        Vec<[Challenge; 32]>,
    );
    type Error = ();

    fn natural_domain_for_degree(&self, degree: usize) -> Self::Domain {
        CircleDomain::standard(log2_strict_usize(degree))
    }

    fn commit(
        &self,
        evaluations: Vec<(Self::Domain, RowMajorMatrix<Val>)>,
    ) -> (Self::Commitment, Self::ProverData) {
        let (committed_domains, ldes): (Vec<_>, Vec<_>) = evaluations
            .into_iter()
            .map(|(domain, evals)| {
                let committed_domain = CircleDomain::standard(domain.log_n + self.log_blowup);
                // bitrev for fri?
                let lde = self.cfft.lde(evals, domain, committed_domain);

                // let lde = PermutedMatrix::<CircleBitrevPermutation, _>::new(lde).to_row_major_matrix();

                (committed_domain, lde)
            })
            .unzip();
        let (comm, mmcs_data) = self.mmcs.commit(ldes);
        (
            comm,
            ProverData {
                committed_domains,
                mmcs_data,
            },
        )
    }

    fn get_evaluations_on_domain(
        &self,
        data: &Self::ProverData,
        idx: usize,
        domain: Self::Domain,
    ) -> RowMajorMatrix<Val> {
        // TODO do this correctly
        let mat = self.mmcs.get_matrices(&data.mmcs_data)[idx];
        assert_eq!(mat.height(), 1 << domain.log_n);
        assert_eq!(domain, data.committed_domains[idx]);
        // PermutedMatrix::<CircleBitrevInvPermutation, _>::new(mat).to_row_major_matrix()
        mat.clone().to_row_major_matrix()
    }

    #[instrument(skip_all)]
    fn open(
        &self,
        // For each round,
        rounds: Vec<(
            &Self::ProverData,
            // for each matrix,
            Vec<
                // points to open
                Vec<Challenge>,
            >,
        )>,
        challenger: &mut Challenger,
    ) -> (OpenedValues<Challenge>, Self::Proof) {
        // Batch combination challenge
        let mu: Challenge = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| (self.mmcs.get_matrices(&data.mmcs_data), points))
            .collect_vec();

        let max_width = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .map(|m| m.width())
            .max()
            .unwrap();

        let mut reduced_openings: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let mu_reducer = PowersReducer::<Val, Challenge>::new(mu, max_width);

        let values: OpenedValues<Challenge> = rounds
            .into_iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(domain, mat, points_for_mat)| {
                        let log_height = log2_strict_usize(mat.height());
                        let reduced_opening_for_log_height: &mut Vec<Challenge> = reduced_openings
                            [log_height]
                            .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                        points_for_mat
                            .into_iter()
                            .map(|zeta| {
                                let zeta_point = univariate_to_point(zeta).unwrap();
                                // todo: cache per domain
                                let basis: Vec<Challenge> = domain.lagrange_basis(zeta_point);

                                let v_n_at_zeta = v_n(zeta_point.real(), log_height)
                                    - v_n(domain.shift.real(), log_height);

                                let mu_pow_offset = mu.exp_u64(num_reduced[log_height] as u64);
                                let mu_pow_width = mu.exp_u64(mat.width() as u64);
                                num_reduced[log_height] += 2 * mat.width();

                                let (lhs_nums, lhs_denoms): (Vec<_>, Vec<_>) = domain
                                    .points()
                                    .map(|x| {
                                        let x_rotate_zeta: Complex<Challenge> =
                                            x.rotate(zeta_point.conjugate());

                                        let v_gamma_re: Challenge =
                                            Challenge::one() - x_rotate_zeta.real();
                                        let v_gamma_im: Challenge = x_rotate_zeta.imag();

                                        (
                                            v_gamma_re - mu_pow_width * v_gamma_im,
                                            v_gamma_re.square() + v_gamma_im.square(),
                                        )
                                    })
                                    .unzip();
                                let inv_lhs_denoms = batch_multiplicative_inverse(&lhs_denoms);

                                // todo: we only need half of the values to interpolate, but how?
                                let ps_at_zeta: Vec<Challenge> =
                                    info_span!("compute opened values with Lagrange interpolation")
                                        .in_scope(|| {
                                            columnwise_dot_product(
                                                &mat.as_view(),
                                                basis.into_iter(),
                                            )
                                            .into_iter()
                                            .map(|x| x * v_n_at_zeta)
                                            .collect()
                                        });

                                let mu_pow_ps_at_zeta = mu_reducer.reduce_ext(&ps_at_zeta);

                                info_span!(
                                    "reduce rows",
                                    log_height = log_height,
                                    width = mat.width()
                                )
                                .in_scope(|| {
                                    izip!(
                                        reduced_opening_for_log_height.par_iter_mut(),
                                        mat.rows(),
                                        lhs_nums,
                                        inv_lhs_denoms,
                                    )
                                    .for_each(
                                        |(reduced_opening, row, lhs_num, inv_lhs_denom)| {
                                            *reduced_opening += lhs_num
                                                * inv_lhs_denom
                                                * mu_pow_offset
                                                * (mu_reducer.reduce_base(row) - mu_pow_ps_at_zeta);
                                        },
                                    )
                                });

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // We do bivariate fold now, so can't have a singleton poly
        assert!(reduced_openings[0].is_none());
        // Do the first circle fold for all polys with the same beta
        let bivariate_beta: Challenge = challenger.sample();

        let fri_input: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|i| {
            let mut ro: Vec<Challenge> = reduced_openings.get(i + 1)?.as_ref()?.clone();
            let _lambda = extract_lambda(
                CircleDomain::standard(i + 1 - self.log_blowup),
                CircleDomain::standard(i + 1),
                &mut ro,
            );
            debug_assert!(is_low_degree(
                &RowMajorMatrix::new_col(ro.clone()).flatten_to_base()
            ));
            let ro_permuted = RowMajorMatrix::new(circle_bitrev_permute(&ro), 2);
            Some(fold_bivariate(ro_permuted, bivariate_beta))
        });

        let (fri_proof, query_indices) = p3_fri::prover::prove::<CircleFriFolder<Val>, _, _, _>(
            &self.fri_config,
            &fri_input,
            challenger,
        );

        // TEMP: pass through reduced openings
        let reduced_query_openings: Vec<[Challenge; 32]> = query_indices
            .into_iter()
            .map(|index| {
                core::array::from_fn(|i| {
                    fri_input[i]
                        .as_ref()
                        .map(|v| v[index])
                        .unwrap_or(Challenge::zero())
                })
            })
            .collect();

        (values, (fri_proof, reduced_query_openings))
    }

    fn verify(
        &self,
        // For each round:
        rounds: Vec<(
            Self::Commitment,
            // for each matrix:
            Vec<(
                // its domain,
                Self::Domain,
                // for each point:
                Vec<(
                    // the point,
                    Challenge,
                    // values at the point
                    Vec<Challenge>,
                )>,
            )>,
        )>,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let (fri_proof, reduced_openings) = proof;
        // Batch combination challenge
        let mu: Challenge = challenger.sample();
        let bivariate_beta: Challenge = challenger.sample();

        let fri_challenges = p3_fri::verifier::verify_shape_and_sample_challenges(
            &self.fri_config,
            &fri_proof,
            challenger,
        )
        .unwrap();

        p3_fri::verifier::verify_challenges::<CircleFriFolder<Val>, _, _, _>(
            &self.fri_config,
            &fri_proof,
            &fri_challenges,
            &reduced_openings,
        )
        .unwrap();

        Ok(())
    }
}

fn reduce_matrix<F: ComplexExtendable>(
    domain: CircleDomain<F>,
    p: RowMajorMatrix<F>,
    zeta: Complex<F>,
    ps_at_zeta: &[F],
    mu: F,
) -> Vec<F> {
    // let num = value - complex_conjugate_line(oods_point, oods_value, point);
    // let denom = pair_vanishing(oods_point, oods_point.complex_conjugate(), point.into_ef(

    p.rows()
        .zip(domain.points())
        .map(|(row, x)| {
            let inv_denom = (Complex::<F>::one() - zeta.conjugate().rotate(x)).inverse();
            izip!(row, ps_at_zeta, mu.square().powers())
                .map(|(&p_at_x, &p_at_zeta, mu2_pow)| {
                    let quotient = Complex::<F>::new_real(p_at_x - p_at_zeta) * inv_denom;
                    mu2_pow * (quotient.real() + mu * quotient.imag())
                })
                .sum()
        })
        .collect()
}
