use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::CanSample;
use p3_commit::{DirectMmcs, OpenedValues, Pcs};
use p3_field::extension::{Complex, ComplexExtendable, HasFrobenius};
use p3_field::{batch_multiplicative_inverse, AbstractField, ExtensionField, Field};
use p3_matrix::dense::{RowMajorMatrix, RowMajorMatrixView};
use p3_matrix::routines::columnwise_dot_product;
use p3_matrix::{Matrix, MatrixRows};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::cfft::Cfft;
use crate::deep_quotient::{frob_line, pair_vanishing};
use crate::domain::CircleDomain;
use crate::util::{univariate_to_point, v_n};

pub struct CirclePcs<Val, InputMmcs> {
    pub log_blowup: usize,
    pub cfft: Cfft<Val>,
    pub mmcs: InputMmcs,
}

pub struct ProverData<Val, MmcsData> {
    committed_domains: Vec<CircleDomain<Val>>,
    mmcs_data: MmcsData,
}

impl<Val, InputMmcs, Challenge, Challenger> Pcs<Challenge, Challenger> for CirclePcs<Val, InputMmcs>
where
    Val: ComplexExtendable,
    InputMmcs: 'static + for<'a> DirectMmcs<Val, Mat<'a> = RowMajorMatrixView<'a, Val>>,
    Challenge: ExtensionField<Val> + HasFrobenius<Val>,
    Challenger: CanSample<Challenge>,
{
    type Domain = CircleDomain<Val>;
    type Commitment = InputMmcs::Commitment;
    type ProverData = ProverData<Val, InputMmcs::ProverData>;
    type Proof = ();
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
        mat.to_row_major_matrix()
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
        let mut reduced_openings: [Option<Vec<Challenge>>; 32] = core::array::from_fn(|_| None);

        let values: OpenedValues<Challenge> = rounds
            .into_iter()
            .map(|(data, points_for_mats)| {
                let mats = self.mmcs.get_matrices(&data.mmcs_data);
                izip!(&data.committed_domains, mats, points_for_mats)
                    .map(|(domain, mat, points_for_mat)| {
                        let log_height = log2_strict_usize(mat.height());
                        let reduced_opening_for_log_height = reduced_openings[log_height]
                            .get_or_insert_with(|| vec![Challenge::zero(); mat.height()]);
                        points_for_mat
                            .into_iter()
                            .map(|zeta| {
                                let zeta_point = univariate_to_point(zeta).unwrap();
                                let basis: Vec<Challenge> = domain.lagrange_basis(zeta_point);
                                let v_n_at_zeta = v_n(zeta_point.real(), log_height)
                                    - v_n(domain.shift.real(), log_height);
                                // todo: we only need half of the values to interpolate, but how?
                                let ps_at_zeta = columnwise_dot_product(&mat, basis.into_iter())
                                    .into_iter()
                                    .map(|x| x * v_n_at_zeta)
                                    .collect();

                                let denoms = domain
                                    .points()
                                    .map(|p| {
                                        pair_vanishing(
                                            zeta_point,
                                            Complex::new(
                                                zeta_point.real().frobenius(),
                                                zeta_point.imag().frobenius(),
                                            ),
                                            Complex::new(p.real().into(), p.imag().into()),
                                        )
                                    })
                                    .collect_vec();
                                let inv_denoms = batch_multiplicative_inverse(&denoms);

                                info_span!("reduce rows").in_scope(|| {
                                    reduced_opening_for_log_height
                                        .par_iter_mut()
                                        .zip_eq(mat.par_rows())
                                        .zip(inv_denoms)
                                        .zip(domain.points())
                                        .for_each(|(((reduced_opening, row), inv_denom), x)| {
                                            for (&p_at_x, &p_at_zeta) in izip!(row, &ps_at_zeta) {
                                                *reduced_opening *= mu;
                                                *reduced_opening +=
                                                    (-frob_line(zeta_point, p_at_zeta, x) + p_at_x)
                                                        * inv_denom;
                                            }
                                        });
                                });

                                ps_at_zeta
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        // todo: fri prove
        (values, ())
    }

    fn verify(
        &self,
        // For each round:
        _rounds: Vec<(
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
        _proof: &Self::Proof,
        _challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        // todo: fri verify
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
