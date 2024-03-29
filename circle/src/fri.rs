use itertools::Itertools;
use p3_challenger::{CanObserve, CanSample, GrindingChallenger};
use p3_commit::{DirectMmcs, Mmcs};
use p3_field::{extension::ComplexExtendable, ExtensionField};
use p3_fri::{CommitPhaseProofStep, FriConfig, FriProof, QueryProof};
use p3_matrix::dense::RowMajorMatrix;
use tracing::{info_span, instrument};

use crate::folding::fold_univariate;

#[instrument(name = "FRI prover", skip_all)]
pub(crate) fn prove<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger,
) -> (FriProof<EF, M, Challenger::Witness>, Vec<usize>)
where
    F: ComplexExtendable,
    EF: ExtensionField<F>,
    M: DirectMmcs<EF>,
    Challenger: GrindingChallenger + CanObserve<M::Commitment> + CanSample<EF>,
{
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();
    let commit_phase_result = commit_phase(config, input, log_max_height, challenger);
    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs = info_span!("query phase").in_scope(|| {
        query_indices
            .iter()
            .map(|&index| answer_query(config, &commit_phase_result.data, index))
            .collect()
    });

    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<F, EF, M, Challenger>(
    config: &FriConfig<M>,
    input: &[Option<Vec<EF>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger,
) -> CommitPhaseResult<EF, M>
where
    F: ComplexExtendable,
    EF: ExtensionField<F>,
    M: DirectMmcs<EF>,
    Challenger: CanObserve<M::Commitment> + CanSample<EF>,
{
    let mut current = input[log_max_height].as_ref().unwrap().clone();
    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        let (commit, prover_data) = config.mmcs.commit_matrix(RowMajorMatrix::new(current, 2));
        challenger.observe(commit.clone());

        let beta: EF = challenger.sample();
        // we passed ownership of `current` to the MMCS, so get a reference to it
        let committed_leaves = config.mmcs.get_matrices(&prover_data).pop().unwrap();
        current = fold_univariate(committed_leaves, beta);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }

        commits.push(commit);
        data.push(prover_data);
    }

    assert_eq!(current.len(), config.blowup());
    let final_poly = current[0];
    for x in current {
        assert_eq!(x, final_poly);
    }

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

struct CommitPhaseResult<F, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData>,
    final_poly: F,
}

fn answer_query<F, EF, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData],
    index: usize,
) -> QueryProof<EF, M>
where
    F: ComplexExtendable,
    EF: ExtensionField<F>,
    M: Mmcs<EF>,
{
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect();
    QueryProof {
        commit_phase_openings,
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_challenger::{HashChallenger, SerializingChallenger32};
    use p3_commit::ExtensionMmcs;
    use p3_field::{extension::BinomialExtensionField, AbstractExtensionField, Field};
    use p3_keccak::Keccak256Hash;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_mersenne_31::Mersenne31;
    use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher32};
    use rand::{thread_rng, Rng};

    use crate::{
        domain::CircleDomain,
        folding::{circle_bitrev_permute, fold_bivariate},
        twiddles::TwiddleCache,
        Cfft,
    };

    use super::*;

    #[test]
    fn test_circle_fold() {
        let mut rng = thread_rng();

        type F = Mersenne31;
        type EF = BinomialExtensionField<F, 3>;

        let log_n = 4;
        let rate_bits = 1;

        let mut evals: Vec<EF> = {
            let evals = RowMajorMatrix::<F>::rand(
                &mut rng,
                1 << log_n,
                <EF as AbstractExtensionField<F>>::D,
            );
            let lde = Cfft::default().lde(
                evals,
                CircleDomain::standard(log_n),
                CircleDomain::standard(log_n + rate_bits),
            );
            lde.rows().map(|r| EF::from_base_slice(r)).collect()
        };

        let mut tc = TwiddleCache::default();
        let twiddles = tc.get_twiddles(
            log_n + rate_bits,
            F::circle_two_adic_generator(log_n + rate_bits + 1),
            true,
        );

        dbg!(twiddles.len());

        for (layer, twiddle) in twiddles.iter().enumerate() {
            let beta: EF = rng.gen();

            let n = evals.len();
            let half_n = evals.len() >> 1;
            assert!(half_n != 0);

            dbg!(layer, twiddle.len(), n, half_n);

            for i in 0..half_n {
                let t = twiddle[i];
                let lo = evals[i];
                let hi = evals[n - i - 1];
                let sum = lo + hi;
                let diff = (lo - hi) * t;
                evals[i] = (sum + beta * diff).halve();
            }

            dbg!(&evals);

            evals.truncate(half_n);
        }

        dbg!(&evals);
    }

    #[test]
    fn test_circle_fri() {
        let mut rng = thread_rng();

        type Val = Mersenne31;
        type Challenge = BinomialExtensionField<Val, 3>;

        type ByteHash = Keccak256Hash;
        type FieldHash = SerializingHasher32<ByteHash>;
        let byte_hash = ByteHash {};
        let field_hash = FieldHash::new(Keccak256Hash {});

        type MyCompress = CompressionFunctionFromHasher<u8, ByteHash, 2, 32>;
        let compress = MyCompress::new(byte_hash);

        type ValMmcs = FieldMerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 32>;
        let val_mmcs = ValMmcs::new(field_hash, compress);

        type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 1,
            proof_of_work_bits: 0,
            mmcs: challenge_mmcs,
        };

        let mut inputs: [_; 32] = core::array::from_fn(|_| None);

        {
            let log_n = 3;
            let evals = RowMajorMatrix::<Val>::rand(
                &mut rng,
                1 << log_n,
                <Challenge as AbstractExtensionField<Val>>::D,
            );
            let lde = Cfft::default().lde(
                evals,
                CircleDomain::standard(log_n),
                CircleDomain::standard(log_n + fri_config.log_blowup),
            );
            let input = lde
                .rows()
                .map(|r| Challenge::from_base_slice(r))
                .collect_vec();
            let input = fold_bivariate::<Val, _>(
                RowMajorMatrix::new(circle_bitrev_permute(&input), 2),
                rng.gen(),
            );
            inputs[log_n + fri_config.log_blowup - 1] = Some(input);
        }

        let mut challenger = Challenger::from_hasher(vec![], byte_hash);
        prove::<Val, Challenge, _, _>(&fri_config, &inputs, &mut challenger);
    }
}
