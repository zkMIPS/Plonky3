use alloc::vec::Vec;
use core::borrow::Borrow;
use core::mem::size_of;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::MatrixRowSlices;
use p3_mds::MdsPermutation;

use crate::columns::PoseidonCols;
use crate::get_num_poseidon_cols;
use crate::round_flags::eval_round_flags;

pub struct PoseidonAir<Mds: Sync, const WIDTH: usize, const ALPHA: u64, const N_ROUNDS: usize> {
    half_num_full_rounds: usize,
    num_partial_rounds: usize,
    round_constants: Vec<u64>,
    mds: Mds,
}

impl<Mds: Sync, const WIDTH: usize, const ALPHA: u64, const N_ROUNDS: usize>
    PoseidonAir<Mds, WIDTH, ALPHA, N_ROUNDS>
{
    pub fn new(
        half_num_full_rounds: usize,
        num_partial_rounds: usize,
        round_constants: Vec<u64>,
        mds: Mds,
    ) -> Self {
        let num_rounds = 2 * half_num_full_rounds + num_partial_rounds;
        assert_eq!(num_rounds, N_ROUNDS);
        assert_eq!(round_constants.len(), WIDTH * num_rounds);

        Self {
            half_num_full_rounds,
            num_partial_rounds,
            round_constants,
            mds,
        }
    }
}

impl<F, Mds: Sync, const WIDTH: usize, const ALPHA: u64, const N_ROUNDS: usize> BaseAir<F>
    for PoseidonAir<Mds, WIDTH, ALPHA, N_ROUNDS>
{
    fn width(&self) -> usize {
        get_num_poseidon_cols!(WIDTH, N_ROUNDS)
    }
}

impl<AB: AirBuilder, Mds: Sync, const WIDTH: usize, const ALPHA: u64, const N_ROUNDS: usize> Air<AB>
    for PoseidonAir<Mds, WIDTH, ALPHA, N_ROUNDS>
where
    Mds: MdsPermutation<AB::Expr, WIDTH>,
{
    fn eval(&self, builder: &mut AB) {
        let num_rounds = 2 * self.half_num_full_rounds + self.num_partial_rounds;
        assert_eq!(num_rounds, N_ROUNDS);

        eval_round_flags::<AB, WIDTH, N_ROUNDS>(builder);

        let main = builder.main();
        let local: &PoseidonCols<AB::Var, WIDTH, N_ROUNDS> = main.row_slice(0).borrow();
        let next: &PoseidonCols<AB::Var, WIDTH, N_ROUNDS> = main.row_slice(1).borrow();

        // The partial round flag must be 0 or 1.
        builder.assert_bool(local.partial_round);

        // check that round constants are added correctly
        let constants = self.round_constants.clone();
        for i in 0..WIDTH {
            let mut round_constant = AB::Expr::zero();
            for r in 0..num_rounds {
                let this_round = local.round_flags[r];
                let this_round_constant = AB::Expr::from_canonical_u64(constants[r * WIDTH + i]);
                round_constant += this_round * this_round_constant;
            }
            let before = local.start_of_round[i];
            let expected = local.after_constants[i];

            builder.assert_eq(expected, before + round_constant);
        }

        // check that sbox layer is correct
        // partial s-box
        let before = local.after_constants[0];
        let expected = local.after_sbox[0];
        let after = before.into().exp_u64(ALPHA);
        builder.assert_eq(expected, after);

        // full s-box
        let full_round = AB::Expr::one() - local.partial_round;
        for i in 0..WIDTH {
            let before = local.after_constants[i];
            let expected = local.after_sbox[i];
            let after = before.into().exp_u64(ALPHA);
            builder.assert_eq(after * full_round.clone(), expected * full_round.clone());
        }

        // check that MDS layer is correct
        let before: [AB::Expr; WIDTH] = local.after_sbox.map(|x| x.into());
        let expected = local.after_mds;
        let after = self.mds.permute(before);
        for i in 0..WIDTH {
            builder.assert_eq(after[i].clone(), expected[i]);
        }

        // check that end of this round matches start of next round
        for i in 0..WIDTH {
            let end = local.after_mds[i];
            let start = next.start_of_round[i];
            builder.assert_eq(end, start);
        }
    }
}

mod tests {
    use alloc::vec::Vec;
    use p3_mds::babybear_extension::MdsMatrixBabyBearExtension;
    use p3_uni_stark::VerificationError;
    use p3_baby_bear::BabyBear;
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_field::extension::BinomialExtensionField;
    use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
    use p3_mds::babybear::MdsMatrixBabyBear;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
    use crate::{generate_trace_rows, PoseidonAir};
    use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge};
    use p3_uni_stark::{prove, verify, StarkConfig, VerificationError};
    use rand::{random, thread_rng};
    use tracing_forest::util::LevelFilter;
    use tracing_forest::ForestLayer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::{EnvFilter, Registry};
    
    const NUM_HASHES: usize = 680;
    
    const WIDTH: usize = 8;
    const ALPHA: u64 = 7;
    const N_ROUNDS: usize = 30;

    #[test]
    fn test_poseidon_air() -> Result<(), VerificationError> {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();

        type Val = BabyBear;
        type Challenge = BinomialExtensionField<Val, 4>;

        type Perm = Poseidon2<Val, DiffusionMatrixBabybear, 24, 7>;
        let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

        type MyHash = PaddingFreeSponge<Perm, 24, 8, 8>;
        let hash = MyHash::new(perm.clone());

        type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
        let compress = MyCompress::new(hash);

        type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
        let val_mmcs = ValMmcs::new(hash, compress);

        type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

        type Dft = Radix2DitParallel;
        let dft = Dft {};

        type Challenger = DuplexChallenger<Val, Perm, 24>;

        let fri_config = FriConfig {
            log_blowup: 1,
            num_queries: 100,
            proof_of_work_bits: 16,
            mmcs: challenge_mmcs,
        };
        type Pcs =
            TwoAdicFriPcs<TwoAdicFriPcsConfig<Val, Challenge, Challenger, Dft, ValMmcs, ChallengeMmcs>>;
        let pcs = Pcs::new(fri_config, dft, val_mmcs);

        type MyConfig = StarkConfig<Val, Challenge, Pcs, Challenger>;
        let config = StarkConfig::new(pcs);

        let mut challenger = Challenger::new(perm.clone());

        let half_num_full_rounds = 4;
        let num_partial_rounds = 22;
        let round_constants = (0..N_ROUNDS * WIDTH).map(|_| random()).collect::<Vec<_>>();
        let inputs = (0..NUM_HASHES).map(|_| random()).collect::<Vec<_>>();
        let mds = MdsMatrixBabyBearExtension {};

        let trace = generate_trace_rows::<Val, WIDTH, ALPHA, N_ROUNDS, MdsMatrixBabyBear>(
            inputs,
            half_num_full_rounds,
            num_partial_rounds,
            round_constants,
            mds,
        );
        dbg!(trace.clone());
        let air = PoseidonAir::new(
            half_num_full_rounds,
            num_partial_rounds,
            round_constants,
            mds,
        );
        let proof = prove::<MyConfig, _>(&config, &air, &mut challenger, trace);

        let mut challenger = Challenger::new(perm);
        verify(&config, &air, &mut challenger, &proof)
    }
}