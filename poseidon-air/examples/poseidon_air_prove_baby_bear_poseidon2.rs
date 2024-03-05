use p3_baby_bear::BabyBear;
use p3_challenger::DuplexChallenger;
use p3_commit::ExtensionMmcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{FriConfig, TwoAdicFriPcs, TwoAdicFriPcsConfig};
use p3_mds::babybear::MdsMatrixBabyBear;
use p3_merkle_tree::FieldMerkleTreeMmcs;
use p3_poseidon2::{DiffusionMatrixBabybear, Poseidon2};
use p3_poseidon_air::{generate_trace_rows, PoseidonAir};
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

fn main() -> Result<(), VerificationError> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2<Val, DiffusionMatrixBabybear, WIDTH, 7>;
    let perm = Perm::new_from_rng(8, 22, DiffusionMatrixBabybear, &mut thread_rng());

    type MyHash = PaddingFreeSponge<Perm, WIDTH, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = CompressionFunctionFromHasher<Val, MyHash, 2, 8>;
    let compress = MyCompress::new(hash);

    type ValMmcs = FieldMerkleTreeMmcs<Val, MyHash, MyCompress, 8>;
    let val_mmcs = ValMmcs::new(hash, compress);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel;
    let dft = Dft {};

    type Challenger = DuplexChallenger<Val, Perm, WIDTH>;

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
    let mds = MdsMatrixBabyBear {};

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
