use std::any::type_name;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use p3_baby_bear::{BabyBear, DiffusionMatrixBabybear, DiffusionMatrixBabybearScalar};
use p3_bn254_fr::{Bn254Fr, DiffusionMatrixBN254};
use p3_field::{PrimeField, PrimeField64};
use p3_goldilocks::{DiffusionMatrixGoldilocks, Goldilocks};
use p3_mersenne_31::{DiffusionMatrixMersenne31, Mersenne31};
use p3_poseidon2::{DiffusionPermutation, Poseidon2, Poseidon2ExternalMatrixGeneral, MDSLightPermutation};
use p3_symmetric::Permutation;
use rand::distributions::{Distribution, Standard};
use rand::thread_rng;

fn bench_poseidon2(c: &mut Criterion) {
    poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 16, 7>(c);
    poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybearScalar, 16, 7>(c);
    poseidon2_p64::<BabyBear, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabybear, 24, 7>(c);

    poseidon2_p64::<Mersenne31, Poseidon2ExternalMatrixGeneral, DiffusionMatrixMersenne31, 16, 5>(c);

    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 8, 7>(c);
    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 12, 7>(c);
    poseidon2_p64::<Goldilocks, Poseidon2ExternalMatrixGeneral, DiffusionMatrixGoldilocks, 16, 7>(c);

    poseidon2::<Bn254Fr, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBN254, 3, 5>(c, 8, 22);
}

// For 64 bit fields we use poseidon2_p64 which chooses the parameters rounds_f, rounds_p as the minimal values 
// to achieve 128-bit soundness.

fn poseidon2<F, MDSLight, Diffusion, const WIDTH: usize, const D: u64>(c: &mut Criterion, rounds_f: usize, rounds_p: usize)
where
    F: PrimeField,
    Standard: Distribution<F>,
    MDSLight: MDSLightPermutation<F, WIDTH> + Default,
    Diffusion: DiffusionPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let internal_layer = Diffusion::default();
    let external_layer = MDSLight::default();

    // TODO: Should be calculated for the particular field, width and ALPHA.

    let poseidon = Poseidon2::<F, MDSLight, Diffusion, WIDTH, D>::new_from_rng_test(
        rounds_f,
        external_layer,
        rounds_p,
        internal_layer,
        &mut rng,
    );
    let input = [F::zero(); WIDTH];
    let name = format!("poseidon2::<{}, {}, {}, {}>", type_name::<F>(), D, rounds_f, rounds_p);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

fn poseidon2_p64<F, MDSLight, Diffusion, const WIDTH: usize, const D: u64>(c: &mut Criterion)
where
    F: PrimeField64,
    Standard: Distribution<F>,
    MDSLight: MDSLightPermutation<F, WIDTH> + Default,
    Diffusion: DiffusionPermutation<F, WIDTH> + Default,
{
    let mut rng = thread_rng();
    let internal_layer = Diffusion::default();
    let external_layer = MDSLight::default();

    let poseidon = Poseidon2::<F, MDSLight, Diffusion, WIDTH, D>::new_from_rng_128(
        external_layer,
        internal_layer,
        &mut rng,
    );
    let input = [F::zero(); WIDTH];
    let name = format!("poseidon2::<{}, {}>", type_name::<F>(), D);
    let id = BenchmarkId::new(name, WIDTH);
    c.bench_with_input(id, &input, |b, &input| b.iter(|| poseidon.permute(input)));
}

criterion_group!(benches, bench_poseidon2);
criterion_main!(benches);
