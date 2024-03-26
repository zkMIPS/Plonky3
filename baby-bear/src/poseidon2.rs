use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{monty_reduce, to_babybear_array, BabyBear};

// Diffusion matrices for Babybear16 and Babybear24.
//
// Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_babybear.rs
const MATRIX_DIAG_16_BABYBEAR_U32: [u32; 16] = [
    0x0a632d94, 0x6db657b7, 0x56fbdc9e, 0x052b3d8a, 0x33745201, 0x5c03108c, 0x0beba37b, 0x258c2e8b,
    0x12029f39, 0x694909ce, 0x6d231724, 0x21c3b222, 0x3c0904a5, 0x01d6acda, 0x27705c83, 0x5231c802,
];

const MATRIX_DIAG_24_BABYBEAR_U32: [u32; 24] = [
    0x409133f0, 0x1667a8a1, 0x06a6c7b6, 0x6f53160e, 0x273b11d1, 0x03176c5d, 0x72f9bbf9, 0x73ceba91,
    0x5cdef81d, 0x01393285, 0x46daee06, 0x065d7ba6, 0x52d72d6f, 0x05dd05e0, 0x3bab4b63, 0x6ada3842,
    0x2fc5fbec, 0x770d61b0, 0x5715aae9, 0x03ef0e90, 0x75b6c770, 0x242adf5f, 0x00d0ca4c, 0x36c0e388,
];

// These correspond to the internal poseidon2 matrix: 1 + D(v)
// Here 1 is the constant matrix of 1's and D(v) is the diagonal matrix with diagonal given by v.

// Convert the above arrays of u32's into arrays of BabyBear field elements saved in MONTY form.
pub(crate) const MATRIX_DIAG_16_BABYBEAR_MONTY: [BabyBear; 16] =
    to_babybear_array(MATRIX_DIAG_16_BABYBEAR_U32);
pub(crate) const MATRIX_DIAG_24_BABYBEAR_MONTY: [BabyBear; 24] =
    to_babybear_array(MATRIX_DIAG_24_BABYBEAR_U32);

// With some more work we can find more optimal choices for v:
// Two optimised diffusion matrices for Babybear16:
// Small entries: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17]
// Power of 2 entries: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 8192, 32768, 65536]
// = [0, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^10, 2^11, 2^13, 2^15, 2^16]

// In order to use these to their fullest potential we need to slightly reimage what the matrix looks like.
// Note that if (1 + D(v)) is a valid matrix then so is r(1 + D(v)) for any constant scalar r. Hence we should operate
// such that (1 + D(v)) is the monty form of the matrix. This should allow for some delayed reduction tricks.

const MATRIX_DIAG_16_MONTY_SHIFTS: [i32; 16] =
    [-64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16];

fn matmul_internal_shift<const WIDTH: usize>(
    state: &mut [BabyBear; WIDTH],
    mat_internal_diag_shifts: [i32; WIDTH],
) {
    let sum: u64 = state.iter().cloned().map(|x| x.value as u64).sum();
    state[0] = BabyBear {
        value: monty_reduce(sum),
    };
    for i in 1..WIDTH {
        let result = ((state[i].value as u64) << mat_internal_diag_shifts[i]) + sum;
        state[i] = BabyBear {
            value: monty_reduce(result),
        };
    }
}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybear;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [BabyBear; 16]) {
        matmul_internal::<BabyBear, BabyBear, 16>(state, MATRIX_DIAG_16_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabybear {}

impl Permutation<[BabyBear; 24]> for DiffusionMatrixBabybear {
    fn permute_mut(&self, state: &mut [BabyBear; 24]) {
        matmul_internal::<BabyBear, BabyBear, 24>(state, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }
}

impl DiffusionPermutation<BabyBear, 24> for DiffusionMatrixBabybear {}

#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixBabybearScalar;

impl Permutation<[BabyBear; 16]> for DiffusionMatrixBabybearScalar {
    fn permute_mut(&self, state: &mut [BabyBear; 16]) {
        matmul_internal_shift::<16>(state, MATRIX_DIAG_16_MONTY_SHIFTS);
    }
}

impl DiffusionPermutation<BabyBear, 16> for DiffusionMatrixBabybearScalar {}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use ark_ff::{BigInteger, PrimeField};
    use p3_field::AbstractField;
    use p3_poseidon2::Poseidon2;
    use p3_symmetric::Permutation;
    use rand::Rng;
    use zkhash::fields::babybear::FpBabyBear;
    use zkhash::poseidon2::poseidon2::Poseidon2 as Poseidon2Ref;
    use zkhash::poseidon2::poseidon2_instance_babybear::{POSEIDON2_BABYBEAR_16_PARAMS, RC16};

    use super::*;

    // These are currently saved as their true values. It will be far more efficient to save them in Monty Form.

    #[test]
    fn test_poseidon2_constants() {
        let monty_constant = MATRIX_DIAG_16_BABYBEAR_U32.map(BabyBear::from_canonical_u32);
        assert_eq!(monty_constant, MATRIX_DIAG_16_BABYBEAR_MONTY);

        let monty_constant = MATRIX_DIAG_24_BABYBEAR_U32.map(BabyBear::from_canonical_u32);
        assert_eq!(monty_constant, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }

    fn babybear_from_ark_ff(input: FpBabyBear) -> BabyBear {
        let as_bigint = input.into_bigint();
        let mut as_bytes = as_bigint.to_bytes_le();
        as_bytes.resize(4, 0);
        let as_u32 = u32::from_le_bytes(as_bytes[0..4].try_into().unwrap());
        BabyBear::from_wrapped_u32(as_u32)
    }

    #[test]
    fn test_poseidon2_babybear_width_16() {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        type F = BabyBear;

        let mut rng = rand::thread_rng();

        // Poiseidon2 reference implementation from zkhash repo.
        let poseidon2_ref = Poseidon2Ref::new(&POSEIDON2_BABYBEAR_16_PARAMS);

        // Copy over round constants from zkhash.
        let round_constants: Vec<[F; WIDTH]> = RC16
            .iter()
            .map(|vec| {
                vec.iter()
                    .cloned()
                    .map(babybear_from_ark_ff)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<BabyBear, DiffusionMatrixBabybear, WIDTH, D> =
            Poseidon2::new(ROUNDS_F, ROUNDS_P, round_constants, DiffusionMatrixBabybear);

        // Generate random input and convert to both BabyBear field formats.
        let input_u32 = rng.gen::<[u32; WIDTH]>();
        let input_ref = input_u32
            .iter()
            .cloned()
            .map(FpBabyBear::from)
            .collect::<Vec<_>>();
        let input = input_u32.map(F::from_wrapped_u32);

        // Check that the conversion is correct.
        assert!(input_ref
            .iter()
            .zip(input.iter())
            .all(|(a, b)| babybear_from_ark_ff(*a) == *b));

        // Run reference implementation.
        let output_ref = poseidon2_ref.permutation(&input_ref);
        let expected: [F; WIDTH] = output_ref
            .iter()
            .cloned()
            .map(babybear_from_ark_ff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // Run our implementation.
        let mut output = input;
        poseidon2.permute_mut(&mut output);

        assert_eq!(output, expected);
    }
}
