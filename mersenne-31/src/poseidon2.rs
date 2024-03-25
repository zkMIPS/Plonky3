use p3_field::AbstractField;
use p3_poseidon2::{matmul_internal, DiffusionPermutation};
use p3_symmetric::Permutation;

use crate::{to_babybear_array, Mersenne31};

// Two optimised diffusion matrices for Mersenne31/16:

// Mersenne31:
// Small entries: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17]
// Power of 2 entries: [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192, 16384, 32768, 65536]
// = [0, 2^0, 2^1, 2^2, 2^3, 2^4, 2^5, 2^6, 2^7, 2^8, 2^9, 2^11, 2^14, 2^15, 2^16, 2^17]

const MATRIX_DIAG_16_MERSENNE31_U32: [u32; 16] = [
    0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 8192, 16384, 32768, 65536
];

// Long term we should instead be doing some sort of delayed reduction strategy using the shifts.
// const MATRIX_DIAG_16_MONTY_SHIFTS: [i32; 16] = [-64, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17];
// Note that the first entry of this constant should never be accessed.

// Need to seperate Vector and Scalar code first to do this properly though.

// fn matmul_internal_shift<const WIDTH: usize>(
//     state: &mut [Mersenne31; WIDTH],
//     mat_internal_diag_shifts: [i32; WIDTH],
// ) {
//     let sum: u64 = state.iter().cloned().map(|x| x.value as u64).sum();
//     state[0] = Mersenne31 { value:monty_reduce(sum) };
//     for i in 1..WIDTH {
//         let result = ((state[i].value as u64) << mat_internal_diag_shifts[i]) + sum.clone();
//         state[i] = Mersenne31 { value:monty_reduce(result) };
//     }
// }


#[derive(Debug, Clone, Default)]
pub struct DiffusionMatrixMersenne31;

impl<AF: AbstractField<F = Mersenne31>> Permutation<[AF; 16]> for DiffusionMatrixMersenne31 {
    fn permute_mut(&self, state: &mut [AF; 16]) {
        matmul_internal::<Mersenne31, AF, 16>(state, MATRIX_DIAG_16_MERSENNE31_U32);
    }
}

impl<AF: AbstractField<F = Mersenne31>> DiffusionPermutation<AF, 16> for DiffusionMatrixMersenne31 {}
