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
    use core::array;
    
    use p3_field::AbstractField;
    use p3_poseidon2::{HLMDSMat4, Poseidon2, Poseidon2MEMatrix, HL_BABYBEAR_16_EXTERNAL_ROUND_CONSTANTS, HL_BABYBEAR_16_INTERNAL_ROUND_CONSTANTS};

    use super::*;

    type F = BabyBear;

    #[test]
    fn test_poseidon2_constants() {
        let monty_constant = MATRIX_DIAG_16_BABYBEAR_U32.map(BabyBear::from_canonical_u32);
        assert_eq!(monty_constant, MATRIX_DIAG_16_BABYBEAR_MONTY);

        let monty_constant = MATRIX_DIAG_24_BABYBEAR_U32.map(BabyBear::from_canonical_u32);
        assert_eq!(monty_constant, MATRIX_DIAG_24_BABYBEAR_MONTY);
    }

    // A function which recreates the poseidon2 implementation in
    // https://github.com/HorizenLabs/poseidon2
    fn hl_poseidon2_babybear_width_16(input: &mut [BabyBear; 16]) {
        const WIDTH: usize = 16;
        const D: u64 = 7;
        const ROUNDS_F: usize = 8;
        const ROUNDS_P: usize = 13;

        let external_linear_layer: Poseidon2MEMatrix<16, _> = Poseidon2MEMatrix::new(HLMDSMat4);

        // Our Poseidon2 implementation.
        let poseidon2: Poseidon2<
            BabyBear,
            Poseidon2MEMatrix<16, _>,
            DiffusionMatrixBabybear,
            WIDTH,
            D,
        > = Poseidon2::new(
            ROUNDS_F,
            HL_BABYBEAR_16_EXTERNAL_ROUND_CONSTANTS.map(to_babybear_array).to_vec(),
            external_linear_layer,
            ROUNDS_P,
            to_babybear_array(HL_BABYBEAR_16_INTERNAL_ROUND_CONSTANTS).to_vec(),
            DiffusionMatrixBabybear,
        );

        poseidon2.permute_mut(input);
    }

    /// Test on the constant 0 input. 
    #[test]
    fn test_poseidon2_width_16_zeroes() {
        let mut input: [BabyBear; 16] = [0_u32; 16].map(F::from_wrapped_u32);

        let expected: [BabyBear; 16] = [
            1337856655, 1843094405, 328115114, 964209316, 1365212758, 1431554563, 210126733, 
            1214932203, 1929553766, 1647595522, 1496863878, 324695999, 1569728319, 1634598391, 
            597968641, 679989771
        ].map(BabyBear::from_canonical_u32);
        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on the input 0..16. 
    #[test]
    fn test_poseidon2_width_16_range() {
        let mut input: [BabyBear; 16] = array::from_fn(|i| F::from_wrapped_u32(i as u32));

        let expected: [BabyBear; 16] = [
            896560466, 771677727, 128113032, 1378976435, 160019712, 1452738514, 682850273,
            223500421, 501450187, 1804685789, 1671399593, 1788755219, 1736880027, 1352180784,
            1928489698, 1128802977,
        ].map(BabyBear::from_canonical_u32);
        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }

    /// Test on a roughly random input. 
    #[test]
    fn test_poseidon2_width_16_random() {
        let mut input: [BabyBear; 16] = [1179785652, 1291567559, 66272299, 471640172, 653876821, 478855335, 871063984, 540251327, 1506944720, 1403776782, 770420443, 126472305, 1535928603, 1017977016, 818646757, 359411429].map(BabyBear::from_canonical_u32);

        let expected: [BabyBear; 16] = [1736862924, 1950079822, 952072292, 1965704005, 236226362, 1113998185, 1624488077, 391891139, 1194078311, 1040746778, 1898067001, 774167026, 193702242, 859952892, 732204701, 1744970965].map(BabyBear::from_canonical_u32);

        hl_poseidon2_babybear_width_16(&mut input);
        assert_eq!(input, expected);
    }
}