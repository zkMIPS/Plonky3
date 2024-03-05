//! MDS matrices over the BabyBear field, and permutations defined by them.
//!
//! NB: Not all sizes have fast implementations of their permutations.
//! Supported sizes: 8, 12, 16, 24, 32, 64.
//! Sizes 8 and 12 are from Plonky2. Other sizes are from Ulrich Haböck's database.

use p3_baby_bear::BabyBear;
use p3_dft::Radix2Bowers;
use p3_field::extension::BinomialExtensionField;
use p3_symmetric::Permutation;

use crate::util::{
    apply_circulant, apply_circulant_12_sml, apply_circulant_8_sml, apply_circulant_fft,
    first_row_to_first_col,
};
use crate::MdsPermutation;

#[derive(Clone, Default)]
pub struct MdsMatrixBabyBearExtension;

const FFT_ALGO: Radix2Bowers = Radix2Bowers;

#[rustfmt::skip]
const MATRIX_CIRC_MDS_16_BABYBEAR: [u64; 16] = [
    0x07801000, 0x4ACAAC32, 0x6A709B76, 0x20413E94,
    0x00928499, 0x31C34CA3, 0x03BBC192, 0x3F20868B,
    0x257FFAAB, 0x5F05F559, 0x55B43EA9, 0x2BC659ED,
    0x2C6D7501, 0x1D110184, 0x0E1F608D, 0x2032F0C6,
];

impl Permutation<[BinomialExtensionField<BabyBear, 4>; 16]> for MdsMatrixBabyBearExtension {
    fn permute(&self, input: [BinomialExtensionField<BabyBear, 4>; 16]) -> [BinomialExtensionField<BabyBear, 4>; 16] {
        const ENTRIES: [u64; 16] = first_row_to_first_col(&MATRIX_CIRC_MDS_16_BABYBEAR);
        apply_circulant_fft(FFT_ALGO, ENTRIES, &input)
    }

    fn permute_mut(&self, input: &mut [BinomialExtensionField<BabyBear, 4>; 16]) {
        *input = self.permute(*input);
    }
}
impl MdsPermutation<BinomialExtensionField<BabyBear, 4>, 16> for MdsMatrixBabyBearExtension {}
