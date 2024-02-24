use core::{borrow::Borrow, mem::{size_of, transmute}};

use p3_util::indices_arr;

use crate::N_ROUNDS;

pub(crate) struct PoseidonCols<T, const WIDTH: usize> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub round_flags: [T; N_ROUNDS],

    pub start_of_round: [[T; WIDTH]; N_ROUNDS],
    pub after_constants: [[T; WIDTH]; N_ROUNDS],
    pub after_sbox: [[T; WIDTH]; N_ROUNDS],
    pub after_mds: [[T; WIDTH]; N_ROUNDS],
}

pub(crate) const fn get_num_poseidon_cols(width: usize) -> usize {
    if width == 8 {
        size_of::<PoseidonCols<u8, 8>>()
    } else if width == 12 {
        size_of::<PoseidonCols<u8, 12>>()
    } else {
        unimplemented!()
    }
}

const fn make_col_map(width: usize) -> PoseidonCols<usize, 8> {
    if width == 8 {
        make_col_map_width_8()
    } else if width == 12 {
        make_col_map_width_12()
    } else {
        unimplemented!()
    }
}

const fn make_col_map_width_8() -> PoseidonCols<usize, 8> {
    let indices_arr = indices_arr::<NUM_POSEIDON_COLS_WIDTH_8>();
    unsafe { transmute::<[usize; NUM_POSEIDON_COLS_WIDTH_8], PoseidonCols<usize, 8>>(indices_arr) }
}

const fn make_col_map_width_12() -> PoseidonCols<usize, 12> {
    let indices_arr = indices_arr::<NUM_POSEIDON_COLS_WIDTH_12>();
    unsafe { transmute::<[usize; NUM_POSEIDON_COLS_WIDTH_12], PoseidonCols<usize, 12>>(indices_arr) }
}

impl<T> Borrow<PoseidonCols<T>> for [T] {
    fn borrow(&self) -> &PoseidonCols<T> {
        debug_assert_eq!(self.len(), NUM_POSEIDON_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PoseidonCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}
