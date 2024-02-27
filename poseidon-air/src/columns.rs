use core::borrow::{Borrow, BorrowMut};
use core::mem::{size_of, transmute};

use p3_util::indices_arr;

use crate::N_ROUNDS;

pub(crate) struct PoseidonCols<T, const WIDTH: usize> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub round_flags: [T; N_ROUNDS],

    /// Set to 1 if we are currently in a partial round, otherwise 0.
    pub partial_round: T,

    pub start_of_round: [T; WIDTH],
    pub after_constants: [T; WIDTH],
    pub after_sbox: [T; WIDTH],
    pub after_mds: [T; WIDTH],
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

const fn make_col_map<const WIDTH: usize>() -> PoseidonCols<usize, WIDTH> {
    if WIDTH == 8 {
        make_col_map_width_8()
    } else if WIDTH == 12 {
        make_col_map_width_12()
    } else {
        unimplemented!()
    }
}

const fn make_col_map_width_8() -> PoseidonCols<usize, 8> {
    let indices_arr = indices_arr::<{ get_num_poseidon_cols(8) }>();
    unsafe {
        transmute::<[usize; size_of::<PoseidonCols<u8, 8>>()], PoseidonCols<usize, 8>>(indices_arr)
    }
}

const fn make_col_map_width_12() -> PoseidonCols<usize, 12> {
    let indices_arr = indices_arr::<{ size_of::<PoseidonCols<u8, 12>>() }>();
    unsafe {
        transmute::<[usize; size_of::<PoseidonCols<u8, 12>>()], PoseidonCols<usize, 12>>(
            indices_arr,
        )
    }
}

impl<T, const WIDTH: usize> Borrow<PoseidonCols<T, WIDTH>> for [T] {
    fn borrow(&self) -> &PoseidonCols<T, WIDTH> {
        debug_assert_eq!(self.len(), get_num_poseidon_cols(WIDTH));
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PoseidonCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize> BorrowMut<PoseidonCols<T, WIDTH>> for [T] {
    fn borrow_mut(&mut self) -> &mut PoseidonCols<T, WIDTH> {
        debug_assert_eq!(self.len(), get_num_poseidon_cols(WIDTH));
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<PoseidonCols<T, WIDTH>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
