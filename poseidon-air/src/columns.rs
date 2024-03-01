use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;

pub(crate) struct PoseidonCols<T, const WIDTH: usize, const N_ROUNDS: usize> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub round_flags: [T; N_ROUNDS],

    /// Set to 1 if we are currently in a partial round, otherwise 0.
    pub partial_round: T,

    pub start_of_round: [T; WIDTH],
    pub after_constants: [T; WIDTH],
    pub after_sbox: [T; WIDTH],
    pub after_mds: [T; WIDTH],
}

#[macro_export]
macro_rules! get_num_poseidon_cols {
    ($width:expr, $n_rounds:expr) => {
        size_of::<PoseidonCols<u8, $width, $n_rounds>>()
    };
}

impl<T, const WIDTH: usize, const N_ROUNDS: usize> Borrow<PoseidonCols<T, WIDTH, N_ROUNDS>> for [T] {
    fn borrow(&self) -> &PoseidonCols<T, WIDTH, N_ROUNDS> {
        debug_assert_eq!(self.len(), get_num_poseidon_cols!(WIDTH, N_ROUNDS));
        let (prefix, shorts, suffix) = unsafe { self.align_to::<PoseidonCols<T, WIDTH, N_ROUNDS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T, const WIDTH: usize, const N_ROUNDS: usize> BorrowMut<PoseidonCols<T, WIDTH, N_ROUNDS>> for [T] {
    fn borrow_mut(&mut self) -> &mut PoseidonCols<T, WIDTH, N_ROUNDS> {
        debug_assert_eq!(self.len(), get_num_poseidon_cols!(WIDTH, N_ROUNDS));
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<PoseidonCols<T, WIDTH, N_ROUNDS>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
