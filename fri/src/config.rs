use alloc::vec::Vec;
use p3_matrix::MatrixRows;

pub struct FriConfig<M> {
    pub log_blowup: usize,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
    pub mmcs: M,
}

impl<M> FriConfig<M> {
    pub fn blowup(&self) -> usize {
        1 << self.log_blowup
    }
}

pub trait FriFolder<F> {
    /// Fold along each row, returning a single column.
    /// Right now this will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_matrix<M: MatrixRows<F>>(m: M, beta: F) -> Vec<F>;
    fn fold_row(index: usize, log_height: usize, evals: &[F], beta: F) -> F;

    fn combine_vec(&self, current: &mut [F], new: &[F]);
    fn combine_row(&self, current: &mut F, new: F, index: usize, log_height: usize);
}
