use core::marker::PhantomData;

use crate::{Matrix, MatrixGet, MatrixRowSlices, MatrixRowSlicesMut, MatrixRows};

pub trait RowPermutation {
    fn permute_index(r: usize, height: usize) -> usize;
}

pub struct PermutedMatrix<Perm, Inner> {
    inner: Inner,
    _phantom: PhantomData<Perm>,
}

impl<Perm: RowPermutation, Inner> PermutedMatrix<Perm, Inner> {
    pub fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<T, Perm: RowPermutation, Inner: Matrix<T>> Matrix<T> for PermutedMatrix<Perm, Inner> {
    fn width(&self) -> usize {
        self.inner.width()
    }
    fn height(&self) -> usize {
        self.inner.height()
    }
}

impl<T, Perm: RowPermutation, Inner: MatrixGet<T>> MatrixGet<T> for PermutedMatrix<Perm, Inner> {
    fn get(&self, r: usize, c: usize) -> T {
        self.inner.get(Perm::permute_index(r, self.height()), c)
    }
}

impl<T, Perm: RowPermutation, Inner: MatrixRows<T>> MatrixRows<T> for PermutedMatrix<Perm, Inner> {
    type Row<'a> = Inner::Row<'a> where Self: 'a;
    fn row(&self, r: usize) -> Self::Row<'_> {
        self.inner.row(Perm::permute_index(r, self.height()))
    }
}

impl<T, Perm: RowPermutation, Inner: MatrixRowSlices<T>> MatrixRowSlices<T>
    for PermutedMatrix<Perm, Inner>
{
    fn row_slice(&self, r: usize) -> &[T] {
        self.inner.row_slice(Perm::permute_index(r, self.height()))
    }
}

impl<T, Perm: RowPermutation, Inner: MatrixRowSlicesMut<T>> MatrixRowSlicesMut<T>
    for PermutedMatrix<Perm, Inner>
{
    fn row_slice_mut(&mut self, r: usize) -> &mut [T] {
        self.inner
            .row_slice_mut(Perm::permute_index(r, self.height()))
    }
}
