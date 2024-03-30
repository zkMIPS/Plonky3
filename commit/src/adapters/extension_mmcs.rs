use alloc::vec::Vec;
use core::marker::PhantomData;
use core::slice;

use p3_field::{ExtensionField, Field};
use p3_matrix::{Dimensions, Matrix, MatrixRowSlices, MatrixRows};

use crate::Mmcs;

#[derive(Clone)]
pub struct ExtensionMmcs<F, EF, InnerMmcs> {
    inner: InnerMmcs,
    _phantom: PhantomData<(F, EF)>,
}

impl<F, EF, InnerMmcs> ExtensionMmcs<F, EF, InnerMmcs> {
    pub fn new(inner: InnerMmcs) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<F, EF, InnerMmcs> Mmcs<EF> for ExtensionMmcs<F, EF, InnerMmcs>
where
    F: Field,
    EF: ExtensionField<F>,
    InnerMmcs: Mmcs<F>,
{
    type ProverData<M> = InnerMmcs::ProverData<FlatMatrix<EF, M>>;
    type Commitment = InnerMmcs::Commitment;
    type Proof = InnerMmcs::Proof;
    type Error = InnerMmcs::Error;

    fn commit<M: MatrixRowSlices<EF>>(
        &self,
        inputs: Vec<M>,
    ) -> (Self::Commitment, Self::ProverData<M>) {
        self.inner
            .commit(inputs.into_iter().map(|mat| FlatMatrix::new(mat)).collect())
    }

    fn open_batch<M: MatrixRowSlices<EF>>(
        &self,
        index: usize,
        prover_data: &Self::ProverData<M>,
    ) -> (Vec<Vec<EF>>, Self::Proof) {
        let (opened_base_values, proof) = self.inner.open_batch(index, prover_data);
        let opened_ext_values = opened_base_values
            .into_iter()
            .map(|row| row.chunks(EF::D).map(EF::from_base_slice).collect())
            .collect();
        (opened_ext_values, proof)
    }

    fn get_matrices<'a, M: MatrixRowSlices<EF>>(
        &self,
        prover_data: &'a Self::ProverData<M>,
    ) -> Vec<&'a M> {
        self.inner
            .get_matrices(prover_data)
            .into_iter()
            .map(|mat| &mat.inner)
            .collect()
    }

    fn verify_batch(
        &self,
        commit: &Self::Commitment,
        dimensions: &[Dimensions],
        index: usize,
        opened_values: &[Vec<EF>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error> {
        let opened_base_values: Vec<Vec<F>> = opened_values
            .iter()
            .map(|row| {
                row.iter()
                    .flat_map(|el| el.as_base_slice())
                    .copied()
                    .collect()
            })
            .collect();
        let base_dimensions = dimensions
            .iter()
            .map(|dim| Dimensions {
                width: dim.width * EF::D,
                height: dim.height,
            })
            .collect::<Vec<_>>();
        self.inner
            .verify_batch(commit, &base_dimensions, index, &opened_base_values, proof)
    }
}

pub struct FlatMatrix<EF, Inner> {
    inner: Inner,
    _phantom: PhantomData<EF>,
}

impl<EF, Inner> FlatMatrix<EF, Inner> {
    fn new(inner: Inner) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

impl<F: Field, EF: ExtensionField<F>, Inner: Matrix<EF>> Matrix<F> for FlatMatrix<EF, Inner> {
    fn width(&self) -> usize {
        self.inner.width() * EF::D
    }
    fn height(&self) -> usize {
        self.inner.height()
    }
}

// this could be Inner: MatrixRows if you write an adapter
impl<F: Field, EF: ExtensionField<F>, Inner: MatrixRowSlices<EF>> MatrixRows<F>
    for FlatMatrix<EF, Inner>
{
    type Row<'a> = core::slice::Iter<'a, F>
    where
        Self: 'a;

    fn row(&self, r: usize) -> Self::Row<'_> {
        self.row_slice(r).iter()
    }
}

impl<F: Field, EF: ExtensionField<F>, Inner: MatrixRowSlices<EF>> MatrixRowSlices<F>
    for FlatMatrix<EF, Inner>
{
    fn row_slice(&self, r: usize) -> &[F] {
        let buf = self.inner.row_slice(r);
        unsafe { slice::from_raw_parts(buf.as_ptr().cast::<F>(), buf.len() * EF::D) }
    }
}
