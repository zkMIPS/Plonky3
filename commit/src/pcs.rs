//! Traits for polynomial commitment schemes.

use p3_field::field::{Field, FieldExtension};
use p3_matrix::dense::RowMajorMatrix;

use alloc::vec::Vec;

/// A (not necessarily hiding) polynomial commitment scheme, for committing to (batches of)
/// polynomials defined over the field `F`.
///
/// This high-level trait is agnostic with respect to the structure of a point; see `UnivariatePCS`
/// and `MultivariatePCS` for more specific subtraits.
pub trait PCS<F: Field> {
    /// The commitment that's sent to the verifier.
    type Commitment;

    /// Data that the prover stores for committed polynomials, to help the prover with opening.
    type ProverData;

    /// The opening argument.
    type Proof;

    type Error;

    fn commit_batches(polynomials: Vec<RowMajorMatrix<F>>) -> (Self::Commitment, Self::ProverData);

    fn get_committed_value(
        prover_data: &Self::ProverData,
        batch: usize,
        poly: usize,
        value: usize,
    ) -> F;
}

pub trait UnivariatePCS<F: Field>: PCS<F> {
    fn open_batches<FE: FieldExtension<F>>(
        points: &[FE],
        prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof);

    fn verify_batches<FE: FieldExtension<F>>(
        commit: &Self::Commitment,
        points: &[FE],
        values: &[Vec<Vec<FE>>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}

pub trait MultivariatePCS<F: Field>: PCS<F> {
    fn open_batches<FE: FieldExtension<F>>(
        points: &[FE],
        prover_data: &[Self::ProverData],
    ) -> (Vec<Vec<Vec<FE>>>, Self::Proof);

    fn verify_batches<FE: FieldExtension<F>>(
        commit: &Self::Commitment,
        points: &[FE],
        values: &[Vec<Vec<FE>>],
        proof: &Self::Proof,
    ) -> Result<(), Self::Error>;
}