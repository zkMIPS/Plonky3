use alloc::vec;
use alloc::vec::Vec;
use core::iter;
use core::mem::size_of;

use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;
use p3_mds::MdsPermutation;
use tracing::instrument;

use crate::columns::PoseidonCols;
use crate::get_num_poseidon_cols;

#[instrument(name = "generate Poseidon trace", skip_all)]
pub fn generate_trace_rows<
    F: PrimeField64,
    const WIDTH: usize,
    const ALPHA: u64,
    const N_ROUNDS: usize,
    Mds,
>(
    inputs: Vec<[u64; WIDTH]>,
    half_num_full_rounds: usize,
    num_partial_rounds: usize,
    round_constants: Vec<u64>,
    mds: Mds,
) -> RowMajorMatrix<F>
where
    Mds: MdsPermutation<F, WIDTH>,
{
    let num_rows = (inputs.len() * N_ROUNDS).next_power_of_two();
    let num_columns = get_num_poseidon_cols!(WIDTH, N_ROUNDS);

    let mut trace = RowMajorMatrix::new(vec![F::zero(); num_rows * num_columns], num_columns);

    let (prefix, rows, suffix) = unsafe {
        trace
            .values
            .align_to_mut::<PoseidonCols<F, WIDTH, N_ROUNDS>>()
    };
    assert!(prefix.is_empty(), "Alignment should match");
    assert!(suffix.is_empty(), "Alignment should match");
    assert_eq!(rows.len(), num_rows);

    let padded_inputs = inputs.into_iter().chain(iter::repeat([0; WIDTH]));
    for (round, (row, input)) in rows.iter_mut().zip(padded_inputs).enumerate() {
        let this_round_constants = round_constants
            .iter()
            .skip(round * WIDTH)
            .take(WIDTH)
            .copied()
            .collect::<Vec<_>>();
        let is_partial_round =
            round >= half_num_full_rounds && round < half_num_full_rounds + num_partial_rounds;
        generate_trace_row_for_round::<F, WIDTH, ALPHA, N_ROUNDS, Mds>(
            row,
            input,
            round,
            is_partial_round,
            this_round_constants,
            mds.clone(),
        );
    }

    trace
}

fn generate_trace_row_for_round<
    F: PrimeField64,
    const WIDTH: usize,
    const ALPHA: u64,
    const N_ROUNDS: usize,
    Mds,
>(
    row: &mut PoseidonCols<F, WIDTH, N_ROUNDS>,
    input: [u64; WIDTH],
    round: usize,
    is_partial_round: bool,
    this_round_constants: Vec<u64>,
    mds: Mds,
) where
    Mds: MdsPermutation<F, WIDTH>,
{
    row.round_flags[round] = F::one();
    row.partial_round = F::from_bool(is_partial_round);

    for i in 0..WIDTH {
        row.start_of_round[i] = F::from_canonical_u64(input[i]);
    }

    // Populate after_constants
    for i in 0..WIDTH {
        row.after_constants[i] =
            row.start_of_round[i] + F::from_canonical_u64(this_round_constants[i]);
    }

    // Populate after_sbox
    if is_partial_round {
        row.after_sbox[0] = row.after_constants[0].exp_u64(ALPHA);
    } else {
        for i in 0..WIDTH {
            row.after_sbox[i] = row.after_constants[i].exp_u64(ALPHA);
        }
    }

    // Populate after_mds
    let mut state = [F::zero(); WIDTH];
    for i in 0..WIDTH {
        state[i] = row.after_sbox[i];
    }
    mds.permute_mut(&mut state);
    for i in 0..WIDTH {
        row.after_mds[i] = state[i];
    }
}
