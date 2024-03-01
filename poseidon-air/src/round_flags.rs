use core::borrow::Borrow;

use p3_air::AirBuilder;
use p3_matrix::MatrixRowSlices;

use crate::columns::PoseidonCols;

pub(crate) fn eval_round_flags<AB: AirBuilder, const WIDTH: usize, const N_ROUNDS: usize>(builder: &mut AB) {
    let main = builder.main();
    let local: &PoseidonCols<AB::Var, WIDTH, N_ROUNDS> = main.row_slice(0).borrow();
    let next: &PoseidonCols<AB::Var, WIDTH, N_ROUNDS> = main.row_slice(1).borrow();

    // Initially, the first step flag should be 1 while the others should be 0.
    builder.when_first_row().assert_one(local.round_flags[0]);
    for i in 1..N_ROUNDS {
        builder.when_first_row().assert_zero(local.round_flags[i]);
    }

    for i in 0..N_ROUNDS {
        let current_round_flag = local.round_flags[i];
        let next_round_flag = next.round_flags[(i + 1) % N_ROUNDS];
        builder
            .when_transition()
            .assert_eq(next_round_flag, current_round_flag);
    }
}
