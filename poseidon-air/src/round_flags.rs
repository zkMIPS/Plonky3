use core::borrow::Borrow;

use p3_air::AirBuilder;
use p3_matrix::MatrixRowSlices;

use crate::columns::PoseidonCols;

pub(crate) fn eval_round_flags<AB: AirBuilder, const WIDTH: usize>(builder: &mut AB, num_rounds: usize) {
    let main = builder.main();
    let local: &PoseidonCols<AB::Var, WIDTH> = main.row_slice(0).borrow();
    let next: &PoseidonCols<AB::Var, WIDTH> = main.row_slice(1).borrow();

    // Initially, the first step flag should be 1 while the others should be 0.
    builder.when_first_row().assert_one(local.round_flags[0]);
    for i in 1..num_rounds {
        builder.when_first_row().assert_zero(local.round_flags[i]);
    }

    for i in 0..num_rounds {
        let current_round_flag = local.round_flags[i];
        let next_round_flag = next.round_flags[(i + 1) % num_rounds];
        builder
            .when_transition()
            .assert_eq(next_round_flag, current_round_flag);
    }
}
