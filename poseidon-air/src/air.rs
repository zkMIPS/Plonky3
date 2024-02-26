use core::borrow::Borrow;

use alloc::vec::Vec;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::AbstractField;
use p3_matrix::MatrixRowSlices;

use crate::columns::{get_num_poseidon_cols, PoseidonCols};
use crate::round_flags::eval_round_flags;
use crate::{HALF_N_FULL_ROUNDS, N_PARTIAL_ROUNDS};

pub struct PoseidonAir<const WIDTH: usize> {
    pub(crate) round_constants: Vec<u64>,
}

impl<F, const WIDTH: usize> BaseAir<F> for PoseidonAir<WIDTH> {
    fn width(&self) -> usize {
        get_num_poseidon_cols(WIDTH)
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for PoseidonAir<WIDTH> {
    fn eval(&self, builder: &mut AB) {
        eval_round_flags::<AB, WIDTH>(builder);
        
        let num_rounds = 2 * HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS;

        let main = builder.main();
        let local: &PoseidonCols<AB::Var, WIDTH> = main.row_slice(0).borrow();
        let next: &PoseidonCols<AB::Var, WIDTH> = main.row_slice(1).borrow();

        // check that round constants are added correctly
        let constants = self.round_constants.clone();
        for i in 0..WIDTH {
            let mut round_constant = AB::Expr::zero();
            for r in 0..num_rounds {
                let this_round = local.round_flags[r];
                let this_round_constant = AB::Expr::from_canonical_u64(constants[r * WIDTH + i]);
                round_constant += this_round * this_round_constant;
            }
            let current = local.start_of_round[i];
            let next = local.after_constants[i];

            builder.assert_eq(next, current + round_constant);
        }
    }
}