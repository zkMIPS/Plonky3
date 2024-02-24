use p3_air::{Air, AirBuilder, BaseAir};

use crate::columns::{PoseidonCols, NUM_POSEIDON_COLS_WIDTH_8};
use crate::round_flags::eval_round_flags;

pub struct PoseidonAirWidth8 {}

impl<F> BaseAir<F> for PoseidonAirWidth8 {
    fn width(&self) -> usize {
        NUM_POSEIDON_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for PoseidonAirWidth8 {
    fn eval(&self, builder: &mut AB) {
        eval_round_flags(builder);

        let main = builder.main();
        let local: &PoseidonCols<AB::Var> = main.row_slice(0).borrow();
        let next: &PoseidonCols<AB::Var> = main.row_slice(1).borrow();


    }
}