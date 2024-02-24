//! And AIR for the Poseidon permutation.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod round_flags;

const HALF_N_FULL_ROUNDS: usize = 4;
const N_FULL_ROUNDS_TOTAL: usize = 2 * HALF_N_FULL_ROUNDS;
pub const N_PARTIAL_ROUNDS: usize = 22;
pub const N_ROUNDS: usize = N_FULL_ROUNDS_TOTAL + N_PARTIAL_ROUNDS;
