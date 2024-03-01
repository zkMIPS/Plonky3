//! And AIR for the Poseidon permutation.

#![no_std]

extern crate alloc;

mod air;
mod columns;
mod generation;
mod round_flags;

pub use air::*;
pub use columns::*;
pub use generation::*;
