//! This file contains simple wrapper structs on top of which we can implement Poseidon2 Internal/ExternalLayer.
//!
//! They are used only in the case that none of the vectorization architectures (AVX2/AVX512/NEON) are available.

use alloc::vec::Vec;

use p3_poseidon2::{ExternalLayerConstants, ExternalLayerConstructor, InternalLayerConstructor};

use serde::{Serialize, Deserialize};
use crate::Mersenne31;

/// The internal layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "Vec<Mersenne31>: Serialize"))]
#[serde(bound(
    deserialize = "Vec<Mersenne31>: Deserialize<'de>"
))]
pub struct Poseidon2InternalLayerMersenne31 {
    pub(crate) internal_constants: Vec<Mersenne31>,
}

/// The external layers of the Poseidon2 permutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "[Mersenne31; WIDTH]: Serialize"))]
#[serde(bound(
    deserialize = "[Mersenne31; WIDTH]: Deserialize<'de>"
))]
pub struct Poseidon2ExternalLayerMersenne31<const WIDTH: usize> {
    pub(crate) external_constants: ExternalLayerConstants<Mersenne31, WIDTH>,
}

impl InternalLayerConstructor<Mersenne31> for Poseidon2InternalLayerMersenne31 {
    fn new_from_constants(internal_constants: Vec<Mersenne31>) -> Self {
        Self { internal_constants }
    }
}

impl<const WIDTH: usize> ExternalLayerConstructor<Mersenne31, WIDTH>
    for Poseidon2ExternalLayerMersenne31<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Mersenne31, WIDTH>) -> Self {
        Self { external_constants }
    }
}
