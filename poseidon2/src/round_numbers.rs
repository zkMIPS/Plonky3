// We compute the required number of partial and full rounds using:
// See https://github.com/0xPolygonZero/hash-constants
// Note the the number of rounds

use p3_field::PrimeField32;
// These round numbers will presumably work for all 31-bit prime fields but we have only checked the following primes.
const TESTED_31_BIT_PRIMES: [u32; 3] = [(1 << 31) - 1, (1 << 31) - (1 << 27) + 1, (1 << 31) - (1 << 24) + 1];

/// Given a field, a width and an alpha return the number of full and partial rounds needed to achieve 128 bit security.
pub fn poseidon_round_numbers<F: PrimeField32>(width: u64, alpha: u64) -> (u32, u32) {
    assert!(TESTED_31_BIT_PRIMES.contains(&F::ORDER_U32));
    match (width, alpha) {
        (16, 3) => (8, 20),
        (16, 5) => (8, 14),
        (16, 7) => (8, 13),
        (16, 11) => (8, 13),
        (24, 3) => (8, 23),
        (24, 5) => (8, 22),
        (24, 7) => (8, 21),
        (24, 11) => (8, 21),
        _ => panic!("The given pair of width and alpha has not been checked"),
    }
}

// (3, 8) => (8, 41)
// (3, 12) => (8, 42)
// (3, 16) => (8, 22)
// (7, 8) => (8, 22)
// (7, 12) => (8, 22)
// (7, 16) => (8, 22)
// (11, 8) => (8, 17)
// (11, 12) => (8, 18)
// (11, 16) => (8, 18)
