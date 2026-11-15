// Copyright (c) 2026 Aldianokto
// Distributed under the Apache License, see the accompanying
// file LICENSE or https://www.apache.org/licenses/LICENSE-2.0.

//! SHA3-256 implementation from scratch, following FIPS 202.
//!
//! This module provides a self-contained SHA3-256 hasher without any
//! external crate dependencies. It implements the Keccak-f[1600]
//! permutation and the sponge construction as specified by NIST.
//!
//! Trade-offs compared to the `sha3` crate:
//! - No hardware acceleration (SHANI, AVX2). Pure software, correct but slower.
//! - Fully self-contained. No supply chain risk.
//!
//! The implementation follows the NIST FIPS 202 standard:
//! <https://csrc.nist.gov/publications/detail/fips/202/final>

/// SHA3-256 output size in bytes.
pub const SHA3_256_OUTPUT_SIZE: usize = 32;

/// SHA3-256 block size (rate) in bytes.
const SHA3_256_RATE: usize = 136;

/// Keccak-f[1600] round constants.
const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];

/// Rotation offsets for the rho step.
const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
];

/// Keccak-f[1600] permutation.
fn keccak_f1600(state: &mut [u64; 25]) {
    for round in 0..24 {
        // Theta step.
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                state[x + 5 * y] ^= d;
            }
        }

        // Rho and Pi steps.
        let mut b = [0u64; 25];
        for x in 0..5 {
            for y in 0..5 {
                let idx = x + 5 * y;
                let rot = ROTATION_OFFSETS[x][y];
                let new_x = y;
                let new_y = (2 * x + 3 * y) % 5;
                let new_idx = new_x + 5 * new_y;
                b[new_idx] = state[idx].rotate_left(rot);
            }
        }

        // Chi step.
        for y in 0..5 {
            for x in 0..5 {
                state[x + 5 * y] =
                    b[x + 5 * y] ^ ((!b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]);
            }
        }

        // Iota step.
        state[0] ^= ROUND_CONSTANTS[round];
    }
}

/// Absorb a full rate-sized block into the sponge state.
fn absorb_block(state: &mut [u64; 25], block: &[u8]) {
    debug_assert_eq!(block.len(), SHA3_256_RATE);
    for i in 0..SHA3_256_RATE / 8 {
        let mut lane = [0u8; 8];
        lane.copy_from_slice(&block[i * 8..(i + 1) * 8]);
        state[i] ^= u64::from_le_bytes(lane);
    }
    keccak_f1600(state);
}

/// Apply SHA3 padding (domain separation 0x06, then 0x80 at the end).
fn pad_block(input: &[u8], padded: &mut [u8; SHA3_256_RATE]) {
    debug_assert!(input.len() < SHA3_256_RATE);
    padded[..input.len()].copy_from_slice(input);
    padded[input.len()] = 0x06;
    padded[SHA3_256_RATE - 1] |= 0x80;
}

/// Compute SHA3-256 of the given data.
pub fn sha3_256(data: &[u8]) -> [u8; SHA3_256_OUTPUT_SIZE] {
    let mut state = [0u64; 25];

    let mut offset = 0;
    while offset + SHA3_256_RATE <= data.len() {
        absorb_block(&mut state, &data[offset..offset + SHA3_256_RATE]);
        offset += SHA3_256_RATE;
    }

    let remaining = &data[offset..];
    let mut padded = [0u8; SHA3_256_RATE];
    pad_block(remaining, &mut padded);
    absorb_block(&mut state, &padded);

    let mut out = [0u8; SHA3_256_OUTPUT_SIZE];
    for i in 0..SHA3_256_OUTPUT_SIZE / 8 {
        let lane = state[i].to_le_bytes();
        out[i * 8..(i + 1) * 8].copy_from_slice(&lane);
    }
    out
}

/// Compute SHA3-256 of the given data, returning a lowercase hex string.
pub fn sha3_256_hex(data: &[u8]) -> String {
    let digest = sha3_256(data);
    hex::encode(digest)
}

/// Streaming SHA3-256 hasher.
pub struct Sha3_256 {
    state: [u64; 25],
    buffer: [u8; SHA3_256_RATE],
    buffer_len: usize,
    finalized: bool,
}

impl Sha3_256 {
    pub fn new() -> Self {
        Sha3_256 {
            state: [0u64; 25],
            buffer: [0u8; SHA3_256_RATE],
            buffer_len: 0,
            finalized: false,
        }
    }

    pub fn update(&mut self, data: &[u8]) {
        assert!(!self.finalized, "cannot update after finalize");

        let mut offset = 0;

        if self.buffer_len > 0 {
            let needed = SHA3_256_RATE - self.buffer_len;
            let to_copy = needed.min(data.len());
            self.buffer[self.buffer_len..self.buffer_len + to_copy]
                .copy_from_slice(&data[..to_copy]);
            self.buffer_len += to_copy;
            offset += to_copy;

            if self.buffer_len == SHA3_256_RATE {
                absorb_block(&mut self.state, &self.buffer);
                self.buffer_len = 0;
            }
        }

        while offset + SHA3_256_RATE <= data.len() {
            absorb_block(&mut self.state, &data[offset..offset + SHA3_256_RATE]);
            offset += SHA3_256_RATE;
        }

        let remaining = &data[offset..];
        if !remaining.is_empty() {
            self.buffer[..remaining.len()].copy_from_slice(remaining);
            self.buffer_len = remaining.len();
        }
    }

    pub fn finalize(mut self) -> [u8; SHA3_256_OUTPUT_SIZE] {
        assert!(!self.finalized, "cannot finalize more than once");
        self.finalized = true;

        let mut padded = [0u8; SHA3_256_RATE];
        pad_block(&self.buffer[..self.buffer_len], &mut padded);
        absorb_block(&mut self.state, &padded);

        let mut out = [0u8; SHA3_256_OUTPUT_SIZE];
        for i in 0..SHA3_256_OUTPUT_SIZE / 8 {
            let lane = self.state[i].to_le_bytes();
            out[i * 8..(i + 1) * 8].copy_from_slice(&lane);
        }
        out
    }

    pub fn finalize_hex(self) -> String {
        hex::encode(self.finalize())
    }
}

impl Default for Sha3_256 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let expected = "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a";
        assert_eq!(sha3_256_hex(b""), expected);
    }

    #[test]
    fn test_abc() {
        let expected = "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532";
        assert_eq!(sha3_256_hex(b"abc"), expected);
    }

    #[test]
    fn test_output_length() {
        let hash = sha3_256(b"XEPAY");
        assert_eq!(hash.len(), SHA3_256_OUTPUT_SIZE);
    }

    #[test]
    fn test_determinism() {
        let a = sha3_256_hex(b"xepay");
        let b = sha3_256_hex(b"xepay");
        assert_eq!(a, b);
    }

    #[test]
    fn test_streaming_matches_oneshot() {
        let mut h = Sha3_256::new();
        h.update(b"hello xepay");
        let streamed = h.finalize_hex();
        let oneshot = sha3_256_hex(b"hello xepay");
        assert_eq!(streamed, oneshot);
    }

    #[test]
    fn test_streaming_byte_by_byte() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let mut h = Sha3_256::new();
        for &b in data {
            h.update(&[b]);
        }
        let streamed = h.finalize_hex();
        let oneshot = sha3_256_hex(data);
        assert_eq!(streamed, oneshot);
    }
}
