//! Integer hashing. The only source of randomness in the engine.

#[inline]
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Seed for an independent stream `n` of a parent seed.
#[inline]
pub fn derive(seed: u64, n: u64) -> u64 {
    splitmix64(seed ^ n.wrapping_mul(0xd6e8_feb8_6659_fd93))
}

#[inline]
pub fn hash2(seed: u64, x: i64, y: i64) -> u64 {
    let mut h = seed;
    h ^= (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = h.rotate_left(27).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    h = (h ^ (h >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 32)
}

#[inline]
pub fn hash3(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let mut h = seed;
    h ^= (x as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h = h.rotate_left(27).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= (y as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    h = h.rotate_left(31).wrapping_mul(0x1656_67b1_9e37_79f9);
    h ^= (z as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
    h = (h ^ (h >> 29)).wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^ (h >> 32)
}

/// FNV-1a over bytes, for golden-hash tests and file checksums.
pub fn fnv1a(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed ^ 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Hasher for integer keys (chunk positions). Much faster than SipHash and
/// independent of process-random state.
#[derive(Default, Clone, Copy)]
pub struct IntHasher(u64);

impl std::hash::Hasher for IntHasher {
    #[inline]
    fn finish(&self) -> u64 {
        splitmix64(self.0)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    #[inline]
    fn write_i32(&mut self, v: i32) {
        self.0 = (self.0.rotate_left(21) ^ v as u32 as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write_i32(v as i32)
    }
    #[inline]
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(21) ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    #[inline]
    fn write_i64(&mut self, v: i64) {
        self.write_u64(v as u64)
    }
    #[inline]
    fn write_usize(&mut self, v: usize) {
        self.write_u64(v as u64)
    }
}

pub type IntBuildHasher = std::hash::BuildHasherDefault<IntHasher>;
pub type IntMap<K, V> = std::collections::HashMap<K, V, IntBuildHasher>;
pub type IntSet<K> = std::collections::HashSet<K, IntBuildHasher>;
