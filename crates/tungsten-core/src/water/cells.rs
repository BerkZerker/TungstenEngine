//! A sparse set of lattice cells, stored as one bit per cell in the chunks
//! that have any.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::IVec3;

use crate::codec::{get_signed, get_varint, put_signed, put_varint};
use crate::coords::{CHUNK_BITS, CHUNK_SIZE, CHUNK_VOLUME, chunk_of, index, local_of};

use super::Mask;

const WORDS: usize = CHUNK_VOLUME / 64;

/// Chunk position as `[y, z, x]`, so chunks sort by height first.
type Key = [i32; 3];

fn key(chunk: IVec3) -> Key {
    [chunk.y, chunk.z, chunk.x]
}

/// Cloning one copies no bits until either copy is changed, so a snapshot
/// can be handed to a worker for nothing.
#[derive(Clone, Default, Debug)]
pub struct Cells {
    chunks: BTreeMap<Key, Arc<[u64; WORDS]>>,
    len: usize,
    /// Bounds of every cell ever inserted; not shrunk by removal.
    bounds: Option<(IVec3, IVec3)>,
}

impl PartialEq for Cells {
    fn eq(&self, other: &Self) -> bool {
        self.chunks == other.chunks
    }
}

impl Eq for Cells {}

impl Extend<IVec3> for Cells {
    fn extend<I: IntoIterator<Item = IVec3>>(&mut self, iter: I) {
        for p in iter {
            self.insert(p);
        }
    }
}

impl FromIterator<IVec3> for Cells {
    fn from_iter<I: IntoIterator<Item = IVec3>>(iter: I) -> Self {
        let mut cells = Self::default();
        for p in iter {
            cells.insert(p);
        }
        cells
    }
}

impl Cells {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A box (inclusive) that holds every cell.
    pub fn bounds(&self) -> Option<(IVec3, IVec3)> {
        self.bounds.filter(|_| self.len > 0)
    }

    pub fn contains(&self, p: IVec3) -> bool {
        let i = index(local_of(p));
        self.chunks.get(&key(chunk_of(p))).is_some_and(|bits| bits[i >> 6] >> (i & 63) & 1 == 1)
    }

    /// Add a cell. True if it was not there.
    pub fn insert(&mut self, p: IVec3) -> bool {
        let i = index(local_of(p));
        let bits = Arc::make_mut(self.chunks.entry(key(chunk_of(p))).or_insert_with(|| Arc::new([0; WORDS])));
        let new = bits[i >> 6] >> (i & 63) & 1 == 0;
        bits[i >> 6] |= 1 << (i & 63);
        self.len += new as usize;
        let (lo, hi) = self.bounds.get_or_insert((p, p));
        (*lo, *hi) = (lo.min(p), hi.max(p));
        new
    }

    /// Take a cell out. True if it was there.
    pub fn remove(&mut self, p: IVec3) -> bool {
        let (k, i) = (key(chunk_of(p)), index(local_of(p)));
        let Some(bits) = self.chunks.get_mut(&k) else { return false };
        let was = bits[i >> 6] >> (i & 63) & 1 == 1;
        if !was {
            return false;
        }
        let bits = Arc::make_mut(bits);
        bits[i >> 6] &= !(1 << (i & 63));
        self.len -= was as usize;
        if was && bits.iter().all(|&w| w == 0) {
            self.chunks.remove(&k);
        }
        was
    }

    /// Every cell, in a fixed order.
    pub fn iter(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.chunks.iter().flat_map(|(k, bits)| {
            let origin = IVec3::new(k[2], k[0], k[1]) * CHUNK_SIZE;
            bits.iter().enumerate().flat_map(move |(w, &word)| {
                Bits(word).map(move |b| {
                    let i = (w * 64 + b as usize) as i32;
                    origin + IVec3::new(i & 31, i >> (2 * CHUNK_BITS), (i >> CHUNK_BITS) & 31)
                })
            })
        })
    }

    /// Cells inside `[min, max]`.
    pub fn iter_box(&self, min: IVec3, max: IVec3) -> impl Iterator<Item = IVec3> + '_ {
        let (c0, c1) = (chunk_of(min), chunk_of(max));
        let overlaps = self.bounds().is_some_and(|(lo, hi)| lo.cmple(max).all() && hi.cmpge(min).all());
        overlaps
            .then(|| self.chunks.range([c0.y, c0.z, c0.x]..=[c1.y, c1.z, c1.x]))
            .into_iter()
            .flatten()
            .filter(move |(k, _)| k[1] >= c0.z && k[1] <= c1.z && k[2] >= c0.x && k[2] <= c1.x)
            .flat_map(move |(k, bits)| {
                let origin = IVec3::new(k[2], k[0], k[1]) * CHUNK_SIZE;
                bits.iter().enumerate().flat_map(move |(w, &word)| {
                    Bits(word).filter_map(move |b| {
                        let i = (w * 64 + b as usize) as i32;
                        let p = origin + IVec3::new(i & 31, i >> (2 * CHUNK_BITS), (i >> CHUNK_BITS) & 31);
                        (p.cmpge(min).all() && p.cmple(max).all()).then_some(p)
                    })
                })
            })
    }

    /// The cells at lattice height `y`, as one mask per chunk column `(x, z)` that has any.
    pub fn layer(&self, y: i32) -> impl Iterator<Item = ((i32, i32), Mask)> + '_ {
        let (cy, ly) = (y >> CHUNK_BITS, (y & (CHUNK_SIZE - 1)) as usize);
        self.chunks.range([cy, i32::MIN, i32::MIN]..=[cy, i32::MAX, i32::MAX]).filter_map(move |(k, bits)| {
            // A layer is 1024 bits: 16 words, two rows of 32 columns in each.
            let words = &bits[ly * 16..ly * 16 + 16];
            let mask: Mask = std::array::from_fn(|z| (words[z / 2] >> (32 * (z % 2))) as u32);
            mask.iter().any(|&row| row != 0).then_some(((k[2], k[1]), mask))
        })
    }

    pub fn encode(&self, out: &mut Vec<u8>) {
        put_varint(out, self.chunks.len() as u64);
        for (k, bits) in &self.chunks {
            k.iter().for_each(|&v| put_signed(out, v as i64));
            // Run lengths of clear and set bits in turn, starting with clear.
            let (mut run, mut state) = (0u64, false);
            for i in 0..CHUNK_VOLUME {
                let bit = bits[i >> 6] >> (i & 63) & 1 == 1;
                if bit != state {
                    put_varint(out, run);
                    (run, state) = (0, bit);
                }
                run += 1;
            }
            put_varint(out, run);
        }
    }

    pub fn decode(bytes: &[u8], at: &mut usize) -> Result<Self, String> {
        let mut cells = Self::default();
        for _ in 0..get_varint(bytes, at)? {
            let mut k = [0i32; 3];
            for v in &mut k {
                *v = i32::try_from(get_signed(bytes, at)?).map_err(|_| "bad chunk position")?;
            }
            let origin = IVec3::new(k[2], k[0], k[1]) * CHUNK_SIZE;
            let (mut i, mut state) = (0usize, false);
            while i < CHUNK_VOLUME {
                let run = get_varint(bytes, at)? as usize;
                if run > CHUNK_VOLUME - i || (run == 0 && i > 0) {
                    return Err("bad run of cells".into());
                }
                if state {
                    for j in i..i + run {
                        let j = j as i32;
                        cells.insert(origin + IVec3::new(j & 31, j >> (2 * CHUNK_BITS), (j >> CHUNK_BITS) & 31));
                    }
                }
                i += run;
                state = !state;
            }
        }
        Ok(cells)
    }
}

/// The positions of the set bits of a word, lowest first.
struct Bits(u64);

impl Iterator for Bits {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        (self.0 != 0).then(|| {
            let b = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            b
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scatter() -> Vec<IVec3> {
        (0..500i64)
            .map(|k| {
                let h = crate::hash::hash2(3, k, 0);
                IVec3::new((h % 90) as i32 - 40, ((h >> 20) % 70) as i32 - 35, ((h >> 40) % 50) as i32 - 5)
            })
            .collect()
    }

    #[test]
    fn holds_what_was_put_in_and_survives_encoding() {
        let points = scatter();
        let mut cells: Cells = points.iter().copied().collect();
        let mut unique = points.clone();
        unique.sort_by_key(|p| (p.x, p.y, p.z));
        unique.dedup();
        assert_eq!(cells.len(), unique.len());
        assert!(points.iter().all(|&p| cells.contains(p)));
        assert!(!cells.contains(IVec3::new(1000, 0, 0)));
        let mut listed: Vec<IVec3> = cells.iter().collect();
        listed.sort_by_key(|p| (p.x, p.y, p.z));
        assert_eq!(listed, unique);
        let (lo, hi) = cells.bounds().unwrap();
        assert!(points.iter().all(|p| p.cmpge(lo).all() && p.cmple(hi).all()));

        let (min, max) = (IVec3::new(-10, -40, 0), IVec3::new(33, 3, 31));
        let inside = |p: &IVec3| p.cmpge(min).all() && p.cmple(max).all();
        let mut boxed: Vec<IVec3> = cells.iter_box(min, max).collect();
        boxed.sort_by_key(|p| (p.x, p.y, p.z));
        assert_eq!(boxed, unique.iter().copied().filter(inside).collect::<Vec<_>>());
        assert!(boxed.len() > 20);

        // One layer, by chunk column.
        let y = unique[0].y;
        let mut layer = 0;
        for ((cx, cz), mask) in cells.layer(y) {
            for (z, row) in mask.iter().enumerate() {
                for x in Bits(*row as u64) {
                    assert!(cells.contains(IVec3::new(cx * 32 + x as i32, y, cz * 32 + z as i32)));
                    layer += 1;
                }
            }
        }
        assert_eq!(layer, unique.iter().filter(|p| p.y == y).count());

        let mut bytes = Vec::new();
        cells.encode(&mut bytes);
        let mut at = 0;
        let back = Cells::decode(&bytes, &mut at).unwrap();
        assert!(at == bytes.len() && back == cells && back.len() == cells.len());
        assert!(Cells::decode(&bytes[..bytes.len() - 1], &mut 0).is_err());

        assert!(cells.remove(points[0]) && !cells.remove(points[0]) && !cells.contains(points[0]));
        assert_eq!(cells.len(), unique.len() - 1);
        for p in &unique {
            cells.remove(*p);
        }
        assert!(cells.is_empty() && cells.bounds().is_none() && cells == Cells::default());
    }
}
