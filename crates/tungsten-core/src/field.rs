//! A dense copy of a box of samples at one level of detail.

use std::sync::Arc;

use glam::IVec3;

use crate::chunk::{Chunk, Dense, Sample, Volume};
use crate::coords::{APRON, CHUNK_SIZE, ChunkPos, NodeKey, PAD, SOLID, chunk_of, chunk_origin, index_xyz};

pub struct LocalField {
    /// Lattice coordinate, at this field's level, of the sample at local (0, 0, 0).
    pub origin: IVec3,
    pub dims: IVec3,
    /// Samples are `2^level` base cells apart.
    pub level: u8,
    pub fill: Vec<u8>,
    pub mat: Vec<u8>,
}

impl LocalField {
    pub fn new(origin: IVec3, dims: IVec3, level: u8) -> Self {
        let n = (dims.x * dims.y * dims.z) as usize;
        Self { origin, dims, level, fill: vec![0; n], mat: vec![0; n] }
    }

    /// The padded block of a node: its own samples plus an apron, all air.
    pub fn block(key: NodeKey) -> Self {
        Self::new(chunk_origin(key.pos) - APRON, IVec3::splat(PAD), key.level)
    }

    #[inline]
    pub fn index(&self, x: i32, y: i32, z: i32) -> usize {
        debug_assert!(x >= 0 && y >= 0 && z >= 0 && x < self.dims.x && y < self.dims.y && z < self.dims.z);
        ((y * self.dims.z + z) * self.dims.x + x) as usize
    }

    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> Sample {
        let i = self.index(x, y, z);
        Sample { fill: self.fill[i], mat: self.mat[i] }
    }

    /// Index strides for +X, +Y and +Z.
    #[inline]
    pub fn strides(&self) -> [usize; 3] {
        [1, (self.dims.z * self.dims.x) as usize, self.dims.x as usize]
    }

    /// Base cells between samples.
    #[inline]
    pub fn step(&self) -> i32 {
        1 << self.level
    }

    pub fn set_all(&mut self, s: Sample) {
        self.fill.fill(s.fill);
        self.mat.fill(s.mat);
    }

    /// True if no surface can pass through the field.
    pub fn is_homogeneous(&self) -> bool {
        let first = self.fill[0] >= SOLID;
        self.fill.iter().all(|&f| (f >= SOLID) == first)
    }

    /// The part of this field inside `[min, max]` (lattice coordinates of this
    /// level, inclusive), as local `(lo, hi)` with `hi` exclusive; `None` if empty.
    pub fn clip(&self, min: IVec3, max: IVec3) -> Option<(IVec3, IVec3)> {
        let lo = (min - self.origin).max(IVec3::ZERO);
        let hi = (max - self.origin + 1).min(self.dims);
        (lo.cmplt(hi).all()).then_some((lo, hi))
    }

    /// Copy the block of `center` out of 27 chunks of its level, indexed
    /// `((dy + 1) * 3 + dz + 1) * 3 + dx + 1`.
    pub fn gather(key: NodeKey, chunks: &[Arc<Chunk>; 27]) -> Self {
        let mut field = Self::block(key);
        const N: i32 = CHUNK_SIZE;
        for cy in -1..=1 {
            for cz in -1..=1 {
                for cx in -1..=1 {
                    let chunk: &Chunk = &chunks[(((cy + 1) * 3 + cz + 1) * 3 + cx + 1) as usize];
                    // Range of this neighbour's local samples inside the block.
                    let range = |c: i32| match c {
                        -1 => (N - APRON, N),
                        0 => (0, N),
                        _ => (0, APRON),
                    };
                    let ((x0, x1), (y0, y1), (z0, z1)) = (range(cx), range(cy), range(cz));
                    let len = (x1 - x0) as usize;
                    for y in y0..y1 {
                        for z in z0..z1 {
                            let dst = field.index(x0 + cx * N + APRON, y + cy * N + APRON, z + cz * N + APRON);
                            match chunk {
                                Chunk::Uniform(s) => {
                                    field.fill[dst..dst + len].fill(s.fill);
                                    field.mat[dst..dst + len].fill(s.mat);
                                }
                                Chunk::Dense(d) => {
                                    let src = index_xyz(x0, y, z);
                                    field.fill[dst..dst + len].copy_from_slice(&d.fill[src..src + len]);
                                    field.mat[dst..dst + len].copy_from_slice(&d.mat[src..src + len]);
                                }
                            }
                        }
                    }
                }
            }
        }
        field
    }

    /// The chunk at the middle of a padded block.
    pub fn interior(&self) -> Chunk {
        debug_assert_eq!(self.dims, IVec3::splat(PAD));
        let mut dense = Dense::new_boxed(Sample::AIR);
        let n = CHUNK_SIZE as usize;
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                let (src, dst) = (self.index(APRON, y + APRON, z + APRON), index_xyz(0, y, z));
                dense.fill[dst..dst + n].copy_from_slice(&self.fill[src..src + n]);
                dense.mat[dst..dst + n].copy_from_slice(&self.mat[src..src + n]);
            }
        }
        let mut chunk = Chunk::Dense(dense);
        chunk.normalize();
        chunk
    }
}

/// For each chunk overlapping the box, call `f(chunk, local_min, local_max_exclusive)`.
fn for_chunks(min: IVec3, dims: IVec3, mut f: impl FnMut(ChunkPos, IVec3, IVec3)) {
    let max = min + dims - 1;
    let (c0, c1) = (chunk_of(min), chunk_of(max));
    for cy in c0.y..=c1.y {
        for cz in c0.z..=c1.z {
            for cx in c0.x..=c1.x {
                let pos = IVec3::new(cx, cy, cz);
                let o = chunk_origin(pos);
                let lo = (min - o).max(IVec3::ZERO);
                let hi = (max - o + 1).min(IVec3::splat(CHUNK_SIZE));
                f(pos, lo, hi);
            }
        }
    }
}

impl Volume {
    /// True if every chunk overlapping the box is loaded.
    pub fn covers(&self, min: IVec3, dims: IVec3) -> bool {
        let mut all = true;
        for_chunks(min, dims, |pos, _, _| all &= self.chunks.contains_key(&pos));
        all
    }

    /// Copy a box of samples. Unloaded space reads as air.
    pub fn copy_region(&self, min: IVec3, dims: IVec3) -> LocalField {
        let mut field = LocalField::new(min, dims, 0);
        for_chunks(min, dims, |pos, lo, hi| {
            let Some(chunk) = self.chunks.get(&pos) else { return };
            let chunk: &Chunk = chunk;
            let o = chunk_origin(pos) - min;
            let len = (hi.x - lo.x) as usize;
            for y in lo.y..hi.y {
                for z in lo.z..hi.z {
                    let dst = field.index(lo.x + o.x, y + o.y, z + o.z);
                    match chunk {
                        Chunk::Uniform(s) => {
                            field.fill[dst..dst + len].fill(s.fill);
                            field.mat[dst..dst + len].fill(s.mat);
                        }
                        Chunk::Dense(d) => {
                            let src = index_xyz(lo.x, y, z);
                            field.fill[dst..dst + len].copy_from_slice(&d.fill[src..src + len]);
                            field.mat[dst..dst + len].copy_from_slice(&d.mat[src..src + len]);
                        }
                    }
                }
            }
        });
        field
    }

    /// Write a field back, touching only samples that differ. Unloaded chunks
    /// are skipped. Appends each chunk that changed to `changed` and returns
    /// the bounds (inclusive) of the samples that did.
    pub fn write_region(&mut self, field: &LocalField, changed: &mut Vec<ChunkPos>) -> Option<(IVec3, IVec3)> {
        debug_assert_eq!(field.level, 0);
        let mut bounds: Option<(IVec3, IVec3)> = None;
        for_chunks(field.origin, field.dims, |pos, lo, hi| {
            let Some(chunk) = self.chunks.get_mut(&pos) else { return };
            let o = chunk_origin(pos) - field.origin;
            let len = (hi.x - lo.x) as usize;
            let mut any = false;
            for y in lo.y..hi.y {
                for z in lo.z..hi.z {
                    let src = field.index(lo.x + o.x, y + o.y, z + o.z);
                    let dst = index_xyz(lo.x, y, z);
                    let (new_fill, new_mat) = (&field.fill[src..src + len], &field.mat[src..src + len]);
                    let same = match &**chunk {
                        Chunk::Uniform(s) => {
                            new_fill.iter().all(|&f| f == s.fill) && new_mat.iter().all(|&m| m == s.mat)
                        }
                        Chunk::Dense(d) => &d.fill[dst..dst + len] == new_fill && &d.mat[dst..dst + len] == new_mat,
                    };
                    if same {
                        continue;
                    }
                    let d = Arc::make_mut(chunk).dense_mut();
                    let differs = |k: &usize| d.fill[dst + k] != new_fill[*k] || d.mat[dst + k] != new_mat[*k];
                    let first = (0..len).find(differs).unwrap() as i32;
                    let last = (0..len).rfind(differs).unwrap() as i32;
                    let row = chunk_origin(pos) + IVec3::new(lo.x, y, z);
                    let (b0, b1) = bounds.get_or_insert((row + IVec3::X * first, row + IVec3::X * last));
                    *b0 = b0.min(row + IVec3::X * first);
                    *b1 = b1.max(row + IVec3::X * last);
                    d.fill[dst..dst + len].copy_from_slice(new_fill);
                    d.mat[dst..dst + len].copy_from_slice(new_mat);
                    any = true;
                }
            }
            if any {
                changed.push(pos);
            }
        });
        bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_and_write_round_trip_across_chunks() {
        let mut vol = Volume::new();
        for x in -1..=0 {
            vol.insert(IVec3::new(x, 0, 0), Chunk::Uniform(Sample { fill: 255, mat: 1 }));
        }
        let min = IVec3::new(-3, 30, 5);
        let mut f = vol.copy_region(min, IVec3::new(6, 4, 2));
        // Rows above y=31 are in an unloaded chunk and read as air.
        assert_eq!(f.get(0, 0, 0), Sample { fill: 255, mat: 1 });
        assert_eq!(f.get(0, 2, 0), Sample::AIR);
        let i = f.index(2, 1, 1);
        f.fill[i] = 10;
        let j = f.index(3, 1, 1);
        f.mat[j] = 2;
        let mut changed = Vec::new();
        let bounds = vol.write_region(&f, &mut changed);
        assert_eq!(bounds, Some((IVec3::new(-1, 31, 6), IVec3::new(0, 31, 6))));
        changed.sort_by_key(|c| c.x);
        assert_eq!(changed, vec![IVec3::new(-1, 0, 0), IVec3::ZERO]);
        assert_eq!(vol.get(IVec3::new(-1, 31, 6)), Some(Sample { fill: 10, mat: 1 }));
        assert_eq!(vol.get(IVec3::new(0, 31, 6)), Some(Sample { fill: 255, mat: 2 }));
        // Writing the same field again changes nothing.
        changed.clear();
        assert!(vol.write_region(&f, &mut changed).is_none());
        assert!(changed.is_empty());
        assert!(vol.covers(min, IVec3::new(6, 2, 2)) && !vol.covers(min, IVec3::new(6, 4, 2)));
    }

    #[test]
    fn a_gathered_block_gives_back_its_middle_chunk() {
        let key = NodeKey::new(0, IVec3::new(1, -2, 3));
        let chunks: [Arc<Chunk>; 27] = std::array::from_fn(|i| {
            let mut c = Chunk::Uniform(Sample { fill: 200, mat: 1 });
            let d = c.dense_mut();
            d.fill[i] = i as u8;
            d.mat[1000 + i] = 2;
            Arc::new(c)
        });
        let block = LocalField::gather(key, &chunks);
        assert!(block.interior() == *chunks[13]);
        // The apron below the block comes from the top of the chunk beneath.
        assert_eq!(block.get(APRON, APRON - 1, APRON), chunks[4].get(index_xyz(0, CHUNK_SIZE - 1, 0)));
        assert_eq!(block.get(PAD - 1, APRON, APRON), chunks[14].get(index_xyz(APRON - 1, 0, 0)));
    }
}
