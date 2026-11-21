//! A dense copy of a box of samples, for work that spans chunk borders.

use std::sync::Arc;

use glam::IVec3;

use crate::chunk::{Chunk, Sample, Volume};
use crate::coords::{CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, chunk_of, chunk_origin, index_xyz};

/// One bit per sample of a chunk: has it ever been written by an edit?
pub type Touched = Box<[u64; CHUNK_VOLUME / 64]>;

pub struct LocalField {
    /// Lattice coordinate of the sample at local (0, 0, 0).
    pub origin: IVec3,
    pub dims: IVec3,
    pub fill: Vec<u8>,
    pub mat: Vec<u8>,
}

impl LocalField {
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
    /// Copy a box of samples. Unloaded space reads as air.
    pub fn copy_region(&self, min: IVec3, dims: IVec3) -> LocalField {
        let n = (dims.x * dims.y * dims.z) as usize;
        let mut field = LocalField { origin: min, dims, fill: vec![0; n], mat: vec![0; n] };
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
    /// are skipped. Appends each chunk that changed to `changed`.
    pub fn write_region(&mut self, field: &LocalField, changed: &mut Vec<ChunkPos>) {
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
                    let touched = self.touched.entry(pos).or_insert_with(|| Box::new([0; CHUNK_VOLUME / 64]));
                    for k in 0..len {
                        if d.fill[dst + k] != new_fill[k] || d.mat[dst + k] != new_mat[k] {
                            d.fill[dst + k] = new_fill[k];
                            d.mat[dst + k] = new_mat[k];
                            touched[(dst + k) / 64] |= 1 << ((dst + k) % 64);
                        }
                    }
                    any = true;
                }
            }
            if any {
                changed.push(pos);
            }
        });
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
        vol.write_region(&f, &mut changed);
        changed.sort_by_key(|c| c.x);
        assert_eq!(changed, vec![IVec3::new(-1, 0, 0), IVec3::ZERO]);
        assert_eq!(vol.get(IVec3::new(-1, 31, 6)), Some(Sample { fill: 10, mat: 1 }));
        assert_eq!(vol.get(IVec3::new(0, 31, 6)), Some(Sample { fill: 255, mat: 2 }));
        assert_eq!(vol.touched[&IVec3::ZERO].iter().map(|w| w.count_ones()).sum::<u32>(), 1);
        // Writing the same field again changes nothing.
        changed.clear();
        vol.write_region(&f, &mut changed);
        assert!(changed.is_empty());
    }
}
