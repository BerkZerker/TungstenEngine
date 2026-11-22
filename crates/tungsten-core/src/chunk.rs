use std::sync::Arc;

use glam::IVec3;

use crate::coords::{CHUNK_VOLUME, ChunkPos, SOLID, chunk_of, index, local_of};
use crate::hash::IntMap;

/// One lattice sample: how much material, and which.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Sample {
    pub fill: u8,
    pub mat: u8,
}

impl Sample {
    pub const AIR: Sample = Sample { fill: 0, mat: 0 };

    #[inline]
    pub fn solid(self) -> bool {
        self.fill >= SOLID
    }
}

/// Sample arrays of a chunk that is not uniform.
#[derive(Clone, PartialEq, Eq)]
pub struct Dense {
    pub fill: [u8; CHUNK_VOLUME],
    pub mat: [u8; CHUNK_VOLUME],
}

impl Dense {
    pub fn new_boxed(sample: Sample) -> Box<Dense> {
        // Built on the heap: the arrays are too large to construct on a stack.
        let mut v = vec![sample.fill; CHUNK_VOLUME * 2];
        v[CHUNK_VOLUME..].fill(sample.mat);
        let raw = Box::into_raw(v.into_boxed_slice()) as *mut Dense;
        // SAFETY: `Dense` is two `[u8; CHUNK_VOLUME]` with alignment 1 and no
        // padding, so it has exactly the layout of the boxed slice.
        unsafe { Box::from_raw(raw) }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum Chunk {
    /// Every sample is the same (all air, or all one full material).
    Uniform(Sample),
    Dense(Box<Dense>),
}

impl Chunk {
    pub const AIR: Chunk = Chunk::Uniform(Sample::AIR);

    #[inline]
    pub fn get(&self, i: usize) -> Sample {
        match self {
            Chunk::Uniform(s) => *s,
            Chunk::Dense(d) => Sample { fill: d.fill[i], mat: d.mat[i] },
        }
    }

    /// Mutable arrays, converting a uniform chunk to dense.
    pub fn dense_mut(&mut self) -> &mut Dense {
        if let Chunk::Uniform(s) = *self {
            *self = Chunk::Dense(Dense::new_boxed(s));
        }
        match self {
            Chunk::Dense(d) => d,
            Chunk::Uniform(_) => unreachable!(),
        }
    }

    /// Collapse to `Uniform` if every sample is identical. Keeps equal worlds
    /// in equal representation regardless of how they got there.
    pub fn normalize(&mut self) {
        if let Chunk::Dense(d) = self {
            let (f, m) = (d.fill[0], d.mat[0]);
            if d.fill.iter().all(|&x| x == f) && d.mat.iter().all(|&x| x == m) {
                *self = Chunk::Uniform(Sample { fill: f, mat: m });
            }
        }
    }

    pub fn is_uniform(&self) -> bool {
        matches!(self, Chunk::Uniform(_))
    }

    /// Content hash, independent of representation.
    pub fn content_hash(&self, seed: u64) -> u64 {
        match self {
            Chunk::Dense(d) => crate::hash::fnv1a(crate::hash::fnv1a(seed, &d.fill), &d.mat),
            Chunk::Uniform(s) => {
                let d = Dense::new_boxed(*s);
                crate::hash::fnv1a(crate::hash::fnv1a(seed, &d.fill), &d.mat)
            }
        }
    }
}

/// A set of loaded chunks addressed by sample coordinate.
///
/// Chunks are shared: a worker can hold one while the simulation carries on,
/// and a write to a chunk that is being read elsewhere copies it first.
#[derive(Default)]
pub struct Volume {
    pub chunks: IntMap<ChunkPos, Arc<Chunk>>,
}

impl Volume {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, pos: ChunkPos, chunk: Chunk) {
        self.chunks.insert(pos, Arc::new(chunk));
    }

    #[inline]
    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos).map(|c| &**c)
    }

    /// Sample at a lattice coordinate, or `None` if its chunk is not loaded.
    #[inline]
    pub fn get(&self, p: IVec3) -> Option<Sample> {
        self.chunks.get(&chunk_of(p)).map(|c| c.get(index(local_of(p))))
    }

    /// Sample at a lattice coordinate, treating unloaded space as air.
    #[inline]
    pub fn get_or_air(&self, p: IVec3) -> Sample {
        self.get(p).unwrap_or(Sample::AIR)
    }

    /// Write one sample. Returns false if the chunk is not loaded.
    pub fn set(&mut self, p: IVec3, s: Sample) -> bool {
        match self.chunks.get_mut(&chunk_of(p)) {
            Some(c) => {
                let i = index(local_of(p));
                if c.get(i) != s {
                    let d = Arc::make_mut(c).dense_mut();
                    d.fill[i] = s.fill;
                    d.mat[i] = s.mat;
                }
                true
            }
            None => false,
        }
    }
}
