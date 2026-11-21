use glam::IVec3;

pub const CHUNK_BITS: i32 = 5;
/// Cells (and owned samples) per chunk axis.
pub const CHUNK_SIZE: i32 = 1 << CHUNK_BITS;
pub const CHUNK_MASK: i32 = CHUNK_SIZE - 1;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Fill value at and above which a sample is solid.
pub const SOLID: u8 = 128;
/// Fill change per cell of distance in generated terrain and brush profiles.
pub const FILL_PER_CELL: i32 = 64;

/// Chunk coordinate. Sample coordinates are `chunk * CHUNK_SIZE + local`.
pub type ChunkPos = IVec3;

/// Per-world constants. Saves are only valid for the parameters they were made with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldParams {
    pub seed: u64,
    /// Metres between lattice samples.
    pub cell_size: f32,
    /// Breaking leaves no isolated piece smaller than a cube of this many
    /// metres a side; smaller ones break off with it. Zero keeps everything.
    pub min_fragment: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self { seed: 0x7475_6e67_7374_656e, cell_size: 0.5, min_fragment: 1.0 }
    }
}

impl WorldParams {
    /// Edge length of a chunk in metres.
    pub fn chunk_metres(&self) -> f32 {
        self.cell_size * CHUNK_SIZE as f32
    }
}

#[inline]
pub fn chunk_of(sample: IVec3) -> ChunkPos {
    IVec3::new(sample.x >> CHUNK_BITS, sample.y >> CHUNK_BITS, sample.z >> CHUNK_BITS)
}

#[inline]
pub fn local_of(sample: IVec3) -> IVec3 {
    IVec3::new(sample.x & CHUNK_MASK, sample.y & CHUNK_MASK, sample.z & CHUNK_MASK)
}

#[inline]
pub fn chunk_origin(chunk: ChunkPos) -> IVec3 {
    IVec3::new(chunk.x << CHUNK_BITS, chunk.y << CHUNK_BITS, chunk.z << CHUNK_BITS)
}

/// Index of a local sample. Y is the outermost axis so a vertical neighbour is
/// one `CHUNK_SIZE * CHUNK_SIZE` stride away.
#[inline]
pub fn index(local: IVec3) -> usize {
    debug_assert!(local.min_element() >= 0 && local.max_element() < CHUNK_SIZE);
    (((local.y << CHUNK_BITS) | local.z) << CHUNK_BITS | local.x) as usize
}

#[inline]
pub fn index_xyz(x: i32, y: i32, z: i32) -> usize {
    (((y << CHUNK_BITS) | z) << CHUNK_BITS | x) as usize
}
