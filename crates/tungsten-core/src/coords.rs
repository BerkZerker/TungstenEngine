use glam::IVec3;

pub const CHUNK_BITS: i32 = 5;
/// Cells (and owned samples) per chunk axis.
pub const CHUNK_SIZE: i32 = 1 << CHUNK_BITS;
pub const CHUNK_MASK: i32 = CHUNK_SIZE - 1;
pub const CHUNK_VOLUME: usize = (CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE) as usize;

/// Fill value at and above which a sample is solid.
pub const SOLID: u8 = 128;
/// Fill change per cell of distance, at whatever level the cell belongs to.
pub const FILL_PER_CELL: i32 = 64;

/// Samples of apron on each side of a node's block: two for the cells a
/// mesh may reach into beyond the chunk, one for gradients.
pub const APRON: i32 = 3;
/// Samples per axis of a node's padded block.
pub const PAD: i32 = CHUNK_SIZE + 2 * APRON;

/// Levels of detail. Level `L` samples the world every `2^L` base cells.
pub const LEVELS: usize = 6;
pub const MAX_LEVEL: u8 = LEVELS as u8 - 1;

/// Chunk coordinate at some level. Sample coordinates of that level are
/// `chunk * CHUNK_SIZE + local`.
pub type ChunkPos = IVec3;

/// One chunk of one level of detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub level: u8,
    pub pos: ChunkPos,
}

impl NodeKey {
    pub const fn new(level: u8, pos: ChunkPos) -> Self {
        Self { level, pos }
    }

    /// Base cells between this node's samples.
    #[inline]
    pub fn step(self) -> i32 {
        1 << self.level
    }

    /// Base-cell coordinate of the node's first own sample.
    #[inline]
    pub fn origin(self) -> IVec3 {
        chunk_origin(self.pos) * self.step()
    }

    /// Edge length in base cells.
    #[inline]
    pub fn size(self) -> i32 {
        CHUNK_SIZE << self.level
    }

    pub fn parent(self) -> NodeKey {
        NodeKey { level: self.level + 1, pos: self.pos >> 1 }
    }

    /// The eight nodes one level finer that fill this one. Level 0 has none.
    pub fn children(self) -> impl Iterator<Item = NodeKey> {
        let (level, base) = (self.level.saturating_sub(1), self.pos * 2);
        (0..8).map(move |i| NodeKey { level, pos: base + IVec3::new(i & 1, i >> 2, (i >> 1) & 1) })
    }
}

/// Per-world constants. Saves are only valid for the parameters they were made with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldParams {
    pub seed: u64,
    /// Metres between lattice samples at the finest level.
    pub cell_size: f32,
    /// Breaking leaves no isolated piece smaller than a cube of this many
    /// metres a side; smaller ones break off with it. Zero keeps everything.
    pub min_fragment: f32,
}

impl Default for WorldParams {
    fn default() -> Self {
        Self { seed: 0x7475_6e67_7374_656e, cell_size: 0.25, min_fragment: 1.0 }
    }
}

impl WorldParams {
    /// Edge length of a finest-level chunk in metres.
    pub fn chunk_metres(&self) -> f32 {
        self.cell_size * CHUNK_SIZE as f32
    }

    /// Metres between samples at `level`.
    pub fn spacing(&self, level: u8) -> f32 {
        self.cell_size * (1u32 << level) as f32
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn children_fill_their_parent() {
        let node = NodeKey::new(3, IVec3::new(-2, 1, 5));
        let mut seen = 0;
        for c in node.children() {
            assert_eq!(c.parent(), node);
            assert_eq!(c.size() * 2, node.size());
            let rel = c.origin() - node.origin();
            assert!(rel.min_element() >= 0 && rel.max_element() <= c.size());
            seen += 1;
        }
        assert_eq!(seen, 8);
    }
}
