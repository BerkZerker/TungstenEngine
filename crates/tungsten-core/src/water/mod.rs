//! Water as a function: bodies with one flat surface each, not a grid of cells.
//!
//! The ocean is a body with a fixed level: open ground below the sea level
//! is under it, decided from the generator's height function alone.
//!
//! This file holds what a node needs to draw the surfaces that pass through
//! it. Surfaces are flat, so a node only says where each one is (a mask over
//! its cells, merged into rectangles); the height comes from the body when
//! the frame is drawn.

pub mod bodies;
pub mod cells;
pub mod flood;

use crate::coords::{CHUNK_SIZE, NodeKey};
use crate::worldgen::{Generator, SEA_LEVEL};

/// A connected piece of water with one surface level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BodyId {
    /// The sea: every part of it, at one fixed level.
    Ocean,
    /// A finite body made by pouring, numbered in the order they were made.
    Pond(u32),
}

/// The part of one body's surface that a node draws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaterPatch {
    pub body: BodyId,
    /// Surface height when the patch was made, fixed-point base cells.
    pub level: i64,
    /// Corners in cells of the node, `0..=CHUNK_SIZE` on X and Z.
    pub vertices: Vec<[u16; 2]>,
    pub indices: Vec<u16>,
}

const N: usize = CHUNK_SIZE as usize;

/// Which cells of a node a surface covers: bit `x` of row `z`.
pub type Mask = [u32; N];

/// Cover a mask with rectangles, two triangles each, facing up.
pub fn mesh_mask(mut mask: Mask) -> (Vec<[u16; 2]>, Vec<u16>) {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    for z0 in 0..N {
        while mask[z0] != 0 {
            let x0 = mask[z0].trailing_zeros();
            let width = (mask[z0] >> x0).trailing_ones();
            let run = if width == 32 { u32::MAX } else { ((1u32 << width) - 1) << x0 };
            let mut z1 = z0;
            while z1 < N && mask[z1] & run == run {
                mask[z1] &= !run;
                z1 += 1;
            }
            let base = vertices.len() as u16;
            let (x0, x1, z0, z1) = (x0 as u16, (x0 + width) as u16, z0 as u16, z1 as u16);
            vertices.extend([[x0, z0], [x0, z1], [x1, z1], [x1, z0]]);
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    (vertices, indices)
}

/// The cells of a node with open ground below sea level at any corner. The
/// terrain itself hides what part of such a cell is dry.
fn ocean_mask(generator: &Generator, key: NodeKey) -> Mask {
    let heights = generator.heights(key.level, key.pos.x, key.pos.z);
    let mut mask = [0u32; N];
    if heights.min() >= SEA_LEVEL {
        return mask;
    }
    let mut above = [false; N + 1];
    for (x, a) in above.iter_mut().enumerate() {
        *a = heights.at(x as i32, 0) < SEA_LEVEL;
    }
    for (z, row) in mask.iter_mut().enumerate() {
        let mut below = [false; N + 1];
        for (x, b) in below.iter_mut().enumerate() {
            *b = heights.at(x as i32, z as i32 + 1) < SEA_LEVEL;
        }
        for x in 0..N {
            *row |= ((above[x] || above[x + 1] || below[x] || below[x + 1]) as u32) << x;
        }
        above = below;
    }
    mask
}

/// Part of a surface that is known by its cells rather than by a rule: the
/// wet lattice columns at its level, in the chunk columns near a node.
#[derive(Clone, Debug)]
pub struct Surface {
    pub body: BodyId,
    pub level: i64,
    /// Chunk column `(x, z)` at the finest level, and its wet columns.
    pub tiles: Vec<((i32, i32), Mask)>,
}

/// The water a node has to draw that the generator knows nothing about.
#[derive(Clone, Debug, Default)]
pub struct NodeWater {
    pub surfaces: Vec<Surface>,
}

/// The cells of a node with a wet lattice column at any corner.
fn cover(key: NodeKey, tiles: &[((i32, i32), Mask)]) -> Mask {
    let (o, step) = (key.origin(), key.step());
    let mut mask = [0u32; N];
    // The node's cells on one axis that touch a lattice column: at most two.
    let touched = |r: i32| {
        let (a, b) = ((r - 1).div_euclid(step), r.div_euclid(step));
        (a.max(0), b.min(N as i32 - 1))
    };
    for &((cx, cz), tile) in tiles {
        for (z, &row) in tile.iter().enumerate() {
            let (z0, z1) = touched(cz * CHUNK_SIZE + z as i32 - o.z);
            let mut bits = row;
            while bits != 0 && z0 <= z1 {
                let x = bits.trailing_zeros() as i32;
                bits &= bits - 1;
                let (x0, x1) = touched(cx * CHUNK_SIZE + x - o.x);
                for z in z0..=z1 {
                    for x in x0..=x1 {
                        mask[z as usize] |= 1 << x;
                    }
                }
            }
        }
    }
    mask
}

/// Water surfaces to draw with a node. A surface belongs to the node that
/// holds its [`anchor`](flood::anchor) layer, so stacked nodes never draw it twice.
pub fn node_patches(generator: &Generator, key: NodeKey, water: &NodeWater) -> Vec<WaterPatch> {
    let mut patches = Vec::new();
    let anchored = flood::anchor(bodies::SEA).div_euclid(key.size()) == key.pos.y;
    let mut sea = if anchored { ocean_mask(generator, key) } else { [0; N] };
    for surface in &water.surfaces {
        let mask = cover(key, &surface.tiles);
        if surface.body == BodyId::Ocean {
            sea.iter_mut().zip(mask).for_each(|(row, more)| *row |= more);
            continue;
        }
        let (vertices, indices) = mesh_mask(mask);
        if !indices.is_empty() {
            patches.push(WaterPatch { body: surface.body, level: surface.level, vertices, indices });
        }
    }
    let (vertices, indices) = mesh_mask(sea);
    if !indices.is_empty() {
        patches.push(WaterPatch { body: BodyId::Ocean, level: bodies::SEA, vertices, indices });
    }
    patches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{MAX_LEVEL, WorldParams};
    use glam::IVec3;

    /// Cells covered by a mesh, counting how often each is covered.
    fn coverage(vertices: &[[u16; 2]], indices: &[u16]) -> [[u8; N]; N] {
        let mut cover = [[0u8; N]; N];
        for quad in indices.chunks_exact(6) {
            let (a, c) = (vertices[quad[0] as usize], vertices[quad[2] as usize]);
            for row in &mut cover[a[1] as usize..c[1] as usize] {
                for cell in &mut row[a[0] as usize..c[0] as usize] {
                    *cell += 1;
                }
            }
        }
        cover
    }

    #[test]
    fn rectangles_cover_a_mask_exactly_once() {
        let mut mask = [0u32; N];
        for (z, row) in mask.iter_mut().enumerate() {
            let h = crate::hash::hash2(7, z as i64, 0);
            *row = (h as u32) | ((h >> 32) as u32) | if z % 5 == 0 { u32::MAX } else { 0 };
        }
        mask[9] = 0;
        let (vertices, indices) = mesh_mask(mask);
        let cover = coverage(&vertices, &indices);
        for (z, row) in cover.iter().enumerate() {
            for (x, &times) in row.iter().enumerate() {
                assert_eq!(times as u32, mask[z] >> x & 1, "cell {x} {z}");
            }
        }
        // A full node is one rectangle.
        let (vertices, indices) = mesh_mask([u32::MAX; N]);
        assert_eq!((vertices.len(), indices.len()), (4, 6));
        assert!(mesh_mask([0; N]).1.is_empty());
    }

    #[test]
    fn the_sea_is_drawn_where_open_ground_is_below_it_at_every_level() {
        let g = Generator::new(WorldParams::default());
        let cell = g.params().cell_size as f64;
        // A stretch across the shore 44 m east of the origin.
        for level in 0..=MAX_LEVEL {
            let spacing = cell * (1u64 << level) as f64;
            let (mut wet, mut dry) = (0, 0);
            let shore = (44.0 / (spacing * N as f64)).floor() as i32;
            for cx in shore - 2..=shore + 2 {
                // The node just under the surface draws it, and no other.
                let key = NodeKey::new(level, IVec3::new(cx, -1, 0));
                let patches = node_patches(&g, key, &NodeWater::default());
                assert!(node_patches(&g, NodeKey::new(level, IVec3::new(cx, 0, 0)), &NodeWater::default()).is_empty());
                assert!(node_patches(&g, NodeKey::new(level, IVec3::new(cx, -2, 0)), &NodeWater::default()).is_empty());
                let cover = match patches.as_slice() {
                    [] => [[0u8; N]; N],
                    [p] => {
                        assert_eq!((p.body, p.level), (BodyId::Ocean, 0));
                        coverage(&p.vertices, &p.indices)
                    }
                    more => panic!("{} patches for one body", more.len()),
                };
                for (z, row) in cover.iter().enumerate() {
                    for (x, &times) in row.iter().enumerate() {
                        let at = |dx: usize, dz: usize| {
                            let wx = (cx as i64 * N as i64 + (x + dx) as i64) as f64 * spacing;
                            g.surface_height(wx, (z + dz) as f64 * spacing)
                        };
                        let under = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)].iter().any(|&h| h < SEA_LEVEL);
                        assert_eq!(times == 1, under, "level {level} node {cx} cell {x} {z}");
                        wet += under as u32;
                        dry += !under as u32;
                    }
                }
            }
            assert!(wet > 200 && dry > 200, "level {level}: {wet} wet, {dry} dry");
        }
    }
}
