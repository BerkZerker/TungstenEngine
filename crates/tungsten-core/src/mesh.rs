//! Surface Nets meshing of one node from its padded block of samples.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};

pub use crate::coords::APRON;
use crate::coords::{self, CHUNK_SIZE, SOLID};
use crate::field::LocalField;
use crate::material;
use crate::water::WaterPatch;

const N: i32 = CHUNK_SIZE;
const PAD: usize = coords::PAD as usize;
/// Cells per axis that can get a vertex: the chunk's own plus two layers
/// below and one above, which a mesh with a margin reaches into.
const CELLS: usize = (N + 3) as usize;
/// Vertex position fixed-point scale: units per cell.
pub const POS_SCALE: u32 = 1024;
const NO_VERTEX: u16 = u16::MAX;

#[inline]
fn pidx(x: i32, y: i32, z: i32) -> usize {
    ((y + APRON) as usize * PAD + (z + APRON) as usize) * PAD + (x + APRON) as usize
}

/// GPU vertex, 12 bytes.
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
#[repr(C)]
pub struct MeshVertex {
    /// Chunk-local position: `(cells + APRON) * POS_SCALE`.
    pub pos: [u16; 3],
    /// Material id in the low byte, sharpness (0..=255) in the high byte.
    pub mat_sharp: u16,
    /// Octahedral-encoded unit normal.
    pub normal: [i16; 2],
}

#[derive(Default)]
pub struct ChunkMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u16>,
    /// Water surfaces drawn with the node; see `water.rs`.
    pub water: Vec<WaterPatch>,
}

impl MeshVertex {
    /// Chunk-local position in cells.
    pub fn cells(&self) -> Vec3 {
        Vec3::new(self.pos[0] as f32, self.pos[1] as f32, self.pos[2] as f32) / POS_SCALE as f32 - APRON as f32
    }
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty() && self.water.is_empty()
    }
}

/// Encode a unit vector into two signed 16-bit values (octahedral mapping).
pub fn oct_encode(n: Vec3) -> [i16; 2] {
    let l1 = n.x.abs() + n.y.abs() + n.z.abs();
    let (mut x, mut y) = (n.x / l1, n.y / l1);
    if n.z < 0.0 {
        let (ox, oy) = (x, y);
        x = (1.0 - oy.abs()) * if ox >= 0.0 { 1.0 } else { -1.0 };
        y = (1.0 - ox.abs()) * if oy >= 0.0 { 1.0 } else { -1.0 };
    }
    [(x * 32767.0).round() as i16, (y * 32767.0).round() as i16]
}

pub fn oct_decode(e: [i16; 2]) -> Vec3 {
    let (x, y) = (e[0] as f32 / 32767.0, e[1] as f32 / 32767.0);
    let z = 1.0 - x.abs() - y.abs();
    let t = (-z).max(0.0);
    Vec3::new(x - t * x.signum(), y - t * y.signum(), z).normalize()
}

const CORNERS: [[i32; 3]; 8] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
/// Cell edges as corner index pairs.
const EDGES: [[usize; 2]; 12] =
    [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];

/// Bit mask of which of a cell's eight corners are solid.
#[inline]
pub fn solid_mask(f: &[u8; 8]) -> u32 {
    let mut mask = 0;
    for (c, &v) in f.iter().enumerate() {
        mask |= ((v >= SOLID) as u32) << c;
    }
    mask
}

/// Where the surface vertex of a cell sits, as a fraction of the cell,
/// snapped to the vertex grid: the mean of the surface crossings on the
/// cell's edges. `mask` must be mixed (neither 0 nor 0xff).
///
/// The signed value 2*fill-255 is odd, so a crossing is never exactly on a corner.
#[inline]
pub fn surface_point(f: &[u8; 8], mask: u32) -> Vec3 {
    let mut sum = Vec3::ZERO;
    let mut count = 0.0f32;
    for e in EDGES {
        let (a, b) = (e[0], e[1]);
        if ((mask >> a) ^ (mask >> b)) & 1 == 0 {
            continue;
        }
        let (sa, sb) = (2 * f[a] as i32 - 255, 2 * f[b] as i32 - 255);
        let t = sa as f32 / (sa - sb) as f32;
        let (pa, pb) = (CORNERS[a], CORNERS[b]);
        sum += Vec3::new(
            pa[0] as f32 + (pb[0] - pa[0]) as f32 * t,
            pa[1] as f32 + (pb[1] - pa[1]) as f32 * t,
            pa[2] as f32 + (pb[2] - pa[2]) as f32 * t,
        );
        count += 1.0;
    }
    (sum / count * POS_SCALE as f32 + 0.5).floor() / POS_SCALE as f32
}

/// Material shown at a cell's vertex: that of the solid corner most exposed
/// to air within the cell, so a thin layer shows as itself rather than as
/// whatever lies beneath it. Ties go to the emptier, then the lower corner.
#[inline]
pub fn surface_material(f: &[u8; 8], mask: u32, mat: impl Fn(usize) -> u8) -> u8 {
    let mut best = (0u32, 0u8, 0usize);
    for (c, &fill) in f.iter().enumerate() {
        if mask >> c & 1 == 0 {
            continue;
        }
        // Corners differing in one axis bit are joined by a cell edge.
        let exposed = (0..3).filter(|a| mask >> (c ^ (1 << a)) & 1 == 0).count() as u32;
        if exposed > best.0 || (exposed == best.0 && exposed > 0 && fill < best.1) {
            best = (exposed, fill, c);
        }
    }
    mat(best.2)
}

/// For each lattice axis: the four cells around an edge along it, as offsets
/// from the cell at the edge's lower sample, counter-clockwise seen from the
/// positive end of the axis.
pub const EDGE_CELLS: [[[i32; 3]; 4]; 3] = [
    [[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]],
    [[-1, 0, -1], [-1, 0, 0], [0, 0, 0], [0, 0, -1]],
    [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]],
];

/// Triangle corner order for a quad: the diagonal whose smaller triangle is
/// larger, which avoids slivers where one exists on only one diagonal.
#[inline]
pub fn quad_split(p: [Vec3; 4]) -> [usize; 6] {
    let area2 = |a: usize, b: usize, c: usize| (p[b] - p[a]).cross(p[c] - p[a]).length_squared();
    let first = area2(0, 1, 2).min(area2(0, 2, 3));
    let second = area2(1, 2, 3).min(area2(1, 3, 0));
    if first >= second { [0, 1, 2, 0, 2, 3] } else { [1, 2, 3, 1, 3, 0] }
}

#[inline]
fn cidx(x: i32, y: i32, z: i32) -> usize {
    ((y + 2) as usize * CELLS + (z + 2) as usize) * CELLS + (x + 2) as usize
}

/// Mesh the padded block of a node. `margin` (0 or 1) is how many cells the
/// mesh reaches beyond the chunk on every side: a coarse node overlaps its
/// neighbours by one cell so that no gap opens against a finer one.
pub fn mesh_block(block: &LocalField, margin: i32) -> ChunkMesh {
    debug_assert!(block.dims == IVec3::splat(coords::PAD) && (0..=1).contains(&margin));
    let mut mesh = ChunkMesh::default();
    if block.is_homogeneous() {
        return mesh;
    }
    let fill = &block.fill;
    // Position in cells of each shared vertex.
    let mut positions: Vec<Vec3> = Vec::new();
    let mut cell_vertex = vec![NO_VERTEX; CELLS * CELLS * CELLS];
    // Offsets from a cell's base sample to its eight corners.
    let corner_off: [usize; 8] = std::array::from_fn(|c| {
        CORNERS[c][1] as usize * PAD * PAD + CORNERS[c][2] as usize * PAD + CORNERS[c][0] as usize
    });

    let (lo, hi) = (-margin, N + margin);
    for y in lo - 1..hi {
        for z in lo - 1..hi {
            for x in lo - 1..hi {
                let base = pidx(x, y, z);
                let f: [u8; 8] = std::array::from_fn(|c| fill[base + corner_off[c]]);
                let mask = solid_mask(&f);
                if mask == 0 || mask == 0xff {
                    continue;
                }

                let frac = surface_point(&f, mask);

                // Quantise the in-cell fraction, then add the cell origin as
                // an integer: both chunks sharing a border cell get identical
                // bits, so seams are watertight.
                let q = frac * POS_SCALE as f32;
                let cell = IVec3::new(x, y, z);
                let raw = (cell + APRON).as_uvec3() * POS_SCALE + q.as_uvec3();

                // Normal from the fill gradient, interpolated at the vertex.
                let grad_at = |c: usize| {
                    let i = base + corner_off[c];
                    Vec3::new(
                        fill[i + 1] as f32 - fill[i - 1] as f32,
                        fill[i + PAD * PAD] as f32 - fill[i - PAD * PAD] as f32,
                        fill[i + PAD] as f32 - fill[i - PAD] as f32,
                    )
                };
                let gx0 = grad_at(0).lerp(grad_at(1), frac.x);
                let gx1 = grad_at(2).lerp(grad_at(3), frac.x);
                let gx2 = grad_at(4).lerp(grad_at(5), frac.x);
                let gx3 = grad_at(6).lerp(grad_at(7), frac.x);
                let g = gx0.lerp(gx1, frac.y).lerp(gx2.lerp(gx3, frac.y), frac.z);
                let normal = if g.length_squared() > 1e-6 { -g.normalize() } else { Vec3::Y };

                let mat = surface_material(&f, mask, |c| block.mat[base + corner_off[c]]);
                let sharp = (material::def(mat).sharpness * 255.0) as u16;

                cell_vertex[cidx(x, y, z)] = mesh.vertices.len() as u16;
                mesh.vertices.push(MeshVertex {
                    pos: [raw.x as u16, raw.y as u16, raw.z as u16],
                    mat_sharp: mat as u16 | sharp << 8,
                    normal: oct_encode(normal),
                });
                positions.push(cell.as_vec3() + frac);
            }
        }
    }

    // One quad per owned lattice edge that crosses the surface. An edge is
    // owned by the chunk holding its lower sample; a margin adds the edges of
    // one more layer of samples all round.
    //
    // A quad is the face of the solid sample on its edge and takes that
    // sample's material whole, so materials meet at a hard line. A shared
    // vertex of another material is copied for it, once per material.
    //
    // Triangles of faceted materials get three vertices of their own carrying
    // the face normal, so they shade flat with no per-pixel derivatives.
    let mut variants = vec![[NO_VERTEX; 4]; mesh.vertices.len()];
    for y in lo..hi {
        for z in lo..hi {
            for x in lo..hi {
                let i = pidx(x, y, z);
                let s0 = fill[i] >= SOLID;
                for (off, cells) in [1, PAD * PAD, PAD].into_iter().zip(EDGE_CELLS) {
                    let s1 = fill[i + off] >= SOLID;
                    if s0 == s1 {
                        continue;
                    }
                    let v: [u16; 4] =
                        std::array::from_fn(|k| cell_vertex[cidx(x + cells[k][0], y + cells[k][1], z + cells[k][2])]);
                    debug_assert!(v.iter().all(|&i| i != NO_VERTEX));
                    let mut order = quad_split(std::array::from_fn(|k| positions[v[k] as usize]));
                    if !s0 {
                        order.reverse();
                    }
                    let tri = |t: usize| -> [u16; 3] { std::array::from_fn(|k| v[order[t * 3 + k]]) };
                    let normal = |t: [u16; 3]| {
                        let [a, b, c] = t.map(|i| positions[i as usize]);
                        (b - a).cross(c - a)
                    };
                    let normals = [normal(tri(0)), normal(tri(1))];
                    let mat = block.mat[if s0 { i } else { i + off }];
                    let def = material::def(mat);
                    let mat_sharp = mat as u16 | ((def.sharpness * 255.0) as u16) << 8;
                    for t in 0..2 {
                        let ids = tri(t);
                        let room = mesh.vertices.len() + 3 <= NO_VERTEX as usize;
                        if def.sharpness < 0.5 || !room {
                            for id in ids {
                                let slot = &mut variants[id as usize][mat as usize & 3];
                                if mesh.vertices[id as usize].mat_sharp == mat_sharp {
                                    *slot = id;
                                } else if *slot == NO_VERTEX && room {
                                    *slot = mesh.vertices.len() as u16;
                                    mesh.vertices.push(MeshVertex { mat_sharp, ..mesh.vertices[id as usize] });
                                }
                                // Out of indices: blend rather than drop the triangle.
                                mesh.indices.push(if *slot == NO_VERTEX { id } else { *slot });
                            }
                            continue;
                        }
                        // A sliver has no stable normal of its own: borrow its
                        // partner's, or fall back to the smooth one.
                        let n = [normals[t], normals[1 - t]].into_iter().find(|n| n.length_squared() > 1e-8);
                        for id in ids {
                            let mut vertex = MeshVertex { mat_sharp, ..mesh.vertices[id as usize] };
                            if let Some(n) = n {
                                vertex.normal = oct_encode(n.normalize());
                            }
                            mesh.indices.push(mesh.vertices.len() as u16);
                            mesh.vertices.push(vertex);
                        }
                    }
                }
            }
        }
    }

    // Drop shared vertices nothing refers to (apron cells, faceted or recoloured ones).
    let mut remap = vec![NO_VERTEX; mesh.vertices.len()];
    for &i in &mesh.indices {
        remap[i as usize] = 0;
    }
    let mut kept = 0u16;
    for (i, slot) in remap.iter_mut().enumerate() {
        if *slot == 0 {
            *slot = kept;
            mesh.vertices[kept as usize] = mesh.vertices[i];
            kept += 1;
        }
    }
    mesh.vertices.truncate(kept as usize);
    for i in &mut mesh.indices {
        *i = remap[*i as usize];
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Dense, Sample, Volume};
    use crate::coords::{NodeKey, chunk_origin, index_xyz};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn block_of(vol: &Volume, pos: IVec3) -> LocalField {
        let chunks: [Arc<Chunk>; 27] = std::array::from_fn(|i| {
            let i = i as i32;
            vol.chunks[&(pos + IVec3::new(i % 3, i / 9, i / 3 % 3) - 1)].clone()
        });
        LocalField::gather(NodeKey::new(0, pos), &chunks)
    }

    /// A ball of stone spanning the eight chunks around the origin.
    fn ball_volume(radius: f32, center: Vec3) -> Volume {
        let mut vol = Volume::new();
        for cy in -2..2 {
            for cz in -2..2 {
                for cx in -2..2 {
                    let pos = IVec3::new(cx, cy, cz);
                    let o = chunk_origin(pos);
                    let mut d = Dense::new_boxed(Sample::AIR);
                    for y in 0..N {
                        for z in 0..N {
                            for x in 0..N {
                                let p = (o + IVec3::new(x, y, z)).as_vec3();
                                let dist = radius - p.distance(center);
                                let f = (127.5 + dist * 64.0).clamp(0.0, 255.0) as u8;
                                let i = index_xyz(x, y, z);
                                d.fill[i] = f;
                                d.mat[i] = if f > 0 { 1 } else { 0 };
                            }
                        }
                    }
                    let mut c = Chunk::Dense(d);
                    c.normalize();
                    vol.insert(pos, c);
                }
            }
        }
        vol
    }

    #[test]
    fn oct_round_trip() {
        for v in [Vec3::X, Vec3::NEG_Y, Vec3::new(0.3, -0.5, -0.8).normalize(), Vec3::NEG_Z] {
            assert!(oct_decode(oct_encode(v)).dot(v) > 0.9999);
        }
    }

    #[test]
    fn ball_is_watertight_across_chunks_and_faces_outward() {
        let center = Vec3::new(0.7, -1.3, 2.1);
        let vol = ball_volume(20.0, center);
        // Directed edge -> count, keyed by world-space quantised positions.
        let mut edges: HashMap<([i32; 3], [i32; 3]), i32> = HashMap::new();
        let mut tris = 0;
        for cy in -1..1 {
            for cz in -1..1 {
                for cx in -1..1 {
                    let pos = IVec3::new(cx, cy, cz);
                    let mesh = mesh_block(&block_of(&vol, pos), 0);
                    let world = |i: u16| {
                        let v = mesh.vertices[i as usize].pos;
                        let o = (chunk_origin(pos) - APRON) * POS_SCALE as i32;
                        [o.x + v[0] as i32, o.y + v[1] as i32, o.z + v[2] as i32]
                    };
                    for t in mesh.indices.chunks(3) {
                        tris += 1;
                        for k in 0..3 {
                            *edges.entry((world(t[k]), world(t[(k + 1) % 3]))).or_default() += 1;
                        }
                        // Winding agrees with the outward direction and the stored normal.
                        let p: Vec<Vec3> = t
                            .iter()
                            .map(|&i| mesh.vertices[i as usize].cells() + chunk_origin(pos).as_vec3())
                            .collect();
                        let face = (p[1] - p[0]).cross(p[2] - p[0]);
                        let out = (p[0] + p[1] + p[2]) / 3.0 - center;
                        assert!(face.dot(out) > 0.0, "triangle faces inward");
                        let n = oct_decode(mesh.vertices[t[0] as usize].normal);
                        assert!(n.dot(out.normalize()) > 0.9, "normal not outward");
                        // The ball is stone: every triangle is flat shaded.
                        let n1 = oct_decode(mesh.vertices[t[1] as usize].normal);
                        assert!(n.dot(n1) > 0.9999 && n.dot(face.normalize()) > 0.999, "not flat shaded");
                    }
                }
            }
        }
        assert!(tris > 5000);
        for (&(a, b), &n) in &edges {
            assert_eq!(n, 1, "edge used {n} times in one direction");
            assert_eq!(edges.get(&(b, a)), Some(&1), "open edge: mesh is not watertight");
        }
    }

    #[test]
    fn homogeneous_blocks_are_empty() {
        let mut vol = Volume::new();
        for cy in -1..=1 {
            for cz in -1..=1 {
                for cx in -1..=1 {
                    vol.insert(IVec3::new(cx, cy, cz), Chunk::Uniform(Sample { fill: 255, mat: 1 }));
                }
            }
        }
        assert!(mesh_block(&block_of(&vol, IVec3::ZERO), 0).is_empty());
        assert!(mesh_block(&block_of(&vol, IVec3::ZERO), 1).is_empty());
    }

    #[test]
    fn a_margin_adds_a_border_and_changes_nothing_else() {
        let vol = ball_volume(20.0, Vec3::new(0.7, -1.3, 2.1));
        let block = block_of(&vol, IVec3::new(-1, 0, 0));
        let triangles = |margin: i32| -> HashSet<[[u16; 3]; 3]> {
            let mesh = mesh_block(&block, margin);
            mesh.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]].map(|i| mesh.vertices[i as usize].pos)).collect()
        };
        let (own, wide) = (triangles(0), triangles(1));
        assert!(own.is_subset(&wide));
        assert!(wide.len() > own.len() + 100, "{} -> {}", own.len(), wide.len());
        // The extra triangles reach past the chunk on both sides.
        let span = |set: &HashSet<[[u16; 3]; 3]>| {
            let xs = set.iter().flatten().map(|p| p[0]);
            (xs.clone().min().unwrap(), xs.max().unwrap())
        };
        let ((a0, a1), (b0, b1)) = (span(&own), span(&wide));
        assert!(b0 < a0 && b1 > a1 && a0 - b0 <= POS_SCALE as u16 * 2);
    }

    #[test]
    fn every_triangle_is_one_material() {
        // The stone ball with a band of dirt and a cap of sand.
        let center = Vec3::new(0.7, -1.3, 2.1);
        let mut vol = ball_volume(20.0, center);
        let keys: Vec<IVec3> = vol.chunks.keys().copied().collect();
        for pos in keys {
            let o = chunk_origin(pos);
            for i in 0..crate::CHUNK_VOLUME as i32 {
                let p = o + IVec3::new(i & 31, i >> 10, (i >> 5) & 31);
                let s = vol.get(p).unwrap();
                if s.fill > 0 {
                    let mat = if p.y > 12 {
                        3
                    } else if p.y > 2 {
                        2
                    } else {
                        1
                    };
                    vol.set(p, Sample { fill: s.fill, mat });
                }
            }
        }
        let mut per_material = [0usize; 4];
        for cy in -1..1 {
            for cz in -1..1 {
                for cx in -1..1 {
                    let pos = IVec3::new(cx, cy, cz);
                    let mesh = mesh_block(&block_of(&vol, pos), 0);
                    for t in mesh.indices.chunks_exact(3) {
                        let m = [t[0], t[1], t[2]].map(|i| mesh.vertices[i as usize].mat_sharp & 0xff);
                        assert!(m[0] == m[1] && m[1] == m[2], "triangle blends materials {m:?}");
                        per_material[m[0] as usize] += 1;
                    }
                }
            }
        }
        assert!(per_material[1..].iter().all(|&n| n > 100), "{per_material:?}");
    }
}
