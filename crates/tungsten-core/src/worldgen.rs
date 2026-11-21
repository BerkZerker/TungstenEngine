//! World generation: a pure function from `(seed, chunk)` to samples.
//!
//! Part of the bit-exact contract. Float use is restricted to `+ - * /`,
//! `sqrt`, `floor` and comparisons; see `noise.rs`.

use std::sync::{Arc, Mutex};

use crate::chunk::{Chunk, Dense, Sample};
use crate::coords::{CHUNK_SIZE, ChunkPos, FILL_PER_CELL, WorldParams, index_xyz};
use crate::hash::{IntMap, derive};
use crate::material::Material;
use crate::noise::{fbm2, noise2, noise3, ridged2, smoothstep};

/// Bump whenever generated output changes; saved deltas are tied to it.
pub const GENERATOR_VERSION: u32 = 2;

/// Mountains: metres between ranges, between ridges, and peak height.
const MASK_WAVE: f64 = 800.0;
const MASK_FROM: f32 = -0.05;
const MASK_TO: f32 = 0.3;
const RIDGE_WAVE: f64 = 450.0;
const MOUNTAIN_HEIGHT: f32 = 170.0;

const N: usize = CHUNK_SIZE as usize;
const AREA: usize = N * N;
/// Cells between points of the coarse lattice that 3D noise is evaluated on.
const COARSE: usize = 4;
const LAT: usize = N / COARSE + 1;
const LAT3: usize = LAT * LAT * LAT;

/// Ground below this height is sand, metres.
pub const SAND_LINE: f32 = 0.0;
/// Ground above this height is bare stone, metres. Dirt lies between the two.
pub const STONE_LINE: f32 = 40.0;
/// How far the two lines wander up and down, metres.
const LINE_WOBBLE: f32 = 3.0;

/// Largest cave radius in noise units; see `cave_radius`.
const CAVE_R_MAX: f32 = 0.09;
/// Metres of distance per unit of cave noise.
const CAVE_SCALE: f32 = 45.0;
/// Depth over which caves fade in below their opening depth, metres.
const CAVE_FADE: f32 = 6.0;

struct Seeds {
    cont: u64,
    hills: u64,
    mask: u64,
    ridge: u64,
    detail: u64,
    cliff: u64,
    band: u64,
    dirt: u64,
    open: u64,
    over: u64,
    cave1: u64,
    cave2: u64,
    cave3: u64,
}

/// Per-column terrain data shared by every chunk in a vertical stack.
struct Column {
    height: [f32; AREA],
    /// 1 / sqrt(1 + slope²): turns vertical depth into approximate distance.
    inv_grad: [f32; AREA],
    /// Amplitude of the 3D cliff/overhang term, metres.
    cliff: [f32; AREA],
    /// Thickness of the loose layer over the bedrock, metres, and what it is.
    cover: [f32; AREA],
    cover_mat: [u8; AREA],
    /// Depth at which caves start to fade in; negative lets them breach the surface.
    open: [f32; AREA],
    min_h: f32,
    max_h: f32,
    max_cliff: f32,
    min_inv_grad: f32,
    min_open: f32,
}

pub struct Generator {
    params: WorldParams,
    seeds: Seeds,
    columns: Mutex<IntMap<(i32, i32), Arc<Column>>>,
}

impl Generator {
    pub fn new(params: WorldParams) -> Self {
        let s = |n| derive(params.seed, n);
        Self {
            params,
            seeds: Seeds {
                cont: s(1),
                hills: s(2),
                mask: s(3),
                ridge: s(4),
                detail: s(5),
                cliff: s(6),
                band: s(7),
                dirt: s(9),
                open: s(10),
                over: s(11),
                cave1: s(12),
                cave2: s(13),
                cave3: s(14),
            },
            columns: Mutex::new(IntMap::default()),
        }
    }

    pub fn params(&self) -> WorldParams {
        self.params
    }

    /// Height of the base terrain surface in metres, before cliffs and caves.
    pub fn surface_height(&self, x: f64, z: f64) -> f32 {
        let s = &self.seeds;
        let cont = fbm2(s.cont, x / 1200.0, z / 1200.0, 4);
        let hills = fbm2(s.hills, x / 220.0, z / 220.0, 4);
        // Mountain ranges: ridged noise, let through where the mask is high.
        let mask = smoothstep(MASK_FROM, MASK_TO, fbm2(s.mask, x / MASK_WAVE, z / MASK_WAVE, 3));
        let ridge = ridged2(s.ridge, x / RIDGE_WAVE, z / RIDGE_WAVE, 4);
        let detail = fbm2(s.detail, x / 36.0, z / 36.0, 3);
        cont * 40.0 + hills * 14.0 + mask * ridge * MOUNTAIN_HEIGHT + detail * 1.2
    }

    fn column(&self, cx: i32, cz: i32) -> Arc<Column> {
        if let Some(c) = self.columns.lock().unwrap().get(&(cx, cz)) {
            return c.clone();
        }
        let col = Arc::new(self.build_column(cx, cz));
        let mut cache = self.columns.lock().unwrap();
        if cache.len() > 8192 {
            cache.clear();
        }
        cache.insert((cx, cz), col.clone());
        col
    }

    fn build_column(&self, cx: i32, cz: i32) -> Column {
        let s = &self.seeds;
        let cell = self.params.cell_size as f64;
        let (ox, oz) = (cx as i64 * N as i64, cz as i64 * N as i64);

        // Heights with a one-sample border for slope by central difference.
        const W: usize = N + 2;
        let mut h = [0.0f32; W * W];
        for z in 0..W {
            for x in 0..W {
                let wx = (ox + x as i64 - 1) as f64 * cell;
                let wz = (oz + z as i64 - 1) as f64 * cell;
                h[z * W + x] = self.surface_height(wx, wz);
            }
        }

        let mut col = Column {
            height: [0.0; AREA],
            inv_grad: [0.0; AREA],
            cliff: [0.0; AREA],
            cover: [0.0; AREA],
            cover_mat: [Material::STONE; AREA],
            open: [0.0; AREA],
            min_h: f32::MAX,
            max_h: f32::MIN,
            max_cliff: 0.0,
            min_inv_grad: 1.0,
            min_open: f32::MAX,
        };
        let inv_2cell = 0.5 / self.params.cell_size;
        for z in 0..N {
            for x in 0..N {
                let i = z * N + x;
                let c = (z + 1) * W + x + 1;
                let height = h[c];
                let gx = (h[c + 1] - h[c - 1]) * inv_2cell;
                let gz = (h[c + W] - h[c - W]) * inv_2cell;
                let slope2 = gx * gx + gz * gz;
                let inv_grad = 1.0 / (1.0 + slope2).sqrt();
                let wx = (ox + x as i64) as f64 * cell;
                let wz = (oz + z as i64) as f64 * cell;

                // One biome, banded by altitude: sand lowlands, dirt midlands,
                // stone highlands. The lines wander a little so they do not
                // trace exact contours.
                let level = height + LINE_WOBBLE * noise2(s.band, wx / 70.0, wz / 70.0);
                let (cover_mat, cover) = if level < SAND_LINE {
                    (Material::SAND, 5.0)
                } else if level < STONE_LINE {
                    (Material::DIRT, 3.0 + 1.5 * noise2(s.dirt, wx / 50.0, wz / 50.0))
                } else {
                    (Material::STONE, 0.0)
                };

                // Cliffs and overhangs belong to the highlands; fading them in
                // above the stone line keeps bedrock from showing through dirt.
                let cliff = 9.0
                    * smoothstep(0.15, 0.6, noise2(s.cliff, wx / 260.0, wz / 260.0))
                    * smoothstep(STONE_LINE + 4.0, STONE_LINE + 24.0, level);

                let open = 7.0 - 10.0 * smoothstep(0.4, 0.55, noise2(s.open, wx / 140.0, wz / 140.0));

                col.height[i] = height;
                col.inv_grad[i] = inv_grad;
                col.cliff[i] = cliff;
                col.cover[i] = cover;
                col.cover_mat[i] = cover_mat;
                col.open[i] = open;
                col.min_h = col.min_h.min(height);
                col.max_h = col.max_h.max(height);
                col.max_cliff = col.max_cliff.max(cliff);
                col.min_inv_grad = col.min_inv_grad.min(inv_grad);
                col.min_open = col.min_open.min(open);
            }
        }
        col
    }

    /// 3D noise on the coarse lattice of a chunk. Lattice points on chunk
    /// faces are shared with the neighbour, so interpolation is seamless.
    fn lattice(&self, seed: u64, origin: [i64; 3], wave: [f64; 3]) -> [f32; LAT3] {
        let cell = self.params.cell_size as f64;
        let mut out = [0.0f32; LAT3];
        for ly in 0..LAT {
            let y = (origin[1] + (ly * COARSE) as i64) as f64 * cell / wave[1];
            for lz in 0..LAT {
                let z = (origin[2] + (lz * COARSE) as i64) as f64 * cell / wave[2];
                for lx in 0..LAT {
                    let x = (origin[0] + (lx * COARSE) as i64) as f64 * cell / wave[0];
                    out[(ly * LAT + lz) * LAT + lx] = noise3(seed, x, y, z);
                }
            }
        }
        out
    }

    pub fn generate(&self, pos: ChunkPos) -> Chunk {
        let cell = self.params.cell_size;
        let col = self.column(pos.x, pos.z);
        let origin = [pos.x as i64 * N as i64, pos.y as i64 * N as i64, pos.z as i64 * N as i64];
        let y_min = (origin[1] as f64 * cell as f64) as f32;
        let y_max = ((origin[1] + N as i64 - 1) as f64 * cell as f64) as f32;
        let band = 2.1 * cell;

        // Entirely above anything the surface terms can reach: air.
        if (y_min - col.max_h) * col.min_inv_grad > 1.1 * col.max_cliff + band {
            return Chunk::AIR;
        }

        // So deep that the surface terms saturate; only caves can matter.
        let top_depth = col.min_h - y_max;
        let deep = top_depth * col.min_inv_grad > 1.1 * col.max_cliff + band && top_depth > 14.0;
        let need_over = !deep && col.max_cliff > 0.0;
        let need_cave = col.max_h - y_min > col.min_open;

        let over = if need_over { self.lattice(self.seeds.over, origin, [34.0, 24.0, 34.0]) } else { [0.0; LAT3] };
        let (c1, c2, c3) = if need_cave {
            (
                self.lattice(self.seeds.cave1, origin, [90.0, 55.0, 90.0]),
                self.lattice(self.seeds.cave2, origin, [90.0, 55.0, 90.0]),
                self.lattice(self.seeds.cave3, origin, [260.0, 160.0, 260.0]),
            )
        } else {
            ([0.0; LAT3], [0.0; LAT3], [0.0; LAT3])
        };

        if deep && !may_have_cave(&c1, &c2) {
            return Chunk::Uniform(Sample { fill: 255, mat: Material::STONE });
        }

        let mut dense = Dense::new_boxed(Sample::AIR);
        let fill_per_metre = FILL_PER_CELL as f32 / cell;
        let (mut r_over, mut r1, mut r2, mut r3) = ([0.0f32; LAT], [0.0f32; LAT], [0.0f32; LAT], [0.0f32; LAT]);
        for y in 0..N {
            let ym = (origin[1] + y as i64) as f64 * cell as f64;
            for z in 0..N {
                if need_over {
                    lattice_row(&over, y, z, &mut r_over);
                }
                if need_cave {
                    lattice_row(&c1, y, z, &mut r1);
                    lattice_row(&c2, y, z, &mut r2);
                    lattice_row(&c3, y, z, &mut r3);
                }
                for x in 0..N {
                    let ci = z * N + x;
                    let depth = (col.height[ci] as f64 - ym) as f32;
                    let mut d = depth * col.inv_grad[ci];
                    if need_over {
                        d += col.cliff[ci] * row_at(&r_over, x);
                    }
                    if need_cave {
                        let fade = ((depth - col.open[ci]) / CAVE_FADE).clamp(0.0, 1.0);
                        if fade > 0.0 {
                            let (a, b) = (row_at(&r1, x), row_at(&r2, x));
                            let r = cave_radius(row_at(&r3, x)) * fade;
                            d = d.min(((a * a + b * b).sqrt() - r) * CAVE_SCALE);
                        }
                    }
                    let fill = ((127.5 + d * fill_per_metre).clamp(0.0, 255.0) + 0.5) as u8;
                    if fill == 0 {
                        continue;
                    }
                    let mat = if depth < col.cover[ci] { col.cover_mat[ci] } else { Material::STONE };
                    let i = index_xyz(x as i32, y as i32, z as i32);
                    dense.fill[i] = fill;
                    dense.mat[i] = mat;
                }
            }
        }
        let mut chunk = Chunk::Dense(dense);
        chunk.normalize();
        chunk
    }
}

#[inline]
fn cave_radius(c3: f32) -> f32 {
    (0.05 + 0.035 * c3).clamp(0.0, CAVE_R_MAX)
}

/// Conservative test: can any interpolated point of these lattices be inside a cave?
fn may_have_cave(c1: &[f32; LAT3], c2: &[f32; LAT3]) -> bool {
    let limit = CAVE_R_MAX * CAVE_R_MAX;
    let min_abs = |lat: &[f32; LAT3], base: usize| {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for dy in 0..2 {
            for dz in 0..2 {
                for dx in 0..2 {
                    let v = lat[base + (dy * LAT + dz) * LAT + dx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) }
    };
    for ly in 0..LAT - 1 {
        for lz in 0..LAT - 1 {
            for lx in 0..LAT - 1 {
                let base = (ly * LAT + lz) * LAT + lx;
                let (a, b) = (min_abs(c1, base), min_abs(c2, base));
                if a * a + b * b <= limit {
                    return true;
                }
            }
        }
    }
    false
}

/// Interpolate a lattice along Y and Z, leaving one value per lattice X.
#[inline]
fn lattice_row(lat: &[f32; LAT3], y: usize, z: usize, row: &mut [f32; LAT]) {
    let (ly, ty) = (y / COARSE, (y % COARSE) as f32 * (1.0 / COARSE as f32));
    let (lz, tz) = (z / COARSE, (z % COARSE) as f32 * (1.0 / COARSE as f32));
    let b = (ly * LAT + lz) * LAT;
    for (lx, out) in row.iter_mut().enumerate() {
        let v00 = lat[b + lx];
        let v01 = lat[b + LAT + lx];
        let v10 = lat[b + LAT * LAT + lx];
        let v11 = lat[b + LAT * LAT + LAT + lx];
        let a = v00 + (v01 - v00) * tz;
        let c = v10 + (v11 - v10) * tz;
        *out = a + (c - a) * ty;
    }
}

#[inline]
fn row_at(row: &[f32; LAT], x: usize) -> f32 {
    let (lx, tx) = (x / COARSE, (x % COARSE) as f32 * (1.0 / COARSE as f32));
    row[lx] + (row[lx + 1] - row[lx]) * tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn deterministic_and_cache_independent() {
        let a = Generator::new(WorldParams::default());
        let b = Generator::new(WorldParams::default());
        for &p in &[IVec3::new(0, 0, 0), IVec3::new(3, -1, -7), IVec3::new(-20, 2, 11)] {
            let x = a.generate(p);
            // Second generator has a cold column cache and a different call order.
            let _ = b.generate(p + IVec3::Y);
            assert!(x == b.generate(p) && x == a.generate(p));
        }
    }

    #[test]
    fn surface_exists_near_origin() {
        let g = Generator::new(WorldParams::default());
        let mut dense = 0;
        for cy in -4..6 {
            if !g.generate(IVec3::new(0, cy, 0)).is_uniform() {
                dense += 1;
            }
        }
        assert!(dense >= 1, "no surface found near the origin");
    }

    #[test]
    fn sky_is_air_and_depths_are_solid() {
        let g = Generator::new(WorldParams::default());
        assert!(g.generate(IVec3::new(0, 40, 0)) == Chunk::AIR);
        let mut solid = 0;
        for x in 0..40 {
            let c = g.generate(IVec3::new(x * 3, -40, 5));
            if c == Chunk::Uniform(Sample { fill: 255, mat: Material::STONE }) {
                solid += 1;
            }
        }
        assert!(solid >= 16, "deep chunks should mostly be uniform stone, got {solid}/40");
    }

    #[test]
    fn lowlands_are_sand_midlands_dirt_and_highlands_stone() {
        let params = WorldParams::default();
        let g = Generator::new(params);
        let cell = params.cell_size as f64;
        let mut seen = [0usize; 4];
        for k in 0..4000i64 {
            // A scatter of columns over a few kilometres.
            let (x, z) = ((k * 7919) % 6000 - 3000, (k * 104_729) % 6000 - 3000);
            let height = g.surface_height(x as f64 * cell, z as f64 * cell);
            // Well inside a band, clear of the wandering lines and of cliffs.
            let expect = if height < SAND_LINE - 4.0 {
                Material::SAND
            } else if height > SAND_LINE + 4.0 && height < STONE_LINE - 4.0 {
                Material::DIRT
            } else if height > STONE_LINE + 4.0 {
                Material::STONE
            } else {
                continue;
            };
            // The highest solid sample of the column, if the surface is where
            // the height says (not cut by a cave mouth or raised by a cliff).
            let y0 = (height as f64 / cell).floor() as i64;
            let sample = |y: i64| {
                let p = IVec3::new(x as i32, y as i32, z as i32);
                g.generate(crate::coords::chunk_of(p)).get(crate::coords::index(crate::coords::local_of(p)))
            };
            if !sample(y0 - 1).solid() || sample(y0 + 2).solid() {
                continue;
            }
            let top = (y0 - 1..=y0 + 1).rev().map(sample).find(|s| s.solid()).unwrap();
            assert_eq!(top.mat, expect, "at {x} {z}, height {height}");
            seen[expect as usize] += 1;
        }
        for m in [Material::SAND, Material::DIRT, Material::STONE] {
            assert!(seen[m as usize] > 150, "too few columns of material {m}: {seen:?}");
        }
    }
}
