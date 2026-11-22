//! World generation: a pure function from `(seed, node)` to samples.
//!
//! Part of the bit-exact contract. Float use is restricted to `+ - * /`,
//! `sqrt`, `floor` and comparisons; see `noise.rs`.
//!
//! Every level of detail samples the same terrain at its own spacing. Caves
//! are left out of the two coarsest levels, whose cells are wider than a tunnel.

use std::sync::{Arc, Mutex};

use crate::chunk::Sample;
use crate::coords::{APRON, CHUNK_SIZE, FILL_PER_CELL, NodeKey, PAD, WorldParams};
use crate::field::LocalField;
use crate::hash::{IntMap, IntSet, derive};
use crate::material::Material;
use crate::noise::{fbm2, noise2, noise3, ridged2, smoothstep};

/// Bump whenever generated output changes; saved edits are tied to it.
pub const GENERATOR_VERSION: u32 = 3;

/// Mountains: metres between ranges, between ridges, and peak height.
const MASK_WAVE: f64 = 800.0;
const MASK_FROM: f32 = -0.05;
const MASK_TO: f32 = 0.3;
const RIDGE_WAVE: f64 = 450.0;
const MOUNTAIN_HEIGHT: f32 = 170.0;

/// Samples per axis of a block, and of one of its horizontal slices.
const P: usize = PAD as usize;
const AREA: usize = P * P;
/// Lattice coordinates of 3D noise start this many samples before the block.
const LAT_PAD: usize = 1;
/// Samples the noise lattice has to span: the block and its padding.
const LAT_SPAN: usize = P + 2 * LAT_PAD - 1;

/// Ground below this height is sand, metres.
pub const SAND_LINE: f32 = 0.0;
/// Ground above this height is bare stone, metres. Dirt lies between the two.
pub const STONE_LINE: f32 = 40.0;
/// How far the two lines wander up and down, metres.
const LINE_WOBBLE: f32 = 3.0;
/// Height of the sea, metres: open ground below it is under water.
pub const SEA_LEVEL: f32 = SAND_LINE;

/// Largest cave radius in noise units; see `cave_radius`.
const CAVE_R_MAX: f32 = 0.09;
/// Metres of distance per unit of cave noise.
const CAVE_SCALE: f32 = 45.0;
/// Depth over which caves fade in below their opening depth, metres.
const CAVE_FADE: f32 = 6.0;
/// Coarsest level that still has caves.
const CAVE_LEVELS: u8 = 3;

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

/// Per-column terrain data shared by every node in a vertical stack.
struct Column {
    height: Vec<f32>,
    /// 1 / sqrt(1 + slope²): turns vertical depth into approximate distance.
    inv_grad: Vec<f32>,
    /// Amplitude of the 3D cliff/overhang term, metres.
    cliff: Vec<f32>,
    /// Thickness of the loose layer over the bedrock, metres, and what it is.
    cover: Vec<f32>,
    cover_mat: Vec<u8>,
    /// Depth at which caves start to fade in; negative lets them breach the surface.
    open: Vec<f32>,
    min_h: f32,
    max_h: f32,
    max_cliff: f32,
    min_inv_grad: f32,
    min_open: f32,
}

/// Base terrain heights of one column of nodes, shared with whoever needs
/// the ground without generating a block (water).
#[derive(Clone)]
pub struct Heights(Arc<Column>);

impl Heights {
    /// Height in metres at a node-local sample, from `-APRON` to `CHUNK_SIZE + APRON - 1`.
    #[inline]
    pub fn at(&self, x: i32, z: i32) -> f32 {
        self.0.height[(z + APRON) as usize * P + (x + APRON) as usize]
    }

    /// Lowest height in the column, apron included.
    pub fn min(&self) -> f32 {
        self.0.min_h
    }
}

/// What a generated block turned out to be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Generated {
    /// Every sample is this one; the field was filled with it.
    Uniform(Sample),
    Mixed,
}

/// 3D noise on a lattice coarser than the samples, interpolated between.
struct Lattice {
    /// Samples between lattice points.
    every: usize,
    /// Lattice points per axis.
    n: usize,
    values: Vec<f32>,
}

impl Lattice {
    fn empty() -> Self {
        Self { every: 1, n: 0, values: Vec::new() }
    }

    /// Interpolate along Y and Z at block sample `(y, z)`, leaving one value per lattice X.
    #[inline]
    fn row(&self, y: usize, z: usize, row: &mut [f32]) {
        let inv = 1.0 / self.every as f32;
        let (y, z) = (y + LAT_PAD, z + LAT_PAD);
        let (ly, ty) = (y / self.every, (y % self.every) as f32 * inv);
        let (lz, tz) = (z / self.every, (z % self.every) as f32 * inv);
        let (n, b) = (self.n, (ly * self.n + lz) * self.n);
        let v = &self.values;
        for (lx, out) in row[..n].iter_mut().enumerate() {
            let a = v[b + lx] + (v[b + n + lx] - v[b + lx]) * tz;
            let c = v[b + n * n + lx] + (v[b + n * n + n + lx] - v[b + n * n + lx]) * tz;
            *out = a + (c - a) * ty;
        }
    }

    #[inline]
    fn at(&self, row: &[f32], x: usize) -> f32 {
        let x = x + LAT_PAD;
        let (lx, tx) = (x / self.every, (x % self.every) as f32 * (1.0 / self.every as f32));
        row[lx] + (row[lx + 1] - row[lx]) * tx
    }
}

/// Whether ground below sea level is joined to a sea is decided for whole
/// cells this many base cells across: the columns of the finest chunks.
pub const SEA_GRID: i32 = CHUNK_SIZE;
/// Cells that low ground has to pass through, one to the next, before it is
/// a sea. Anything smaller is a dry hollow. At the default cell size this
/// is 1.6 hectares, an eighth of a kilometre across.
pub const SEA_CELLS: usize = 256;
/// Sea cells along the side of a cached tile.
const SEA_TILE: i32 = 32;
type SeaTile = Arc<[u32; SEA_TILE as usize]>;
/// Base cells between the points of a cell's side at which the ground is
/// looked at: low ground narrower than this may pass between them unseen.
const SEA_STEP: i32 = 4;

pub struct Generator {
    params: WorldParams,
    seeds: Seeds,
    columns: Mutex<IntMap<(u8, i32, i32), Arc<Column>>>,
    /// For each tile of the sea lattice, its points that are in a sea: bit `x` of row `z`.
    seas: Mutex<IntMap<(i32, i32), SeaTile>>,
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
            seas: Mutex::new(IntMap::default()),
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

    /// True if the ground is below sea level somewhere along a side of sea
    /// cell `(i, j)`: its side towards -Z if `along_x`, towards -X if not.
    fn low_side(&self, i: i32, j: i32, along_x: bool) -> bool {
        let cell = self.params.cell_size as f64;
        (0..=SEA_GRID / SEA_STEP).any(|k| {
            let (dx, dz) = if along_x { (k * SEA_STEP, 0) } else { (0, k * SEA_STEP) };
            let (x, z) = (i as i64 * SEA_GRID as i64 + dx as i64, j as i64 * SEA_GRID as i64 + dz as i64);
            self.surface_height(x as f64 * cell, z as f64 * cell) < SEA_LEVEL
        })
    }

    /// Which cells of a tile are in a sea: those from which ground below sea
    /// level runs on, from cell to neighbouring cell across a side that is
    /// low somewhere, through at least [`SEA_CELLS`] of them. That depends
    /// on nothing but the height function, so it is the same wherever it is
    /// asked from.
    fn sea_tile(&self, tx: i32, tz: i32) -> SeaTile {
        if let Some(t) = self.seas.lock().unwrap().get(&(tx, tz)) {
            return t.clone();
        }
        const N: i32 = SEA_TILE;
        let (i0, j0) = (tx * N, tz * N);
        let mut sea = [0u32; N as usize];
        let mut done = [0u32; N as usize];
        // Low or not, for every side looked at, in or out of the tile.
        let mut sides: IntMap<(i32, i32, bool), bool> = IntMap::default();
        let mut queue: Vec<(i32, i32)> = Vec::new();
        for z in 0..N {
            for x in 0..N {
                if done[z as usize] >> x & 1 == 1 {
                    continue;
                }
                // Follow the low ground from here until it is seen to be a
                // sea, or there is no more of it.
                queue.clear();
                queue.push((i0 + x, j0 + z));
                let mut seen: IntSet<(i32, i32)> = IntSet::default();
                seen.insert((i0 + x, j0 + z));
                let (mut next, mut is_sea) = (0, false);
                while next < queue.len() && !is_sea {
                    let (i, j) = queue[next];
                    next += 1;
                    // Each neighbour, and the side shared with it.
                    for (q, side) in [
                        ((i - 1, j), (i, j, false)),
                        ((i + 1, j), (i + 1, j, false)),
                        ((i, j - 1), (i, j, true)),
                        ((i, j + 1), (i, j + 1, true)),
                    ] {
                        if seen.contains(&q)
                            || !*sides.entry(side).or_insert_with(|| self.low_side(side.0, side.1, side.2))
                        {
                            continue;
                        }
                        let (lx, lz) = (q.0 - i0, q.1 - j0);
                        if (0..N).contains(&lx) && (0..N).contains(&lz) && done[lz as usize] >> lx & 1 == 1 {
                            // Joined to ground that is already decided, which
                            // can only be a sea: one left dry was followed to its end.
                            is_sea = true;
                            break;
                        }
                        seen.insert(q);
                        queue.push(q);
                    }
                    is_sea |= queue.len() >= SEA_CELLS;
                }
                for &(i, j) in &queue {
                    let (lx, lz) = (i - i0, j - j0);
                    if (0..N).contains(&lx) && (0..N).contains(&lz) {
                        done[lz as usize] |= 1 << lx;
                        sea[lz as usize] |= (is_sea as u32) << lx;
                    }
                }
            }
        }
        let tile = Arc::new(sea);
        let mut cache = self.seas.lock().unwrap();
        if cache.len() > 4000 {
            cache.clear();
        }
        cache.insert((tx, tz), tile.clone());
        tile
    }

    /// For the square of `n` by `n` sea cells from `(ci, cj)`, row by row,
    /// whether open ground below sea level in each is under the sea. Other
    /// low ground is a dry hollow until something joins it to a sea.
    pub fn sea_cells(&self, ci: i32, cj: i32, n: i32) -> Vec<bool> {
        let mut out = Vec::with_capacity((n * n) as usize);
        let mut tile: Option<((i32, i32), SeaTile)> = None;
        for j in cj..cj + n {
            for i in ci..ci + n {
                let t = (i.div_euclid(SEA_TILE), j.div_euclid(SEA_TILE));
                if tile.as_ref().is_none_or(|(at, _)| *at != t) {
                    tile = Some((t, self.sea_tile(t.0, t.1)));
                }
                let bits = &tile.as_ref().unwrap().1;
                out.push(bits[j.rem_euclid(SEA_TILE) as usize] >> i.rem_euclid(SEA_TILE) & 1 == 1);
            }
        }
        out
    }

    /// [`sea_cells`](Self::sea_cells) for one cell.
    pub fn sea_reaches(&self, ci: i32, cj: i32) -> bool {
        self.sea_cells(ci, cj, 1)[0]
    }

    /// Heights the terrain surface can reach, metres: lowest valley floor to highest peak.
    pub fn height_range(&self) -> (f32, f32) {
        (-60.0, 60.0 + MOUNTAIN_HEIGHT)
    }

    /// Base terrain heights under the nodes at `(cx, cz)` of `level`.
    pub fn heights(&self, level: u8, cx: i32, cz: i32) -> Heights {
        Heights(self.column(level, cx, cz))
    }

    fn column(&self, level: u8, cx: i32, cz: i32) -> Arc<Column> {
        if let Some(c) = self.columns.lock().unwrap().get(&(level, cx, cz)) {
            return c.clone();
        }
        let col = Arc::new(self.build_column(level, cx, cz));
        let mut cache = self.columns.lock().unwrap();
        if cache.len() > 3000 {
            cache.clear();
        }
        cache.insert((level, cx, cz), col.clone());
        col
    }

    fn build_column(&self, level: u8, cx: i32, cz: i32) -> Column {
        let s = &self.seeds;
        // Metres per sample of this level, and the lattice coordinate of the block's first sample.
        let spacing = self.params.cell_size as f64 * (1u64 << level) as f64;
        let (ox, oz) = (cx as i64 * CHUNK_SIZE as i64 - APRON as i64, cz as i64 * CHUNK_SIZE as i64 - APRON as i64);

        // Heights with a one-sample border for slope by central difference.
        const W: usize = P + 2;
        let mut h = vec![0.0f32; W * W];
        for z in 0..W {
            for x in 0..W {
                let wx = (ox + x as i64 - 1) as f64 * spacing;
                let wz = (oz + z as i64 - 1) as f64 * spacing;
                h[z * W + x] = self.surface_height(wx, wz);
            }
        }

        let mut col = Column {
            height: vec![0.0; AREA],
            inv_grad: vec![0.0; AREA],
            cliff: vec![0.0; AREA],
            cover: vec![0.0; AREA],
            cover_mat: vec![Material::STONE; AREA],
            open: vec![0.0; AREA],
            min_h: f32::MAX,
            max_h: f32::MIN,
            max_cliff: 0.0,
            min_inv_grad: 1.0,
            min_open: f32::MAX,
        };
        let inv_2step = (0.5 / spacing) as f32;
        for z in 0..P {
            for x in 0..P {
                let i = z * P + x;
                let c = (z + 1) * W + x + 1;
                let height = h[c];
                let gx = (h[c + 1] - h[c - 1]) * inv_2step;
                let gz = (h[c + W] - h[c - W]) * inv_2step;
                let slope2 = gx * gx + gz * gz;
                let inv_grad = 1.0 / (1.0 + slope2).sqrt();
                let wx = (ox + x as i64) as f64 * spacing;
                let wz = (oz + z as i64) as f64 * spacing;

                // One biome, banded by altitude: sand lowlands, dirt midlands,
                // stone highlands. The lines wander a little so they do not
                // trace exact contours.
                let band = height + LINE_WOBBLE * noise2(s.band, wx / 70.0, wz / 70.0);
                let (cover_mat, cover) = if band < SAND_LINE {
                    (Material::SAND, 5.0)
                } else if band < STONE_LINE {
                    (Material::DIRT, 3.0 + 1.5 * noise2(s.dirt, wx / 50.0, wz / 50.0))
                } else {
                    (Material::STONE, 0.0)
                };

                // Cliffs and overhangs belong to the highlands; fading them in
                // above the stone line keeps bedrock from showing through dirt.
                let cliff = 9.0
                    * smoothstep(0.15, 0.6, noise2(s.cliff, wx / 260.0, wz / 260.0))
                    * smoothstep(STONE_LINE + 4.0, STONE_LINE + 24.0, band);

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

    /// 3D noise on the lattice of a block. Lattice points sit at multiples of
    /// `every` samples of the level, so neighbouring blocks agree on them.
    fn lattice(&self, seed: u64, level: u8, origin: [i64; 3], wave: [f64; 3]) -> Lattice {
        // About a metre apart at the fine levels, never further than the samples.
        let every = (4usize >> level.saturating_sub(2)).max(1);
        let n = LAT_SPAN.div_ceil(every) + 1;
        let spacing = self.params.cell_size as f64 * (1u64 << level) as f64;
        let mut values = vec![0.0f32; n * n * n];
        let at = |a: usize, l: usize| (origin[a] - LAT_PAD as i64 + (l * every) as i64) as f64 * spacing / wave[a];
        for ly in 0..n {
            let y = at(1, ly);
            for lz in 0..n {
                let z = at(2, lz);
                for lx in 0..n {
                    values[(ly * n + lz) * n + lx] = noise3(seed, at(0, lx), y, z);
                }
            }
        }
        Lattice { every, n, values }
    }

    /// Fill `field`, the padded block of `key`, with generated terrain.
    pub fn fill_block(&self, key: NodeKey, field: &mut LocalField) -> Generated {
        debug_assert!(field.dims == glam::IVec3::splat(PAD) && field.level == key.level);
        let level = key.level;
        let spacing = self.params.spacing(level);
        let col = self.column(level, key.pos.x, key.pos.z);
        // Lattice coordinate of the block's first sample.
        let origin: [i64; 3] = std::array::from_fn(|a| key.pos[a] as i64 * CHUNK_SIZE as i64 - APRON as i64);
        debug_assert!(origin.iter().all(|o| (o - LAT_PAD as i64).rem_euclid(4) == 0));
        let y_min = (origin[1] as f64 * spacing as f64) as f32;
        let y_max = ((origin[1] + P as i64 - 1) as f64 * spacing as f64) as f32;
        let band = 2.1 * spacing;

        // Entirely above anything the surface terms can reach: air.
        if (y_min - col.max_h) * col.min_inv_grad > 1.1 * col.max_cliff + band {
            field.set_all(Sample::AIR);
            return Generated::Uniform(Sample::AIR);
        }

        // So deep that the surface terms saturate; only caves can matter.
        let top_depth = col.min_h - y_max;
        let deep = top_depth * col.min_inv_grad > 1.1 * col.max_cliff + band && top_depth > 14.0;
        let need_over = !deep && col.max_cliff > 0.0;
        let need_cave = level <= CAVE_LEVELS && col.max_h - y_min > col.min_open;

        let over =
            if need_over { self.lattice(self.seeds.over, level, origin, [34.0, 24.0, 34.0]) } else { Lattice::empty() };
        let (c1, c2, c3) = if need_cave {
            (
                self.lattice(self.seeds.cave1, level, origin, [90.0, 55.0, 90.0]),
                self.lattice(self.seeds.cave2, level, origin, [90.0, 55.0, 90.0]),
                self.lattice(self.seeds.cave3, level, origin, [260.0, 160.0, 260.0]),
            )
        } else {
            (Lattice::empty(), Lattice::empty(), Lattice::empty())
        };

        let rock = Sample { fill: 255, mat: Material::STONE };
        if deep && !(need_cave && may_have_cave(&c1, &c2)) {
            field.set_all(rock);
            return Generated::Uniform(rock);
        }

        field.set_all(Sample::AIR);
        let fill_per_metre = FILL_PER_CELL as f32 / spacing;
        let skin = 1.5 * spacing;
        let rows = LAT_SPAN + 2;
        let (mut r_over, mut r1, mut r2, mut r3) =
            (vec![0.0f32; rows], vec![0.0f32; rows], vec![0.0f32; rows], vec![0.0f32; rows]);
        let (mut any_air, mut all_rock) = (false, true);
        for y in 0..P {
            let ym = (origin[1] + y as i64) as f64 * spacing as f64;
            for z in 0..P {
                if need_over {
                    over.row(y, z, &mut r_over);
                }
                if need_cave {
                    c1.row(y, z, &mut r1);
                    c2.row(y, z, &mut r2);
                    c3.row(y, z, &mut r3);
                }
                let base = (y * P + z) * P;
                for x in 0..P {
                    let ci = z * P + x;
                    let depth = (col.height[ci] as f64 - ym) as f32;
                    let mut d = depth * col.inv_grad[ci];
                    if need_over {
                        d += col.cliff[ci] * over.at(&r_over, x);
                    }
                    if need_cave {
                        let fade = ((depth - col.open[ci]) / CAVE_FADE).clamp(0.0, 1.0);
                        if fade > 0.0 {
                            let (a, b) = (c1.at(&r1, x), c2.at(&r2, x));
                            let r = cave_radius(c3.at(&r3, x)) * fade;
                            d = d.min(((a * a + b * b).sqrt() - r) * CAVE_SCALE);
                        }
                    }
                    let fill = ((127.5 + d * fill_per_metre).clamp(0.0, 255.0) + 0.5) as u8;
                    if fill == 0 {
                        any_air = true;
                        continue;
                    }
                    // The sample nearest the surface is what gets drawn. Where
                    // the cover is thinner than the cells of this level, that
                    // sample still has to be cover, or bedrock shows through.
                    let covered = depth < col.cover[ci] || depth * col.inv_grad[ci] < skin;
                    let mat = if covered { col.cover_mat[ci] } else { Material::STONE };
                    all_rock &= fill == 255 && mat == Material::STONE;
                    field.fill[base + x] = fill;
                    field.mat[base + x] = mat;
                }
            }
        }
        if all_rock && !any_air {
            return Generated::Uniform(rock);
        }
        if field.fill.iter().all(|&f| f == 0) { Generated::Uniform(Sample::AIR) } else { Generated::Mixed }
    }
}

#[inline]
fn cave_radius(c3: f32) -> f32 {
    (0.05 + 0.035 * c3).clamp(0.0, CAVE_R_MAX)
}

/// Conservative test: can any interpolated point of these lattices be inside a cave?
fn may_have_cave(c1: &Lattice, c2: &Lattice) -> bool {
    let limit = CAVE_R_MAX * CAVE_R_MAX;
    let n = c1.n;
    let min_abs = |lat: &Lattice, base: usize| {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for dy in 0..2 {
            for dz in 0..2 {
                for dx in 0..2 {
                    let v = lat.values[base + (dy * n + dz) * n + dx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        if lo <= 0.0 && hi >= 0.0 { 0.0 } else { lo.abs().min(hi.abs()) }
    };
    for ly in 0..n - 1 {
        for lz in 0..n - 1 {
            for lx in 0..n - 1 {
                let base = (ly * n + lz) * n + lx;
                let (a, b) = (min_abs(c1, base), min_abs(c2, base));
                if a * a + b * b <= limit {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{MAX_LEVEL, SOLID};
    use glam::IVec3;

    fn block(g: &Generator, level: u8, pos: IVec3) -> (LocalField, Generated) {
        let key = NodeKey::new(level, pos);
        let mut f = LocalField::block(key);
        let kind = g.fill_block(key, &mut f);
        (f, kind)
    }

    #[test]
    fn deterministic_and_cache_independent() {
        let a = Generator::new(WorldParams::default());
        let b = Generator::new(WorldParams::default());
        for (level, p) in [(0, IVec3::new(0, 0, 0)), (1, IVec3::new(3, -1, -7)), (4, IVec3::new(-20, 0, 11))] {
            let x = block(&a, level, p).0;
            // Second generator has a cold column cache and a different call order.
            let _ = block(&b, level, p + IVec3::Y);
            let (y, z) = (block(&b, level, p).0, block(&a, level, p).0);
            assert!(x.fill == y.fill && x.mat == y.mat && x.fill == z.fill && x.mat == z.mat);
        }
    }

    #[test]
    fn neighbouring_blocks_agree_where_they_overlap() {
        let g = Generator::new(WorldParams::default());
        for level in 0..=MAX_LEVEL {
            // A column of blocks that crosses the surface near the origin.
            let ground = g.surface_height(0.0, 0.0) / g.params().spacing(level);
            let cy = (ground / CHUNK_SIZE as f32).floor() as i32;
            let (a, _) = block(&g, level, IVec3::new(0, cy, 0));
            for axis in 0..3 {
                let mut offset = IVec3::ZERO;
                offset[axis] = 1;
                let (b, _) = block(&g, level, IVec3::new(0, cy, 0) + offset);
                // `b`'s low apron is `a`'s high interior and the other way round.
                for u in 0..PAD {
                    for v in 0..PAD {
                        for w in 0..2 * APRON {
                            let mut pa = IVec3::new(u, v, 0);
                            pa[axis] = CHUNK_SIZE + w;
                            pa[(axis + 1) % 3] = u;
                            pa[(axis + 2) % 3] = v;
                            let mut pb = pa;
                            pb[axis] = w;
                            assert_eq!(a.get(pa.x, pa.y, pa.z), b.get(pb.x, pb.y, pb.z), "level {level} axis {axis}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn every_level_has_the_surface_at_the_same_height() {
        let g = Generator::new(WorldParams::default());
        let (x, z) = (40.0f64, -24.0f64);
        let height = g.surface_height(x, z);
        for level in 0..=MAX_LEVEL {
            let spacing = g.params().spacing(level) as f64;
            // The sample column at (x, z), which is on every level's lattice.
            let (sx, sz) = ((x / spacing) as i32, (z / spacing).floor() as i32);
            let sy = (height as f64 / spacing).floor() as i32;
            let sample = |y: i32| {
                let p = IVec3::new(sx, y, sz);
                let (f, _) = block(&g, level, crate::coords::chunk_of(p));
                let l = crate::coords::local_of(p) + APRON;
                f.get(l.x, l.y, l.z)
            };
            assert!(sample(sy - 2).fill >= SOLID, "level {level}: ground missing below the surface");
            assert!(sample(sy + 3).fill < SOLID, "level {level}: ground above the surface");
        }
    }

    #[test]
    fn sky_is_air_and_depths_are_solid() {
        let g = Generator::new(WorldParams::default());
        assert_eq!(block(&g, 0, IVec3::new(0, 80, 0)).1, Generated::Uniform(Sample::AIR));
        assert_eq!(block(&g, MAX_LEVEL, IVec3::new(0, 3, 0)).1, Generated::Uniform(Sample::AIR));
        let rock = Generated::Uniform(Sample { fill: 255, mat: Material::STONE });
        let solid = (0..40).filter(|&x| block(&g, 1, IVec3::new(x * 3, -40, 5)).1 == rock).count();
        assert!(solid >= 16, "deep chunks should mostly be uniform stone, got {solid}/40");
        // Uniform results fill the field too.
        let (f, kind) = block(&g, MAX_LEVEL, IVec3::new(0, -3, 0));
        assert_eq!(kind, rock);
        assert!(f.fill.iter().all(|&v| v == 255) && f.mat.iter().all(|&m| m == Material::STONE));
    }

    #[test]
    fn lowlands_are_sand_midlands_dirt_and_highlands_stone() {
        let params = WorldParams { cell_size: 0.5, ..WorldParams::default() };
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
            let p = IVec3::new(x as i32, y0 as i32, z as i32);
            let chunk = crate::coords::chunk_of(p);
            let (f, _) = block(&g, 0, chunk);
            let l = p - crate::coords::chunk_origin(chunk) + APRON;
            let sample = |dy: i32| f.get(l.x, l.y + dy, l.z);
            if !sample(-1).solid() || sample(2).solid() {
                continue;
            }
            let top = (-1..=1).rev().map(sample).find(|s| s.solid()).unwrap();
            assert_eq!(top.mat, expect, "at {x} {z}, height {height}");
            seen[expect as usize] += 1;
        }
        for m in [Material::SAND, Material::DIRT, Material::STONE] {
            assert!(seen[m as usize] > 150, "too few columns of material {m}: {seen:?}");
        }
    }
}
