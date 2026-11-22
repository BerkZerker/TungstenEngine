//! Edit operations: the only way the world changes after generation.
//!
//! Part of the bit-exact contract. An [`Op`] carries fixed-point positions
//! and everything here is integer arithmetic, so the same ops in the same
//! order give identical samples on every machine.
//!
//! An op is a shape, not a list of samples. Applying it takes each sample to
//! the minimum (or maximum) of what it held and a function of its position:
//! `SOLID` plus the signed distance to the shape's surface, rounded down to
//! the fill scale of the sample's level. Whether that is solid depends only
//! on the sign of the distance, so a point that several levels sample is
//! solid in all of them or in none, and of the same material.

use glam::IVec3;

use crate::chunk::Volume;
use crate::coords::{ChunkPos, FILL_PER_CELL, MAX_LEVEL, SOLID, WorldParams};
use crate::field::LocalField;
use crate::hash::{derive, hash3};
use crate::material::{self, CarveMode, Material};

/// Fixed-point units per base cell for edit positions and radii.
pub const FP: i64 = 256;
/// Fixed-point distance over which fill changes by one unit, at level 0.
const FP_PER_FILL: i64 = FP / FILL_PER_CELL as i64;

/// A vertical capsule that a `Place` leaves untouched (the player's body).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exclude {
    pub x: i64,
    pub z: i64,
    pub y0: i64,
    pub y1: i64,
    pub radius: i64,
}

/// One edit. Positions and radii are fixed-point base cells.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Cut a ball out of smooth materials. Stone gives way in whole shards:
    /// those whose centre is inside the ball and, if `nearest`, the one the
    /// ball's centre is in.
    Break { center: [i64; 3], radius: i64, nearest: bool },
    /// Add a ball of `material` (stone: the shards centred inside it). Never
    /// repaints what is already solid and leaves `exclude` empty.
    Place { center: [i64; 3], radius: i64, material: u8, exclude: Option<Exclude> },
    /// Empty these lattice points: a piece that broke off. `cells` are
    /// offsets from `origin`, in base cells.
    Clear { origin: IVec3, cells: Box<[[u8; 3]]> },
}

/// World constants the edit rules depend on.
#[derive(Clone, Copy, Debug)]
pub struct EditRules {
    seed: u64,
    /// Spacing of the shard pattern stone breaks along, fixed-point base cells.
    pub shard_spacing: i64,
    /// Pieces of fewer base samples than this are broken off; see `WorldParams::min_fragment`.
    pub fragment_samples: u32,
}

impl EditRules {
    pub fn new(params: &WorldParams) -> Self {
        // About 1.5 m, and never finer than two cells: smaller shards cannot
        // be resolved by the lattice.
        let spacing = ((1.5 / params.cell_size as f64) * FP as f64).round() as i64;
        let side = params.min_fragment.max(0.0) as f64 / params.cell_size as f64;
        Self {
            seed: derive(params.seed, 0x53_4841_5244),
            shard_spacing: spacing.max(2 * FP),
            fragment_samples: (side * side * side).round().min(4096.0) as u32,
        }
    }

    /// No point is further than this from the centre of the shard it is in.
    fn shard_reach(&self) -> i64 {
        self.shard_spacing * 8 / 5
    }

    /// Grid cell of the shard pattern holding the site nearest to `w`.
    fn nearest_site(&self, w: [i64; 3]) -> [i64; 3] {
        let g: [i64; 3] = w.map(|v| v.div_euclid(self.shard_spacing));
        let (mut best, mut best_d2) = (g, i64::MAX);
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let c = [g[0] + dx, g[1] + dy, g[2] + dz];
                    let s = self.site(c);
                    let d2 = len2([w[0] - s[0], w[1] - s[1], w[2] - s[2]]);
                    if d2 < best_d2 {
                        (best, best_d2) = (c, d2);
                    }
                }
            }
        }
        best
    }

    /// World position of the shard centre in grid cell `g`.
    #[inline]
    fn site(&self, g: [i64; 3]) -> [i64; 3] {
        let sp = self.shard_spacing;
        let (inset, span) = (sp / 10, (sp * 8 / 10) as u64);
        let h = hash3(self.seed, g[0], g[1], g[2]);
        let jitter = [h & 0xffff, (h >> 16) & 0xffff, (h >> 32) & 0xffff];
        std::array::from_fn(|a| g[a] * sp + inset + (jitter[a] % span) as i64)
    }
}

impl Op {
    /// Lattice bounds at `level` (inclusive) of the samples this op can change.
    pub fn bounds(&self, rules: &EditRules, level: u8) -> (IVec3, IVec3) {
        let step = 1i64 << level;
        let ball = |center: &[i64; 3], radius: i64, shards: bool| {
            // Fill saturates two cells of the level away from a surface.
            let soft = 2 * FP * step;
            let reach =
                if shards { radius.max(rules.shard_reach()) + rules.shard_reach() + soft } else { radius + soft };
            let unit = FP * step;
            (
                IVec3::from_array(center.map(|c| (c - reach).div_euclid(unit) as i32)),
                IVec3::from_array(center.map(|c| (c + reach).div_euclid(unit) as i32 + 1)),
            )
        };
        match self {
            // Stone may be anywhere under a break brush.
            Op::Break { center, radius, .. } => ball(center, *radius, true),
            Op::Place { center, radius, material, .. } => {
                ball(center, *radius, material::def(*material).carve == CarveMode::Shards)
            }
            Op::Clear { origin, cells } => {
                let reach =
                    cells.iter().fold(IVec3::ZERO, |m, c| m.max(IVec3::new(c[0] as i32, c[1] as i32, c[2] as i32)));
                let s = IVec3::splat(step as i32);
                (origin.div_euclid(s), (*origin + reach).div_euclid(s))
            }
        }
    }

    /// Base-cell bounds (inclusive) of everything this op can change at `levels`.
    pub fn bounds_over(&self, rules: &EditRules, levels: std::ops::Range<u8>) -> (IVec3, IVec3) {
        levels.fold((IVec3::MAX, IVec3::MIN), |(lo, hi), level| {
            let (a, b) = self.bounds(rules, level);
            (lo.min(a << level as i32), hi.max(b << level as i32))
        })
    }

    /// Base-cell bounds (inclusive) of everything this op can change at any level.
    pub fn bounds_all(&self, rules: &EditRules) -> (IVec3, IVec3) {
        self.bounds_over(rules, 0..MAX_LEVEL + 1)
    }
}

#[inline]
fn clamp_fill(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn len2(p: [i64; 3]) -> i64 {
    p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
}

/// Where the samples of a field are.
#[derive(Clone, Copy)]
struct Grid {
    origin: IVec3,
    dims: IVec3,
    /// Fixed-point distance between samples.
    unit: i64,
}

impl Grid {
    fn of(field: &LocalField) -> Self {
        Self { origin: field.origin, dims: field.dims, unit: FP * field.step() as i64 }
    }

    /// World fixed-point position of a local sample.
    fn world(&self, local: IVec3) -> [i64; 3] {
        (self.origin + local).to_array().map(|v| v as i64 * self.unit)
    }

    /// Visit the samples of local box `[lo, hi)` as (index, world position,
    /// position relative to `center`).
    fn for_samples(&self, lo: IVec3, hi: IVec3, center: [i64; 3], mut f: impl FnMut(usize, [i64; 3], [i64; 3])) {
        let o = self.origin;
        for y in lo.y..hi.y {
            let wy = (o.y + y) as i64 * self.unit;
            for z in lo.z..hi.z {
                let wz = (o.z + z) as i64 * self.unit;
                let row = ((y * self.dims.z + z) * self.dims.x) as usize;
                for x in lo.x..hi.x {
                    let w = [(o.x + x) as i64 * self.unit, wy, wz];
                    f(row + x as usize, w, [w[0] - center[0], w[1] - center[1], w[2] - center[2]]);
                }
            }
        }
    }
}

/// The shard centres around a box of samples, relative to a brush centre.
struct Sites {
    spacing: i64,
    /// Grid cell of index (0, 0, 0).
    base: [i64; 3],
    dims: [i64; 3],
    /// Position of each site relative to the brush centre.
    pos: Vec<[i64; 3]>,
}

impl Sites {
    /// Sites for every grid cell a sample between world positions `min` and
    /// `max` can ask about.
    fn new(rules: &EditRules, center: [i64; 3], min: [i64; 3], max: [i64; 3]) -> Self {
        let sp = rules.shard_spacing;
        let base: [i64; 3] = std::array::from_fn(|a| min[a].div_euclid(sp) - 1);
        let dims: [i64; 3] = std::array::from_fn(|a| max[a].div_euclid(sp) + 1 - base[a] + 1);
        let mut pos = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
        for gy in 0..dims[1] {
            for gz in 0..dims[2] {
                for gx in 0..dims[0] {
                    let s = rules.site([base[0] + gx, base[1] + gy, base[2] + gz]);
                    pos.push([s[0] - center[0], s[1] - center[1], s[2] - center[2]]);
                }
            }
        }
        Self { spacing: sp, base, dims, pos }
    }

    /// Which sites a ball takes: those inside it and, if `nearest`, the one
    /// whose shard holds its centre.
    fn taken(&self, rules: &EditRules, center: [i64; 3], radius: i64, nearest: bool) -> Vec<bool> {
        let r2 = radius * radius;
        let mut taken: Vec<bool> = self.pos.iter().map(|&s| len2(s) <= r2).collect();
        if nearest {
            let g = rules.nearest_site(center);
            let l: [i64; 3] = std::array::from_fn(|a| g[a] - self.base[a]);
            if (0..3).all(|a| l[a] >= 0 && l[a] < self.dims[a]) {
                taken[self.index(l)] = true;
            }
        }
        taken
    }

    #[inline]
    fn index(&self, g: [i64; 3]) -> usize {
        ((g[1] * self.dims[2] + g[2]) * self.dims[0] + g[0]) as usize
    }

    /// The 27 candidate sites around world fixed-point position `w`, as
    /// (site index, squared distance from `p`), where `p` is `w` relative to
    /// the brush centre.
    #[inline]
    fn candidates(&self, w: [i64; 3], p: [i64; 3]) -> [(u32, i64); 27] {
        let g: [i64; 3] = std::array::from_fn(|a| w[a].div_euclid(self.spacing) - self.base[a]);
        let mut out = [(0u32, 0i64); 27];
        let mut n = 0;
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    let i = self.index([g[0] + dx, g[1] + dy, g[2] + dz]);
                    let s = self.pos[i];
                    let d = [p[0] - s[0], p[1] - s[1], p[2] - s[2]];
                    out[n] = (i as u32, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    n += 1;
                }
            }
        }
        out
    }

    /// Index into `candidates` of the nearest site; ties go to the first.
    #[inline]
    fn nearest(c: &[(u32, i64); 27]) -> usize {
        let mut best = 0;
        for k in 1..27 {
            if c[k].1 < c[best].1 {
                best = k;
            }
        }
        best
    }

    /// Distance from a point to the plane halfway between its own site `a`
    /// and another site `b`, given the point's squared distances to each.
    #[inline]
    fn bisector(&self, a: u32, da2: i64, b: u32, db2: i64) -> i64 {
        let (sa, sb) = (self.pos[a as usize], self.pos[b as usize]);
        let d = [sa[0] - sb[0], sa[1] - sb[1], sa[2] - sb[2]];
        let len = ((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) as u64).isqrt().max(1) as i64;
        (db2 - da2) / (2 * len)
    }

    /// Signed distance from a point to the surface of the taken shards:
    /// positive inside them. `None` if no face between a taken and an untaken
    /// shard is near; then the bool says which side the point is on.
    #[inline]
    fn depth(&self, taken: &[bool], w: [i64; 3], p: [i64; 3]) -> Result<i64, bool> {
        let c = self.candidates(w, p);
        let own = c[Self::nearest(&c)];
        let inside = taken[own.0 as usize];
        let mut face = i64::MAX;
        for &(j, dj2) in &c {
            if j != own.0 && taken[j as usize] != inside {
                face = face.min(self.bisector(own.0, own.1, j, dj2));
            }
        }
        if face == i64::MAX { Err(inside) } else { Ok(if inside { face } else { -face }) }
    }

    /// Bounds, relative to the brush centre, of the taken sites.
    fn extent(&self, taken: &[bool]) -> Option<([i64; 3], [i64; 3])> {
        let mut bounds: Option<([i64; 3], [i64; 3])> = None;
        for (s, _) in self.pos.iter().zip(taken).filter(|(_, t)| **t) {
            let (lo, hi) = bounds.get_or_insert((*s, *s));
            for a in 0..3 {
                lo[a] = lo[a].min(s[a]);
                hi[a] = hi[a].max(s[a]);
            }
        }
        bounds
    }
}

/// Apply one op to the samples of `field`, optionally only those inside
/// `clip` (lattice bounds of the field's level, inclusive).
pub fn apply(op: &Op, rules: &EditRules, field: &mut LocalField, clip: Option<(IVec3, IVec3)>) {
    let (mut min, mut max) = op.bounds(rules, field.level);
    if let Some((clip_min, clip_max)) = clip {
        (min, max) = (min.max(clip_min), max.min(clip_max));
    }
    let Some((lo, hi)) = field.clip(min, max) else { return };
    let grid = Grid::of(field);
    match op {
        Op::Break { center, radius, nearest } => {
            break_brush(grid, &mut field.fill, &mut field.mat, rules, *center, *radius, *nearest, lo, hi)
        }
        Op::Place { center, radius, material, exclude } => {
            place_brush(grid, &mut field.fill, &mut field.mat, rules, *center, *radius, *material, exclude, lo, hi)
        }
        Op::Clear { origin, cells } => {
            let step = field.step();
            for c in cells.iter() {
                let base = *origin + IVec3::new(c[0] as i32, c[1] as i32, c[2] as i32);
                if base.rem_euclid(IVec3::splat(step)) != IVec3::ZERO {
                    continue;
                }
                let l = base / step - field.origin;
                if l.cmpge(lo).all() && l.cmplt(hi).all() {
                    let i = field.index(l.x, l.y, l.z);
                    field.fill[i] = 0;
                    field.mat[i] = 0;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn break_brush(
    grid: Grid,
    fill: &mut [u8],
    mat: &mut [u8],
    rules: &EditRules,
    center: [i64; 3],
    radius: i64,
    nearest: bool,
    lo: IVec3,
    hi: IVec3,
) {
    let per_fill = FP_PER_FILL * grid.unit / FP;
    let soft = radius + 2 * grid.unit;
    let soft2 = soft * soft;
    let mut any_stone = false;

    // Smooth materials: cut to the ball.
    grid.for_samples(lo, hi, center, |i, _, p| {
        if fill[i] == 0 {
            return;
        }
        if material::def(mat[i]).carve == CarveMode::Shards {
            any_stone = true;
            return;
        }
        let d2 = len2(p);
        if d2 >= soft2 {
            return;
        }
        let d = (d2 as u64).isqrt() as i64;
        fill[i] = fill[i].min(clamp_fill(SOLID as i64 + (d - radius).div_euclid(per_fill)));
        if fill[i] == 0 {
            mat[i] = 0;
        }
    });
    if !any_stone {
        return;
    }

    // Stone: whole shards break off, leaving flat faces where they met their
    // neighbours.
    let sites = Sites::new(rules, center, grid.world(lo), grid.world(hi - 1));
    let taken = sites.taken(rules, center, radius, nearest);
    let Some((near, far)) = sites.extent(&taken) else { return };
    let margin = rules.shard_reach() + 2 * grid.unit;
    grid.for_samples(lo, hi, center, |i, w, p| {
        if fill[i] == 0 || material::def(mat[i]).carve != CarveMode::Shards {
            return;
        }
        if (0..3).any(|a| p[a] < near[a] - margin || p[a] > far[a] + margin) {
            return;
        }
        let target = match sites.depth(&taken, w, p) {
            Ok(depth) => clamp_fill(SOLID as i64 + (-depth).div_euclid(per_fill)),
            Err(true) => 0,
            Err(false) => return,
        };
        fill[i] = fill[i].min(target);
        if fill[i] == 0 {
            mat[i] = 0;
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn place_brush(
    grid: Grid,
    fill: &mut [u8],
    mat: &mut [u8],
    rules: &EditRules,
    center: [i64; 3],
    radius: i64,
    material: u8,
    exclude: &Option<Exclude>,
    lo: IVec3,
    hi: IVec3,
) {
    if material == Material::AIR || material as usize >= material::MATERIAL_COUNT {
        return;
    }
    let per_fill = FP_PER_FILL * grid.unit / FP;
    let excluded = |w: [i64; 3]| {
        exclude.is_some_and(|e| {
            let dy = w[1] - w[1].clamp(e.y0, e.y1);
            len2([w[0] - e.x, dy, w[2] - e.z]) < e.radius * e.radius
        })
    };
    let mut stamp = |i: usize, target: u8| {
        if target > fill[i] {
            if fill[i] < SOLID {
                mat[i] = material;
            }
            fill[i] = target;
        }
    };

    if material::def(material).carve != CarveMode::Shards {
        // A ball: fill rises to the ball's profile, never falls.
        let soft = radius + 2 * grid.unit;
        let soft2 = soft * soft;
        grid.for_samples(lo, hi, center, |i, w, p| {
            let d2 = len2(p);
            if d2 >= soft2 {
                return;
            }
            let d = (d2 as u64).isqrt() as i64;
            let target = clamp_fill(SOLID as i64 + (radius - d).div_euclid(per_fill));
            if target > 0 && !excluded(w) {
                stamp(i, target);
            }
        });
        return;
    }

    // Stone: the union of the shards centred inside the brush, so placed
    // stone is angular and lines up with the pattern it later breaks along.
    // A brush smaller than the pattern still places the shard it is in.
    let sites = Sites::new(rules, center, grid.world(lo), grid.world(hi - 1));
    let taken = sites.taken(rules, center, radius, true);
    let Some((near, far)) = sites.extent(&taken) else { return };
    let margin = rules.shard_reach() + 2 * grid.unit;
    grid.for_samples(lo, hi, center, |i, w, p| {
        if (0..3).any(|a| p[a] < near[a] - margin || p[a] > far[a] + margin) {
            return;
        }
        let target = match sites.depth(&taken, w, p) {
            Ok(depth) => clamp_fill(SOLID as i64 + depth.div_euclid(per_fill)),
            Err(true) => 255,
            Err(false) => return,
        };
        if target > 0 && !excluded(w) {
            stamp(i, target);
        }
    });
}

/// Break off what an edit left too small to keep: a connected run of solid
/// samples of one material, open to the air, of fewer than `limit` samples.
/// `before` is the fill as it was before the edit; only pieces beside a sample
/// the edit changed are considered, so nothing far from the brush is touched.
/// Returns the indices of the samples it emptied.
fn remove_fragments(field: &mut LocalField, before: &[u8], limit: u32) -> Vec<usize> {
    let mut removed = Vec::new();
    // A piece cannot be smaller than one sample.
    if limit <= 1 {
        return removed;
    }
    let [sx, sy, sz] = field.strides();
    let around = |i: usize| [i - sx, i + sx, i - sz, i + sz, i - sy, i + sy];
    // Samples on the faces of the field have neighbours we cannot see.
    let mut edge = vec![false; field.fill.len()];
    let mut seeds = Vec::new();
    let mut i = 0;
    for y in 0..field.dims.y {
        for z in 0..field.dims.z {
            for x in 0..field.dims.x {
                edge[i] = x == 0
                    || y == 0
                    || z == 0
                    || x == field.dims.x - 1
                    || y == field.dims.y - 1
                    || z == field.dims.z - 1;
                if !edge[i] && field.fill[i] != before[i] {
                    seeds.push(i);
                    seeds.extend(around(i));
                }
                i += 1;
            }
        }
    }

    // 0: unseen, 1: part of something large enough to stay, else the id of the search that reached it.
    let mut seen = vec![0u32; field.fill.len()];
    let mut search = 1u32;
    let mut piece = Vec::new();
    while let Some(seed) = seeds.pop() {
        if field.fill[seed] < SOLID || seen[seed] != 0 {
            continue;
        }
        search += 1;
        let kind = field.mat[seed];
        piece.clear();
        piece.push(seed);
        seen[seed] = search;
        let (mut keep, mut exposed) = (false, false);
        let mut next = 0;
        while next < piece.len() && !keep {
            let c = piece[next];
            next += 1;
            if edge[c] {
                // It runs out of view: assume it is large.
                keep = true;
                break;
            }
            for n in around(c) {
                if field.fill[n] < SOLID {
                    exposed = true;
                } else if field.mat[n] == kind {
                    if seen[n] == 1 || piece.len() as u32 + 1 >= limit {
                        keep = true;
                        break;
                    }
                    if seen[n] != search {
                        seen[n] = search;
                        piece.push(n);
                    }
                }
            }
        }
        if keep || !exposed {
            // Buried pockets stay too: removing them would leave hidden voids.
            for &c in &piece {
                seen[c] = if keep { 1 } else { 0 };
            }
            continue;
        }
        for &c in &piece {
            field.fill[c] = 0;
            field.mat[c] = 0;
            seen[c] = 0;
            removed.push(c);
            // Whatever this was holding on to may now be loose as well.
            seeds.extend(around(c));
        }
    }
    removed
}

/// What an edit did to the loaded samples.
#[derive(Debug, Default)]
pub struct EditOutcome {
    /// Chunks whose samples changed.
    pub chunks: Vec<ChunkPos>,
    /// Bounds (inclusive) of the loaded samples that changed.
    pub changed: Option<(IVec3, IVec3)>,
    /// Sample bounds of the region the op could have changed (inclusive).
    pub min: IVec3,
    pub max: IVec3,
    /// Pieces that broke off as a result: an op to log after this one.
    pub fragments: Option<Op>,
    /// False if part of that region was not loaded, so the op may have done
    /// more than `chunks` shows.
    pub covered: bool,
}

/// Apply an op to the finest level as held in `volume`. A break also removes
/// the fragments it leaves, and reports them as a further op.
pub fn edit_volume(volume: &mut Volume, rules: &EditRules, op: &Op) -> EditOutcome {
    let (min, max) = op.bounds(rules, 0);
    let dims = max - min + 1;
    let mut field = volume.copy_region(min, dims);
    let covered = volume.covers(min, dims);
    let mut out = EditOutcome { chunks: Vec::new(), changed: None, min, max, fragments: None, covered };

    let before = matches!(op, Op::Break { .. }).then(|| field.fill.clone());
    apply(op, rules, &mut field, None);
    // Unloaded space reads as air, which would make whatever borders it look loose.
    if let (Some(before), true) = (before, covered) {
        let removed = remove_fragments(&mut field, &before, rules.fragment_samples);
        if !removed.is_empty() {
            let local = |i: usize| {
                let i = i as i32;
                IVec3::new(i % dims.x, i / (dims.x * dims.z), i / dims.x % dims.z)
            };
            let lo = removed.iter().fold(IVec3::MAX, |m, &i| m.min(local(i)));
            let cells = removed.iter().map(|&i| (local(i) - lo).to_array().map(|v| v as u8)).collect();
            out.fragments = Some(Op::Clear { origin: min + lo, cells });
        }
    }
    out.changed = volume.write_region(&field, &mut out.chunks);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Sample};

    const STONE: Sample = Sample { fill: 255, mat: Material::STONE };
    const DIRT: Sample = Sample { fill: 255, mat: Material::DIRT };

    fn rules() -> EditRules {
        EditRules::new(&WorldParams::default())
    }

    fn volume_of(sample: Sample) -> Volume {
        let mut vol = Volume::new();
        for y in -2..2 {
            for z in -2..2 {
                for x in -2..2 {
                    vol.insert(IVec3::new(x, y, z), Chunk::Uniform(sample));
                }
            }
        }
        vol
    }

    fn solid_count(vol: &Volume) -> usize {
        vol.chunks
            .values()
            .map(|c| match &**c {
                Chunk::Uniform(s) => (s.fill >= SOLID) as usize * crate::CHUNK_VOLUME,
                Chunk::Dense(d) => d.fill.iter().filter(|&&f| f >= SOLID).count(),
            })
            .sum()
    }

    fn ball(radius: f64) -> f64 {
        4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)
    }

    #[test]
    fn breaking_dirt_cuts_a_ball() {
        let mut vol = volume_of(DIRT);
        let before = solid_count(&vol);
        let op = Op::Break { center: [100, -300, 500], radius: 9 * FP, nearest: false };
        let out = edit_volume(&mut vol, &rules(), &op);
        assert!(out.covered && out.fragments.is_none() && !out.chunks.is_empty());
        let removed = (before - solid_count(&vol)) as f64;
        assert!((removed / ball(9.0) - 1.0).abs() < 0.03, "removed {removed}, a ball is {}", ball(9.0));
        assert!(!vol.get(IVec3::new(0, -1, 2)).unwrap().solid());
        assert_eq!(vol.get(IVec3::new(0, -1, 2)).unwrap().mat, 0);
        assert_eq!(vol.get(IVec3::new(14, -1, 2)).unwrap(), DIRT);
        // The same op again changes nothing.
        assert!(edit_volume(&mut vol, &rules(), &op).chunks.is_empty());
    }

    #[test]
    fn stone_breaks_in_whole_shards() {
        let rules = rules();
        let mut vol = volume_of(STONE);
        let before = solid_count(&vol);
        let center = [300, 200, -700];
        // Too small to hold a shard centre: without `nearest` nothing gives.
        let small = Op::Break { center, radius: FP, nearest: false };
        assert!(edit_volume(&mut vol, &rules, &small).chunks.is_empty());
        let op = Op::Break { center, radius: FP, nearest: true };
        let out = edit_volume(&mut vol, &rules, &op);
        assert!(!out.chunks.is_empty());
        // One shard of about spacing³ went, not a one-cell ball.
        let removed = (before - solid_count(&vol)) as f64;
        let shard = (rules.shard_spacing as f64 / FP as f64).powi(3);
        assert!(removed > shard * 0.2 && removed < shard * 4.0, "removed {removed}, a shard is about {shard}");
        // The shard that went is the one the centre is in.
        assert!(!vol.get(IVec3::new(1, 0, -3)).unwrap().solid());

        // A larger ball takes every shard centred in it, and only whole ones:
        // each sample's fate follows that of its shard.
        let mut vol = volume_of(STONE);
        let radius = 14 * FP;
        edit_volume(&mut vol, &rules, &Op::Break { center, radius, nearest: false });
        let (mut gone, mut kept) = (0, 0);
        for y in -30..30 {
            for z in -30..30 {
                for x in -30..30 {
                    let p = IVec3::new(x, y, z);
                    let w = p.to_array().map(|v| v as i64 * FP);
                    let site = rules.site(rules.nearest_site(w));
                    let d = [site[0] - center[0], site[1] - center[1], site[2] - center[2]];
                    let taken = len2(d) <= radius * radius;
                    let solid = vol.get(p).unwrap().solid();
                    // Near a face between a taken and a kept shard either is possible.
                    if taken && !solid {
                        gone += 1;
                    } else if !taken && solid {
                        kept += 1;
                    }
                }
            }
        }
        let total = 60 * 60 * 60;
        assert!(gone > 5000 && gone + kept > total * 97 / 100, "gone {gone}, kept {kept} of {total}");
    }

    #[test]
    fn placing_adds_a_ball_and_never_repaints() {
        let rules = rules();
        let mut vol = volume_of(Sample::AIR);
        let dirt = Op::Place { center: [0, 0, 0], radius: 8 * FP, material: Material::DIRT, exclude: None };
        edit_volume(&mut vol, &rules, &dirt);
        let first = solid_count(&vol) as f64;
        assert!((first / ball(8.0) - 1.0).abs() < 0.04, "placed {first}, a ball is {}", ball(8.0));
        // Sand over half of it: the dirt stays dirt, the new part is sand.
        let sand = Op::Place { center: [8 * FP, 0, 0], radius: 8 * FP, material: Material::SAND, exclude: None };
        edit_volume(&mut vol, &rules, &sand);
        assert!(solid_count(&vol) as f64 > first * 1.3);
        assert_eq!(vol.get(IVec3::new(1, 0, 0)).unwrap().mat, Material::DIRT);
        assert_eq!(vol.get(IVec3::new(12, 0, 0)).unwrap().mat, Material::SAND);
        // Placed stone is made of shards, so it is not a ball.
        let mut vol = volume_of(Sample::AIR);
        let stone = Op::Place { center: [0, 0, 0], radius: 8 * FP, material: Material::STONE, exclude: None };
        edit_volume(&mut vol, &rules, &stone);
        let placed = solid_count(&vol) as f64;
        assert!(placed > ball(8.0) * 0.3 && placed < ball(8.0) * 3.0, "{placed}");
    }

    #[test]
    fn placing_leaves_the_excluded_capsule_empty() {
        let mut vol = volume_of(Sample::AIR);
        let exclude = Exclude { x: 0, z: 0, y0: -2 * FP, y1: 4 * FP, radius: 3 * FP };
        let op = Op::Place { center: [0, 0, 0], radius: 10 * FP, material: Material::DIRT, exclude: Some(exclude) };
        edit_volume(&mut vol, &rules(), &op);
        for y in -2..=4 {
            assert_eq!(vol.get(IVec3::new(1, y, -1)).unwrap().fill, 0, "inside the capsule at y {y}");
        }
        assert!(vol.get(IVec3::new(5, 0, 0)).unwrap().solid());
        assert!(vol.get(IVec3::new(0, -7, 0)).unwrap().solid());
    }

    #[test]
    fn small_exposed_pieces_break_off_and_are_reported() {
        let rules = rules();
        // A stone floor with a short dirt post standing on it.
        let mut vol = volume_of(Sample::AIR);
        for y in -20..0 {
            for z in -30..30 {
                for x in -30..30 {
                    vol.set(IVec3::new(x, y, z), STONE);
                }
            }
        }
        for y in 0..3 {
            vol.set(IVec3::new(0, y, 0), DIRT);
        }
        let before = solid_count(&vol);
        // Dig beside the post, clipping its top: what is left is too small to keep.
        let op = Op::Break { center: [3 * FP, 2 * FP, 0], radius: 3 * FP, nearest: false };
        let mut replay = Volume::new();
        for (&p, c) in &vol.chunks {
            replay.chunks.insert(p, c.clone());
        }
        let out = edit_volume(&mut vol, &rules, &op);
        let Some(Op::Clear { cells, .. }) = &out.fragments else { panic!("nothing broke off") };
        assert_eq!(cells.len(), 3, "the ball only grazed the post, so all of it broke off");
        assert!((0..3).all(|y| !vol.get(IVec3::new(0, y, 0)).unwrap().solid()), "the post is still there");
        // The floor far from the brush is untouched.
        assert_eq!(vol.get(IVec3::new(20, -1, 20)).unwrap(), STONE);
        assert!(before - solid_count(&vol) < 1500);

        // Replaying the op and the reported fragments reproduces the result,
        // without looking for fragments again.
        let mut field = replay.copy_region(out.min, out.max - out.min + 1);
        apply(&op, &rules, &mut field, None);
        apply(out.fragments.as_ref().unwrap(), &rules, &mut field, None);
        let now = vol.copy_region(out.min, out.max - out.min + 1);
        assert!(field.fill == now.fill && field.mat == now.mat);
    }

    /// A session that digs and builds with every material.
    fn session() -> Vec<Op> {
        let mut ops = Vec::new();
        for k in 0..24i64 {
            let center = [k * 310 - 3000, 200 - k * 90, k * 170 - 1500];
            ops.push(match k % 4 {
                0 => Op::Break { center, radius: 5 * FP + k * 40, nearest: false },
                1 => Op::Place { center, radius: 4 * FP, material: Material::SAND, exclude: None },
                2 => Op::Break { center, radius: 2 * FP, nearest: true },
                _ => Op::Place { center, radius: 6 * FP, material: Material::STONE, exclude: None },
            });
        }
        ops.push(Op::Clear {
            origin: IVec3::new(-4, 0, -4),
            cells: Box::new([[0, 0, 0], [4, 0, 4], [3, 1, 2], [8, 0, 8]]),
        });
        ops
    }

    /// Half stone, half dirt, at any level.
    fn ground(origin: IVec3, dims: IVec3, level: u8) -> LocalField {
        let mut f = LocalField::new(origin, dims, level);
        for y in 0..dims.y {
            for z in 0..dims.z {
                for x in 0..dims.x {
                    let i = f.index(x, y, z);
                    let base = (origin + IVec3::new(x, y, z)) << level as i32;
                    if base.y < 2 {
                        f.fill[i] = 255;
                        f.mat[i] = if base.x < 0 { Material::STONE } else { Material::DIRT };
                    }
                }
            }
        }
        f
    }

    #[test]
    fn an_op_gives_the_same_samples_whole_or_in_pieces() {
        let rules = rules();
        let (origin, dims) = (IVec3::new(-40, -30, -40), IVec3::new(80, 60, 80));
        let mut whole = ground(origin, dims, 0);
        let mut pieces = ground(origin, dims, 0);
        // The same field again, but resolved through eight clip boxes.
        for op in session() {
            apply(&op, &rules, &mut whole, None);
            for i in 0..8 {
                let half = IVec3::new(i & 1, (i >> 1) & 1, i >> 2);
                let lo = origin + half * dims / 2;
                apply(&op, &rules, &mut pieces, Some((lo, lo + dims / 2 - 1)));
            }
        }
        assert!(whole.fill == pieces.fill && whole.mat == pieces.mat);
        // And a small field of its own agrees with the same samples of the large one.
        let (o2, d2) = (IVec3::new(-9, -7, -11), IVec3::new(17, 12, 19));
        let mut small = ground(o2, d2, 0);
        for op in session() {
            apply(&op, &rules, &mut small, None);
        }
        for y in 0..d2.y {
            for z in 0..d2.z {
                for x in 0..d2.x {
                    let l = o2 + IVec3::new(x, y, z) - origin;
                    assert_eq!(small.get(x, y, z), whole.get(l.x, l.y, l.z));
                }
            }
        }
    }

    #[test]
    fn every_level_gets_the_same_surface() {
        let rules = rules();
        let mut fine = ground(IVec3::new(-64, -48, -64), IVec3::new(128, 96, 128), 0);
        for op in session() {
            apply(&op, &rules, &mut fine, None);
        }
        for level in 1..=3u8 {
            let step = 1 << level;
            let mut coarse = ground(fine.origin >> level as i32, fine.dims >> level as i32, level);
            for op in session() {
                apply(&op, &rules, &mut coarse, None);
            }
            // Where the two lattices share a point, they agree on which side
            // of the surface it is and what it is made of.
            let (mut solid, mut air) = (0, 0);
            for y in 0..coarse.dims.y {
                for z in 0..coarse.dims.z {
                    for x in 0..coarse.dims.x {
                        let c = coarse.get(x, y, z);
                        let f = fine.get(x * step, y * step, z * step);
                        assert_eq!(c.solid(), f.solid(), "level {level} at {x} {y} {z}: {c:?} against {f:?}");
                        if c.solid() {
                            assert_eq!(c.mat, f.mat, "level {level} at {x} {y} {z}");
                            solid += 1;
                        } else {
                            air += 1;
                        }
                    }
                }
            }
            assert!(solid > 100 && air > 100, "level {level}: {solid} solid, {air} air");
        }
    }

    #[test]
    fn bounds_cover_everything_an_op_changes() {
        let rules = rules();
        for level in [0u8, 2, 5] {
            let reach = 80 >> level;
            let origin = IVec3::new(-reach, -reach, -reach);
            for op in session() {
                let mut field = ground(origin, IVec3::splat(2 * reach), level);
                let before = (field.fill.clone(), field.mat.clone());
                apply(&op, &rules, &mut field, None);
                let (min, max) = op.bounds(&rules, level);
                let (all_min, all_max) = op.bounds_all(&rules);
                assert!((min << level as i32).cmpge(all_min).all() && (max << level as i32).cmple(all_max).all());
                for y in 0..field.dims.y {
                    for z in 0..field.dims.z {
                        for x in 0..field.dims.x {
                            let i = field.index(x, y, z);
                            if field.fill[i] != before.0[i] || field.mat[i] != before.1[i] {
                                let p = origin + IVec3::new(x, y, z);
                                assert!(
                                    p.cmpge(min).all() && p.cmple(max).all(),
                                    "{op:?} changed {p} at level {level}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
