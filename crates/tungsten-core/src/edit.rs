//! Edit operations: the only way the world changes after generation.
//!
//! Part of the bit-exact contract. An [`EditOp`] carries fixed-point
//! positions and everything in [`apply`] is integer arithmetic, so applying
//! the same ops in the same order gives identical samples on every machine.

use glam::IVec3;

use crate::chunk::Volume;
use crate::coords::{ChunkPos, FILL_PER_CELL, SOLID, WorldParams};
use crate::field::LocalField;
use crate::hash::{derive, hash3};
use crate::material::{self, CarveMode, Material};

/// Fixed-point units per cell for edit positions and radii.
pub const FP: i64 = 256;
/// Fixed-point distance over which fill changes by one unit.
const FP_PER_FILL: i64 = FP / FILL_PER_CELL as i64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditKind {
    /// One tick of mining with the brush.
    Break,
    /// One stamp of `material`.
    Place,
}

/// A vertical capsule that a `Place` leaves untouched (the player's body).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exclude {
    pub x: i64,
    pub z: i64,
    pub y0: i64,
    pub y1: i64,
    pub radius: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditOp {
    pub kind: EditKind,
    /// Brush centre in fixed-point cells.
    pub center: [i64; 3],
    /// Brush radius in fixed-point cells.
    pub radius: i64,
    pub material: u8,
    /// Simulation tick; seeds the dither that makes slow carving smooth.
    pub tick: u64,
    pub exclude: Option<Exclude>,
}

/// World constants the edit rules depend on.
#[derive(Clone, Copy, Debug)]
pub struct EditRules {
    seed: u64,
    /// Spacing of the shard pattern stone breaks along, fixed-point cells.
    pub shard_spacing: i64,
    /// Pieces of fewer samples than this are broken off; see `WorldParams::min_fragment`.
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
}

#[derive(Debug, Default)]
pub struct EditOutcome {
    /// Chunks whose samples changed.
    pub chunks: Vec<ChunkPos>,
    /// Sample bounds of the region the op could have changed (inclusive).
    pub min: IVec3,
    pub max: IVec3,
}

#[inline]
fn clamp_fill(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

/// Fill removed this tick at squared distance `d2` from the centre of a
/// brush of squared radius `r2`, for a material with `rate` (1/256 fill per
/// tick at the centre). `dither` spreads the fractional part over ticks.
fn erode_amount(rate: u32, d2: i64, r2: i64, dither: u64) -> u8 {
    if d2 >= r2 {
        return 0;
    }
    let k = 65536 - (d2 << 16) / r2;
    let falloff = (k * k) >> 16;
    let a = (rate as i64 * falloff) >> 16;
    let whole = (a >> 8) + ((dither & 255) < (a & 255) as u64) as i64;
    whole.min(255) as u8
}

/// The lattice of shard centres covering a region, relative to the brush centre.
struct Sites {
    spacing: i64,
    /// Grid cell of index (0, 0, 0).
    base: [i64; 3],
    dims: [i64; 3],
    /// Position of each site relative to the brush centre.
    pos: Vec<[i64; 3]>,
}

impl Sites {
    /// Sites for every grid cell a sample of `field` can ask about.
    fn new(rules: &EditRules, center: [i64; 3], field: &LocalField) -> Self {
        let sp = rules.shard_spacing;
        let min = field.origin.to_array().map(|v| v as i64 * FP);
        let max = (field.origin + field.dims - 1).to_array().map(|v| v as i64 * FP);
        let base: [i64; 3] = std::array::from_fn(|a| min[a].div_euclid(sp) - 1);
        let dims: [i64; 3] = std::array::from_fn(|a| max[a].div_euclid(sp) + 1 - base[a] + 1);
        let mut pos = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
        let (inset, span) = (sp / 10, (sp * 8 / 10) as u64);
        for gy in 0..dims[1] {
            for gz in 0..dims[2] {
                for gx in 0..dims[0] {
                    let g = [base[0] + gx, base[1] + gy, base[2] + gz];
                    let h = hash3(rules.seed, g[0], g[1], g[2]);
                    let jitter = [h & 0xffff, (h >> 16) & 0xffff, (h >> 32) & 0xffff];
                    pos.push(std::array::from_fn(|a| g[a] * sp + inset + (jitter[a] % span) as i64 - center[a]));
                }
            }
        }
        Self { spacing: sp, base, dims, pos }
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
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shard {
    Untouched,
    /// Worn down to this fill level, still solid.
    Worn(u8),
    /// Broke off this tick.
    Popped,
}

/// Apply one edit. Chunks that are not loaded are left alone.
pub fn apply(volume: &mut Volume, rules: &EditRules, op: &EditOp) -> EditOutcome {
    let shards = match op.kind {
        // Stone may be anywhere under a break brush.
        EditKind::Break => true,
        EditKind::Place => material::def(op.material).carve == CarveMode::Shards,
    };
    let reach = op.radius + 2 * FP + if shards { 2 * rules.shard_spacing } else { 0 };
    let lo: [i64; 3] = std::array::from_fn(|a| op.center[a] - reach);
    let hi: [i64; 3] = std::array::from_fn(|a| op.center[a] + reach);
    let min = IVec3::from_array(lo.map(|v| v.div_euclid(FP) as i32));
    let max = IVec3::from_array(hi.map(|v| v.div_euclid(FP) as i32 + 1));
    let mut field = volume.copy_region(min, max - min + 1);
    let mut out = EditOutcome { chunks: Vec::new(), min, max };

    match op.kind {
        EditKind::Break => {
            let before = field.fill.clone();
            break_brush(&mut field, rules, op);
            remove_fragments(&mut field, &before, rules.fragment_samples);
        }
        EditKind::Place => place_brush(&mut field, rules, op),
    }
    volume.write_region(&field, &mut out.chunks);
    out
}

/// Visit every sample of a field as (index, world fixed position, position
/// relative to the brush centre).
fn for_samples(field: &LocalField, center: [i64; 3], mut f: impl FnMut(usize, [i64; 3], [i64; 3])) {
    let o = field.origin;
    let mut i = 0;
    for y in 0..field.dims.y {
        let wy = (o.y + y) as i64 * FP;
        for z in 0..field.dims.z {
            let wz = (o.z + z) as i64 * FP;
            for x in 0..field.dims.x {
                let w = [(o.x + x) as i64 * FP, wy, wz];
                f(i, w, [w[0] - center[0], w[1] - center[1], w[2] - center[2]]);
                i += 1;
            }
        }
    }
}

#[inline]
fn len2(p: [i64; 3]) -> i64 {
    p[0] * p[0] + p[1] * p[1] + p[2] * p[2]
}

fn break_brush(field: &mut LocalField, rules: &EditRules, op: &EditOp) {
    let r2 = op.radius * op.radius;
    let dither_seed = derive(rules.seed, op.tick);
    let stone = Material::STONE;
    let mut any_stone = false;

    // Materials that erode: fill wears away under a smooth falloff.
    let (fill, mat) = (&mut field.fill, &mut field.mat);
    let dims = field.dims;
    let origin = field.origin;
    let view = LocalField { origin, dims, fill: Vec::new(), mat: Vec::new() };
    for_samples(&view, op.center, |i, w, p| {
        if fill[i] == 0 {
            return;
        }
        let def = material::def(mat[i]);
        if def.carve == CarveMode::Shards {
            any_stone = true;
            return;
        }
        let d2 = len2(p);
        if d2 < r2 {
            let dither = hash3(dither_seed, w[0], w[1], w[2]);
            fill[i] = fill[i].saturating_sub(erode_amount(def.carve_rate, d2, r2, dither));
            if fill[i] == 0 {
                mat[i] = 0;
            }
        }
    });
    if !any_stone {
        return;
    }

    // Stone: each shard wears as a unit and breaks off whole.
    let sites = Sites::new(rules, op.center, &view);
    let mut level = vec![0u8; sites.pos.len()];
    let mut nearest = vec![u32::MAX; fill.len()];
    for_samples(&view, op.center, |i, w, p| {
        if fill[i] == 0 || mat[i] != stone {
            return;
        }
        let c = sites.candidates(w, p);
        let n = c[Sites::nearest(&c)].0;
        nearest[i] = n;
        level[n as usize] = level[n as usize].max(fill[i]);
    });

    let reach = op.radius + sites.spacing / 2;
    let reach2 = reach * reach;
    let rate = material::def(stone).carve_rate;
    let mut any_popped = false;
    let state: Vec<Shard> = (0..sites.pos.len())
        .map(|n| {
            if level[n] < SOLID {
                return Shard::Untouched;
            }
            let s = sites.pos[n];
            let dither = hash3(dither_seed, s[0] + op.center[0], s[1] + op.center[1], s[2] + op.center[2]);
            let amount = erode_amount(rate, len2(s), reach2, dither);
            if amount == 0 {
                return Shard::Untouched;
            }
            let worn = level[n].saturating_sub(amount);
            if worn < SOLID {
                any_popped = true;
                Shard::Popped
            } else {
                Shard::Worn(worn)
            }
        })
        .collect();

    for_samples(&view, op.center, |i, w, p| {
        let n = nearest[i];
        if n == u32::MAX {
            return;
        }
        match state[n as usize] {
            Shard::Worn(l) => fill[i] = fill[i].min(l),
            Shard::Popped => {
                // Cut the shard's volume away, leaving flat faces where it
                // met its neighbours.
                let c = sites.candidates(w, p);
                let own = c[Sites::nearest(&c)];
                let mut inside = i64::MAX;
                for &(j, dj2) in &c {
                    if j != own.0 {
                        inside = inside.min(sites.bisector(own.0, own.1, j, dj2));
                    }
                }
                fill[i] = fill[i].min(clamp_fill(SOLID as i64 - 1 - inside / FP_PER_FILL));
            }
            Shard::Untouched => {}
        }
        if any_popped && state[n as usize] != Shard::Popped {
            // Next to a shard that just broke off: bevel towards the shared face.
            let c = sites.candidates(w, p);
            let own = c[Sites::nearest(&c)];
            for &(j, dj2) in &c {
                if state[j as usize] == Shard::Popped {
                    let d = sites.bisector(own.0, own.1, j, dj2);
                    fill[i] = fill[i].min(clamp_fill(SOLID as i64 - 1 + d / FP_PER_FILL));
                }
            }
        }
        if fill[i] == 0 {
            mat[i] = 0;
        }
    });
}

/// Break off what an edit left too small to keep: a connected run of solid
/// samples of one material, open to the air, of fewer than `limit` samples.
/// `before` is the fill as it was before the edit; only pieces beside a sample
/// the edit changed are considered, so nothing far from the brush is touched.
fn remove_fragments(field: &mut LocalField, before: &[u8], limit: u32) {
    // A piece cannot be smaller than one sample.
    if limit <= 1 {
        return;
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
            // Whatever this was holding on to may now be loose as well.
            seeds.extend(around(c));
        }
    }
}

fn place_brush(field: &mut LocalField, rules: &EditRules, op: &EditOp) {
    if op.material == Material::AIR || op.material as usize >= material::MATERIAL_COUNT {
        return;
    }
    let excluded = |w: [i64; 3]| {
        op.exclude.is_some_and(|e| {
            let dy = w[1] - w[1].clamp(e.y0, e.y1);
            len2([w[0] - e.x, dy, w[2] - e.z]) < e.radius * e.radius
        })
    };
    let (fill, mat) = (&mut field.fill, &mut field.mat);
    let view = LocalField { origin: field.origin, dims: field.dims, fill: Vec::new(), mat: Vec::new() };
    let mut stamp = |i: usize, target: u8| {
        if target > fill[i] {
            if fill[i] < SOLID {
                mat[i] = op.material;
            }
            fill[i] = target;
        }
    };

    if material::def(op.material).carve != CarveMode::Shards {
        // A ball: fill rises to the ball's profile, never falls.
        for_samples(&view, op.center, |i, w, p| {
            let d = (len2(p) as u64).isqrt() as i64;
            let target = clamp_fill(SOLID as i64 + (op.radius - d) / FP_PER_FILL);
            if target > 0 && !excluded(w) {
                stamp(i, target);
            }
        });
        return;
    }

    // Stone: the union of the shards centred inside the brush, so placed
    // stone is angular and lines up with the pattern it later breaks along.
    let sites = Sites::new(rules, op.center, &view);
    let r2 = op.radius * op.radius;
    let mut placed: Vec<bool> = sites.pos.iter().map(|&s| len2(s) <= r2).collect();
    // A brush smaller than the pattern still places the shard it is in.
    let center_candidates = sites.candidates(op.center, [0, 0, 0]);
    placed[center_candidates[Sites::nearest(&center_candidates)].0 as usize] = true;

    for_samples(&view, op.center, |i, w, p| {
        let c = sites.candidates(w, p);
        let own = c[Sites::nearest(&c)];
        let own_placed = placed[own.0 as usize];
        // Distance to the nearest face between placed and unplaced shards.
        let mut face = i64::MAX;
        for &(j, dj2) in &c {
            if j != own.0 && placed[j as usize] != own_placed {
                face = face.min(sites.bisector(own.0, own.1, j, dj2));
            }
        }
        let target = match (own_placed, face) {
            (true, i64::MAX) => 255,
            (true, d) => clamp_fill(SOLID as i64 + d / FP_PER_FILL),
            (false, i64::MAX) => 0,
            (false, d) => clamp_fill(SOLID as i64 - 1 - d / FP_PER_FILL),
        };
        if target > 0 && !excluded(w) {
            stamp(i, target);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Sample};

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

    fn hash(vol: &Volume) -> u64 {
        let mut keys: Vec<_> = vol.chunks.keys().copied().collect();
        keys.sort_by_key(|k| (k.x, k.y, k.z));
        keys.iter().fold(0, |h, k| vol.chunks[k].content_hash(h))
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

    fn op(kind: EditKind, material: u8, radius_cells: i64, tick: u64) -> EditOp {
        EditOp { kind, center: [70, -40, 130], radius: radius_cells * FP, material, tick, exclude: None }
    }

    fn rules() -> EditRules {
        EditRules::new(&WorldParams::default())
    }

    #[test]
    fn dirt_erodes_into_a_ball_at_the_documented_speed() {
        let mut vol = volume_of(Sample { fill: 255, mat: Material::DIRT });
        let full = solid_count(&vol);
        // Dirt takes 0.4 s to open at the centre: nothing solid is lost at
        // 0.3 s, a hole exists by 0.5 s.
        for tick in 0..18 {
            apply(&mut vol, &rules(), &op(EditKind::Break, 0, 4, tick));
        }
        assert_eq!(solid_count(&vol), full);
        for tick in 18..30 {
            apply(&mut vol, &rules(), &op(EditKind::Break, 0, 4, tick));
        }
        assert!(vol.get(IVec3::ZERO).unwrap().fill < SOLID);
        // Held long enough, the hole approaches the brush ball (268 samples)
        // from inside: the rim wears ever more slowly.
        for tick in 30..600 {
            apply(&mut vol, &rules(), &op(EditKind::Break, 0, 4, tick));
        }
        let hole = full - solid_count(&vol);
        assert!((150..=268).contains(&hole), "hole of {hole} samples");
        assert_eq!(vol.get(IVec3::new(5, 0, 0)).unwrap().fill, 255);
        assert_eq!(vol.get(IVec3::ZERO).unwrap(), Sample::AIR);
    }

    #[test]
    fn stone_holds_then_breaks_off_in_whole_shards() {
        let mut vol = volume_of(Sample { fill: 255, mat: Material::STONE });
        let full = solid_count(&vol);
        let mut popped_at = None;
        for tick in 0..90 {
            let before = solid_count(&vol);
            apply(&mut vol, &rules(), &op(EditKind::Break, 0, 3, tick));
            let lost = before - solid_count(&vol);
            if lost > 0 && popped_at.is_none() {
                popped_at = Some(tick);
                // The first break removes a shard-sized piece at once, not a sliver.
                assert!(lost >= 8, "first break removed only {lost} samples");
            }
        }
        // Stone takes 1.2 s (72 ticks) at the centre of the brush.
        let t = popped_at.expect("stone never broke");
        assert!((60..=80).contains(&t), "first shard broke at tick {t}");
        assert!(full - solid_count(&vol) < 600);
    }

    #[test]
    fn placing_dirt_makes_a_ball_and_respects_the_exclusion() {
        let mut vol = volume_of(Sample::AIR);
        let mut o = op(EditKind::Place, Material::DIRT, 4, 0);
        apply(&mut vol, &rules(), &o);
        let n = solid_count(&vol);
        assert!((230..=300).contains(&n), "ball of {n} samples");
        assert_eq!(vol.get(IVec3::ZERO).unwrap(), Sample { fill: 255, mat: Material::DIRT });
        // Placing again changes nothing: a stamp only raises fill.
        let h = hash(&vol);
        assert!(apply(&mut vol, &rules(), &o).chunks.is_empty());
        assert_eq!(h, hash(&vol));

        let mut vol = volume_of(Sample::AIR);
        o.exclude = Some(Exclude { x: 0, z: 0, y0: -2 * FP, y1: 2 * FP, radius: 2 * FP });
        apply(&mut vol, &rules(), &o);
        assert_eq!(vol.get(IVec3::ZERO).unwrap(), Sample::AIR);
        assert_eq!(vol.get(IVec3::new(1, 1, 0)).unwrap(), Sample::AIR);
        assert!(vol.get(IVec3::new(3, 0, 0)).unwrap().fill >= SOLID);
    }

    #[test]
    fn placing_stone_makes_angular_solid_and_never_repaints() {
        let mut vol = volume_of(Sample::AIR);
        apply(&mut vol, &rules(), &op(EditKind::Place, Material::STONE, 3, 0));
        let n = solid_count(&vol);
        assert!(n > 20, "placed only {n} samples");
        assert!(vol.chunks.values().all(|c| match &**c {
            Chunk::Uniform(_) => true,
            Chunk::Dense(d) => d.fill.iter().zip(d.mat.iter()).all(|(&f, &m)| (f == 0) == (m == 0) && m <= 1),
        }));
        // Even a brush smaller than the shard pattern places something.
        let mut small = volume_of(Sample::AIR);
        let mut o = op(EditKind::Place, Material::STONE, 1, 0);
        o.radius = FP / 2;
        apply(&mut small, &rules(), &o);
        assert!(solid_count(&small) > 4);

        // Dirt placed over the stone does not change stone samples.
        let stone_before: Vec<_> = (0..4).map(|x| vol.get(IVec3::new(x, 0, 0)).unwrap()).collect();
        apply(&mut vol, &rules(), &op(EditKind::Place, Material::DIRT, 5, 1));
        for (x, s) in stone_before.iter().enumerate() {
            if s.fill >= SOLID {
                assert_eq!(vol.get(IVec3::new(x as i32, 0, 0)).unwrap().mat, Material::STONE);
            }
        }
    }

    #[test]
    fn same_ops_give_same_world_regardless_of_representation() {
        let ops: Vec<EditOp> = (0..120)
            .map(|t| EditOp {
                kind: if t % 40 == 39 { EditKind::Place } else { EditKind::Break },
                center: [t as i64 * 37 - 900, (t as i64 * 13) % 500 - 250, 300 - t as i64 * 11],
                radius: 2 * FP + (t as i64 % 5) * 100,
                material: 1 + (t % 3) as u8,
                tick: t,
                exclude: None,
            })
            .collect();
        let run = |dense: bool| {
            let mut vol = volume_of(Sample { fill: 255, mat: Material::STONE });
            // A dirt layer so both carve modes meet.
            for (pos, c) in vol.chunks.iter_mut() {
                if pos.y == 0 {
                    *c = std::sync::Arc::new(Chunk::Uniform(Sample { fill: 255, mat: Material::DIRT }));
                }
                if dense {
                    std::sync::Arc::make_mut(c).dense_mut();
                }
            }
            for o in &ops {
                apply(&mut vol, &rules(), o);
            }
            hash(&vol)
        };
        let a = run(false);
        assert_eq!(a, run(true));
        assert_eq!(a, run(false));
    }

    #[test]
    fn small_exposed_pieces_break_off_and_nothing_else_does() {
        let dims = IVec3::splat(14);
        let n = (14 * 14 * 14) as usize;
        let mut f = LocalField { origin: IVec3::ZERO, dims, fill: vec![0; n], mat: vec![0; n] };
        let put = |f: &mut LocalField, x: i32, y: i32, z: i32, mat: u8| {
            let i = f.index(x, y, z);
            f.fill[i] = 255;
            f.mat[i] = mat;
        };
        // A stone floor four samples thick.
        for y in 0..4 {
            for z in 0..14 {
                for x in 0..14 {
                    put(&mut f, x, y, z, Material::STONE);
                }
            }
        }
        // On it: a sliver of dirt with a crumb of sand on top, and a mound of dirt.
        put(&mut f, 3, 4, 3, Material::DIRT);
        put(&mut f, 4, 4, 3, Material::DIRT);
        put(&mut f, 3, 5, 3, Material::SAND);
        for y in 4..6 {
            for z in 8..10 {
                for x in 8..10 {
                    put(&mut f, x, y, z, Material::DIRT);
                }
            }
        }
        // In it: a pocket of dirt that never sees daylight.
        put(&mut f, 6, 2, 6, Material::DIRT);

        // Pretend the edit touched the air beside each of them.
        let mut before = f.fill.clone();
        for (x, y, z) in [(5, 4, 3), (7, 4, 8), (6, 3, 6)] {
            before[f.index(x, y, z)] ^= 1;
        }
        let solid_before = f.fill.iter().filter(|&&v| v >= SOLID).count();
        remove_fragments(&mut f, &before, 8);

        // The sliver goes, and the crumb it carried goes with it.
        for (x, y, z) in [(3, 4, 3), (4, 4, 3), (3, 5, 3)] {
            assert_eq!(f.get(x, y, z), Sample::AIR, "left behind at {x} {y} {z}");
        }
        // The mound (8 samples), the pocket and the floor are untouched.
        assert_eq!(f.fill.iter().filter(|&&v| v >= SOLID).count(), solid_before - 3);
        assert_eq!(f.get(6, 2, 6).mat, Material::DIRT);

        // A limit of zero keeps everything.
        put(&mut f, 3, 4, 3, Material::DIRT);
        remove_fragments(&mut f, &before, 0);
        assert_eq!(f.get(3, 4, 3).mat, Material::DIRT);
    }
}
