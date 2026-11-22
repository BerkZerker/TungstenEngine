//! The edit log: every change made to a world, filed by where it applies.
//!
//! Part of the bit-exact contract. The world is its generator plus this log.
//! Ops are filed in edit cells, each op in every cell it can reach. A sample
//! is resolved from its own cell alone: the cell's baked overlay if it has
//! one, then its ops in order.
//!
//! An op reaches two cells of a level beyond its shape, which is far at the
//! coarse levels. So there are two tiers of cells: fine ones of [`CELL_SIZE`]
//! base cells serving the three finest levels, and coarse ones eight times
//! the size serving the rest. Either way a cell is 128 samples across at the
//! finest level it serves, and an op lands in a handful of cells, not dozens.
//!
//! Baking replaces a cell's ops with an overlay of the samples they produced
//! at each level. The samples are the same before and after, so nothing that
//! was built from them needs to change.

use std::sync::Arc;

use glam::IVec3;

use crate::codec::{get_signed, get_u8, get_varint, overlay_apply, overlay_encode, put_signed, put_varint};
use crate::coords::{APRON, CHUNK_SIZE, LEVELS, NodeKey, chunk_of, chunk_origin};
use crate::edit::{self, EditRules, Exclude, Op};
use crate::field::LocalField;
use crate::hash::{IntMap, IntSet};
use crate::worldgen::Generator;

pub const CELL_BITS: i32 = 7;
/// Base cells along the edge of a fine edit cell.
pub const CELL_SIZE: i32 = 1 << CELL_BITS;
/// Levels each tier of cells serves.
pub const TIER_LEVELS: usize = 3;
pub const TIERS: usize = LEVELS / TIER_LEVELS;
const _: () = assert!(TIERS * TIER_LEVELS == LEVELS);

/// An edit cell: a cube of `CELL_SIZE << (3 * tier)` base cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CellPos {
    pub tier: u8,
    pub pos: IVec3,
}

impl CellPos {
    pub const fn new(tier: u8, pos: IVec3) -> Self {
        Self { tier, pos }
    }

    /// The tier that serves a level.
    #[inline]
    pub fn tier_of(level: u8) -> u8 {
        level / TIER_LEVELS as u8
    }

    /// Bits of base-cell coordinate below a cell coordinate of this tier.
    #[inline]
    pub fn bits(tier: u8) -> i32 {
        CELL_BITS + TIER_LEVELS as i32 * tier as i32
    }

    /// Levels this cell's tier serves.
    pub fn levels(self) -> std::ops::Range<u8> {
        let first = self.tier * TIER_LEVELS as u8;
        first..first + TIER_LEVELS as u8
    }

    fn order(&self) -> (u8, i32, i32, i32) {
        (self.tier, self.pos.y, self.pos.z, self.pos.x)
    }
}

const FORMAT: u8 = 1;

/// What a cell's ops produced, as an overlay on the generator at each level
/// its tier serves, finest first.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Baked {
    pub levels: [Vec<u8>; TIER_LEVELS],
}

/// The edits of one cell.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CellEdits {
    pub baked: Option<Arc<Baked>>,
    /// Ops since the bake, oldest first.
    pub ops: Vec<Op>,
}

/// Cells and their edits, as handed to whatever resolves samples.
pub type CellSet = Vec<(CellPos, Arc<CellEdits>)>;

#[derive(Default)]
pub struct EditLog {
    cells: IntMap<CellPos, Arc<CellEdits>>,
    /// Cells changed since [`take_dirty`](Self::take_dirty) was last called.
    dirty: IntSet<CellPos>,
}

/// Samples per axis, at `level`, of the edit cells that serve it.
#[inline]
fn cell_span(level: u8) -> i32 {
    (1 << CellPos::bits(CellPos::tier_of(level))) >> level
}

impl EditLog {
    pub fn new() -> Self {
        Self::default()
    }

    /// The cells of each tier that `op` can reach.
    pub fn cells_of(rules: &EditRules, op: &Op) -> Vec<CellPos> {
        let mut out = Vec::new();
        for tier in 0..TIERS as u8 {
            let (lo, hi) = op.bounds_over(rules, CellPos::new(tier, IVec3::ZERO).levels());
            let bits = CellPos::bits(tier);
            let (c0, c1) = (lo >> bits, hi >> bits);
            for y in c0.y..=c1.y {
                for z in c0.z..=c1.z {
                    for x in c0.x..=c1.x {
                        out.push(CellPos::new(tier, IVec3::new(x, y, z)));
                    }
                }
            }
        }
        out
    }

    /// File `op` in every cell it can reach.
    pub fn push(&mut self, rules: &EditRules, op: &Op) {
        for pos in Self::cells_of(rules, op) {
            Arc::make_mut(self.cells.entry(pos).or_default()).ops.push(op.clone());
            self.dirty.insert(pos);
        }
    }

    /// Put a cell's edits in place, as loaded from a save.
    pub fn insert(&mut self, pos: CellPos, edits: CellEdits) {
        self.cells.insert(pos, Arc::new(edits));
    }

    pub fn cell(&self, pos: CellPos) -> Option<&Arc<CellEdits>> {
        self.cells.get(&pos)
    }

    /// Replace a cell's first `ops` ops, and whatever it had baked, with `baked`.
    pub fn rebase(&mut self, pos: CellPos, ops: usize, baked: Arc<Baked>) {
        let Some(cell) = self.cells.get_mut(&pos) else { return };
        let cell = Arc::make_mut(cell);
        cell.ops.drain(..ops.min(cell.ops.len()));
        cell.baked = Some(baked);
        self.dirty.insert(pos);
    }

    /// Every cell of `tier` with edits that overlaps base-cell bounds `[min, max]`.
    pub fn gather(&self, tier: u8, min: IVec3, max: IVec3) -> CellSet {
        let bits = CellPos::bits(tier);
        let (c0, c1) = (min >> bits, max >> bits);
        let span = (c1 - c0 + 1).as_i64vec3();
        let mut out: CellSet = if span.x * span.y * span.z > self.cells.len() as i64 {
            let inside = |p: &CellPos| p.tier == tier && p.pos.cmpge(c0).all() && p.pos.cmple(c1).all();
            self.cells.iter().filter(|(p, _)| inside(p)).map(|(p, c)| (*p, c.clone())).collect()
        } else {
            let mut out = Vec::new();
            for y in c0.y..=c1.y {
                for z in c0.z..=c1.z {
                    for x in c0.x..=c1.x {
                        let pos = CellPos::new(tier, IVec3::new(x, y, z));
                        if let Some(c) = self.cells.get(&pos) {
                            out.push((pos, c.clone()));
                        }
                    }
                }
            }
            out
        };
        out.sort_unstable_by_key(|(p, _)| p.order());
        out
    }

    /// The cells a node's padded block needs.
    pub fn gather_node(&self, key: NodeKey) -> CellSet {
        if self.cells.is_empty() {
            return Vec::new();
        }
        let step = key.step();
        let min = (chunk_origin(key.pos) - APRON) * step;
        self.gather(CellPos::tier_of(key.level), min, min + (CHUNK_SIZE + 2 * APRON - 1) * step)
    }

    pub fn take_dirty(&mut self) -> Vec<CellPos> {
        let mut dirty: Vec<CellPos> = self.dirty.drain().collect();
        dirty.sort_unstable_by_key(|p| p.order());
        dirty
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Ops held, counting one filed in several cells once per cell.
    pub fn op_count(&self) -> usize {
        self.cells.values().map(|c| c.ops.len()).sum()
    }

    pub fn cells(&self) -> impl Iterator<Item = (&CellPos, &Arc<CellEdits>)> {
        self.cells.iter()
    }
}

/// Bring generated samples up to date with the log: for each cell, the
/// samples of `field` inside it take the cell's overlay and then its ops.
pub fn resolve(field: &mut LocalField, rules: &EditRules, cells: &[(CellPos, Arc<CellEdits>)]) {
    let n = cell_span(field.level);
    let tier = CellPos::tier_of(field.level);
    for (pos, edits) in cells {
        debug_assert_eq!(pos.tier, tier);
        let min = pos.pos * n;
        let clip = (min, min + n - 1);
        let Some((lo, hi)) = field.clip(clip.0, clip.1) else { continue };
        if let Some(bytes) = edits.baked.as_ref().map(|b| &b.levels[field.level as usize % TIER_LEVELS]) {
            let offset = min - field.origin;
            // A damaged overlay leaves the generated samples as they are.
            let _ = overlay_apply(bytes, (n * n * n) as usize, |i, fill, mat| {
                let i = i as i32;
                let l = IVec3::new(i % n, i / (n * n), i / n % n) + offset;
                if l.cmpge(lo).all() && l.cmplt(hi).all() {
                    let at = field.index(l.x, l.y, l.z);
                    field.fill[at] = fill;
                    field.mat[at] = mat;
                }
            });
        }
        for op in &edits.ops {
            edit::apply(op, rules, field, Some(clip));
        }
    }
}

/// Resolve a cell at every level it serves and return the result as an
/// overlay on the generator. Resolving with the overlay and no ops gives the
/// same samples as resolving with `edits`.
pub fn bake(generator: &Generator, rules: &EditRules, pos: CellPos, edits: &Arc<CellEdits>) -> Baked {
    let mut baked = Baked::default();
    let cells = [(pos, edits.clone())];
    for level in pos.levels() {
        let n = cell_span(level);
        let origin = pos.pos * n;
        let mut base = LocalField::new(origin, IVec3::splat(n), level);
        // Generated samples of the cell, block by block.
        let (c0, c1) = (chunk_of(origin), chunk_of(origin + n - 1));
        for cy in c0.y..=c1.y {
            for cz in c0.z..=c1.z {
                for cx in c0.x..=c1.x {
                    let key = NodeKey::new(level, IVec3::new(cx, cy, cz));
                    let mut block = LocalField::block(key);
                    generator.fill_block(key, &mut block);
                    let Some((lo, hi)) = base.clip(chunk_origin(key.pos), chunk_origin(key.pos) + CHUNK_SIZE - 1)
                    else {
                        continue;
                    };
                    let len = (hi.x - lo.x) as usize;
                    let shift = base.origin - block.origin;
                    for y in lo.y..hi.y {
                        for z in lo.z..hi.z {
                            let dst = base.index(lo.x, y, z);
                            let src = block.index(lo.x + shift.x, y + shift.y, z + shift.z);
                            base.fill[dst..dst + len].copy_from_slice(&block.fill[src..src + len]);
                            base.mat[dst..dst + len].copy_from_slice(&block.mat[src..src + len]);
                        }
                    }
                }
            }
        }
        let mut now = LocalField::new(origin, IVec3::splat(n), level);
        now.fill.copy_from_slice(&base.fill);
        now.mat.copy_from_slice(&base.mat);
        resolve(&mut now, rules, &cells);
        baked.levels[level as usize % TIER_LEVELS] = overlay_encode((&base.fill, &base.mat), (&now.fill, &now.mat));
    }
    baked
}

fn put_op(out: &mut Vec<u8>, op: &Op) {
    match op {
        Op::Break { center, radius, nearest } => {
            out.push(*nearest as u8);
            center.iter().for_each(|&c| put_signed(out, c));
            put_signed(out, *radius);
        }
        Op::Place { center, radius, material, exclude } => {
            out.push(2 + exclude.is_some() as u8);
            center.iter().for_each(|&c| put_signed(out, c));
            put_signed(out, *radius);
            out.push(*material);
            if let Some(e) = exclude {
                [e.x, e.z, e.y0, e.y1, e.radius].iter().for_each(|&v| put_signed(out, v));
            }
        }
        Op::Chip { center } => {
            out.push(5);
            center.iter().for_each(|&c| put_signed(out, c));
        }
        Op::Clear { origin, cells } => {
            out.push(4);
            origin.to_array().iter().for_each(|&c| put_signed(out, c as i64));
            put_varint(out, cells.len() as u64);
            cells.iter().for_each(|c| out.extend_from_slice(c));
        }
    }
}

fn get_op(bytes: &[u8], at: &mut usize) -> Result<Op, String> {
    let tag = get_u8(bytes, at)?;
    let three = |at: &mut usize| -> Result<[i64; 3], String> {
        Ok([get_signed(bytes, at)?, get_signed(bytes, at)?, get_signed(bytes, at)?])
    };
    Ok(match tag {
        0 | 1 => Op::Break { center: three(at)?, radius: get_signed(bytes, at)?, nearest: tag == 1 },
        5 => Op::Chip { center: three(at)? },
        2 | 3 => {
            let (center, radius, material) = (three(at)?, get_signed(bytes, at)?, get_u8(bytes, at)?);
            let exclude = if tag == 3 {
                let mut v = [0i64; 5];
                for slot in &mut v {
                    *slot = get_signed(bytes, at)?;
                }
                Some(Exclude { x: v[0], z: v[1], y0: v[2], y1: v[3], radius: v[4] })
            } else {
                None
            };
            Op::Place { center, radius, material, exclude }
        }
        4 => {
            let o = three(at)?;
            let origin = IVec3::new(o[0] as i32, o[1] as i32, o[2] as i32);
            let count = get_varint(bytes, at)? as usize;
            let end = at.checked_add(count.checked_mul(3).ok_or("bad cell count")?).ok_or("bad cell count")?;
            let raw = bytes.get(*at..end).ok_or("truncated")?;
            *at = end;
            Op::Clear { origin, cells: raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect() }
        }
        _ => return Err(format!("unknown op {tag}")),
    })
}

impl CellEdits {
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty() && self.baked.is_none()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![FORMAT];
        match &self.baked {
            Some(b) => {
                out.push(1);
                for level in &b.levels {
                    put_varint(&mut out, level.len() as u64);
                    out.extend_from_slice(level);
                }
            }
            None => out.push(0),
        }
        put_varint(&mut out, self.ops.len() as u64);
        self.ops.iter().for_each(|op| put_op(&mut out, op));
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut at = 0;
        if get_u8(bytes, &mut at)? != FORMAT {
            return Err(format!("unsupported edit format {}", bytes[0]));
        }
        let baked = match get_u8(bytes, &mut at)? {
            0 => None,
            1 => {
                let mut baked = Baked::default();
                for level in &mut baked.levels {
                    let len = get_varint(bytes, &mut at)? as usize;
                    let end = at.checked_add(len).ok_or("bad overlay length")?;
                    *level = bytes.get(at..end).ok_or("truncated")?.to_vec();
                    at = end;
                }
                Some(Arc::new(baked))
            }
            _ => return Err("bad bake flag".into()),
        };
        let count = get_varint(bytes, &mut at)? as usize;
        let mut ops = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            ops.push(get_op(bytes, &mut at)?);
        }
        if at != bytes.len() {
            return Err("trailing bytes".into());
        }
        Ok(Self { baked, ops })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{MAX_LEVEL, WorldParams};
    use crate::edit::FP;
    use crate::material::Material;

    /// Digging and building around the surface at the origin, across cell borders.
    fn session(generator: &Generator) -> Vec<Op> {
        let cell = generator.params().cell_size;
        let ground = (generator.surface_height(0.0, 0.0) / cell) as i64 * FP;
        (0..60i64)
            .map(|k| {
                let center = [k * 700 - 20_000, ground + 600 - k * 230, k * 450 - 14_000];
                match k % 5 {
                    0 | 1 => Op::Break { center, radius: 5 * FP + k * 30, nearest: k % 2 == 0 },
                    2 => Op::Place { center, radius: 6 * FP, material: Material::SAND, exclude: None },
                    3 if k % 2 == 0 => Op::Chip { center },
                    3 => Op::Place { center, radius: 7 * FP, material: Material::STONE, exclude: None },
                    _ => Op::Clear {
                        origin: IVec3::new((center[0] / FP) as i32, (center[1] / FP) as i32, (center[2] / FP) as i32),
                        cells: (0..40u8).map(|i| [i % 8, i / 8, i % 3]).collect(),
                    },
                }
            })
            .collect()
    }

    fn resolved(generator: &Generator, rules: &EditRules, log: &EditLog, key: NodeKey) -> LocalField {
        let mut block = LocalField::block(key);
        generator.fill_block(key, &mut block);
        resolve(&mut block, rules, &log.gather_node(key));
        block
    }

    /// Nodes of every level around the session.
    fn nodes(generator: &Generator) -> Vec<NodeKey> {
        let cell = generator.params().cell_size;
        let ground = (generator.surface_height(0.0, 0.0) / cell) as i32;
        let mut out = Vec::new();
        for level in 0..=MAX_LEVEL {
            let c = chunk_of(IVec3::new(-40, ground, -30) >> level as i32);
            for d in [IVec3::ZERO, IVec3::X, IVec3::NEG_Y, IVec3::new(-1, 0, 1)] {
                out.push(NodeKey::new(level, c + d));
            }
        }
        out
    }

    #[test]
    fn ops_are_filed_wherever_they_reach() {
        let params = WorldParams::default();
        let (generator, rules) = (Generator::new(params), EditRules::new(&params));
        let mut log = EditLog::new();
        let op = Op::Break { center: [0, 0, 0], radius: 4 * FP, nearest: false };
        log.push(&rules, &op);
        // At the corner of eight cells, in both tiers.
        assert_eq!(log.cell_count(), 16);
        assert_eq!(log.op_count(), 16);
        for tier in 0..TIERS as u8 {
            let far = 4 << CellPos::bits(tier);
            assert_eq!(log.gather(tier, IVec3::splat(-1), IVec3::splat(-1)).len(), 1);
            assert_eq!(log.gather(tier, IVec3::splat(-far), IVec3::splat(far)).len(), 8);
            assert!(log.gather(tier, IVec3::splat(far / 2), IVec3::splat(far)).is_empty());
        }
        assert_eq!(log.take_dirty().len(), 16);
        assert!(log.take_dirty().is_empty());
        // Well inside a cell an op is filed once per tier, not once per neighbour.
        let mut log = EditLog::new();
        let middle = (CELL_SIZE * 4 + CELL_SIZE / 2) as i64 * FP;
        log.push(&rules, &Op::Break { center: [middle; 3], radius: 4 * FP, nearest: false });
        assert_eq!(log.op_count(), 2);

        // A block resolved through the log is the block with the ops applied.
        let mut log = EditLog::new();
        let ops = session(&generator);
        ops.iter().for_each(|op| log.push(&rules, op));
        let mut changed = 0;
        for key in nodes(&generator) {
            let mut direct = LocalField::block(key);
            generator.fill_block(key, &mut direct);
            let generated = direct.fill.clone();
            ops.iter().for_each(|op| edit::apply(op, &rules, &mut direct, None));
            let through_log = resolved(&generator, &rules, &log, key);
            assert!(direct.fill == through_log.fill && direct.mat == through_log.mat, "{key:?}");
            changed += (generated != direct.fill) as usize;
        }
        assert!(changed >= 8, "the session should reach most of the nodes, changed {changed}");
    }

    #[test]
    fn baking_changes_no_sample_at_any_level() {
        let params = WorldParams::default();
        let (generator, rules) = (Generator::new(params), EditRules::new(&params));
        let mut log = EditLog::new();
        let ops = session(&generator);
        ops[..40].iter().for_each(|op| log.push(&rules, op));
        let before: Vec<LocalField> =
            nodes(&generator).iter().map(|&k| resolved(&generator, &rules, &log, k)).collect();

        // Bake every cell, then again after more ops on top of the bake.
        let bake_all = |log: &mut EditLog| {
            let cells: Vec<CellPos> = log.cells.keys().copied().collect();
            assert!(cells.iter().any(|c| c.tier == 0) && cells.iter().any(|c| c.tier == 1));
            for pos in cells {
                let edits = log.cell(pos).unwrap().clone();
                let baked = bake(&generator, &rules, pos, &edits);
                log.rebase(pos, edits.ops.len(), Arc::new(baked));
            }
        };
        bake_all(&mut log);
        assert_eq!(log.op_count(), 0);
        for (key, was) in nodes(&generator).iter().zip(&before) {
            let now = resolved(&generator, &rules, &log, *key);
            assert!(now.fill == was.fill && now.mat == was.mat, "{key:?} changed when baked");
        }

        let mut plain = EditLog::new();
        ops.iter().for_each(|op| plain.push(&rules, op));
        ops[40..].iter().for_each(|op| log.push(&rules, op));
        for stage in 0..2 {
            for key in nodes(&generator) {
                let (a, b) = (resolved(&generator, &rules, &plain, key), resolved(&generator, &rules, &log, key));
                assert!(a.fill == b.fill && a.mat == b.mat, "{key:?} differs, stage {stage}");
            }
            bake_all(&mut log);
        }
    }

    #[test]
    fn cells_round_trip_through_bytes() {
        let params = WorldParams::default();
        let (generator, rules) = (Generator::new(params), EditRules::new(&params));
        let mut log = EditLog::new();
        let exclude = Exclude { x: -5, z: 9, y0: -700, y1: 300, radius: 2 * FP };
        log.push(
            &rules,
            &Op::Place { center: [1, -2, 3], radius: FP, material: Material::DIRT, exclude: Some(exclude) },
        );
        session(&generator).iter().for_each(|op| log.push(&rules, op));
        let pos = *log.cells.keys().max_by_key(|p| log.cells[*p].ops.len()).unwrap();
        let mut edits = (**log.cell(pos).unwrap()).clone();
        assert_eq!(CellEdits::decode(&edits.encode()).unwrap(), edits);
        let baked = bake(&generator, &rules, pos, &Arc::new(edits.clone()));
        assert!(baked.levels.iter().any(|l| !l.is_empty()));
        edits.baked = Some(Arc::new(baked));
        let bytes = edits.encode();
        assert_eq!(CellEdits::decode(&bytes).unwrap(), edits);
        assert_eq!(bytes, CellEdits::decode(&bytes).unwrap().encode());
        // Damage is an error, never a panic.
        for cut in 0..bytes.len().min(400) {
            assert!(CellEdits::decode(&bytes[..cut]).is_err());
        }
        assert!(CellEdits::decode(&[9, 0, 0]).is_err());
    }
}
