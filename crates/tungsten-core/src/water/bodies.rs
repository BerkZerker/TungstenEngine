//! The water of a world beyond what the generator gives: ponds made by
//! pouring, and where the sea has come in through dug openings.
//!
//! Part of the bit-exact contract: integers only, and every collection is
//! walked in a fixed order.
//!
//! A pond is a seed cell and a volume. Where its water lies is a function of
//! those two and the terrain ([`fill`]), worked out again whenever the ground
//! or the water around it changes; the cells kept with it are that answer,
//! for drawing and for telling whose water a cell is under. So a pond lowers
//! when a channel is dug from it and rises when something is put in it,
//! with nothing tracked per cell and its volume untouched.
//!
//! All of this happens on the finest lattice, in loaded terrain near the
//! player. What cannot be worked out because the terrain is not loaded waits.

use std::collections::{BTreeMap, VecDeque};

use glam::IVec3;

use crate::chunk::Volume;
use crate::codec::{get_signed, get_u8, get_varint, put_signed, put_varint};
use crate::coords::{CHUNK_SIZE, NodeKey, chunk_of};
use crate::edit::FP;
use crate::worldgen::{Generator, SEA_LEVEL};

use super::cells::Cells;
use super::flood::{Filled, Grid, Run, bottom, fill, run_down, spread, top_layer};
use super::{BodyId, NodeWater, Surface};

/// Level of the sea, fixed-point base cells. The sea's surface is anchored
/// at lattice height zero, which is where `SEA_LEVEL` is.
pub const SEA: i64 = 0;
/// A pond of less than this soaks away: an eighth of a cubic metre at the
/// default cell size.
pub const MIN_POND: i64 = 8 * FP;
/// How many hollows in a row an overflow is followed through.
const MAX_HOPS: u8 = 48;
/// Cells the sea floods per tick through an opening.
const SPREAD_BUDGET: usize = 30_000;
/// Ponds settled per tick.
const SETTLE_PER_TICK: usize = 4;
/// Lots of water waiting for terrain that are looked at per tick.
const PARKED_PER_TICK: usize = 4;
/// Pending cells of the sea's advance looked at per tick.
const FRONTIER_PER_TICK: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pond {
    /// A cell of the pond, the lowest when it was made.
    pub seed: IVec3,
    /// Units of `1 / FP` of a base cell cubed.
    pub volume: i64,
    /// Surface height, fixed-point base cells.
    pub level: i64,
    /// The cells under its water as of when it last settled.
    pub cells: Cells,
    /// Something changed near it since.
    unsettled: bool,
}

/// Water on its way downhill: an overflow, or what the player poured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pour {
    at: IVec3,
    volume: i64,
    hops: u8,
}

#[derive(Default, Debug)]
pub struct Water {
    ponds: BTreeMap<u32, Pond>,
    next_pond: u32,
    /// Cells the sea has come into through openings dug in its shore.
    sea: Cells,
    /// Where the sea's advance carries on from.
    frontier: VecDeque<IVec3>,
    pours: VecDeque<Pour>,
    /// Water that ran out of the loaded terrain. It carries on from where it
    /// was when that terrain is loaded again.
    parked: Vec<Pour>,
    /// Volume that ran into the sea.
    drained: i64,
    /// Volume that is gone: puddles too small to keep, ponds buried whole,
    /// and overflows followed through too many hollows.
    lost: i64,
    /// Boxes (inclusive, with the surface layers as their height) in which
    /// surfaces changed and have to be drawn again.
    changed: Vec<(IVec3, IVec3)>,
    unsaved: bool,
}

/// A summary for display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterStats {
    pub ponds: usize,
    /// Volume held by ponds, on its way downhill (or waiting for terrain
    /// to load so it can go on), taken by the sea, and lost.
    pub held: i64,
    pub moving: i64,
    pub drained: i64,
    pub lost: i64,
    /// Cells the sea has come into, and how many more it is waiting to look at.
    pub sea_cells: usize,
    pub sea_pending: usize,
}

fn overlaps(a: (IVec3, IVec3), b: (IVec3, IVec3)) -> bool {
    a.0.cmple(b.1).all() && a.1.cmpge(b.0).all()
}

const MAGIC: &[u8; 4] = b"TNGW";
const VERSION: u8 = 1;

fn put_point(out: &mut Vec<u8>, p: IVec3) {
    p.to_array().iter().for_each(|&v| put_signed(out, v as i64));
}

fn get_point(bytes: &[u8], at: &mut usize) -> Result<IVec3, String> {
    let mut p = [0i32; 3];
    for v in &mut p {
        *v = i32::try_from(get_signed(bytes, at)?).map_err(|_| "bad position")?;
    }
    Ok(IVec3::from_array(p))
}

impl Water {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ponds(&self) -> impl Iterator<Item = (u32, &Pond)> {
        self.ponds.iter().map(|(&id, p)| (id, p))
    }

    pub fn stats(&self) -> WaterStats {
        WaterStats {
            ponds: self.ponds.len(),
            held: self.ponds.values().map(|p| p.volume).sum(),
            moving: self.pours.iter().chain(&self.parked).map(|p| p.volume).sum(),
            drained: self.drained,
            lost: self.lost,
            sea_cells: self.sea.len(),
            sea_pending: self.frontier.len(),
        }
    }

    /// Surface level of every body whose level can change.
    pub fn levels(&self) -> impl Iterator<Item = (BodyId, i64)> + '_ {
        self.ponds.iter().map(|(&id, p)| (BodyId::Pond(id), p.level))
    }

    /// True if anything changed since the last call; for knowing when to save.
    pub fn take_unsaved(&mut self) -> bool {
        std::mem::take(&mut self.unsaved)
    }

    /// Boxes in which water surfaces changed since the last call. Height is
    /// the lattice layers the surfaces were and are in.
    pub fn take_changed(&mut self) -> Vec<(IVec3, IVec3)> {
        std::mem::take(&mut self.changed)
    }

    /// The body whose water the cell at `p` is under, whether or not the
    /// cell is solid.
    pub fn holder(&self, generator: &Generator, p: IVec3) -> Option<BodyId> {
        if self.sea_holds(generator, p) {
            return Some(BodyId::Ocean);
        }
        self.ponds.iter().find(|(_, pond)| pond.cells.contains(p)).map(|(&id, _)| BodyId::Pond(id))
    }

    fn sea_holds(&self, generator: &Generator, p: IVec3) -> bool {
        if bottom(p.y) >= SEA {
            return false;
        }
        let chunk = chunk_of(p);
        let o = chunk * CHUNK_SIZE;
        self.sea.contains(p) || generator.heights(0, chunk.x, chunk.z).at(p.x - o.x, p.z - o.z) < SEA_LEVEL
    }

    /// Let `volume` of water go at `at`. It runs downhill from there on the
    /// next tick.
    pub fn pour(&mut self, at: IVec3, volume: i64) {
        if volume > 0 {
            self.pours.push_back(Pour { at, volume, hops: 0 });
            self.unsaved = true;
        }
    }

    fn note(&mut self, cells: Option<(IVec3, IVec3)>, levels: [i64; 2]) {
        let Some((lo, hi)) = cells else { return };
        let (a, b) = (top_layer(levels[0]), top_layer(levels[1]));
        // A wet column is drawn by the cells on every side of it.
        self.changed.push((IVec3::new(lo.x - 1, a.min(b), lo.z - 1), IVec3::new(hi.x + 1, a.max(b), hi.z + 1)));
        self.unsaved = true;
    }

    /// Mark in `grid` the cells that are under water already, other than
    /// pond `skip`'s.
    fn mark(&self, grid: &mut Grid, generator: &Generator, skip: Option<u32>) {
        let (min, max) = (grid.origin, grid.origin + grid.dims - 1);
        if bottom(min.y) < SEA {
            let top = top_layer(SEA).min(max.y);
            // The sea as generated: open ground below it, and whatever is hollow under that.
            let (c0, c1) = (chunk_of(min), chunk_of(max));
            for cz in c0.z..=c1.z {
                for cx in c0.x..=c1.x {
                    let heights = generator.heights(0, cx, cz);
                    if heights.min() >= SEA_LEVEL {
                        continue;
                    }
                    let (ox, oz) = (cx * CHUNK_SIZE, cz * CHUNK_SIZE);
                    for z in min.z.max(oz)..=max.z.min(oz + CHUNK_SIZE - 1) {
                        for x in min.x.max(ox)..=max.x.min(ox + CHUNK_SIZE - 1) {
                            if heights.at(x - ox, z - oz) < SEA_LEVEL {
                                (min.y..=top).for_each(|y| grid.hold(IVec3::new(x, y, z), BodyId::Ocean));
                            }
                        }
                    }
                }
            }
            for p in self.sea.iter_box(min, IVec3::new(max.x, top, max.z)) {
                grid.hold(p, BodyId::Ocean);
            }
        }
        for (&id, pond) in &self.ponds {
            if Some(id) != skip {
                for p in pond.cells.iter_box(min, max) {
                    grid.hold(p, BodyId::Pond(id));
                }
            }
        }
    }

    /// Give a pond the cells, level and volume it settled to; one too small
    /// to keep soaks away.
    fn place(&mut self, id: u32, cells: Vec<IVec3>, level: i64, volume: i64) {
        let pond = self.ponds.get_mut(&id).unwrap();
        let cells: Cells = cells.into_iter().collect();
        let old = (pond.cells.bounds(), pond.level);
        let same = cells == pond.cells && level == pond.level;
        pond.unsettled = false;
        self.unsaved |= !same || volume != pond.volume;
        (pond.cells, pond.level, pond.volume) = (cells, level, volume);
        let new = pond.cells.bounds();
        if volume < MIN_POND {
            self.lost += volume;
            self.ponds.remove(&id);
            self.note(old.0, [old.1; 2]);
        } else if !same {
            self.note(old.0, [old.1, level]);
            self.note(new, [old.1, level]);
        }
    }

    fn remove(&mut self, id: u32) -> Option<Pond> {
        let pond = self.ponds.remove(&id)?;
        self.note(pond.cells.bounds(), [pond.level; 2]);
        Some(pond)
    }

    /// Work out again where a pond's water lies. False if the terrain around
    /// it is not loaded, in which case it stays as it was.
    fn settle(&mut self, id: u32, volume: &Volume, generator: &Generator, hops: u8) -> bool {
        loop {
            let Some(pond) = self.ponds.get(&id) else { return true };
            let Some(mut grid) = Grid::new(volume, Grid::around(pond.seed), Grid::DIMS) else { return false };
            if !grid.is_open(pond.seed) {
                // The seed was filled in: carry on from any cell that is left.
                let (left, held) = (pond.cells.iter().find(|&p| grid.is_open(p)), pond.volume);
                match left {
                    Some(p) => self.ponds.get_mut(&id).unwrap().seed = p,
                    None => {
                        // Buried whole.
                        self.lost += held;
                        self.remove(id);
                    }
                }
                continue;
            }
            self.mark(&mut grid, generator, Some(id));
            let (seed, held) = (pond.seed, pond.volume);
            let (cells, how) = match grid.holder(seed) {
                // Other water has come over the seed.
                Some(with) => (Vec::new(), Filled::Met { level: bottom(seed.y), kept: 0, with }),
                None => fill(&grid, seed, held),
            };
            match how {
                Filled::Settled { level } => self.place(id, cells, level, held),
                Filled::Spilled { level, kept, at } => {
                    self.place(id, cells, level, kept);
                    match at {
                        Some(at) if hops < MAX_HOPS => {
                            self.pours.push_back(Pour { at, volume: held - kept, hops: hops + 1 })
                        }
                        _ => self.lost += held - kept,
                    }
                }
                Filled::Met { with: BodyId::Pond(other), .. } => {
                    // Two ponds in one hollow are one pond.
                    let other = self.remove(other).unwrap();
                    self.ponds.get_mut(&id).unwrap().volume += other.volume;
                    continue;
                }
                // Above the sea: what does not fit below the rim runs down into it.
                Filled::Met { level, kept, with: BodyId::Ocean } if level >= SEA => {
                    self.drained += held - kept;
                    self.place(id, cells, level, kept);
                }
                Filled::Met { with: BodyId::Ocean, .. } => {
                    // Below sea level and open to it: the sea comes in.
                    self.drained += held;
                    self.remove(id);
                    self.frontier.push_back(seed);
                }
            }
            return true;
        }
    }

    /// Follow poured water downhill through one box: to where it comes to
    /// rest, or to the edge, from where it carries on next tick.
    fn run(&mut self, pour: Pour, volume: &Volume, generator: &Generator) {
        let Some(mut grid) = Grid::new(volume, Grid::around(pour.at), Grid::DIMS) else {
            self.parked.push(pour);
            return;
        };
        self.mark(&mut grid, generator, None);
        match run_down(&grid, pour.at) {
            Run::Out(at) => self.pours.push_front(Pour { at, ..pour }),
            Run::Blocked => self.lost += pour.volume,
            // The sea takes it without rising.
            Run::Into(BodyId::Ocean) => self.drained += pour.volume,
            Run::Into(BodyId::Pond(id)) => {
                self.ponds.get_mut(&id).unwrap().volume += pour.volume;
                self.settle(id, volume, generator, pour.hops);
            }
            Run::Rest(seed) => {
                let id = self.next_pond;
                self.next_pond += 1;
                let pond = Pond {
                    seed,
                    volume: pour.volume,
                    level: bottom(seed.y),
                    cells: Cells::default(),
                    unsettled: false,
                };
                self.ponds.insert(id, pond);
                self.settle(id, volume, generator, pour.hops);
            }
        }
    }

    /// Let the sea into one more box of the space it has an opening to.
    fn advance_sea(&mut self, volume: &Volume, generator: &Generator) {
        for _ in 0..self.frontier.len().min(FRONTIER_PER_TICK) {
            let p = self.frontier.pop_front().unwrap();
            match volume.get(p) {
                // Not loaded: it waits.
                None => {
                    self.frontier.push_back(p);
                    continue;
                }
                Some(s) if s.solid() => continue,
                Some(_) if self.sea_holds(generator, p) => continue,
                Some(_) => {}
            }
            let Some(mut grid) = Grid::new(volume, Grid::around(p), Grid::DIMS) else {
                self.frontier.push_back(p);
                continue;
            };
            self.mark(&mut grid, generator, None);
            let mut seeds = vec![p];
            self.frontier.retain(|&q| {
                let inside = grid.contains(q);
                if inside {
                    seeds.push(q);
                }
                !inside
            });
            let found = spread(&grid, &seeds, SEA, BodyId::Ocean, SPREAD_BUDGET);
            let mut bounds: Option<(IVec3, IVec3)> = None;
            for &c in &found.cells {
                self.sea.insert(c);
                let (lo, hi) = bounds.get_or_insert((c, c));
                (*lo, *hi) = (lo.min(c), hi.max(c));
            }
            self.note(bounds, [SEA; 2]);
            self.frontier.extend(found.open);
            for (with, at) in found.met {
                // A pond the sea has reached is part of the sea.
                if let BodyId::Pond(id) = with
                    && let Some(pond) = self.remove(id)
                {
                    self.drained += pond.volume;
                    self.frontier.push_back(at);
                }
            }
            return;
        }
    }

    /// The terrain in `[min, max]` changed. Ponds near it settle again on the
    /// next tick; where it opened ground beside the sea, the sea comes in.
    pub fn edited(&mut self, volume: &Volume, generator: &Generator, min: IVec3, max: IVec3) {
        let (min, max) = (min - 1, max + 1);
        for pond in self.ponds.values_mut() {
            pond.unsettled |= pond.cells.bounds().is_some_and(|b| overlaps(b, (min, max)));
        }
        if bottom(min.y) >= SEA {
            return;
        }
        let Some(mut grid) = Grid::new(volume, min, max - min + 1) else { return };
        self.mark(&mut grid, generator, None);
        let buried: Vec<IVec3> = self.sea.iter_box(min, max).filter(|&p| !grid.is_open(p)).collect();
        let mut bounds: Option<(IVec3, IVec3)> = None;
        for p in buried {
            self.sea.remove(p);
            let (lo, hi) = bounds.get_or_insert((p, p));
            (*lo, *hi) = (lo.min(p), hi.max(p));
        }
        self.note(bounds, [SEA; 2]);
        let beside_sea = |p: IVec3| {
            [IVec3::NEG_Y, IVec3::NEG_X, IVec3::X, IVec3::NEG_Z, IVec3::Z, IVec3::Y]
                .iter()
                .any(|&d| grid.holder(p + d) == Some(BodyId::Ocean))
        };
        for y in min.y..=max.y.min(top_layer(SEA)) {
            for z in min.z..=max.z {
                for x in min.x..=max.x {
                    let p = IVec3::new(x, y, z);
                    if grid.is_free(p) && beside_sea(p) {
                        self.frontier.push_back(p);
                    }
                }
            }
        }
    }

    /// One tick: settle ponds whose surroundings changed, follow one lot of
    /// water downhill, and let the sea a little further in.
    pub fn step(&mut self, volume: &Volume, generator: &Generator) {
        let unsettled: Vec<u32> =
            self.ponds.iter().filter(|(_, p)| p.unsettled).map(|(&id, _)| id).take(SETTLE_PER_TICK).collect();
        for id in unsettled {
            self.settle(id, volume, generator, 0);
        }
        // Water that was waiting for terrain which is there now.
        let loaded = |p: &Pour| volume.covers(Grid::around(p.at), Grid::DIMS);
        if let Some(i) = self.parked.iter().take(PARKED_PER_TICK).position(loaded) {
            self.pours.push_back(self.parked.remove(i));
        } else if self.parked.len() > 1 {
            let n = PARKED_PER_TICK.min(self.parked.len());
            self.parked.rotate_left(n);
        }
        if let Some(pour) = self.pours.pop_front() {
            self.run(pour, volume, generator);
            self.unsaved = true;
        }
        self.advance_sea(volume, generator);
    }

    /// True if a tick would do nothing more with the terrain that is loaded.
    pub fn at_rest(&self, volume: &Volume) -> bool {
        self.pours.is_empty()
            && self.parked.iter().all(|p| !volume.covers(Grid::around(p.at), Grid::DIMS))
            && self.ponds.values().all(|p| !p.unsettled || !volume.covers(Grid::around(p.seed), Grid::DIMS))
            && self.frontier.iter().all(|&p| volume.get(p).is_none())
    }

    /// The surfaces of ponds, and of the sea where it has come in, that pass
    /// through a node.
    pub fn node_water(&self, key: NodeKey) -> NodeWater {
        let mut out = NodeWater::default();
        let (o, size) = (key.origin(), key.size());
        let mut add = |body: BodyId, level: i64, cells: &Cells| {
            let layer = top_layer(level);
            let Some((lo, hi)) = cells.bounds() else { return };
            // A node draws a wet column on its far edge too.
            let outside = layer < o.y || layer >= o.y + size || hi.x < o.x || lo.x > o.x + size;
            if outside || hi.z < o.z || lo.z > o.z + size {
                return;
            }
            let near = |c: i32, o: i32| c * CHUNK_SIZE + CHUNK_SIZE > o && c * CHUNK_SIZE <= o + size;
            let tiles: Vec<_> = cells.layer(layer).filter(|((cx, cz), _)| near(*cx, o.x) && near(*cz, o.z)).collect();
            if !tiles.is_empty() {
                out.surfaces.push(Surface { body, level, tiles });
            }
        };
        add(BodyId::Ocean, SEA, &self.sea);
        for (&id, pond) in &self.ponds {
            add(BodyId::Pond(id), pond.level, &pond.cells);
        }
        out
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        put_varint(&mut out, self.next_pond as u64);
        put_signed(&mut out, self.drained);
        put_signed(&mut out, self.lost);
        put_varint(&mut out, self.ponds.len() as u64);
        for (&id, pond) in &self.ponds {
            put_varint(&mut out, id as u64);
            put_point(&mut out, pond.seed);
            put_signed(&mut out, pond.volume);
            put_signed(&mut out, pond.level);
            out.push(pond.unsettled as u8);
            pond.cells.encode(&mut out);
        }
        self.sea.encode(&mut out);
        put_varint(&mut out, self.frontier.len() as u64);
        self.frontier.iter().for_each(|&p| put_point(&mut out, p));
        // Water on its way and water waiting are the same thing to a new session.
        put_varint(&mut out, (self.pours.len() + self.parked.len()) as u64);
        for pour in self.pours.iter().chain(&self.parked) {
            put_point(&mut out, pour.at);
            put_signed(&mut out, pour.volume);
            out.push(pour.hops);
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.get(..4) != Some(MAGIC) {
            return Err("not a water file".into());
        }
        let mut at = 4;
        let version = get_u8(bytes, &mut at)?;
        if version != VERSION {
            return Err(format!("unsupported water version {version}"));
        }
        let at = &mut at;
        let mut water = Water {
            next_pond: u32::try_from(get_varint(bytes, at)?).map_err(|_| "bad pond count")?,
            drained: get_signed(bytes, at)?,
            lost: get_signed(bytes, at)?,
            ..Self::default()
        };
        for _ in 0..get_varint(bytes, at)? {
            let id = u32::try_from(get_varint(bytes, at)?).map_err(|_| "bad pond number")?;
            let pond = Pond {
                seed: get_point(bytes, at)?,
                volume: get_signed(bytes, at)?,
                level: get_signed(bytes, at)?,
                unsettled: get_u8(bytes, at)? != 0,
                cells: Cells::decode(bytes, at)?,
            };
            water.ponds.insert(id, pond);
        }
        water.sea = Cells::decode(bytes, at)?;
        for _ in 0..get_varint(bytes, at)? {
            water.frontier.push_back(get_point(bytes, at)?);
        }
        for _ in 0..get_varint(bytes, at)? {
            let pour = Pour { at: get_point(bytes, at)?, volume: get_signed(bytes, at)?, hops: get_u8(bytes, at)? };
            water.pours.push_back(pour);
        }
        if *at != bytes.len() {
            return Err("trailing bytes".into());
        }
        Ok(water)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::WorldParams;
    use crate::edit::{EditRules, Op, edit_volume};
    use crate::field::LocalField;
    use crate::material::Material;

    struct World {
        generator: Generator,
        rules: EditRules,
        volume: Volume,
        water: Water,
    }

    impl World {
        /// The finest level for two chunks each way around the ground at a column.
        fn around(x: i32, z: i32) -> (Self, IVec3) {
            let params = WorldParams::default();
            let generator = Generator::new(params);
            let cell = params.cell_size as f64;
            let ground = (generator.surface_height(x as f64 * cell, z as f64 * cell) as f64 / cell).floor() as i32;
            let here = IVec3::new(x, ground, z);
            let mut volume = Volume::new();
            let c = chunk_of(here);
            for cy in -2..=2 {
                for cz in -3..=3 {
                    for cx in -3..=3 {
                        let key = NodeKey::new(0, c + IVec3::new(cx, cy, cz));
                        let mut field = LocalField::block(key);
                        generator.fill_block(key, &mut field);
                        volume.insert(key.pos, field.interior());
                    }
                }
            }
            (Self { generator, rules: EditRules::new(&params), volume, water: Water::new() }, here)
        }

        fn edit(&mut self, op: Op) {
            let outcome = edit_volume(&mut self.volume, &self.rules, &op);
            assert!(outcome.covered);
            if let Some((min, max)) = outcome.changed {
                self.water.edited(&self.volume, &self.generator, min, max);
            }
        }

        fn dig(&mut self, at: IVec3, radius: i64) {
            let center = [at.x as i64 * FP, at.y as i64 * FP, at.z as i64 * FP];
            self.edit(Op::Break { center, radius: radius * FP, nearest: false });
        }

        fn put(&mut self, at: IVec3, radius: i64) {
            let center = [at.x as i64 * FP, at.y as i64 * FP, at.z as i64 * FP];
            self.edit(Op::Place { center, radius: radius * FP, material: Material::SAND, exclude: None });
        }

        /// Tick until the water stops moving.
        fn rest(&mut self) {
            for _ in 0..400 {
                self.water.step(&self.volume, &self.generator);
                if self.water.at_rest(&self.volume) {
                    return;
                }
            }
            panic!("water did not come to rest: {:?}", self.water.stats());
        }

        fn open(&self, p: IVec3) -> bool {
            !self.volume.get(p).unwrap().solid()
        }

        /// Everything poured is in a pond, on its way, or accounted lost.
        fn total(&self) -> i64 {
            let s = self.water.stats();
            s.held + s.moving + s.drained + s.lost
        }
    }

    /// Check a pond against the terrain: its cells are open, joined up, below
    /// its level, and hold its volume.
    fn check(world: &World, pond: &Pond) {
        let top = top_layer(pond.level);
        let mut holds = 0;
        for p in pond.cells.iter() {
            assert!(world.open(p), "pond cell {p} is solid");
            assert!(p.y <= top, "pond cell {p} above its level");
            holds += (pond.level - bottom(p.y)).min(FP);
            // Nothing open and dry beside or below a wet cell, short of the surface layer.
            for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                assert!(!world.open(q) || pond.cells.contains(q), "dry cell {q} beside wet {p}");
            }
        }
        let surface = pond.cells.iter().filter(|p| p.y == top).count() as i64;
        // The level is rounded up to a whole unit.
        assert!(holds >= pond.volume && holds - pond.volume < surface, "{holds} for {}", pond.volume);
    }

    #[test]
    fn a_pond_keeps_its_volume_through_digging_and_filling() {
        let (mut world, here) = World::around(0, 0);
        // A bowl two metres across in the ground, and water let go above it.
        let pit = here - IVec3::Y * 3;
        world.dig(pit, 8);
        assert!(world.water.stats() == WaterStats::default() && world.water.take_changed().is_empty());
        let poured = 300 * FP;
        world.water.pour(here + IVec3::Y * 6, poured);
        assert_eq!(world.water.stats().moving, poured);
        world.rest();
        let ponds: Vec<(u32, Pond)> = world.water.ponds().map(|(id, p)| (id, p.clone())).collect();
        let [(id, pond)] = ponds.as_slice() else { panic!("{} ponds", ponds.len()) };
        check(&world, pond);
        assert_eq!((pond.volume, world.total()), (poured, poured));
        assert!(pond.cells.contains(pond.seed) && (pond.seed - pit).abs().max_element() <= 9, "seed {}", pond.seed);
        assert!(pond.level < bottom(here.y - 1), "the pond is not below the rim");
        assert_eq!(world.water.holder(&world.generator, pond.seed), Some(BodyId::Pond(*id)));
        let drawn = world.water.take_changed();
        assert!(!drawn.is_empty());
        let (lo, hi) = pond.cells.bounds().unwrap();
        assert!(drawn.iter().any(|(a, b)| a.x < lo.x && b.x > hi.x && a.y <= top_layer(pond.level)));

        // A second bowl cut into the side of the first: the same water lies lower.
        world.dig(pit + IVec3::new(9, 0, 0), 8);
        world.rest();
        let lower = world.water.ponds().next().unwrap().1.clone();
        check(&world, &lower);
        assert_eq!((lower.volume, world.water.ponds().count()), (poured, 1));
        assert!(lower.level < pond.level - FP / 4, "{} then {}", pond.level, lower.level);
        assert!(lower.cells.len() > pond.cells.len());
        assert!(!world.water.take_changed().is_empty());

        // Sand dropped into it: the water is pushed up.
        world.put(pit + IVec3::new(0, -4, 0), 5);
        world.rest();
        let higher = world.water.ponds().next().unwrap().1.clone();
        check(&world, &higher);
        assert_eq!(higher.volume, poured);
        assert!(higher.level > lower.level + FP / 8, "{} then {}", lower.level, higher.level);

        // It all survives a save.
        let bytes = world.water.encode();
        let back = Water::decode(&bytes).unwrap();
        assert!(back.ponds == world.water.ponds && back.next_pond == world.water.next_pond);
        assert_eq!(back.encode(), bytes);
        assert!(Water::decode(&bytes[..bytes.len() - 3]).is_err());

        // Nothing moves when nothing changes.
        assert!(!world.water.take_changed().is_empty() && world.water.take_unsaved());
        world.water.step(&world.volume, &world.generator);
        assert!(world.water.take_changed().is_empty() && !world.water.take_unsaved());
    }

    #[test]
    fn more_than_a_hollow_holds_runs_on_downhill() {
        let (mut world, here) = World::around(0, 0);
        let pit = here - IVec3::Y * 2;
        world.dig(pit, 6);
        let poured = 4000 * FP;
        world.water.pour(here + IVec3::Y * 4, poured);
        world.rest();
        assert_eq!(world.total(), poured);
        // The bowl is full to its rim, and the rest went elsewhere.
        let (_, first) = world.water.ponds().find(|(_, p)| (p.seed - pit).abs().max_element() <= 6).unwrap();
        check(&world, first);
        assert!(first.volume < poured / 4 && first.volume > 100 * FP, "{}", first.volume);
        for (_, pond) in world.water.ponds() {
            check(&world, pond);
            assert!(pond.volume >= MIN_POND);
        }
        // More water changes nothing in a full bowl.
        let before = first.clone();
        world.water.pour(here + IVec3::Y * 4, 500 * FP);
        world.rest();
        let (_, after) = world.water.ponds().find(|(_, p)| p.seed == before.seed).unwrap();
        assert_eq!((after.volume, after.level), (before.volume, before.level));
        assert_eq!(world.total(), poured + 500 * FP);
    }

    #[test]
    fn two_ponds_that_meet_become_one() {
        let (mut world, here) = World::around(0, 0);
        let (a, b) = (here - IVec3::Y * 5, here + IVec3::new(18, -5, 0));
        let (va, vb) = (100 * FP, 80 * FP);
        world.dig(a, 7);
        world.dig(b, 7);
        world.water.pour(a + IVec3::Y * 8, va);
        world.water.pour(b + IVec3::Y * 8, vb);
        world.rest();
        assert_eq!((world.water.ponds().count(), world.total()), (2, va + vb));
        // A trench between them, below both surfaces.
        for x in (a.x..=b.x).step_by(3) {
            world.dig(IVec3::new(x, a.y - 2, a.z), 4);
        }
        world.rest();
        let ponds: Vec<Pond> = world.water.ponds().map(|(_, p)| p.clone()).collect();
        let [pond] = ponds.as_slice() else { panic!("{} ponds", ponds.len()) };
        check(&world, pond);
        assert_eq!((pond.volume, world.total()), (va + vb, va + vb));
    }

    /// The first column east of the origin where the ground is below the sea.
    fn shore() -> i32 {
        let g = Generator::new(WorldParams::default());
        (0..400).find(|&x| g.surface_height(x as f64 * 0.25, 0.0) < SEA_LEVEL).unwrap()
    }

    #[test]
    fn the_sea_comes_in_through_a_dug_channel() {
        let wet = shore();
        let (mut world, _) = World::around(wet - 12, 0);
        let sea_floor = IVec3::new(wet + 6, -1, 0);
        assert_eq!(world.water.holder(&world.generator, sea_floor), Some(BodyId::Ocean));
        assert_eq!(world.water.holder(&world.generator, IVec3::new(wet + 6, 1, 0)), None);
        assert_eq!(world.water.holder(&world.generator, IVec3::new(wet - 4, -1, 0)), None);

        // A pit behind the beach, reaching below sea level but short of the water.
        let pit = IVec3::new(wet - 14, -3, 0);
        world.dig(pit, 6);
        world.rest();
        assert!(world.open(pit) && world.water.holder(&world.generator, pit).is_none());
        assert_eq!(world.water.stats(), WaterStats::default());

        // A channel from it to the sea.
        for x in (wet - 10..=wet + 2).step_by(3) {
            world.dig(IVec3::new(x, -2, 0), 4);
        }
        world.rest();
        let stats = world.water.stats();
        assert!(stats.sea_cells > 500 && stats.sea_pending == 0, "{stats:?}");
        assert_eq!(world.water.holder(&world.generator, pit), Some(BodyId::Ocean));
        for p in world.water.sea.iter() {
            assert!(world.open(p) && bottom(p.y) < SEA, "sea cell {p}");
            for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                assert!(!world.open(q) || world.water.sea_holds(&world.generator, q), "dry {q} beside sea {p}");
            }
        }
        // The surface over the pit is handed to the node that draws it.
        let key = NodeKey::new(0, chunk_of(IVec3::new(pit.x, 0, pit.z)));
        let surfaces = world.water.node_water(key).surfaces;
        assert!(matches!(surfaces.as_slice(), [Surface { body: BodyId::Ocean, level: SEA, .. }]));
        let patches = crate::water::node_patches(&world.generator, key, &world.water.node_water(key));
        assert!(patches.len() == 1 && patches[0].body == BodyId::Ocean);
        let local = pit - key.origin();
        let covers = |p: &crate::water::WaterPatch| {
            p.indices.chunks_exact(6).any(|q| {
                let (a, c) = (p.vertices[q[0] as usize], p.vertices[q[2] as usize]);
                (a[0] as i32..c[0] as i32).contains(&local.x) && (a[1] as i32..c[1] as i32).contains(&local.z)
            })
        };
        assert!(covers(&patches[0]), "no sea drawn over the pit");
        assert!(!crate::water::node_patches(&world.generator, key, &NodeWater::default()).iter().any(covers));

        // Filled in again, the cells are given back.
        world.put(pit, 9);
        world.rest();
        assert!(world.water.stats().sea_cells < stats.sea_cells);
        assert!(world.water.sea.iter().all(|p| world.open(p)));
        let bytes = world.water.encode();
        assert!(Water::decode(&bytes).unwrap().sea == world.water.sea);
    }

    #[test]
    fn a_pond_dug_through_to_the_sea_joins_it_or_drains_into_it() {
        let wet = shore();
        let (mut world, _) = World::around(wet - 12, 0);
        // Below sea level: once joined, it is sea.
        let low = IVec3::new(wet - 14, -4, 0);
        world.dig(low, 6);
        world.water.pour(low + IVec3::Y * 2, 60 * FP);
        world.rest();
        assert_eq!(world.water.ponds().count(), 1);
        assert!(world.water.ponds().next().unwrap().1.level < SEA);
        for x in (wet - 10..=wet + 2).step_by(3) {
            world.dig(IVec3::new(x, -2, 0), 4);
        }
        world.rest();
        assert_eq!(world.water.ponds().count(), 0);
        assert_eq!(world.water.holder(&world.generator, low), Some(BodyId::Ocean));
    }
}
