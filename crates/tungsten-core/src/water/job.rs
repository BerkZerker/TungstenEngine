//! The part of water that takes time: searching a box of terrain for where
//! water lies or goes. A job is everything such a search needs, copied out
//! of the world so that it can run on a worker, and its outcome is a pure
//! function of the job: the same on every machine, however long it took.
//!
//! Part of the bit-exact contract: integers only, fixed orders.

use std::sync::Arc;

use glam::IVec3;

use crate::chunk::{Chunk, Dense, Sample, Volume};
use crate::coords::{CHUNK_SIZE, ChunkPos, LEVELS, NodeKey, chunk_of, chunk_origin};
use crate::edit::{self, EditRules, Op};
use crate::field::LocalField;
use crate::hash::IntMap;
use crate::log::{self, CellSet, EditLog};
use crate::worldgen::{Generator, SEA_GRID, SEA_LEVEL};

use super::BodyId;
use super::cells::Cells;
use super::flood::{Filled, Grid, Part, Run, Spread, fill, parts, run_down, spread};

/// The lattice cell of level of detail `lod` that holds a point of the finest lattice.
#[inline]
pub fn cell_of(base: IVec3, lod: u8) -> IVec3 {
    (base + ((1 << lod) >> 1)) >> lod as i32
}

/// The point of the finest lattice at the middle of a cell of `lod`.
#[inline]
pub fn base_of(cell: IVec3, lod: u8) -> IVec3 {
    cell << lod as i32
}

/// A box of terrain a job is to look at: whole chunks of one level of detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub lod: u8,
    pub origin: ChunkPos,
    pub size: IVec3,
    /// Only terrain that is loaded will do: the job is given up otherwise.
    pub loaded: bool,
}

impl Request {
    /// The box in the finest lattice, inclusive.
    pub fn bounds(&self) -> (IVec3, IVec3) {
        let (lo, hi) = (self.origin * CHUNK_SIZE, (self.origin + self.size) * CHUNK_SIZE - 1);
        let half = (1 << self.lod) >> 1;
        (base_of(lo, self.lod) - half, base_of(hi, self.lod) + half)
    }
}

enum Piece {
    Chunk(Arc<Chunk>),
    /// To be generated and have these edits applied.
    Edits(CellSet),
    /// A chunk an earlier job made, and the edits made since.
    Patch(Arc<Chunk>, Vec<Op>),
}

/// A chunk a job had to make, for the next job to start from: making one
/// from the log costs far more than any search does.
pub type Made = (NodeKey, Arc<Chunk>);

#[derive(Default)]
struct Kept {
    /// `None` while the first search to want it is still making it.
    chunk: Option<Arc<Chunk>>,
    /// Edits that reach the chunk and are not in it yet.
    pending: Vec<Op>,
    /// How many of `pending` the search that is making it anew was given.
    sent: usize,
    /// The step it was last asked for, and which gathering that was.
    used: u64,
    gather: u64,
}

/// The chunks searches had to make, kept for the searches after them. A
/// pond that is settling is searched for over and over, and where nothing is
/// loaded its terrain would be made from the generator and the log each time.
#[derive(Default)]
pub struct KeptTerrain {
    chunks: IntMap<NodeKey, Kept>,
    /// Make everything afresh instead, as a check on the keeping.
    pub off: bool,
    steps: u64,
    /// Gatherings of terrain so far, and the one the search under way came from.
    gathers: u64,
    flight: u64,
}

/// Chunks kept before those long unused are let go: 16 MB if all are dense.
const KEPT_CHUNKS: usize = 256;
const KEPT_STEPS: u64 = 600;
/// A kept chunk owed more edits than this is made again from the log.
const KEPT_EDITS: usize = 64;

impl KeptTerrain {
    /// Once a step of the water.
    pub fn tick(&mut self) {
        self.steps += 1;
        if self.chunks.len() > KEPT_CHUNKS {
            let now = self.steps;
            self.chunks.retain(|_, kept| now - kept.used < KEPT_STEPS);
        }
    }

    /// An edit was added to the log.
    pub fn edited(&mut self, rules: &EditRules, op: &Op) {
        if self.chunks.is_empty() {
            return;
        }
        let bounds: [(IVec3, IVec3); LEVELS] = std::array::from_fn(|level| op.bounds(rules, level as u8));
        for (key, kept) in &mut self.chunks {
            let (min, max) = bounds[key.level as usize];
            let origin = chunk_origin(key.pos);
            if min.cmpgt(origin + CHUNK_SIZE - 1).any() || max.cmplt(origin).any() {
                continue;
            }
            if kept.pending.len() < KEPT_EDITS {
                kept.pending.push(op.clone());
            } else {
                // Too far behind to be worth bringing up to date: it is made
                // again from the log, and what a search under way makes of
                // it is not wanted.
                *kept = Kept { used: kept.used, ..Kept::default() };
            }
        }
    }

    /// As [`Terrain::gather`], with what is kept where nothing is loaded.
    pub fn gather(&mut self, volume: &Volume, log: &EditLog, request: &Request) -> Option<Terrain> {
        if self.off {
            return Terrain::gather(volume, log, request);
        }
        self.gathers += 1;
        let (now, gather) = (self.steps, self.gathers);
        Terrain::gather_from(volume, log, request, |key| {
            let kept = self.chunks.entry(key).or_default();
            (kept.used, kept.gather) = (now, gather);
            if kept.chunk.is_none() {
                // To be made from the log as it is now: nothing is owed.
                kept.pending.clear();
            }
            kept.sent = kept.pending.len();
            kept.chunk.clone().map(|chunk| (chunk, kept.pending.clone()))
        })
    }

    /// The terrain last gathered went to a search.
    pub fn launched(&mut self) {
        self.flight = self.gathers;
    }

    /// What that search made.
    pub fn store(&mut self, made: Vec<Made>) {
        for (key, chunk) in made {
            // Only if it is still as the search was given it: not let go
            // since, nor fallen too far behind, nor gathered again.
            if let Some(kept) = self.chunks.get_mut(&key).filter(|k| k.gather == self.flight) {
                kept.chunk = Some(chunk);
                kept.pending.drain(..kept.sent);
                kept.sent = 0;
            }
        }
    }
}

/// `chunk` with `ops` applied: what making it afresh from a log that has
/// since gained those ops would give, since an op is a function of each
/// sample's position alone.
fn patched(key: NodeKey, chunk: &Chunk, ops: &[Op], rules: &EditRules) -> Chunk {
    let mut field = LocalField::new(chunk_origin(key.pos), IVec3::splat(CHUNK_SIZE), key.level);
    match chunk {
        Chunk::Uniform(sample) => field.set_all(*sample),
        Chunk::Dense(dense) => {
            field.fill.copy_from_slice(&dense.fill);
            field.mat.copy_from_slice(&dense.mat);
        }
    }
    for op in ops {
        edit::apply(op, rules, &mut field, None);
    }
    let mut dense = Dense::new_boxed(Sample::AIR);
    dense.fill.copy_from_slice(&field.fill);
    dense.mat.copy_from_slice(&field.mat);
    let mut chunk = Chunk::Dense(dense);
    chunk.normalize();
    chunk
}

/// The terrain in a [`Request`]'s box: the chunks that are loaded as they
/// are, the rest as the edits to make them from. Either way the samples are
/// the same, so what a job finds does not depend on what happened to be
/// loaded.
pub struct Terrain {
    request: Request,
    /// In order of Y, then Z, then X.
    pieces: Vec<Piece>,
}

impl Terrain {
    /// `None` if the request is for loaded terrain and some is not.
    pub fn gather(volume: &Volume, log: &EditLog, request: &Request) -> Option<Self> {
        Self::gather_from(volume, log, request, |_| None)
    }

    /// As [`Terrain::gather`], given the chunks kept from earlier jobs:
    /// `kept` hands over one, with the edits made since it was made.
    fn gather_from(
        volume: &Volume,
        log: &EditLog,
        request: &Request,
        mut kept: impl FnMut(NodeKey) -> Option<(Arc<Chunk>, Vec<Op>)>,
    ) -> Option<Self> {
        let mut pieces = Vec::with_capacity((request.size.x * request.size.y * request.size.z) as usize);
        for y in 0..request.size.y {
            for z in 0..request.size.z {
                for x in 0..request.size.x {
                    let pos = request.origin + IVec3::new(x, y, z);
                    let loaded = if request.lod == 0 { volume.chunks.get(&pos) } else { None };
                    let key = NodeKey::new(request.lod, pos);
                    pieces.push(match loaded {
                        Some(chunk) => Piece::Chunk(chunk.clone()),
                        None if request.loaded => return None,
                        None => match kept(key) {
                            Some((chunk, ops)) if ops.is_empty() => Piece::Chunk(chunk),
                            Some((chunk, ops)) => Piece::Patch(chunk, ops),
                            None => Piece::Edits(log.gather_node(key)),
                        },
                    });
                }
            }
        }
        Some(Self { request: *request, pieces })
    }

    /// Terrain given as its chunks, in order of Y, then Z, then X.
    pub fn of_chunks(request: &Request, chunks: Vec<Arc<Chunk>>) -> Self {
        assert_eq!(chunks.len() as i32, request.size.x * request.size.y * request.size.z);
        Self { request: *request, pieces: chunks.into_iter().map(Piece::Chunk).collect() }
    }

    /// The terrain as a grid, and the chunks that had to be made for it.
    fn grid(self, generator: &Generator, rules: &EditRules) -> (Grid, Vec<Made>) {
        let Request { lod, origin, size, .. } = self.request;
        let mut made = Vec::new();
        let chunks: Vec<Arc<Chunk>> = self
            .pieces
            .into_iter()
            .enumerate()
            .map(|(n, piece)| {
                let n = n as i32;
                let key =
                    NodeKey::new(lod, origin + IVec3::new(n % size.x, n / (size.x * size.z), n / size.x % size.z));
                let chunk = match piece {
                    Piece::Chunk(chunk) => return chunk,
                    Piece::Edits(cells) => {
                        let mut block = LocalField::block(key);
                        generator.fill_block(key, &mut block);
                        log::resolve(&mut block, rules, &cells);
                        Arc::new(block.interior())
                    }
                    Piece::Patch(chunk, ops) => Arc::new(patched(key, &chunk, &ops, rules)),
                };
                made.push((key, chunk.clone()));
                chunk
            })
            .collect();
        (Grid::from_chunks(origin, size, &chunks), made)
    }
}

/// Another pond near a job's box.
pub(super) struct Other {
    pub id: u32,
    pub lod: u8,
    pub cells: Cells,
}

pub(super) enum Task {
    /// Work out where a pond's water lies.
    Settle {
        /// In the lattice of the job, as is everything below.
        seed: IVec3,
        volume: i64,
        sealed: bool,
        /// The cells it had and its level, to tell whether it has been cut in two.
        was: Cells,
        was_level: i64,
    },
    /// Find the way downhill from a flow's source.
    Route { from: IVec3 },
    /// Let the sea further into what it has an opening to.
    Sea { seeds: Vec<IVec3>, reach: u32 },
}

/// A search for a worker to run.
pub struct WaterJob {
    pub(super) task: Task,
    pub(super) terrain: Terrain,
    /// Cells the sea has come into.
    pub(super) sea: Cells,
    pub(super) others: Vec<Other>,
}

/// What a job found. Lattice positions are in the job's level of detail.
pub enum Outcome {
    Settled {
        /// The cell the search started from: the pond's seed, or a cell near it if that was buried.
        seed: IVec3,
        cells: Cells,
        how: Filled,
        /// The edge of the box had a part in it.
        edge: bool,
    },
    /// The pond is in pieces: something was built across it.
    Split {
        parts: Vec<(IVec3, i64, Cells)>,
    },
    /// There is no open cell left where the pond was.
    Buried,
    Routed {
        path: Vec<IVec3>,
        end: Run,
    },
    Sea(Spread),
}

/// Mark in `grid`, a box of level of detail `lod`, the cells that are under
/// water already: the sea as generated, the cells it has come into, and
/// the ponds in `others`.
pub(super) fn mark(grid: &mut Grid, lod: u8, generator: &Generator, sea: &Cells, others: &[Other]) {
    let (min, max) = (grid.origin, grid.origin + grid.dims - 1);
    // The cells at lattice height zero and below are the ones under sea level, at every level of detail.
    if min.y <= 0 {
        let top = max.y.min(0);
        // The sea as generated: open ground below it that is joined to a
        // sea, and whatever is hollow under that.
        let mut reached: IntMap<(i32, i32), bool> = IntMap::default();
        let (c0, c1) = (chunk_of(min), chunk_of(max));
        for cz in c0.z..=c1.z {
            for cx in c0.x..=c1.x {
                let heights = generator.heights(lod, cx, cz);
                if heights.min() >= SEA_LEVEL {
                    continue;
                }
                let (ox, oz) = (cx * CHUNK_SIZE, cz * CHUNK_SIZE);
                for z in min.z.max(oz)..=max.z.min(oz + CHUNK_SIZE - 1) {
                    for x in min.x.max(ox)..=max.x.min(ox + CHUNK_SIZE - 1) {
                        if heights.at(x - ox, z - oz) >= SEA_LEVEL {
                            continue;
                        }
                        let at = ((x << lod).div_euclid(SEA_GRID), (z << lod).div_euclid(SEA_GRID));
                        if *reached.entry(at).or_insert_with(|| generator.sea_reaches(at.0, at.1)) {
                            (min.y..=top).for_each(|y| grid.hold(IVec3::new(x, y, z), BodyId::Ocean));
                        }
                    }
                }
            }
        }
        let low = IVec3::new(max.x, top, max.z);
        if lod == 0 {
            for p in sea.iter_box(min, low) {
                grid.hold(p, BodyId::Ocean);
            }
        } else if sea
            .bounds()
            .is_some_and(|(lo, hi)| lo.cmple(base_of(low, lod)).all() && hi.cmpge(base_of(min, lod)).all())
        {
            for_box(min, low, |p| {
                if sea.contains(base_of(p, lod)) {
                    grid.hold(p, BodyId::Ocean);
                }
            });
        }
    }
    for other in others {
        let body = BodyId::Pond(other.id);
        if other.lod == lod {
            for p in other.cells.iter_box(min, max) {
                grid.hold(p, body);
            }
        } else if other.lod < lod {
            // Finer than the box: any of its cells wets the cell of the box it is in.
            let (lo, hi) = (cell_of(base_of(min, lod), other.lod) - 1, cell_of(base_of(max, lod), other.lod) + 1);
            for p in other.cells.iter_box(lo, hi) {
                grid.hold(cell_of(base_of(p, other.lod), lod), body);
            }
        } else {
            for_box(min, max, |p| {
                if other.cells.contains(cell_of(base_of(p, lod), other.lod)) {
                    grid.hold(p, body);
                }
            });
        }
    }
}

fn for_box(min: IVec3, max: IVec3, mut f: impl FnMut(IVec3)) {
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                f(IVec3::new(x, y, z));
            }
        }
    }
}

impl WaterJob {
    /// The box the job looks at, in the finest lattice.
    pub fn bounds(&self) -> (IVec3, IVec3) {
        self.terrain.request.bounds()
    }

    pub fn run(self, generator: &Generator, rules: &EditRules) -> Outcome {
        self.run_keeping(generator, rules).0
    }

    /// Run the search, and hand back the chunks it had to make as well.
    pub fn run_keeping(self, generator: &Generator, rules: &EditRules) -> (Outcome, Vec<Made>) {
        let lod = self.terrain.request.lod;
        let (mut grid, made) = self.terrain.grid(generator, rules);
        mark(&mut grid, lod, generator, &self.sea, &self.others);
        (Self::search(self.task, &grid), made)
    }

    fn search(task: Task, grid: &Grid) -> Outcome {
        match task {
            Task::Settle { mut seed, volume, sealed, was, was_level } => {
                // A pond that reaches out of the box cannot be told to be in pieces.
                let (lo, hi) = (grid.origin, grid.origin + grid.dims - 1);
                let whole = was.bounds().is_some_and(|(a, b)| a.cmpge(lo).all() && b.cmple(hi).all());
                let pieces = if whole { parts(grid, was.iter(), was_level) } else { Vec::new() };
                if pieces.len() > 1 {
                    let pieces = pieces
                        .into_iter()
                        .map(|Part { seed, content, cells }| (seed, content, cells.into_iter().collect()));
                    return Outcome::Split { parts: pieces.collect() };
                }
                if !grid.is_open(seed) {
                    // The seed was filled in: carry on from what is left of
                    // the pond, or from the first open cell over it.
                    let over = (seed.y..grid.origin.y + grid.dims.y).map(|y| IVec3::new(seed.x, y, seed.z));
                    match pieces.first().map(|p| p.seed).or_else(|| over.into_iter().find(|&p| grid.is_open(p))) {
                        Some(p) => seed = p,
                        None => return Outcome::Buried,
                    }
                }
                let (cells, how, edge) = match grid.holder(seed) {
                    // Other water has come over the seed.
                    Some(with) => {
                        (Vec::new(), Filled::Met { level: super::flood::bottom(seed.y), kept: 0, with }, false)
                    }
                    None => fill(grid, seed, volume, sealed),
                };
                Outcome::Settled { seed, cells: cells.into_iter().collect(), how, edge }
            }
            Task::Route { from } => {
                let (path, end) = run_down(grid, from);
                Outcome::Routed { path, end }
            }
            Task::Sea { seeds, reach } => Outcome::Sea(spread(grid, &seeds, super::bodies::SEA, BodyId::Ocean, reach)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::WorldParams;
    use crate::edit::FP;
    use crate::material::Material;

    /// A column of the default world where stone is close under the soil,
    /// and the ops of a session of digging and building there.
    fn session(generator: &Generator) -> (IVec3, Vec<Op>) {
        let cell = generator.params().cell_size as f64;
        let ground = (generator.surface_height(4.0, 4.0) as f64 / cell).floor() as i32;
        let here = IVec3::new(16, ground, 16);
        let at = |x: i32, y: i32, z: i32| {
            let p = here + IVec3::new(x, y, z);
            [p.x as i64 * FP, p.y as i64 * FP, p.z as i64 * FP]
        };
        let ops = vec![
            Op::Break { center: at(0, -2, 0), radius: 6 * FP, nearest: false },
            Op::Break { center: at(3, -9, 1), radius: 6 * FP, nearest: false },
            Op::Break { center: at(2, -17, -1), radius: 5 * FP, nearest: true },
            Op::Chip { center: at(1, -23, 0) },
            Op::Place { center: at(-5, -3, 2), radius: 5 * FP, material: Material::SAND, exclude: None },
            Op::Break { center: at(4, -24, 3), radius: 7 * FP, nearest: false },
            Op::Chip { center: at(6, -30, 2) },
            Op::Break { center: at(-30, -1, 40), radius: 12 * FP, nearest: false },
            Op::Place { center: at(2, -20, 0), radius: 4 * FP, material: Material::DIRT, exclude: None },
            Op::Break { center: at(2, -36, 0), radius: 9 * FP, nearest: true },
        ];
        (here, ops)
    }

    fn made(generator: &Generator, rules: &EditRules, log: &EditLog, key: NodeKey) -> Chunk {
        let mut block = LocalField::block(key);
        generator.fill_block(key, &mut block);
        log::resolve(&mut block, rules, &log.gather_node(key));
        block.interior()
    }

    #[test]
    fn a_chunk_brought_up_to_date_is_the_chunk_made_afresh() {
        let params = WorldParams::default();
        let (generator, rules) = (Generator::new(params), EditRules::new(&params));
        let (here, ops) = session(&generator);
        let logs: Vec<EditLog> = (0..=ops.len())
            .map(|n| {
                let mut log = EditLog::new();
                ops[..n].iter().for_each(|op| log.push(&rules, op));
                log
            })
            .collect();
        let mut changed = 0;
        for level in 0..4 {
            for d in 0..8 {
                let pos = chunk_of(cell_of(here, level)) - IVec3::new(d & 1, d >> 1 & 1, d >> 2);
                let key = NodeKey::new(level, pos);
                let fresh = made(&generator, &rules, &logs[ops.len()], key);
                changed += (fresh != made(&generator, &rules, &logs[0], key)) as usize;
                for from in [0, 1, 4, 7, ops.len() - 1] {
                    let old = made(&generator, &rules, &logs[from], key);
                    assert!(patched(key, &old, &ops[from..], &rules) == fresh, "{key:?} from {from}");
                }
            }
        }
        assert!(changed >= 8, "the session changed {changed} chunks");
    }

    #[test]
    fn kept_terrain_is_the_terrain_made_afresh_whenever_the_edits_come() {
        let params = WorldParams::default();
        let (generator, rules) = (Generator::new(params), EditRules::new(&params));
        let (here, ops) = session(&generator);
        let (volume, mut log, mut kept) = (Volume::new(), EditLog::new(), KeptTerrain::default());
        let requests: Vec<Request> = (0..3)
            .map(|lod| Request { lod, origin: chunk_of(cell_of(here, lod)) - 1, size: IVec3::splat(2), loaded: false })
            .collect();
        let mut reused = 0;
        // Each op comes in while a search is under way; what that search
        // made is kept without it, and the searches before it are owed it.
        for round in 0..=ops.len() {
            for (n, request) in requests.iter().enumerate() {
                kept.tick();
                let terrain = kept.gather(&volume, &log, request).unwrap();
                kept.launched();
                let fresh = Terrain::gather(&volume, &log, request).unwrap();
                if n == 1 && round < ops.len() {
                    log.push(&rules, &ops[round]);
                    kept.edited(&rules, &ops[round]);
                }
                reused += terrain.pieces.iter().filter(|p| !matches!(p, Piece::Edits(_))).count();
                let ((grid, made), (want, _)) = (terrain.grid(&generator, &rules), fresh.grid(&generator, &rules));
                for_box(grid.origin, grid.origin + grid.dims - 1, |p| {
                    assert_eq!(grid.is_open(p), want.is_open(p), "round {round} lod {} at {p}", request.lod);
                });
                kept.store(made);
            }
        }
        assert_eq!(reused, ops.len() * 3 * 8);

        // More edits than are kept track of, some of them with a search under way.
        let check = |kept: &mut KeptTerrain, log: &mut EditLog, during: std::ops::Range<i32>| {
            for request in &requests {
                kept.tick();
                let terrain = kept.gather(&volume, log, request).unwrap();
                kept.launched();
                let fresh = Terrain::gather(&volume, log, request).unwrap();
                for n in during.clone() {
                    let p = here + IVec3::new(n % 9 - 4, -(n % 7), n % 5 - 2);
                    let center = [p.x as i64 * FP, p.y as i64 * FP, p.z as i64 * FP];
                    let op = Op::Break { center, radius: 2 * FP, nearest: false };
                    log.push(&rules, &op);
                    kept.edited(&rules, &op);
                }
                let ((grid, made), (want, _)) = (terrain.grid(&generator, &rules), fresh.grid(&generator, &rules));
                for_box(grid.origin, grid.origin + grid.dims - 1, |p| {
                    assert_eq!(grid.is_open(p), want.is_open(p), "lod {} at {p}", request.lod);
                });
                kept.store(made);
            }
        };
        let many = KEPT_EDITS as i32 + 8;
        check(&mut kept, &mut log, 0..many);
        check(&mut kept, &mut log, 0..0);
        check(&mut kept, &mut log, many..many + 3);
        check(&mut kept, &mut log, 0..0);
    }
}
