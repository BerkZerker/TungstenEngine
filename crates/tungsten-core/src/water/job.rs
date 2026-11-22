//! The part of water that takes time: searching a box of terrain for where
//! water lies or goes. A job is everything such a search needs, copied out
//! of the world so that it can run on a worker, and its outcome is a pure
//! function of the job: the same on every machine, however long it took.
//!
//! Part of the bit-exact contract: integers only, fixed orders.

use std::sync::Arc;

use glam::IVec3;

use crate::chunk::{Chunk, Volume};
use crate::coords::{CHUNK_SIZE, ChunkPos, NodeKey, chunk_of};
use crate::edit::EditRules;
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
        let mut pieces = Vec::with_capacity((request.size.x * request.size.y * request.size.z) as usize);
        for y in 0..request.size.y {
            for z in 0..request.size.z {
                for x in 0..request.size.x {
                    let pos = request.origin + IVec3::new(x, y, z);
                    let loaded = if request.lod == 0 { volume.chunks.get(&pos) } else { None };
                    pieces.push(match loaded {
                        Some(chunk) => Piece::Chunk(chunk.clone()),
                        None if request.loaded => return None,
                        None => Piece::Edits(log.gather_node(NodeKey::new(request.lod, pos))),
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

    fn grid(self, generator: &Generator, rules: &EditRules) -> Grid {
        let Request { lod, origin, size, .. } = self.request;
        let chunks: Vec<Arc<Chunk>> = self
            .pieces
            .into_iter()
            .enumerate()
            .map(|(n, piece)| match piece {
                Piece::Chunk(chunk) => chunk,
                Piece::Edits(cells) => {
                    let n = n as i32;
                    let pos = origin + IVec3::new(n % size.x, n / (size.x * size.z), n / size.x % size.z);
                    let mut block = LocalField::block(NodeKey::new(lod, pos));
                    generator.fill_block(NodeKey::new(lod, pos), &mut block);
                    log::resolve(&mut block, rules, &cells);
                    Arc::new(block.interior())
                }
            })
            .collect();
        Grid::from_chunks(origin, size, &chunks)
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
        let lod = self.terrain.request.lod;
        let mut grid = self.terrain.grid(generator, rules);
        mark(&mut grid, lod, generator, &self.sea, &self.others);
        match self.task {
            Task::Settle { mut seed, volume, sealed, was, was_level } => {
                // A pond that reaches out of the box cannot be told to be in pieces.
                let (lo, hi) = (grid.origin, grid.origin + grid.dims - 1);
                let whole = was.bounds().is_some_and(|(a, b)| a.cmpge(lo).all() && b.cmple(hi).all());
                let pieces = if whole { parts(&grid, was.iter(), was_level) } else { Vec::new() };
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
                    None => fill(&grid, seed, volume, sealed),
                };
                Outcome::Settled { seed, cells: cells.into_iter().collect(), how, edge }
            }
            Task::Route { from } => {
                let (path, end) = run_down(&grid, from);
                Outcome::Routed { path, end }
            }
            Task::Sea { seeds, reach } => Outcome::Sea(spread(&grid, &seeds, super::bodies::SEA, BodyId::Ocean, reach)),
        }
    }
}
