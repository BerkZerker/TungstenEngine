//! The water of a world beyond what the generator gives: ponds, water on
//! its way downhill, and where the sea has come in through dug openings.
//!
//! Part of the bit-exact contract: integers only, and every collection is
//! walked in a fixed order.
//!
//! A pond is a seed cell and a volume. Where its water lies is a function of
//! those two and the terrain ([`fill`](super::flood::fill)), worked out again
//! whenever the ground or the water around it changes; the cells kept with
//! it are that answer, for drawing and for telling whose water a cell is
//! under. So a pond lowers when a channel is dug from it and rises when
//! something is put in it, with nothing tracked per cell and its volume
//! untouched. A pond too wide for the box it is searched in is searched on
//! the lattice of a coarser level of detail, where the same box is twice
//! the size.
//!
//! A flow is water on its way: a route from where it was let go to where it
//! ends up, and the lots of water let go into it. Where the water is along
//! the route is a formula of the time since, at a fixed speed, and it is
//! handed over at the end as it arrives.
//!
//! The searches (where a pond lies, which way a flow runs, how far the sea
//! gets) are [`WaterJob`]s, run one at a time. A job is started on one tick
//! and its outcome is put to use exactly [`DELAY`] ticks later, however
//! long it took to work out: the world is the same on every machine.

use std::collections::{BTreeMap, VecDeque};

use glam::IVec3;

use crate::chunk::Volume;
use crate::codec::{get_signed, get_u8, get_varint, put_signed, put_varint};
use crate::coords::{CHUNK_BITS, CHUNK_SIZE, MAX_LEVEL, NodeKey};
use crate::edit::FP;
use crate::worldgen::Generator;

use super::cells::Cells;
use super::flood::{Filled, Grid, Run, anchor, bottom, top_layer};
use super::job::{Other, Outcome, Request, Task, Terrain, WaterJob, base_of, cell_of, mark};
use super::{BodyId, NodeWater, Surface};

/// Level of the sea, fixed-point base cells. The sea's surface is anchored
/// at lattice height zero, which is where `SEA_LEVEL` is.
pub const SEA: i64 = 0;
/// A pond of less than this soaks away once nothing runs into it: an eighth
/// of a cubic metre at the default cell size.
pub const MIN_POND: i64 = 8 * FP;
/// Ticks from starting a search to using what it found.
pub const DELAY: u64 = 4;
/// Speed of water on its way, in `1 / FP` of a base cell per tick: 5.6 m/s
/// at the default cell size.
pub const SPEED: i64 = 96;
/// Chunks along each side of the box a way downhill or the sea's advance is
/// searched in, and across the box a pond is: 16 m and 32 m at the finest
/// level and the default cell size. Both are 16 m high.
const BOX: i32 = 2;
const WIDE: i32 = 4;
/// How far the sea gets per search, in cells: with `DELAY`, 11 m/s.
const SEA_REACH: u32 = 12;
/// Cells of the sea's advance tried per tick for one in loaded terrain.
const FRONTIER_TRIES: usize = 64;
/// The most a pond lets go over its rim per tick: a quarter of a cubic metre.
const SPILL_RATE: i64 = 16 * FP;
/// Water poured within this many cells of a flow's source joins that flow.
pub const JOIN: i32 = 4;
/// Ticks a flow with no water in it is kept, in case more comes.
const LINGER: u64 = 30;
/// Ticks a pond that has gone down is still drawn over the ground it left,
/// so that its surface can be seen to sink.
pub const EBB_TICKS: u64 = 90;

/// The ground a pond's surface covered before its level fell.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Ebb {
    cells: Cells,
    lod: u8,
    level: i64,
    until: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pond {
    /// A cell of the pond on the finest lattice, the lowest when it was made.
    pub seed: IVec3,
    /// Units of `1 / FP` of a base cell cubed.
    pub volume: i64,
    /// Surface height, fixed-point base cells.
    pub level: i64,
    /// The level of detail whose lattice `cells` are on.
    pub lod: u8,
    /// The cells under its water as of when it last settled.
    pub cells: Cells,
    /// The level of detail it is to be searched at next: coarser than `lod`
    /// once it has outgrown the box.
    want: u8,
    /// Even the coarsest box is too small: its sides count as shore.
    sealed: bool,
    /// Something changed near it since it settled.
    unsettled: bool,
    /// It has taken in another pond since it settled: its cells are both
    /// ponds', which says nothing of where the water of the two lies.
    joined: bool,
    /// The box on the finest lattice it is known to reach.
    span: Option<(IVec3, IVec3)>,
    /// Full to its rim: where more water goes without asking again.
    over: Option<Over>,
    /// The height the water that made it came down from; see [`Flow::head`].
    head: i64,
    /// Its own overflow came back to it, so it keeps what it is given;
    /// see [`Water::returned`].
    holds: bool,
    ebb: Option<Ebb>,
}

/// Where what a full pond cannot hold goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Over {
    /// Over the rim at this cell of the finest lattice.
    At(IVec3),
    Sea,
}

/// Water let go into a flow over some ticks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Lot {
    start: u64,
    ticks: u32,
    volume: i64,
    /// How much of it has been handed over at the end of the route.
    delivered: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum End {
    /// A hollow with no water in it yet.
    Rest(IVec3),
    Into(BodyId),
    /// Still falling at the bottom of the box the route was searched in: another flow carries on.
    Out(IVec3),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Route {
    /// The cells where the way turns, on the finest lattice.
    path: Vec<IVec3>,
    /// Distance along the path to each of them, in `1 / FP` of a cell.
    along: Vec<i64>,
    end: End,
}

impl Route {
    fn new(path: Vec<IVec3>, end: End) -> Self {
        let mut along = Vec::with_capacity(path.len());
        let mut at = 0;
        for (i, p) in path.iter().enumerate() {
            if i > 0 {
                at += (*p - path[i - 1]).abs().element_sum() as i64 * FP;
            }
            along.push(at);
        }
        Self { path, along, end }
    }

    fn length(&self) -> i64 {
        self.along.last().copied().unwrap_or(0)
    }

    /// Ticks water takes from one end to the other.
    fn travel(&self) -> u64 {
        ((self.length() + SPEED - 1) / SPEED) as u64
    }

    fn bounds(&self) -> (IVec3, IVec3) {
        self.path.iter().fold((self.path[0], self.path[0]), |(lo, hi), &p| (lo.min(p), hi.max(p)))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Flow {
    /// Where the water is let go, on the finest lattice.
    source: IVec3,
    /// The surface level of the lowest pond this water has run over the rim
    /// of, and `i64::MAX` for water that was only ever poured. Water runs
    /// downhill, so a pond it reaches lies lower than that, or would if
    /// every pond were on the same lattice.
    head: i64,
    /// `None` until it has been searched for, and again when the ground changes.
    route: Option<Route>,
    lots: VecDeque<Lot>,
    /// Tick it was last given water.
    fed: u64,
}

impl Flow {
    fn pending(&self) -> i64 {
        self.lots.iter().map(|l| l.volume - l.delivered).sum()
    }
}

/// A stretch of a flow with water in it, for drawing.
#[derive(Clone, Debug, PartialEq)]
pub struct Stream {
    /// The turns of the route, on the finest lattice.
    pub path: Vec<IVec3>,
    /// Distance along the path to each turn, and of the stretch's two ends, in cells.
    pub along: Vec<f32>,
    pub from: f32,
    pub to: f32,
    /// Base cells cubed per tick.
    pub rate: f32,
}

enum Kind {
    /// A pond, the lattice it is searched on, and its volume at the start.
    Settle {
        id: u32,
        lod: u8,
        held: i64,
    },
    Route(u32),
    /// The cells the sea's advance was carried on from.
    Sea(Vec<IVec3>),
}

/// A search that is under way.
struct Flight {
    due: u64,
    kind: Kind,
    /// The box it looks at, on the finest lattice.
    bounds: (IVec3, IVec3),
    /// The ground in the box changed since: what it finds is not used.
    stale: bool,
}

#[derive(Default)]
pub struct Water {
    tick: u64,
    ponds: BTreeMap<u32, Pond>,
    next_pond: u32,
    /// Cells the sea has come into through openings dug in its shore.
    sea: Cells,
    /// Where the sea's advance carries on from.
    frontier: VecDeque<IVec3>,
    /// None of the frontier was in loaded terrain when last tried.
    stuck: bool,
    flows: BTreeMap<u32, Flow>,
    next_flow: u32,
    /// Volume that ran into the sea.
    drained: i64,
    /// Volume that is gone: puddles too small to keep.
    lost: i64,
    flight: Option<Flight>,
    /// Which kind of search has the next turn, and who had the last of each.
    turn: u8,
    last_settled: u32,
    last_routed: u32,
    /// Boxes (inclusive, with the surfaces' anchor layers as their height)
    /// in which surfaces changed and have to be drawn again.
    changed: Vec<(IVec3, IVec3)>,
    unsaved: bool,
}

/// A summary for display.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaterStats {
    pub ponds: usize,
    /// Volume held by ponds, on its way downhill, taken by the sea, and lost.
    pub held: i64,
    pub moving: i64,
    pub drained: i64,
    pub lost: i64,
    pub flows: usize,
    /// Cells the sea has come into, and how many more it is waiting to look at.
    pub sea_cells: usize,
    pub sea_pending: usize,
}

fn overlaps(a: (IVec3, IVec3), b: (IVec3, IVec3)) -> bool {
    a.0.cmple(b.1).all() && a.1.cmpge(b.0).all()
}

fn union(a: Option<(IVec3, IVec3)>, b: Option<(IVec3, IVec3)>) -> Option<(IVec3, IVec3)> {
    match (a, b) {
        (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1))),
        (a, b) => a.or(b),
    }
}

/// The box on the finest lattice that cells of level of detail `lod` take up.
fn reach(cells: &Cells, lod: u8) -> Option<(IVec3, IVec3)> {
    let half = (1 << lod) >> 1;
    cells.bounds().map(|(lo, hi)| (base_of(lo, lod) - half, base_of(hi, lod) + half))
}

/// The box searched for water at `at`, a cell of level of detail `lod`:
/// whole chunks, `across` of them each way, with the cell within half a
/// chunk of the middle and a quarter of a chunk or more above the bottom.
fn search(at: IVec3, lod: u8, across: i32, loaded: bool) -> Request {
    let half = CHUNK_SIZE / 2;
    let middle = (at + IVec3::new(half, half + half / 2, half)) >> CHUNK_BITS;
    let origin = middle - IVec3::new(across / 2, 1, across / 2);
    Request { lod, origin, size: IVec3::new(across, BOX, across), loaded }
}

/// Ticks over which a pond lets `volume` go over its rim.
fn spill_ticks(volume: i64) -> u32 {
    (volume / SPILL_RATE).clamp(2 * DELAY as i64, 6000) as u32
}

const MAGIC: &[u8; 4] = b"TNGW";
const VERSION: u8 = 2;

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

impl Pond {
    fn new(seed: IVec3, volume: i64, level: i64, lod: u8, cells: Cells, head: i64) -> Self {
        Self {
            seed,
            volume,
            level,
            lod,
            span: reach(&cells, lod),
            cells,
            want: lod,
            sealed: false,
            unsettled: true,
            joined: false,
            over: None,
            head,
            holds: false,
            ebb: None,
        }
    }

    /// True if a change to the ground in a box may move its water.
    fn unsettled_by(&self, changed: (IVec3, IVec3)) -> bool {
        // The cell above the surface counts: it may be filled in from there.
        self.span.is_some_and(|(lo, hi)| overlaps((lo, hi + IVec3::Y * (1 << self.lod)), changed))
    }
}

impl Water {
    pub fn new() -> Self {
        Self::default()
    }

    /// The body whose water a cell of the finest lattice is under, if any.
    pub fn holder(&self, generator: &Generator, p: IVec3) -> Option<BodyId> {
        let pond = |(&id, pond): (&u32, &Pond)| pond.cells.contains(cell_of(p, pond.lod)).then_some(BodyId::Pond(id));
        if let Some(body) = self.ponds.iter().find_map(pond) {
            return Some(body);
        }
        if bottom(p.y) >= SEA {
            return None;
        }
        let open_sea = || {
            let heights = generator.heights(0, p.x >> CHUNK_BITS, p.z >> CHUNK_BITS);
            heights.at(p.x & (CHUNK_SIZE - 1), p.z & (CHUNK_SIZE - 1)) < crate::worldgen::SEA_LEVEL
                && generator.sea_reaches(p.x >> CHUNK_BITS, p.z >> CHUNK_BITS)
        };
        (self.sea.contains(p) || open_sea()).then_some(BodyId::Ocean)
    }

    pub fn ponds(&self) -> impl Iterator<Item = (u32, &Pond)> {
        self.ponds.iter().map(|(&id, p)| (id, p))
    }

    pub fn stats(&self) -> WaterStats {
        WaterStats {
            ponds: self.ponds.len(),
            held: self.ponds.values().map(|p| p.volume).sum(),
            moving: self.flows.values().map(Flow::pending).sum(),
            drained: self.drained,
            lost: self.lost,
            flows: self.flows.values().filter(|f| !f.lots.is_empty()).count(),
            sea_cells: self.sea.len(),
            sea_pending: self.frontier.len(),
        }
    }

    /// Surface level of every body whose level can change.
    pub fn levels(&self) -> impl Iterator<Item = (BodyId, i64)> + '_ {
        self.ponds.iter().map(|(&id, p)| (BodyId::Pond(id), p.level))
    }

    /// The stretches of every flow that have water in them now.
    pub fn streams(&self) -> Vec<Stream> {
        self.streams_at(0.0)
    }

    /// The stretches of every flow with water in them at fractional tick `alpha` in `[0, 1]`,
    /// for smooth client-side interpolation between fixed ticks.
    pub fn streams_at(&self, alpha: f32) -> Vec<Stream> {
        let speed = SPEED as f32 / FP as f32;
        let alpha = alpha.clamp(0.0, 1.0);
        let mut out = Vec::new();
        for flow in self.flows.values() {
            let Some(route) = &flow.route else { continue };
            let length = route.length() as f32 / FP as f32;
            if length <= 0.0 {
                continue;
            }
            for lot in &flow.lots {
                let since = (self.tick - lot.start) as f32 + alpha;
                let head = (since * speed).min(length);
                let tail = ((since - lot.ticks as f32) * speed).clamp(0.0, length);
                if head > tail {
                    out.push(Stream {
                        path: route.path.clone(),
                        along: route.along.iter().map(|&a| a as f32 / FP as f32).collect(),
                        from: tail,
                        to: head,
                        rate: lot.volume as f32 / lot.ticks as f32 / FP as f32,
                    });
                }
            }
        }
        out
    }

    /// True if anything changed since the last call; for knowing when to save.
    pub fn take_unsaved(&mut self) -> bool {
        std::mem::take(&mut self.unsaved)
    }

    /// Boxes in which water surfaces changed since the last call. Height is
    /// the layers the surfaces were and are anchored in.
    pub fn take_changed(&mut self) -> Vec<(IVec3, IVec3)> {
        std::mem::take(&mut self.changed)
    }

    /// Let `volume` of water go at `at`, a cell of the finest lattice. It
    /// joins a flow whose source is within `join` cells, or starts one.
    pub fn pour(&mut self, at: IVec3, volume: i64, join: i32) {
        self.feed(at, volume, 1, i64::MAX, join);
    }

    /// Put `volume` into the flow from `source`, let go over `ticks` ticks from now.
    fn feed(&mut self, source: IVec3, volume: i64, ticks: u32, head: i64, join: i32) {
        if volume <= 0 {
            return;
        }
        let near = self.flows.iter().find(|(_, f)| (f.source - source).abs().max_element() <= join).map(|(&id, _)| id);
        let id = near.unwrap_or_else(|| {
            let id = self.next_flow;
            self.next_flow += 1;
            self.flows.insert(id, Flow { source, head, route: None, lots: VecDeque::new(), fed: 0 });
            id
        });
        let (now, flow) = (self.tick, self.flows.get_mut(&id).unwrap());
        (flow.fed, flow.head) = (now, flow.head.min(head));
        match flow.lots.back_mut() {
            // It runs on from the lot before: one stretch of water.
            Some(last) if now <= last.start + last.ticks as u64 => {
                let end = (last.start + last.ticks as u64).max(now + ticks as u64);
                last.ticks = (end - last.start) as u32;
                last.volume += volume;
            }
            _ => flow.lots.push_back(Lot { start: now, ticks: ticks.max(1), volume, delivered: 0 }),
        }
        self.unsaved = true;
    }

    fn note(&mut self, cells: Option<(IVec3, IVec3)>, levels: [i64; 2]) {
        let Some((lo, hi)) = cells else { return };
        let (a, b) = (anchor(levels[0]), anchor(levels[1]));
        // A wet column is drawn by the cells on every side of it.
        self.changed.push((IVec3::new(lo.x - 1, a.min(b), lo.z - 1), IVec3::new(hi.x + 1, a.max(b), hi.z + 1)));
        self.unsaved = true;
    }

    /// True if water is on its way into a pond.
    fn fed(&self, id: u32) -> bool {
        let into = |f: &Flow| f.route.as_ref().is_some_and(|r| r.end == End::Into(BodyId::Pond(id)));
        self.flows.values().any(|f| !f.lots.is_empty() && into(f))
    }

    /// Give a pond the cells and level it settled to.
    fn place(&mut self, id: u32, lod: u8, seed: IVec3, cells: Cells, level: i64) {
        let until = self.tick + EBB_TICKS;
        let pond = self.ponds.get_mut(&id).unwrap();
        let old = (pond.span, pond.level);
        let same = cells == pond.cells && level == pond.level && lod == pond.lod;
        let was = std::mem::replace(&mut pond.cells, cells);
        if level < pond.level && !was.is_empty() {
            // Keep the widest it has been while it sinks.
            match &mut pond.ebb {
                Some(ebb) => ebb.until = until,
                None => pond.ebb = Some(Ebb { cells: was, lod: pond.lod, level: pond.level, until }),
            }
        }
        (pond.lod, pond.level, pond.seed) = (lod, level, base_of(seed, lod));
        pond.span = reach(&pond.cells, lod);
        let new = pond.span;
        if !same {
            self.note(union(old.0, new), [old.1, level]);
        }
    }

    fn remove(&mut self, id: u32) -> Option<Pond> {
        let pond = self.ponds.remove(&id)?;
        let wide = union(pond.span, pond.ebb.as_ref().and_then(|e| reach(&e.cells, e.lod)));
        self.note(wide, [pond.level, pond.ebb.as_ref().map_or(pond.level, |e| e.level)]);
        Some(pond)
    }

    /// A pond with too little in it to keep, and nothing coming, soaks away.
    fn soak(&mut self, id: u32) {
        if self.ponds.get(&id).is_some_and(|p| p.volume < MIN_POND) && !self.fed(id) {
            let pond = self.remove(id).unwrap();
            self.lost += pond.volume;
            self.forget(id, None);
        }
    }

    /// A pond is gone. Flows that ran into it run into `now` instead, or
    /// are searched for again.
    fn forget(&mut self, id: u32, now: Option<u32>) {
        for flow in self.flows.values_mut() {
            if flow.route.as_ref().is_some_and(|r| r.end == End::Into(BodyId::Pond(id))) {
                match now {
                    Some(now) => flow.route.as_mut().unwrap().end = End::Into(BodyId::Pond(now)),
                    None => flow.route = None,
                }
            }
        }
    }

    /// The box a pond was searched in was too small for it. Returns false
    /// if there is no larger one: its sides are already taken for shore.
    fn outgrow(&mut self, id: u32) -> bool {
        let Some(pond) = self.ponds.get_mut(&id) else { return false };
        if pond.sealed {
            return false;
        }
        if pond.want < MAX_LEVEL {
            pond.want += 1;
        } else {
            pond.sealed = true;
        }
        (pond.unsettled, pond.holds) = (true, false);
        true
    }

    /// Water has come to a pond from no higher than its own surface: its
    /// own overflow, most likely, by a way that led back to it. That does
    /// not happen on one lattice, where water only goes down. It happens
    /// when the box a pond was searched in is too small to show that the
    /// ground its overflow ran off to is part of the same hollow, and then a
    /// wider box will do. And it happens to a pond on a coarse lattice,
    /// whose rim is not where the finest lattice has it; such a pond keeps
    /// what it is given from then on, until the ground changes.
    fn returned(&mut self, id: u32, volume: i64) {
        let pond = self.ponds.get_mut(&id).unwrap();
        pond.volume += volume;
        (pond.unsettled, pond.over) = (true, None);
        if pond.want > 0 || !self.outgrow(id) {
            self.ponds.get_mut(&id).unwrap().holds = true;
        }
    }

    fn new_pond(&mut self, pond: Pond) -> u32 {
        let id = self.next_pond;
        self.next_pond += 1;
        self.ponds.insert(id, pond);
        self.unsaved = true;
        id
    }

    /// The ponds whose water may lie in a box, other than `skip`.
    fn others(&self, bounds: (IVec3, IVec3), skip: Option<u32>) -> Vec<Other> {
        let near = |p: &Pond| p.span.is_some_and(|s| overlaps(s, bounds));
        let other = |(&id, p): (&u32, &Pond)| Other { id, lod: p.lod, cells: p.cells.clone() };
        self.ponds.iter().filter(|(id, p)| Some(**id) != skip && near(p)).map(other).collect()
    }

    fn launch(&mut self, kind: Kind, task: Task, terrain: Terrain, request: &Request, skip: Option<u32>) -> WaterJob {
        let bounds = request.bounds();
        self.flight = Some(Flight { due: self.tick + DELAY, kind, bounds, stale: false });
        WaterJob { task, terrain, sea: self.sea.clone(), others: self.others(bounds, skip) }
    }

    fn begin_settle(&mut self, terrain: &mut impl FnMut(&Request) -> Option<Terrain>) -> Option<WaterJob> {
        let ready = |(&id, p): (&u32, &Pond)| (p.unsettled && p.volume > 0).then_some(id);
        let after = self.ponds.range(self.last_settled + 1..).find_map(ready);
        let id = after.or_else(|| self.ponds.iter().find_map(ready))?;
        self.last_settled = id;
        let pond = self.ponds.get_mut(&id).unwrap();
        // A level of detail whose box holds what the pond is known to reach
        // and has something of its volume in every cell.
        let request = loop {
            let request = search(cell_of(pond.seed, pond.want), pond.want, WIDE, false);
            let (lo, hi) = request.bounds();
            let margin = 2 << pond.want;
            let fits = pond.span.is_none_or(|s| s.0.cmpge(lo + margin).all() && s.1.cmple(hi - margin).all());
            if pond.volume >> (3 * pond.want) == 0 {
                (pond.want, pond.span) = (0, None);
            } else if fits || pond.want == MAX_LEVEL {
                break request;
            } else {
                pond.want += 1;
            }
        };
        let lod = pond.want;
        let task = Task::Settle {
            seed: cell_of(pond.seed, lod),
            volume: pond.volume >> (3 * lod),
            sealed: pond.sealed,
            was: if lod == pond.lod && !pond.joined { pond.cells.clone() } else { Cells::default() },
            was_level: pond.level >> lod,
        };
        let kind = Kind::Settle { id, lod, held: pond.volume };
        let terrain = terrain(&request)?;
        Some(self.launch(kind, task, terrain, &request, Some(id)))
    }

    fn begin_route(&mut self, terrain: &mut impl FnMut(&Request) -> Option<Terrain>) -> Option<WaterJob> {
        let lost = |(&id, f): (&u32, &Flow)| (f.route.is_none() && !f.lots.is_empty()).then_some(id);
        let after = self.flows.range(self.last_routed + 1..).find_map(lost);
        let id = after.or_else(|| self.flows.iter().find_map(lost))?;
        self.last_routed = id;
        let from = self.flows[&id].source;
        let request = search(from, 0, BOX, false);
        let terrain = terrain(&request)?;
        Some(self.launch(Kind::Route(id), Task::Route { from }, terrain, &request, None))
    }

    fn begin_sea(&mut self, terrain: &mut impl FnMut(&Request) -> Option<Terrain>) -> Option<WaterJob> {
        for _ in 0..self.frontier.len().min(FRONTIER_TRIES) {
            let p = self.frontier.pop_front().unwrap();
            let request = search(p, 0, BOX, true);
            // Not loaded: it waits.
            let Some(terrain) = terrain(&request) else {
                self.frontier.push_back(p);
                continue;
            };
            let (lo, hi) = request.bounds();
            let mut seeds = vec![p];
            self.frontier.retain(|&q| {
                let inside = q.cmpge(lo).all() && q.cmple(hi).all();
                if inside {
                    seeds.push(q);
                }
                !inside
            });
            self.stuck = false;
            let task = Task::Sea { seeds: seeds.clone(), reach: SEA_REACH };
            return Some(self.launch(Kind::Sea(seeds), task, terrain, &request, None));
        }
        self.stuck = !self.frontier.is_empty();
        None
    }

    /// Start the next search, if there is anything to search for. `terrain`
    /// gives the terrain in a box, or `None` if it has to be loaded and is not.
    fn begin(&mut self, mut terrain: impl FnMut(&Request) -> Option<Terrain>) -> Option<WaterJob> {
        for _ in 0..3 {
            let turn = self.turn;
            self.turn = (self.turn + 1) % 3;
            let job = match turn {
                0 => self.begin_settle(&mut terrain),
                1 => self.begin_route(&mut terrain),
                _ => self.begin_sea(&mut terrain),
            };
            if job.is_some() {
                return job;
            }
        }
        None
    }

    /// Put what a search found to use.
    fn finish(&mut self, kind: Kind, outcome: Outcome) {
        match (kind, outcome) {
            (Kind::Settle { id, lod, held }, Outcome::Settled { seed, cells, how, edge }) => {
                let Some(pond) = self.ponds.get_mut(&id) else { return };
                // Water that has run in since the search began: it settles
                // again, unless the pond was found full, when that is over too.
                let late = pond.volume - held;
                let holds = pond.holds;
                pond.unsettled = late != 0;
                pond.over = None;
                pond.joined = false;
                let unit = 3 * lod;
                // What is over when `kept` cells of this lattice are full. The part of
                // the volume that is less than one such cell stays with the pond.
                let over = |kept: i64| ((held >> unit) << unit) - (kept << unit) + late;
                match how {
                    // Up against the side of the box, a wider one will tell; until then it stays as it was.
                    Filled::Settled { .. } | Filled::Spilled { at: None, .. } if edge && self.outgrow(id) => {}
                    Filled::Settled { level } => {
                        self.ponds.get_mut(&id).unwrap().holds = false;
                        self.place(id, lod, seed, cells, level << lod);
                    }
                    // A space with no way out that is full: the rest is held under pressure.
                    Filled::Spilled { level, at: None, .. } => self.place(id, lod, seed, cells, level << lod),
                    // Full to the rim, and keeping what is over; see `returned`.
                    Filled::Spilled { level, .. } if holds => self.place(id, lod, seed, cells, level << lod),
                    Filled::Spilled { level, kept, at: Some(at) } => {
                        let (over, at, level) = (over(kept), base_of(at, lod), level << lod);
                        let pond = self.ponds.get_mut(&id).unwrap();
                        pond.volume -= over;
                        pond.unsettled = false;
                        pond.over = Some(Over::At(at));
                        let head = pond.head.min(level);
                        self.place(id, lod, seed, cells, level);
                        self.feed(at, over, spill_ticks(over), head, 0);
                    }
                    Filled::Met { with: BodyId::Pond(other), .. } => {
                        if !self.ponds.contains_key(&other) {
                            self.ponds.get_mut(&id).unwrap().unsettled = true;
                            return;
                        }
                        // Two ponds in one hollow are one pond, as wide as
                        // both. The one on the coarser lattice, or else the
                        // larger, takes in the other.
                        let rank = |id: u32| (self.ponds[&id].lod, self.ponds[&id].volume, id);
                        let (keep, gone) = if rank(other) > rank(id) { (other, id) } else { (id, other) };
                        let gone_pond = self.remove(gone).unwrap();
                        let pond = self.ponds.get_mut(&keep).unwrap();
                        pond.volume += gone_pond.volume;
                        pond.span = union(pond.span, gone_pond.span);
                        pond.want = pond.want.max(gone_pond.want);
                        pond.sealed |= gone_pond.sealed;
                        (pond.unsettled, pond.joined, pond.over) = (true, true, None);
                        if pond.lod == gone_pond.lod {
                            // Until it has settled, it is drawn where both were.
                            pond.cells.extend(gone_pond.cells.iter());
                            pond.head = pond.head.max(gone_pond.head);
                        } else if gone_pond.head <= pond.level {
                            // Its own overflow, come to rest in a dip that its lattice has for part of it.
                            pond.holds = true;
                        }
                        self.forget(gone, Some(keep));
                    }
                    // Above the sea: what does not fit below the rim runs down into it.
                    Filled::Met { level, kept, with: BodyId::Ocean } if level << lod >= SEA => {
                        let over = over(kept);
                        let pond = self.ponds.get_mut(&id).unwrap();
                        pond.volume -= over;
                        (pond.unsettled, pond.holds) = (false, false);
                        pond.over = Some(Over::Sea);
                        self.drained += over;
                        self.place(id, lod, seed, cells, level << lod);
                    }
                    Filled::Met { with: BodyId::Ocean, .. } => {
                        // Below sea level and open to it: the sea comes in.
                        let pond = self.remove(id).unwrap();
                        self.drained += pond.volume;
                        self.frontier.push_back(base_of(seed, lod));
                        self.forget(id, None);
                    }
                }
                self.soak(id);
            }
            (Kind::Settle { id, lod, .. }, Outcome::Split { parts }) => {
                let Some(pond) = self.ponds.get_mut(&id) else { return };
                let (held, level, whole) = (pond.volume, pond.level, pond.span);
                let total: i64 = parts.iter().map(|p| p.1).sum();
                // Each piece keeps the share of the water that it held.
                let mut shares: Vec<i64> = match total {
                    0 => vec![0; parts.len()],
                    _ => parts.iter().map(|p| (held as i128 * p.1 as i128 / total as i128) as i64).collect(),
                };
                shares[0] += held - shares.iter().sum::<i64>();
                let head = pond.head;
                for (i, ((seed, _, cells), share)) in parts.into_iter().zip(shares).enumerate() {
                    let piece = Pond::new(base_of(seed, lod), share, level, lod, cells, head);
                    if i == 0 {
                        *self.ponds.get_mut(&id).unwrap() = piece;
                    } else if share > 0 {
                        self.new_pond(piece);
                    }
                }
                self.note(whole, [level; 2]);
            }
            (Kind::Settle { id, .. }, Outcome::Buried) => {
                // Filled in whole: its water is pushed up and runs off.
                let Some(pond) = self.remove(id) else { return };
                self.forget(id, None);
                self.feed(pond.seed, pond.volume, spill_ticks(pond.volume), pond.head, 0);
            }
            (Kind::Route(flow), Outcome::Routed { path, end }) => {
                let Some(found) = self.flows.get_mut(&flow) else { return };
                let end = match end {
                    Run::Rest(at) => End::Rest(at),
                    Run::Into(body) => End::Into(body),
                    Run::Out(at) => End::Out(at),
                };
                found.route = Some(Route::new(path, end));
            }
            (Kind::Sea(_), Outcome::Sea(found)) => {
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
                        self.forget(id, None);
                    }
                }
            }
            _ => debug_assert!(false, "a search came back with the wrong kind of outcome"),
        }
    }

    /// Hand over the water that has reached the end of its route this tick.
    fn run_flows(&mut self) {
        let now = self.tick;
        let mut arrived: Vec<(u32, i64)> = Vec::new();
        for (&id, flow) in &mut self.flows {
            let Some(route) = &flow.route else { continue };
            let travel = route.travel();
            let mut total = 0;
            for lot in &mut flow.lots {
                if now < lot.start + travel {
                    continue;
                }
                let since = now - lot.start - travel;
                let due = match since >= lot.ticks as u64 {
                    true => lot.volume,
                    false => (lot.volume as i128 * since as i128 / lot.ticks as i128) as i64,
                };
                if due > lot.delivered {
                    total += due - lot.delivered;
                    lot.delivered = due;
                }
            }
            flow.lots.retain(|l| l.delivered < l.volume || now < l.start + l.ticks as u64);
            if total > 0 {
                arrived.push((id, total));
            }
        }
        for (id, volume) in arrived {
            let flow = self.flows.get_mut(&id).unwrap();
            let head = flow.head;
            match flow.route.as_ref().unwrap().end {
                // The sea takes it without rising.
                End::Into(BodyId::Ocean) => self.drained += volume,
                End::Into(BodyId::Pond(pond)) => match self.ponds.get_mut(&pond) {
                    Some(Pond { level, .. }) if head <= *level => self.returned(pond, volume),
                    // Full to its rim, with nothing changed since: the water goes straight on.
                    Some(Pond { unsettled: false, over: Some(over), level, .. }) => match *over {
                        Over::At(at) => {
                            let head = head.min(*level);
                            self.feed(at, volume, 1, head, 0)
                        }
                        Over::Sea => self.drained += volume,
                    },
                    Some(pond) => {
                        pond.volume += volume;
                        pond.unsettled = true;
                    }
                    None => {
                        // The pond has gone: the water looks for its way again.
                        flow.route = None;
                        flow.lots.push_front(Lot { start: now, ticks: 1, volume, delivered: 0 });
                    }
                },
                End::Rest(at) => {
                    let pond = self.new_pond(Pond::new(at, volume, bottom(at.y), 0, Cells::default(), head));
                    self.flows.get_mut(&id).unwrap().route.as_mut().unwrap().end = End::Into(BodyId::Pond(pond));
                }
                End::Out(at) => self.feed(at, volume, 1, head, 0),
            }
            self.unsaved = true;
        }
        // Flows that have run dry and been left alone go.
        let done: Vec<(u32, Option<End>)> = self
            .flows
            .iter()
            .filter(|(_, f)| f.lots.is_empty() && now >= f.fed + LINGER)
            .map(|(&id, f)| (id, f.route.as_ref().map(|r| r.end)))
            .collect();
        for (id, end) in done {
            self.flows.remove(&id);
            // What it ran into may now be a puddle with nothing coming.
            if let Some(End::Into(BodyId::Pond(pond))) = end {
                self.soak(pond);
            }
        }
    }

    /// One tick. If a search started [`DELAY`] ticks ago, `outcome` is
    /// called for what it found, and must wait for it if need be. Returns
    /// the next search to run, if one is started; `terrain` gives it the
    /// terrain in a box, or `None` if that has to be loaded and is not.
    pub fn step(
        &mut self,
        outcome: impl FnOnce() -> Outcome,
        terrain: impl FnMut(&Request) -> Option<Terrain>,
    ) -> Option<WaterJob> {
        self.tick += 1;
        if self.flight.as_ref().is_some_and(|f| f.due <= self.tick) {
            let flight = self.flight.take().unwrap();
            let outcome = outcome();
            match flight.kind {
                Kind::Sea(seeds) if flight.stale => self.frontier.extend(seeds),
                _ if flight.stale => {}
                kind => self.finish(kind, outcome),
            }
        }
        self.run_flows();
        let now = self.tick;
        let ebbed: Vec<u32> = self
            .ponds
            .iter()
            .filter(|(_, p)| p.ebb.as_ref().is_some_and(|e| e.until <= now))
            .map(|(&id, _)| id)
            .collect();
        for id in ebbed {
            let pond = self.ponds.get_mut(&id).unwrap();
            let ebb = pond.ebb.take().unwrap();
            let (wide, levels) = (reach(&ebb.cells, ebb.lod), [pond.level, ebb.level]);
            self.note(wide, levels);
        }
        match self.flight {
            Some(_) => None,
            None => self.begin(terrain),
        }
    }

    /// True while a search is under way.
    pub fn searching(&self) -> bool {
        self.flight.is_some()
    }

    /// True if ticks would change nothing more: no search under way, every
    /// pond settled, no water on its way, and the sea as far in as the
    /// loaded terrain lets it.
    pub fn at_rest(&self) -> bool {
        self.flight.is_none()
            && self.ponds.values().all(|p| !p.unsettled && p.ebb.is_none())
            && self.flows.values().all(|f| f.lots.is_empty())
            && (self.frontier.is_empty() || self.stuck)
    }

    /// The terrain in `[min, max]` changed. Ponds and flows near it are
    /// looked at again; where it opened ground beside the sea, the sea
    /// comes in.
    pub fn edited(&mut self, volume: &Volume, generator: &Generator, min: IVec3, max: IVec3) {
        let (min, max) = (min - 1, max + 1);
        for pond in self.ponds.values_mut() {
            if pond.unsettled_by((min, max)) {
                (pond.unsettled, pond.holds) = (true, false);
            }
        }
        for flow in self.flows.values_mut() {
            if flow.route.as_ref().is_some_and(|r| overlaps(r.bounds(), (min, max))) {
                flow.route = None;
            }
        }
        if let Some(flight) = &mut self.flight {
            flight.stale |= overlaps(flight.bounds, (min, max));
            // Its pond may reach out of the box it is searched in.
            if let Kind::Settle { id, .. } = flight.kind {
                flight.stale |= self.ponds.get(&id).is_some_and(|p| p.unsettled_by((min, max)));
            }
        }
        self.stuck = false;
        if bottom(min.y) >= SEA {
            return;
        }
        let Some(mut grid) = Grid::new(volume, min, max - min + 1) else { return };
        mark(&mut grid, 0, generator, &self.sea, &[]);
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

    /// The surfaces of ponds, and of the sea where it has come in, that pass
    /// through a node.
    pub fn node_water(&self, key: NodeKey) -> NodeWater {
        let mut out = NodeWater::default();
        let (o, size) = (key.origin(), key.size());
        // `shown` is the level the cells were under water at; `level` the body's own.
        let mut add = |body: BodyId, level: i64, cells: &Cells, lod: u8, shown: i64| {
            let Some((lo, hi)) = reach(cells, lod) else { return };
            // A node draws a wet column on its far edge too.
            let (anchor, far) = (anchor(level), o + size);
            if anchor < o.y || anchor >= far.y || hi.x < o.x - 1 || lo.x > far.x || hi.z < o.z - 1 || lo.z > far.z {
                return;
            }
            let chunk = CHUNK_SIZE << lod;
            let near = |c: i32, o: i32| (c + 1) * chunk + 1 > o && c * chunk - chunk <= o + size;
            let layer = cells.layer(top_layer(shown >> lod));
            let tiles: Vec<_> = layer.filter(|((cx, cz), _)| near(*cx, o.x) && near(*cz, o.z)).collect();
            if !tiles.is_empty() {
                out.surfaces.push(Surface { body, level, lod, tiles });
            }
        };
        add(BodyId::Ocean, SEA, &self.sea, 0, SEA);
        for (&id, pond) in &self.ponds {
            add(BodyId::Pond(id), pond.level, &pond.cells, pond.lod, pond.level);
            if let Some(ebb) = &pond.ebb {
                add(BodyId::Pond(id), pond.level, &ebb.cells, ebb.lod, ebb.level);
            }
        }
        out
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(VERSION);
        put_varint(&mut out, self.tick);
        put_varint(&mut out, self.next_pond as u64);
        put_varint(&mut out, self.next_flow as u64);
        put_signed(&mut out, self.drained);
        put_signed(&mut out, self.lost);
        put_varint(&mut out, self.ponds.len() as u64);
        for (&id, pond) in &self.ponds {
            put_varint(&mut out, id as u64);
            put_point(&mut out, pond.seed);
            put_signed(&mut out, pond.volume);
            put_signed(&mut out, pond.level);
            out.extend([pond.lod, pond.want, pond.sealed as u8, pond.unsettled as u8, pond.holds as u8]);
            put_signed(&mut out, pond.head);
            match pond.over {
                None => out.push(0),
                Some(Over::Sea) => out.push(1),
                Some(Over::At(at)) => {
                    out.push(2);
                    put_point(&mut out, at);
                }
            }
            pond.cells.encode(&mut out);
        }
        self.sea.encode(&mut out);
        // A search that is under way is started again: what it was for is still to do.
        let searching = match &self.flight {
            Some(Flight { kind: Kind::Sea(seeds), .. }) => seeds.as_slice(),
            _ => &[],
        };
        put_varint(&mut out, (self.frontier.len() + searching.len()) as u64);
        self.frontier.iter().chain(searching).for_each(|&p| put_point(&mut out, p));
        // Routes are found again; the water on them keeps its place in time.
        put_varint(&mut out, self.flows.len() as u64);
        for (&id, flow) in &self.flows {
            put_varint(&mut out, id as u64);
            put_point(&mut out, flow.source);
            put_signed(&mut out, flow.head);
            put_varint(&mut out, flow.fed);
            put_varint(&mut out, flow.lots.len() as u64);
            for lot in &flow.lots {
                put_varint(&mut out, lot.start);
                put_varint(&mut out, lot.ticks as u64);
                put_signed(&mut out, lot.volume);
                put_signed(&mut out, lot.delivered);
            }
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
        let small = |v: u64| u32::try_from(v).map_err(|_| "bad number");
        let mut water = Water {
            tick: get_varint(bytes, at)?,
            next_pond: small(get_varint(bytes, at)?)?,
            next_flow: small(get_varint(bytes, at)?)?,
            drained: get_signed(bytes, at)?,
            lost: get_signed(bytes, at)?,
            ..Self::default()
        };
        for _ in 0..get_varint(bytes, at)? {
            let id = small(get_varint(bytes, at)?)?;
            let (seed, volume, level) = (get_point(bytes, at)?, get_signed(bytes, at)?, get_signed(bytes, at)?);
            let (lod, want) = (get_u8(bytes, at)?, get_u8(bytes, at)?);
            let (sealed, unsettled, holds) =
                (get_u8(bytes, at)? != 0, get_u8(bytes, at)? != 0, get_u8(bytes, at)? != 0);
            let head = get_signed(bytes, at)?;
            if lod > MAX_LEVEL || want > MAX_LEVEL {
                return Err("bad level of detail".into());
            }
            let over = match get_u8(bytes, at)? {
                0 => None,
                1 => Some(Over::Sea),
                2 => Some(Over::At(get_point(bytes, at)?)),
                _ => return Err("bad overflow".into()),
            };
            let cells = Cells::decode(bytes, at)?;
            let pond =
                Pond { want, sealed, unsettled, over, holds, ..Pond::new(seed, volume, level, lod, cells, head) };
            water.ponds.insert(id, pond);
        }
        water.sea = Cells::decode(bytes, at)?;
        for _ in 0..get_varint(bytes, at)? {
            water.frontier.push_back(get_point(bytes, at)?);
        }
        for _ in 0..get_varint(bytes, at)? {
            let id = small(get_varint(bytes, at)?)?;
            let source = get_point(bytes, at)?;
            let head = get_signed(bytes, at)?;
            let fed = get_varint(bytes, at)?;
            let mut lots = VecDeque::new();
            for _ in 0..get_varint(bytes, at)? {
                let (start, ticks) = (get_varint(bytes, at)?, small(get_varint(bytes, at)?)?);
                let (volume, delivered) = (get_signed(bytes, at)?, get_signed(bytes, at)?);
                if ticks == 0 || delivered < 0 || delivered > volume || start > water.tick {
                    return Err("bad lot of water".into());
                }
                lots.push_back(Lot { start, ticks, volume, delivered });
            }
            water.flows.insert(id, Flow { source, head, route: None, lots, fed });
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
    use crate::chunk::Chunk;
    use crate::coords::{WorldParams, chunk_of, index_xyz};
    use crate::edit::{EditRules, Op, edit_volume};
    use crate::field::LocalField;
    use crate::log::EditLog;
    use crate::material::Material;
    use crate::worldgen::SEA_LEVEL;
    use std::sync::Arc;

    /// A world to pour water in. Searches are run the moment they are
    /// started, and their outcomes held back until they are asked for.
    struct World {
        generator: Generator,
        rules: EditRules,
        volume: Volume,
        log: EditLog,
        /// Terrain that is a rule instead of the generator's: true where solid.
        solid: Option<Box<dyn Fn(IVec3) -> bool>>,
        water: Water,
        found: Option<(u64, Outcome)>,
        tick: u64,
        poured: i64,
    }

    impl World {
        fn new(solid: Option<Box<dyn Fn(IVec3) -> bool>>) -> Self {
            let params = WorldParams::default();
            Self {
                generator: Generator::new(params),
                rules: EditRules::new(&params),
                volume: Volume::new(),
                log: EditLog::new(),
                solid,
                water: Water::new(),
                found: None,
                tick: 0,
                poured: 0,
            }
        }

        /// The generated world, with the finest level loaded around the ground at a column.
        fn around(x: i32, z: i32) -> (Self, IVec3) {
            let mut world = Self::new(None);
            let cell = world.generator.params().cell_size as f64;
            let ground =
                (world.generator.surface_height(x as f64 * cell, z as f64 * cell) as f64 / cell).floor() as i32;
            let here = IVec3::new(x, ground, z);
            let c = chunk_of(here);
            for cy in -2..=2 {
                for cz in -3..=3 {
                    for cx in -3..=3 {
                        let key = NodeKey::new(0, c + IVec3::new(cx, cy, cz));
                        let mut field = LocalField::block(key);
                        world.generator.fill_block(key, &mut field);
                        world.volume.insert(key.pos, field.interior());
                    }
                }
            }
            (world, here)
        }

        /// Terrain that is a rule, far above the sea.
        fn made(solid: impl Fn(IVec3) -> bool + 'static) -> Self {
            Self::new(Some(Box::new(solid)))
        }

        fn step(&mut self) {
            self.tick += 1;
            let (volume, log, solid, tick) = (&self.volume, &self.log, &self.solid, self.tick);
            let mut found = self.found.take();
            let terrain = |request: &Request| match solid {
                None => Terrain::gather(volume, log, request),
                Some(solid) => {
                    let mut chunks = Vec::new();
                    for i in 0..request.size.element_product() {
                        let (sx, sz) = (request.size.x, request.size.z);
                        let pos = request.origin + IVec3::new(i % sx, i / (sx * sz), i / sx % sz);
                        let mut chunk = Chunk::AIR;
                        for j in 0..CHUNK_SIZE.pow(3) {
                            let local =
                                IVec3::new(j % CHUNK_SIZE, j / CHUNK_SIZE / CHUNK_SIZE, j / CHUNK_SIZE % CHUNK_SIZE);
                            if solid(base_of(pos * CHUNK_SIZE + local, request.lod)) {
                                let dense = chunk.dense_mut();
                                dense.fill[index_xyz(local.x, local.y, local.z)] = 255;
                            }
                        }
                        chunks.push(Arc::new(chunk));
                    }
                    Some(Terrain::of_chunks(request, chunks))
                }
            };
            // What a search found is asked for exactly `DELAY` ticks after it was started.
            let outcome = || {
                let (started, outcome) = found.take().expect("no search was under way");
                assert_eq!(tick, started + DELAY);
                outcome
            };
            let job = self.water.step(outcome, terrain);
            assert!(found.is_none() || job.is_none());
            self.found = found;
            if let Some(job) = job {
                self.found = Some((self.tick, job.run(&self.generator, &self.rules)));
            }
            assert_eq!(self.total(), self.poured, "water was made or lost at tick {}", self.tick);
        }

        fn pour(&mut self, at: IVec3, volume: i64) {
            self.poured += volume;
            self.water.pour(at, volume, JOIN);
        }

        fn edit(&mut self, op: Op) {
            let outcome = edit_volume(&mut self.volume, &self.rules, &op);
            assert!(outcome.covered);
            self.log.push(&self.rules, &op);
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

        /// Tick until the water stops moving. Returns the ticks it took.
        fn rest(&mut self) -> u64 {
            let from = self.tick;
            while self.tick < from + 30_000 {
                self.step();
                if self.water.at_rest() {
                    return self.tick - from;
                }
            }
            let ponds: Vec<_> = self
                .ponds()
                .into_iter()
                .map(|(id, p)| {
                    (
                        id,
                        p.seed,
                        p.volume,
                        p.level,
                        p.lod,
                        p.want,
                        p.sealed,
                        p.unsettled,
                        p.over,
                        p.span,
                        p.ebb.is_some(),
                    )
                })
                .collect();
            panic!("water did not come to rest: {:?}\n{:?}\n{ponds:?}", self.water.stats(), self.water.flows);
        }

        /// Not solid, on the lattice of a level of detail.
        fn open(&self, p: IVec3, lod: u8) -> bool {
            match (&self.solid, lod) {
                (Some(solid), _) => !solid(base_of(p, lod)),
                (None, 0) => !self.volume.get(p).unwrap().solid(),
                (None, _) => unimplemented!(),
            }
        }

        /// Everything poured is in a pond, on its way, or accounted for.
        fn total(&self) -> i64 {
            let s = self.water.stats();
            s.held + s.moving + s.drained + s.lost
        }

        fn ponds(&self) -> Vec<(u32, Pond)> {
            self.water.ponds().map(|(id, p)| (id, p.clone())).collect()
        }

        fn only_pond(&self) -> (u32, Pond) {
            let ponds = self.ponds();
            let [pond] = ponds.as_slice() else { panic!("{} ponds: {:?}", ponds.len(), self.water.stats()) };
            pond.clone()
        }

        /// Check a pond against the terrain: its cells are open, joined up,
        /// below its level, and hold its volume.
        fn check(&self, pond: &Pond) {
            let (lod, level) = (pond.lod, pond.level >> pond.lod);
            let top = top_layer(level);
            let mut holds = 0;
            for p in pond.cells.iter() {
                assert!(self.open(p, lod), "pond cell {p} is solid");
                assert!(p.y <= top, "pond cell {p} above its level");
                holds += (level - bottom(p.y)).min(FP);
                // Nothing open and dry beside or below a wet cell, short of the surface layer.
                for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                    let q = p + d;
                    assert!(!self.open(q, lod) || pond.cells.contains(q), "dry cell {q} beside wet {p}");
                }
            }
            let surface = pond.cells.iter().filter(|p| p.y == top).count() as i64;
            // The level is rounded up to a whole unit, and the volume down to whole cells of the lattice.
            let volume = pond.volume >> (3 * lod);
            assert!(holds >= volume && holds - volume < surface.max(1), "{holds} for {volume}");
        }
    }

    #[test]
    fn a_pond_keeps_its_volume_through_digging_and_filling() {
        let (mut world, here) = World::around(0, 0);
        // A bowl two metres across in the ground, and water let go above it.
        let pit = here - IVec3::Y * 3;
        world.dig(pit, 8);
        assert!(world.water.stats() == WaterStats::default() && world.water.take_changed().is_empty());
        let poured = 300 * FP;
        world.pour(here + IVec3::Y * 6, poured);
        assert_eq!((world.water.stats().moving, world.water.stats().flows), (poured, 1));
        // It is on its way for a while: found a route, then fell to the bottom of the bowl.
        world.step();
        assert!(world.water.searching() && world.water.streams().is_empty());
        (0..DELAY).for_each(|_| world.step());
        let streams = world.water.streams();
        let [stream] = streams.as_slice() else { panic!("{} streams", streams.len()) };
        assert_eq!(stream.path[0], here + IVec3::Y * 6);
        assert!(stream.to > stream.from && stream.to < *stream.along.last().unwrap());
        assert_eq!((world.water.ponds().count(), world.water.stats().moving), (0, poured));
        let took = world.rest();
        assert!(took > 20, "at rest after {took} ticks");
        let (id, pond) = world.only_pond();
        world.check(&pond);
        assert_eq!((pond.volume, pond.lod), (poured, 0));
        assert!(pond.cells.contains(pond.seed) && (pond.seed - pit).abs().max_element() <= 9, "seed {}", pond.seed);
        assert!(pond.level < bottom(here.y - 1), "the pond is not below the rim");
        assert_eq!(world.water.holder(&world.generator, pond.seed), Some(BodyId::Pond(id)));
        let drawn = world.water.take_changed();
        assert!(!drawn.is_empty());
        let (lo, hi) = pond.cells.bounds().unwrap();
        assert!(drawn.iter().any(|(a, b)| a.x < lo.x && b.x > hi.x && a.y <= anchor(pond.level)));

        // A second bowl cut into the side of the first: the same water lies lower,
        // and for a while it is still drawn over the ground it left.
        world.dig(pit + IVec3::new(9, 0, 0), 8);
        world.step();
        assert!(world.water.searching() && world.only_pond().1.level == pond.level);
        (0..DELAY).for_each(|_| world.step());
        let lower = world.only_pond().1;
        world.check(&lower);
        assert_eq!(lower.volume, poured);
        assert!(lower.level < pond.level - FP / 4, "{} then {}", pond.level, lower.level);
        assert!(lower.cells.len() > pond.cells.len());
        assert!(!world.water.take_changed().is_empty());
        let key = NodeKey::new(0, chunk_of(IVec3::new(pit.x, anchor(lower.level), pit.z)));
        assert_eq!(world.water.node_water(key).surfaces.len(), 2);
        assert!(world.rest() >= EBB_TICKS - DELAY && !world.water.take_changed().is_empty());
        assert_eq!(world.water.node_water(key).surfaces.len(), 1);

        // Sand dropped into it: the water is pushed up.
        world.put(pit + IVec3::new(0, -4, 0), 5);
        world.rest();
        let higher = world.only_pond().1;
        world.check(&higher);
        assert_eq!(higher.volume, poured);
        assert!(higher.level > lower.level + FP / 8, "{} then {}", lower.level, higher.level);

        // It all survives a save.
        let bytes = world.water.encode();
        let back = Water::decode(&bytes).unwrap();
        assert!(
            back.ponds == world.water.ponds && back.next_pond == world.water.next_pond && back.tick == world.water.tick
        );
        assert_eq!(back.encode(), bytes);
        assert!(Water::decode(&bytes[..bytes.len() - 3]).is_err());

        // Nothing moves when nothing changes.
        assert!(!world.water.take_changed().is_empty() && world.water.take_unsaved());
        world.step();
        assert!(world.water.take_changed().is_empty() && !world.water.take_unsaved() && !world.water.searching());
    }

    #[test]
    fn more_than_a_hollow_holds_runs_on_downhill() {
        let (mut world, here) = World::around(0, 0);
        let pit = here - IVec3::Y * 2;
        world.dig(pit, 6);
        let poured = 4000 * FP;
        world.pour(here + IVec3::Y * 4, poured);
        world.rest();
        // The bowl is full to its rim, and the rest went elsewhere.
        let ponds = world.ponds();
        let (_, first) = ponds.iter().find(|(_, p)| (p.seed - pit).abs().max_element() <= 6).unwrap();
        world.check(first);
        assert!(first.volume < poured / 4 && first.volume > 100 * FP, "{}", first.volume);
        for (_, pond) in &ponds {
            world.check(pond);
            assert!(pond.volume >= MIN_POND);
        }
        // More water changes nothing in a full bowl: it goes straight on, with no new search for the bowl.
        world.pour(here + IVec3::Y * 4, 500 * FP);
        world.rest();
        let ponds = world.ponds();
        let (_, after) = ponds.iter().find(|(_, p)| p.seed == first.seed).unwrap();
        assert_eq!((after.volume, after.level), (first.volume, first.level));
        assert_eq!(world.total(), poured + 500 * FP);
    }

    #[test]
    fn a_hollow_wider_than_the_box_is_one_pond_on_a_coarser_lattice() {
        // A pit 19 m across with a floor of bowls and ridges, and a cavern of
        // the same 36 m across, dug level into the hillside.
        for (across, lod) in [(5, 0), (10, 1)] {
            let (mut world, here) = World::around(16, if lod == 0 { 16 } else { 300 });
            let bowls: Vec<IVec3> = (0..across * across)
                .map(|i| IVec3::new((i % across - across / 2) * 14, 0, (i / across - across / 2) * 14))
                .collect();
            let ground =
                |p: IVec3| (world.generator.surface_height(p.x as f64 * 0.25, p.z as f64 * 0.25) * 4.0).floor() as i32;
            let lowest = bowls.iter().map(|&b| ground(here + b)).min().unwrap();
            let floor = if lod == 0 { here.y - 6 } else { lowest - 8 };
            assert!(floor > 20, "floor at {floor}");
            for bowl in bowls {
                world.dig(IVec3::new(here.x, floor, here.z) + bowl, 10);
            }
            let poured = 160 * (across * across) as i64 * FP;
            world.pour(IVec3::new(here.x, floor + 8, here.z), poured);
            world.rest();
            // All of it is held, nearly all in one pond that covers the bowls.
            assert_eq!(world.water.stats().held, poured, "{:?}", world.water.stats());
            let (_, pond) = world.ponds().into_iter().max_by_key(|(_, p)| p.volume).unwrap();
            let (lo, hi) = pond.span.unwrap();
            let wide = hi.x - lo.x > 14 * (across - 1) && hi.z - lo.z > 14 * (across - 1);
            assert!(
                pond.volume > poured * 9 / 10 && wide && pond.lod == lod,
                "{} in {lo} to {hi} at {}",
                pond.volume,
                pond.lod
            );
            if lod == 0 {
                world.check(&pond);
            }
        }
    }

    #[test]
    fn two_ponds_that_meet_become_one() {
        let (mut world, here) = World::around(0, 0);
        let (a, b) = (here - IVec3::Y * 5, here + IVec3::new(18, -5, 0));
        let (va, vb) = (100 * FP, 80 * FP);
        world.dig(a, 7);
        world.dig(b, 7);
        world.pour(a + IVec3::Y * 8, va);
        world.pour(b + IVec3::Y * 8, vb);
        world.rest();
        assert_eq!(world.water.ponds().count(), 2);
        // A trench between them, below both surfaces.
        for x in (a.x..=b.x).step_by(3) {
            world.dig(IVec3::new(x, a.y - 2, a.z), 4);
        }
        world.rest();
        let pond = world.only_pond().1;
        world.check(&pond);
        assert_eq!(pond.volume, va + vb);
    }

    /// Ground whose top is at `GROUND`, far above the sea, with the cells for which `open` is true dug out of it.
    const GROUND: i32 = 4096;
    fn ground(open: impl Fn(IVec3) -> bool + 'static) -> World {
        World::made(move |p| p.y < GROUND && !open(p))
    }

    #[test]
    fn a_wall_across_a_pond_makes_two_ponds_with_their_share_each() {
        use std::cell::Cell;
        use std::rc::Rc;
        // A trench 40 by 10 and 8 deep, and a wall that can be put across it a quarter of the way along.
        let wall = Rc::new(Cell::new(false));
        let built = wall.clone();
        let trench = |p: IVec3| p.y >= GROUND - 8 && (0..40).contains(&p.x) && (0..10).contains(&p.z);
        let mut world = ground(move |p| trench(p) && !(built.get() && p.x == 10));
        let poured = 400 * FP * 3 + 100 * FP;
        world.pour(IVec3::new(20, GROUND + 4, 5), poured);
        world.rest();
        let (id, whole) = world.only_pond();
        assert_eq!((whole.volume, whole.level), (poured, bottom(GROUND - 5) + FP / 4));

        wall.set(true);
        world.water.edited(&Volume::new(), &world.generator, IVec3::new(10, GROUND - 8, 0), IVec3::new(10, GROUND, 9));
        world.rest();
        let ponds = world.ponds();
        let [(a, west), (b, east)] = ponds.as_slice() else { panic!("{} ponds", ponds.len()) };
        // Each holds its share of all the water, by what was over its part of the
        // floor, and stands a little higher for what the wall takes up.
        assert!(*a == id && *b != id);
        assert_eq!((west.volume, east.volume), (poured * 100 / 390 + 1, poured * 290 / 390));
        assert!(west.level == east.level && (1..FP / 4).contains(&(west.level - whole.level)));
        assert!(west.cells.iter().all(|p| p.x < 10) && east.cells.iter().all(|p| p.x > 10));
        world.check(west);
        world.check(east);

        // Water added on one side raises that side only.
        world.pour(IVec3::new(5, GROUND + 4, 5), 100 * FP);
        world.rest();
        let ponds = world.ponds();
        assert_eq!((ponds[0].1.level, ponds[1].1.level), (west.level + FP, east.level));

        // The wall taken away, they are one pond again, with all the water.
        wall.set(false);
        world.water.edited(&Volume::new(), &world.generator, IVec3::new(10, GROUND - 8, 0), IVec3::new(10, GROUND, 9));
        world.rest();
        let (_, pond) = world.only_pond();
        world.check(&pond);
        assert_eq!(pond.volume, poured + 100 * FP);
    }

    #[test]
    fn no_water_is_lost_in_a_hollow_wider_than_any_box_of_the_finest_lattice() {
        // A flat-floored basin 75 m across.
        let mut world = ground(|p| p.y >= GROUND - 64 && p.x.abs() < 150 && p.z.abs() < 150);
        let poured = 300_000 * FP;
        world.pour(IVec3::new(3, GROUND, -5), poured);
        world.rest();
        let (_, pond) = world.only_pond();
        world.check(&pond);
        let stats = world.water.stats();
        assert_eq!((pond.volume, stats.held, stats.lost, stats.drained), (poured, poured, 0, 0));
        // It took the lattice two levels up to see all of it, where the basin is 75 cells across.
        assert_eq!((pond.lod, pond.cells.len()), (2, 75 * 75));
        let (lo, hi) = pond.span.unwrap();
        assert!(lo.x <= -148 && hi.x >= 148 && lo.z <= -148 && hi.z >= 148);
        // 300,000 cells over 75 by 75 cells of 16: a little over three cells deep
        // on that lattice, whose floor is half a cell of its own below its lowest open point.
        let depth = pond.level - (bottom((GROUND - 64) >> 2) << 2);
        assert!((3 * FP..4 * FP).contains(&depth), "depth {depth}");
    }

    #[test]
    fn no_water_is_lost_in_water_deeper_than_a_box() {
        // A shaft 2 m square and 50 m deep.
        let mut world = ground(|p| p.y >= GROUND - 200 && (-4..4).contains(&p.x) && (-4..4).contains(&p.z));
        let poured = 64 * 150 * FP;
        world.pour(IVec3::new(0, GROUND + 2, 0), poured);
        world.rest();
        let (_, pond) = world.only_pond();
        world.check(&pond);
        assert_eq!((pond.volume, world.water.stats().lost), (poured, 0));
        // 150 cells deep, as near as the lattice it ended on can tell.
        let depth = pond.level - bottom(GROUND - 200);
        assert!(pond.lod >= 1 && (depth - 150 * FP).abs() <= (FP << pond.lod), "depth {depth} at {}", pond.lod);
    }

    #[test]
    fn water_runs_as_far_downhill_as_the_hill_goes() {
        // Steps down to the east for 600 m, 2 m wide, to a pit at the bottom.
        let stairs = |p: IVec3| (0..2400).contains(&p.x) && (0..8).contains(&p.z) && p.y >= GROUND - p.x / 4;
        let pit = |p: IVec3| (2400..2420).contains(&p.x) && (0..8).contains(&p.z) && p.y >= GROUND - 640;
        let mut world = ground(move |p| stairs(p) || pit(p));
        let poured = 2000 * FP;
        world.pour(IVec3::new(2, GROUND + 3, 4), poured);
        // On its way it is drawn, and it does not get there at once.
        (0..600).for_each(|_| world.step());
        let stats = world.water.stats();
        assert!(stats.moving > poured * 9 / 10 && stats.held < poured / 10, "{stats:?}");
        let streams = world.water.streams();
        assert!(!streams.is_empty() && streams.iter().all(|s| s.path[0].x < 600 && s.rate > 0.0), "{streams:?}");
        let took = world.rest();
        let (_, pond) = world.only_pond();
        world.check(&pond);
        assert_eq!((pond.volume, world.water.stats().lost), (poured, 0));
        assert!(pond.seed.x >= 2400 && pond.seed.y == GROUND - 640, "seed {}", pond.seed);
        // 3,000 cells of way at the speed of water, with the searches along
        // it, less the steps on which it rested and was passed straight on.
        let way = (3000 * FP / SPEED) as u64;
        assert!((way * 3 / 4..way * 3 / 2).contains(&(took + 600)), "{took} ticks");
    }

    #[test]
    fn a_pond_that_is_filled_in_is_pushed_out_not_lost() {
        use std::cell::Cell;
        use std::rc::Rc;
        // A pit, and beside it a lower one.
        let filled = Rc::new(Cell::new(false));
        let now = filled.clone();
        let first = |p: IVec3| p.y >= GROUND - 6 && (0..8).contains(&p.x) && (0..8).contains(&p.z);
        let second = |p: IVec3| p.y >= GROUND - 12 && (12..24).contains(&p.x) && (0..8).contains(&p.z);
        let mut world = ground(move |p| (first(p) && !now.get()) || second(p));
        let poured = 64 * 4 * FP;
        world.pour(IVec3::new(4, GROUND + 2, 4), poured);
        world.rest();
        assert!(first(world.only_pond().1.seed));
        filled.set(true);
        world.water.edited(
            &Volume::new(),
            &world.generator,
            IVec3::new(0, GROUND - 6, 0),
            IVec3::new(7, GROUND - 1, 7),
        );
        world.rest();
        // It came up through the fill and ran into the other pit.
        let (_, pond) = world.only_pond();
        world.check(&pond);
        assert_eq!((pond.volume, world.water.stats().lost), (poured, 0));
        assert!(pond.cells.iter().all(|p| p.x >= 12));
    }

    #[test]
    fn only_a_puddle_with_nothing_running_into_it_soaks_away() {
        let mut world = ground(|p| p.y >= GROUND - 4 && (0..6).contains(&p.x) && (0..6).contains(&p.z));
        // A trickle too small to keep, for as long as it runs.
        for _ in 0..200 {
            world.pour(IVec3::new(3, GROUND + 2, 3), 1);
            world.step();
        }
        let stats = world.water.stats();
        assert_eq!((stats.ponds, stats.lost), (1, 0));
        world.rest();
        // It stopped at less than a puddle: gone, and counted.
        assert_eq!(world.water.stats(), WaterStats { lost: 200, ..WaterStats::default() });
        // The same trickle for longer leaves a pond.
        for _ in 0..3000 {
            world.pour(IVec3::new(3, GROUND + 2, 3), 1);
            world.step();
        }
        world.rest();
        assert_eq!((world.only_pond().1.volume, world.water.stats().lost), (3000, 200));
    }

    #[test]
    fn a_save_in_the_middle_of_things_loses_nothing() {
        let (mut world, here) = World::around(0, 0);
        let pit = here - IVec3::Y * 2;
        world.dig(pit, 6);
        world.pour(here + IVec3::Y * 4, 4000 * FP);
        let mut saves = Vec::new();
        for tick in 0..120 {
            world.step();
            if tick % 7 == 0 {
                saves.push(world.water.encode());
            }
        }
        world.rest();
        let end = world.water.stats();
        for bytes in saves {
            let mut again = World::around(0, 0).0;
            again.dig(pit, 6);
            again.water = Water::decode(&bytes).unwrap();
            again.poured = 4000 * FP;
            again.rest();
            // The same water in the same hollows; when it got there may differ.
            let stats = again.water.stats();
            assert_eq!(
                (stats.ponds, stats.held, stats.drained, stats.lost),
                (end.ponds, end.held, end.drained, end.lost)
            );
            let levels = |w: &World| w.ponds().iter().map(|(_, p)| (p.seed, p.level, p.volume)).collect::<Vec<_>>();
            let (mut a, mut b) = (levels(&world), levels(&again));
            a.sort_by_key(|p| p.0.to_array());
            b.sort_by_key(|p| p.0.to_array());
            assert_eq!(a, b);
        }
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
        assert!(world.open(pit, 0) && world.water.holder(&world.generator, pit).is_none());
        assert_eq!(world.water.stats(), WaterStats::default());

        // A channel from it to the sea. The water comes along it a stretch at a time.
        for x in (wet - 10..=wet + 2).step_by(3) {
            world.dig(IVec3::new(x, -2, 0), 4);
        }
        let mut counts = Vec::new();
        while !world.water.at_rest() {
            world.step();
            let cells = world.water.stats().sea_cells;
            if counts.last() != Some(&cells) {
                counts.push(cells);
                assert!(world.water.sea.iter().all(|p| p.x > wet - 3 - SEA_REACH as i32 * counts.len() as i32));
            }
        }
        assert!(counts.len() >= 3, "the sea came in in {} steps", counts.len());
        let stats = world.water.stats();
        assert!(stats.sea_cells > 500 && stats.sea_pending == 0, "{stats:?}");
        assert_eq!(world.water.holder(&world.generator, pit), Some(BodyId::Ocean));
        for p in world.water.sea.iter() {
            assert!(world.open(p, 0) && bottom(p.y) < SEA, "sea cell {p}");
            for d in [IVec3::NEG_Y, IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                let q = p + d;
                assert!(
                    !world.open(q, 0) || world.water.holder(&world.generator, q).is_some(),
                    "dry {q} beside sea {p}"
                );
            }
        }
        // The surface over the pit is handed to the node that draws it.
        let key = NodeKey::new(0, chunk_of(IVec3::new(pit.x, anchor(SEA), pit.z)));
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
        assert!(world.water.sea.iter().all(|p| world.open(p, 0)));
        let bytes = world.water.encode();
        assert!(Water::decode(&bytes).unwrap().sea == world.water.sea);
    }

    #[test]
    fn a_pond_dug_through_to_the_sea_joins_it() {
        let wet = shore();
        let (mut world, _) = World::around(wet - 12, 0);
        // Below sea level: once joined, it is sea.
        let low = IVec3::new(wet - 14, -4, 0);
        world.dig(low, 6);
        world.pour(low + IVec3::Y * 2, 60 * FP);
        world.rest();
        assert!(world.only_pond().1.level < SEA);
        for x in (wet - 10..=wet + 2).step_by(3) {
            world.dig(IVec3::new(x, -2, 0), 4);
        }
        world.rest();
        assert_eq!((world.water.ponds().count(), world.water.stats().drained), (0, 60 * FP));
        assert_eq!(world.water.holder(&world.generator, low), Some(BodyId::Ocean));
    }
}
