//! Where water goes on a lattice: filling a hollow from its lowest point,
//! spreading at a fixed level, and running downhill.
//!
//! Part of the bit-exact contract: integers only. Everything here works in a
//! [`Grid`], a copy of a box of the world small enough to search in one go;
//! what happens beyond the box is for the caller to follow up. The lattice
//! may be that of any level of detail: lengths and volumes are in its cells.
//!
//! A cell is a lattice sample that is not solid. The cell at lattice height
//! `y` holds the slab of water from half a cell below it to half a cell
//! above, so a full cell is `FP` units of volume and the slab starts at
//! [`bottom`]. Water reaches a cell once the level is above that.

use std::collections::VecDeque;

use glam::IVec3;

use std::sync::Arc;

use crate::chunk::{Chunk, Volume};
use crate::coords::{CHUNK_SIZE, ChunkPos, SOLID, index_xyz};
use crate::edit::FP;

use super::BodyId;

/// Level at which the cell at lattice height `y` starts to hold water.
#[inline]
pub fn bottom(y: i32) -> i64 {
    y as i64 * FP - FP / 2
}

/// The highest lattice layer with water in it at `level`.
#[inline]
pub fn top_layer(level: i64) -> i32 {
    (level + FP / 2 - 1).div_euclid(FP) as i32
}

/// The lattice layer that says which of a stack of nodes draws a surface at
/// `level`: the one under the highest with water in it, so that the surface
/// is drawn with the ground seen through it, at the same level of detail.
#[inline]
pub fn anchor(level: i64) -> i32 {
    top_layer(level) - 1
}

const ROCK: u8 = 0;
const FREE: u8 = 1;
/// Cells of value `HELD + i` hold water of `owners[i]`.
const HELD: u8 = 2;

/// Cells searched before a flood gives up: the box is too fine for the water.
const BUDGET: usize = 250_000;

const NEIGHBOURS: [IVec3; 6] = [IVec3::NEG_Y, IVec3::NEG_X, IVec3::X, IVec3::NEG_Z, IVec3::Z, IVec3::Y];

/// A box of the finest lattice: which cells are solid, free, or already
/// under some body's water.
pub struct Grid {
    pub origin: IVec3,
    pub dims: IVec3,
    cells: Vec<u8>,
    owners: Vec<BodyId>,
}

impl Grid {
    /// A box with the cells for which `solid` is false free, asked in the
    /// order of [`index`](Self::index): X fastest, then Z, then Y.
    pub fn from_solid(origin: IVec3, dims: IVec3, solid: impl Iterator<Item = bool>) -> Self {
        let cells: Vec<u8> = solid.map(|s| if s { ROCK } else { FREE }).collect();
        assert_eq!(cells.len(), (dims.x * dims.y * dims.z) as usize);
        Self { origin, dims, cells, owners: Vec::new() }
    }

    /// The box of `size` chunks from chunk `origin`, with `chunks` in order of Y, then Z, then X.
    pub fn from_chunks(origin: ChunkPos, size: IVec3, chunks: &[Arc<Chunk>]) -> Self {
        let dims = size * CHUNK_SIZE;
        let mut cells = vec![ROCK; (dims.x * dims.y * dims.z) as usize];
        let n = CHUNK_SIZE as usize;
        for (i, chunk) in chunks.iter().enumerate() {
            let i = i as i32;
            let c = IVec3::new(i % size.x, i / (size.x * size.z), i / size.x % size.z) * CHUNK_SIZE;
            if matches!(**chunk, Chunk::Uniform(s) if s.solid()) {
                continue;
            }
            for y in 0..n {
                for z in 0..n {
                    let row =
                        (((c.y as usize + y) * dims.z as usize + c.z as usize + z) * dims.x as usize) + c.x as usize;
                    for x in 0..n {
                        if !chunk.get(index_xyz(x as i32, y as i32, z as i32)).solid() {
                            cells[row + x] = FREE;
                        }
                    }
                }
            }
        }
        Self { origin: origin * CHUNK_SIZE, dims, cells, owners: Vec::new() }
    }

    /// Copy a box out of the loaded world. `None` if part of it is not loaded.
    pub fn new(volume: &Volume, origin: IVec3, dims: IVec3) -> Option<Self> {
        if !volume.covers(origin, dims) {
            return None;
        }
        let field = volume.copy_region(origin, dims);
        let cells = field.fill.iter().map(|&f| if f >= SOLID { ROCK } else { FREE }).collect();
        Some(Self { origin, dims, cells, owners: Vec::new() })
    }

    #[inline]
    fn inside(&self, p: IVec3) -> bool {
        let l = p - self.origin;
        l.cmpge(IVec3::ZERO).all() && l.cmplt(self.dims).all()
    }

    #[inline]
    pub fn index(&self, p: IVec3) -> usize {
        let l = p - self.origin;
        ((l.y * self.dims.z + l.z) * self.dims.x + l.x) as usize
    }

    pub fn point(&self, i: usize) -> IVec3 {
        let (i, dx, dz) = (i as i32, self.dims.x, self.dims.z);
        self.origin + IVec3::new(i % dx, i / (dx * dz), i / dx % dz)
    }

    /// Lattice height of every cell in the box, lowest to highest.
    pub fn layers(&self) -> std::ops::Range<i32> {
        self.origin.y..self.origin.y + self.dims.y
    }

    pub fn contains(&self, p: IVec3) -> bool {
        self.inside(p)
    }

    /// Not solid. Cells outside the box are not free.
    pub fn is_open(&self, p: IVec3) -> bool {
        self.inside(p) && self.cells[self.index(p)] != ROCK
    }

    /// Not solid and under nobody's water.
    pub fn is_free(&self, p: IVec3) -> bool {
        self.inside(p) && self.cells[self.index(p)] == FREE
    }

    /// The body whose water is at `p`, if any.
    pub fn holder(&self, p: IVec3) -> Option<BodyId> {
        if !self.inside(p) {
            return None;
        }
        self.cells[self.index(p)].checked_sub(HELD).map(|i| self.owners[i as usize])
    }

    /// Record that a free cell is under `body`'s water. Solid cells and cells
    /// outside the box are left alone.
    pub fn hold(&mut self, p: IVec3, body: BodyId) {
        if !self.inside(p) {
            return;
        }
        let i = self.index(p);
        if self.cells[i] != FREE {
            return;
        }
        let owner = self.owners.iter().position(|&b| b == body).unwrap_or_else(|| {
            self.owners.push(body);
            self.owners.len() - 1
        });
        // Past 254 bodies in one box the rest are treated as one with the last.
        self.cells[i] = HELD + owner.min((u8::MAX - HELD) as usize) as u8;
    }
}

/// How filling a hollow ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filled {
    /// The whole volume fits, with its surface at `level`.
    Settled { level: i64 },
    /// Full to a rim at `level` holding `kept`; the rest runs over it,
    /// towards `at` if there is lower ground to run to.
    Spilled { level: i64, kept: i64, at: Option<IVec3> },
    /// On reaching `level`, holding `kept` below it, the water met `with`'s.
    Met { level: i64, kept: i64, with: BodyId },
}

/// Fill the hollow around `seed` with `volume` of water, lowest cells first.
/// Returns the cells that end up wet, how it ended, and whether the edge of
/// the box had a part in that, so that a wider box might find otherwise.
///
/// The search goes up one layer at a time. Reaching layer `k` opens every
/// cell that can be got to without going above it, including lower ground
/// beyond a rim. A hollow there fills before the water rises further; if
/// there is not enough to fill it, the water this side of the rim stays
/// level with it and the rest runs over, to be another body. Ground that
/// goes on down past the box takes everything above the rim, unless the box
/// is `sealed`: then its sides are walls.
pub fn fill(grid: &Grid, seed: IVec3, volume: i64, sealed: bool) -> (Vec<IVec3>, Filled, bool) {
    debug_assert!(grid.is_free(seed) && volume > 0);
    // For each cell found, one more than the index of the layer that opens it.
    let mut seen = vec![0u8; grid.cells.len()];
    let mut order: Vec<u32> = Vec::new();
    // Cells waiting for the level to reach their layer.
    let mut waiting: Vec<Vec<u32>> = vec![Vec::new(); grid.dims.y as usize];
    // Scratch for telling the hollows beyond a rim apart.
    let mut grouped = vec![false; grid.cells.len()];
    let mut group: Vec<u32> = Vec::new();
    let start = grid.index(seed);
    seen[start] = (seed.y - grid.origin.y + 1) as u8;
    waiting[(seed.y - grid.origin.y) as usize].push(start as u32);
    let point = |i: u32| grid.point(i as usize);
    // Cells that are wet whatever happens next: every one is full.
    let mut wet: Vec<IVec3> = Vec::new();
    let mut edge = false;

    for k in seed.y..grid.origin.y + grid.dims.y {
        // Everything wet so far is below this layer, and full.
        let (level, kept) = (bottom(k), FP * wet.len() as i64);
        let phase = (k - grid.origin.y + 1) as u8;
        let opened = order.len();
        order.append(&mut waiting[(k - grid.origin.y) as usize]);
        if order.len() == opened {
            // A closed space, full to its roof.
            return (wet, Filled::Spilled { level, kept, at: None }, edge);
        }
        let mut met = None;
        let mut next = opened;
        while next < order.len() {
            let p = point(order[next]);
            next += 1;
            for d in NEIGHBOURS {
                let q = p + d;
                if !grid.inside(q) {
                    edge |= q.y <= k;
                    continue;
                }
                let j = grid.index(q);
                match grid.cells[j] {
                    ROCK => {}
                    FREE if seen[j] != 0 => {}
                    FREE if q.y <= k => {
                        seen[j] = phase;
                        order.push(j as u32);
                    }
                    FREE => {
                        seen[j] = phase + 1;
                        waiting[(q.y - grid.origin.y) as usize].push(j as u32);
                    }
                    held => met = met.or(Some(grid.owners[(held - HELD) as usize])),
                }
            }
            if order.len() > BUDGET {
                return (wet, Filled::Spilled { level, kept, at: None }, true);
            }
        }
        if let Some(with) = met {
            return (wet, Filled::Met { level, kept, with }, edge);
        }

        // Lower ground beyond a rim, one connected hollow at a time, nearest
        // first. One that reaches the edge of the box is a way out.
        let mut way_out = None;
        for &first in &order[opened..] {
            if point(first).y >= k || std::mem::replace(&mut grouped[first as usize], true) {
                continue;
            }
            group.clear();
            group.push(first);
            let (mut open, mut next) = (false, 0);
            while next < group.len() {
                let p = point(group[next]);
                next += 1;
                for d in NEIGHBOURS {
                    let q = p + d;
                    if q.y >= k {
                        continue;
                    }
                    if !grid.inside(q) {
                        open |= !sealed;
                        continue;
                    }
                    let j = grid.index(q);
                    if seen[j] == phase && !std::mem::replace(&mut grouped[j], true) {
                        group.push(j as u32);
                    }
                }
            }
            let rest = volume - FP * wet.len() as i64;
            if open {
                way_out = way_out.or(Some(point(first)));
            } else if rest < FP * group.len() as i64 {
                // Not enough to fill it: the water parts at the rim.
                return (wet, Filled::Spilled { level, kept: volume - rest, at: Some(point(first)) }, edge);
            } else {
                wet.extend(group.iter().map(|&i| point(i)));
            }
        }
        if way_out.is_some() {
            let kept = FP * wet.len() as i64;
            return (wet, Filled::Spilled { level, kept, at: way_out }, edge);
        }

        // The layer itself. Where it runs past the box, the edge of the box
        // stands in for a shore.
        let flat: Vec<IVec3> = order[opened..].iter().map(|&i| point(i)).filter(|p| p.y == k).collect();
        let rest = volume - FP * wet.len() as i64;
        let n = flat.len() as i64;
        wet.extend(flat);
        if rest <= FP * n {
            return (wet, Filled::Settled { level: level + (rest + n - 1) / n }, edge);
        }
    }
    // Still rising at the top of the box.
    let (level, kept) = (bottom(grid.origin.y + grid.dims.y), FP * wet.len() as i64);
    (wet, Filled::Spilled { level, kept, at: None }, true)
}

/// What spreading at a fixed level found.
#[derive(Default, Debug)]
pub struct Spread {
    /// Cells newly under water.
    pub cells: Vec<IVec3>,
    /// Cells to carry on from: beyond the box, or not reached within the budget.
    pub open: Vec<IVec3>,
    /// Other bodies' water it ran into, and where.
    pub met: Vec<(BodyId, IVec3)>,
}

/// Flood from `seeds` through the free cells below `level`: what an opening
/// lets in from a body too large to notice the loss. It goes `reach` cells
/// from the seeds and no further, so that the water is seen to advance.
pub fn spread(grid: &Grid, seeds: &[IVec3], level: i64, body: BodyId, reach: u32) -> Spread {
    let mut out = Spread::default();
    let mut seen = vec![false; grid.cells.len()];
    let mut queue: VecDeque<(IVec3, u32)> = VecDeque::new();
    for &s in seeds {
        if grid.is_free(s) && bottom(s.y) < level && !std::mem::replace(&mut seen[grid.index(s)], true) {
            queue.push_back((s, 0));
        }
    }
    while let Some((p, steps)) = queue.pop_front() {
        if steps > reach {
            out.open.push(p);
            continue;
        }
        out.cells.push(p);
        for d in NEIGHBOURS {
            let q = p + d;
            if bottom(q.y) >= level {
                continue;
            }
            if !grid.inside(q) {
                out.open.push(q);
                continue;
            }
            let j = grid.index(q);
            match grid.cells[j] {
                ROCK => {}
                FREE => {
                    if !std::mem::replace(&mut seen[j], true) {
                        queue.push_back((q, steps + 1));
                    }
                }
                held => {
                    let with = grid.owners[(held - HELD) as usize];
                    if with != body && !std::mem::replace(&mut seen[j], true) {
                        out.met.push((with, q));
                    }
                }
            }
        }
    }
    out.open.sort_unstable_by_key(|p| (p.y, p.z, p.x));
    out.open.dedup();
    out
}

/// Where water let go at a point ends up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run {
    /// At rest, with nowhere lower to go that could be seen from there.
    Rest(IVec3),
    /// Into another body's water.
    Into(BodyId),
    /// Still falling at this cell, the last of the box on its way.
    Out(IVec3),
}

/// Follow water downhill from `start`: straight down while it can fall, and
/// along the ground to the nearest drop when it cannot. Returns the cells
/// where its way turns, first to last, and how it ends. Level ground that
/// runs out of the box is a place to rest: whether it goes on down beyond
/// is for whoever fills it to find out, in a box of their own.
pub fn run_down(grid: &Grid, start: IVec3) -> (Vec<IVec3>, Run) {
    let mut p = start;
    // Let go inside the ground, it is pushed up to the first open cell.
    while grid.inside(p) && !grid.is_open(p) {
        p.y += 1;
    }
    if !grid.inside(p) {
        return (vec![start], Run::Out(p));
    }
    /// Cells of level ground searched for a way down before calling it the bottom.
    const LEVEL_GROUND: usize = 1024;
    let mut path = vec![p];
    // For each cell of level ground reached, one more than the index of the cell it was reached from.
    let mut from = vec![0u32; grid.cells.len()];
    let mut seen = vec![0u32; grid.cells.len()];
    let mut search = 0u32;
    let mut queue: VecDeque<IVec3> = VecDeque::new();
    // The way along the ground from the start of a search to `to`.
    let walk = |path: &mut Vec<IVec3>, from: &[u32], first: IVec3, to: IVec3| {
        let at = path.len();
        let mut c = to;
        while c != first {
            path.push(c);
            c = grid.point(from[grid.index(c)] as usize - 1);
        }
        path[at..].reverse();
    };
    let end = 'falling: loop {
        loop {
            if let Some(body) = grid.holder(p) {
                path.push(p);
                break 'falling Run::Into(body);
            }
            let under = p - IVec3::Y;
            if !grid.inside(under) {
                path.push(p);
                break 'falling Run::Out(p);
            }
            if !grid.is_open(under) {
                break;
            }
            p = under;
        }
        path.push(p);
        // On the ground: look along this layer for the nearest way down.
        search += 1;
        queue.clear();
        queue.push_back(p);
        seen[grid.index(p)] = search;
        let mut visited = 0;
        while let Some(c) = queue.pop_front() {
            visited += 1;
            if visited > LEVEL_GROUND {
                break;
            }
            for d in [IVec3::NEG_X, IVec3::X, IVec3::NEG_Z, IVec3::Z] {
                let q = c + d;
                if !grid.inside(q) {
                    continue;
                }
                let j = grid.index(q);
                if seen[j] == search || grid.cells[j] == ROCK {
                    continue;
                }
                seen[j] = search;
                from[j] = grid.index(c) as u32 + 1;
                if let Some(body) = grid.holder(q) {
                    walk(&mut path, &from, p, q);
                    break 'falling Run::Into(body);
                }
                let under = q - IVec3::Y;
                if grid.is_open(under) {
                    walk(&mut path, &from, p, q);
                    p = under;
                    continue 'falling;
                }
                queue.push_back(q);
            }
        }
        break Run::Rest(p);
    };
    // Keep the turns only.
    let mut turns: Vec<IVec3> = Vec::with_capacity(path.len());
    for q in path {
        let n = turns.len();
        if turns.last() == Some(&q) {
            continue;
        }
        if n >= 2 && (turns[n - 1] - turns[n - 2]).signum() == (q - turns[n - 1]).signum() {
            turns[n - 1] = q;
        } else {
            turns.push(q);
        }
    }
    (turns, end)
}

/// A piece of what was one body's water, after the ground changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// Its lowest cell.
    pub seed: IVec3,
    /// The water it held at the old level.
    pub content: i64,
    pub cells: Vec<IVec3>,
}

/// The pieces that the cells `was`, which were under one body's water at
/// `level`, are in now: those of them still open, joined through open cells
/// that the level reaches. More than one piece means something was built
/// across the water.
pub fn parts(grid: &Grid, was: impl Iterator<Item = IVec3>, level: i64) -> Vec<Part> {
    // 1: was wet and is open; 2: gone through, whether it was wet or not.
    let mut state = vec![0u8; grid.cells.len()];
    let mut all = Vec::new();
    for p in was {
        if grid.is_open(p) && bottom(p.y) < level {
            state[grid.index(p)] = 1;
            all.push(p);
        }
    }
    let mut out = Vec::new();
    let mut queue = Vec::new();
    for first in all {
        if std::mem::replace(&mut state[grid.index(first)], 2) == 2 {
            continue;
        }
        let mut part = Part { seed: first, content: 0, cells: vec![first] };
        queue.clear();
        queue.push(first);
        while let Some(p) = queue.pop() {
            for d in NEIGHBOURS {
                let q = p + d;
                if !grid.is_open(q) || bottom(q.y) > level {
                    continue;
                }
                match std::mem::replace(&mut state[grid.index(q)], 2) {
                    2 => continue,
                    1 => part.cells.push(q),
                    _ => {}
                }
                queue.push(q);
            }
        }
        part.cells.sort_unstable_by_key(|p| (p.y, p.z, p.x));
        part.seed = part.cells[0];
        part.content = part.cells.iter().map(|p| (level - bottom(p.y)).clamp(0, FP)).sum();
        out.push(part);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Sample};

    /// A grid of solid rock with the cells `open` says are free, in a box from the origin.
    fn carve(dims: IVec3, open: impl Fn(IVec3) -> bool) -> Grid {
        let mut volume = Volume::new();
        for i in 0..27 {
            volume.insert(IVec3::new(i % 3, i / 9, i / 3 % 3), Chunk::Uniform(Sample { fill: 255, mat: 1 }));
        }
        for y in 0..dims.y {
            for z in 0..dims.z {
                for x in 0..dims.x {
                    let p = IVec3::new(x, y, z);
                    if open(p) {
                        volume.set(p, Sample::AIR);
                    }
                }
            }
        }
        Grid::new(&volume, IVec3::ZERO, dims).unwrap()
    }

    /// A pit 10 by 10 from height 4 up, open to the sky, in ground whose top is at 12.
    fn pit(p: IVec3) -> bool {
        p.y >= 12 || (p.y >= 4 && (10..20).contains(&p.x) && (10..20).contains(&p.z))
    }

    #[test]
    fn levels_and_layers() {
        assert_eq!((bottom(0), bottom(1), bottom(-1)), (-FP / 2, FP / 2, -3 * FP / 2));
        for (level, layer) in [(0, 0), (FP / 2, 0), (FP / 2 + 1, 1), (-FP / 2, -1), (-FP / 2 + 1, 0), (3 * FP / 2, 1)] {
            assert_eq!(top_layer(level), layer, "level {level}");
            assert!(bottom(layer) < level && bottom(layer + 1) >= level);
        }
    }

    #[test]
    fn a_pit_fills_from_the_bottom_and_volume_gives_the_level() {
        let grid = carve(IVec3::new(48, 40, 48), pit);
        let seed = IVec3::new(15, 4, 15);
        // Three layers and a quarter of the fourth.
        let volume = 100 * FP * 3 + 100 * FP / 4;
        let (cells, how, _) = fill(&grid, seed, volume, false);
        assert_eq!(how, Filled::Settled { level: bottom(7) + FP / 4 });
        assert_eq!(cells.len(), 400);
        assert!(cells.iter().all(|p| pit(*p) && (4..8).contains(&p.y)));
        // The same from anywhere in the pit, however it is seeded.
        let (again, how_again, _) = fill(&grid, IVec3::new(10, 4, 19), volume, false);
        assert_eq!(how_again, how);
        assert_eq!(again.len(), 400);
        // One unit of water wets the floor.
        let (cells, how, _) = fill(&grid, seed, 1, false);
        assert_eq!((cells.len(), how), (100, Filled::Settled { level: bottom(4) + 1 }));
    }

    #[test]
    fn too_much_water_spills_over_the_rim_into_open_ground() {
        // The ground outside the pit slopes away to the east.
        let grid = carve(IVec3::new(48, 40, 48), |p| pit(p) || (p.x >= 24 && p.y >= 12 - (p.x - 24) / 2));
        let (cells, how, _) = fill(&grid, IVec3::new(15, 4, 15), 100 * FP * 20, false);
        // Eight layers fit below the rim at 12.
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((level, kept, cells.len()), (bottom(12), 800 * FP, 800));
        assert!(at.x >= 24 && at.y < 12, "spills at {at}");
        // Down the slope to the last step in the box.
        let (path, end) = run_down(&grid, at);
        let last = IVec3::new(46, 1, at.z);
        assert_eq!((path[0], *path.last().unwrap(), end), (at, last, Run::Rest(last)));
        // Dropped over a hole in the floor of the box, it is still falling.
        let shaft = carve(IVec3::new(48, 40, 48), |p| p.x == 5 && p.z == 5);
        let (from, to) = (IVec3::new(5, 30, 5), IVec3::new(5, 0, 5));
        assert_eq!(run_down(&shaft, from), (vec![from, to], Run::Out(to)));
    }

    #[test]
    fn a_second_hollow_beyond_a_rim_fills_only_when_the_water_reaches_it() {
        // Two pits, 10 by 10 and 5 by 10, joined by a notch whose floor is at 8.
        let second = |p: IVec3| p.y >= 4 && (25..30).contains(&p.x) && (10..20).contains(&p.z);
        let notch = |p: IVec3| p.y >= 8 && (20..25).contains(&p.x) && p.z == 15;
        let open = move |p: IVec3| p.y < 30 && (pit(p) || second(p) || notch(p)) && (p.y < 12 || p.x < 35);
        let grid = carve(IVec3::new(48, 40, 48), |p| open(p) && p.y < 12);
        let seed = IVec3::new(15, 4, 15);
        let first = 100 * FP * 4;
        // Up to the notch, the first pit keeps everything.
        let (cells, how, _) = fill(&grid, seed, first, false);
        assert_eq!((cells.len(), how), (400, Filled::Settled { level: bottom(8) }));
        // A little more runs through the notch and belongs to the second pit.
        let (cells, how, _) = fill(&grid, seed, first + 50 * FP, false);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (400, bottom(8), first));
        assert!(second(at) && at.y < 8, "spills at {at}");
        assert_eq!(run_down(&grid, at), (vec![at, IVec3::new(at.x, 4, at.z)], Run::Rest(IVec3::new(at.x, 4, at.z))));
        // Enough to fill the second pit to the notch too, and both rise as one.
        let second_below = 50 * FP * 4;
        let both = first + second_below + (100 + 50 + 5) * FP / 2;
        let (cells, how, _) = fill(&grid, seed, both, false);
        assert_eq!(how, Filled::Settled { level: bottom(8) + FP / 2 });
        assert_eq!(cells.len(), 400 + 200 + 155);
        // Seeded from the second pit the answer is the same.
        let (cells, how, _) = fill(&grid, IVec3::new(27, 4, 12), both, false);
        assert_eq!((cells.len(), how), (755, Filled::Settled { level: bottom(8) + FP / 2 }));
    }

    #[test]
    fn a_hollow_beside_open_ground_fills_before_the_rest_runs_off() {
        // The pit, a second one west of it behind a notch at 8, and to the
        // east a notch at 8 onto ground that slopes away out of the box.
        let second = |p: IVec3| p.y >= 4 && p.y < 12 && (2..7).contains(&p.x) && (10..20).contains(&p.z);
        let west = |p: IVec3| p.y >= 8 && p.y < 12 && (7..10).contains(&p.x) && p.z == 15;
        let east = |p: IVec3| p.y >= 8 && (20..24).contains(&p.x) && p.z == 15;
        let slope = |p: IVec3| p.x >= 24 && p.y >= 8 - (p.x - 24) / 2;
        let grid = carve(IVec3::new(48, 40, 48), |p| pit(p) || second(p) || west(p) || east(p) || slope(p));
        let seed = IVec3::new(15, 4, 15);
        let (first, other) = (100 * FP * 4, 50 * FP * 4);
        // A little over the rim: it goes to the nearer hollow or off down the slope, not both.
        let (cells, how, _) = fill(&grid, seed, first + 10 * FP, false);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (400, bottom(8), first));
        assert!(second(at) && at.y < 8, "spills at {at}");
        // Plenty: both hollows are full to the rim and the rest leaves down the slope.
        let (cells, how, _) = fill(&grid, seed, first + other + 1000 * FP, false);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (600, bottom(8), first + other));
        assert!(slope(at), "spills at {at}");
        assert!(cells.iter().all(|&p| (pit(p) || second(p)) && p.y < 8));
        // The same seen from the second hollow, so neither hands water back to the other.
        let (cells, how, _) = fill(&grid, IVec3::new(4, 4, 12), first + other + 1000 * FP, false);
        assert_eq!((cells.len(), how), (600, Filled::Spilled { level, kept, at: Some(at) }));
    }

    #[test]
    fn rising_water_stops_where_it_meets_another_body() {
        let mut grid = carve(IVec3::new(48, 40, 48), pit);
        for x in 10..20 {
            for z in 10..20 {
                grid.hold(IVec3::new(x, 9, z), BodyId::Ocean);
            }
        }
        assert_eq!(grid.holder(IVec3::new(12, 9, 12)), Some(BodyId::Ocean));
        assert!(grid.holder(IVec3::new(12, 8, 12)).is_none() && !grid.is_free(IVec3::new(12, 9, 12)));
        let (cells, how, _) = fill(&grid, IVec3::new(15, 4, 15), 100 * FP * 30, false);
        assert_eq!(how, Filled::Met { level: bottom(8), kept: 400 * FP, with: BodyId::Ocean });
        assert_eq!(cells.len(), 400);
        assert_eq!(run_down(&grid, IVec3::new(15, 20, 15)).1, Run::Into(BodyId::Ocean));
    }

    #[test]
    fn spreading_floods_what_is_below_the_level_and_reports_the_rest() {
        let mut grid = carve(IVec3::new(48, 40, 48), pit);
        grid.hold(IVec3::new(19, 5, 19), BodyId::Pond(7));
        let seed = IVec3::new(15, 4, 15);
        let all = spread(&grid, &[seed], bottom(8), BodyId::Ocean, u32::MAX);
        assert_eq!(all.cells.len(), 399);
        assert!(all.open.is_empty() && all.cells.iter().all(|p| p.y < 8));
        assert_eq!(all.met.iter().map(|m| m.0).collect::<Vec<_>>(), vec![BodyId::Pond(7); 1]);
        // Out of reach: what was not got to is handed back, to carry on from.
        let part = spread(&grid, &[seed], bottom(8), BodyId::Ocean, 3);
        assert!(part.cells.iter().all(|&p| (p - seed).abs().element_sum() <= 3) && part.cells.len() == 44);
        assert!(!part.open.is_empty() && part.open.iter().all(|&p| grid.is_free(p)));
        assert!(part.open.iter().all(|&p| (p - seed).abs().element_sum() == 4));
        // A box that cuts the pit in two: the far half is reported as open.
        let half = carve(IVec3::new(15, 40, 48), pit);
        let cut = spread(&half, &[IVec3::new(12, 4, 15)], bottom(8), BodyId::Ocean, u32::MAX);
        assert_eq!((cut.cells.len(), cut.open.len()), (200, 40));
        assert!(cut.open.iter().all(|p| p.x == 15));
        // Seeds that are solid or above the level are ignored.
        let none = spread(&grid, &[IVec3::new(0, 0, 0), IVec3::new(15, 9, 15)], bottom(8), BodyId::Ocean, 10);
        assert!(none.cells.is_empty());
    }

    #[test]
    fn water_runs_downhill_to_the_bottom_of_a_hollow() {
        // Steps down to the west into a pit, from a start high above the east side.
        let hole = |p: IVec3| p.y >= 4 && (10..20).contains(&p.x) && (10..20).contains(&p.z);
        let grid = carve(IVec3::new(48, 40, 48), move |p| hole(p) || p.y >= 12 + (p.x - 20).max(0) / 3);
        let (path, end) = run_down(&grid, IVec3::new(40, 30, 15));
        assert_eq!(end, Run::Rest(IVec3::new(19, 4, 15)));
        // Down onto the top step, west along each and down to the next, and into the pit.
        assert_eq!(path[..3], [IVec3::new(40, 30, 15), IVec3::new(40, 18, 15), IVec3::new(37, 18, 15)]);
        assert_eq!(path[path.len() - 3..], [IVec3::new(22, 12, 15), IVec3::new(19, 12, 15), IVec3::new(19, 4, 15)]);
        assert!(path.windows(2).all(|w| (w[1] - w[0]).abs().min_element() == 0 && w[1].y <= w[0].y));
        assert_eq!(run_down(&grid, IVec3::new(15, 30, 12)).1, Run::Rest(IVec3::new(15, 4, 12)));
        // Let go inside the ground, water comes up to the first open cell.
        let (path, end) = run_down(&grid, IVec3::new(5, 2, 5));
        assert_eq!((path[0], end), (IVec3::new(5, 12, 5), Run::Rest(IVec3::new(10, 4, 10))));
    }

    #[test]
    fn a_wall_built_across_water_makes_two_parts() {
        // The pit with a wall across it at x = 14 up to height 7.
        let walled = |p: IVec3| pit(p) && !(p.x == 14 && p.y < 8);
        let grid = carve(IVec3::new(48, 40, 48), walled);
        let was: Vec<IVec3> = (0..48 * 40 * 48)
            .map(|i| IVec3::new(i % 48, i / (48 * 48), i / 48 % 48))
            .filter(|&p| pit(p) && p.y < 7)
            .collect();
        // Water three layers deep, a quarter into the third, is cut in two by it.
        let level = bottom(6) + FP / 4;
        let found = parts(&grid, was.iter().copied(), level);
        assert_eq!(found.len(), 2);
        assert_eq!((found[0].seed, found[1].seed), (IVec3::new(10, 4, 10), IVec3::new(15, 4, 10)));
        assert_eq!((found[0].cells.len(), found[1].cells.len()), (40 * 3, 50 * 3));
        assert_eq!((found[0].content, found[1].content), (40 * (2 * FP + FP / 4), 50 * (2 * FP + FP / 4)));
        // Water that reaches the top of the wall is one part still, and so
        // is water that was already over it.
        assert_eq!(parts(&grid, was.iter().copied(), bottom(8)).len(), 1);
        let over = was.iter().copied().chain((10..20).flat_map(|x| (10..20).map(move |z| IVec3::new(x, 7, z))));
        let found = parts(&grid, over, bottom(8) + 1);
        assert_eq!((found.len(), found[0].cells.len()), (1, 400 - 4 * 10));
    }
}
