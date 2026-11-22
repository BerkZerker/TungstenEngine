//! Where water goes on the finest lattice: filling a hollow from its lowest
//! point, spreading at a fixed level, and running downhill.
//!
//! Part of the bit-exact contract: integers only. Everything here works in a
//! [`Grid`], a copy of a box of the world small enough to search in one go;
//! what happens beyond the box is for the caller to follow up.
//!
//! A cell is a lattice sample that is not solid. The cell at lattice height
//! `y` holds the slab of water from half a cell below it to half a cell
//! above, so a full cell is `FP` units of volume and the slab starts at
//! [`bottom`]. Water reaches a cell once the level is above that.

use std::collections::VecDeque;

use glam::IVec3;

use crate::chunk::Volume;
use crate::coords::SOLID;
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

/// Cells searched before a flood gives up and calls the space open.
const BUDGET: usize = 120_000;

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
    /// Size of the box a flood is searched in: 16 m by 12 m at the default cell size.
    pub const DIMS: IVec3 = IVec3::new(64, 48, 64);

    /// The search box for water at `p`: centred on it, with more room above than below.
    pub fn around(p: IVec3) -> IVec3 {
        p - IVec3::new(Self::DIMS.x / 2, Self::DIMS.y / 3, Self::DIMS.z / 2)
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
    fn index(&self, p: IVec3) -> usize {
        let l = p - self.origin;
        ((l.y * self.dims.z + l.z) * self.dims.x + l.x) as usize
    }

    fn point(&self, i: usize) -> IVec3 {
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
/// Returns the cells that end up wet and how it ended.
///
/// The search goes up one layer at a time. Reaching layer `k` opens every
/// cell that can be got to without going above it, including lower ground
/// beyond a rim. A hollow there fills before the water rises further; if
/// there is not enough to fill it, the water this side of the rim stays
/// level with it and the rest runs over, to be another body. Ground that
/// goes on down past the box takes everything above the rim.
pub fn fill(grid: &Grid, seed: IVec3, volume: i64) -> (Vec<IVec3>, Filled) {
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

    for k in seed.y..grid.origin.y + grid.dims.y {
        // Everything wet so far is below this layer, and full.
        let (level, kept) = (bottom(k), FP * wet.len() as i64);
        let phase = (k - grid.origin.y + 1) as u8;
        let opened = order.len();
        order.append(&mut waiting[(k - grid.origin.y) as usize]);
        if order.len() == opened {
            // A closed space, full to its roof.
            return (wet, Filled::Spilled { level, kept, at: None });
        }
        let (mut cut_short, mut met) = (false, None);
        let mut next = opened;
        while next < order.len() {
            let p = point(order[next]);
            next += 1;
            for d in NEIGHBOURS {
                let q = p + d;
                if !grid.inside(q) {
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
                cut_short = true;
                break;
            }
        }
        if let Some(with) = met {
            return (wet, Filled::Met { level, kept, with });
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
            let (mut open, mut next) = (cut_short, 0);
            while next < group.len() {
                let p = point(group[next]);
                next += 1;
                for d in NEIGHBOURS {
                    let q = p + d;
                    if q.y >= k {
                        continue;
                    }
                    if !grid.inside(q) {
                        open = true;
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
                return (wet, Filled::Spilled { level, kept: volume - rest, at: Some(point(first)) });
            } else {
                wet.extend(group.iter().map(|&i| point(i)));
            }
        }
        if way_out.is_some() {
            let kept = FP * wet.len() as i64;
            return (wet, Filled::Spilled { level, kept, at: way_out });
        }

        // The layer itself. Where it runs past the box, the edge of the box
        // stands in for a shore.
        let flat: Vec<IVec3> = order[opened..].iter().map(|&i| point(i)).filter(|p| p.y == k).collect();
        let rest = volume - FP * wet.len() as i64;
        let n = flat.len() as i64;
        wet.extend(flat);
        if rest <= FP * n {
            return (wet, Filled::Settled { level: level + (rest + n - 1) / n });
        }
    }
    // Still rising at the top of the box.
    let (level, kept) = (bottom(grid.origin.y + grid.dims.y), FP * wet.len() as i64);
    (wet, Filled::Spilled { level, kept, at: None })
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

/// Flood from `seeds` through every free cell below `level`: what an opening
/// lets in from a body too large to notice the loss. At most `budget` cells.
pub fn spread(grid: &Grid, seeds: &[IVec3], level: i64, body: BodyId, budget: usize) -> Spread {
    let mut out = Spread::default();
    let mut seen = vec![false; grid.cells.len()];
    let mut queue: VecDeque<IVec3> = VecDeque::new();
    for &s in seeds {
        if grid.is_free(s) && bottom(s.y) < level && !std::mem::replace(&mut seen[grid.index(s)], true) {
            queue.push_back(s);
        }
    }
    while let Some(p) = queue.pop_front() {
        if out.cells.len() >= budget {
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
                        queue.push_back(q);
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
    /// At rest in a hollow, with nowhere lower to go.
    Rest(IVec3),
    /// Into another body's water.
    Into(BodyId),
    /// Still going at the edge of the box.
    Out(IVec3),
    /// Nowhere: the start is buried.
    Blocked,
}

/// Follow water downhill from `p`: straight down while it can fall, and
/// along the ground to the nearest drop when it cannot.
pub fn run_down(grid: &Grid, mut p: IVec3) -> Run {
    // A start just inside the ground counts as on it.
    for _ in 0..4 {
        if grid.is_open(p) {
            break;
        }
        p.y += 1;
    }
    if !grid.is_open(p) {
        return Run::Blocked;
    }
    /// Cells of level ground searched for a way down before calling it the bottom.
    const LEVEL_GROUND: usize = 1024;
    let mut seen = vec![0u32; grid.cells.len()];
    let mut search = 0u32;
    let mut queue: VecDeque<IVec3> = VecDeque::new();
    'falling: loop {
        loop {
            if let Some(body) = grid.holder(p) {
                return Run::Into(body);
            }
            let under = p - IVec3::Y;
            if !grid.inside(under) {
                return Run::Out(p);
            }
            if !grid.is_open(under) {
                break;
            }
            p = under;
        }
        // On the ground: look along this layer for the nearest way down.
        search += 1;
        queue.clear();
        queue.push_back(p);
        seen[grid.index(p)] = search;
        let (mut visited, mut edge) = (0, None);
        while let Some(c) = queue.pop_front() {
            visited += 1;
            if visited > LEVEL_GROUND {
                break;
            }
            for d in [IVec3::NEG_X, IVec3::X, IVec3::NEG_Z, IVec3::Z] {
                let q = c + d;
                if !grid.inside(q) {
                    edge = edge.or(Some(q));
                    continue;
                }
                let j = grid.index(q);
                if seen[j] == search || grid.cells[j] == ROCK {
                    continue;
                }
                seen[j] = search;
                if let Some(body) = grid.holder(q) {
                    return Run::Into(body);
                }
                let under = q - IVec3::Y;
                if grid.is_open(under) {
                    p = under;
                    continue 'falling;
                }
                queue.push_back(q);
            }
        }
        return match edge {
            Some(q) if visited <= LEVEL_GROUND => Run::Out(q),
            _ => Run::Rest(p),
        };
    }
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
        let (cells, how) = fill(&grid, seed, volume);
        assert_eq!(how, Filled::Settled { level: bottom(7) + FP / 4 });
        assert_eq!(cells.len(), 400);
        assert!(cells.iter().all(|p| pit(*p) && (4..8).contains(&p.y)));
        // The same from anywhere in the pit, however it is seeded.
        let (again, how_again) = fill(&grid, IVec3::new(10, 4, 19), volume);
        assert_eq!(how_again, how);
        assert_eq!(again.len(), 400);
        // One unit of water wets the floor.
        let (cells, how) = fill(&grid, seed, 1);
        assert_eq!((cells.len(), how), (100, Filled::Settled { level: bottom(4) + 1 }));
    }

    #[test]
    fn too_much_water_spills_over_the_rim_into_open_ground() {
        // The ground outside the pit slopes away to the east.
        let grid = carve(IVec3::new(48, 40, 48), |p| pit(p) || (p.x >= 24 && p.y >= 12 - (p.x - 24) / 2));
        let (cells, how) = fill(&grid, IVec3::new(15, 4, 15), 100 * FP * 20);
        // Eight layers fit below the rim at 12.
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((level, kept, cells.len()), (bottom(12), 800 * FP, 800));
        assert!(at.x >= 24 && at.y < 12, "spills at {at}");
        assert_eq!(run_down(&grid, at), Run::Out(IVec3::new(48, at.y - (47 - at.x) / 2, at.z)));
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
        let (cells, how) = fill(&grid, seed, first);
        assert_eq!((cells.len(), how), (400, Filled::Settled { level: bottom(8) }));
        // A little more runs through the notch and belongs to the second pit.
        let (cells, how) = fill(&grid, seed, first + 50 * FP);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (400, bottom(8), first));
        assert!(second(at) && at.y < 8, "spills at {at}");
        assert_eq!(run_down(&grid, at), Run::Rest(IVec3::new(at.x, 4, at.z)));
        // Enough to fill the second pit to the notch too, and both rise as one.
        let second_below = 50 * FP * 4;
        let both = first + second_below + (100 + 50 + 5) * FP / 2;
        let (cells, how) = fill(&grid, seed, both);
        assert_eq!(how, Filled::Settled { level: bottom(8) + FP / 2 });
        assert_eq!(cells.len(), 400 + 200 + 155);
        // Seeded from the second pit the answer is the same.
        let (cells, how) = fill(&grid, IVec3::new(27, 4, 12), both);
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
        let (cells, how) = fill(&grid, seed, first + 10 * FP);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (400, bottom(8), first));
        assert!(second(at) && at.y < 8, "spills at {at}");
        // Plenty: both hollows are full to the rim and the rest leaves down the slope.
        let (cells, how) = fill(&grid, seed, first + other + 1000 * FP);
        let Filled::Spilled { level, kept, at: Some(at) } = how else { panic!("{how:?}") };
        assert_eq!((cells.len(), level, kept), (600, bottom(8), first + other));
        assert!(slope(at), "spills at {at}");
        assert!(cells.iter().all(|&p| (pit(p) || second(p)) && p.y < 8));
        // The same seen from the second hollow, so neither hands water back to the other.
        let (cells, how) = fill(&grid, IVec3::new(4, 4, 12), first + other + 1000 * FP);
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
        let (cells, how) = fill(&grid, IVec3::new(15, 4, 15), 100 * FP * 30);
        assert_eq!(how, Filled::Met { level: bottom(8), kept: 400 * FP, with: BodyId::Ocean });
        assert_eq!(cells.len(), 400);
        assert_eq!(run_down(&grid, IVec3::new(15, 20, 15)), Run::Into(BodyId::Ocean));
    }

    #[test]
    fn spreading_floods_what_is_below_the_level_and_reports_the_rest() {
        let mut grid = carve(IVec3::new(48, 40, 48), pit);
        grid.hold(IVec3::new(19, 5, 19), BodyId::Pond(7));
        let seed = IVec3::new(15, 4, 15);
        let all = spread(&grid, &[seed], bottom(8), BodyId::Ocean, usize::MAX);
        assert_eq!(all.cells.len(), 399);
        assert!(all.open.is_empty() && all.cells.iter().all(|p| p.y < 8));
        assert_eq!(all.met.iter().map(|m| m.0).collect::<Vec<_>>(), vec![BodyId::Pond(7); 1]);
        // Out of budget: what was not reached is handed back.
        let part = spread(&grid, &[seed], bottom(8), BodyId::Ocean, 150);
        assert_eq!(part.cells.len(), 150);
        assert!(!part.open.is_empty() && part.open.iter().all(|&p| grid.is_free(p)));
        // A box that cuts the pit in two: the far half is reported as open.
        let half = carve(IVec3::new(15, 40, 48), pit);
        let cut = spread(&half, &[IVec3::new(12, 4, 15)], bottom(8), BodyId::Ocean, usize::MAX);
        assert_eq!((cut.cells.len(), cut.open.len()), (200, 40));
        assert!(cut.open.iter().all(|p| p.x == 15));
        // Seeds that are solid or above the level are ignored.
        let none = spread(&grid, &[IVec3::new(0, 0, 0), IVec3::new(15, 9, 15)], bottom(8), BodyId::Ocean, 10);
        assert!(none.cells.is_empty());
    }

    #[test]
    fn water_runs_downhill_to_the_bottom_of_a_hollow() {
        // Steps down to the west into the pit, from a start high above the east side.
        let grid = carve(IVec3::new(48, 40, 48), |p| pit(p) || p.y >= 12 + (p.x - 20).max(0) / 3);
        assert_eq!(run_down(&grid, IVec3::new(40, 30, 15)), Run::Rest(IVec3::new(19, 4, 15)));
        assert_eq!(run_down(&grid, IVec3::new(15, 30, 12)), Run::Rest(IVec3::new(15, 4, 12)));
        assert_eq!(run_down(&grid, IVec3::new(5, 2, 5)), Run::Blocked);
    }
}
