use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::{DVec3, IVec3};
use tungsten_core::coords::{APRON, CHUNK_BITS, CHUNK_SIZE, LEVELS, MAX_LEVEL, NodeKey, chunk_of};
use tungsten_core::edit::{self, EditOutcome, EditRules, Op};
use tungsten_core::field::LocalField;
use tungsten_core::hash::{IntMap, IntSet};
use tungsten_core::log::{self, Baked, CellEdits, CellPos, CellSet, EditLog, TIERS};
use tungsten_core::mesh::{ChunkMesh, mesh_block};
use tungsten_core::water::bodies::{JOIN, Water};
use tungsten_core::water::job::{KeptTerrain, Made, Outcome, Request, WaterJob};
use tungsten_core::water::{self, NodeWater};
use tungsten_core::worldgen::{Generated, Generator};
use tungsten_core::{Chunk, ChunkPos, Volume, WorldParams};

use crate::pool::Pool;
use crate::store::{REGION_BITS, RegionPos, Store};

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Metres of terrain drawn around the focus.
    pub view_distance: f32,
    /// A node is split into finer ones while the focus is within this many
    /// of its own lengths. Higher draws fine detail further out.
    pub detail: f32,
    pub workers: usize,
    /// An edit cell is baked once it holds this many ops.
    pub bake_ops: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self { view_distance: 2048.0, detail: 2.0, workers: cores.saturating_sub(2).max(1), bake_ops: 512 }
    }
}

/// A change to what should be drawn. Events must be applied in order.
pub enum MeshEvent {
    /// `urgent` marks the result of an edit, which must reach the screen
    /// together with the rest of its group and without delay.
    Upload { key: NodeKey, mesh: ChunkMesh, urgent: bool },
    /// The node no longer has a mesh.
    Remove { key: NodeKey, urgent: bool },
    /// Which nodes are on screen changed. Nodes are hidden until shown. To be
    /// applied whole: taking half would leave a hole or draw ground twice.
    Visible { show: Vec<NodeKey>, hide: Vec<NodeKey> },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub nodes: usize,
    /// Nodes with a mesh, on screen or held in reserve.
    pub meshed: usize,
    /// Nodes on screen, by level.
    pub drawn: [u32; LEVELS],
    /// Finest-level chunks held for collision and editing, and how many are not uniform.
    pub chunks: usize,
    pub dense: usize,
    pub jobs_queued: usize,
    pub jobs_running: usize,
    /// Edit cells with edits, ops in them, and cells baked this session.
    pub cells: usize,
    pub ops: usize,
    pub bakes: usize,
}

enum Job {
    /// Generate a node's block, resolve the edits in `cells` and mesh it.
    Node {
        key: NodeKey,
        version: u64,
        group: u32,
        cells: CellSet,
        water: NodeWater,
    },
    /// Remesh a finest-level node from its chunk and 26 neighbours, indexed
    /// as `neighbours()` yields them.
    Mesh {
        key: NodeKey,
        version: u64,
        group: u32,
        chunks: Box<[Arc<Chunk>; 27]>,
        water: NodeWater,
    },
    Bake {
        cell: CellPos,
        edits: Arc<CellEdits>,
    },
    /// A water search (pond settle, flow route, or sea advance).
    Water(WaterJob),
    /// Free unloaded chunks off the main thread.
    Release(Vec<Arc<Chunk>>),
}

enum Done {
    /// A mesh and, from a finest-level `Job::Node`, the chunk it was made from.
    Node { key: NodeKey, version: u64, group: u32, mesh: ChunkMesh, chunk: Option<Chunk> },
    /// The overlay that replaces the first `ops` ops of a cell.
    Baked { cell: CellPos, ops: usize, baked: Baked },
}

struct Node {
    /// Changes whenever the samples behind this node's mesh do. Never reused.
    version: u64,
    /// Version a job was last started for.
    requested: Option<u64>,
    /// Version of the mesh the renderer holds (which may be no mesh at all).
    shown: Option<u64>,
    has_mesh: bool,
    /// Part of the tree the focus currently asks for.
    wanted: bool,
    /// Wanted as finer nodes rather than as itself.
    refine: bool,
    /// How many of its eight finer nodes exist.
    children: u8,
    /// All eight exist and can be drawn without a hole; as of the last `redraw`.
    children_cover: bool,
    /// The renderer has been told to draw it.
    drawn: bool,
    /// Scratch for `redraw`.
    marked: bool,
}

/// Meshes of one edit, held back until all have arrived so they swap in together.
struct Group {
    remaining: usize,
    meshes: Vec<(NodeKey, u64, ChunkMesh)>,
}

/// Where a world's water is saved, beside its regions.
const WATER_FILE: &str = "water.bin";
const URGENT: u64 = 0;
const IDLE: u64 = u64::MAX;
/// Priority offset of meshes that only stand in while finer ones load.
const STAND_IN: u64 = 1 << 32;
/// Main-thread time per update for accepting finished streaming work.
const COLLECT_BUDGET: std::time::Duration = std::time::Duration::from_micros(1500);
/// Base cells the focus moves before the tree is looked at again.
const RETARGET: f64 = 8.0;

pub struct World {
    params: WorldParams,
    rules: EditRules,
    config: StreamConfig,
    generator: Arc<Generator>,
    /// The finest level around the focus, with every edit applied.
    volume: Volume,
    log: EditLog,
    /// Ponds, and where the sea has come in. The sea itself is a function of the generator.
    water: Water,
    store: Option<Arc<Store>>,
    /// Regions whose saved cells are in the log.
    regions: IntSet<RegionPos>,
    nodes: IntMap<NodeKey, Node>,
    /// The coarsest nodes in view.
    roots: Vec<NodeKey>,
    pool: Pool<Job>,
    done: Receiver<Done>,
    /// Results of edit remeshes, kept apart so streaming cannot delay them.
    done_urgent: Receiver<Done>,
    /// Outcome of the in-flight water search, awaited at its fixed tick, and
    /// the chunks it had to make.
    done_water: Receiver<(Outcome, Vec<Made>)>,
    /// Terrain made for water searches where none is loaded.
    water_terrain: KeptTerrain,
    /// Focus in base cells, as of the last look at the tree.
    focus: Option<DVec3>,
    groups: IntMap<u32, Group>,
    next_group: u32,
    next_version: u64,
    /// Cells with a bake under way.
    baking: IntSet<CellPos>,
    bakes: usize,
    meshed: usize,
    /// The set of nodes may need to change.
    tree_dirty: bool,
    /// What is drawn may need to change.
    drawn_dirty: bool,
    drawn_levels: [u32; LEVELS],
}

fn neighbours() -> impl Iterator<Item = IVec3> {
    (-1..=1).flat_map(|y| (-1..=1).flat_map(move |z| (-1..=1).map(move |x| IVec3::new(x, y, z))))
}

/// Distance in base cells from a point to a node's box.
fn distance(key: NodeKey, focus: DVec3) -> f64 {
    let min = key.origin().as_dvec3();
    let max = min + key.size() as f64;
    (min - focus).max(focus - max).max(DVec3::ZERO).length()
}

/// Job priority: nearest first, measured in the node's own lengths so that
/// every level fills in from the focus outwards together.
fn priority(key: NodeKey, focus: DVec3, stand_in: bool) -> u64 {
    1 + (distance(key, focus) / key.size() as f64 * 64.0) as u64 + if stand_in { STAND_IN } else { 0 }
}

/// Call `f` for every chunk position in `[min, max]`.
fn for_box(min: IVec3, max: IVec3, mut f: impl FnMut(IVec3)) {
    for y in min.y..=max.y {
        for z in min.z..=max.z {
            for x in min.x..=max.x {
                f(IVec3::new(x, y, z));
            }
        }
    }
}

/// Load the saved edits of every region of `tier` overlapping base-cell bounds `[min, max]`.
fn load_regions(
    store: Option<&Store>,
    log: &mut EditLog,
    regions: &mut IntSet<RegionPos>,
    tier: u8,
    min: IVec3,
    max: IVec3,
) {
    let Some(store) = store else { return };
    let shift = CellPos::bits(tier) + REGION_BITS;
    for_box(min >> shift, max >> shift, |r| {
        if !regions.insert((tier, r)) {
            return;
        }
        for (pos, bytes) in store.region((tier, r)) {
            match CellEdits::decode(&bytes) {
                Ok(edits) => log.insert(pos, edits),
                // Edits that cannot be read leave the generated terrain as it is.
                Err(e) => eprintln!("tungsten: saved edits of cell {pos:?} are unreadable ({e}); ignored"),
            }
        }
    });
}

impl World {
    /// With a `store`, saved edits are loaded as their region comes into use
    /// and new ones are written back on [`save`](Self::save).
    pub fn new(params: WorldParams, config: StreamConfig, store: Option<Arc<Store>>) -> Self {
        let generator = Arc::new(Generator::new(params));
        let rules = EditRules::new(&params);
        let (tx, done): (Sender<Done>, _) = unbounded();
        let (tx_urgent, done_urgent): (Sender<Done>, _) = unbounded();
        let (tx_water, done_water): (Sender<(Outcome, Vec<Made>)>, _) = unbounded();
        let worker_gen = generator.clone();
        let pool = Pool::new(config.workers, move |job: Job| {
            let result = match job {
                Job::Release(chunks) => {
                    drop(chunks);
                    return;
                }
                Job::Water(job) => {
                    let _ = tx_water.send(job.run_keeping(&worker_gen, &rules));
                    return;
                }
                Job::Node { key, version, group, cells, water } => {
                    let mut block = LocalField::block(key);
                    let generated = worker_gen.fill_block(key, &mut block);
                    log::resolve(&mut block, &rules, &cells);
                    let untouched = |s| matches!(generated, Generated::Uniform(u) if u == s) && cells.is_empty();
                    // Coarse nodes overlap their neighbours; see `mesh_block`.
                    let margin = (key.level > 0) as i32;
                    let mut mesh = match generated {
                        Generated::Uniform(_) if cells.is_empty() => ChunkMesh::default(),
                        _ => mesh_block(&block, margin),
                    };
                    mesh.water = water::node_patches(&worker_gen, key, &water);
                    let chunk = (key.level == 0).then(|| match generated {
                        Generated::Uniform(s) if untouched(s) => Chunk::Uniform(s),
                        _ => block.interior(),
                    });
                    Done::Node { key, version, group, mesh, chunk }
                }
                Job::Mesh { key, version, group, chunks, water } => {
                    // Copying 27 chunks into a block is the expensive part of
                    // starting a mesh, so it happens here, not on the main thread.
                    let block = LocalField::gather(key, &chunks);
                    let mut mesh = mesh_block(&block, 0);
                    mesh.water = water::node_patches(&worker_gen, key, &water);
                    Done::Node { key, version, group, mesh, chunk: None }
                }
                Job::Bake { cell, edits } => {
                    Done::Baked { cell, ops: edits.ops.len(), baked: log::bake(&worker_gen, &rules, cell, &edits) }
                }
            };
            // The receiver is gone only during shutdown.
            let urgent = matches!(result, Done::Node { group, .. } if group != 0);
            let _ = if urgent { tx_urgent.send(result) } else { tx.send(result) };
        });

        let water = match store.as_ref().and_then(|s| s.read_file(WATER_FILE)) {
            Some(bytes) => Water::decode(&bytes).unwrap_or_else(|e| {
                eprintln!("tungsten: saved water is unreadable ({e}); ignored");
                Water::new()
            }),
            None => Water::new(),
        };

        Self {
            params,
            rules,
            config,
            generator,
            volume: Volume::new(),
            log: EditLog::new(),
            water,
            store,
            regions: IntSet::default(),
            nodes: IntMap::default(),
            roots: Vec::new(),
            pool,
            done,
            done_urgent,
            done_water,
            water_terrain: KeptTerrain::default(),
            focus: None,
            groups: IntMap::default(),
            next_group: 1,
            next_version: 1,
            baking: IntSet::default(),
            bakes: 0,
            meshed: 0,
            tree_dirty: false,
            drawn_dirty: false,
            drawn_levels: [0; LEVELS],
        }
    }

    pub fn params(&self) -> WorldParams {
        self.params
    }

    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn generator(&self) -> &Generator {
        &self.generator
    }

    pub fn volume(&self) -> &Volume {
        &self.volume
    }

    pub fn log(&self) -> &EditLog {
        &self.log
    }

    pub fn water(&self) -> &Water {
        &self.water
    }

    /// True once the finest-level chunk and its 26 neighbours hold data.
    pub fn neighbourhood_loaded(&self, pos: ChunkPos) -> bool {
        neighbours().all(|n| self.volume.chunks.contains_key(&(pos + n)))
    }

    pub fn stats(&self) -> Stats {
        let (jobs_queued, jobs_running) = self.pool.load();
        Stats {
            nodes: self.nodes.len(),
            meshed: self.meshed,
            drawn: self.drawn_levels,
            chunks: self.volume.chunks.len(),
            dense: self.volume.chunks.values().filter(|c| !c.is_uniform()).count(),
            jobs_queued,
            jobs_running,
            cells: self.log.cell_count(),
            ops: self.log.op_count(),
            bakes: self.bakes,
        }
    }

    /// True when nothing is being generated, meshed or baked and nothing is
    /// waiting to be shown.
    pub fn settled(&self) -> bool {
        let (q, r) = self.pool.load();
        self.focus.is_some()
            && q == 0
            && r == 0
            && self.done.is_empty()
            && self.done_urgent.is_empty()
            && !self.tree_dirty
            && !self.drawn_dirty
    }

    /// Advance streaming around `focus`, a position in metres. Mesh changes
    /// are appended to `events`.
    pub fn update(&mut self, focus: DVec3, events: &mut Vec<MeshEvent>) {
        let focus = focus / self.params.cell_size as f64;
        let moved = self.focus.is_none_or(|f| (f - focus).abs().max_element() >= RETARGET);
        if moved {
            self.focus = Some(focus);
        }
        if moved || self.tree_dirty {
            self.retarget(moved, events);
        }
        self.collect(events);
        if self.tree_dirty {
            self.retarget(false, events);
        }
        if self.drawn_dirty {
            self.redraw(events);
        }
    }

    /// Load the saved edits of every region of `tier` overlapping base-cell
    /// bounds `[min, max]`.
    fn ensure_regions(&mut self, tier: u8, min: IVec3, max: IVec3) {
        load_regions(self.store.as_deref(), &mut self.log, &mut self.regions, tier, min, max);
    }

    fn node_job(&mut self, key: NodeKey, version: u64, group: u32) -> Job {
        let step = key.step();
        let min = (key.pos * CHUNK_SIZE - APRON) * step;
        self.ensure_regions(CellPos::tier_of(key.level), min, min + (CHUNK_SIZE + 2 * APRON) * step);
        Job::Node { key, version, group, cells: self.log.gather_node(key), water: self.water.node_water(key) }
    }

    /// A job that meshes a finest-level node again: from the loaded chunks if
    /// all 27 are there, from scratch otherwise.
    fn remesh_job(&mut self, key: NodeKey, version: u64, group: u32) -> Job {
        let chunks: Option<Vec<Arc<Chunk>>> =
            neighbours().map(|n| self.volume.chunks.get(&(key.pos + n)).cloned()).collect();
        match chunks {
            Some(chunks) => {
                let chunks = chunks.into_boxed_slice().try_into().ok().expect("27 neighbours");
                Job::Mesh { key, version, group, chunks, water: self.water.node_water(key) }
            }
            None => self.node_job(key, version, group),
        }
    }

    /// Decide which nodes the focus asks for, start what they need and drop
    /// the rest. `moved` also re-prioritises what is queued.
    fn retarget(&mut self, moved: bool, events: &mut Vec<MeshEvent>) {
        self.tree_dirty = false;
        let Some(focus) = self.focus else { return };
        for n in self.nodes.values_mut() {
            n.wanted = false;
        }

        // The coarsest nodes within view, over the whole height of the
        // terrain and around the focus wherever it is.
        let cell = self.params.cell_size as f64;
        let size = (CHUNK_SIZE << MAX_LEVEL) as f64;
        let view = self.config.view_distance as f64 / cell;
        let (low, high) = self.generator.height_range();
        let reach = (CHUNK_SIZE * 8) as f64;
        let y0 = ((low as f64 / cell).min(focus.y - reach) / size).floor() as i32;
        let y1 = ((high as f64 / cell).max(focus.y + reach) / size).floor() as i32;
        let r = (view / size).ceil() as i32 + 1;
        let c = (focus / size).floor().as_ivec3();
        let mut jobs = Vec::new();
        let mut roots = std::mem::take(&mut self.roots);
        roots.clear();
        for_box(IVec3::new(c.x - r, y0, c.z - r), IVec3::new(c.x + r, y1, c.z + r), |pos| {
            let key = NodeKey::new(MAX_LEVEL, pos);
            let level_with = DVec3::new(focus.x, key.origin().y as f64, focus.z);
            if distance(key, level_with) <= view {
                roots.push(key);
                self.visit(key, focus, &mut jobs);
            }
        });
        self.roots = roots;

        // Say what is drawn now before taking anything away, so that the
        // renderer never holds a frame with a node gone and nothing in its place.
        self.redraw(events);

        // Finer nodes under a node that is wanted whole stay until its own
        // mesh is ready; everything else unwanted goes.
        let gone: Vec<NodeKey> =
            self.nodes.iter().filter(|(k, n)| !n.wanted && !self.kept(**k)).map(|(k, _)| *k).collect();
        let mut released = Vec::new();
        for key in gone {
            let node = self.nodes.remove(&key).unwrap();
            if let Some(parent) = (key.level < MAX_LEVEL).then(|| self.nodes.get_mut(&key.parent())).flatten() {
                parent.children -= 1;
            }
            if node.has_mesh {
                self.meshed -= 1;
                events.push(MeshEvent::Remove { key, urgent: false });
            }
            if key.level == 0 {
                released.extend(self.volume.chunks.remove(&key.pos));
            }
        }
        if !released.is_empty() {
            self.pool.push(IDLE, Job::Release(released));
        }

        if moved {
            // Re-prioritise what is still queued; drop jobs nobody wants any more.
            let nodes = &self.nodes;
            self.pool.rekey(|job| match job {
                Job::Node { key, version, group: 0, .. } => {
                    nodes.get(key).filter(|n| n.version == *version).map(|n| priority(*key, focus, n.refine))
                }
                Job::Node { .. } | Job::Mesh { .. } | Job::Water(_) => Some(URGENT),
                Job::Bake { .. } | Job::Release(_) => Some(IDLE),
            });
        }
        self.pool.push_all(jobs);
    }

    fn visit(&mut self, key: NodeKey, focus: DVec3, jobs: &mut Vec<(u64, Job)>) {
        if !self.nodes.contains_key(&key) {
            self.next_version += 1;
            self.nodes.insert(
                key,
                Node {
                    version: self.next_version,
                    requested: None,
                    shown: None,
                    has_mesh: false,
                    wanted: false,
                    refine: false,
                    children: 0,
                    children_cover: false,
                    drawn: false,
                    marked: false,
                },
            );
            // Nodes are visited from the top down, so the parent is there.
            if key.level < MAX_LEVEL {
                self.nodes.get_mut(&key.parent()).unwrap().children += 1;
            }
        }
        let node = self.nodes.get_mut(&key).unwrap();
        node.wanted = true;
        // A node already split stays split a little longer, so standing on a
        // boundary does not flip it back and forth.
        let limit = self.config.detail as f64 * key.size() as f64 * if node.refine { 1.1 } else { 1.0 };
        node.refine = key.level > 0 && distance(key, focus) < limit;
        let (refine, version) = (node.refine, node.version);
        // A split node only needs a mesh to stand in until its children have theirs.
        if node.requested != Some(version) && (!refine || node.shown.is_none()) {
            node.requested = Some(version);
            let job = self.node_job(key, version, 0);
            jobs.push((priority(key, focus, refine), job));
        }
        if refine {
            for child in key.children() {
                self.visit(child, focus, jobs);
            }
        }
    }

    /// Is this unwanted node standing in for a wanted coarser one whose mesh
    /// is not ready?
    fn kept(&self, mut key: NodeKey) -> bool {
        while key.level < MAX_LEVEL {
            key = key.parent();
            if let Some(n) = self.nodes.get(&key).filter(|n| n.wanted) {
                return !n.refine && n.shown != Some(n.version);
            }
        }
        false
    }

    fn collect(&mut self, events: &mut Vec<MeshEvent>) {
        // Results of edits first, and all of them: they are few and must not wait.
        while let Ok(done) = self.done_urgent.try_recv() {
            self.accept(done, events);
        }
        // Streaming results for a bounded time; the rest waits one update.
        let start = std::time::Instant::now();
        let mut taken = 0u32;
        loop {
            if taken.is_multiple_of(32) && start.elapsed() >= COLLECT_BUDGET {
                break;
            }
            let Ok(done) = self.done.try_recv() else { break };
            self.accept(done, events);
            taken += 1;
        }
    }

    fn accept(&mut self, done: Done, events: &mut Vec<MeshEvent>) {
        match done {
            Done::Baked { cell, ops, baked } => {
                // The cell resolves to the same samples as before, so nothing is remeshed.
                self.log.rebase(cell, ops, Arc::new(baked));
                self.baking.remove(&cell);
                self.bakes += 1;
            }
            Done::Node { key, version, group, mesh, chunk } => {
                // A result for an older version is out of date: a newer job is on its way.
                let current = self.nodes.get(&key).is_some_and(|n| n.version == version);
                if let (true, Some(chunk)) = (current, chunk) {
                    self.volume.insert(key.pos, chunk);
                }
                if group == 0 {
                    if current {
                        self.show(key, version, mesh, false, events);
                    }
                    return;
                }
                let Some(g) = self.groups.get_mut(&group) else { return };
                if current {
                    g.meshes.push((key, version, mesh));
                }
                g.remaining -= 1;
                if g.remaining == 0 {
                    for (key, version, mesh) in self.groups.remove(&group).unwrap().meshes {
                        self.show(key, version, mesh, true, events);
                    }
                }
            }
        }
    }

    fn show(&mut self, key: NodeKey, version: u64, mesh: ChunkMesh, urgent: bool, events: &mut Vec<MeshEvent>) {
        let Some(n) = self.nodes.get_mut(&key) else { return };
        if n.version != version {
            return;
        }
        n.shown = Some(version);
        match (mesh.is_empty(), n.has_mesh) {
            (true, true) => {
                n.has_mesh = false;
                self.meshed -= 1;
                events.push(MeshEvent::Remove { key, urgent });
            }
            (true, false) => {}
            (false, had) => {
                if !had {
                    n.has_mesh = true;
                    self.meshed += 1;
                }
                events.push(MeshEvent::Upload { key, mesh, urgent });
            }
        }
        self.drawn_dirty = true;
        // Finer nodes that stood in for this one can go now.
        if !n.refine && n.children > 0 {
            self.tree_dirty = true;
        }
    }

    /// Can this node's volume be drawn without a hole, by itself or by finer
    /// nodes? Records the answer for the finer nodes alone in `children_cover`.
    fn cover(&mut self, key: NodeKey) -> bool {
        let Some(n) = self.nodes.get(&key) else { return false };
        let (shown, children) = (n.shown.is_some(), n.children);
        let mut all = children == 8;
        if children > 0 {
            for child in key.children() {
                all &= self.cover(child);
            }
        }
        self.nodes.get_mut(&key).unwrap().children_cover = all;
        shown || all
    }

    /// Choose what to draw for a node's volume: itself or its children,
    /// never both and never part of each. Needs `cover` to have run.
    fn draw(&self, key: NodeKey, out: &mut Vec<NodeKey>) {
        let Some(n) = self.nodes.get(&key) else { return };
        let whole = !n.refine && if n.wanted { n.shown == Some(n.version) } else { n.shown.is_some() };
        if whole {
            out.push(key);
        } else if n.children_cover {
            // Wanted finer, or wanted whole but its mesh is not ready (or is
            // out of date) while finer ones are still here.
            key.children().for_each(|c| self.draw(c, out));
        } else if n.shown.is_some() {
            out.push(key);
        } else if n.children > 0 {
            // Nothing complete yet: show what there is.
            key.children().for_each(|c| self.draw(c, out));
        }
    }

    /// Work out what should be on screen and tell the renderer what changed.
    fn redraw(&mut self, events: &mut Vec<MeshEvent>) {
        self.drawn_dirty = false;
        let roots = std::mem::take(&mut self.roots);
        let mut draw = Vec::new();
        for &root in &roots {
            self.cover(root);
            self.draw(root, &mut draw);
        }
        self.roots = roots;
        for key in &draw {
            self.nodes.get_mut(key).unwrap().marked = true;
        }
        let (mut show, mut hide) = (Vec::new(), Vec::new());
        self.drawn_levels = [0; LEVELS];
        for (key, n) in &mut self.nodes {
            match (n.drawn, n.marked) {
                (false, true) => show.push(*key),
                (true, false) => hide.push(*key),
                _ => {}
            }
            n.drawn = std::mem::take(&mut n.marked);
            self.drawn_levels[key.level as usize] += n.drawn as u32;
        }
        if !show.is_empty() || !hide.is_empty() {
            events.push(MeshEvent::Visible { show, hide });
        }
    }

    /// Apply an edit: change the loaded samples, log it, and start the
    /// remeshing, whose results arrive through [`update`](Self::update).
    pub fn edit(&mut self, op: &Op) -> EditOutcome {
        let (lo, hi) = op.bounds_all(&self.rules);
        for tier in 0..TIERS as u8 {
            self.ensure_regions(tier, lo, hi);
        }
        let outcome = edit::edit_volume(&mut self.volume, &self.rules, op);
        // An op that changes nothing is not worth keeping.
        if outcome.chunks.is_empty() && outcome.covered {
            return outcome;
        }
        self.log.push(&self.rules, op);
        self.water_terrain.edited(&self.rules, op);
        if let Some(fragments) = &outcome.fragments {
            self.log.push(&self.rules, fragments);
            self.water_terrain.edited(&self.rules, fragments);
        }

        // Notify water of the terrain change; settling runs on workers on the
        // fixed simulation tick.
        if let Some((min, max)) = outcome.changed {
            self.water.edited(&self.volume, &self.generator, min, max);
        }

        let focus = self.focus.unwrap_or(DVec3::ZERO);
        let group = self.next_group;
        self.next_group = self.next_group.wrapping_add(1).max(1);
        let mut jobs = Vec::new();
        // Nodes given a job here; water that changed in them needs no second one.
        let mut done = IntSet::default();

        // The finest level: what changed is already in `volume`, so remesh
        // from it, all at once.
        let mut affected = Vec::new();
        for_box(chunk_of(outcome.min - APRON), chunk_of(outcome.max + APRON), |pos| affected.push(pos));
        for pos in affected {
            let key = NodeKey::new(0, pos);
            if !self.nodes.contains_key(&key) {
                continue;
            }
            let loaded = self.volume.chunks.contains_key(&pos);
            let block = (pos * CHUNK_SIZE - APRON, pos * CHUNK_SIZE + CHUNK_SIZE + APRON - 1);
            let touched = outcome.changed.is_some_and(|(a, b)| a.cmple(block.1).all() && b.cmpge(block.0).all());
            // A chunk still being generated was generated without this op.
            if loaded && !touched {
                continue;
            }
            let version = self.bump(key);
            let job = self.remesh_job(key, version, group);
            jobs.push((URGENT, job));
            done.insert(key);
        }
        if !jobs.is_empty() {
            self.groups.insert(group, Group { remaining: jobs.len(), meshes: Vec::with_capacity(jobs.len()) });
        }

        // Coarser levels resolve the log again. Only nodes on screen as
        // themselves need it now; the rest wait until they are.
        for level in 1..=MAX_LEVEL {
            let (min, max) = op.bounds(&self.rules, level);
            let mut affected = Vec::new();
            for_box(chunk_of(min - APRON), chunk_of(max + APRON), |pos| affected.push(NodeKey::new(level, pos)));
            for key in affected {
                let Some(n) = self.nodes.get(&key) else { continue };
                let (wanted, refine) = (n.wanted, n.refine);
                let version = self.bump(key);
                done.insert(key);
                if wanted && !refine {
                    self.nodes.get_mut(&key).unwrap().requested = Some(version);
                    let job = self.node_job(key, version, 0);
                    jobs.push((priority(key, focus, false), job));
                }
            }
        }
        self.pool.push_all(jobs);
        self.redraw_water(&mut done);

        // Bake cells that have collected enough ops.
        for cell in EditLog::cells_of(&self.rules, op) {
            let Some(edits) = self.log.cell(cell) else { continue };
            if edits.ops.len() >= self.config.bake_ops && self.baking.insert(cell) {
                self.pool.push(IDLE - 1, Job::Bake { cell, edits: edits.clone() });
            }
        }
        outcome
    }

    /// Let `volume` of water go at a finest-level cell; see [`Water::pour`].
    pub fn pour(&mut self, at: IVec3, volume: i64) {
        self.water.pour(at, volume, JOIN);
    }

    /// True when water has nothing more to do in the terrain that is loaded.
    pub fn water_at_rest(&self) -> bool {
        self.water.at_rest()
    }

    /// One simulation tick of water. Searches run on the worker pool and their
    /// outcomes are applied at a fixed tick delay; if a worker has not finished
    /// when its tick arrives, this waits so every machine stays identical.
    pub fn step_water(&mut self) {
        self.water_terrain.tick();
        let (volume, log, store, regions, done_water) =
            (&self.volume, &mut self.log, &self.store, &mut self.regions, &self.done_water);
        // Both closures use the kept terrain, one after the other.
        let kept = std::cell::RefCell::new(&mut self.water_terrain);
        let job = self.water.step(
            || {
                let (outcome, made) = done_water.recv().expect("water worker exited");
                kept.borrow_mut().store(made);
                outcome
            },
            |req: &Request| {
                if !req.loaded {
                    let step = 1 << req.lod;
                    let min = (req.origin * CHUNK_SIZE - APRON) * step;
                    let max = ((req.origin + req.size) * CHUNK_SIZE + APRON) * step;
                    load_regions(store.as_deref(), log, regions, CellPos::tier_of(req.lod), min, max);
                }
                kept.borrow_mut().gather(volume, log, req)
            },
        );
        if let Some(job) = job {
            self.water_terrain.launched();
            self.pool.push(URGENT, Job::Water(job));
        }
        self.redraw_water(&mut IntSet::default());
    }

    /// Mesh again the nodes, at every level, in which water surfaces changed,
    /// other than those in `done`.
    fn redraw_water(&mut self, done: &mut IntSet<NodeKey>) {
        let changed = self.water.take_changed();
        if changed.is_empty() {
            return;
        }
        let focus = self.focus.unwrap_or(DVec3::ZERO);
        let mut jobs = Vec::new();
        for (min, max) in changed {
            for level in 0..=MAX_LEVEL {
                let shift = CHUNK_BITS + level as i32;
                // A node also draws the wet columns along its far edges.
                let lo = IVec3::new((min.x - 1) >> shift, min.y >> shift, (min.z - 1) >> shift);
                let mut keys = Vec::new();
                for_box(lo, max >> shift, |pos| keys.push(NodeKey::new(level, pos)));
                for key in keys {
                    let Some(n) = self.nodes.get(&key) else { continue };
                    let (wanted, refine) = (n.wanted, n.refine);
                    if !done.insert(key) {
                        continue;
                    }
                    let version = self.bump(key);
                    if level == 0 {
                        let job = self.remesh_job(key, version, 0);
                        jobs.push((URGENT + 1, job));
                    } else if wanted && !refine {
                        self.nodes.get_mut(&key).unwrap().requested = Some(version);
                        let job = self.node_job(key, version, 0);
                        jobs.push((priority(key, focus, false), job));
                    }
                }
            }
        }
        self.pool.push_all(jobs);
    }

    /// Give a node a new version and return it.
    fn bump(&mut self, key: NodeKey) -> u64 {
        self.next_version += 1;
        let n = self.nodes.get_mut(&key).unwrap();
        n.version = self.next_version;
        if key.level == 0 {
            n.requested = Some(n.version);
        }
        self.drawn_dirty = true;
        n.version
    }

    /// Hand every unsaved edit to the store and write it to disk; in the
    /// background if `wait` is false.
    pub fn save(&mut self, wait: bool) -> std::io::Result<()> {
        let Some(store) = self.store.clone() else { return Ok(()) };
        for pos in self.log.take_dirty() {
            if let Some(cell) = self.log.cell(pos) {
                store.put(pos, cell.encode());
            }
        }
        let water = self.water.take_unsaved().then(|| self.water.encode());
        let write = move || {
            if let Some(bytes) = water {
                store.write_file(WATER_FILE, &bytes)?;
            }
            store.flush().map(|_| ())
        };
        if wait {
            write()
        } else {
            std::thread::spawn(move || {
                if let Err(e) = write() {
                    eprintln!("tungsten: autosave failed: {e}");
                }
            });
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use tungsten_core::Material;
    use tungsten_core::edit::FP;

    fn config() -> StreamConfig {
        StreamConfig { view_distance: 300.0, detail: 1.0, workers: 4, bake_ops: usize::MAX }
    }

    fn settle(world: &mut World, focus: DVec3, events: &mut Vec<MeshEvent>) {
        let start = Instant::now();
        loop {
            world.update(focus, events);
            if world.settled() {
                world.update(focus, events);
                if world.settled() {
                    return;
                }
            }
            assert!(start.elapsed() < Duration::from_secs(120), "streaming did not settle");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A point on the ground near the origin, metres.
    fn ground(world: &World) -> DVec3 {
        DVec3::new(4.0, world.generator().surface_height(4.0, 4.0) as f64, 4.0)
    }

    /// What the renderer would hold after these events: meshes and what is on screen.
    #[derive(Default)]
    struct Screen {
        meshes: IntSet<NodeKey>,
        visible: IntSet<NodeKey>,
    }

    impl Screen {
        fn apply(&mut self, events: &mut Vec<MeshEvent>) {
            for e in events.drain(..) {
                match e {
                    MeshEvent::Upload { key, .. } => {
                        self.meshes.insert(key);
                    }
                    MeshEvent::Remove { key, .. } => {
                        self.meshes.remove(&key);
                    }
                    MeshEvent::Visible { show, hide } => {
                        hide.iter().for_each(|k| assert!(self.visible.remove(k), "{k:?} hidden twice"));
                        show.iter().for_each(|k| assert!(self.visible.insert(*k), "{k:?} shown twice"));
                    }
                }
            }
        }
    }

    /// Every wanted root's volume is drawn exactly once: by itself, or by all
    /// eight children each drawn exactly once.
    fn drawn_once(screen: &Screen, key: NodeKey) -> bool {
        if screen.visible.contains(&key) {
            fn none_below(screen: &Screen, key: NodeKey) -> bool {
                key.level == 0 || key.children().all(|c| !screen.visible.contains(&c) && none_below(screen, c))
            }
            return none_below(screen, key);
        }
        key.level > 0 && key.children().all(|c| drawn_once(screen, c))
    }

    fn roots(world: &World) -> Vec<NodeKey> {
        assert!(!world.roots.is_empty() && world.roots.iter().all(|r| world.nodes[r].wanted));
        world.roots.clone()
    }

    /// Does anything exist that could draw all of this node's volume?
    fn covered(world: &World, key: NodeKey) -> bool {
        world
            .nodes
            .get(&key)
            .is_some_and(|n| n.shown.is_some() || (key.level > 0 && key.children().all(|c| covered(world, c))))
    }

    #[test]
    fn builds_a_tree_around_the_focus_and_follows_it() {
        let mut world = World::new(WorldParams::default(), config(), None);
        let (mut events, mut screen) = (Vec::new(), Screen::default());
        let here = ground(&world);
        settle(&mut world, here, &mut events);
        screen.apply(&mut events);

        let s = world.stats();
        assert!(roots(&world).len() >= 8, "{} roots", roots(&world).len());
        assert!(roots(&world).iter().all(|&r| drawn_once(&screen, r)), "holes or overlaps");
        // Fine detail is near the focus and every level is in use.
        assert!(s.drawn.iter().all(|&n| n > 0), "{:?}", s.drawn);
        let chunk = (here / world.params().chunk_metres() as f64).floor().as_ivec3();
        assert!(world.neighbourhood_loaded(chunk));
        assert!(screen.visible.contains(&NodeKey::new(0, chunk)));
        assert_eq!(screen.meshes.len(), s.meshed);
        assert!(screen.meshes.iter().any(|k| k.level == 0) && screen.meshes.iter().any(|k| k.level == MAX_LEVEL));
        // Every leaf is up to date.
        for (key, n) in &world.nodes {
            assert!(n.wanted, "{key:?} kept for nothing");
            assert!(n.refine || n.shown == Some(n.version), "{key:?} not meshed");
        }
        assert_eq!(s.chunks, world.nodes.keys().filter(|k| k.level == 0).count());

        // Walk in steps: never a hole or an overlap on the way.
        let mut at = here;
        for _ in 0..12 {
            at += DVec3::new(7.0, 0.0, 3.0);
            for _ in 0..4 {
                world.update(at, &mut events);
                screen.apply(&mut events);
                let complete = roots(&world).iter().filter(|&&r| covered(&world, r)).all(|&r| drawn_once(&screen, r));
                assert!(complete, "hole or overlap while moving");
                std::thread::sleep(Duration::from_millis(2));
            }
        }

        // Far away: nothing of the old place is left.
        let far = here + DVec3::new(5000.0, 0.0, -3000.0);
        settle(&mut world, far, &mut events);
        screen.apply(&mut events);
        let near = |k: &NodeKey| distance(*k, far / world.params().cell_size as f64) < 400.0 / 0.25 * 1.5;
        assert!(world.nodes.keys().all(near) && screen.meshes.iter().all(near) && screen.visible.iter().all(near));
        assert_eq!(screen.meshes.len(), world.stats().meshed);
        assert!(roots(&world).iter().all(|&r| drawn_once(&screen, r)));
        assert_eq!(world.volume.chunks.len(), world.nodes.keys().filter(|k| k.level == 0).count());
    }

    fn dig_and_build(world: &mut World) -> Vec<ChunkPos> {
        let here = ground(world) / world.params().cell_size as f64;
        let at = |dx: f64, dy: f64| {
            [((here.x + dx) * FP as f64) as i64, ((here.y + dy) * FP as f64) as i64, (here.z * FP as f64) as i64]
        };
        let mut chunks = Vec::new();
        for k in 0..30 {
            let k = k as f64;
            let ops = [
                Op::Break { center: at(k * 1.5 - 20.0, 2.0 - k * 0.4), radius: 5 * FP, nearest: true },
                Op::Place { center: at(k * 1.5 - 20.0, 9.0), radius: 3 * FP, material: Material::SAND, exclude: None },
            ];
            for op in &ops {
                chunks.extend(world.edit(op).chunks);
            }
        }
        chunks.sort_unstable_by_key(|c| (c.x, c.y, c.z));
        chunks.dedup();
        assert!(chunks.len() >= 4, "the session should span chunks, touched {}", chunks.len());
        chunks
    }

    #[test]
    fn edits_survive_unload_and_restart() {
        let dir = std::env::temp_dir().join(format!("tungsten-test-world-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let params = WorldParams::default();
        let open = || {
            let (store, params) = Store::open(&dir, params).unwrap();
            World::new(params, config(), Some(Arc::new(store)))
        };
        let hashes = |w: &World, chunks: &[ChunkPos]| -> Vec<u64> {
            chunks.iter().map(|c| w.volume.chunks[c].content_hash(0)).collect()
        };
        let mut events = Vec::new();

        let mut world = open();
        let here = ground(&world);
        settle(&mut world, here, &mut events);
        let edited = dig_and_build(&mut world);
        let untouched = World::new(params, config(), None);
        settle(&mut world, here, &mut events);
        let before = hashes(&world, &edited);
        assert!(world.stats().ops > 0 && world.groups.is_empty());

        // The chunks in memory are what the log resolves to from scratch.
        for &pos in &edited {
            let key = NodeKey::new(0, pos);
            let mut block = LocalField::block(key);
            untouched.generator().fill_block(key, &mut block);
            let plain = block.interior();
            log::resolve(&mut block, &world.rules, &world.log.gather_node(key));
            assert!(block.interior() == *world.volume.chunks[&pos], "chunk {pos} differs from its log");
            assert!(plain != *world.volume.chunks[&pos], "chunk {pos} reported as changed but was not");
        }

        // Walk away (chunks unload), come back (they are resolved again).
        settle(&mut world, here + DVec3::X * 3000.0, &mut events);
        assert!(edited.iter().all(|c| !world.volume.chunks.contains_key(c)));
        settle(&mut world, here, &mut events);
        assert_eq!(before, hashes(&world, &edited), "edits changed across unload");

        // Quit and restart: same again from disk.
        world.save(true).unwrap();
        drop(world);
        let mut world = open();
        assert_eq!(world.stats().ops, 0);
        settle(&mut world, here, &mut events);
        assert_eq!(before, hashes(&world, &edited), "edits changed across restart");
        assert!(world.stats().ops > 0);
        drop(world);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edits_reach_the_screen_together_and_at_every_level() {
        let mut world = World::new(WorldParams::default(), config(), None);
        let (mut events, mut screen) = (Vec::new(), Screen::default());
        let here = ground(&world);
        settle(&mut world, here, &mut events);
        screen.apply(&mut events);

        let cells = here / world.params().cell_size as f64;
        let center = [(cells.x * FP as f64) as i64, (cells.y * FP as f64) as i64, (cells.z * FP as f64) as i64];
        let outcome = world.edit(&Op::Break { center, radius: 6 * FP, nearest: true });
        assert!(!outcome.chunks.is_empty());
        // The samples change at once; the meshes follow as one urgent batch.
        assert!(!world.volume.get(cells.floor().as_ivec3()).unwrap().solid());
        settle(&mut world, here, &mut events);
        let urgent: Vec<NodeKey> = events
            .iter()
            .filter_map(|e| match e {
                MeshEvent::Upload { key, urgent: true, .. } => Some(*key),
                _ => None,
            })
            .collect();
        assert!(!urgent.is_empty() && urgent.iter().all(|k| k.level == 0), "{urgent:?}");
        assert!(world.groups.is_empty());
        screen.apply(&mut events);
        assert!(roots(&world).iter().all(|&r| drawn_once(&screen, r)));

        // From far enough away the same hole is drawn by a coarser level.
        let away = here + DVec3::new(60.0, 0.0, 0.0);
        settle(&mut world, away, &mut events);
        screen.apply(&mut events);
        let hole = cells.floor().as_ivec3();
        let shows_hole = screen.visible.iter().find(|k| {
            let rel = hole - k.origin();
            rel.min_element() >= 0 && rel.max_element() < k.size()
        });
        let key = *shows_hole.expect("nothing drawn where the hole is");
        assert!(key.level > 0, "still drawn at the finest level from 60 m");
        let mut block = LocalField::block(key);
        world.generator().fill_block(key, &mut block);
        let generated = block.fill.clone();
        log::resolve(&mut block, &world.rules, &world.log.gather_node(key));
        assert!(block.fill != generated, "the coarse node does not see the edit");
        let n = &world.nodes[&key];
        assert_eq!(n.shown, Some(n.version));
    }

    #[test]
    fn water_is_drawn_follows_edits_and_survives_restart() {
        use tungsten_core::water::BodyId;
        let dir = std::env::temp_dir().join(format!("tungsten-test-water-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let open = || {
            let (store, params) = Store::open(&dir, WorldParams::default()).unwrap();
            World::new(params, config(), Some(Arc::new(store)))
        };
        /// The nodes whose latest mesh draws a body, by level.
        fn drawing(events: &[MeshEvent], body: impl Fn(BodyId) -> bool) -> [usize; LEVELS] {
            let mut latest: IntMap<NodeKey, bool> = IntMap::default();
            for e in events {
                match e {
                    MeshEvent::Upload { key, mesh, .. } => {
                        latest.insert(*key, mesh.water.iter().any(|p| body(p.body)));
                    }
                    MeshEvent::Remove { key, .. } => {
                        latest.remove(key);
                    }
                    MeshEvent::Visible { .. } => {}
                }
            }
            let mut levels = [0; LEVELS];
            latest.iter().filter(|(_, has)| **has).for_each(|(k, _)| levels[k.level as usize] += 1);
            levels
        }
        let rest = |world: &mut World, events: &mut Vec<MeshEvent>, here: DVec3| {
            for _ in 0..600 {
                world.step_water();
                if world.water_at_rest() {
                    break;
                }
            }
            assert!(world.water_at_rest());
            settle(world, here, events);
        };

        let mut world = open();
        let mut events = Vec::new();
        let here = ground(&world);
        settle(&mut world, here, &mut events);
        // The sea is 30 m from here, beyond the finest nodes: every coarser level draws some of it.
        let sea = drawing(&events, |b| b == BodyId::Ocean);
        assert!(sea[1..].iter().all(|&n| n > 0), "sea drawn by level {sea:?}");
        assert_eq!(drawing(&events, |b| b != BodyId::Ocean), [0; LEVELS]);

        // A bowl in the ground with water let go over it.
        let cells = (here / world.params().cell_size as f64).floor().as_ivec3();
        let at = |p: IVec3| [p.x as i64 * FP, p.y as i64 * FP, p.z as i64 * FP];
        world.edit(&Op::Break { center: at(cells - IVec3::Y * 3), radius: 8 * FP, nearest: false });
        world.pour(cells + IVec3::Y * 4, 300 * FP);
        events.clear();
        rest(&mut world, &mut events, here);
        let stats = world.water().stats();
        assert_eq!((stats.ponds, stats.held + stats.moving + stats.drained + stats.lost), (1, 300 * FP));
        let (id, pond) = world.water().ponds().next().map(|(id, p)| (id, p.clone())).unwrap();
        let ponds = drawing(&events, |b| b == BodyId::Pond(id));
        assert!(ponds[0] >= 1 && ponds[1..] == [0; LEVELS - 1], "pond drawn by level {ponds:?}");
        assert!(world.water().levels().any(|l| l == (BodyId::Pond(id), pond.level)));

        // Dug wider, the pond lies lower and is drawn again; from further off a coarser node draws it.
        events.clear();
        world.edit(&Op::Break { center: at(cells + IVec3::new(9, -3, 0)), radius: 8 * FP, nearest: false });
        rest(&mut world, &mut events, here);
        let lower = world.water().ponds().next().unwrap().1.clone();
        assert!(lower.level < pond.level && lower.volume == pond.volume);
        assert!(drawing(&events, |b| b == BodyId::Pond(id))[0] >= 1);
        events.clear();
        settle(&mut world, here + DVec3::new(60.0, 0.0, 0.0), &mut events);
        let far = drawing(&events, |b| b == BodyId::Pond(id));
        assert!(far[0] == 0 && far[1..].iter().sum::<usize>() >= 1, "pond drawn by level {far:?} from 60 m");

        // Quit and restart: the pond is as it was, and drawn.
        world.save(true).unwrap();
        drop(world);
        let mut world = open();
        events.clear();
        settle(&mut world, here, &mut events);
        assert_eq!(world.water().stats(), stats);
        assert!(*world.water().ponds().next().unwrap().1 == lower);
        assert!(drawing(&events, |b| b == BodyId::Pond(id))[0] >= 1);
        drop(world);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn water_on_workers_matches_across_thread_counts() {
        let run = |workers: usize| {
            let mut world = World::new(WorldParams::default(), StreamConfig { workers, ..config() }, None);
            let mut events = Vec::new();
            let here = ground(&world);
            settle(&mut world, here, &mut events);
            let cells = (here / world.params().cell_size as f64).floor().as_ivec3();
            let at = |p: IVec3| [p.x as i64 * FP, p.y as i64 * FP, p.z as i64 * FP];
            world.edit(&Op::Break { center: at(cells - IVec3::Y * 3), radius: 7 * FP, nearest: false });
            world.pour(cells + IVec3::Y * 4, 2000 * FP);
            for tick in 0..200 {
                if tick == 80 {
                    world.edit(&Op::Break { center: at(cells + IVec3::new(8, -3, 0)), radius: 6 * FP, nearest: false });
                }
                world.step_water();
            }
            world.water().encode()
        };
        assert_eq!(run(1), run(4), "worker count changed water state");
    }

    #[test]
    fn baking_changes_nothing_but_the_log() {
        let run = |bake_ops: usize| {
            let mut world = World::new(WorldParams::default(), StreamConfig { bake_ops, ..config() }, None);
            let mut events = Vec::new();
            let here = ground(&world);
            settle(&mut world, here, &mut events);
            let edited = dig_and_build(&mut world);
            settle(&mut world, here, &mut events);
            // Leave and return so every chunk is resolved from the log as it now is.
            settle(&mut world, here + DVec3::X * 3000.0, &mut events);
            settle(&mut world, here, &mut events);
            let hashes: Vec<u64> = edited.iter().map(|c| world.volume.chunks[c].content_hash(0)).collect();
            (hashes, world.stats())
        };
        let (plain, plain_stats) = run(usize::MAX);
        let (baked, baked_stats) = run(8);
        assert_eq!(plain, baked);
        assert_eq!(plain_stats.bakes, 0);
        assert!(baked_stats.bakes > 0 && baked_stats.ops < plain_stats.ops, "{baked_stats:?} against {plain_stats:?}");
    }
}
