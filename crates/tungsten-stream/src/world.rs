use std::collections::VecDeque;
use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, unbounded};
use glam::IVec3;
use tungsten_core::codec::{self, Decoded};
use tungsten_core::coords::{CHUNK_BITS, CHUNK_MASK, chunk_origin, index, local_of};
use tungsten_core::edit::{self, EditOp, EditOutcome, EditRules};
use tungsten_core::hash::{IntMap, IntSet};
use tungsten_core::mesh::{ChunkMesh, PaddedBlock, mesh_block};
use tungsten_core::sand::SandSim;
use tungsten_core::worldgen::Generator;
use tungsten_core::{Chunk, ChunkPos, Volume, WorldParams};

use crate::pool::Pool;
use crate::store::Store;

#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    /// Horizontal radius, in chunks, inside which chunks are meshed.
    pub radius_h: i32,
    /// Vertical radius, in chunks, inside which chunks are meshed.
    pub radius_v: i32,
    pub workers: usize,
    /// Chunks considered for meshing per update at most; bounds main-thread time.
    pub mesh_starts_per_update: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self { radius_h: 12, radius_v: 6, workers: cores.saturating_sub(2).max(1), mesh_starts_per_update: 512 }
    }
}

/// A change to what should be drawn.
pub enum MeshEvent {
    /// `urgent` marks the result of an edit, which must reach the screen
    /// together with the rest of its group and without delay.
    Upload {
        pos: ChunkPos,
        mesh: ChunkMesh,
        urgent: bool,
    },
    Remove {
        pos: ChunkPos,
        urgent: bool,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub loading: usize,
    pub loaded: usize,
    pub dense: usize,
    pub meshed: usize,
    pub awaiting_mesh: usize,
    pub jobs_queued: usize,
    pub jobs_running: usize,
}

enum Job {
    Generate(ChunkPos),
    /// Free unloaded chunks off the main thread.
    Release(Vec<Arc<Chunk>>),
    /// The chunk and its 26 neighbours, indexed as `neighbours()` yields them.
    Mesh {
        pos: ChunkPos,
        version: u32,
        group: u32,
        chunks: Box<[Arc<Chunk>; 27]>,
    },
}

enum Done {
    /// A generated chunk, with its saved edits already overlaid.
    Generated(ChunkPos, Chunk, Option<Decoded>),
    Meshed {
        pos: ChunkPos,
        version: u32,
        group: u32,
        mesh: ChunkMesh,
    },
}

struct Entry {
    loaded: bool,
    /// Bumped whenever this chunk's block (own samples or apron) changes.
    version: u32,
    /// Version a mesh job was last started for.
    requested: Option<u32>,
    /// Version currently on screen.
    shown: Option<u32>,
    has_mesh: bool,
}

/// Meshes of one edit, held back until all have arrived so they swap in together.
struct Group {
    remaining: usize,
    meshes: Vec<(ChunkPos, u32, ChunkMesh)>,
}

const URGENT: u64 = 0;
/// Main-thread time per update for accepting finished streaming work.
const COLLECT_BUDGET: std::time::Duration = std::time::Duration::from_micros(1500);

pub struct World {
    params: WorldParams,
    rules: EditRules,
    config: StreamConfig,
    generator: Arc<Generator>,
    volume: Volume,
    sand: SandSim,
    store: Option<Arc<Store>>,
    /// Chunks changed since they were last handed to the store.
    unsaved: IntSet<ChunkPos>,
    entries: IntMap<ChunkPos, Entry>,
    pool: Pool<Job>,
    done: Receiver<Done>,
    /// Results of edit remeshes, kept apart so streaming cannot delay them.
    done_urgent: Receiver<Done>,
    center: Option<ChunkPos>,
    /// Offsets to mesh, nearest first.
    mesh_offsets: Vec<IVec3>,
    mesh_set: IntSet<IVec3>,
    /// Offsets to hold data for: the mesh set plus its neighbours.
    data_offsets: Vec<IVec3>,
    /// Offsets kept loaded once present: one ring of hysteresis beyond data.
    keep_set: IntSet<IVec3>,
    /// Chunks whose mesh may need (re)starting.
    check: VecDeque<ChunkPos>,
    queued_check: IntSet<ChunkPos>,
    groups: IntMap<u32, Group>,
    next_group: u32,
    loading: usize,
    meshed: usize,
}

fn neighbours() -> impl Iterator<Item = IVec3> {
    (-1..=1).flat_map(|y| (-1..=1).flat_map(move |z| (-1..=1).map(move |x| IVec3::new(x, y, z))))
}

fn dilate(set: &IntSet<IVec3>) -> IntSet<IVec3> {
    set.iter().flat_map(|&p| neighbours().map(move |n| p + n)).collect()
}

fn dist_key(offset: IVec3) -> u64 {
    // Vertical distance counts double: the load volume is flattened.
    let d = offset * IVec3::new(1, 2, 1);
    1 + d.length_squared() as u64
}

fn sorted(set: &IntSet<IVec3>) -> Vec<IVec3> {
    let mut v: Vec<IVec3> = set.iter().copied().collect();
    v.sort_by_key(|&o| (dist_key(o), o.x, o.y, o.z));
    v
}

impl World {
    /// With a `store`, saved edits are loaded with their chunks and edits are
    /// written back when chunks unload and on [`save`](Self::save).
    pub fn new(params: WorldParams, config: StreamConfig, store: Option<Arc<Store>>) -> Self {
        let generator = Arc::new(Generator::new(params));
        let (tx, done): (Sender<Done>, _) = unbounded();
        let (tx_urgent, done_urgent): (Sender<Done>, _) = unbounded();
        let worker_gen = generator.clone();
        let worker_store = store.clone();
        let pool = Pool::new(config.workers, move |job: Job| {
            let result = match job {
                Job::Release(chunks) => {
                    drop(chunks);
                    return;
                }
                Job::Generate(pos) => {
                    let mut chunk = worker_gen.generate(pos);
                    let saved = worker_store.as_ref().and_then(|s| s.get(pos)).and_then(|bytes| {
                        // A delta that cannot be read leaves the generated chunk as it is.
                        codec::apply(&mut chunk, &bytes)
                            .map_err(|e| eprintln!("tungsten: saved chunk {pos} is unreadable ({e}); regenerated"))
                            .ok()
                    });
                    Done::Generated(pos, chunk, saved)
                }
                Job::Mesh { pos, version, group, chunks } => {
                    // Copying 27 chunks into a block is the expensive part of
                    // starting a mesh, so it happens here, not on the main thread.
                    let block = PaddedBlock::gather(pos, |p| {
                        let d = p - pos + 1;
                        Some(&*chunks[((d.y * 3 + d.z) * 3 + d.x) as usize])
                    })
                    .expect("all 27 chunks are present");
                    Done::Meshed { pos, version, group, mesh: mesh_block(&block) }
                }
            };
            // The receiver is gone only during shutdown.
            let urgent = matches!(result, Done::Meshed { group, .. } if group != 0);
            let _ = if urgent { tx_urgent.send(result) } else { tx.send(result) };
        });

        let (rh, rv) = (config.radius_h.max(1), config.radius_v.max(1));
        let mut mesh_set = IntSet::default();
        for y in -rv..=rv {
            for z in -rh..=rh {
                for x in -rh..=rh {
                    let (fx, fy, fz) = (x as f32 / rh as f32, y as f32 / rv as f32, z as f32 / rh as f32);
                    if fx * fx + fy * fy + fz * fz <= 1.0 {
                        mesh_set.insert(IVec3::new(x, y, z));
                    }
                }
            }
        }
        let data_set = dilate(&mesh_set);
        let keep_set = dilate(&data_set);
        Self {
            params,
            rules: EditRules::new(&params),
            config,
            generator,
            volume: Volume::new(),
            sand: SandSim::new(),
            store,
            unsaved: IntSet::default(),
            entries: IntMap::default(),
            pool,
            done,
            done_urgent,
            center: None,
            mesh_offsets: sorted(&mesh_set),
            mesh_set,
            data_offsets: sorted(&data_set),
            keep_set,
            check: VecDeque::new(),
            queued_check: IntSet::default(),
            groups: IntMap::default(),
            next_group: 1,
            loading: 0,
            meshed: 0,
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

    /// True once the chunk and its 26 neighbours hold data.
    pub fn neighbourhood_loaded(&self, pos: ChunkPos) -> bool {
        neighbours().all(|n| self.volume.chunks.contains_key(&(pos + n)))
    }

    pub fn stats(&self) -> Stats {
        let (jobs_queued, jobs_running) = self.pool.load();
        Stats {
            loading: self.loading,
            loaded: self.volume.chunks.len(),
            dense: self.volume.chunks.values().filter(|c| !c.is_uniform()).count(),
            meshed: self.meshed,
            awaiting_mesh: self.check.len(),
            jobs_queued,
            jobs_running,
        }
    }

    /// True when every chunk in range is loaded and every mesh is current.
    pub fn settled(&self) -> bool {
        let (q, r) = self.pool.load();
        self.center.is_some()
            && q == 0
            && r == 0
            && self.loading == 0
            && self.check.is_empty()
            && self.done.is_empty()
            && self.done_urgent.is_empty()
    }

    fn queue_check(&mut self, pos: ChunkPos) {
        if self.queued_check.insert(pos) {
            self.check.push_back(pos);
        }
    }

    /// Advance streaming around `focus`. Mesh changes are appended to `events`.
    pub fn update(&mut self, focus: ChunkPos, events: &mut Vec<MeshEvent>) {
        if self.center != Some(focus) {
            self.recenter(focus, events);
        }
        self.collect(events);
        self.start_meshes(events);
    }

    fn recenter(&mut self, focus: ChunkPos, events: &mut Vec<MeshEvent>) {
        self.center = Some(focus);

        // Unload what fell outside the hysteresis ring.
        let keep = &self.keep_set;
        let drop: Vec<ChunkPos> = self.entries.keys().filter(|&&p| !keep.contains(&(p - focus))).copied().collect();
        let mut released = Vec::new();
        for pos in drop {
            let e = self.entries.remove(&pos).unwrap();
            if e.loaded {
                self.persist(pos, true);
                released.extend(self.volume.chunks.remove(&pos));
                self.volume.touched.remove(&pos);
            } else {
                self.loading -= 1;
            }
            if e.has_mesh {
                self.meshed -= 1;
                events.push(MeshEvent::Remove { pos, urgent: false });
            }
        }

        if !released.is_empty() {
            self.pool.push(u64::MAX, Job::Release(released));
        }

        // Re-prioritise what is still queued; drop jobs for unloaded chunks.
        let entries = &self.entries;
        self.pool.rekey(|job| match job {
            Job::Release(_) => Some(u64::MAX),
            Job::Generate(pos) => entries.contains_key(pos).then(|| dist_key(*pos - focus)),
            Job::Mesh { pos, group, .. } => {
                if *group != 0 {
                    Some(URGENT)
                } else {
                    entries.contains_key(pos).then(|| dist_key(*pos - focus))
                }
            }
        });

        let mut wanted = Vec::new();
        for &off in &self.data_offsets {
            let pos = focus + off;
            if let std::collections::hash_map::Entry::Vacant(slot) = self.entries.entry(pos) {
                slot.insert(Entry { loaded: false, version: 0, requested: None, shown: None, has_mesh: false });
                self.loading += 1;
                wanted.push((dist_key(off), Job::Generate(pos)));
            }
        }
        self.pool.push_all(wanted);
        // Chunks newly inside the mesh radius, or whose job was dropped.
        for i in 0..self.mesh_offsets.len() {
            let pos = focus + self.mesh_offsets[i];
            if self.entries.get(&pos).is_some_and(|e| e.loaded && e.requested != Some(e.version)) {
                self.queue_check(pos);
            }
        }
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
            Done::Generated(pos, chunk, saved) => {
                let Some(e) = self.entries.get_mut(&pos) else { return };
                if e.loaded {
                    return;
                }
                e.loaded = true;
                self.loading -= 1;
                self.volume.insert(pos, chunk);
                if let Some(saved) = saved {
                    self.volume.touched.insert(pos, saved.touched);
                    let origin = chunk_origin(pos);
                    for i in saved.awake {
                        let i = i as i32;
                        let local = IVec3::new(i & CHUNK_MASK, i >> (2 * CHUNK_BITS), (i >> CHUNK_BITS) & CHUNK_MASK);
                        self.sand.wake(origin + local);
                    }
                }
                for n in neighbours() {
                    self.queue_check(pos + n);
                }
            }
            Done::Meshed { pos, version, group, mesh } => {
                if group == 0 {
                    self.show(pos, version, mesh, false, events);
                    return;
                }
                let Some(g) = self.groups.get_mut(&group) else { return };
                g.meshes.push((pos, version, mesh));
                g.remaining -= 1;
                if g.remaining == 0 {
                    for (pos, version, mesh) in self.groups.remove(&group).unwrap().meshes {
                        self.show(pos, version, mesh, true, events);
                    }
                }
            }
        }
    }

    fn show(&mut self, pos: ChunkPos, version: u32, mesh: ChunkMesh, urgent: bool, events: &mut Vec<MeshEvent>) {
        let Some(e) = self.entries.get_mut(&pos) else { return };
        // Never replace a newer mesh with an older one.
        if e.shown.is_some_and(|s| s >= version) {
            return;
        }
        e.shown = Some(version);
        match (mesh.is_empty(), e.has_mesh) {
            (true, true) => {
                e.has_mesh = false;
                self.meshed -= 1;
                events.push(MeshEvent::Remove { pos, urgent });
            }
            (true, false) => {}
            (false, had) => {
                if !had {
                    e.has_mesh = true;
                    self.meshed += 1;
                }
                events.push(MeshEvent::Upload { pos, mesh, urgent });
            }
        }
    }

    /// Start a mesh job for `pos` if it is in range, ready and out of date.
    fn start_mesh(&mut self, pos: ChunkPos, group: u32, events: &mut Vec<MeshEvent>) -> bool {
        let Some(center) = self.center else { return false };
        if !self.mesh_set.contains(&(pos - center)) {
            return false;
        }
        let Some(e) = self.entries.get(&pos) else { return false };
        if !e.loaded || e.requested == Some(e.version) {
            return false;
        }
        let version = e.version;
        let mut chunks = Vec::with_capacity(27);
        for n in neighbours() {
            let Some(c) = self.volume.chunks.get(&(pos + n)) else { return false };
            chunks.push(c.clone());
        }
        self.entries.get_mut(&pos).unwrap().requested = Some(version);
        // Open sky and solid rock, the common cases, need no job at all.
        let side = |c: &Chunk| match c {
            Chunk::Uniform(s) => Some(s.solid()),
            Chunk::Dense(_) => None,
        };
        if let Some(first) = side(&chunks[0]) {
            if chunks.iter().all(|c| side(c) == Some(first)) {
                self.show(pos, version, ChunkMesh::default(), group != 0, events);
                return false;
            }
        }
        let chunks: Box<[Arc<Chunk>; 27]> = chunks.into_boxed_slice().try_into().ok().expect("27 neighbours");
        let key = if group != 0 { URGENT } else { dist_key(pos - center) };
        self.pool.push(key, Job::Mesh { pos, version, group, chunks });
        true
    }

    fn start_meshes(&mut self, events: &mut Vec<MeshEvent>) {
        // Bound the work of one update, whether or not a check starts a job.
        for _ in 0..self.config.mesh_starts_per_update {
            let Some(pos) = self.check.pop_front() else { break };
            self.queued_check.remove(&pos);
            self.start_mesh(pos, 0, events);
        }
    }

    /// Record that samples of `chunks` changed and remesh everything whose
    /// block includes them, as one group that swaps in together.
    pub fn chunks_edited(&mut self, chunks: &[ChunkPos], events: &mut Vec<MeshEvent>) {
        let mut affected: IntSet<ChunkPos> = IntSet::default();
        for &c in chunks {
            self.unsaved.insert(c);
            affected.extend(neighbours().map(|n| c + n));
        }
        let group = self.next_group;
        self.next_group = self.next_group.wrapping_add(1).max(1);
        let mut started = 0;
        for pos in affected {
            if let Some(e) = self.entries.get_mut(&pos) {
                e.version += 1;
            }
            if self.start_mesh(pos, group, events) {
                started += 1;
            }
        }
        if started > 0 {
            self.groups.insert(group, Group { remaining: started, meshes: Vec::with_capacity(started) });
        }
    }

    /// Apply an edit and remesh what it changed.
    pub fn edit(&mut self, op: &EditOp, events: &mut Vec<MeshEvent>) -> EditOutcome {
        let outcome = edit::apply(&mut self.volume, &self.rules, op);
        if !outcome.chunks.is_empty() {
            self.chunks_edited(&outcome.chunks, events);
            // Loose material in and just around the edit may now be unsupported.
            self.sand.wake_region(&self.volume, outcome.min - 1, outcome.max + 1);
        }
        outcome
    }

    /// Hand a chunk's edits to the store if they have changed since last time.
    /// `unloading` also takes its settling sand with it.
    fn persist(&mut self, pos: ChunkPos, unloading: bool) {
        let awake = if unloading { self.sand.take_chunk(pos) } else { self.sand.awake_in(pos) };
        let changed = self.unsaved.remove(&pos);
        let (Some(store), Some(chunk)) = (&self.store, self.volume.chunks.get(&pos)) else { return };
        // Sand still settling must resume after a reload even if no sample changed yet.
        if !changed && !(unloading && !awake.is_empty()) {
            return;
        }
        let awake: Vec<u16> = awake.iter().map(|&p| index(local_of(p)) as u16).collect();
        let empty;
        let touched = match self.volume.touched.get(&pos) {
            Some(t) => t,
            None => {
                empty = Box::new([0u64; tungsten_core::CHUNK_VOLUME / 64]);
                &empty
            }
        };
        store.put(pos, codec::encode(chunk, touched, &awake));
    }

    /// Hand every unsaved edit to the store and write it to disk; in the
    /// background if `wait` is false.
    pub fn save(&mut self, wait: bool) -> std::io::Result<()> {
        let Some(store) = self.store.clone() else { return Ok(()) };
        let pending: Vec<ChunkPos> = self.unsaved.iter().copied().collect();
        for pos in pending {
            self.persist(pos, false);
        }
        if wait {
            store.flush().map(|_| ())
        } else {
            std::thread::spawn(move || {
                if let Err(e) = store.flush() {
                    eprintln!("tungsten: autosave failed: {e}");
                }
            });
            Ok(())
        }
    }

    /// One step of the sand simulation.
    pub fn step_sand(&mut self, events: &mut Vec<MeshEvent>) {
        let mut changed = Vec::new();
        self.sand.step(&mut self.volume, &mut changed);
        if !changed.is_empty() {
            self.chunks_edited(&changed, events);
        }
    }

    /// Samples of loose material currently settling.
    pub fn sand_awake(&self) -> usize {
        self.sand.awake()
    }

    pub fn volume_mut(&mut self) -> &mut Volume {
        &mut self.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn settle(world: &mut World, focus: ChunkPos, events: &mut Vec<MeshEvent>) {
        let start = Instant::now();
        loop {
            world.update(focus, events);
            if world.settled() {
                world.update(focus, events);
                if world.settled() {
                    return;
                }
            }
            assert!(start.elapsed() < Duration::from_secs(60), "streaming did not settle");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn loads_meshes_and_unloads() {
        let config = StreamConfig { radius_h: 3, radius_v: 2, workers: 4, mesh_starts_per_update: 16 };
        let mut world = World::new(WorldParams::default(), config, None);
        let mut events = Vec::new();
        settle(&mut world, IVec3::ZERO, &mut events);
        let s = world.stats();
        assert_eq!(s.loading, 0);
        assert_eq!(s.loaded, world.data_offsets.len());
        let uploads = events.iter().filter(|e| matches!(e, MeshEvent::Upload { .. })).count();
        assert!(uploads > 10 && uploads == s.meshed, "{uploads} uploads, {} meshed", s.meshed);
        // Every mesh in range is current.
        for off in &world.mesh_offsets {
            let e = &world.entries[off];
            assert_eq!(e.shown, Some(e.version), "chunk {off} not meshed");
        }

        // Move far away: everything old unloads, and removals match uploads.
        events.clear();
        let far = IVec3::new(40, 0, 0);
        settle(&mut world, far, &mut events);
        let removed = events.iter().filter(|e| matches!(e, MeshEvent::Remove { .. })).count();
        assert_eq!(removed, uploads);
        assert!(world.volume.chunks.keys().all(|&p| world.keep_set.contains(&(p - far))));
        assert_eq!(world.stats().loaded, world.data_offsets.len());
    }

    #[test]
    fn edits_and_settling_sand_survive_unload_and_restart() {
        use tungsten_core::edit::{EditKind, EditOp, FP};
        let dir = std::env::temp_dir().join(format!("tungsten-test-world-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let config = StreamConfig { radius_h: 2, radius_v: 2, workers: 4, mesh_starts_per_update: 64 };
        let params = WorldParams::default();
        let open = || {
            let (store, params) = Store::open(&dir, params).unwrap();
            World::new(params, config, Some(Arc::new(store)))
        };
        let hash = |w: &World, pos: ChunkPos| w.volume.chunks[&pos].content_hash(0);
        let mut events = Vec::new();

        let mut world = open();
        settle(&mut world, IVec3::ZERO, &mut events);
        let untouched = hash(&world, IVec3::ZERO);
        // Dig, then drop a ball of sand that is still settling when we leave.
        let ground = world.generator().surface_height(4.0, 4.0) as f64 / params.cell_size as f64;
        let at = |dy: f64| [8 * FP, ((ground + dy) * FP as f64) as i64, 8 * FP];
        for tick in 0..40 {
            let op =
                EditOp { kind: EditKind::Break, center: at(0.0), radius: 4 * FP, material: 0, tick, exclude: None };
            world.edit(&op, &mut events);
        }
        let op =
            EditOp { kind: EditKind::Place, center: at(8.0), radius: 3 * FP, material: 3, tick: 40, exclude: None };
        let outcome = world.edit(&op, &mut events);
        world.step_sand(&mut events);
        assert!(world.sand_awake() > 0);
        let edited: Vec<ChunkPos> = outcome.chunks.clone();
        let before: Vec<u64> = edited.iter().map(|&c| hash(&world, c)).collect();
        let awake_before = world.sand_awake();

        // Walk away (chunks unload), come back (they reload from the store).
        settle(&mut world, IVec3::new(30, 0, 0), &mut events);
        assert_eq!(world.sand_awake(), 0);
        assert!(world.volume.touched.is_empty());
        settle(&mut world, IVec3::ZERO, &mut events);
        let after: Vec<u64> = edited.iter().map(|&c| hash(&world, c)).collect();
        assert_eq!(before, after, "edits changed across unload");
        assert_eq!(world.sand_awake(), awake_before, "settling sand was forgotten");

        // Quit and restart: same again from disk.
        world.save(true).unwrap();
        drop(world);
        let mut world = open();
        settle(&mut world, IVec3::ZERO, &mut events);
        let restarted: Vec<u64> = edited.iter().map(|&c| hash(&world, c)).collect();
        assert_eq!(before, restarted, "edits changed across restart");
        assert_eq!(world.sand_awake(), awake_before);
        assert!(edited.iter().any(|&c| c == IVec3::ZERO || hash(&world, c) != untouched));
        drop(world);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn edits_remesh_as_a_group() {
        let config = StreamConfig { radius_h: 2, radius_v: 2, workers: 4, mesh_starts_per_update: 64 };
        let mut world = World::new(WorldParams::default(), config, None);
        let mut events = Vec::new();
        settle(&mut world, IVec3::ZERO, &mut events);
        events.clear();

        // Fill the corner sample shared by eight chunks' aprons.
        let p = IVec3::ZERO;
        world.volume_mut().set(p, tungsten_core::Sample { fill: 255, mat: 1 });
        world.volume_mut().set(p + IVec3::Y * 40, tungsten_core::Sample { fill: 255, mat: 1 });
        world.chunks_edited(&[IVec3::ZERO, IVec3::new(0, 1, 0)], &mut events);
        settle(&mut world, IVec3::ZERO, &mut events);
        assert!(events.iter().any(|e| matches!(e, MeshEvent::Upload { pos, .. } if *pos == IVec3::new(0, 1, 0))));
        assert!(world.groups.is_empty());
    }
}
