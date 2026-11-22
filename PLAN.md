# Tungsten Engine Plan

Companion to `SPEC.md`. The first section describes the engine as it is now (v0.2).
Sections 0 to 12 are the v0.1 plan, kept for the mechanics that have not changed
(materials, meshing rules, the player, rendering, the determinism boundary). Where
they disagree with the v0.2 section, v0.2 is what is built: in particular 4.1 (cell
size), 4.2 and 9.3 (chunks and their lifecycle), 7.2 and 7.3 (how brushes are
applied), 7.5 (sand slump, removed) and 9.6 (saves).

## v0.2: function terrain and LOD (2026-11-21)

Decided with the user: the world is a function (generator plus a log of edit
operations), drawn at 0.25 m next to the player and out to 2 km at lower detail.
Sand is static for now and becomes a pile function with the water phase, which is
next. Old worlds are not kept readable.

### How it works

- **Truth.** Seed plus an edit log. An edit is a shape (`Op`): `Break` (sphere;
  stone gives way in whole shards whose centre is inside it), `Place` (sphere
  of a material, never repainting solid, leaving the player's capsule out) and
  `Clear` (lattice points of a fragment that broke off). Positions are
  fixed-point, 1/256 of the base cell.
- **Levels.** Level L samples every `cell_size * 2^L`; six levels, 0.25 m to
  8 m. A node is a 32-cell chunk of one level. An op is a pure per-sample
  min/max with a function of position, and whether a sample ends up solid
  depends only on the sign of its distance to the shape, so a lattice point
  shared by several levels is solid, and of the same material, in all of them.
- **Log.** Ops are filed in edit cells, each op in every cell it can reach. A
  sample is resolved from its own cell only: generator, then the cell's baked
  overlay, then its ops in order. There are two tiers of cells: 32 m cells for
  levels 0 to 2 and 256 m cells for levels 3 to 5. An op reaches two cells of
  a level beyond its shape, so one tier of 32 m cells put every edit in 27 of
  them.
- **Baking.** A cell with 512 ops is resolved on a worker at each level its
  tier serves and stored as a run-length overlay on the generator; the ops
  are dropped. The samples are identical by construction (tested), so nothing
  is remeshed.
- **Streaming.** An octree by distance. Roots are level-5 nodes within the
  view distance; a node splits while the player is within two node lengths of
  it (`--detail`). Every node job is stateless: generate the padded block,
  resolve the log, mesh. Level-0 jobs also return the chunk, which goes into
  `Volume` for collision, raycasts and editing. An edit changes `Volume` in
  place and remeshes the level-0 chunks from it within the tick; coarser
  nodes re-resolve on workers.
- **Seams.** Coarser nodes mesh one cell beyond their bounds and are pushed
  back in depth by two of their cells, so the finer mesh wins wherever both
  exist. A parent stays on screen until all eight children are ready, and
  children until the parent is; each such swap reaches the renderer as one
  event.
- **Generator** (version 3). Same terrain as v0.1 at every level. Caves are
  generated at levels 0 to 3 only. Where the dirt or sand cover is thinner
  than a coarse cell, the cover material still wins, so bedrock does not show
  through at a distance.
- **Mining.** Dirt and sand: every 6 ticks a ball is cut, sunk 0.1 s of the
  material's dig speed (dirt 2.5 m/s, sand 5 m/s) into the surface along its
  normal. Stone: after 0.75 m of work at 1 m/s the shard under the crosshair
  breaks off with every other shard whose centre is in the brush. The brush
  goes down to 0.25 m.
- **Saves.** Region files of 16^3 edit cells per tier, each cell its ops plus
  baked overlay. World format 2.

### Status

**Verified by running it**

- 45 tests pass; clippy is clean with warnings as errors.
- Golden hashes (generated blocks at all six levels near and 500 km from the
  origin, a 120-op editing session, the same session resolved from the log at
  every level, and the encoded log before and after baking) are identical on
  ARM64 and on x86-64 (under Rosetta), at 1 and 7 threads. The same test
  checks that the log reproduces the edited volume exactly and that baking
  changes no sample.
- Benchmark at the defaults (2048 m, 0.25 m cells, 10 workers): cold load of
  3,400 nodes, 1,683 meshes and 4.0M triangles in 0.18 s. Flying at 60 m/s:
  0.31 ms of main-thread time per tick on average, 1.0 ms worst, and no tick
  spent waiting for terrain.
- The window on Metal at 3024 x 1800: about 1,200 fps standing still, worst
  frame 5 ms; 0.9 to 1.2M triangles in 400 to 500 draws.
- Headless captures: terrain continuous to 2 km with no visible seams from
  the ground; digging, placing sand and stone; the same edits after a reload
  and seen from 40 m at a coarser level.

**Not verified**

- How it feels in the hand. I can only drive it with scripted input.
- Vulkan and DX12. Only Metal has run.
- Baking in a long play session (covered by tests only).

**Known limits**

- Material boundaries are blocky far away: they are hard edges on 4 to 8 m cells.
- Cave mouths close beyond about 250 m, where level 4 takes over.
- Collision and raycasts use level-0 samples, loaded within about 16 m of the
  player; flying faster than terrain loads is allowed, walking waits for it.
- Sand does not move.

## 0. v0.1 status (2026-11-21, historical)

Milestones 0 to 6 are implemented and most of 7. `README.md` has the controls and
commands.

**Verified by running it**

- 38 tests pass; clippy is clean with warnings as errors.
- Golden hashes (generation, an editing session, settled sand, encoded saves) are
  identical on ARM64 and on x86-64 (run under Rosetta), at 1 and 7 threads.
- The window opens on Metal, runs and exits cleanly: about 650 fps at 3024 x 1800 at the
  default view distance, worst steady-state frame 3.6 ms.
- Streaming benchmark: cold load of 5,599 chunks in 0.09 s; flying at 60 m/s costs
  0.15 ms of main-thread time per tick on average. The worst tick was 1.2 to 2.5 ms
  across runs, so the 2 ms budget is met at walking speeds and is marginal at 8x run speed.
- Headless captures of digging, placing each material, sand settling, and a world
  reloaded after quitting (pixel-identical terrain).
- Scripted movement on generated terrain: running 200 ticks in each of 8 directions
  covers the same 24.6 m with no sideways drift. Tests cover walking and running up,
  down and across slopes from 5% to 100%, and sliding off anything steeper than 50 degrees.

**Not verified**

- How it feels in the hand: mouse look, cursor capture, Ctrl + click on macOS. I can
  only drive it with scripted input.
- Vulkan and DX12. Only Metal has run. The CI workflow has never run (no remote yet).

**Where the build differs from the plan below**

| Plan | Built |
|---|---|
| Crisp vertex placement for stone (6) | Not built. Stone's angular look comes from the shard geometry plus true flat shading |
| Faceted shading from screen-space derivatives (6) | Faceted triangles carry their own vertices with the face normal. Derivatives produced speckled edges on thin triangles |
| Vertex material from the fullest corner (6) | From the solid corner most exposed to air, so a thin layer shows as itself |
| Collision against stored chunk meshes (6, 8) | Collision triangles are rebuilt each tick from the samples around the player, with the mesher's own vertex rule. No mesh storage, and no dependence on which meshes have arrived |
| Mesh jobs get a copied block (6, 9.2) | Chunks are shared by reference and copied on a worker; a write to a chunk a worker is reading copies it first |
| Saves compressed (9.6, 12.5) | Run-length coding only, no compression crate. A crater costs a few KB |
| Sand spreads to 4 neighbours | 8 neighbours, with a longer threshold on diagonals, so piles are rounder |
| Player pushed out along every contact normal (8) | Deepest contact first; ground snap and step-down move vertically only; ground too steep to stand on acts as a wall. The first version shoved the player sideways on any descent steeper than 7% |

**Added beyond the plan**: fly mode (double-tap Space), scripted headless capture, the benchmark,
`--exit-after` frame statistics, and lifting the player out when buried by placed or
slumping material.

**Changed on request, 2026-11-21** (generator version 2; version 1 worlds do not open)

- Terrain is one biome banded by height: sand below 0 m, dirt to 40 m, stone above. The
  desert patches and the slope-dependent dirt are gone. Mountains are more frequent
  (a range every 800 m or so, up to about 180 m), and cliffs and overhangs occur only
  above the stone line.
- Materials meet at a hard edge: each quad takes the material of the solid sample it
  is the face of, and vertices are copied per material instead of blending.
- `min_fragment` is a per-world setting beside `cell_size` (default 1 m, `--fragment`).
  After a break, any connected run of one material beside the edit that is open to the
  air and smaller than a cube of that size is removed. Buried pockets are left alone.
- Sand settles into the partly filled samples just above a surface and takes them
  over. Before, they held it up and piles rested a sample above the ground.
- Double-tap Space toggles fly.

Known and left: a settled sand pile can keep a one-sample dimple that shows the ground
beneath, because a pit shallower than the repose angle is stable.

**Section 12 is partly open.** Items 1 and 2 are confirmed; the build uses the default
listed for the rest.

## 1. Decisions made (2026-11-21)

| Topic | Decision |
|---|---|
| World volume | Full 3D field in cubic chunks, streamed on all three axes, no height or depth limit |
| Jagged / smooth | Comes from both the mining brush and the mesh, driven by per-material parameters |
| Sand | Slumps to its angle of repose after edits, then sleeps. No continuous falling-sand sim |
| Multiplayer target | Server-authoritative. World generation and edit application must be bit-identical everywhere; player movement need not be |
| Dependencies | Low-level crates only (wgpu, winit, glam, channels). No engine, physics or noise crates |
| Saves | Per-chunk deltas against the seed-generated chunk, in region files |
| Camera | First person |
| Detail levels | One level of detail inside a fixed radius, fog at the edge. Format leaves room for LOD later |
| Break | Hold to carve, applied every fixed tick, speed set by material hardness |
| Place | Not hold-to-carve: one stamp per click (see 12.1) |
| Brush | Mouse wheel resizes it. Placing is unlimited, no inventory |
| Look | Flat colour per material, one sun light plus ambient, distance fog. No textures or shadows |

## 2. Non-goals for v0.1

Water, LOD, shadows, textures, audio, day/night, inventory, networking, structural collapse
(floating dirt or stone stays where it is), entities other than the player, menus.

## 3. The determinism boundary

The one rule the whole architecture follows: **floats in, integers through.**

| Bit-exact on every machine (the "core") | Free to differ (the "shell") |
|---|---|
| World generation: seed to samples | Input, camera, player physics |
| Edit application: samples + edit op to samples | Meshing, rendering |
| Sand slump simulation | Job scheduling, streaming order |
| Delta encoding and decoding | Frame rate |

- An **edit op** is the unit that crosses the boundary: `{tick, kind, centre, radius,
  material}` with centre and radius in fixed-point lattice units. The aim ray is float;
  the op it produces is quantised; everything applied to the world from there is integer
  maths. Later, edit ops are exactly what the server broadcasts.
- Core results must not depend on thread count or the order chunks happen to load.
- Enforced by golden-hash tests (section 10), run on macOS arm64, Linux x86-64 and
  Windows x86-64.

## 4. World data model

### 4.1 Lattice and samples

The world is an infinite 3D lattice of samples, spaced `cell_size` apart (default 0.5 m).
Each sample is two bytes:

- `fill: u8`, how much solid material sits at the sample. 0 is empty, 255 is full.
  The surface is where fill crosses the halfway point (solid means `fill >= 128`).
- `material: u8`, 0 air, 1 stone, 2 dirt, 3 sand. One material per sample. Canonical
  form: `fill == 0` implies `material == 0`.

Fill is an occupancy value, not a distance. That choice is what makes the rest simple:
mining subtracts fill, placing adds fill, sand slumping moves fill between samples and
conserves the total, and all three are integer operations.

Generated terrain maps signed distance to fill over a band of about two cells either
side of the surface, so the mesher has a smooth gradient to interpolate.

### 4.2 Chunks

- 32 x 32 x 32 cells per chunk, so 16 m at the default cell size. Chunk coordinates are
  `i32` on each axis.
- A chunk is either `Uniform { fill, material }` (all air, or all one solid material:
  no array allocated) or `Dense` (65 KB). Most chunks in range are uniform; only those
  the surface or a cave passes through are dense.
- Positions: sample coordinates are integers. The player position is `f64` world space.
  Rendering is camera-relative (each chunk is drawn at its offset from the camera's
  chunk) so precision does not degrade far from the origin.

### 4.3 Materials

A static table; adding a material later is one row.

| | Stone | Dirt | Sand |
|---|---|---|---|
| Carve mode | Shards | Smooth erosion | Smooth erosion |
| Time to carve at brush centre | 1.2 s | 0.4 s | 0.15 s |
| Mesh sharpness | 1.0 (faceted) | 0.0 (rounded) | 0.0 (rounded) |
| Slumps | no | no | yes, repose about 34 degrees |
| Place shape | Angular shard cluster | Smooth sphere | Sphere, then slumps |

## 5. World generation

Pure function `(seed, chunk coord) -> chunk`. Runs on worker threads.

- **Noise**: our own gradient noise over an integer hash (SplitMix64 for per-layer seeds).
  Bit-exactness: `f32` using only add, subtract, multiply, divide, sqrt and floor, with no
  fused multiply-add and no platform maths functions (sin, pow and friends differ per
  platform). Output is quantised to `u8` fill. See 12.3.
- **Shape**: surface height from layered noise (continents, hills, ridged mountains
  under a mask), plus a 3D noise term for cliffs and overhangs, minus tunnel caves
  (two noise fields intersected) that fade out near the surface and occasionally open.
- **Materials**: stone everywhere by default; a dirt layer 0 to 4 m thick on gentle
  slopes; sand in low basins and in dune regions picked by a low-frequency mask; steep
  slopes expose stone. Generated sand is never steeper than its repose angle.
- **Fast reject**: before sampling 32,768 points, bound the height range over the
  chunk's footprint. Chunks entirely above it are uniform air with no further work.
- **Generator version** is stored in the world header. Any change to generation output
  bumps it, because saved deltas are only meaningful against the terrain they were
  made on.

## 6. Meshing

Surface Nets, one mesh per chunk, on worker threads.

- A job gets an immutable copy of the chunk's samples plus a two-sample apron from its
  26 neighbours (so a chunk can only mesh once all 26 neighbours have data). No locks.
- **Vertices**: one per cell whose eight corners are not all solid or all empty. Signed
  value is `2 * fill - 255`, which is always odd, so there is never an exact-zero corner
  case.
- **Faces**: one quad per lattice edge that crosses the surface, joining the four cells
  around it. Each edge belongs to exactly one chunk. Border vertices are recomputed from
  the same samples on both sides, so seams are watertight by construction.
- **Normals**: from the fill gradient (hence the apron), packed.
- **Material**: each vertex takes the material of its fullest solid corner (ties to the
  lowest id). Colours interpolate across triangles, which blends material borders free.
- **Sharpness** (per material, interpolated per vertex):
  - Placement: 0 puts the vertex at the average of its edge crossings (rounded).
    1 solves for the point that best fits the crossing planes, clamped to the cell
    (crisp edges, the dual-contouring placement). Same mesher, one parameter.
  - Shading: the fragment shader blends the smooth normal toward the true face normal
    (from screen-space derivatives) by sharpness. Stone is faceted, dirt and sand are
    smooth, with no duplicated vertices.
- **Output**: packed 12-byte vertices and `u32` indices for the GPU, plus the plain
  triangle list and a small spatial grid kept on the CPU for chunks near the player
  (collision, section 8).

## 7. Editing mechanics

### 7.1 Aiming

Each fixed tick the game casts a ray from the eye through the fill field (lattice walk,
then refine to the surface) up to a reach of 6 m. The result is the hit point, its
normal, and the sample's material and fill (shown in the debug text). A crosshair and a
faint ring of the brush radius are drawn at the hit.

### 7.2 Breaking (left button held)

One edit op per tick while held, centred on the hit point. As material disappears the
hit point moves deeper, so holding digs a tunnel and sweeping cuts a trench. Each
sample under the brush reacts according to **its own** material:

- **Dirt and sand, smooth erosion.** `fill -= rate(material) * falloff(distance / radius)`.
  The falloff is smooth, so the result is a smooth bowl. Sand's rate is higher, so a
  brush over mixed ground washes the sand out first.
- **Stone, shards.** Stone space is divided into world-anchored, seed-hashed angular
  cells about 1.2 m across (a Voronoi pattern computed in integers). The brush wears down
  every shard it overlaps as a unit: all of a shard's samples lose fill together, at a
  rate set by how close the shard's centre is to the brush centre. When a shard drops
  below half fill it breaks off whole: its volume is subtracted exactly, leaving flat
  faces along the shard boundaries. Wear is stored in the fill values themselves, so
  half-broken stone survives save and load with no extra state.

Because the shard pattern depends only on seed and position, the same op produces the
same break on every machine.

### 7.3 Placing (right button, one stamp per click)

The brush centre is pushed out from the hit point along the surface normal so the new
material attaches to what you aimed at. The stamp is a union: fill is raised to the
shape, never lowered.

- Dirt: a smooth sphere.
- Stone: the union of the shards whose centres fall inside the brush, so placed stone is
  angular and lines up with the pattern it will later break along.
- Sand: a sphere, which wakes the sand simulation and settles into a pile.

A sample takes the placed material only if it was not already solid, so placing never
repaints existing ground. Samples inside the player's capsule are left untouched.

### 7.4 Selection and brush size

Keys 1, 2, 3 select stone, dirt, sand. The mouse wheel changes the radius from 0.5 m
to 4 m in 0.25 m steps (default 1.5 m).

### 7.5 Sand slump

An integer cellular simulation over awake sand samples, at a fixed sub-rate of the
simulation tick (20 Hz).

1. **Fall**: sand moves straight down into any free capacity below it.
2. **Spread**: supported sand moves sideways into a lower neighbour when the local slope
   exceeds the repose threshold.

- Total fill is conserved exactly; transfers are integers.
- Iteration order is fixed (by chunk coordinate, then sample index) and the sideways
  direction order rotates with the tick number to avoid a directional bias.
- **Waking**: an edit wakes sand in and around its bounds. A sample that changes wakes
  its neighbours. A region with no transfers in a tick goes back to sleep.
- Untouched generated sand never simulates. Only woken regions cost anything.
- Sand only simulates where all neighbouring chunks are loaded. If a chunk unloads while
  awake, that flag is saved with it and the settle resumes on load.
- Work per tick is capped; a large avalanche spreads over several ticks instead of
  stalling a frame.
- Limitation accepted for v0.1: one material per sample, so sand cannot enter a partly
  filled dirt or stone sample.

### 7.6 From edit to pixels

1. The op is applied to every chunk it touches, in the simulation tick.
2. Those chunks, and neighbours whose apron changed, are queued in an urgent lane ahead
   of all streaming work.
3. The new meshes for one op are swapped in together in the same frame, so there is
   never a crack between an updated chunk and a stale neighbour.

Target: an edit is visible within two frames.

## 8. Player

Fixed 60 Hz simulation tick with an accumulator; rendering interpolates between the last
two states, so movement is smooth at any frame rate and identical in feel at 60 or 120.

| | Value |
|---|---|
| Capsule | radius 0.35 m, height 1.8 m (eye 1.62 m); crouched 1.2 m (eye 1.0 m) |
| Speed | walk 4.5 m/s, run 7.5 m/s (Shift), crouch 2.0 m/s (Ctrl, held) |
| Jump | 1.25 m apex, with 0.1 s of coyote time and 0.1 s of input buffering |
| Gravity | 20 m/s² |
| Slopes | walkable up to 50 degrees, steeper slides |
| Step-up | 0.3 m, which matters on faceted stone |

- **Collision**: capsule against the CPU-side triangles of nearby chunks,
  collide-and-slide. What you stand on is exactly what you see. The player is held in
  place until the chunks around them have meshed (spawn and teleport). See 12.4.
- **Edits under the player**: digging the ground out drops you; material is never placed
  inside your capsule. Standing up from a crouch is blocked while there is no headroom.
- **Mouse**: captured for look; Esc releases it; a click recaptures. Ctrl + left click
  must reach the game as "crouch and break" (macOS otherwise treats it as a right click).
- **Spawn**: the surface above the origin column, found by scanning the generated field.

## 9. Runtime architecture

### 9.1 Crates

| Crate | Contents | Depends on |
|---|---|---|
| `tungsten-core` | Coordinates, fixed-point, hash, noise, generation, chunk storage, edit ops, sand sim, mesher, delta codec | glam, bytemuck |
| `tungsten-stream` | Job pool, chunk lifecycle, load/unload, region files | core, crossbeam-channel |
| `tungsten-render` | wgpu device, terrain pipeline, GPU buffer arena, debug text, brush outline | core, wgpu 30 |
| `tungsten-game` (binary) | winit loop, input, player, glue | all, winit 0.30, pollster |

`core` and `stream` have no GPU or window dependency, so a future server binary is those
two crates plus networking.

### 9.2 Threads

- **Main thread**: window events, simulation ticks (player, edits, sand), render.
- **Worker pool** (cores minus two): generate, mesh, compress and write saves. Jobs carry
  a priority (urgent edits, then distance to the player, with chunks in view first) and
  are dropped if their chunk left range before they ran.
- Workers never touch live world state: jobs take immutable snapshots and send results
  back over a channel. The main thread applies results under a per-frame time budget.

### 9.3 Chunk lifecycle

```
Unloaded -> Generating -> Loaded -> Meshing -> Meshed <-> Dirty (edited, remeshing)
                (seed, then saved delta)   (needs all 26 neighbours Loaded)
Meshed/Loaded -> Saving (only if modified) -> Unloaded
```

- Data is loaded one chunk further out than meshes, since meshing needs neighbours.
- Load volume: 12 chunks horizontally, 6 vertically (192 m by 96 m) by default, with a
  one-chunk hysteresis before unloading so walking along a boundary does not thrash.
- Uniform chunks surrounded by matching uniform chunks skip meshing entirely.

### 9.4 Rendering

- wgpu, so Metal on the Mac, Vulkan and DX12 elsewhere, from one codebase. No
  backend-specific features in v0.1.
- Meshes live in a few large GPU buffers with sub-allocation; one draw per visible chunk.
- CPU frustum culling per chunk, drawn front to back. Reverse-Z depth.
- One terrain shader: material colour, sun plus ambient, sharpness-blended normal, fog
  that reaches full strength at the mesh radius so pop-in is hidden.
- VSync on by default, with a toggle to uncap for profiling.

### 9.5 Debug view (F3)

Own bitmap font, one small pipeline. Shows: fps and frame time (average and worst 1%),
position (world, chunk, local), facing, velocity, grounded/crouching, selected material,
brush radius, aimed material and fill, chunks by state, job queue depths, draw calls,
triangles, awake sand samples, tick rate, seed.

### 9.6 Persistence

- **What is saved per chunk**: a bitmap of samples that have ever been edited, and the
  current fill and material of those samples. Loading is generate, then overlay.
  Untouched chunks cost nothing on disk.
- **Encoding**: run-length bitmap, fill and material as separate streams, then
  compressed (12.5).
- **Region files**: 16 x 16 x 16 chunks per file, with an offset table. A chunk is written
  when it unloads modified, on a 30 s autosave, and on quit. Writes never leave a
  half-written region on a crash.
- **World header**: format version, seed, cell size, chunk size, generator version.
- **Player file**: position, look direction, selected material, brush radius.

## 10. Testing and performance

- **Golden hashes**: a fixed set of (seed, chunk) pairs hashes to committed values. A
  scripted list of edit ops and sand ticks applied to them hashes to committed values.
  Same on all three platforms, at 1 thread and at 12.
- **Mesher**: border vertices of adjacent chunks match exactly; no open edges.
- **Sand**: total fill is conserved; a dropped column settles to a cone of 30 to 38
  degrees; a settled region goes to sleep.
- **Saves**: edit, save, reload, compare sample for sample.
- **Headless benchmark**: generate and mesh a fixed fly-through with no window, to track
  chunk throughput over time.

Targets on the M6 MacBook Pro, to be measured at milestone 2 and revised with data:

| | Target |
|---|---|
| Frame rate at default view distance | 120 fps |
| Main-thread time spent on streaming | under 2 ms in any frame |
| Edit to visible | 2 frames or fewer |
| Flying at run speed | no visible holes in the terrain ahead |

## 11. Milestones

Each one ends with something you can run and judge.

| # | Milestone | You can |
|---|---|---|
| 0 | **Scaffold**: workspace, window, wgpu, fixed-tick loop, fly camera, debug text, CI on three platforms | Fly around an empty sky with an fps counter |
| 1 | **Core**: coordinates, hash, noise, generation, Surface Nets, golden-hash tests | See one generated chunk, with material colours |
| 2 | **Streaming**: job pool, lifecycle, neighbour-aware meshing, culling, fog, buffer arena, headless benchmark | Fly forever over seamless terrain; first performance numbers |
| 3 | **Player**: capsule collision, walk, run, jump, crouch, spawn | Walk the world in first person |
| 4 | **Editing**: aim ray, erosion brush for all three materials, place stamp, 1-3 keys, wheel resize, urgent remesh, sharpness shading | Dig and build, with stone faceted and dirt smooth |
| 5 | **Material behaviour**: stone shards with crisp vertex placement, sand slump | Break stone into angular chunks and watch sand settle |
| 6 | **Persistence**: deltas, region files, autosave, header, player file | Quit and come back to the same world |
| 7 | **Hardening**: cross-platform determinism run, profiling passes, Vulkan test on the Ubuntu tower, tuning | Call it v0.1 |

Milestone 4 deliberately ships plain erosion for stone first, so editing, remeshing and
shading are proven before the two most novel mechanics arrive in milestone 5.

## 12. Needs your sign-off

Things the spec and your answers do not settle. Each has the default this plan assumes.

1. **Placing. Confirmed 2026-11-21:** one stamp per click, skipping the space the player
   stands in and never overwriting existing terrain.
   Still assumed: you cannot place in mid-air (the ray must hit something within reach).
2. **"Adjustable mesh resolution." Confirmed 2026-11-21:** a per-world setting (`cell_size`,
   chosen at world creation), because the lattice is the saved data. The alternative is
   a live graphics setting that meshes every second sample, which is the first step of
   LOD and was scoped out.
3. **Float generation.** Assumed `f32` with a restricted operation set, policed by
   golden hashes. The alternative is fixed-point integers throughout generation: immune
   by construction, slower to write and to run.
4. **Collision against triangles** (assumed) versus directly against the fill field. The
   field needs no mesh, so it is lighter for a future server, but is less exact on
   faceted stone.
5. **Small crates.** Is a pure-Rust compression crate (`lz4_flex`) acceptable under
   "low-level crates only", or should saves use our own run-length coding alone? Same
   question for `bytemuck` (vertex casting) and a benchmark harness as a dev dependency.
6. **Save location.** Assumed `./worlds/<name>/` beside the binary during development.
7. **Crouch.** Assumed hold (not toggle), with no edge guard.
8. **Numbers.** Every value in this document (cell size, chunk size, view distance,
   speeds, carve times, brush range, reach, repose angle) is a proposed starting point.
9. **Non-goals.** The list in section 2, in particular no water and no collapse of
   unsupported dirt or stone.

Crate versions were pinned at milestone 0: wgpu 30, winit 0.30, glam 0.34.
