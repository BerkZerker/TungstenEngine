# Tungsten Engine

A procedurally generated, chunk-streamed, editable world engine in Rust.
`SPEC.md` is the brief, `PLAN.md` the design and its current status.

## Run

```
./start.sh
```

It builds an optimised binary and starts it; arguments are passed on, e.g.
`./start.sh --world test --seed 7`.

The first click captures the mouse; Esc releases it. The world is saved under
`./worlds/default` every 30 seconds and on exit.

| Input | Action |
|---|---|
| W A S D | Move |
| Space | Jump |
| Shift | Run |
| Ctrl | Crouch |
| Left mouse (hold) | Break |
| Right mouse (click) | Place the selected material, or pour water |
| 1 / 2 / 3 | Select stone / dirt / sand |
| 4 | Select water |
| Mouse wheel | Brush radius |
| Space twice | Toggle fly mode (no collision; Space up, Ctrl down, Shift fast) |
| V | Toggle vsync |
| F3 | Toggle the debug overlay |

`cargo run --release -- --help` lists the options: world name, seed, the headless
modes below, and these:

| Option | Meaning | Default |
|---|---|---|
| `--view METRES` | How far terrain is drawn | 2048 |
| `--detail N` | Terrain is drawn at a finer level within N node lengths of the player | 2 |
| `--cell METRES` | Spacing of the finest terrain lattice. Fixed when a world is created | 0.25 |
| `--fragment METRES` | Breaking removes any isolated piece smaller than a cube this size; 0 keeps everything. Fixed when a world is created | 1 |

## The world

One biome, banded by height: sand below 0 m, dirt up to 40 m, bare stone above, with
the two lines wandering by a few metres. Mountain ranges reach about 180 m and carry
the cliffs and overhangs; caves run through all of it. Sand and dirt meet along a
smooth line; stone meets everything at a hard, angular edge.

The world is a function: the terrain generator plus a log of every edit as a shape
(a ball broken out, a ball placed). It is sampled every 0.25 m beside the player and
at five coarser levels, up to 8 m, out to the view distance, and an edit shows at
every level. Only the log is saved. Sand stays where it is put for now.

Water is a function too. The sea lies at 0 m wherever open ground is below it. A
pond is a volume of water in a hollow: pour some with key 4 and right click, and
it runs downhill to where it can lie. Dig a pond wider and it sinks, fill it in
and it rises, dig through its rim and it runs on, into the next hollow or the
sea; dig below 0 m beside the sea and the sea comes in. Water does not yet act
on the player or on anything else. Ponds are saved in `water.bin` in the world's
folder.

Worlds are not kept compatible between versions of the engine yet. A world records
the terrain generator it was made with and will not open after generation changes:
start another with `--world NAME`, or remove its folder under `./worlds`.

## Headless modes

```
# Render one frame to a file, optionally after playing scripted input.
tungsten --capture shot.bmp --script 'look 20,-25; tick 60; hold lmb; tick 90'

# Streaming benchmark: cold load, then a real-time fly-through.
tungsten --bench 10

# Open the window, run for a while, print frame statistics, quit.
tungsten --no-save --no-vsync --exit-after 10
```

## Layout

| Crate | Contents |
|---|---|
| `tungsten-core` | Deterministic core: generation, edit shapes, the edit log and its baking, water; plus mesher, raycast, collision geometry |
| `tungsten-stream` | Worker pool, the level-of-detail tree around the player, region files |
| `tungsten-render` | wgpu renderer: terrain, water, brush outline, debug text |
| `tungsten-game` | The `tungsten` binary: window, input, player, HUD |

## Checks

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`crates/tungsten-core/tests/golden.rs` pins hashes of generated terrain at every
level, a scripted editing session, the same session resolved from its log, the
encoded saves, and a scripted session with water. They must be identical on
every platform; if a change to generation or to the edit rules is intended, bump
`GENERATOR_VERSION` and update the constants from the failure message. A change
to how water behaves needs only its own constant updated.

## License

MIT with one added condition: a commercial product built with the engine must
show the credit "Made with Tungsten Engine by Sam Berkebile" where its other
credits are shown. See [LICENSE](LICENSE).
