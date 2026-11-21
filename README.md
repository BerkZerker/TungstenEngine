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
| Right mouse (click) | Place the selected material |
| 1 / 2 / 3 | Select stone / dirt / sand |
| Mouse wheel | Brush radius |
| Space twice | Toggle fly mode (no collision; Space up, Ctrl down, Shift fast) |
| V | Toggle vsync |
| F3 | Toggle the debug overlay |

`cargo run --release -- --help` lists the options: world name, seed, view radius,
the headless modes below, and the two settings fixed when a world is created:

| Option | Meaning | Default |
|---|---|---|
| `--cell METRES` | Spacing of the terrain lattice: smaller is smoother and heavier | 0.5 |
| `--fragment METRES` | Breaking removes any isolated piece smaller than a cube this size; 0 keeps everything | 1 |

## The world

One biome, banded by height: sand below 0 m, dirt up to 40 m, bare stone above, with
the two lines wandering by a few metres. Mountain ranges reach about 180 m and carry
the cliffs and overhangs; caves run through all of it. Materials meet at a hard edge.

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
| `tungsten-core` | Deterministic core: generation, edits, sand, save codec; plus mesher, raycast, collision geometry |
| `tungsten-stream` | Worker pool, chunk lifecycle around the player, region files |
| `tungsten-render` | wgpu renderer: terrain, brush outline, debug text |
| `tungsten-game` | The `tungsten` binary: window, input, player, HUD |

## Checks

```
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

`crates/tungsten-core/tests/golden.rs` pins hashes of generated terrain, a
scripted editing session, settled sand and the encoded saves. They must be
identical on every platform; if a change to generation is intended, bump
`GENERATOR_VERSION` and update the constants from the failure message.

## License

MIT with one added condition: a commercial product built with the engine must
show the credit "Made with Tungsten Engine by Sam Berkebile" where its other
credits are shown. See [LICENSE](LICENSE).
