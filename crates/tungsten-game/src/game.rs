use std::path::PathBuf;
use std::sync::Arc;

use glam::{DVec3, IVec3, Vec3};
use tungsten_core::edit::{EditRules, Exclude, FP, Op};
use tungsten_core::material::{self, CarveMode};
use tungsten_core::raycast::{RayHit, raycast};
use tungsten_core::{Material, WorldParams};
use tungsten_render::{Camera, PourArc};
use tungsten_stream::store::{read_kv, write_kv};
use tungsten_stream::{MeshEvent, Store, World};
use winit::keyboard::KeyCode;

use crate::Args;
use crate::input::{DoubleTap, Input};
use crate::player::{Intent, Mode, Player, RADIUS, TICK};

pub const BRUSH_MIN: f32 = 0.25;
pub const BRUSH_MAX: f32 = 4.0;
pub const BRUSH_STEP: f32 = 0.25;
/// Ticks between cuts while mining something that gives way smoothly.
const CUT_TICKS: u64 = 6;
/// Metres of mining it takes to break stone, which then goes all at once.
const STONE_WORK: f32 = 0.75;
/// Chips knocked off, evenly spaced, before the shard goes.
const STONE_CHIPS: u32 = 3;
/// How far the player can reach to break or place, metres.
pub const REACH: f32 = 6.0;
/// The number key that selects water, and what the player file calls it.
const WATER_SLOT: u8 = 4;
/// Ticks between autosaves: 30 seconds.
const AUTOSAVE_TICKS: u64 = 30 * tungsten_core::material::TICK_HZ as u64;

/// Simulation state: the world and the player in it.
pub struct Game {
    pub world: World,
    pub player: Player,
    pub selected: Material,
    /// Right click pours water instead of placing the selected material.
    pub pouring: bool,
    /// True while water is actively streaming from the pour tool.
    pouring_active: bool,
    pub brush_radius: f32,
    pub tick: u64,
    /// What the player is looking at, as of the last tick.
    pub aim: Option<RayHit>,
    /// False until the player has been put on the ground at the spawn point.
    spawned: bool,
    accumulator: f64,
    space_taps: DoubleTap,
    /// Ticks the break button has been held, and the mining done on stone since it last broke.
    mining_ticks: u64,
    stone_work: f32,
    stone_chips: u32,
    /// Where the player's state is saved, if the world is persistent.
    player_file: Option<PathBuf>,
    /// Mesh changes not yet handed to the renderer.
    pub events: Vec<MeshEvent>,
}

fn to_fp(cells: f64) -> i64 {
    (cells * FP as f64).round() as i64
}

impl Game {
    /// Open the world named by `args` (or a throwaway one) and put the player in it.
    pub fn open(args: &Args) -> Result<Self, String> {
        let (store, params) = match &args.world {
            Some(dir) => {
                let (store, params) = Store::open(dir, args.params)?;
                if args.explicit_params && params != args.params {
                    return Err(format!(
                        "world '{}' already exists with seed {}, cell size {} m and fragment size {} m; pick another --world name for different settings",
                        dir.display(),
                        params.seed,
                        params.cell_size,
                        params.min_fragment
                    ));
                }
                (Some(Arc::new(store)), params)
            }
            None => (None, args.params),
        };
        let world = World::new(params, args.stream, store);
        let (x, z) = (8.25, 8.25);
        let ground = world.generator().surface_height(x, z) as f64;
        let mut game = Self {
            player: Player::new(DVec3::new(x, ground + 2.0, z)),
            world,
            selected: Material::Stone,
            pouring: false,
            pouring_active: false,
            brush_radius: 1.5,
            tick: 0,
            aim: None,
            spawned: false,
            accumulator: 0.0,
            space_taps: DoubleTap::default(),
            mining_ticks: 0,
            stone_work: 0.0,
            stone_chips: 0,
            player_file: args.world.as_ref().map(|d| d.join("player.txt")),
            events: Vec::new(),
        };
        game.load_player();
        if args.fly {
            game.player.mode = Mode::Fly;
        }
        if let Some(pos) = args.pos {
            game.place_player(pos);
        }
        if let Some((yaw, pitch)) = args.look {
            game.player.yaw = yaw.to_radians();
            game.player.pitch = pitch.to_radians();
        }
        Ok(game)
    }

    fn load_player(&mut self) {
        let Some(kv) = self.player_file.as_deref().and_then(read_kv) else { return };
        let num = |k: &str| kv.get(k).and_then(|v| v.parse::<f64>().ok());
        if let (Some(x), Some(y), Some(z)) = (num("x"), num("y"), num("z")) {
            self.place_player(DVec3::new(x, y, z));
        }
        if let (Some(yaw), Some(pitch)) = (num("yaw"), num("pitch")) {
            self.player.yaw = yaw as f32;
            self.player.pitch = (pitch as f32).clamp(-1.55, 1.55);
        }
        if let Some(r) = num("brush_radius") {
            self.brush_radius = (r as f32).clamp(BRUSH_MIN, BRUSH_MAX);
        }
        if let Some(m) = num("material") {
            self.pouring = m as u8 == WATER_SLOT;
            self.selected = match Material::from_u8(m as u8) {
                Material::Air => Material::Stone,
                m => m,
            };
        }
        if kv.get("mode").map(String::as_str) == Some("fly") {
            self.player.mode = Mode::Fly;
        }
    }

    /// Save edits and the player. `wait` blocks until the data is on disk.
    pub fn save(&mut self, wait: bool) -> Result<(), String> {
        let Some(path) = &self.player_file else { return Ok(()) };
        self.world.save(wait).map_err(|e| format!("saving the world failed: {e}"))?;
        // Before the first spawn the position is a placeholder, not somewhere to return to.
        if !self.spawned {
            return Ok(());
        }
        let p = &self.player;
        write_kv(
            path,
            &[
                ("x", p.pos.x.to_string()),
                ("y", p.pos.y.to_string()),
                ("z", p.pos.z.to_string()),
                ("yaw", p.yaw.to_string()),
                ("pitch", p.pitch.to_string()),
                ("brush_radius", self.brush_radius.to_string()),
                ("material", if self.pouring { WATER_SLOT } else { self.selected as u8 }.to_string()),
                ("mode", if p.mode == Mode::Fly { "fly" } else { "walk" }.to_string()),
            ],
        )
        .map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn params(&self) -> WorldParams {
        self.world.params()
    }

    /// Finest-level chunk containing a position in metres.
    pub fn chunk_at(&self, pos: DVec3) -> IVec3 {
        (pos / self.params().chunk_metres() as f64).floor().as_ivec3()
    }

    /// Put the player at an explicit position instead of the spawn point.
    pub fn place_player(&mut self, pos: DVec3) {
        self.player.teleport(pos);
        self.spawned = true;
    }

    /// True while the terrain around the player is still loading.
    pub fn waiting_for_terrain(&self) -> bool {
        !self.world.neighbourhood_loaded(self.chunk_at(self.player.pos))
    }

    /// Advance by `dt` seconds of real time. Returns the number of ticks run.
    pub fn advance(&mut self, dt: f64, input: &mut Input) -> u32 {
        self.player.look(input.look.0, input.look.1);
        for &key in &input.pressed {
            match key {
                KeyCode::Digit1 => (self.selected, self.pouring) = (Material::Stone, false),
                KeyCode::Digit2 => (self.selected, self.pouring) = (Material::Dirt, false),
                KeyCode::Digit3 => (self.selected, self.pouring) = (Material::Sand, false),
                KeyCode::Digit4 => self.pouring = true,
                KeyCode::Space if self.space_taps.press(self.tick) => self.player.toggle_fly(),
                _ => {}
            }
        }
        if input.scroll != 0.0 {
            let steps = input.scroll.round().clamp(-8.0, 8.0);
            self.brush_radius = (self.brush_radius + steps * BRUSH_STEP).clamp(BRUSH_MIN, BRUSH_MAX);
        }

        self.accumulator += dt.min(0.25);
        let mut ticks = 0;
        while self.accumulator >= TICK {
            self.accumulator -= TICK;
            self.step(input);
            ticks += 1;
        }
        self.stream();
        ticks
    }

    /// One fixed simulation tick.
    pub fn step(&mut self, input: &mut Input) {
        self.tick += 1;
        let intent = Intent::from_input(input);
        if self.waiting_for_terrain() {
            // Never walk on or edit terrain that is not there yet. Flying
            // touches nothing, so it carries on.
            match self.player.mode {
                Mode::Fly if self.spawned => self.player.tick_fly(&intent),
                _ => self.player.tick_idle(),
            }
            self.aim = None;
            self.pouring_active = false;
            input.place_clicks = 0;
            return;
        }
        if !self.spawned {
            self.spawn();
        }
        match self.player.mode {
            Mode::Fly => self.player.tick_fly(&intent),
            Mode::Walk => {
                let params = self.params();
                self.player.tick_walk(&intent, self.world.volume(), &params);
            }
        }
        self.edit(input);
        self.world.step_water();
        if self.tick.is_multiple_of(AUTOSAVE_TICKS) {
            if let Err(e) = self.save(false) {
                eprintln!("tungsten: {e}");
            }
        }
    }

    /// Drop the player onto the actual surface at the spawn column.
    fn spawn(&mut self) {
        self.spawned = true;
        let cell = self.params().cell_size as f64;
        let from = self.player.pos + DVec3::Y * 30.0;
        if let Some(hit) = raycast(self.world.volume(), from / cell, Vec3::NEG_Y, (80.0 / cell) as f32) {
            self.player.teleport(hit.pos * cell + DVec3::Y * 0.05);
        }
    }

    /// Knock a chip off the stone beside the point aimed at, on a different
    /// side each time: the sign that a shard is giving way.
    fn chip(&mut self, hit: &RayHit, eye: DVec3, n: u32) {
        let cell = self.params().cell_size;
        let size = EditRules::new(&self.params()).chips.spacing as f32 / FP as f32;
        let side = hit.normal.any_orthonormal_vector();
        let up = hit.normal.cross(side);
        // Three places round the aim point, then the point itself.
        for attempt in 0..4 {
            let turn = (n + attempt) as f32 * std::f32::consts::TAU / 3.0 + 0.5;
            let offset = if attempt < 3 { (side * turn.cos() + up * turn.sin()) * (0.7 * size) } else { Vec3::ZERO };
            // Onto the surface there, and just under it.
            let dir = (hit.pos + offset.as_dvec3() - eye).as_vec3().normalize_or_zero();
            let Some(at) = raycast(self.world.volume(), eye, dir, REACH / cell + size) else { continue };
            if material::def(at.material).carve != CarveMode::Shards {
                continue;
            }
            let center = at.pos - (at.normal * (0.3 * size)).as_dvec3();
            if !self.world.edit(&Op::Chip { center: center.to_array().map(to_fp) }).chunks.is_empty() {
                return;
            }
        }
    }

    fn edit(&mut self, input: &mut Input) {
        let cell = self.params().cell_size as f64;
        let eye = self.player.eye(1.0) / cell;
        let dir = self.player.view_dir();
        self.aim = raycast(self.world.volume(), eye, dir, REACH / cell as f32);
        self.pouring_active = false;
        let clicks = std::mem::take(&mut input.place_clicks);
        if !input.break_held {
            (self.mining_ticks, self.stone_work, self.stone_chips) = (0, 0.0, 0);
        }
        let Some(hit) = self.aim else { return };
        let radius = (self.brush_radius as f64 / cell * FP as f64).round() as i64;

        if input.break_held {
            let def = material::def(hit.material);
            // Where the ball goes: sitting on the surface aimed at, sunk
            // `depth` metres into it. Measured along the surface normal, so a
            // glancing aim bites no deeper than a square one.
            let ball = |depth: f32| {
                let center = hit.pos + (hit.normal * ((self.brush_radius - depth) / cell as f32)).as_dvec3();
                center.to_array().map(to_fp)
            };
            let op = match def.carve {
                // A cut every few ticks, each a little deeper than the last.
                CarveMode::Smooth => self.mining_ticks.is_multiple_of(CUT_TICKS).then(|| {
                    let depth = (def.dig_speed * CUT_TICKS as f32 * TICK as f32).min(self.brush_radius);
                    Op::Break { center: ball(depth), radius, nearest: false }
                }),
                // Stone chips as the work goes in, then the shard aimed at
                // breaks off with every other the ball has reached.
                CarveMode::Shards => {
                    self.stone_work += def.dig_speed * TICK as f32;
                    if self.stone_work >= STONE_WORK {
                        (self.stone_work, self.stone_chips) = (0.0, 0);
                        // Centred just inside the surface, so the shard it is
                        // in is the one being looked at.
                        let center = hit.pos - (hit.normal * (self.brush_radius.min(0.4) / cell as f32)).as_dvec3();
                        Some(Op::Break { center: center.to_array().map(to_fp), radius, nearest: true })
                    } else {
                        let due = (self.stone_work / STONE_WORK * (STONE_CHIPS + 1) as f32) as u32;
                        if due > self.stone_chips {
                            self.chip(&hit, eye, self.stone_chips);
                            self.stone_chips = due;
                        }
                        None
                    }
                }
            };
            self.mining_ticks += 1;
            if let Some(op) = op {
                self.world.edit(&op);
            }
        }
        if self.pouring && (input.place_held || clicks > 0) {
            // Holding right click pours a steady flow at a rate set by the brush size.
            // Let go in the first open cell over the surface: from any higher
            // it would fall there as a column of its own.
            let at = hit.pos + (hit.normal * 0.6).as_dvec3();
            let r = self.brush_radius as f64 / cell;
            let rate = (r * r * 30.0).round().max(1.0) as i64;
            let mult = if input.place_held { 1 } else { clicks as i64 };
            self.world.pour(at.floor().as_ivec3(), rate * mult);
            self.pouring_active = input.place_held;
        } else if clicks > 0 {
            // Centre the stamp just outside the surface so it attaches to it.
            let center = hit.pos + (hit.normal * (self.brush_radius * 0.6 / cell as f32)).as_dvec3();
            let feet = self.player.pos / cell;
            // The surface forms about half a cell outside the last full sample.
            let margin = 1.0;
            let op = Op::Place {
                center: center.to_array().map(to_fp),
                radius,
                material: self.selected as u8,
                exclude: (self.player.mode == Mode::Walk).then(|| Exclude {
                    x: to_fp(feet.x),
                    z: to_fp(feet.z),
                    y0: to_fp(feet.y + RADIUS as f64 / cell),
                    y1: to_fp(feet.y + (self.player.height() - RADIUS) as f64 / cell),
                    radius: to_fp(RADIUS as f64 / cell + margin),
                }),
            };
            self.world.edit(&op);
        }
    }

    /// Advance streaming around the player.
    pub fn stream(&mut self) {
        self.world.update(self.player.pos, &mut self.events);
    }

    /// Fraction of a tick elapsed since the last one, for interpolation.
    pub fn alpha(&self) -> f64 {
        self.accumulator / TICK
    }

    pub fn camera(&self) -> Camera {
        let eye = self.player.eye(self.alpha());
        let chunk = self.chunk_at(eye);
        let local = eye - chunk.as_dvec3() * self.params().chunk_metres() as f64;
        Camera {
            chunk,
            local: local.as_vec3(),
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            fov_y: 70f32.to_radians(),
        }
    }

    /// Active cosmetic pour stream from the player's tool to the aimed surface.
    pub fn pour_arc(&self) -> Option<PourArc> {
        if !self.pouring_active {
            return None;
        }
        let hit = self.aim?;
        let cell = self.params().cell_size as f64;
        let eye = self.player.eye(self.alpha());
        let forward = self.player.view_dir().as_dvec3();
        let (sy, cy) = self.player.yaw.sin_cos();
        let right = DVec3::new(cy as f64, 0.0, sy as f64);
        let up = right.cross(forward).normalize_or(DVec3::Y);
        let from = eye + forward * 0.4 + right * 0.3 - up * 0.35;
        let to = (hit.pos + (hit.normal * 0.5).as_dvec3()) * cell;
        Some(PourArc { from, to, radius: self.brush_radius })
    }

    /// Distance in metres at which terrain stops being drawn.
    pub fn view_distance(&self) -> f32 {
        self.world.config().view_distance
    }
}
