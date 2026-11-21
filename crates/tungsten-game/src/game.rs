use std::path::PathBuf;
use std::sync::Arc;

use glam::{DVec3, IVec3, Vec3};
use tungsten_core::edit::{EditKind, EditOp, Exclude, FP};
use tungsten_core::raycast::{RayHit, raycast};
use tungsten_core::{Material, WorldParams};
use tungsten_render::Camera;
use tungsten_stream::store::{read_kv, write_kv};
use tungsten_stream::{MeshEvent, Store, World};
use winit::keyboard::KeyCode;

use crate::Args;
use crate::input::{DoubleTap, Input};
use crate::player::{Intent, Mode, Player, RADIUS, TICK};

pub const BRUSH_MIN: f32 = 0.5;
pub const BRUSH_MAX: f32 = 4.0;
pub const BRUSH_STEP: f32 = 0.25;
/// How far the player can reach to break or place, metres.
pub const REACH: f32 = 6.0;
/// Ticks between autosaves: 30 seconds.
const AUTOSAVE_TICKS: u64 = 30 * tungsten_core::material::TICK_HZ as u64;

/// Simulation state: the world and the player in it.
pub struct Game {
    pub world: World,
    pub player: Player,
    pub selected: Material,
    pub brush_radius: f32,
    pub tick: u64,
    /// What the player is looking at, as of the last tick.
    pub aim: Option<RayHit>,
    /// False until the player has been put on the ground at the spawn point.
    spawned: bool,
    accumulator: f64,
    space_taps: DoubleTap,
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
            brush_radius: 1.5,
            tick: 0,
            aim: None,
            spawned: false,
            accumulator: 0.0,
            space_taps: DoubleTap::default(),
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
                ("material", (self.selected as u8).to_string()),
                ("mode", if p.mode == Mode::Fly { "fly" } else { "walk" }.to_string()),
            ],
        )
        .map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn params(&self) -> WorldParams {
        self.world.params()
    }

    /// Chunk containing a position in metres.
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
                KeyCode::Digit1 => self.selected = Material::Stone,
                KeyCode::Digit2 => self.selected = Material::Dirt,
                KeyCode::Digit3 => self.selected = Material::Sand,
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
            // Never simulate against terrain that is not there yet.
            self.player.tick_idle();
            self.aim = None;
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
        if self.tick.is_multiple_of(tungsten_core::sand::SAND_INTERVAL) {
            self.world.step_sand(&mut self.events);
        }
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

    fn edit(&mut self, input: &mut Input) {
        let cell = self.params().cell_size as f64;
        let eye = self.player.eye(1.0) / cell;
        let dir = self.player.view_dir();
        self.aim = raycast(self.world.volume(), eye, dir, REACH / cell as f32);
        let clicks = std::mem::take(&mut input.place_clicks);
        let Some(hit) = self.aim else { return };
        let radius = (self.brush_radius as f64 / cell * FP as f64).round() as i64;

        if input.break_held {
            let op = EditOp {
                kind: EditKind::Break,
                center: hit.pos.to_array().map(to_fp),
                radius,
                material: 0,
                tick: self.tick,
                exclude: None,
            };
            self.world.edit(&op, &mut self.events);
        }
        if clicks > 0 {
            // Centre the stamp just outside the surface so it attaches to it.
            let center = hit.pos + (hit.normal * (self.brush_radius * 0.6 / cell as f32)).as_dvec3();
            let feet = self.player.pos / cell;
            // The surface forms about half a cell outside the last full sample.
            let margin = 1.0;
            let op = EditOp {
                kind: EditKind::Place,
                center: center.to_array().map(to_fp),
                radius,
                material: self.selected as u8,
                tick: self.tick,
                exclude: (self.player.mode == Mode::Walk).then(|| Exclude {
                    x: to_fp(feet.x),
                    z: to_fp(feet.z),
                    y0: to_fp(feet.y + RADIUS as f64 / cell),
                    y1: to_fp(feet.y + (self.player.height() - RADIUS) as f64 / cell),
                    radius: to_fp(RADIUS as f64 / cell + margin),
                }),
            };
            self.world.edit(&op, &mut self.events);
        }
    }

    /// Advance chunk streaming around the player.
    pub fn stream(&mut self) {
        let focus = self.chunk_at(self.player.pos);
        self.world.update(focus, &mut self.events);
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

    /// Distance in metres at which terrain stops being drawn.
    pub fn view_distance(&self) -> f32 {
        self.world.config().radius_h as f32 * self.params().chunk_metres()
    }
}
