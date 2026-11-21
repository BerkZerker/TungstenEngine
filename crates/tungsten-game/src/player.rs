use glam::{DVec3, Vec3};
use tungsten_core::collide::{Contact, closest_point_on_triangle, sphere_triangle, triangles};
use tungsten_core::field::LocalField;
use tungsten_core::{Volume, WorldParams};
use winit::keyboard::KeyCode;

use crate::input::Input;

pub const TICK: f64 = 1.0 / tungsten_core::material::TICK_HZ as f64;
const DT: f32 = TICK as f32;

pub const RADIUS: f32 = 0.35;
pub const HEIGHT_STANDING: f32 = 1.8;
pub const HEIGHT_CROUCHED: f32 = 1.2;
pub const EYE_STANDING: f32 = 1.62;
pub const EYE_CROUCHED: f32 = 1.0;

const WALK_SPEED: f32 = 4.5;
const RUN_SPEED: f32 = 7.5;
const CROUCH_SPEED: f32 = 2.0;
const GRAVITY: f32 = 20.0;
const TERMINAL_SPEED: f32 = 50.0;
/// Launch speed for a 1.25 m apex: sqrt(2 * g * h).
const JUMP_SPEED: f32 = 7.071;
const GROUND_ACCEL: f32 = 60.0;
const AIR_ACCEL: f32 = 12.0;
/// Ledges up to this height are walked over.
const STEP_HEIGHT: f32 = 0.3;
/// Ground steeper than 50 degrees is slid down, not stood on.
const WALKABLE_Y: f32 = 0.643;
/// Ticks after leaving the ground during which a jump still works.
const COYOTE_TICKS: u8 = 6;
/// Ticks a jump press is remembered before landing.
const JUMP_BUFFER_TICKS: u8 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Walk,
    /// No gravity and no collision.
    Fly,
}

/// What the player wants to do this tick, sampled from input.
#[derive(Clone, Copy, Debug, Default)]
pub struct Intent {
    /// Local movement: x right, y up (fly only), z forward.
    pub wish: Vec3,
    pub jump: bool,
    pub run: bool,
    pub crouch: bool,
}

impl Intent {
    pub fn from_input(input: &Input) -> Self {
        let crouch = input.any_held(&[KeyCode::ControlLeft, KeyCode::ControlRight]);
        let jump = input.held(KeyCode::Space);
        Self {
            wish: Vec3::new(
                input.axis(KeyCode::KeyA, KeyCode::KeyD),
                jump as i32 as f32 - crouch as i32 as f32,
                input.axis(KeyCode::KeyS, KeyCode::KeyW),
            ),
            jump,
            run: input.any_held(&[KeyCode::ShiftLeft, KeyCode::ShiftRight]),
            crouch,
        }
    }
}

/// Terrain triangles near the player, in metres relative to an anchor point.
pub struct Collider {
    tris: Vec<[Vec3; 3]>,
    /// Triangles near the body for the current query.
    near: Vec<[Vec3; 3]>,
    field: LocalField,
    /// Position of the field's origin sample relative to the anchor, metres.
    offset: Vec3,
    cell: f32,
}

#[derive(Clone, Copy, Default)]
struct Hits {
    ground: bool,
    /// Touched something too steep to stand on.
    blocked: bool,
}

impl Collider {
    /// Collect the surface within `min..max` (metres relative to `anchor`).
    pub fn gather(volume: &Volume, params: &WorldParams, anchor: DVec3, min: Vec3, max: Vec3) -> Self {
        let cell = params.cell_size as f64;
        let lo = ((anchor + min.as_dvec3()) / cell).floor().as_ivec3() - 1;
        let hi = ((anchor + max.as_dvec3()) / cell).ceil().as_ivec3() + 2;
        let field = volume.copy_region(lo, hi - lo);
        let mut tris = Vec::new();
        triangles(&field, &mut tris);
        let offset = (lo.as_dvec3() * cell - anchor).as_vec3();
        for t in &mut tris {
            for p in t {
                *p = *p * params.cell_size + offset;
            }
        }
        Self { tris, near: Vec::new(), field, offset, cell: params.cell_size }
    }

    /// Is this point (metres relative to the anchor) inside solid material?
    fn solid_at(&self, p: Vec3) -> bool {
        let c = ((p - self.offset) / self.cell).round().as_ivec3();
        let inside = c.cmpge(glam::IVec3::ZERO).all() && c.cmplt(self.field.dims).all();
        inside && self.field.get(c.x, c.y, c.z).solid()
    }

    fn select_near(&mut self, pos: Vec3, height: f32, margin: f32) {
        let lo = pos - Vec3::new(RADIUS + margin, margin, RADIUS + margin);
        let hi = pos + Vec3::new(RADIUS + margin, height + margin, RADIUS + margin);
        self.near.clear();
        self.near.extend(self.tris.iter().copied().filter(|t| {
            let (tmin, tmax) = (t[0].min(t[1]).min(t[2]), t[0].max(t[1]).max(t[2]));
            tmin.cmple(hi).all() && tmax.cmpge(lo).all()
        }));
    }

    /// Push a body made of stacked spheres out of the terrain, clipping its velocity.
    ///
    /// One contact is resolved at a time, deepest first. Where a sphere sinks
    /// into a flat stretch of triangles, the one beneath it is the deepest and
    /// lifting off it clears the rest; taking them in any other order would act
    /// on the slanted normals their edges give and push the body sideways.
    fn depenetrate(&self, pos: &mut Vec3, height: f32, vel: &mut Vec3) -> Hits {
        let mut hits = Hits::default();
        let spheres = ((height - 2.0 * RADIUS) / 0.3).ceil() as usize + 1;
        let spacing = (height - 2.0 * RADIUS) / (spheres - 1) as f32;
        for _ in 0..12 {
            let mut deepest: Option<(usize, Contact)> = None;
            for &tri in &self.near {
                for k in 0..spheres {
                    let center = *pos + Vec3::Y * (RADIUS + spacing * k as f32);
                    let Some(c) = sphere_triangle(center, RADIUS, tri) else { continue };
                    if deepest.is_none_or(|(_, d)| c.depth > d.depth) {
                        deepest = Some((k, c));
                    }
                }
            }
            let Some((k, c)) = deepest else { break };
            if c.depth < 1e-5 {
                break;
            }
            if k == 0 && c.normal.y >= WALKABLE_Y {
                // Standing: lift straight up so slopes cause no creep.
                pos.y += c.depth / c.normal.y;
                vel.y = vel.y.max(0.0);
                hits.ground = true;
                continue;
            }
            // Ground too steep to stand on acts as a wall unless the body is
            // rising: clear it sideways, so that walking into the foot of a
            // bank does not lift the body off the floor it stands on.
            let flat = Vec3::new(c.normal.x, 0.0, c.normal.z);
            let push = if c.normal.y > 0.0 && c.normal.y < 0.9 && vel.y <= 0.0 {
                flat / flat.length_squared()
            } else {
                c.normal
            };
            *pos += push * c.depth;
            let vn = vel.dot(c.normal);
            if vn < 0.0 {
                let rising = vel.y > 0.0;
                *vel -= c.normal * vn;
                // Deflecting off a steep slope must not carry the body up it.
                if !rising && vel.y > 0.0 {
                    vel.y = 0.0;
                }
            }
            hits.blocked = true;
        }
        hits
    }

    /// Move by `vel * DT` in small steps, sliding along whatever is hit.
    fn slide(&mut self, pos: &mut Vec3, height: f32, vel: &mut Vec3) -> Hits {
        let distance = vel.length() * DT;
        self.select_near(*pos, height, distance + STEP_HEIGHT + 0.5);
        let steps = (distance / 0.12).ceil().max(1.0) as usize;
        let mut hits = Hits::default();
        for _ in 0..steps {
            *pos += *vel * (DT / steps as f32);
            let h = self.depenetrate(pos, height, vel);
            hits.ground |= h.ground;
            hits.blocked |= h.blocked;
        }
        hits
    }

    /// Would a body of this height overlap terrain at `pos`?
    fn overlaps(&mut self, pos: Vec3, height: f32) -> bool {
        self.select_near(pos, height, 0.1);
        self.overlaps_near(pos, height, 0)
    }

    /// Overlap test against the current selection, from sphere `first` upward.
    fn overlaps_near(&self, pos: Vec3, height: f32, first: usize) -> bool {
        let spheres = ((height - 2.0 * RADIUS) / 0.3).ceil() as usize + 1;
        let spacing = (height - 2.0 * RADIUS) / (spheres - 1) as f32;
        self.near.iter().any(|&tri| {
            (first..spheres).any(|k| {
                // A little slack so resting contact does not count.
                sphere_triangle(pos + Vec3::Y * (RADIUS + spacing * k as f32), RADIUS - 0.03, tri).is_some()
            })
        })
    }

    /// Lower the body by up to `drop` onto walkable ground, moving only
    /// vertically. Returns false and leaves `pos` alone if there is none.
    ///
    /// Sinking the body and pushing it out along contact normals would not do:
    /// the edges of neighbouring triangles give slanted normals that shove it
    /// sideways.
    fn settle(&mut self, pos: &mut Vec3, height: f32, drop: f32) -> bool {
        let lowest = *pos - Vec3::Y * drop;
        self.select_near(lowest, height, 0.1);
        let mut foot = lowest + Vec3::Y * RADIUS;
        let mut touched = false;
        // Raise the foot until it clears the point it is deepest into. That
        // never overshoots, and closes in on resting contact within a few rounds.
        for _ in 0..8 {
            let mut lift = 0.0f32;
            for &tri in &self.near {
                let d = foot - closest_point_on_triangle(foot, tri);
                if d.length_squared() < RADIUS * RADIUS {
                    lift = lift.max((RADIUS * RADIUS - d.x * d.x - d.z * d.z).sqrt() - d.y);
                }
            }
            if lift <= 1e-5 {
                break;
            }
            foot.y += lift;
            touched = true;
        }
        let settled = foot - Vec3::Y * RADIUS;
        if !touched || settled.y > pos.y + 0.02 {
            return false;
        }
        // What the foot now rests on decides whether this is ground.
        let support = self
            .near
            .iter()
            .map(|&tri| foot - closest_point_on_triangle(foot, tri))
            .min_by(|a, b| a.length_squared().total_cmp(&b.length_squared()));
        let walkable = support.is_some_and(|d| d.y >= WALKABLE_Y * d.length());
        if !walkable || self.overlaps_near(settled, height, 1) {
            return false;
        }
        *pos = settled;
        true
    }
}

pub struct Player {
    /// Feet position in metres.
    pub pos: DVec3,
    /// Position at the previous tick, for render interpolation.
    pub prev_pos: DVec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: Mode,
    pub grounded: bool,
    pub crouching: bool,
    pub eye_height: f32,
    pub prev_eye_height: f32,
    jump_was_held: bool,
    jump_buffer: u8,
    coyote: u8,
}

fn move_toward(current: Vec3, target: Vec3, max_delta: f32) -> Vec3 {
    let d = target - current;
    let len = d.length();
    if len <= max_delta || len < 1e-6 { target } else { current + d * (max_delta / len) }
}

impl Player {
    pub fn new(pos: DVec3) -> Self {
        Self {
            pos,
            prev_pos: pos,
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            mode: Mode::Walk,
            grounded: false,
            crouching: false,
            eye_height: EYE_STANDING,
            prev_eye_height: EYE_STANDING,
            jump_was_held: false,
            jump_buffer: 0,
            coyote: 0,
        }
    }

    pub fn look(&mut self, dx: f64, dy: f64) {
        const SENSITIVITY: f32 = 0.0022;
        self.yaw = (self.yaw + dx as f32 * SENSITIVITY).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch - dy as f32 * SENSITIVITY).clamp(-1.55, 1.55);
    }

    /// Unit vectors for forward and right on the horizontal plane.
    pub fn basis(&self) -> (Vec3, Vec3) {
        let (s, c) = self.yaw.sin_cos();
        (Vec3::new(s, 0.0, -c), Vec3::new(c, 0.0, s))
    }

    pub fn view_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    pub fn height(&self) -> f32 {
        if self.crouching { HEIGHT_CROUCHED } else { HEIGHT_STANDING }
    }

    /// Eye position interpolated between the last two ticks.
    pub fn eye(&self, alpha: f64) -> DVec3 {
        let feet = self.prev_pos.lerp(self.pos, alpha);
        let h = self.prev_eye_height + (self.eye_height - self.prev_eye_height) * alpha as f32;
        feet + DVec3::new(0.0, h as f64, 0.0)
    }

    pub fn teleport(&mut self, pos: DVec3) {
        self.pos = pos;
        self.prev_pos = pos;
        self.vel = Vec3::ZERO;
    }

    /// Keep interpolation state current on ticks where the player does not move.
    pub fn tick_idle(&mut self) {
        self.prev_pos = self.pos;
        self.prev_eye_height = self.eye_height;
    }

    /// Switch between walking and flying.
    pub fn toggle_fly(&mut self) {
        self.mode = if self.mode == Mode::Fly { Mode::Walk } else { Mode::Fly };
        // The key press that switched modes must not also count as a jump.
        self.jump_was_held = true;
        self.jump_buffer = 0;
        self.coyote = 0;
    }

    pub fn tick_fly(&mut self, intent: &Intent) {
        self.tick_idle();
        let (forward, right) = self.basis();
        let speed = if intent.run { 60.0 } else { 12.0 };
        let wish = (forward * intent.wish.z + right * intent.wish.x + Vec3::Y * intent.wish.y).normalize_or_zero();
        // Ease towards the wished velocity so starts and stops are not abrupt.
        self.vel += (wish * speed - self.vel) * 0.25;
        self.pos += (self.vel * DT).as_dvec3();
        self.grounded = false;
        self.crouching = false;
        self.eye_height = EYE_STANDING;
    }

    pub fn tick_walk(&mut self, intent: &Intent, volume: &Volume, params: &WorldParams) {
        self.tick_idle();
        let reach = self.vel.abs() * DT + Vec3::new(1.2, 1.5, 1.2);
        let mut terrain =
            Collider::gather(volume, params, self.pos, -reach, reach + Vec3::Y * (HEIGHT_STANDING + STEP_HEIGHT));
        // Positions below are relative to where the tick started.
        let mut pos = Vec3::ZERO;

        // Buried (material placed or slumped onto the body): there is no
        // surface to stand on inside solid ground, so rise until clear.
        if terrain.solid_at(Vec3::Y * (self.height() * 0.5)) {
            self.pos.y += 0.2;
            self.vel = Vec3::ZERO;
            self.grounded = false;
            return;
        }

        // Crouch immediately; stand only when there is headroom.
        if intent.crouch {
            self.crouching = true;
        } else if self.crouching && !terrain.overlaps(pos, HEIGHT_STANDING) {
            self.crouching = false;
        }
        let height = self.height();
        let eye_target = if self.crouching { EYE_CROUCHED } else { EYE_STANDING };
        self.eye_height += (eye_target - self.eye_height).clamp(-6.0 * DT, 6.0 * DT);

        // Horizontal velocity follows the wish direction.
        let (forward, right) = self.basis();
        let wish = (forward * intent.wish.z + right * intent.wish.x).normalize_or_zero();
        let speed = if self.crouching {
            CROUCH_SPEED
        } else if intent.run {
            RUN_SPEED
        } else {
            WALK_SPEED
        };
        let horizontal = Vec3::new(self.vel.x, 0.0, self.vel.z);
        let horizontal = if self.grounded {
            move_toward(horizontal, wish * speed, GROUND_ACCEL * DT)
        } else if wish != Vec3::ZERO {
            move_toward(horizontal, wish * speed.max(horizontal.length()), AIR_ACCEL * DT)
        } else {
            horizontal
        };
        self.vel = Vec3::new(horizontal.x, self.vel.y, horizontal.z);

        // Jump on a press, remembered briefly before landing and after leaving a ledge.
        if intent.jump && !self.jump_was_held {
            self.jump_buffer = JUMP_BUFFER_TICKS;
        }
        self.jump_was_held = intent.jump;
        let mut jumped = false;
        if self.jump_buffer > 0 && self.coyote > 0 {
            self.vel.y = JUMP_SPEED;
            self.jump_buffer = 0;
            self.coyote = 0;
            self.grounded = false;
            jumped = true;
        }
        self.jump_buffer = self.jump_buffer.saturating_sub(1);
        self.vel.y = (self.vel.y - GRAVITY * DT).max(-TERMINAL_SPEED);

        let was_grounded = self.grounded && !jumped;
        let (start_vel, mut vel) = (self.vel, self.vel);
        let mut hits = terrain.slide(&mut pos, height, &mut vel);

        // Blocked by a low ledge: retry lifted by a step, then settle down onto it.
        if hits.blocked && (was_grounded || hits.ground) && wish != Vec3::ZERO {
            let mut stepped = Vec3::new(0.0, STEP_HEIGHT, 0.0);
            let mut step_vel = Vec3::new(start_vel.x, 0.0, start_vel.z);
            if !terrain.overlaps(stepped, height) {
                terrain.slide(&mut stepped, height, &mut step_vel);
                let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z).length();
                if terrain.settle(&mut stepped, height, STEP_HEIGHT + 0.05) && flat(stepped) > flat(pos) + 0.005 {
                    pos = stepped;
                    vel = Vec3::new(step_vel.x, 0.0, step_vel.z);
                    hits = Hits { ground: true, blocked: false };
                }
            }
        }

        // Walking down a slope or off a small step: stay attached to the ground.
        if was_grounded && !hits.ground && vel.y <= 0.0 && terrain.settle(&mut pos, height, STEP_HEIGHT - 0.05) {
            vel.y = 0.0;
            hits.ground = true;
        }

        self.grounded = hits.ground;
        self.coyote = if self.grounded { COYOTE_TICKS } else { self.coyote.saturating_sub(1) };
        self.vel = vel;
        self.pos += pos.as_dvec3();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;
    use tungsten_core::{Chunk, Sample};

    /// Flat dirt with its surface at y = -0.25 m (cells below y=0 are full).
    fn flat_world() -> (Volume, WorldParams) {
        let mut vol = Volume::new();
        for y in -2..2 {
            for z in -2..2 {
                for x in -2..2 {
                    let s = if y < 0 { Sample { fill: 255, mat: 2 } } else { Sample::AIR };
                    vol.insert(IVec3::new(x, y, z), Chunk::Uniform(s));
                }
            }
        }
        (vol, WorldParams::default())
    }

    /// Ground rising `grade` metres per metre towards +Z, level across X.
    fn slope_world(grade: f32) -> (Volume, WorldParams) {
        let params = WorldParams::default();
        let mut vol = Volume::new();
        for cy in -2..2 {
            for cz in -2..2 {
                for cx in -2..2 {
                    let pos = IVec3::new(cx, cy, cz);
                    let mut d = tungsten_core::Dense::new_boxed(Sample::AIR);
                    for i in 0..tungsten_core::CHUNK_VOLUME {
                        let (y, z) = ((i >> 10) as i32 + cy * 32, ((i >> 5) & 31) as i32 + cz * 32);
                        let depth = (grade * z as f32 - y as f32) * 64.0;
                        d.fill[i] = (127.5 + depth).clamp(0.0, 255.0) as u8;
                        d.mat[i] = if d.fill[i] > 0 { 2 } else { 0 };
                    }
                    vol.insert(pos, Chunk::Dense(d));
                }
            }
        }
        (vol, params)
    }

    #[test]
    fn keeps_speed_and_heading_on_slopes() {
        // Down, up and across, at grades from gentle to near the walkable limit.
        for grade in [0.05f32, 0.2, 0.5, 1.0] {
            let (vol, params) = slope_world(grade);
            for (yaw, name) in [(0.0f32, "down"), (180.0, "up"), (90.0, "across")] {
                for (run_held, speed) in [(false, WALK_SPEED), (true, RUN_SPEED)] {
                    let mut p = Player::new(DVec3::new(0.3, 4.0, 0.3));
                    p.yaw = yaw.to_radians();
                    run(&mut p, &vol, &params, Intent::default(), 120);
                    assert!(p.grounded, "grade {grade}: did not land");
                    let rest = p.pos;
                    run(&mut p, &vol, &params, Intent::default(), 60);
                    assert!(
                        p.pos.distance(rest) < 1e-3,
                        "grade {grade}: slid {} m while standing",
                        p.pos.distance(rest)
                    );

                    let intent = Intent { wish: Vec3::Z, run: run_held, ..Default::default() };
                    run(&mut p, &vol, &params, intent, 20);
                    let from = p.pos;
                    for _ in 0..60 {
                        p.tick_walk(&intent, &vol, &params);
                        assert!(p.grounded, "grade {grade} {name}: left the ground");
                    }
                    let d = p.pos - from;
                    let (along, aside) = if name == "across" { (d.x, d.z) } else { (d.z.abs(), d.x) };
                    let what = format!("grade {grade} {name} at {speed} m/s: went {along:.3} m, {aside:.3} m aside");
                    assert!((along - speed as f64).abs() < 0.03 * speed as f64, "{what}");
                    assert!(aside.abs() < 0.02, "{what}");
                    // And it stayed on the surface.
                    let expected = d.z * grade as f64;
                    assert!((d.y - expected).abs() < 0.05, "{what}; height changed {:.3}, slope {expected:.3}", d.y);
                }
            }
        }
    }

    #[test]
    fn slides_down_ground_too_steep_to_stand_on() {
        // 63 degrees: well past the 50 degree limit.
        let (vol, params) = slope_world(2.0);
        let mut p = Player::new(DVec3::new(0.3, 3.0, 0.3));
        run(&mut p, &vol, &params, Intent::default(), 40);
        let from = p.pos;
        run(&mut p, &vol, &params, Intent::default(), 60);
        let d = p.pos - from;
        assert!(d.z < -1.0 && d.y < -2.0, "moved {d:?} in a second");
        assert!((d.y - d.z * 2.0).abs() < 0.3, "left the slope: {d:?}");
        // Walking at it does not climb it.
        let mut p = Player::new(DVec3::new(0.3, 3.0, 0.3));
        p.yaw = 180f32.to_radians();
        let top = p.pos.y;
        for _ in 0..120 {
            p.tick_walk(&Intent { wish: Vec3::Z, run: true, ..Default::default() }, &vol, &params);
            assert!(p.pos.y <= top + 1e-3, "climbed to {}", p.pos.y);
        }
    }

    fn run(player: &mut Player, vol: &Volume, params: &WorldParams, intent: Intent, ticks: usize) {
        for _ in 0..ticks {
            player.tick_walk(&intent, vol, params);
        }
    }

    #[test]
    fn lands_stands_still_and_walks_at_walk_speed() {
        let (vol, params) = flat_world();
        let mut p = Player::new(DVec3::new(0.3, 2.0, 0.3));
        run(&mut p, &vol, &params, Intent::default(), 90);
        assert!(p.grounded);
        assert!((p.pos.y + 0.25).abs() < 0.02, "resting at y={}", p.pos.y);
        let rest = p.pos;
        run(&mut p, &vol, &params, Intent::default(), 60);
        assert!(p.pos.distance(rest) < 1e-4, "drifted while standing");

        let forward = Intent { wish: Vec3::Z, ..Default::default() };
        run(&mut p, &vol, &params, forward, 30);
        let a = p.pos;
        run(&mut p, &vol, &params, forward, 60);
        let moved = (p.pos - a).length();
        assert!((moved - WALK_SPEED as f64).abs() < 0.05, "walked {moved} m in a second");
        assert!((p.pos.y - rest.y).abs() < 0.01 && p.grounded);
        // Facing -Z at yaw 0.
        assert!(p.pos.z < a.z - 4.0);
    }

    #[test]
    fn jump_reaches_its_apex_and_crouch_lowers_the_eye() {
        let (vol, params) = flat_world();
        let mut p = Player::new(DVec3::new(0.3, 0.0, 0.3));
        run(&mut p, &vol, &params, Intent::default(), 60);
        let ground = p.pos.y;
        let mut apex = ground;
        p.tick_walk(&Intent { jump: true, ..Default::default() }, &vol, &params);
        for _ in 0..80 {
            p.tick_walk(&Intent { jump: true, ..Default::default() }, &vol, &params);
            apex = apex.max(p.pos.y);
        }
        let h = apex - ground;
        assert!((1.1..=1.3).contains(&h), "jump height {h}");
        // Holding jump does not bounce: back on the ground.
        assert!(p.grounded && (p.pos.y - ground).abs() < 0.02);

        run(&mut p, &vol, &params, Intent { crouch: true, ..Default::default() }, 30);
        assert!(p.crouching && (p.eye_height - EYE_CROUCHED).abs() < 1e-3);
        run(&mut p, &vol, &params, Intent::default(), 30);
        assert!(!p.crouching && (p.eye_height - EYE_STANDING).abs() < 1e-3);
    }

    #[test]
    fn rises_out_when_buried() {
        let (mut vol, params) = flat_world();
        let mut p = Player::new(DVec3::new(0.3, 0.0, 0.3));
        run(&mut p, &vol, &params, Intent::default(), 60);
        // Bury the player under 3 m of sand.
        for y in 0..6 {
            for z in -4..4 {
                for x in -4..4 {
                    vol.set(IVec3::new(x, y, z), Sample { fill: 255, mat: 3 });
                }
            }
        }
        run(&mut p, &vol, &params, Intent::default(), 120);
        // Sand surface is at 5.5 cells = 2.75 m.
        assert!(p.grounded && (p.pos.y - 2.75).abs() < 0.05, "y = {}", p.pos.y);
    }

    #[test]
    fn walks_over_a_low_step_but_not_a_wall() {
        let (mut vol, params) = flat_world();
        // A 0.25 m (half-cell) rise for z <= -4 cells, and a 2 m wall at z <= -16 cells.
        for z in -30..=-4 {
            for x in -8..8 {
                let top = if z <= -16 { 4 } else { 0 };
                for y in 0..top {
                    vol.set(IVec3::new(x, y, z), Sample { fill: 255, mat: 1 });
                }
                if top == 0 {
                    vol.set(IVec3::new(x, 0, z), Sample { fill: 127, mat: 1 });
                    vol.set(IVec3::new(x, -1, z), Sample { fill: 255, mat: 1 });
                }
            }
        }
        let mut p = Player::new(DVec3::new(0.3, 0.5, 0.3));
        run(&mut p, &vol, &params, Intent::default(), 60);
        let low = p.pos.y;
        run(&mut p, &vol, &params, Intent { wish: Vec3::Z, ..Default::default() }, 240);
        assert!(p.pos.y > low + 0.15, "did not climb the step: y {} -> {}", low, p.pos.y);
        // Stopped by the wall, standing in front of it.
        assert!(p.pos.z > -8.0 - 0.1 && p.pos.z < -6.5, "z = {}", p.pos.z);
        assert!(p.grounded);
    }
}
