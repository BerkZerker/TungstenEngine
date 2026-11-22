use std::collections::VecDeque;
use std::fmt::Write;

use tungsten_render::{Camera, FrameStats, LineBatch, TextBatch};

use crate::game::Game;
use crate::player::Mode;

/// Rolling frame-time statistics.
#[derive(Default)]
pub struct FrameTimes {
    recent: VecDeque<f32>,
    scratch: Vec<f32>,
    /// (average ms, 1% worst ms), refreshed a few times a second.
    shown: (f32, f32),
    since_refresh: f32,
}

impl FrameTimes {
    pub fn push(&mut self, dt: f32) {
        self.recent.push_back(dt);
        while self.recent.len() > 600 {
            self.recent.pop_front();
        }
        self.since_refresh += dt;
        if self.since_refresh >= 0.25 {
            self.since_refresh = 0.0;
            self.scratch.clear();
            self.scratch.extend(self.recent.iter().copied());
            self.scratch.sort_unstable_by(|a, b| b.total_cmp(a));
            let worst = (self.scratch.len() / 100).max(1);
            let low = self.scratch[..worst].iter().sum::<f32>() / worst as f32;
            let avg = self.scratch.iter().sum::<f32>() / self.scratch.len() as f32;
            self.shown = (avg * 1000.0, low * 1000.0);
        }
    }

    pub fn average_ms(&self) -> f32 {
        self.shown.0
    }

    pub fn low_ms(&self) -> f32 {
        self.shown.1
    }
}

pub struct HudInfo<'a> {
    pub game: &'a Game,
    pub frame: FrameStats,
    pub times: &'a FrameTimes,
    pub gpu: &'a str,
    pub vsync: bool,
    pub size: (u32, u32),
    pub scale: f32,
    pub show_debug: bool,
}

const WHITE: [u8; 4] = [255, 255, 255, 255];
const DIM: [u8; 4] = [200, 210, 220, 255];

/// Outline of the brush at the point the player is aiming at.
pub fn brush_outline(lines: &mut LineBatch, game: &Game, camera: &Camera) {
    lines.clear();
    let Some(hit) = &game.aim else { return };
    let params = game.params();
    let world = hit.pos * params.cell_size as f64;
    let center = (world - camera.chunk.as_dvec3() * params.chunk_metres() as f64).as_vec3();
    // Lifted slightly so it does not shimmer against the surface it lies on.
    lines.ring(center + hit.normal * 0.03, hit.normal, game.brush_radius, [255, 255, 255, 110]);
}

pub fn build(text: &mut TextBatch, info: &HudInfo) {
    text.clear();
    let s = (info.scale * 3.0).round().max(1.0);

    // Crosshair.
    let c = (info.scale * 2.0).round().max(1.0);
    let (cx, cy) = (info.size.0 as f32 * 0.5, info.size.1 as f32 * 0.5);
    text.push(cx - 2.5 * c, cy - 3.5 * c, c, [255, 255, 255, 220], "+");

    if !info.show_debug {
        return;
    }
    let g = info.game;
    let p = g.player.pos;
    let chunk = g.chunk_at(p);
    let stats = g.world.stats();
    let params = g.params();
    let (avg, low) = (info.times.average_ms(), info.times.low_ms());
    let mut out = String::with_capacity(512);
    let _ = writeln!(
        out,
        "{:.0} fps  {:.2} ms  (1% low {:.0} fps, {:.2} ms){}",
        1000.0 / avg.max(0.001),
        avg,
        1000.0 / low.max(0.001),
        low,
        if info.vsync { "  vsync" } else { "" }
    );
    let _ = writeln!(out, "pos   {:.2} {:.2} {:.2}", p.x, p.y, p.z);
    let _ = writeln!(out, "chunk {} {} {}", chunk.x, chunk.y, chunk.z);
    let _ = writeln!(
        out,
        "look  yaw {:.0} pitch {:.0}   speed {:.1} m/s",
        g.player.yaw.to_degrees(),
        g.player.pitch.to_degrees(),
        g.player.vel.length()
    );
    let _ = writeln!(
        out,
        "mode  {}{}{}",
        match g.player.mode {
            Mode::Walk => "walk",
            Mode::Fly => "fly",
        },
        if g.player.grounded { "  grounded" } else { "" },
        if g.player.crouching { "  crouching" } else { "" }
    );
    let brush = if g.pouring { "water" } else { g.selected.def().name };
    let _ = writeln!(out, "brush {}  radius {:.2} m", brush, g.brush_radius);
    match &g.aim {
        Some(hit) => {
            let _ = writeln!(
                out,
                "aim   {}  fill {}  {:.1} m",
                tungsten_core::Material::from_u8(hit.material).def().name,
                hit.fill,
                hit.distance * params.cell_size
            );
        }
        None => {
            let _ = writeln!(out, "aim   -");
        }
    }
    if g.waiting_for_terrain() {
        let _ = writeln!(out, "waiting for terrain");
    }
    let _ = writeln!(out);
    let d = stats.drawn;
    let _ = writeln!(
        out,
        "nodes   {} ({} meshed)  drawn by level {} {} {} {} {} {}",
        stats.nodes, stats.meshed, d[0], d[1], d[2], d[3], d[4], d[5]
    );
    let _ = writeln!(
        out,
        "near    {} chunks ({} dense)  jobs {} queued {} running",
        stats.chunks, stats.dense, stats.jobs_queued, stats.jobs_running
    );
    let _ = writeln!(out, "edits   {} ops in {} cells  {} baked", stats.ops, stats.cells, stats.bakes);
    let w = g.world.water().stats();
    // Water volumes are in 1/256 of a cell cubed.
    let m3 = |v: i64| v as f64 / 256.0 * (params.cell_size as f64).powi(3);
    let _ = writeln!(
        out,
        "water   {} ponds {:.1} m3  moving {:.1}  to sea {:.1}  lost {:.1}",
        w.ponds,
        m3(w.held),
        m3(w.moving),
        m3(w.drained),
        m3(w.lost)
    );
    let _ = writeln!(out, "sea     in {} cells  {} pending", w.sea_cells, w.sea_pending);
    let _ = writeln!(
        out,
        "draw    {} calls  {:.0}k tris  gpu {:.0}/{:.0} MB",
        info.frame.draws,
        info.frame.triangles as f32 / 1000.0,
        info.frame.gpu_bytes_used as f32 / 1048576.0,
        info.frame.gpu_bytes_reserved as f32 / 1048576.0
    );
    let _ = writeln!(
        out,
        "world   seed {}  cell {} m  fragment {} m  view {} m  tick {}",
        params.seed,
        params.cell_size,
        params.min_fragment,
        g.view_distance(),
        g.tick
    );
    let _ = write!(out, "{}", info.gpu);
    text.push(6.0 * s, 6.0 * s, s, WHITE, &out);

    // The key list, on as many lines as the window's width needs.
    let help = "WASD move  Space jump  Shift run  Ctrl crouch  LMB break  RMB place  1-3 material  4 water  wheel brush  Space x2 fly  V vsync  F3 debug";
    let columns = ((info.size.0 as f32 - 12.0 * s) / TextBatch::advance(s)) as usize;
    let mut lines: Vec<String> = Vec::new();
    for item in help.split("  ") {
        match lines.last_mut() {
            Some(line) if line.len() + 2 + item.len() <= columns => {
                line.push_str("  ");
                line.push_str(item);
            }
            _ => lines.push(item.to_string()),
        }
    }
    let y = info.size.1 as f32 - TextBatch::line_height(s) * lines.len() as f32 - 4.0 * s;
    text.push(6.0 * s, y, s, DIM, &lines.join("\n"));
}
