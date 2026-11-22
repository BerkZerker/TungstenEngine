//! Headless rendering of a single frame to a BMP file.

use std::io::Write;
use std::time::{Duration, Instant};

use tungsten_render::{FrameStats, Gpu, LineBatch, Renderer, Sky, TextBatch};
use tungsten_stream::MeshEvent;

use crate::Args;
use crate::game::Game;
use crate::hud::{self, FrameTimes, HudInfo};
use crate::input::Input;
use crate::player::Mode;
use winit::keyboard::KeyCode;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Hand pending mesh changes to the renderer, oldest first.
///
/// With a `budget`, streaming uploads stop once it is spent and wait for the
/// next frame; results of edits always go through, so an edit is never late
/// and its chunks never appear one at a time.
pub fn apply_events(game: &mut Game, gpu: &Gpu, renderer: &mut Renderer, budget: Option<Duration>) {
    let urgent =
        |e: &MeshEvent| matches!(e, MeshEvent::Upload { urgent: true, .. } | MeshEvent::Remove { urgent: true, .. });
    let must = game.events.iter().rposition(urgent).map_or(0, |i| i + 1);
    let start = Instant::now();
    let mut done = 0;
    for event in &game.events {
        if done >= must && budget.is_some_and(|b| start.elapsed() >= b) {
            break;
        }
        match event {
            MeshEvent::Upload { key, mesh, .. } => renderer.upload_node(gpu, *key, mesh),
            MeshEvent::Remove { key, .. } => renderer.remove_node(*key),
            MeshEvent::Visible { show, hide } => renderer.set_visible(show, hide),
        }
        done += 1;
    }
    game.events.drain(..done);
}

pub fn run(args: Args) -> Result<(), String> {
    let path = args.capture.clone().unwrap();
    let (width, height) = args.size;
    let instance = Gpu::instance();
    let gpu = pollster::block_on(Gpu::new(&instance, None))?;
    let mut game = Game::open(&args)?;
    let mut renderer = Renderer::new(&gpu, FORMAT, width, height, game.params());
    // Without a script nothing would put the player on the ground: look from where they are.
    if args.script.is_none() && args.world.is_none() {
        game.player.mode = Mode::Fly;
    }

    let start = Instant::now();
    settle(&mut game, &gpu, &mut renderer)?;
    let load_time = start.elapsed();
    if let Some(script) = &args.script {
        run_script(script, &mut game, &gpu, &mut renderer)?;
        settle(&mut game, &gpu, &mut renderer)?;
    }
    game.save(true)?;

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("capture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sky = Sky::daylight(game.view_distance());
    let mut text = TextBatch::default();
    let mut lines = LineBatch::default();
    hud::brush_outline(&mut lines, &game, &game.camera());
    let mut times = FrameTimes::default();
    let gpu_name = gpu.describe();

    // Render twice: the first frame supplies statistics for the overlay.
    let mut stats = FrameStats::default();
    let mut frame_time = 0.0;
    for _ in 0..2 {
        times.push(0.25f32.max(frame_time));
        hud::build(
            &mut text,
            &HudInfo {
                game: &game,
                frame: stats,
                times: &times,
                gpu: &gpu_name,
                vsync: false,
                size: (width, height),
                scale: 1.0,
                show_debug: true,
            },
        );
        let t = Instant::now();
        // Ripples are placed by simulation time, so a capture is repeatable.
        let time = game.tick as f32 * crate::player::TICK as f32;
        stats = renderer.render(&gpu, &view, &game.camera(), &sky, &lines, &text, time);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
        frame_time = t.elapsed().as_secs_f32();
    }

    let row_bytes = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture readback"),
        size: row_bytes as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("capture") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    buffer.map_async(wgpu::MapMode::Read, .., |r| r.expect("map capture buffer"));
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| e.to_string())?;
    let data = buffer.get_mapped_range(..).map_err(|e| e.to_string())?;

    write_bmp(&path, width, height, row_bytes as usize, &data).map_err(|e| format!("{}: {e}", path.display()))?;
    let s = game.world.stats();
    println!(
        "{}: {}x{}  loaded {} nodes ({} meshed, {} near chunks) in {:.2} s  {} draws  {} tris  frame {:.2} ms  {}",
        path.display(),
        width,
        height,
        s.nodes,
        s.meshed,
        s.chunks,
        load_time.as_secs_f32(),
        stats.draws,
        stats.triangles,
        frame_time * 1000.0,
        gpu_name
    );
    Ok(())
}

/// Stream until everything in range is loaded and meshed.
fn settle(game: &mut Game, gpu: &Gpu, renderer: &mut Renderer) -> Result<(), String> {
    let start = Instant::now();
    loop {
        game.stream();
        apply_events(game, gpu, renderer, None);
        if game.world.settled() {
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(120) {
            return Err("world did not finish loading within 120 s".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn key_code(name: &str) -> Result<KeyCode, String> {
    Ok(match name {
        "w" => KeyCode::KeyW,
        "a" => KeyCode::KeyA,
        "s" => KeyCode::KeyS,
        "d" => KeyCode::KeyD,
        "space" => KeyCode::Space,
        "shift" => KeyCode::ShiftLeft,
        "ctrl" => KeyCode::ControlLeft,
        "1" => KeyCode::Digit1,
        "2" => KeyCode::Digit2,
        "3" => KeyCode::Digit3,
        _ => return Err(format!("script: unknown key '{name}'")),
    })
}

/// Play semicolon-separated input commands against the simulation, one
/// fixed tick at a time, so a capture can show the result of playing.
fn run_script(script: &str, game: &mut Game, gpu: &Gpu, renderer: &mut Renderer) -> Result<(), String> {
    let mut input = Input::default();
    for command in script.split(';').map(str::trim).filter(|c| !c.is_empty()) {
        let (name, arg) = command.split_once(' ').unwrap_or((command, ""));
        let arg = arg.trim();
        match name {
            "tick" => {
                let n: u32 = arg.parse().map_err(|_| format!("script: tick needs a count, got '{arg}'"))?;
                for _ in 0..n {
                    game.step(&mut input);
                    game.stream();
                    apply_events(game, gpu, renderer, None);
                }
            }
            "hold" | "release" => {
                let down = name == "hold";
                if arg == "lmb" {
                    input.break_held = down;
                } else {
                    input.key(key_code(arg)?, down);
                    // Let the game see the press itself, as it would between frames.
                    game.advance(0.0, &mut input);
                    input.end_frame();
                }
            }
            "click" => input.place_clicks += 1,
            "key" => {
                let code = key_code(arg)?;
                input.key(code, true);
                game.advance(0.0, &mut input);
                input.end_frame();
                input.key(code, false);
            }
            "scroll" => {
                input.scroll = arg.parse().map_err(|_| format!("script: scroll needs a number, got '{arg}'"))?;
                game.advance(0.0, &mut input);
                input.end_frame();
            }
            "look" => {
                let v = crate::parse_list::<f32>(arg, ',', 2)?;
                game.player.yaw = v[0].to_radians();
                game.player.pitch = v[1].to_radians();
            }
            other => return Err(format!("script: unknown command '{other}'")),
        }
    }
    Ok(())
}

/// Write RGBA rows as a 24-bit BMP.
fn write_bmp(path: &std::path::Path, width: u32, height: u32, stride: usize, rgba: &[u8]) -> std::io::Result<()> {
    let row = (width as usize * 3).next_multiple_of(4);
    let size = 54 + row * height as usize;
    let mut out = Vec::with_capacity(size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(size as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&(width as i32).to_le_bytes());
    out.extend_from_slice(&(height as i32).to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in (0..height as usize).rev() {
        let line = &rgba[y * stride..y * stride + width as usize * 4];
        for px in line.chunks_exact(4) {
            out.extend_from_slice(&[px[2], px[1], px[0]]);
        }
        out.resize(out.len() + row - width as usize * 3, 0);
    }
    std::fs::File::create(path)?.write_all(&out)
}
