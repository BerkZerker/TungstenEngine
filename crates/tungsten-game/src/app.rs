//! The windowed application: winit event loop, surface and frame pacing.

use std::sync::Arc;
use std::time::Instant;

use tungsten_render::{FrameStats, Gpu, LineBatch, Renderer, Sky, TextBatch};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use crate::Args;
use crate::capture::apply_events;
use crate::game::Game;
use crate::hud::{self, FrameTimes, HudInfo};
use crate::input::Input;

/// Main-thread time per frame for uploading streamed-in chunk meshes.
const UPLOAD_BUDGET: std::time::Duration = std::time::Duration::from_millis(2);

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    gpu_name: String,
    renderer: Renderer,
    game: Game,
    input: Input,
    text: TextBatch,
    lines: LineBatch,
    times: FrameTimes,
    frames: u64,
    worst_frame: f64,
    last_frame: Instant,
    last_stats: FrameStats,
    grabbed: bool,
    vsync: bool,
    show_debug: bool,
    started: Instant,
    exit_after: Option<f64>,
}

struct App {
    args: Option<Args>,
    running: Option<Running>,
    error: Option<String>,
}

pub fn run(args: Args) -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| e.to_string())?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { args: Some(args), running: None, error: None };
    event_loop.run_app(&mut app).map_err(|e| e.to_string())?;
    app.error.map_or(Ok(()), Err)
}

fn present_mode(vsync: bool) -> wgpu::PresentMode {
    if vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync }
}

impl Running {
    fn new(event_loop: &ActiveEventLoop, args: Args) -> Result<Self, String> {
        let attrs = Window::default_attributes()
            .with_title("Tungsten")
            .with_inner_size(LogicalSize::new(args.size.0, args.size.1));
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| e.to_string())?);
        let instance = Gpu::instance();
        let surface = instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let gpu = pollster::block_on(Gpu::new(&instance, Some(&surface)))?;

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .ok_or("the surface is not supported by the selected GPU")?;
        let caps = surface.get_capabilities(&gpu.adapter);
        // Shading is linear; an sRGB surface format encodes on write.
        config.format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);
        config.present_mode = present_mode(!args.no_vsync);
        config.desired_maximum_frame_latency = 2;
        surface.configure(&gpu.device, &config);

        let game = Game::open(&args)?;
        let renderer = Renderer::new(&gpu, config.format, config.width, config.height, game.params());
        Ok(Self {
            gpu_name: gpu.describe(),
            window,
            surface,
            config,
            gpu,
            renderer,
            game,
            input: Input::default(),
            text: TextBatch::default(),
            lines: LineBatch::default(),
            times: FrameTimes::default(),
            frames: 0,
            worst_frame: 0.0,
            last_frame: Instant::now(),
            last_stats: FrameStats::default(),
            grabbed: false,
            vsync: !args.no_vsync,
            show_debug: true,
            started: Instant::now(),
            exit_after: args.exit_after,
        })
    }

    fn set_grab(&mut self, grab: bool) {
        if grab == self.grabbed {
            return;
        }
        if grab {
            let ok = self
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined))
                .is_ok();
            if !ok {
                return;
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.input.release_all();
        }
        self.window.set_cursor_visible(!grab);
        self.grabbed = grab;
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.renderer.resize(&self.gpu, width, height);
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64();
        self.last_frame = now;
        self.times.push(dt as f32);
        self.frames += 1;
        // The first second includes start-up work and says nothing about steady state.
        if self.started.elapsed().as_secs_f64() > 1.0 {
            self.worst_frame = self.worst_frame.max(dt);
        }

        for i in 0..self.input.pressed.len() {
            match self.input.pressed[i] {
                KeyCode::F3 => self.show_debug = !self.show_debug,
                KeyCode::KeyV => {
                    self.vsync = !self.vsync;
                    self.config.present_mode = present_mode(self.vsync);
                    self.surface.configure(&self.gpu.device, &self.config);
                }
                _ => {}
            }
        }

        self.game.advance(dt, &mut self.input);
        self.input.end_frame();
        apply_events(&mut self.game, &self.gpu, &mut self.renderer, Some(UPLOAD_BUDGET));

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            // Occluded, timed out, or a validation error already reported.
            _ => return,
        };
        hud::build(
            &mut self.text,
            &HudInfo {
                game: &self.game,
                frame: self.last_stats,
                times: &self.times,
                gpu: &self.gpu_name,
                vsync: self.vsync,
                size: (self.config.width, self.config.height),
                scale: self.window.scale_factor() as f32 * 0.5,
                show_debug: self.show_debug,
            },
        );
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sky = Sky::daylight(self.game.view_distance());
        let camera = self.game.camera();
        hud::brush_outline(&mut self.lines, &self.game, &camera);
        self.last_stats = self.renderer.render(&self.gpu, &view, &camera, &sky, &self.lines, &self.text);
        self.window.pre_present_notify();
        self.gpu.queue.present(frame);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        let Some(args) = self.args.take() else { return };
        match Running::new(event_loop, args) {
            Ok(r) => self.running = Some(r),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(r) = self.running.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => r.resize(size.width, size.height),
            WindowEvent::Focused(false) => r.set_grab(false),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = event.state == ElementState::Pressed;
                    if code == KeyCode::Escape {
                        if down {
                            r.set_grab(false);
                        }
                    } else if r.grabbed || !down {
                        r.input.key(code, down);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                if !r.grabbed {
                    if down {
                        r.set_grab(true);
                    }
                    return;
                }
                match button {
                    MouseButton::Left => r.input.break_held = down,
                    MouseButton::Right if down => r.input.place_clicks += 1,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if r.grabbed => {
                r.input.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
            }
            WindowEvent::RedrawRequested => r.frame(),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        let Some(r) = self.running.as_mut() else { return };
        if let (true, DeviceEvent::MouseMotion { delta }) = (r.grabbed, event) {
            r.input.look.0 += delta.0;
            r.input.look.1 += delta.1;
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.running {
            if r.exit_after.is_some_and(|t| r.started.elapsed().as_secs_f64() >= t) {
                event_loop.exit();
            }
            r.window.request_redraw();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = self.running.as_mut() {
            if let Err(e) = r.game.save(true) {
                self.error = Some(e);
            }
            if r.exit_after.is_some() {
                let secs = r.started.elapsed().as_secs_f64();
                println!(
                    "{} frames in {:.1} s: {:.1} fps average, worst frame {:.2} ms, {}x{}, {}",
                    r.frames,
                    secs,
                    r.frames as f64 / secs,
                    r.worst_frame * 1000.0,
                    r.config.width,
                    r.config.height,
                    r.gpu_name
                );
            }
        }
    }
}
