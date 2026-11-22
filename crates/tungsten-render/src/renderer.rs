use glam::{IVec3, Vec3};
use tungsten_core::material::MATERIALS;
use tungsten_core::mesh::ChunkMesh;
use tungsten_core::{NodeKey, WorldParams};

use crate::gpu::Gpu;
use crate::lines::{LineBatch, LinePipeline};
use crate::terrain::{Globals, Terrain};
use crate::text::{TextBatch, TextPipeline};

pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SAMPLES: u32 = 4;
const Z_NEAR: f32 = 0.08;

/// Where the frame is seen from. Position is split into a finest-level chunk
/// and an offset inside it so rendering stays precise at any distance from
/// the origin.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub chunk: IVec3,
    /// Metres from the chunk's origin.
    pub local: Vec3,
    /// Radians; 0 looks along -Z, positive turns right.
    pub yaw: f32,
    /// Radians; positive looks up.
    pub pitch: f32,
    pub fov_y: f32,
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }
}

/// Lighting and fog, linear RGB.
#[derive(Clone, Copy, Debug)]
pub struct Sky {
    pub sun_dir: Vec3,
    pub sun_color: Vec3,
    pub sky_color: Vec3,
    pub ground_color: Vec3,
    pub fog_color: Vec3,
    pub fog_start: f32,
    pub fog_end: f32,
}

impl Sky {
    /// Daylight with fog reaching full strength at `view_distance` metres.
    pub fn daylight(view_distance: f32) -> Self {
        Self {
            sun_dir: Vec3::new(0.55, 0.6, 0.4).normalize(),
            sun_color: Vec3::new(1.0, 0.94, 0.82) * 0.95,
            sky_color: Vec3::new(0.42, 0.55, 0.75) * 0.5,
            ground_color: Vec3::new(0.30, 0.26, 0.22) * 0.35,
            fog_color: Vec3::new(0.56, 0.69, 0.86),
            fog_start: view_distance * 0.6,
            fog_end: view_distance * 0.97,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub draws: u32,
    pub triangles: u32,
    pub chunks: u32,
    pub gpu_bytes_used: u64,
    pub gpu_bytes_reserved: u64,
}

pub struct Renderer {
    format: wgpu::TextureFormat,
    size: (u32, u32),
    params: WorldParams,
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    terrain: Terrain,
    lines: LinePipeline,
    text: TextPipeline,
}

fn attachment(device: &wgpu::Device, label: &str, format: wgpu::TextureFormat, size: (u32, u32)) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Renderer {
    pub fn new(gpu: &Gpu, format: wgpu::TextureFormat, width: u32, height: u32, params: WorldParams) -> Self {
        let size = (width.max(1), height.max(1));
        Self {
            format,
            size,
            params,
            msaa: attachment(&gpu.device, "msaa color", format, size),
            depth: attachment(&gpu.device, "depth", DEPTH_FORMAT, size),
            terrain: Terrain::new(&gpu.device, format, SAMPLES),
            lines: LinePipeline::new(&gpu.device, format, SAMPLES),
            text: TextPipeline::new(&gpu.device, format, SAMPLES),
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        let size = (width.max(1), height.max(1));
        if size != self.size {
            self.size = size;
            self.msaa = attachment(&gpu.device, "msaa color", self.format, size);
            self.depth = attachment(&gpu.device, "depth", DEPTH_FORMAT, size);
        }
    }

    pub fn upload_node(&mut self, gpu: &Gpu, key: NodeKey, mesh: &ChunkMesh) {
        self.terrain.upload(&gpu.device, &gpu.queue, key, mesh);
    }

    pub fn remove_node(&mut self, key: NodeKey) {
        self.terrain.remove(key);
    }

    /// Change which nodes are drawn. A node is hidden until shown.
    pub fn set_visible(&mut self, show: &[NodeKey], hide: &[NodeKey]) {
        self.terrain.set_visible(show, hide);
    }

    /// Draw one frame into `target`, which must have this renderer's format and size.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        camera: &Camera,
        sky: &Sky,
        lines: &LineBatch,
        text: &TextBatch,
    ) -> FrameStats {
        let aspect = self.size.0 as f32 / self.size.1 as f32;
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(camera.fov_y, aspect, Z_NEAR);
        let view = glam::camera::rh::view::look_to_mat4(camera.local, camera.forward(), Vec3::Y);
        let view_proj = proj * view;

        let v4 = |v: Vec3, w: f32| [v.x, v.y, v.z, w];
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            camera: v4(camera.local, self.params.cell_size),
            sun_dir: v4(sky.sun_dir.normalize(), 0.0),
            sun_color: v4(sky.sun_color, 0.0),
            sky_color: v4(sky.sky_color, 0.0),
            ground_color: v4(sky.ground_color, 0.0),
            fog_color: v4(sky.fog_color, 0.0),
            fog: [sky.fog_start, sky.fog_end, 0.0, 0.0],
            materials: std::array::from_fn(|i| {
                let c = MATERIALS[i].color;
                [c[0], c[1], c[2], 1.0]
            }),
        };
        let (draws, triangles) = self.terrain.prepare(
            &gpu.device,
            &gpu.queue,
            &globals,
            &view_proj,
            camera.chunk,
            camera.local,
            self.params.cell_size,
        );
        let line_vertices = self.lines.prepare(&gpu.device, &gpu.queue, &view_proj, lines);
        let glyphs = self.text.prepare(&gpu.device, &gpu.queue, self.size, text);

        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.msaa,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: sky.fog_color.x as f64,
                            g: sky.fog_color.y as f64,
                            b: sky.fog_color.z as f64,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.terrain.draw(&mut pass);
            self.lines.draw(&mut pass, line_vertices);
            self.text.draw(&mut pass, glyphs);
        }
        gpu.queue.submit([encoder.finish()]);

        let (gpu_bytes_used, gpu_bytes_reserved) = self.terrain.memory();
        FrameStats { draws, triangles, chunks: self.terrain.chunk_count() as u32, gpu_bytes_used, gpu_bytes_reserved }
    }
}
