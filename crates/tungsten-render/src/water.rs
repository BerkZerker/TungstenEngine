//! Water surfaces and moving water: flat patches drawn over the terrain at
//! smoothly eased client-side levels, plus dynamic flowing streams and the
//! active pour arc.

use bytemuck::{Pod, Zeroable};
use glam::{DVec3, IVec3, Mat4, Vec3};
use tungsten_core::edit::FP;
use tungsten_core::hash::{IntMap, IntSet};
use tungsten_core::water::bodies::Stream;
use tungsten_core::water::{BodyId, WaterPatch};
use tungsten_core::{CHUNK_SIZE, NodeKey};

use crate::arena::{Alloc, Arena};
use crate::terrain::{aabb_visible, frustum_planes};

/// Exponential ease rate (per second) for cosmetic client-side level transitions.
const LEVEL_EASE_RATE: f64 = 5.5;
/// Cross-section sides of extruded flowing streams and pour arcs.
const STREAM_SIDES: usize = 12;

/// Active pour stream from the player's tool to the aimed surface, purely cosmetic.
#[derive(Clone, Copy, Debug)]
pub struct PourArc {
    /// Start position in world metres (near the player's tool/camera).
    pub from: DVec3,
    /// Target position in world metres (just off the aimed surface).
    pub to: DVec3,
    /// Brush radius in metres, setting stream thickness.
    pub radius: f32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct StreamVertex {
    pos: [f32; 3],
    along: f32,
    normal: [f32; 3],
    foam: f32,
    /// Across the flow: metres from its middle, that as a share of its half
    /// width, and how far it is a sheet on the ground and not a jet, 0 to 1.
    across: [f32; 3],
}

struct PatchGpu {
    body: BodyId,
    /// Level when the patch was made; used unless the body has a newer one.
    level: i64,
    vertices: Alloc,
    /// In units of two indices, so byte offsets stay 4-aligned.
    indices: Alloc,
    index_count: u32,
}

struct Visible {
    distance: f32,
    vertex_page: u32,
    index_page: u32,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
    origin: [f32; 4],
}

pub(crate) struct Water {
    pipeline: wgpu::RenderPipeline,
    stream_pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    vertex_arena: Arena,
    index_arena: Arena,
    nodes: IntMap<NodeKey, Vec<PatchGpu>>,
    /// True simulation surface levels, fixed-point base cells.
    targets: IntMap<BodyId, i64>,
    /// Client-side eased surface levels, fixed-point base cells.
    drawn: IntMap<BodyId, f64>,
    visible: Vec<Visible>,
    origins: Vec<[f32; 4]>,
    streams: Vec<Stream>,
    pour: Option<PourArc>,
    stream_vertices: Vec<StreamVertex>,
    stream_indices: Vec<u32>,
    stream_vbuf: wgpu::Buffer,
    stream_vcap: usize,
    stream_ibuf: wgpu::Buffer,
    stream_icap: usize,
    stream_index_count: u32,
}

impl Water {
    /// `globals` is the terrain's uniform buffer, which the water shader
    /// shares; `distance` is the attachment the terrain writes its distance to.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        globals: &wgpu::Buffer,
        distance: &wgpu::TextureView,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water"),
            source: wgpu::ShaderSource::Wgsl(include_str!("water.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = Self::bind(device, &layout, globals, distance);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: crate::renderer::DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let color_target = [Some(wgpu::ColorTargetState {
            format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Uint16x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![1 => Float32x4],
                    }),
                ],
            },
            // Seen from both sides: the camera can be under the surface.
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            // Hidden by terrain in front of it, but never hides anything itself.
            depth_stencil: depth_stencil.clone(),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            multiview_mask: None,
            cache: None,
        });
        let stream_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water streams"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_stream"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<StreamVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32, 4 => Float32x3],
                })],
            },
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil,
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_stream"),
                compilation_options: Default::default(),
                targets: &color_target,
            }),
            multiview_mask: None,
            cache: None,
        });
        let instance_capacity = 1024;
        let stream_vcap = 1024;
        let stream_icap = 4096;
        Self {
            pipeline,
            stream_pipeline,
            layout,
            bind_group,
            instances: Self::instance_buffer(device, instance_capacity),
            instance_capacity,
            vertex_arena: Arena::new("water vertices", wgpu::BufferUsages::VERTEX, 4, 1 << 16),
            index_arena: Arena::new("water indices", wgpu::BufferUsages::INDEX, 4, 1 << 16),
            nodes: IntMap::default(),
            targets: IntMap::default(),
            drawn: IntMap::default(),
            visible: Vec::new(),
            origins: Vec::new(),
            streams: Vec::new(),
            pour: None,
            stream_vertices: Vec::new(),
            stream_indices: Vec::new(),
            stream_vbuf: Self::stream_vertex_buffer(device, stream_vcap),
            stream_vcap,
            stream_ibuf: Self::stream_index_buffer(device, stream_icap),
            stream_icap,
            stream_index_count: 0,
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        distance: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(distance) },
            ],
        })
    }

    /// Point at a new distance attachment, after a resize.
    pub fn rebind(&mut self, device: &wgpu::Device, globals: &wgpu::Buffer, distance: &wgpu::TextureView) {
        self.bind_group = Self::bind(device, &self.layout, globals, distance);
    }

    fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water origins"),
            size: capacity as u64 * 16,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn stream_vertex_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water stream vertices"),
            size: (capacity * size_of::<StreamVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn stream_index_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water stream indices"),
            size: (capacity * size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: NodeKey, patches: &[WaterPatch]) {
        self.remove(key);
        let gpu: Vec<PatchGpu> = patches
            .iter()
            .filter(|p| !p.indices.is_empty())
            .map(|p| {
                debug_assert!(p.indices.len().is_multiple_of(2));
                let vertices = self.vertex_arena.alloc(device, p.vertices.len() as u32);
                let indices = self.index_arena.alloc(device, (p.indices.len() / 2) as u32);
                self.vertex_arena.write(queue, vertices, bytemuck::cast_slice(&p.vertices));
                self.index_arena.write(queue, indices, bytemuck::cast_slice(&p.indices));
                PatchGpu { body: p.body, level: p.level, vertices, indices, index_count: p.indices.len() as u32 }
            })
            .collect();
        if !gpu.is_empty() {
            self.nodes.insert(key, gpu);
        }
    }

    pub fn remove(&mut self, key: NodeKey) {
        for p in self.nodes.remove(&key).unwrap_or_default() {
            self.vertex_arena.free(p.vertices);
            self.index_arena.free(p.indices);
        }
    }

    /// Update the true levels of bodies whose surface can move, easing the
    /// client-side drawn levels toward them over `dt` seconds.
    pub fn set_levels(&mut self, levels: impl IntoIterator<Item = (BodyId, i64)>, dt: f32) {
        self.targets.clear();
        self.targets.extend(levels);
        self.drawn.retain(|body, _| self.targets.contains_key(body));
        let alpha = if dt > 0.0 { 1.0 - (-(dt as f64) * LEVEL_EASE_RATE).exp() } else { 0.0 };
        for (&body, &target) in &self.targets {
            let target = target as f64;
            match self.drawn.get_mut(&body) {
                Some(cur) if dt > 0.0 => {
                    *cur += (target - *cur) * alpha;
                    if (target - *cur).abs() < 0.25 {
                        *cur = target;
                    }
                }
                Some(_) => {}
                None => {
                    self.drawn.insert(body, target);
                }
            }
        }
    }

    /// Set the active moving-water streams and optional cosmetic pour arc.
    pub fn set_streams(&mut self, streams: &[Stream], pour: Option<PourArc>) {
        self.streams.clear();
        self.streams.extend_from_slice(streams);
        self.pour = pour;
    }

    /// Cull, sort back to front, build stream geometry, and upload. Returns (draws, triangles).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        camera_chunk: IVec3,
        camera_local: Vec3,
        cell_size: f32,
        shown: &IntSet<NodeKey>,
    ) -> (u32, u32) {
        let planes = frustum_planes(view_proj);
        self.visible.clear();
        let mut triangles = 0;
        for (key, patches) in &self.nodes {
            if !shown.contains(key) {
                continue;
            }
            let step = key.step();
            let base = key.origin() - camera_chunk * CHUNK_SIZE;
            let size = (CHUNK_SIZE * step) as f32 * cell_size;
            for p in patches {
                let level = self.drawn.get(&p.body).copied().unwrap_or(p.level as f64);
                // In base cells above the camera chunk's origin.
                let y = (level / FP as f64 - (camera_chunk.y * CHUNK_SIZE) as f64) as f32;
                let min = Vec3::new(base.x as f32, y, base.z as f32) * cell_size;
                let max = min + Vec3::new(size, 0.0, size);
                if !aabb_visible(&planes, min, max) {
                    continue;
                }
                triangles += p.index_count / 3;
                self.visible.push(Visible {
                    distance: ((min + max) * 0.5).distance_squared(camera_local),
                    vertex_page: p.vertices.page,
                    index_page: p.indices.page,
                    first_index: p.indices.offset * 2,
                    index_count: p.index_count,
                    base_vertex: p.vertices.offset as i32,
                    origin: [base.x as f32, y, base.z as f32, step as f32],
                });
            }
        }
        self.visible.sort_unstable_by(|a, b| b.distance.total_cmp(&a.distance));
        if self.visible.len() > self.instance_capacity {
            self.instance_capacity = self.visible.len().next_power_of_two();
            self.instances = Self::instance_buffer(device, self.instance_capacity);
        }
        self.origins.clear();
        self.origins.extend(self.visible.iter().map(|v| v.origin));
        if !self.origins.is_empty() {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&self.origins));
        }

        // Build flowing stream and pour-arc triangles.
        self.stream_vertices.clear();
        self.stream_indices.clear();
        let origin_cells = camera_chunk * CHUNK_SIZE;
        let origin_metres = origin_cells.as_dvec3() * cell_size as f64;
        for stream in &self.streams {
            append_stream(
                stream,
                origin_cells,
                cell_size,
                &planes,
                &mut self.stream_vertices,
                &mut self.stream_indices,
            );
        }
        if let Some(pour) = self.pour {
            append_pour_arc(pour, origin_metres, &planes, &mut self.stream_vertices, &mut self.stream_indices);
        }
        self.stream_index_count = self.stream_indices.len() as u32;
        if self.stream_index_count > 0 {
            if self.stream_vertices.len() > self.stream_vcap {
                self.stream_vcap = self.stream_vertices.len().next_power_of_two();
                self.stream_vbuf = Self::stream_vertex_buffer(device, self.stream_vcap);
            }
            if self.stream_indices.len() > self.stream_icap {
                self.stream_icap = self.stream_indices.len().next_power_of_two();
                self.stream_ibuf = Self::stream_index_buffer(device, self.stream_icap);
            }
            queue.write_buffer(&self.stream_vbuf, 0, bytemuck::cast_slice(&self.stream_vertices));
            queue.write_buffer(&self.stream_ibuf, 0, bytemuck::cast_slice(&self.stream_indices));
        }
        let stream_draws = u32::from(self.stream_index_count > 0);
        triangles += self.stream_index_count / 3;
        (self.visible.len() as u32 + stream_draws, triangles)
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.visible.is_empty() {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(1, self.instances.slice(..));
            let (mut vp, mut ip) = (u32::MAX, u32::MAX);
            for (i, v) in self.visible.iter().enumerate() {
                if v.vertex_page != vp {
                    vp = v.vertex_page;
                    pass.set_vertex_buffer(0, self.vertex_arena.buffer(vp).slice(..));
                }
                if v.index_page != ip {
                    ip = v.index_page;
                    pass.set_index_buffer(self.index_arena.buffer(ip).slice(..), wgpu::IndexFormat::Uint16);
                }
                let i = i as u32;
                pass.draw_indexed(v.first_index..v.first_index + v.index_count, v.base_vertex, i..i + 1);
            }
        }
        if self.stream_index_count > 0 {
            pass.set_pipeline(&self.stream_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.stream_vbuf.slice(..));
            pass.set_index_buffer(self.stream_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.stream_index_count, 0, 0..1);
        }
    }
}

/// Sample a polyline along `[from, to]` in distance units, returning `(pos, along_metres)`.
fn clip_stream(stream: &Stream, origin_cells: IVec3, cell_size: f32) -> Vec<(Vec3, f32)> {
    let n = stream.path.len().min(stream.along.len());
    if n < 2 || stream.to <= stream.from {
        return Vec::new();
    }
    let pt = |i: usize| (stream.path[i] - origin_cells).as_vec3() * cell_size;
    let mut out: Vec<(Vec3, f32)> = Vec::with_capacity(n + 2);
    for i in 0..n - 1 {
        let (a0, a1) = (stream.along[i], stream.along[i + 1]);
        if a1 <= stream.from || a0 >= stream.to || a1 <= a0 {
            continue;
        }
        let (p0, p1) = (pt(i), pt(i + 1));
        let t0 = ((stream.from - a0) / (a1 - a0)).clamp(0.0, 1.0);
        let t1 = ((stream.to - a0) / (a1 - a0)).clamp(0.0, 1.0);
        let q0 = p0.lerp(p1, t0);
        let s0 = (a0 + (a1 - a0) * t0) * cell_size;
        if out.last().is_none_or(|(last, _)| last.distance_squared(q0) > 1e-6) {
            out.push((q0, s0));
        }
        let q1 = p0.lerp(p1, t1);
        let s1 = (a0 + (a1 - a0) * t1) * cell_size;
        if out.last().is_none_or(|(last, _)| last.distance_squared(q1) > 1e-6) {
            out.push((q1, s1));
        }
    }
    out
}

/// Resample a lattice polyline at uniform spacing and smooth out single-cell
/// stair steps so water on a slope becomes a continuous chute with slowly
/// varying tangent, preventing tube folds at tight corners.
fn smooth_polyline(pts: &[(Vec3, f32)], spacing: f32) -> Vec<(Vec3, f32)> {
    if pts.len() < 2 {
        return pts.to_vec();
    }
    // 1. Resample uniformly along the polyline's arc length.
    let mut cum = Vec::with_capacity(pts.len());
    cum.push(0.0f32);
    for i in 1..pts.len() {
        cum.push(cum[i - 1] + pts[i].0.distance(pts[i - 1].0));
    }
    let total = *cum.last().unwrap();
    if total < 1e-4 {
        return pts.to_vec();
    }
    let steps = ((total / spacing).ceil() as usize).clamp(2, 256);
    let mut resampled = Vec::with_capacity(steps + 1);
    let mut seg = 0usize;
    for k in 0..=steps {
        let target = total * (k as f32 / steps as f32);
        while seg + 1 < cum.len() - 1 && cum[seg + 1] < target {
            seg += 1;
        }
        let span = (cum[seg + 1] - cum[seg]).max(1e-6);
        let t = ((target - cum[seg]) / span).clamp(0.0, 1.0);
        let p = pts[seg].0.lerp(pts[seg + 1].0, t);
        let s = pts[seg].1 + (pts[seg + 1].1 - pts[seg].1) * t;
        resampled.push((p, s));
    }
    // 2. Laplacian smoothing on interior points removes single-cell stair steps
    // while keeping start and end anchored.
    let mut next = resampled.clone();
    for _ in 0..4 {
        for i in 1..resampled.len() - 1 {
            next[i].0 = resampled[i - 1].0 * 0.25 + resampled[i].0 * 0.5 + resampled[i + 1].0 * 0.25;
        }
        std::mem::swap(&mut resampled, &mut next);
    }
    resampled
}

fn append_stream(
    stream: &Stream,
    origin_cells: IVec3,
    cell_size: f32,
    planes: &[glam::Vec4; 5],
    vertices: &mut Vec<StreamVertex>,
    indices: &mut Vec<u32>,
) {
    let clipped = clip_stream(stream, origin_cells, cell_size);
    if clipped.len() < 2 {
        return;
    }
    let smoothed = smooth_polyline(&clipped, (cell_size * 1.6).max(0.35));
    // Half the width of the water, from cubic metres a second of it.
    let rate = stream.rate * cell_size.powi(3) * 60.0;
    let base_r = (0.2 * rate.max(0.0).sqrt()).clamp(0.1, 0.8);
    let pad = Vec3::splat(base_r * 1.8);
    let (mut min, mut max) = (smoothed[0].0, smoothed[0].0);
    for &(p, _) in &smoothed {
        min = min.min(p);
        max = max.max(p);
    }
    if !aabb_visible(planes, min - pad, max + pad) {
        return;
    }
    let s_start = smoothed.first().unwrap().1;
    let s_end = smoothed.last().unwrap().1;
    let total_len = (s_end - s_start).max(1e-3);
    let taper_dist = (1.5 * base_r).max(0.3).min(total_len * 0.5);
    // The water narrows to its front and its tail, not where it comes from
    // or where the way ends: there it runs on into other water.
    let length = stream.along.last().copied().unwrap_or(0.0);
    let (source, arrived) = (stream.from <= 0.01, stream.to >= length - 0.01);
    extrude_tube(
        &smoothed,
        |i, tangent| {
            let s = smoothed[i].1;
            let tail = if source { 1.0 } else { (s - s_start) / taper_dist };
            let front = if arrived { 1.0 } else { (s_end - s) / taper_dist };
            let end_taper = tail.min(front).clamp(0.04, 1.0).sqrt();
            let steep = tangent.y.abs();
            // On slopes, a wide shallow sheet lying on the ground; on steep drops, a rounded column.
            let rx = base_r * (1.0 - 0.6 * steep) * end_taper;
            let ry = (0.05 + (base_r * 0.4 - 0.05) * steep) * end_taper;
            let foam = (0.15 + 0.85 * steep).clamp(0.15, 1.0);
            (rx, ry, ry * 0.5, foam, 1.0 - steep)
        },
        vertices,
        indices,
    );
}

fn append_pour_arc(
    pour: PourArc,
    origin_metres: DVec3,
    planes: &[glam::Vec4; 5],
    vertices: &mut Vec<StreamVertex>,
    indices: &mut Vec<u32>,
) {
    let from = (pour.from - origin_metres).as_vec3();
    let to = (pour.to - origin_metres).as_vec3();
    let span = to - from;
    let dist = span.length();
    if dist < 0.05 {
        return;
    }
    let horiz = Vec3::new(span.x, 0.0, span.z).length();
    // Gentle parabolic arch peaking near the midpoint.
    let lift = (horiz * 0.14).min(0.45);
    const SEGMENTS: usize = 16;
    let mut pts = Vec::with_capacity(SEGMENTS + 1);
    let mut along = 0.0f32;
    let mut prev = from;
    for i in 0..=SEGMENTS {
        let t = i as f32 / SEGMENTS as f32;
        let p = from.lerp(to, t) + Vec3::Y * (4.0 * t * (1.0 - t) * lift);
        if i > 0 {
            along += p.distance(prev);
        }
        pts.push((p, along));
        prev = p;
    }
    let base_r = (0.025 + 0.022 * pour.radius).clamp(0.03, 0.11);
    let pad = Vec3::splat(base_r * 1.5);
    let (mut min, mut max) = (pts[0].0, pts[0].0);
    for &(p, _) in &pts {
        min = min.min(p);
        max = max.max(p);
    }
    if !aabb_visible(planes, min - pad, max + pad) {
        return;
    }
    extrude_tube(
        &pts,
        |i, _| {
            let t = i as f32 / SEGMENTS as f32;
            let r = base_r * (0.55 + 0.70 * t);
            let foam = 0.45 + 0.35 * t;
            (r, r, 0.0, foam, 0.0)
        },
        vertices,
        indices,
    );
}

/// Extrude a closed `STREAM_SIDES`-sided tube along `pts` using a stable
/// transport frame. `profile(i, tangent)` returns `(rx, ry, y_shift, foam, flat)`.
fn extrude_tube(
    pts: &[(Vec3, f32)],
    profile: impl Fn(usize, Vec3) -> (f32, f32, f32, f32, f32),
    vertices: &mut Vec<StreamVertex>,
    indices: &mut Vec<u32>,
) {
    let n = pts.len();
    if n < 2 {
        return;
    }
    let base = vertices.len() as u32;
    let mut prev_right = Vec3::X;
    for i in 0..n {
        let tangent = if i == 0 {
            (pts[1].0 - pts[0].0).normalize_or(Vec3::NEG_Y)
        } else if i + 1 == n {
            (pts[i].0 - pts[i - 1].0).normalize_or(Vec3::NEG_Y)
        } else {
            ((pts[i].0 - pts[i - 1].0).normalize_or(Vec3::ZERO) + (pts[i + 1].0 - pts[i].0).normalize_or(Vec3::ZERO))
                .normalize_or(Vec3::NEG_Y)
        };
        // Keep horizontal cross-sections upright (`up.y > 0`) whenever not
        // nearly vertical; smoothly transport `prev_right` through vertical drops.
        let right = if tangent.y.abs() < 0.92 {
            tangent.cross(Vec3::Y).normalize_or(prev_right)
        } else {
            let proj = prev_right - tangent * prev_right.dot(tangent);
            if proj.length_squared() > 1e-6 { proj.normalize() } else { tangent.any_orthonormal_vector() }
        };
        let up = right.cross(tangent).normalize_or(Vec3::Y);
        prev_right = right;

        let (rx, ry, y_shift, foam, flat) = profile(i, tangent);
        let center = pts[i].0 + Vec3::Y * y_shift;
        let along = pts[i].1;
        for s in 0..STREAM_SIDES {
            let ang = s as f32 * (std::f32::consts::TAU / STREAM_SIDES as f32);
            let (sin, cos) = ang.sin_cos();
            let offset = right * (cos * rx) + up * (sin * ry);
            let normal = (right * (cos / rx.max(1e-4)) + up * (sin / ry.max(1e-4))).normalize_or(Vec3::Y);
            let across = [cos * rx, cos, flat];
            vertices.push(StreamVertex {
                pos: (center + offset).to_array(),
                along,
                normal: normal.to_array(),
                foam,
                across,
            });
        }
    }
    let stride = STREAM_SIDES as u32;
    for i in 0..(n - 1) as u32 {
        let a = base + i * stride;
        let b = a + stride;
        for s in 0..stride {
            let next = (s + 1) % stride;
            // Winding produces outward-facing CCW triangles.
            indices.extend_from_slice(&[a + s, b + next, a + next, a + s, b + s, b + next]);
        }
    }
    // The ends are left open: they are drawn out thin, or run into other water.
}
