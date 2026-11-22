//! Water surfaces: flat patches drawn over the terrain, each at the level
//! its body has this frame.

use glam::{IVec3, Mat4, Vec3};
use tungsten_core::edit::FP;
use tungsten_core::hash::{IntMap, IntSet};
use tungsten_core::water::{BodyId, WaterPatch};
use tungsten_core::{CHUNK_SIZE, NodeKey};

use crate::arena::{Alloc, Arena};
use crate::terrain::{aabb_visible, frustum_planes};

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
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    vertex_arena: Arena,
    index_arena: Arena,
    nodes: IntMap<NodeKey, Vec<PatchGpu>>,
    /// Current surface level of bodies whose level moves, fixed-point base cells.
    levels: IntMap<BodyId, i64>,
    visible: Vec<Visible>,
    origins: Vec<[f32; 4]>,
}

impl Water {
    /// `globals` is the terrain's uniform buffer, which the water shader
    /// shares; `depth` is the depth attachment the terrain is drawn to.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        samples: u32,
        globals: &wgpu::Buffer,
        depth: &wgpu::TextureView,
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: true,
                    },
                    count: None,
                },
            ],
        });
        let bind_group = Self::bind(device, &layout, globals, depth);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let instance_capacity = 1024;
        Self {
            pipeline,
            layout,
            bind_group,
            instances: Self::instance_buffer(device, instance_capacity),
            instance_capacity,
            vertex_arena: Arena::new("water vertices", wgpu::BufferUsages::VERTEX, 4, 1 << 16),
            index_arena: Arena::new("water indices", wgpu::BufferUsages::INDEX, 4, 1 << 16),
            nodes: IntMap::default(),
            levels: IntMap::default(),
            visible: Vec::new(),
            origins: Vec::new(),
        }
    }

    fn bind(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
            ],
        })
    }

    /// Point at a new depth attachment, after a resize.
    pub fn rebind(&mut self, device: &wgpu::Device, globals: &wgpu::Buffer, depth: &wgpu::TextureView) {
        self.bind_group = Self::bind(device, &self.layout, globals, depth);
    }

    fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("water origins"),
            size: capacity as u64 * 16,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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

    /// Replace the levels of the bodies whose surface has moved since their
    /// patches were made.
    pub fn set_levels(&mut self, levels: impl IntoIterator<Item = (BodyId, i64)>) {
        self.levels.clear();
        self.levels.extend(levels);
    }

    /// Cull, sort back to front and upload per-patch origins. Returns (draws, triangles).
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
                let level = self.levels.get(&p.body).copied().unwrap_or(p.level);
                // In base cells above the camera chunk's origin.
                let y = (level as f64 / FP as f64 - (camera_chunk.y * CHUNK_SIZE) as f64) as f32;
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
        (self.visible.len() as u32, triangles)
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.visible.is_empty() {
            return;
        }
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
}
