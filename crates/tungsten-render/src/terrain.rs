use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Mat4, Vec3, Vec4};
use tungsten_core::hash::{IntMap, IntSet};
use tungsten_core::mesh::{APRON, ChunkMesh, MeshVertex};
use tungsten_core::{CHUNK_SIZE, NodeKey};

use crate::arena::{Alloc, Arena};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub camera: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_color: [f32; 4],
    pub ground_color: [f32; 4],
    pub fog_color: [f32; 4],
    pub fog: [f32; 4],
    pub materials: [[f32; 4]; 4],
    pub water: [f32; 4],
}

struct ChunkGpu {
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

pub(crate) struct Terrain {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    globals: wgpu::Buffer,
    instances: wgpu::Buffer,
    instance_capacity: usize,
    vertex_arena: Arena,
    index_arena: Arena,
    chunks: IntMap<NodeKey, ChunkGpu>,
    /// Nodes that are to be drawn, whether or not they have a mesh.
    shown: IntSet<NodeKey>,
    visible: Vec<Visible>,
    origins: Vec<[f32; 4]>,
}

/// Frustum planes (left, right, bottom, top, near) of a reverse-Z clip matrix.
pub(crate) fn frustum_planes(m: &Mat4) -> [Vec4; 5] {
    let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2]
}

pub(crate) fn aabb_visible(planes: &[Vec4; 5], min: Vec3, max: Vec3) -> bool {
    let (c, e) = ((min + max) * 0.5, (max - min) * 0.5);
    planes.iter().all(|p| {
        let n = p.truncate();
        n.dot(c) + p.w + n.abs().dot(e) >= 0.0
    })
}

impl Terrain {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("terrain"),
            source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain globals"),
            size: size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("terrain"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("terrain"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: size_of::<MeshVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Uint16x4, 1 => Snorm16x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x4],
                    }),
                ],
            },
            primitive: wgpu::PrimitiveState {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                // Reverse-Z: nearer fragments have larger depth.
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(crate::renderer::DISTANCE_FORMAT.into()),
                ],
            }),
            multiview_mask: None,
            cache: None,
        });
        let instance_capacity = 2048;
        Self {
            pipeline,
            bind_group,
            globals,
            instances: Self::instance_buffer(device, instance_capacity),
            instance_capacity,
            vertex_arena: Arena::new(
                "terrain vertices",
                wgpu::BufferUsages::VERTEX,
                size_of::<MeshVertex>() as u64,
                1 << 20,
            ),
            index_arena: Arena::new("terrain indices", wgpu::BufferUsages::INDEX, 4, 1 << 21),
            chunks: IntMap::default(),
            shown: IntSet::default(),
            visible: Vec::new(),
            origins: Vec::new(),
        }
    }

    fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk origins"),
            size: capacity as u64 * 16,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The uniform buffer every world-space shader reads.
    pub fn globals(&self) -> &wgpu::Buffer {
        &self.globals
    }

    /// Nodes that are to be drawn.
    pub fn shown(&self) -> &IntSet<NodeKey> {
        &self.shown
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn memory(&self) -> (u64, u64) {
        (
            self.vertex_arena.used_bytes() + self.index_arena.used_bytes(),
            self.vertex_arena.capacity_bytes() + self.index_arena.capacity_bytes(),
        )
    }

    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, key: NodeKey, mesh: &ChunkMesh) {
        self.remove(key);
        if mesh.indices.is_empty() {
            return;
        }
        debug_assert!(mesh.indices.len().is_multiple_of(2));
        let vertices = self.vertex_arena.alloc(device, mesh.vertices.len() as u32);
        let indices = self.index_arena.alloc(device, (mesh.indices.len() / 2) as u32);
        self.vertex_arena.write(queue, vertices, bytemuck::cast_slice(&mesh.vertices));
        self.index_arena.write(queue, indices, bytemuck::cast_slice(&mesh.indices));
        self.chunks.insert(key, ChunkGpu { vertices, indices, index_count: mesh.indices.len() as u32 });
    }

    pub fn remove(&mut self, key: NodeKey) {
        if let Some(c) = self.chunks.remove(&key) {
            self.vertex_arena.free(c.vertices);
            self.index_arena.free(c.indices);
        }
    }

    /// Change which nodes are drawn.
    pub fn set_visible(&mut self, show: &[NodeKey], hide: &[NodeKey]) {
        for key in hide {
            self.shown.remove(key);
        }
        self.shown.extend(show.iter().copied());
    }

    /// Cull, sort front to back and upload per-node origins. Returns (draws, triangles).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        globals: &Globals,
        view_proj: &Mat4,
        camera_chunk: IVec3,
        camera_local: Vec3,
        cell_size: f32,
    ) -> (u32, u32) {
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(globals));
        let planes = frustum_planes(view_proj);
        self.visible.clear();
        let mut triangles = 0;
        for (key, c) in &self.chunks {
            if !self.shown.contains(key) {
                continue;
            }
            // In base cells from the camera chunk's origin: whole numbers, exact in f32.
            let step = key.step();
            let base = key.origin() - camera_chunk * CHUNK_SIZE;
            let cells = (base - APRON * step).as_vec3();
            let spacing = cell_size * step as f32;
            let origin = base.as_vec3() * cell_size;
            // Meshes reach up to two cells beyond the chunk's own.
            let (min, max) = (origin - 2.0 * spacing, origin + (CHUNK_SIZE + 2) as f32 * spacing);
            if !aabb_visible(&planes, min, max) {
                continue;
            }
            triangles += c.index_count / 3;
            self.visible.push(Visible {
                distance: ((min + max) * 0.5).distance_squared(camera_local),
                vertex_page: c.vertices.page,
                index_page: c.indices.page,
                first_index: c.indices.offset * 2,
                index_count: c.index_count,
                base_vertex: c.vertices.offset as i32,
                origin: [cells.x, cells.y, cells.z, step as f32],
            });
        }
        self.visible.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
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
