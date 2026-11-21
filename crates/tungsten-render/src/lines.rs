use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct LineVertex {
    pos: [f32; 3],
    color: [u8; 4],
}

/// World-space line segments queued for one frame, drawn over the terrain.
/// Positions are metres relative to the camera chunk's origin.
#[derive(Default)]
pub struct LineBatch {
    vertices: Vec<LineVertex>,
}

impl LineBatch {
    pub fn clear(&mut self) {
        self.vertices.clear();
    }

    pub fn segment(&mut self, a: Vec3, b: Vec3, color: [u8; 4]) {
        self.vertices.push(LineVertex { pos: a.to_array(), color });
        self.vertices.push(LineVertex { pos: b.to_array(), color });
    }

    /// A circle of `radius` around `center` in the plane facing `normal`.
    pub fn ring(&mut self, center: Vec3, normal: Vec3, radius: f32, color: [u8; 4]) {
        const SEGMENTS: usize = 48;
        let n = normal.normalize_or(Vec3::Y);
        let u = n.any_orthonormal_vector();
        let v = n.cross(u);
        let point = |i: usize| {
            let (s, c) = (i as f32 / SEGMENTS as f32 * std::f32::consts::TAU).sin_cos();
            center + (u * c + v * s) * radius
        };
        for i in 0..SEGMENTS {
            self.segment(point(i), point(i + 1), color);
        }
    }
}

pub(crate) struct LinePipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    capacity: usize,
}

impl LinePipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lines"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lines.wgsl").into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lines view"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lines"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<LineVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Unorm8x4],
                })],
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineList, ..Default::default() },
            // Drawn over everything: a brush outline must stay readable inside a hole.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::renderer::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
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
        let capacity = 1024;
        Self { pipeline, bind_group, uniform, vertices: Self::buffer(device, capacity), capacity }
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("line vertices"),
            size: (capacity * size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Upload this frame's lines. Returns the vertex count to draw.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view_proj: &Mat4, batch: &LineBatch) -> u32 {
        if batch.vertices.is_empty() {
            return 0;
        }
        if batch.vertices.len() > self.capacity {
            self.capacity = batch.vertices.len().next_power_of_two();
            self.vertices = Self::buffer(device, self.capacity);
        }
        queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&view_proj.to_cols_array()));
        queue.write_buffer(&self.vertices, 0, bytemuck::cast_slice(&batch.vertices));
        batch.vertices.len() as u32
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.draw(0..count, 0..1);
    }
}
