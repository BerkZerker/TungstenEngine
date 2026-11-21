use bytemuck::{Pod, Zeroable};

use crate::font;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Glyph {
    origin: [f32; 3],
    bits: [u32; 2],
    color: [u8; 4],
}

/// Text queued for one frame. Coordinates are physical pixels from the top-left.
#[derive(Default)]
pub struct TextBatch {
    glyphs: Vec<Glyph>,
}

impl TextBatch {
    pub fn clear(&mut self) {
        self.glyphs.clear();
    }

    /// Advance in pixels per character at `scale`.
    pub fn advance(scale: f32) -> f32 {
        (font::GLYPH_W + 1) as f32 * scale
    }

    /// Height in pixels of a line at `scale`.
    pub fn line_height(scale: f32) -> f32 {
        (font::GLYPH_H + 2) as f32 * scale
    }

    /// Queue `text` with its top-left at `(x, y)`, with a one-font-pixel drop shadow.
    pub fn push(&mut self, x: f32, y: f32, scale: f32, color: [u8; 4], text: &str) {
        for (pass, color) in [(scale, [0, 0, 0, 200]), (0.0, color)] {
            let (mut cx, mut cy) = (x, y);
            for c in text.chars() {
                if c == '\n' {
                    cx = x;
                    cy += Self::line_height(scale);
                    continue;
                }
                let bits = font::glyph(c);
                if bits != 0 {
                    self.glyphs.push(Glyph {
                        origin: [cx + pass, cy + pass, scale],
                        bits: [bits as u32, (bits >> 32) as u32],
                        color,
                    });
                }
                cx += Self::advance(scale);
            }
        }
    }
}

pub(crate) struct TextPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    instances: wgpu::Buffer,
    capacity: usize,
}

impl TextPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, samples: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text"),
            source: wgpu::ShaderSource::Wgsl(include_str!("text.wgsl").into()),
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text screen"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text"),
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
            label: Some("text"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("text"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<Glyph>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32x2, 2 => Unorm8x4],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
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
        let capacity = 4096;
        Self { pipeline, bind_group, uniform, instances: Self::buffer(device, capacity), capacity }
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("text glyphs"),
            size: (capacity * size_of::<Glyph>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Upload this frame's glyphs. Returns the instance count to draw.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: (u32, u32), batch: &TextBatch) -> u32 {
        if batch.glyphs.is_empty() {
            return 0;
        }
        if batch.glyphs.len() > self.capacity {
            self.capacity = batch.glyphs.len().next_power_of_two();
            self.instances = Self::buffer(device, self.capacity);
        }
        let scale = [2.0 / size.0 as f32, 2.0 / size.1 as f32, 0.0, 0.0];
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&scale));
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&batch.glyphs));
        batch.glyphs.len() as u32
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        if count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        pass.draw(0..6, 0..count);
    }
}
