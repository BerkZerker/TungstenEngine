struct Screen {
    // xy: 2 / viewport size in pixels.
    scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> screen: Screen;

struct VsIn {
    @builtin(vertex_index) vertex: u32,
    // xy: top-left in pixels, z: pixels per font pixel.
    @location(0) origin: vec3<f32>,
    @location(1) bits: vec2<u32>,
    @location(2) color: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) bits: vec2<u32>,
    @location(2) @interpolate(flat) color: vec4<f32>,
}

@vertex
fn vs(in: VsIn) -> VsOut {
    // Two triangles covering a 5x9 glyph.
    let corner = vec2<f32>(f32((in.vertex + 1u) / 3u % 2u), f32(in.vertex % 2u));
    let cell = corner * vec2<f32>(5.0, 9.0);
    let px = in.origin.xy + cell * in.origin.z;
    var out: VsOut;
    out.clip = vec4<f32>(px.x * screen.scale.x - 1.0, 1.0 - px.y * screen.scale.y, 0.0, 1.0);
    out.uv = cell;
    out.bits = in.bits;
    out.color = in.color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let p = vec2<u32>(clamp(in.uv, vec2<f32>(0.0), vec2<f32>(4.999, 8.999)));
    let i = p.y * 5u + p.x;
    var bit: u32;
    if i < 32u {
        bit = (in.bits.x >> i) & 1u;
    } else {
        bit = (in.bits.y >> (i - 32u)) & 1u;
    }
    if bit == 0u {
        discard;
    }
    return in.color;
}
