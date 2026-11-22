struct Screen {
    // xy: 2 / viewport size in pixels.
    scale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> screen: Screen;

struct VsIn {
    @builtin(vertex_index) vertex: u32,
    // xy: top-left in pixels, zw: pixels per font pixel across and down.
    @location(0) origin: vec4<f32>,
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
    let px = in.origin.xy + cell * in.origin.zw;
    var out: VsOut;
    out.clip = vec4<f32>(px.x * screen.scale.x - 1.0, 1.0 - px.y * screen.scale.y, 0.0, 1.0);
    out.uv = cell;
    out.bits = in.bits;
    out.color = in.color;
    return out;
}

// 1 where the glyph has a font pixel at `uv`, 0 elsewhere.
fn lit(bits: vec2<u32>, uv: vec2<f32>) -> f32 {
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x >= 5.0 || uv.y >= 9.0) {
        return 0.0;
    }
    let i = u32(uv.y) * 5u + u32(uv.x);
    if (i < 32u) {
        return f32((bits.x >> i) & 1u);
    }
    return f32((bits.y >> (i - 32u)) & 1u);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    // How much of this screen pixel the glyph covers, from four points in
    // it. At a whole number of screen pixels per font pixel all four agree
    // and the text is as crisp as the font; in between, edges are evened out.
    let dx = dpdx(in.uv);
    let dy = dpdy(in.uv);
    var cover = 0.0;
    cover += lit(in.bits, in.uv + dx * 0.125 + dy * 0.375);
    cover += lit(in.bits, in.uv - dx * 0.375 + dy * 0.125);
    cover += lit(in.bits, in.uv + dx * 0.375 - dy * 0.125);
    cover += lit(in.bits, in.uv - dx * 0.125 - dy * 0.375);
    if (cover == 0.0) {
        discard;
    }
    return vec4<f32>(in.color.rgb, in.color.a * cover * 0.25);
}
