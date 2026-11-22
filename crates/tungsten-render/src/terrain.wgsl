struct Globals {
    view_proj: mat4x4<f32>,
    // xyz: camera position relative to the camera chunk's origin, w: cell size.
    camera: vec4<f32>,
    // xyz: direction towards the sun.
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    // rgb: fog colour, w: unused.
    fog_color: vec4<f32>,
    // x: fog start distance, y: fog end distance.
    fog: vec4<f32>,
    materials: array<vec4<f32>, 4>,
    // Used by water.wgsl.
    water: vec4<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;

// How far back coarser nodes are depth-tested, in their own cells.
const DEPTH_PUSH: f32 = 2.0;

struct VsIn {
    // xyz: (cells + 3) * 1024. w: two materials in the low byte, four bits
    // each, and the second one's share of this vertex in the high byte.
    @location(0) pos: vec4<u32>,
    @location(1) normal: vec2<f32>,
    // xyz: the node's origin relative to the camera chunk's origin, in base
    // cells, less its apron: always a whole number. w: base cells per cell of
    // the node (1 at the finest level, doubling with each coarser one).
    @location(2) chunk: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // The triangle's two materials: its three vertices agree.
    @location(2) @interpolate(flat) mats: u32,
    // The second material's share, 0 to 1.
    @location(3) share: f32,
}

fn oct_decode(e: vec2<f32>) -> vec3<f32> {
    let z = 1.0 - abs(e.x) - abs(e.y);
    let t = max(-z, 0.0);
    let s = vec2<f32>(select(-1.0, 1.0, e.x >= 0.0), select(-1.0, 1.0, e.y >= 0.0));
    return normalize(vec3<f32>(e - t * s, z));
}

@vertex
fn vs(in: VsIn) -> VsOut {
    var out: VsOut;
    // Both terms are exact in f32, so a vertex shared by two nodes lands on
    // identical coordinates from either side and seams cannot crack.
    let cells = vec3<f32>(in.pos.xyz) * (in.chunk.w / 1024.0) + in.chunk.xyz;
    out.world = cells * g.camera.w;
    out.clip = g.view_proj * vec4<f32>(out.world, 1.0);
    // Coarser nodes overlap the finer ones beside them by a cell. Depth-test
    // them as if they were a little further away, so the finer mesh wins
    // wherever both exist and the coarse one only shows through gaps.
    // (Reverse-Z with an infinite far plane: depth is near / distance.)
    if (in.chunk.w > 1.0 && out.clip.w > 0.0) {
        let push = DEPTH_PUSH * in.chunk.w * g.camera.w;
        out.clip.z = out.clip.z * out.clip.w / (out.clip.w + push);
    }
    out.normal = oct_decode(in.normal);
    out.mats = in.pos.w & 0xffu;
    out.share = f32(in.pos.w >> 8u) / 255.0;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    // Distance from the camera, for water drawn over this to tell how much
    // of it there is in front. The depth buffer will not do: it has coarse
    // nodes pushed back.
    @location(1) distance: vec4<f32>,
}

@fragment
fn fs(in: VsOut) -> FsOut {
    // Two soft materials meet where their shares are equal: a sharp line,
    // smoothed over one pixel, not a fade.
    let second = clamp((in.share - 0.5) / max(fwidth(in.share), 1e-5) + 0.5, 0.0, 1.0);
    let albedo = mix(g.materials[in.mats & 3u].rgb, g.materials[(in.mats >> 4u) & 3u].rgb, second);
    let to_camera = g.camera.xyz - in.world;
    // Faceted materials carry their face normal in the vertices.
    let n = normalize(in.normal);

    let hemi = mix(g.ground_color.rgb, g.sky_color.rgb, n.y * 0.5 + 0.5);
    let sun = g.sun_color.rgb * max(dot(n, g.sun_dir.xyz), 0.0);
    var color = albedo * (hemi + sun);

    let distance = length(to_camera);
    let fog = smoothstep(g.fog.x, g.fog.y, distance);
    color = mix(color, g.fog_color.rgb, fog);
    return FsOut(vec4<f32>(color, 1.0), vec4<f32>(distance, 0.0, 0.0, 0.0));
}
