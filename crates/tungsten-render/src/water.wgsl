struct Globals {
    view_proj: mat4x4<f32>,
    // xyz: camera position relative to the camera chunk's origin, w: cell size.
    camera: vec4<f32>,
    // xyz: direction towards the sun.
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_color: vec4<f32>,
    ground_color: vec4<f32>,
    fog_color: vec4<f32>,
    // x: fog start distance, y: fog end distance.
    fog: vec4<f32>,
    materials: array<vec4<f32>, 4>,
    // xy: where the camera chunk's origin is in the ripple pattern, metres.
    // z: time in seconds.
    water: vec4<f32>,
}

@group(0) @binding(0) var<uniform> g: Globals;
// Distance to the terrain already drawn, to tell how much water is in front of it.
@group(0) @binding(1) var terrain_distance: texture_multisampled_2d<f32>;

// The ripple pattern repeats every this many metres, so it can be placed
// from a position kept small enough to stay precise.
const PERIOD: f32 = 64.0;
const TAU: f32 = 6.2831853;

const DEEP: vec3<f32> = vec3<f32>(0.010, 0.055, 0.085);
// How fast water hides what is under it, per metre looked through.
const MURK: f32 = 0.45;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
}

@vertex
fn vs(
    // Corner in cells of the node.
    @location(0) pos: vec2<u32>,
    // xz: the node's origin relative to the camera chunk's origin, in base
    // cells; y: the surface height likewise; w: base cells per cell of the node.
    @location(1) node: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    let cells = vec3<f32>(f32(pos.x) * node.w + node.x, node.y, f32(pos.y) * node.w + node.z);
    out.world = cells * g.camera.w;
    // Not pushed back in depth like its node's terrain: the ground under it
    // can be a finer node's, pushed back less, which would hide shallow
    // water along the seam. Terrain that is in front of it is told by the
    // distance attachment instead.
    out.clip = g.view_proj * vec4<f32>(out.world, 1.0);
    return out;
}

// One wave: `dir` in whole cycles per PERIOD on each axis, so the pattern
// tiles; `steep` is its largest slope.
fn wave(p: vec2<f32>, t: f32, dir: vec2<f32>, steep: f32) -> vec2<f32> {
    let k = dir * (TAU / PERIOD);
    let len = length(k);
    // Deep-water waves: longer ones travel faster.
    let phase = dot(k, p) + sqrt(9.8 * len) * t;
    return k / len * steep * cos(phase);
}

// Slope of the surface at a point.
fn ripple(p: vec2<f32>, t: f32) -> vec2<f32> {
    return wave(p, t, vec2<f32>(9.0, 4.0), 0.030)
        + wave(p, t, vec2<f32>(-5.0, 11.0), 0.028)
        + wave(p, t, vec2<f32>(17.0, -8.0), 0.024)
        + wave(p, t, vec2<f32>(23.0, 19.0), 0.020)
        + wave(p, t, vec2<f32>(-31.0, 37.0), 0.016);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let to_camera = g.camera.xyz - in.world;
    let dist = length(to_camera);
    let v = to_camera / dist;
    // Ripples fade with distance, where they would be smaller than a pixel.
    let slope = ripple(in.world.xz + g.water.xy, g.water.z) / (1.0 + dist * dist / 22500.0);
    var n = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    let fog = smoothstep(g.fog.x, g.fog.y, dist);
    // Metres of water looked through before the ground behind it.
    let ground = textureLoad(terrain_distance, vec2<i32>(in.clip.xy), 0).r;
    let through = max(ground - dist, 0.0);

    if (v.y < 0.0) {
        // Seen from below: a bright, mostly clear ceiling.
        let glow = mix(DEEP * 4.0, g.fog_color.rgb, 0.6 * max(dot(-n, v), 0.0));
        return vec4<f32>(mix(glow, g.fog_color.rgb, fog), select(0.0, 0.55, through > 0.0));
    }

    // The thinnest water at a shore is not drawn: the surface is flat and
    // the lattice it was worked out on is not, so its edge is only near.
    let shore = smoothstep(0.0, 0.08, through * v.y);

    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);
    let r = reflect(-v, n);
    let zenith = g.fog_color.rgb * vec3<f32>(0.42, 0.60, 0.92);
    let sky = mix(g.fog_color.rgb, zenith, sqrt(clamp(r.y, 0.0, 1.0)));
    let sun = g.sun_color.rgb * pow(max(dot(r, g.sun_dir.xyz), 0.0), 400.0) * 3.0;
    let body = DEEP * (1.0 + 2.0 * max(g.sun_dir.y, 0.0));
    let color = mix(body, sky, fresnel) + sun;
    // What gets through: not reflected, and not lost in the water.
    let clear = 0.84 * exp(-through * MURK) * (1.0 - fresnel);
    let alpha = clamp(1.0 - clear + sun.r, 0.0, 1.0) * shore;
    return vec4<f32>(mix(color, g.fog_color.rgb, fog), alpha);
}

// Moving water: pouring arcs and downhill streams.

// Must match `SPEED` of tungsten-core's water at the default cell size:
// the pattern is carried along at the speed the water runs.
const STREAM_SPEED: f32 = 5.6;
const SHALLOW: vec3<f32> = vec3<f32>(0.020, 0.085, 0.200);

struct StreamIn {
    @location(0) pos: vec3<f32>,
    @location(1) along: f32,
    @location(2) normal: vec3<f32>,
    @location(3) foam: f32,
    @location(4) across: vec3<f32>,
}

struct StreamOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) along: f32,
    @location(3) foam: f32,
    @location(4) across: vec3<f32>,
}

@vertex
fn vs_stream(in: StreamIn) -> StreamOut {
    var out: StreamOut;
    out.world = in.pos;
    out.clip = g.view_proj * vec4<f32>(in.pos, 1.0);
    out.normal = in.normal;
    out.along = in.along;
    out.foam = in.foam;
    out.across = in.across;
    return out;
}

fn hash(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Smooth noise in 0..1, one feature per unit.
fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(hash(i), hash(i + vec2<f32>(1.0, 0.0)), u.x);
    let b = mix(hash(i + vec2<f32>(0.0, 1.0)), hash(i + vec2<f32>(1.0, 1.0)), u.x);
    return mix(a, b, u.y);
}

@fragment
fn fs_stream(in: StreamOut) -> @location(0) vec4<f32> {
    let to_camera = g.camera.xyz - in.world;
    let dist = length(to_camera);
    let v = to_camera / max(dist, 1e-4);

    // Streaks drawn out along the flow and carried down it.
    let run = in.along - g.water.z * STREAM_SPEED;
    let a = noise(vec2<f32>(in.across.x * 13.0, run * 0.6));
    let b = noise(vec2<f32>(in.across.x * 37.0 + 7.3, run * 1.7 + a));
    let streak = a * 0.6 + b * 0.4;
    let rough = vec3<f32>(a - 0.5, 0.0, b - 0.5) * 0.45 / (1.0 + dist * dist / 4000.0);
    var n = normalize(in.normal + rough);
    if (dot(n, v) < 0.0) {
        n = -n;
    }

    let fresnel = 0.03 + 0.97 * pow(1.0 - max(dot(n, v), 0.0), 4.5);
    let r = reflect(-v, n);
    let zenith = g.fog_color.rgb * vec3<f32>(0.42, 0.60, 0.92);
    let sky = mix(g.fog_color.rgb, zenith, sqrt(clamp(r.y, 0.0, 1.0)));
    let sun = g.sun_color.rgb * pow(max(dot(r, g.sun_dir.xyz), 0.0), 260.0) * 2.0;
    let body = SHALLOW * (0.5 + 1.5 * max(g.sun_dir.y, 0.0));

    // White water: in streaks where it is steep, and along the edges of
    // water on the ground.
    let edge = abs(in.across.y);
    let torn = in.foam * smoothstep(0.5, 0.8, streak);
    let fringe = in.across.z * smoothstep(0.5, 0.95, edge) * smoothstep(0.35, 0.7, b) * 0.4;
    let white = clamp(torn + fringe, 0.0, 0.9);
    let foam_col = mix(g.fog_color.rgb, vec3<f32>(0.90, 0.95, 0.99), 0.75);
    let color = mix(mix(body, sky, fresnel) + sun, foam_col, white);

    // Thin towards its edges and where the ground comes up through it, so
    // that it lies on the ground instead of standing on it.
    let ground = textureLoad(terrain_distance, vec2<i32>(in.clip.xy), 0).r;
    let through = max(ground - dist, 0.0);
    let contact = smoothstep(0.0, 0.05, through);
    let sheet = 1.0 - smoothstep(0.72, 1.0, edge);
    // A jet thins towards its outline instead, whichever way it is seen from.
    let jet = smoothstep(0.0, 0.55, abs(dot(normalize(in.normal), v)));
    let side = mix(jet, sheet, in.across.z);
    let alpha = clamp(0.62 + 0.3 * fresnel + 0.3 * white + sun.r, 0.0, 0.95) * side * contact;
    let fog = smoothstep(g.fog.x, g.fog.y, dist);
    return vec4<f32>(mix(color, g.fog_color.rgb, fog), alpha);
}
