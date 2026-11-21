//! Gradient noise over an integer hash.
//!
//! Bit-exact across platforms: only `+ - * /`, `floor` and comparisons on
//! IEEE floats, no fused multiply-add and no platform maths functions.
//! Coordinates are `f64` so precision does not degrade far from the origin;
//! everything after the lattice split is `f32`.

use crate::hash::{hash2, hash3};

#[inline]
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

const D: f32 = std::f32::consts::FRAC_1_SQRT_2;
const GRAD2: [[f32; 2]; 8] = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [D, D], [-D, D], [D, -D], [-D, -D]];

const GRAD3: [[f32; 3]; 16] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [0.0, -1.0, -1.0],
];

#[inline]
fn grad2(seed: u64, ix: i64, iy: i64, fx: f32, fy: f32) -> f32 {
    let g = GRAD2[(hash2(seed, ix, iy) & 7) as usize];
    g[0] * fx + g[1] * fy
}

#[inline]
fn grad3(seed: u64, ix: i64, iy: i64, iz: i64, fx: f32, fy: f32, fz: f32) -> f32 {
    let g = GRAD3[(hash3(seed, ix, iy, iz) & 15) as usize];
    g[0] * fx + g[1] * fy + g[2] * fz
}

/// 2D gradient noise in roughly [-1, 1].
pub fn noise2(seed: u64, x: f64, y: f64) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = ((x - x0) as f32, (y - y0) as f32);
    let (ix, iy) = (x0 as i64, y0 as i64);
    let (u, v) = (fade(fx), fade(fy));
    let a = lerp(grad2(seed, ix, iy, fx, fy), grad2(seed, ix + 1, iy, fx - 1.0, fy), u);
    let b = lerp(grad2(seed, ix, iy + 1, fx, fy - 1.0), grad2(seed, ix + 1, iy + 1, fx - 1.0, fy - 1.0), u);
    lerp(a, b, v) * std::f32::consts::SQRT_2
}

/// 3D gradient noise in roughly [-1, 1].
pub fn noise3(seed: u64, x: f64, y: f64, z: f64) -> f32 {
    let (x0, y0, z0) = (x.floor(), y.floor(), z.floor());
    let (fx, fy, fz) = ((x - x0) as f32, (y - y0) as f32, (z - z0) as f32);
    let (ix, iy, iz) = (x0 as i64, y0 as i64, z0 as i64);
    let (u, v, w) = (fade(fx), fade(fy), fade(fz));
    let g = |dx: i64, dy: i64, dz: i64| {
        grad3(seed, ix + dx, iy + dy, iz + dz, fx - dx as f32, fy - dy as f32, fz - dz as f32)
    };
    let x00 = lerp(g(0, 0, 0), g(1, 0, 0), u);
    let x10 = lerp(g(0, 1, 0), g(1, 1, 0), u);
    let x01 = lerp(g(0, 0, 1), g(1, 0, 1), u);
    let x11 = lerp(g(0, 1, 1), g(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w)
}

/// Fractal sum of `octaves` layers of 2D noise, normalised to roughly [-1, 1].
pub fn fbm2(seed: u64, x: f64, y: f64, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut freq) = (0.0f32, 1.0f32, 0.0f32, 1.0f64);
    for o in 0..octaves {
        sum += amp * noise2(seed.wrapping_add(o as u64), x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

/// Ridged fractal noise in [0, 1], sharp crests at 1.
pub fn ridged2(seed: u64, x: f64, y: f64, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut freq) = (0.0f32, 1.0f32, 0.0f32, 1.0f64);
    for o in 0..octaves {
        let n = 1.0 - noise2(seed.wrapping_add(o as u64), x * freq, y * freq).abs().min(1.0);
        sum += amp * n * n;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}

#[inline]
pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_and_continuity() {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        let mut prev = noise3(1, 0.0, 0.3, 0.7);
        for i in 0..20_000 {
            let t = i as f64 * 0.013;
            let n = noise3(1, t, 0.3 + t * 0.37, 0.7 - t * 0.21);
            assert!((n - prev).abs() < 0.08, "discontinuity at {t}");
            prev = n;
            lo = lo.min(n);
            hi = hi.max(n);
            let m = noise2(2, t, t * 0.61);
            assert!((-1.01..=1.01).contains(&m));
        }
        assert!(lo < -0.5 && hi > 0.5 && lo >= -1.01 && hi <= 1.01, "{lo} {hi}");
    }

    #[test]
    fn far_from_origin_keeps_precision() {
        let base = 3.0e9;
        let a = noise2(5, base + 0.25, 0.5);
        let b = noise2(5, base + 0.26, 0.5);
        assert!(a != b && (a - b).abs() < 0.05);
    }
}
