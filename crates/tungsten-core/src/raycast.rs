//! Ray against the surface of the fill field. Outside the bit-exact contract.

use glam::{DVec3, IVec3, Vec3};

use crate::chunk::Volume;
use crate::field::LocalField;

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Hit position in cells (world lattice space).
    pub pos: DVec3,
    /// Unit surface normal, pointing out of the solid.
    pub normal: Vec3,
    /// Distance from the ray origin in cells.
    pub distance: f32,
    /// The fullest sample of the cell that was hit, and its contents.
    pub sample: IVec3,
    pub material: u8,
    pub fill: u8,
}

struct Sampler<'a> {
    field: &'a LocalField,
}

impl Sampler<'_> {
    /// Trilinear fill at a position in cells relative to the field origin.
    fn fill(&self, p: Vec3) -> f32 {
        let f = self.field;
        let max = (f.dims - 2).as_vec3();
        let p = p.clamp(Vec3::ZERO, max + 0.999);
        let b = p.floor();
        let t = p - b;
        let i = f.index(b.x as i32, b.y as i32, b.z as i32);
        let [sx, sy, sz] = f.strides();
        let v = |o: usize| f.fill[i + o] as f32;
        let x00 = v(0) + (v(sx) - v(0)) * t.x;
        let x10 = v(sy) + (v(sy + sx) - v(sy)) * t.x;
        let x01 = v(sz) + (v(sz + sx) - v(sz)) * t.x;
        let x11 = v(sz + sy) + (v(sz + sy + sx) - v(sz + sy)) * t.x;
        let y0 = x00 + (x10 - x00) * t.y;
        let y1 = x01 + (x11 - x01) * t.y;
        y0 + (y1 - y0) * t.z
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let e = 0.5;
        Vec3::new(
            self.fill(p + Vec3::X * e) - self.fill(p - Vec3::X * e),
            self.fill(p + Vec3::Y * e) - self.fill(p - Vec3::Y * e),
            self.fill(p + Vec3::Z * e) - self.fill(p - Vec3::Z * e),
        )
    }
}

/// Cast a ray from `origin` (cells) along unit `dir` for up to `max_distance` cells.
pub fn raycast(volume: &Volume, origin: DVec3, dir: Vec3, max_distance: f32) -> Option<RayHit> {
    const STEP: f32 = 0.2;
    const SURFACE: f32 = 127.5;
    let end = origin + (dir * max_distance).as_dvec3();
    let min = origin.min(end).floor().as_ivec3() - 2;
    let max = origin.max(end).ceil().as_ivec3() + 3;
    let field = volume.copy_region(min, max - min);
    let s = Sampler { field: &field };
    let local = (origin - min.as_dvec3()).as_vec3();

    let mut t_prev = 0.0f32;
    if s.fill(local) >= SURFACE {
        return Some(finish(&s, local, origin, -dir, 0.0, true));
    }
    let mut t = STEP;
    while t_prev < max_distance {
        let t_now = t.min(max_distance);
        if s.fill(local + dir * t_now) >= SURFACE {
            // Bisect to the crossing.
            let (mut lo, mut hi) = (t_prev, t_now);
            for _ in 0..10 {
                let mid = (lo + hi) * 0.5;
                if s.fill(local + dir * mid) >= SURFACE { hi = mid } else { lo = mid }
            }
            let p = local + dir * hi;
            return Some(finish(&s, p, origin + (dir * hi).as_dvec3(), -dir, hi, false));
        }
        t_prev = t_now;
        t += STEP;
    }
    None
}

fn finish(s: &Sampler, local: Vec3, pos: DVec3, fallback_normal: Vec3, distance: f32, inside: bool) -> RayHit {
    let g = s.gradient(local);
    let normal = if !inside && g.length_squared() > 1e-4 { -g.normalize() } else { fallback_normal };
    // Fullest corner of the cell containing the hit.
    let f = s.field;
    let b = local.floor().as_ivec3().clamp(IVec3::ZERO, f.dims - 2);
    let mut best = (b, f.get(b.x, b.y, b.z));
    for c in 1..8 {
        let q = b + IVec3::new(c & 1, (c >> 1) & 1, (c >> 2) & 1);
        let v = f.get(q.x, q.y, q.z);
        if v.fill > best.1.fill {
            best = (q, v);
        }
    }
    RayHit { pos, normal, distance, sample: f.origin + best.0, material: best.1.mat, fill: best.1.fill }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Sample};

    #[test]
    fn hits_flat_ground_from_above() {
        let mut vol = Volume::new();
        vol.insert(IVec3::new(0, 0, 0), Chunk::AIR);
        vol.insert(IVec3::new(0, -1, 0), Chunk::Uniform(Sample { fill: 255, mat: 2 }));
        // Fill 255 at y=-1 and 0 at y=0: the surface sits halfway, at y=-0.5.
        let hit = raycast(&vol, DVec3::new(10.3, 6.0, 12.7), Vec3::NEG_Y, 20.0).unwrap();
        assert!((hit.pos.y + 0.5).abs() < 0.02, "{}", hit.pos.y);
        assert!((hit.distance - 6.5).abs() < 0.02);
        assert!(hit.normal.y > 0.99);
        assert_eq!((hit.material, hit.fill, hit.sample.y), (2, 255, -1));
        assert!(raycast(&vol, DVec3::new(10.3, 6.0, 12.7), Vec3::Y, 20.0).is_none());
        assert!(raycast(&vol, DVec3::new(10.3, 6.0, 12.7), Vec3::NEG_Y, 5.0).is_none());
    }
}
