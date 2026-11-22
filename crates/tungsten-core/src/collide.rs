//! Collision geometry taken straight from the samples.
//!
//! Triangles are produced with the mesher's own vertex placement, so what a
//! body touches is what is drawn, with no dependence on which chunk meshes
//! happen to exist. Outside the bit-exact contract.

use glam::Vec3;

use crate::coords::SOLID;
use crate::field::LocalField;
use crate::mesh::{EDGE_CELLS, quad_split, solid_mask, surface_point};

/// Append the surface triangles inside `field`, in cells relative to the
/// field's origin, wound counter-clockwise seen from outside.
pub fn triangles(field: &LocalField, out: &mut Vec<[Vec3; 3]>) {
    let d = field.dims;
    if d.min_element() < 3 {
        return;
    }
    let (cx, cy, cz) = ((d.x - 1) as usize, (d.y - 1) as usize, (d.z - 1) as usize);
    let [sx, sy, sz] = field.strides();
    let mut verts: Vec<Option<Vec3>> = vec![None; cx * cy * cz];
    let cell = |x: i32, y: i32, z: i32| (y as usize * cz + z as usize) * cx + x as usize;
    for y in 0..d.y - 1 {
        for z in 0..d.z - 1 {
            for x in 0..d.x - 1 {
                let b = field.index(x, y, z);
                let f = [
                    field.fill[b],
                    field.fill[b + sx],
                    field.fill[b + sy],
                    field.fill[b + sy + sx],
                    field.fill[b + sz],
                    field.fill[b + sz + sx],
                    field.fill[b + sz + sy],
                    field.fill[b + sz + sy + sx],
                ];
                let mask = solid_mask(&f);
                if mask != 0 && mask != 0xff {
                    verts[cell(x, y, z)] = Some(Vec3::new(x as f32, y as f32, z as f32) + surface_point(&f, mask));
                }
            }
        }
    }
    for y in 1..d.y - 1 {
        for z in 1..d.z - 1 {
            for x in 1..d.x - 1 {
                let i = field.index(x, y, z);
                let s0 = field.fill[i] >= SOLID;
                for (stride, cells) in [sx, sy, sz].into_iter().zip(EDGE_CELLS) {
                    if (field.fill[i + stride] >= SOLID) == s0 {
                        continue;
                    }
                    // A mixed edge guarantees all four cells have a vertex.
                    let p: [Vec3; 4] = std::array::from_fn(|k| {
                        verts[cell(x + cells[k][0], y + cells[k][1], z + cells[k][2])].unwrap()
                    });
                    let t = quad_split(p);
                    for tri in t.chunks(3) {
                        out.push(if s0 {
                            [p[tri[0]], p[tri[1]], p[tri[2]]]
                        } else {
                            [p[tri[2]], p[tri[1]], p[tri[0]]]
                        });
                    }
                }
            }
        }
    }
}

/// Closest point to `p` on triangle `abc`.
pub fn closest_point_on_triangle(p: Vec3, [a, b, c]: [Vec3; 3]) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    /// Unit direction to push the sphere out along.
    pub normal: Vec3,
    pub depth: f32,
}

/// Overlap of a sphere with a one-sided triangle.
pub fn sphere_triangle(center: Vec3, radius: f32, tri: [Vec3; 3]) -> Option<Contact> {
    let closest = closest_point_on_triangle(center, tri);
    let delta = center - closest;
    let dist2 = delta.length_squared();
    if dist2 >= radius * radius {
        return None;
    }
    let face = (tri[1] - tri[0]).cross(tri[2] - tri[0]).normalize_or_zero();
    let dist = dist2.sqrt();
    let along = delta.dot(face);
    if along < 0.0 && along * along > 0.98 * dist2 {
        // The centre is directly beneath the face: push out through it. Being
        // behind the plane of a triangle whose nearest point is on its rim is
        // different, and normal at the crest of a ridge.
        return Some(Contact { normal: face, depth: radius + dist });
    }
    let normal = if dist > 1e-5 { delta / dist } else { face };
    Some(Contact { normal, depth: radius - dist })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::IVec3;

    #[test]
    fn flat_ground_gives_upward_triangles_at_the_surface() {
        let dims = IVec3::new(6, 6, 6);
        let mut field = LocalField::new(IVec3::ZERO, dims, 0);
        // Solid below y = 2.5.
        for y in 0..6 {
            for z in 0..6 {
                for x in 0..6 {
                    let i = field.index(x, y, z);
                    field.fill[i] = (127.5 + (2.5 - y as f32) * 64.0).clamp(0.0, 255.0) as u8;
                }
            }
        }
        let mut tris = Vec::new();
        triangles(&field, &mut tris);
        assert_eq!(tris.len(), 4 * 4 * 2);
        for t in &tris {
            let n = (t[1] - t[0]).cross(t[2] - t[0]).normalize();
            assert!(n.y > 0.999);
            assert!(t.iter().all(|p| (p.y - 2.5).abs() < 0.01), "{t:?}");
        }
        let c = sphere_triangle(Vec3::new(2.5, 2.8, 2.5), 0.5, tris[0]);
        let hit = tris.iter().filter_map(|&t| sphere_triangle(Vec3::new(2.5, 2.8, 2.5), 0.5, t)).next().unwrap();
        assert!(c.is_some() || hit.depth > 0.0);
        assert!((hit.depth - 0.2).abs() < 0.02 && hit.normal.y > 0.99);
    }
}
