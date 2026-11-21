//! Sand slump: loose material settles to its angle of repose after edits.
//!
//! Part of the bit-exact contract: integer transfers of fill between
//! samples, processed in a fixed order. Total fill is conserved exactly.

use std::collections::{BTreeMap, BTreeSet};

use glam::IVec3;

use crate::chunk::Volume;
use crate::coords::{CHUNK_SIZE, ChunkPos, SOLID, chunk_of, chunk_origin};
use crate::material;

/// Samples of margin copied around a chunk: a sample reads two below and one to each side.
const MARGIN: i32 = 3;
/// Height difference between neighbouring columns above which sand moves,
/// in fill units (255 per cell): tan(34 degrees) of a cell.
const REPOSE: i32 = 171;
/// The same for diagonal neighbours, which are sqrt(2) further apart.
const REPOSE_DIAGONAL: i32 = 242;
/// Simulation ticks between sand steps.
pub const SAND_INTERVAL: u64 = 3;

/// The eight horizontal neighbours as (dx, dz), alternating straight and diagonal.
const DIRS: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Sort key that orders samples bottom-up, then by z, then x.
type Key = (i32, i32, i32);

#[inline]
fn key(p: IVec3) -> Key {
    (p.y, p.z, p.x)
}

#[derive(Default)]
pub struct SandSim {
    awake: BTreeSet<Key>,
    steps: u64,
}

impl SandSim {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn awake(&self) -> usize {
        self.awake.len()
    }

    pub fn wake(&mut self, p: IVec3) {
        self.awake.insert(key(p));
    }

    /// Wake every slumping sample in an inclusive box, e.g. around an edit.
    pub fn wake_region(&mut self, volume: &Volume, min: IVec3, max: IVec3) {
        let field = volume.copy_region(min, max - min + 1);
        let mut i = 0;
        for y in 0..field.dims.y {
            for z in 0..field.dims.z {
                for x in 0..field.dims.x {
                    if field.fill[i] > 0 && material::def(field.mat[i]).slumps {
                        self.awake.insert(key(field.origin + IVec3::new(x, y, z)));
                    }
                    i += 1;
                }
            }
        }
    }

    /// The awake samples inside one chunk.
    pub fn awake_in(&self, pos: ChunkPos) -> Vec<IVec3> {
        let o = chunk_origin(pos);
        self.awake
            .range((o.y, i32::MIN, i32::MIN)..(o.y + CHUNK_SIZE, i32::MIN, i32::MIN))
            .map(|&(y, z, x)| IVec3::new(x, y, z))
            .filter(|&p| chunk_of(p) == pos)
            .collect()
    }

    /// Remove and return the awake samples of a chunk that is being unloaded.
    pub fn take_chunk(&mut self, pos: ChunkPos) -> Vec<IVec3> {
        let taken = self.awake_in(pos);
        for &p in &taken {
            self.awake.remove(&key(p));
        }
        taken
    }

    /// One step of settling. Appends chunks whose samples changed to `changed`.
    pub fn step(&mut self, volume: &mut Volume, changed: &mut Vec<ChunkPos>) {
        if self.awake.is_empty() {
            return;
        }
        self.steps += 1;
        let rotation = (self.steps % 8) as usize;
        let previous = std::mem::take(&mut self.awake);

        // Group by chunk; BTreeMap keeps both chunks and samples in a fixed order.
        let mut buckets: BTreeMap<Key, Vec<IVec3>> = BTreeMap::new();
        for (y, z, x) in previous {
            let p = IVec3::new(x, y, z);
            buckets.entry(key(chunk_of(p))).or_default().push(p);
        }

        for ((cy, cz, cx), samples) in buckets {
            let chunk = IVec3::new(cx, cy, cz);
            // Never move sand into or out of space that is not loaded: keep
            // the samples awake and wait.
            let loaded = (-1..=1).all(|y| {
                (-1..=1).all(|z| (-1..=1).all(|x| volume.chunks.contains_key(&(chunk + IVec3::new(x, y, z)))))
            });
            if !loaded {
                self.awake.extend(samples.into_iter().map(key));
                continue;
            }

            let origin = chunk_origin(chunk) - MARGIN;
            let mut field = volume.copy_region(origin, IVec3::splat(CHUNK_SIZE + 2 * MARGIN));
            let [sx, sy, sz] = field.strides();
            let (fill, mat) = (&mut field.fill, &mut field.mat);
            // Sand can enter more sand and anything that is not solid. That
            // includes the thin, partly filled layer of other material just
            // above every surface, which it takes over as it fills; were that
            // layer to hold it up, a pile would rest a sample above the
            // ground with a gap beneath it.
            let open = |fill: &[u8], mat: &[u8], i: usize| fill[i] < SOLID || material::def(mat[i]).slumps;

            for p in samples {
                let l = p - origin;
                let i = ((l.y * field.dims.z + l.z) * field.dims.x + l.x) as usize;
                if fill[i] == 0 || !material::def(mat[i]).slumps {
                    continue;
                }
                let kind = mat[i];
                let mut moved = false;

                // Fall straight down into free capacity.
                let below = i - sy;
                if open(fill, mat, below) && fill[below] < 255 {
                    let m = fill[i].min(255 - fill[below]);
                    fill[below] += m;
                    mat[below] = kind;
                    fill[i] -= m;
                    moved = true;
                    self.awake.insert(key(p - IVec3::Y));
                }

                // Resting on something: spread to lower neighbouring columns.
                if fill[i] > 0 && !(open(fill, mat, below) && fill[below] < 255) {
                    for k in 0..8 {
                        let (dx, dz) = DIRS[(k + rotation) % 8];
                        let n = (i as isize + dx as isize * sx as isize + dz as isize * sz as isize) as usize;
                        if !open(fill, mat, n) || fill[n] == 255 {
                            continue;
                        }
                        // Top of the neighbouring column relative to this
                        // sample's level, in fill units.
                        let neighbour_top = if fill[n] > 0 {
                            fill[n] as i32
                        } else if !open(fill, mat, n - sy) {
                            0
                        } else if fill[n - sy] > 0 {
                            fill[n - sy] as i32 - 255
                        } else if !open(fill, mat, n - 2 * sy) {
                            -255
                        } else {
                            fill[n - 2 * sy] as i32 - 510
                        };
                        let limit = if dx != 0 && dz != 0 { REPOSE_DIAGONAL } else { REPOSE };
                        let excess = fill[i] as i32 - neighbour_top - limit;
                        if excess <= 0 {
                            continue;
                        }
                        let m = ((excess + 1) / 2).min(fill[i] as i32).min(255 - fill[n] as i32) as u8;
                        fill[n] += m;
                        mat[n] = kind;
                        fill[i] -= m;
                        moved = true;
                        self.awake.insert(key(p + IVec3::new(dx, 0, dz)));
                        if fill[i] == 0 {
                            break;
                        }
                    }
                }

                if fill[i] == 0 {
                    mat[i] = 0;
                }
                if moved {
                    // Anything that leaned on this sample may now move.
                    for dy in 0..=2 {
                        for dz in -1..=1 {
                            for dx in -1..=1 {
                                self.awake.insert(key(p + IVec3::new(dx, dy, dz)));
                            }
                        }
                    }
                }
            }
            volume.write_region(&field, changed);
        }
        changed.sort_unstable_by_key(|c| key(*c));
        changed.dedup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Chunk, Sample};
    use crate::material::Material;

    fn total_sand(vol: &Volume) -> u64 {
        vol.chunks
            .values()
            .map(|c| match &**c {
                Chunk::Uniform(s) if s.mat == Material::SAND => s.fill as u64 * crate::CHUNK_VOLUME as u64,
                Chunk::Uniform(_) => 0,
                Chunk::Dense(d) => {
                    d.fill.iter().zip(d.mat.iter()).filter(|&(_, &m)| m == Material::SAND).map(|(&f, _)| f as u64).sum()
                }
            })
            .sum()
    }

    /// Air above a stone floor whose top full layer is y = -1.
    fn floor() -> Volume {
        let mut vol = Volume::new();
        for y in -2..2 {
            for z in -2..2 {
                for x in -2..2 {
                    let s = if y < 0 { Sample { fill: 255, mat: Material::STONE } } else { Sample::AIR };
                    vol.insert(IVec3::new(x, y, z), Chunk::Uniform(s));
                }
            }
        }
        vol
    }

    fn column_height(vol: &Volume, x: i32, z: i32) -> f32 {
        (0..40).map(|y| vol.get(IVec3::new(x, y, z)).unwrap().fill as f32 / 255.0).sum()
    }

    fn settle(vol: &mut Volume, sim: &mut SandSim) -> usize {
        let mut changed = Vec::new();
        for step in 0..5000 {
            if sim.awake() == 0 {
                return step;
            }
            sim.step(vol, &mut changed);
        }
        panic!("sand never settled: {} samples still awake", sim.awake());
    }

    #[test]
    fn a_tower_slumps_into_a_pile_at_the_repose_angle() {
        let mut vol = floor();
        // A 4x4 tower, 16 cells tall, straddling a chunk corner.
        for y in 0..16 {
            for z in -2..2 {
                for x in -2..2 {
                    vol.set(IVec3::new(x, y, z), Sample { fill: 255, mat: Material::SAND });
                }
            }
        }
        let before = total_sand(&vol);
        let mut sim = SandSim::new();
        sim.wake_region(&vol, IVec3::new(-3, -1, -3), IVec3::new(3, 17, 3));
        let steps = settle(&mut vol, &mut sim);
        assert_eq!(total_sand(&vol), before, "sand was created or destroyed");
        assert!(steps > 20, "settled suspiciously fast: {steps} steps");

        let peak = column_height(&vol, 0, 0);
        assert!(peak < 9.0 && peak > 3.0, "peak {peak}");
        // Slope along an axis from the centre: within a few degrees of 34.
        let mut run = 0;
        while column_height(&vol, run + 1, 0) > 0.05 {
            run += 1;
        }
        let angle = (peak / run as f32).atan().to_degrees();
        assert!((27.0..=40.0).contains(&angle), "pile angle {angle} (peak {peak}, run {run})");
        // Nothing left floating: every column is solid from the floor up.
        for x in -12..12 {
            let mut seen_gap = false;
            for y in 0..20 {
                let f = vol.get(IVec3::new(x, y, 0)).unwrap().fill;
                assert!(!(seen_gap && f > 0), "floating sand at x={x} y={y}");
                seen_gap |= f < 255;
            }
        }
    }

    #[test]
    fn settling_is_deterministic_and_freezes_at_unloaded_space() {
        let build = || {
            let mut vol = floor();
            for y in 0..10 {
                for x in 20..24 {
                    vol.set(IVec3::new(x, y, 5), Sample { fill: 200, mat: Material::SAND });
                }
            }
            vol
        };
        let run = || {
            let mut vol = build();
            let mut sim = SandSim::new();
            sim.wake_region(&vol, IVec3::new(18, -1, 3), IVec3::new(26, 11, 7));
            settle(&mut vol, &mut sim);
            let mut keys: Vec<_> = vol.chunks.keys().copied().collect();
            keys.sort_by_key(|k| (k.x, k.y, k.z));
            keys.iter().fold(0, |h, k| vol.chunks[k].content_hash(h))
        };
        assert_eq!(run(), run());

        // Sand in a chunk at the edge of the loaded world stays put and awake.
        let mut vol = floor();
        vol.set(IVec3::new(60, 20, 0), Sample { fill: 255, mat: Material::SAND });
        let mut sim = SandSim::new();
        sim.wake(IVec3::new(60, 20, 0));
        let mut changed = Vec::new();
        for _ in 0..10 {
            sim.step(&mut vol, &mut changed);
        }
        assert!(changed.is_empty() && sim.awake() == 1);
        assert_eq!(sim.take_chunk(IVec3::new(1, 0, 0)), vec![IVec3::new(60, 20, 0)]);
        assert_eq!(sim.awake(), 0);
    }

    #[test]
    fn a_pile_sits_on_sloping_ground_with_no_gap_beneath() {
        // Dirt whose surface climbs one cell in four along x, as generated
        // terrain does: partly filled samples lie just above the surface.
        let mut vol = Volume::new();
        for cy in -2..2 {
            for cz in -2..2 {
                for cx in -2..2 {
                    let mut d = crate::chunk::Dense::new_boxed(Sample::AIR);
                    for i in 0..crate::CHUNK_VOLUME {
                        let (x, y) = ((i & 31) as i32 + cx * 32, (i >> 10) as i32 + cy * 32);
                        let fill = (127.5 + (x as f32 * 0.25 - y as f32) * 64.0).clamp(0.0, 255.0) as u8;
                        d.fill[i] = fill;
                        d.mat[i] = if fill > 0 { Material::DIRT } else { 0 };
                    }
                    vol.insert(IVec3::new(cx, cy, cz), Chunk::Dense(d));
                }
            }
        }
        // Drop a block of sand onto it from above.
        for y in 8..14 {
            for z in -3..3 {
                for x in -3..3 {
                    vol.set(IVec3::new(x, y, z), Sample { fill: 255, mat: Material::SAND });
                }
            }
        }
        let mut sim = SandSim::new();
        sim.wake_region(&vol, IVec3::new(-4, 7, -4), IVec3::new(4, 15, 4));
        settle(&mut vol, &mut sim);

        // Every column is solid from the ground up to its top: nowhere does
        // air lie between the sand and what it rests on.
        let mut sand_columns = 0;
        for z in -14..14 {
            for x in -14..14 {
                let column: Vec<Sample> = (-10..20).map(|y| vol.get(IVec3::new(x, y, z)).unwrap()).collect();
                let top = column.iter().rposition(|s| s.solid()).unwrap();
                assert!(column[..top].iter().all(|s| s.solid()), "gap under the pile at x {x} z {z}");
                sand_columns += (column[top].mat == Material::SAND) as usize;
            }
        }
        assert!(sand_columns > 60, "only {sand_columns} columns ended up topped with sand");
    }
}
