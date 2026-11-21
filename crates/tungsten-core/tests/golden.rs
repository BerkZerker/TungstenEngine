//! Golden hashes for the bit-exact contract.
//!
//! These values must be identical on every platform and at any thread count.
//! If a change is *meant* to alter generation, bump `GENERATOR_VERSION` and
//! update the constants from the failure message; if it is not, the change
//! has broken determinism or compatibility with existing saves.

use glam::IVec3;
use tungsten_core::edit::{EditKind, EditOp, EditRules, FP, apply};
use tungsten_core::sand::SandSim;
use tungsten_core::worldgen::{GENERATOR_VERSION, Generator};
use tungsten_core::{Volume, WorldParams, codec};

const GENERATED: u64 = 0x830afe7567390059;
const EDITED: u64 = 0xcbb62a87bd7c9187;
const SETTLED: u64 = 0x48bcd97b878f0de1;
const ENCODED: u64 = 0x65d1d2718ed97d13;

fn hash_volume(vol: &Volume) -> u64 {
    let mut keys: Vec<IVec3> = vol.chunks.keys().copied().collect();
    keys.sort_by_key(|k| (k.x, k.y, k.z));
    keys.iter().fold(0, |h, k| vol.chunks[k].content_hash(h))
}

/// A slab of world around the origin and one far from it, for two seeds.
fn generate(threads: usize) -> Vec<(WorldParams, Volume)> {
    [WorldParams::default(), WorldParams { seed: 12345, cell_size: 0.25, min_fragment: 0.5 }]
        .into_iter()
        .map(|params| {
            let generator = Generator::new(params);
            let mut positions = Vec::new();
            for base in [IVec3::ZERO, IVec3::new(70_000, -3, -41_000)] {
                for y in -3..=2 {
                    for z in -2..=2 {
                        for x in -2..=2 {
                            positions.push(base + IVec3::new(x, y, z));
                        }
                    }
                }
            }
            // Generation order and thread count must not matter.
            let chunks = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..threads)
                    .map(|t| {
                        let (generator, positions) = (&generator, &positions);
                        scope.spawn(move || {
                            positions
                                .iter()
                                .enumerate()
                                .filter(|(i, _)| i % threads == t)
                                .map(|(_, &p)| (p, generator.generate(p)))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect::<Vec<_>>()
            });
            let mut vol = Volume::new();
            for (p, c) in chunks {
                vol.insert(p, c);
            }
            (params, vol)
        })
        .collect()
}

#[test]
fn generation_edits_sand_and_saves_match_golden_hashes() {
    let mut worlds = generate(1);
    let generated = worlds.iter().fold(0u64, |h, (_, v)| h.rotate_left(17) ^ hash_volume(v));
    let threaded = generate(7).iter().fold(0u64, |h, (_, v)| h.rotate_left(17) ^ hash_volume(v));
    assert_eq!(generated, threaded, "generation depends on thread count or order");

    // A scripted session near the surface: dig, then place each material.
    let mut edited = 0u64;
    let mut settled = 0u64;
    let mut encoded = 0u64;
    for (params, vol) in &mut worlds {
        let rules = EditRules::new(params);
        let generator = Generator::new(*params);
        let ground = (generator.surface_height(0.0, 0.0) / params.cell_size) as i64;
        let mut sand = SandSim::new();
        for tick in 0..200u64 {
            let t = tick as i64;
            let place = tick % 50 == 49;
            let op = EditOp {
                kind: if place { EditKind::Place } else { EditKind::Break },
                center: [t * 9 - 600, (ground + 2) * FP - t * 5, 400 - t * 7],
                radius: 3 * FP + (t % 4) * 70,
                // Sand, dirt, stone, then sand again so the session ends with a pile settling.
                material: 3 - (tick / 50 % 3) as u8,
                tick,
                exclude: None,
            };
            let out = apply(vol, &rules, &op);
            sand.wake_region(vol, out.min - 1, out.max + 1);
            if tick % 3 == 0 {
                sand.step(vol, &mut Vec::new());
            }
        }
        edited = edited.rotate_left(17) ^ hash_volume(vol);
        let mut steps = 0;
        while sand.awake() > 0 && steps < 3000 {
            sand.step(vol, &mut Vec::new());
            steps += 1;
        }
        assert!(steps > 5, "the script should leave sand still settling");
        assert_eq!(sand.awake(), 0, "sand did not settle");
        settled = settled.rotate_left(17) ^ hash_volume(vol);

        let mut touched: Vec<IVec3> = vol.touched.keys().copied().collect();
        touched.sort_by_key(|k| (k.x, k.y, k.z));
        assert!(!touched.is_empty());
        for pos in touched {
            let bytes = codec::encode(&vol.chunks[&pos], &vol.touched[&pos], &[]);
            encoded = tungsten_core::hash::fnv1a(encoded, &bytes);
        }
    }

    let actual = [generated, edited, settled, encoded];
    let golden = [GENERATED, EDITED, SETTLED, ENCODED];
    assert_eq!(
        actual, golden,
        "\ngolden hashes changed (generator version {GENERATOR_VERSION}):\n\
         const GENERATED: u64 = {:#018x};\nconst EDITED: u64 = {:#018x};\n\
         const SETTLED: u64 = {:#018x};\nconst ENCODED: u64 = {:#018x};\n",
        actual[0], actual[1], actual[2], actual[3]
    );
}
