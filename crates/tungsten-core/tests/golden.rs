//! Golden hashes for the bit-exact contract.
//!
//! These values must be identical on every platform and at any thread count.
//! If a change is *meant* to alter generation or the edit rules, bump
//! `GENERATOR_VERSION` and update the constants from the failure message; if
//! it is not, the change has broken determinism.

use std::sync::Arc;

use glam::IVec3;
use tungsten_core::coords::{chunk_of, chunk_origin};
use tungsten_core::edit::{EditRules, FP, Op, edit_volume};
use tungsten_core::field::LocalField;
use tungsten_core::hash::fnv1a;
use tungsten_core::log::{CellPos, EditLog, bake, resolve};
use tungsten_core::water::bodies::Water;
use tungsten_core::worldgen::{GENERATOR_VERSION, Generator};
use tungsten_core::{MAX_LEVEL, NodeKey, Volume, WorldParams};

const GENERATED: u64 = 0xf80a0bb91e1930b0;
const EDITED: u64 = 0x05c7725beb7c0b7f;
const RESOLVED: u64 = 0x9126b43589a2934f;
const SAVED: u64 = 0x419dcb1ebe6f224e;
const BAKED: u64 = 0xe240962c290a1032;
const WATER: u64 = 0x8cf5fe850ad536ce;

fn worlds() -> [WorldParams; 2] {
    [WorldParams::default(), WorldParams { seed: 12345, cell_size: 0.5, min_fragment: 0.75 }]
}

/// Nodes of every level around a point, near the surface.
fn nodes(generator: &Generator, x: i32, z: i32) -> Vec<NodeKey> {
    let cell = generator.params().cell_size as f64;
    let ground = (generator.surface_height(x as f64 * cell, z as f64 * cell) as f64 / cell) as i32;
    let mut out = Vec::new();
    for level in 0..=MAX_LEVEL {
        let c = chunk_of(IVec3::new(x, ground, z) >> level as i32);
        for d in [IVec3::ZERO, IVec3::NEG_X, IVec3::NEG_Y, IVec3::new(0, 1, -1), IVec3::new(1, -1, 1)] {
            out.push(NodeKey::new(level, c + d));
        }
    }
    out
}

fn block(generator: &Generator, key: NodeKey) -> LocalField {
    let mut field = LocalField::block(key);
    generator.fill_block(key, &mut field);
    field
}

fn hash_field(h: u64, field: &LocalField) -> u64 {
    fnv1a(fnv1a(h, &field.fill), &field.mat)
}

/// Blocks near the origin and far from it, generated on `threads` threads.
fn generate(params: WorldParams, threads: usize) -> u64 {
    let generator = Generator::new(params);
    let mut keys = nodes(&generator, 0, 0);
    keys.extend(nodes(&generator, 2_300_000, -1_300_000));
    // Generation order and thread count must not matter.
    let hashes = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (generator, keys) = (&generator, &keys);
                scope.spawn(move || {
                    keys.iter()
                        .enumerate()
                        .filter(|(i, _)| i % threads == t)
                        .map(|(i, &k)| (i, hash_field(0, &block(generator, k))))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut all: Vec<(usize, u64)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
        all.sort_unstable();
        all
    });
    hashes.iter().fold(0, |h, (_, x)| h.rotate_left(17) ^ x)
}

#[test]
fn generation_edits_and_saves_match_golden_hashes() {
    let generated = worlds().iter().fold(0u64, |h, &p| h.rotate_left(29) ^ generate(p, 1));
    let threaded = worlds().iter().fold(0u64, |h, &p| h.rotate_left(29) ^ generate(p, 7));
    assert_eq!(generated, threaded, "generation depends on thread count or order");

    let (mut edited, mut resolved, mut saved, mut baked) = (0u64, 0u64, 0u64, 0u64);
    for params in worlds() {
        let generator = Generator::new(params);
        let rules = EditRules::new(&params);
        let keys = nodes(&generator, 0, 0);
        let ground = (generator.surface_height(0.0, 0.0) / params.cell_size) as i64;

        // The finest level around the origin, as the game holds it.
        let mut volume = Volume::new();
        let center = keys[0].pos;
        for y in -2..=2 {
            for z in -2..=2 {
                for x in -2..=2 {
                    let pos = center + IVec3::new(x, y, z);
                    volume.insert(pos, block(&generator, NodeKey::new(0, pos)).interior());
                }
            }
        }

        // A scripted session near the surface: dig, break stone, place each material.
        let mut log = EditLog::new();
        for k in 0..120i64 {
            let c = [k * 37 - 2200, (ground + 3) * FP - k * 21, 1500 - k * 29];
            let op = match k % 6 {
                0..=2 => Op::Break { center: c, radius: 4 * FP + (k % 5) * 90, nearest: false },
                3 => Op::Break { center: c, radius: 3 * FP, nearest: true },
                4 if k % 4 == 0 => Op::Chip { center: c },
                _ => Op::Place { center: c, radius: 5 * FP, material: (k % 3) as u8 + 1, exclude: None },
            };
            let outcome = edit_volume(&mut volume, &rules, &op);
            assert!(outcome.covered, "the session left the loaded volume");
            if !outcome.chunks.is_empty() {
                log.push(&rules, &op);
                if let Some(fragments) = &outcome.fragments {
                    log.push(&rules, fragments);
                }
            }
        }
        assert!(log.op_count() > 100);
        let mut positions: Vec<IVec3> = volume.chunks.keys().copied().collect();
        positions.sort_by_key(|k| (k.x, k.y, k.z));
        edited = positions.iter().fold(edited, |h, k| volume.chunks[k].content_hash(h));

        // The log gives back the same finest level, and every other level.
        for &pos in &positions {
            let key = NodeKey::new(0, pos);
            let mut field = block(&generator, key);
            resolve(&mut field, &rules, &log.gather_node(key));
            assert!(field.interior() == *volume.chunks[&pos], "chunk {pos} differs from its log");
        }
        let resolve_all = |log: &EditLog| {
            keys.iter().fold(0u64, |h, &key| {
                let mut field = block(&generator, key);
                resolve(&mut field, &rules, &log.gather_node(key));
                hash_field(h, &field)
            })
        };
        let before = resolve_all(&log);
        resolved = resolved.rotate_left(17) ^ before;
        assert_ne!(before, keys.iter().fold(0u64, |h, &key| hash_field(h, &block(&generator, key))));

        let mut cells: Vec<CellPos> = log.cells().map(|(p, _)| *p).collect();
        cells.sort_by_key(|p| (p.tier, p.pos.x, p.pos.y, p.pos.z));
        for pos in &cells {
            saved = fnv1a(saved, &log.cell(*pos).unwrap().encode());
        }

        // Baked, the world is the same and so are the bytes that say so.
        for pos in &cells {
            let edits = log.cell(*pos).unwrap().clone();
            log.rebase(*pos, edits.ops.len(), Arc::new(bake(&generator, &rules, *pos, &edits)));
            baked = fnv1a(baked, &log.cell(*pos).unwrap().encode());
        }
        assert_eq!(log.op_count(), 0);
        assert_eq!(resolve_all(&log), before, "baking changed the world");
        // Chunk origins are where the blocks say they are.
        assert_eq!(chunk_origin(center), block(&generator, keys[0]).origin + 3);
    }

    let actual = [generated, edited, resolved, saved, baked];
    let golden = [GENERATED, EDITED, RESOLVED, SAVED, BAKED];
    assert_eq!(
        actual, golden,
        "\ngolden hashes changed (generator version {GENERATOR_VERSION}):\n\
         const GENERATED: u64 = {:#018x};\nconst EDITED: u64 = {:#018x};\n\
         const RESOLVED: u64 = {:#018x};\nconst SAVED: u64 = {:#018x};\nconst BAKED: u64 = {:#018x};\n",
        actual[0], actual[1], actual[2], actual[3], actual[4]
    );
}

/// Ponds poured, dug into, filled in and overflowed near the origin, and the
/// sea let in through a channel, all on the finest level as the game holds it.
#[test]
fn water_matches_golden_hash() {
    let mut hash = 0u64;
    for params in worlds() {
        let generator = Generator::new(params);
        let rules = EditRules::new(&params);
        let cell = params.cell_size as f64;
        // The first column east of the origin that is under the sea.
        let shore = (0..4000).find(|&x| generator.surface_height(x as f64 * cell, 0.0) < 0.0).unwrap();
        let mut held = 0;
        for x in [0, shore - 10] {
            let ground = (generator.surface_height(x as f64 * cell, 0.0) as f64 / cell).floor() as i32;
            let here = IVec3::new(x, ground, 0);
            let mut volume = Volume::new();
            for i in 0..125 {
                let pos = chunk_of(here) + IVec3::new(i % 5, i / 25, i / 5 % 5) - 2;
                volume.insert(pos, block(&generator, NodeKey::new(0, pos)).interior());
            }
            let mut water = Water::new();
            let edit = |water: &mut Water, volume: &mut Volume, at: IVec3, radius: i64, place: bool| {
                let center = [at.x as i64 * FP, at.y as i64 * FP, at.z as i64 * FP];
                let op = match place {
                    true => Op::Place { center, radius: radius * FP, material: 3, exclude: None },
                    false => Op::Break { center, radius: radius * FP, nearest: false },
                };
                if let Some((min, max)) = edit_volume(volume, &rules, &op).changed {
                    water.edited(volume, &generator, min, max);
                }
            };
            let pit = here - IVec3::Y * 4;
            for step in 0..240 {
                match step {
                    0 => edit(&mut water, &mut volume, pit, 7, false),
                    5 => water.pour(here + IVec3::Y * 5, 250 * FP),
                    40 => edit(&mut water, &mut volume, pit + IVec3::new(8, -1, 2), 6, false),
                    70 => edit(&mut water, &mut volume, pit + IVec3::new(-2, -5, 0), 4, true),
                    100 => water.pour(here + IVec3::new(3, 6, -2), 3000 * FP),
                    // Towards the sea, where there is one within reach.
                    130..=150 if step % 4 == 0 => {
                        edit(&mut water, &mut volume, pit + IVec3::new(2 * (step - 126), 0, 0), 5, false)
                    }
                    _ => {}
                }
                water.step(&volume, &generator);
            }
            let stats = water.stats();
            assert_eq!(stats.held + stats.moving + stats.drained + stats.lost, 3250 * FP, "water was not conserved");
            held += stats.ponds + stats.sea_cells;
            let bytes = water.encode();
            assert!(Water::decode(&bytes).unwrap().encode() == bytes);
            hash = fnv1a(hash, &bytes);
        }
        assert!(held > 0, "the session left no water behind");
    }
    assert_eq!(
        hash, WATER,
        "\nwater changed (generator version {GENERATOR_VERSION}):\nconst WATER: u64 = {hash:#018x};\n"
    );
}
