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
use tungsten_core::worldgen::{GENERATOR_VERSION, Generator};
use tungsten_core::{MAX_LEVEL, NodeKey, Volume, WorldParams};

const GENERATED: u64 = 0xf80a0bb91e1930b0;
const EDITED: u64 = 0x37c3dd5efb1f4d93;
const RESOLVED: u64 = 0x88b617e9558fa8d1;
const SAVED: u64 = 0x6fabf7af924c70be;
const BAKED: u64 = 0xff72da7184c9c3d7;

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
