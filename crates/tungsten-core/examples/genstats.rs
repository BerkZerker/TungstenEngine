//! Prints what generating and meshing costs at each level of detail:
//! `cargo run --release -p tungsten-core --example genstats`

use std::time::Instant;

use glam::IVec3;
use tungsten_core::field::LocalField;
use tungsten_core::mesh::mesh_block;
use tungsten_core::worldgen::{Generated, Generator};
use tungsten_core::{CHUNK_SIZE, MAX_LEVEL, NodeKey, WorldParams};

fn main() {
    let params = WorldParams::default();
    let generator = Generator::new(params);
    println!("level  spacing  nodes  mixed  generate ms  mesh ms  triangles (per node with a surface)");
    for level in 0..=MAX_LEVEL {
        // A slab of nodes around the surface, 400 m across at every level.
        let size = params.spacing(level) * CHUNK_SIZE as f32;
        let r = ((200.0 / size).ceil() as i32).clamp(1, 12);
        let (mut nodes, mut mixed, mut triangles) = (0u32, 0u32, 0usize);
        let (mut generate, mut mesh) = (0.0f64, 0.0f64);
        for z in -r..r {
            for x in -r..r {
                let ground = generator.surface_height((x as f32 * size) as f64, (z as f32 * size) as f64);
                let cy = (ground / size).floor() as i32;
                for y in cy - 1..=cy + 1 {
                    let key = NodeKey::new(level, IVec3::new(x, y, z));
                    let mut block = LocalField::block(key);
                    let t = Instant::now();
                    let kind = generator.fill_block(key, &mut block);
                    generate += t.elapsed().as_secs_f64();
                    nodes += 1;
                    if kind == Generated::Mixed {
                        let t = Instant::now();
                        let m = mesh_block(&block, (level > 0) as i32);
                        mesh += t.elapsed().as_secs_f64();
                        if !m.is_empty() {
                            mixed += 1;
                            triangles += m.indices.len() / 3;
                        }
                    }
                }
            }
        }
        println!(
            "{level:>5}  {:>6.2}m  {nodes:>5}  {mixed:>5}  {:>11.3}  {:>7.3}  {:>9}",
            params.spacing(level),
            generate * 1000.0 / nodes as f64,
            mesh * 1000.0 / mixed.max(1) as f64,
            triangles / mixed.max(1) as usize
        );
    }
}
