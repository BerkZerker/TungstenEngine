//! Generation statistics and timing: `cargo run --release -p tungsten-core --example genstats`

use std::time::Instant;

use glam::IVec3;
use tungsten_core::mesh::{PaddedBlock, mesh_block};
use tungsten_core::worldgen::Generator;
use tungsten_core::{Chunk, Volume, WorldParams};

fn main() {
    let g = Generator::new(WorldParams::default());
    for (label, cy) in [("deep (y=-40)", -40), ("shallow (y=-3)", -3), ("sky (y=30)", 30)] {
        let (mut uniform, mut air, mut total) = (0, 0u64, 0u64);
        let t = Instant::now();
        let n = 400;
        for i in 0..n {
            let c = g.generate(IVec3::new(i % 20 * 3, cy, i / 20 * 3));
            match &c {
                Chunk::Uniform(_) => uniform += 1,
                Chunk::Dense(d) => {
                    air += d.fill.iter().filter(|&&f| f < 128).count() as u64;
                    total += d.fill.len() as u64;
                }
            }
        }
        println!(
            "{label:16} {:6.1} us/chunk  uniform {uniform}/{n}  air in dense {:.1}%",
            t.elapsed().as_secs_f64() * 1e6 / n as f64,
            100.0 * air as f64 / total.max(1) as f64
        );
    }

    // Surface heights around the origin, and a meshing pass over that area.
    let mut vol = Volume::new();
    let t = Instant::now();
    let r = 6;
    for cy in -6..=6 {
        for cz in -r..=r {
            for cx in -r..=r {
                let p = IVec3::new(cx, cy, cz);
                vol.insert(p, g.generate(p));
            }
        }
    }
    let count = vol.chunks.len();
    let dense = vol.chunks.values().filter(|c| !c.is_uniform()).count();
    println!(
        "area: {count} chunks ({dense} dense) in {:.1} ms, {:.1} us/chunk",
        t.elapsed().as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64() * 1e6 / count as f64
    );
    let t = Instant::now();
    let (mut tris, mut meshed, mut nonempty) = (0, 0, 0);
    for cy in -5..=5 {
        for cz in -r + 1..r {
            for cx in -r + 1..r {
                let p = IVec3::new(cx, cy, cz);
                let block = PaddedBlock::gather(p, |q| vol.chunk(q)).unwrap();
                let m = mesh_block(&block);
                meshed += 1;
                if !m.is_empty() {
                    nonempty += 1;
                    tris += m.indices.len() / 3;
                }
            }
        }
    }
    println!(
        "mesh: {meshed} chunks ({nonempty} non-empty, {tris} tris) in {:.1} ms, {:.1} us/chunk",
        t.elapsed().as_secs_f64() * 1e3,
        t.elapsed().as_secs_f64() * 1e6 / meshed as f64
    );
    let h: Vec<String> = (-3..=3).map(|i| format!("{:.0}", g.surface_height(i as f64 * 200.0, 0.0))).collect();
    println!("heights along x every 200 m: {}", h.join(" "));
}
