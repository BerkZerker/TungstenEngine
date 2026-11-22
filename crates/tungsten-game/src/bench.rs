//! Headless streaming benchmark: load a world cold, then fly through it at
//! speed in real time, with no window or GPU.

use std::time::{Duration, Instant};

use tungsten_stream::MeshEvent;
use winit::keyboard::KeyCode;

use crate::Args;
use crate::game::Game;
use crate::input::Input;
use crate::player::{Mode, TICK};

/// Drain mesh events, tallying uploads and their triangles.
fn count(events: &mut Vec<MeshEvent>, uploads: &mut usize, triangles: &mut usize) {
    for e in events.drain(..) {
        if let MeshEvent::Upload { mesh, .. } = e {
            *uploads += 1;
            *triangles += mesh.indices.len() / 3;
        }
    }
}

pub fn run(args: Args) -> Result<(), String> {
    let seconds = args.bench.unwrap();
    let mut game = Game::open(&args)?;
    game.player.mode = Mode::Fly;
    let config = game.world.config();
    println!(
        "bench: view {} m, detail {}, {} workers, cell {} m",
        config.view_distance,
        config.detail,
        config.workers,
        game.params().cell_size
    );

    // Cold load of everything in range.
    let start = Instant::now();
    let (mut uploads, mut triangles) = (0usize, 0usize);
    while !game.world.settled() {
        game.stream();
        count(&mut game.events, &mut uploads, &mut triangles);
        std::thread::sleep(Duration::from_micros(200));
    }
    let cold = start.elapsed().as_secs_f64();
    let stats = game.world.stats();
    println!(
        "cold load: {} nodes, {} meshes, {:.2}M triangles in {:.3} s = {:.0} nodes/s; drawn by level {:?}",
        stats.nodes,
        uploads,
        triangles as f64 / 1e6,
        cold,
        stats.nodes as f64 / cold,
        stats.drawn
    );

    // Fly forward at fly-run speed, paced to real time like the game loop.
    let mut input = Input::default();
    input.key(KeyCode::KeyW, true);
    input.key(KeyCode::ShiftLeft, true);
    let from = game.player.pos;
    let ticks = (seconds / TICK) as u64;
    let (mut worst, mut total) = (Duration::ZERO, Duration::ZERO);
    let (mut waiting, mut backlog_max, mut backlog_sum) = (0u64, 0usize, 0usize);
    uploads = 0;
    let start = Instant::now();
    for tick in 0..ticks {
        let t = Instant::now();
        game.step(&mut input);
        game.stream();
        let spent = t.elapsed();
        worst = worst.max(spent);
        total += spent;
        count(&mut game.events, &mut uploads, &mut triangles);
        let s = game.world.stats();
        let backlog = s.jobs_queued + s.jobs_running;
        backlog_max = backlog_max.max(backlog);
        backlog_sum += backlog;
        waiting += game.waiting_for_terrain() as u64;
        let due = Duration::from_secs_f64((tick + 1) as f64 * TICK);
        if let Some(rest) = due.checked_sub(start.elapsed()) {
            std::thread::sleep(rest);
        }
    }
    let flown = (game.player.pos - from).length();
    println!(
        "fly-through: {:.0} m in {:.1} s ({:.0} m/s), {} meshes uploaded",
        flown,
        start.elapsed().as_secs_f64(),
        flown / start.elapsed().as_secs_f64(),
        uploads
    );
    println!(
        "main thread per tick: {:.3} ms average, {:.3} ms worst (budget 2 ms)",
        total.as_secs_f64() * 1000.0 / ticks.max(1) as f64,
        worst.as_secs_f64() * 1000.0
    );
    println!(
        "backlog (node jobs queued or running): {:.0} average, {} worst; ticks spent waiting for terrain: {}",
        backlog_sum as f64 / ticks.max(1) as f64,
        backlog_max,
        waiting
    );
    Ok(())
}
