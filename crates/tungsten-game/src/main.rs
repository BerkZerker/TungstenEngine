//! Tungsten: an infinite, editable, procedurally generated world.

mod app;
mod bench;
mod capture;
mod game;
mod hud;
mod input;
mod player;

use std::path::PathBuf;

use glam::DVec3;
use tungsten_core::WorldParams;
use tungsten_stream::StreamConfig;

pub struct Args {
    pub params: WorldParams,
    pub stream: StreamConfig,
    /// Render one frame to this BMP file without opening a window.
    pub capture: Option<PathBuf>,
    /// Start position in metres (feet), instead of the spawn point.
    pub pos: Option<DVec3>,
    /// Start yaw and pitch in degrees.
    pub look: Option<(f32, f32)>,
    pub size: (u32, u32),
    pub fly: bool,
    /// World directory to load and save, or `None` to keep nothing.
    pub world: Option<PathBuf>,
    /// The seed or cell size was given on the command line.
    pub explicit_params: bool,
    /// Quit after this many seconds (for smoke tests).
    pub exit_after: Option<f64>,
    pub no_vsync: bool,
    /// Run the streaming benchmark for this many seconds instead of playing.
    pub bench: Option<f64>,
    /// Scripted input to play before a capture; see `capture::run_script`.
    pub script: Option<String>,
}

const USAGE: &str = "\
tungsten [options]
  --world NAME        world to load or create under ./worlds (default: default)
  --no-save           do not load or save a world
  --seed N            seed for a new world (default: built in)
  --cell METRES       finest lattice spacing for a new world (default 0.25)
  --fragment METRES   for a new world: breaking removes isolated pieces smaller than a
                      cube this size (default 1, 0 keeps everything)
  --view METRES       how far terrain is drawn (default 2048)
  --detail N          how far out fine detail reaches; a node is drawn finer while
                      the player is within N of its own lengths (default 2)
  --workers N         worker threads (default: cores - 2)
  --pos X,Y,Z         start position in metres
  --look YAW,PITCH    start view direction in degrees
  --fly               start in fly mode
  --size WxH          window or capture size in pixels
  --exit-after SECS   quit by itself after this long, printing frame statistics
  --no-vsync          do not wait for the display between frames
  --bench SECS        headless: fly at speed for this long and report streaming throughput
  --capture FILE.bmp  render one frame headless and exit
  --script CMDS       with --capture: play input first, e.g. 'look 0,-40; hold lmb; tick 90'
                      commands: tick N | hold K | release K | click | key K | look YAW,PITCH | scroll N
                      keys: w a s d space shift ctrl lmb 1 2 3 4";

fn parse_list<T: std::str::FromStr>(s: &str, sep: char, n: usize) -> Result<Vec<T>, String> {
    let v: Vec<T> = s.split(sep).filter_map(|p| p.trim().parse().ok()).collect();
    if v.len() == n { Ok(v) } else { Err(format!("expected {n} values separated by '{sep}', got '{s}'")) }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        params: WorldParams::default(),
        stream: StreamConfig::default(),
        capture: None,
        pos: None,
        look: None,
        size: (1600, 900),
        fly: false,
        world: None,
        explicit_params: false,
        exit_after: None,
        no_vsync: false,
        bench: None,
        script: None,
    };
    let (mut world_name, mut no_save) = (None, false);
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--seed" => {
                a.params.seed = value()?.parse().map_err(|e| format!("--seed: {e}"))?;
                a.explicit_params = true;
            }
            "--world" => {
                let name = value()?;
                if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
                    return Err(format!("--world: '{name}' is not a valid world name"));
                }
                world_name = Some(name);
            }
            "--no-save" => no_save = true,
            "--no-vsync" => a.no_vsync = true,
            "--bench" => {
                a.bench = Some(value()?.parse().map_err(|e| format!("--bench: {e}"))?);
                no_save = true;
            }
            "--exit-after" => a.exit_after = Some(value()?.parse().map_err(|e| format!("--exit-after: {e}"))?),
            "--cell" => {
                a.explicit_params = true;
                a.params.cell_size = value()?.parse().map_err(|e| format!("--cell: {e}"))?;
                if !(0.1..=4.0).contains(&a.params.cell_size) {
                    return Err("--cell must be between 0.1 and 4 metres".into());
                }
            }
            "--fragment" => {
                a.explicit_params = true;
                a.params.min_fragment = value()?.parse().map_err(|e| format!("--fragment: {e}"))?;
                if !(0.0..=4.0).contains(&a.params.min_fragment) {
                    return Err("--fragment must be between 0 and 4 metres".into());
                }
            }
            "--view" => {
                a.stream.view_distance = value()?.parse().map_err(|e| format!("--view: {e}"))?;
                if !(32.0..=16384.0).contains(&a.stream.view_distance) {
                    return Err("--view must be between 32 and 16384 metres".into());
                }
            }
            "--detail" => {
                a.stream.detail = value()?.parse().map_err(|e| format!("--detail: {e}"))?;
                if !(1.0..=8.0).contains(&a.stream.detail) {
                    return Err("--detail must be between 1 and 8".into());
                }
            }
            "--workers" => a.stream.workers = value()?.parse().map_err(|e| format!("--workers: {e}"))?,
            "--pos" => {
                let v = parse_list::<f64>(&value()?, ',', 3)?;
                a.pos = Some(DVec3::new(v[0], v[1], v[2]));
            }
            "--look" => {
                let v = parse_list::<f32>(&value()?, ',', 2)?;
                a.look = Some((v[0], v[1]));
            }
            "--size" => {
                let v = parse_list::<u32>(&value()?, 'x', 2)?;
                a.size = (v[0].max(16), v[1].max(16));
            }
            "--fly" => a.fly = true,
            "--script" => a.script = Some(value()?),
            "--capture" => a.capture = Some(PathBuf::from(value()?)),
            "--help" | "-h" => return Err(USAGE.into()),
            other => return Err(format!("unknown option '{other}'\n{USAGE}")),
        }
    }
    // A capture leaves no trace unless a world is named; playing saves by default.
    if !no_save && (world_name.is_some() || a.capture.is_none()) {
        a.world = Some(PathBuf::from("worlds").join(world_name.as_deref().unwrap_or("default")));
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let result = if args.bench.is_some() {
        bench::run(args)
    } else if args.capture.is_some() {
        capture::run(args)
    } else {
        app::run(args)
    };
    if let Err(e) = result {
        eprintln!("tungsten: {e}");
        std::process::exit(1);
    }
}
