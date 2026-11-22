//! On-disk storage of a world: a header, and region files of edit cells.
//!
//! A region file holds the saved edit cells of a 16x16x16 block of them and
//! is always replaced whole by writing a temporary file and renaming it, so a
//! crash can never leave one half written.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use glam::IVec3;
use tungsten_core::hash::{IntMap, fnv1a};
use tungsten_core::log::CellPos;
use tungsten_core::worldgen::GENERATOR_VERSION;
use tungsten_core::{CHUNK_SIZE, WorldParams};

/// Edit cells along the edge of a region.
pub const REGION_BITS: i32 = 4;
const REGION_MAGIC: &[u8; 4] = b"TNGR";
const REGION_VERSION: u32 = 2;
const WORLD_FORMAT: u32 = 2;

#[derive(Default)]
struct Region {
    chunks: HashMap<u16, Arc<[u8]>>,
    dirty: bool,
}

/// A region: a tier of edit cells and a position in units of `1 << REGION_BITS` of them.
pub type RegionPos = (u8, IVec3);

pub struct Store {
    root: PathBuf,
    regions: Mutex<IntMap<RegionPos, Region>>,
    temp_counter: AtomicU64,
}

/// The region an edit cell is saved in, and its index there.
pub fn region_of(cell: CellPos) -> (RegionPos, u16) {
    let r = cell.pos >> REGION_BITS;
    let l = cell.pos - (r << REGION_BITS);
    ((cell.tier, r), ((l.y << (2 * REGION_BITS)) | (l.z << REGION_BITS) | l.x) as u16)
}

/// Replace `path` with `bytes` atomically.
fn write_atomic(path: &Path, temp_id: u64, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension(format!("tmp{temp_id}"));
    let mut f = fs::File::create(&tmp)?;
    f.write_all(bytes)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)
}

fn parse_region(bytes: &[u8]) -> Result<HashMap<u16, Arc<[u8]>>, String> {
    let u32_at = |at: usize| -> Result<u32, String> {
        Ok(u32::from_le_bytes(bytes.get(at..at + 4).ok_or("truncated")?.try_into().unwrap()))
    };
    if bytes.get(..4) != Some(REGION_MAGIC) {
        return Err("not a region file".into());
    }
    if u32_at(4)? != REGION_VERSION {
        return Err(format!("unsupported region version {}", u32_at(4)?));
    }
    let count = u32_at(8)? as usize;
    let mut table = 12;
    let mut data = 12 + count.checked_mul(10).ok_or("bad count")?;
    let mut chunks = HashMap::with_capacity(count);
    for _ in 0..count {
        let index = u16::from_le_bytes(bytes.get(table..table + 2).ok_or("truncated")?.try_into().unwrap());
        let (len, sum) = (u32_at(table + 2)? as usize, u32_at(table + 6)?);
        table += 10;
        let blob = bytes.get(data..data + len).ok_or("truncated")?;
        data += len;
        // A damaged chunk falls back to its generated state; the rest survive.
        if fnv1a(0, blob) as u32 == sum {
            chunks.insert(index, Arc::from(blob));
        } else {
            eprintln!("tungsten: discarding corrupt saved chunk {index} in a region file");
        }
    }
    Ok(chunks)
}

fn serialize_region(region: &Region) -> Vec<u8> {
    let mut keys: Vec<u16> = region.chunks.keys().copied().collect();
    keys.sort_unstable();
    let mut out = Vec::new();
    out.extend_from_slice(REGION_MAGIC);
    out.extend_from_slice(&REGION_VERSION.to_le_bytes());
    out.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for k in &keys {
        let blob = &region.chunks[k];
        out.extend_from_slice(&k.to_le_bytes());
        out.extend_from_slice(&(blob.len() as u32).to_le_bytes());
        out.extend_from_slice(&(fnv1a(0, blob) as u32).to_le_bytes());
    }
    for k in &keys {
        out.extend_from_slice(&region.chunks[k]);
    }
    out
}

impl Store {
    /// Open the world at `root`, creating it with `params` if it is new.
    /// Returns the store and the parameters the world actually has.
    pub fn open(root: &Path, params: WorldParams) -> Result<(Self, WorldParams), String> {
        let err = |e: io::Error| format!("{}: {e}", root.display());
        fs::create_dir_all(root.join("regions")).map_err(err)?;
        let header = root.join("world.txt");
        let params = match read_kv(&header) {
            Some(kv) => {
                let get = |k: &str| kv.get(k).ok_or_else(|| format!("{}: missing '{k}'", header.display()));
                let num = |k: &str| -> Result<u64, String> {
                    get(k)?.parse().map_err(|_| format!("{}: bad '{k}'", header.display()))
                };
                if num("format")? != WORLD_FORMAT as u64 {
                    return Err(format!(
                        "{}: world format {} is not the format {} this build reads. \
                         Start a new world with --world NAME, or move or delete {}",
                        header.display(),
                        num("format")?,
                        WORLD_FORMAT,
                        root.display()
                    ));
                }
                if num("generator")? != GENERATOR_VERSION as u64 {
                    return Err(format!(
                        "{}: made with terrain generator {} but this build has generator {}, so its edits would not line up. \
                         Start a new world with --world NAME, or move or delete {}",
                        header.display(),
                        num("generator")?,
                        GENERATOR_VERSION,
                        root.display()
                    ));
                }
                if num("chunk_size")? != CHUNK_SIZE as u64 {
                    return Err(format!("{}: chunk size {} is not supported", header.display(), num("chunk_size")?));
                }
                let cell_bits = u32::from_str_radix(get("cell_size_bits")?, 16)
                    .map_err(|_| format!("{}: bad 'cell_size_bits'", header.display()))?;
                let fragment_bits = u32::from_str_radix(get("min_fragment_bits")?, 16)
                    .map_err(|_| format!("{}: bad 'min_fragment_bits'", header.display()))?;
                WorldParams {
                    seed: num("seed")?,
                    cell_size: f32::from_bits(cell_bits),
                    min_fragment: f32::from_bits(fragment_bits),
                }
            }
            None => {
                write_kv(
                    &header,
                    &[
                        ("format", WORLD_FORMAT.to_string()),
                        ("seed", params.seed.to_string()),
                        ("cell_size", params.cell_size.to_string()),
                        // Exact bits: the decimal above is for people.
                        ("cell_size_bits", format!("{:08x}", params.cell_size.to_bits())),
                        ("min_fragment", params.min_fragment.to_string()),
                        ("min_fragment_bits", format!("{:08x}", params.min_fragment.to_bits())),
                        ("chunk_size", CHUNK_SIZE.to_string()),
                        ("generator", GENERATOR_VERSION.to_string()),
                    ],
                )
                .map_err(err)?;
                params
            }
        };
        Ok((Self { root: root.to_path_buf(), regions: Mutex::default(), temp_counter: AtomicU64::new(0) }, params))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn region_path(&self, (tier, r): RegionPos) -> PathBuf {
        self.root.join("regions").join(format!("r{tier}.{}.{}.{}.tngr", r.x, r.y, r.z))
    }

    fn with_region<T>(&self, r: RegionPos, f: impl FnOnce(&mut Region) -> T) -> T {
        let mut regions = self.regions.lock().unwrap();
        let region = regions.entry(r).or_insert_with(|| {
            let path = self.region_path(r);
            let chunks = match fs::read(&path) {
                Ok(bytes) => parse_region(&bytes).unwrap_or_else(|e| {
                    eprintln!("tungsten: ignoring {}: {e}", path.display());
                    HashMap::new()
                }),
                Err(_) => HashMap::new(),
            };
            Region { chunks, dirty: false }
        });
        f(region)
    }

    /// Saved edits of a cell, if it has any.
    pub fn get(&self, pos: CellPos) -> Option<Arc<[u8]>> {
        let (r, index) = region_of(pos);
        self.with_region(r, |region| region.chunks.get(&index).cloned())
    }

    /// Every saved cell of a region.
    pub fn region(&self, r: RegionPos) -> Vec<(CellPos, Arc<[u8]>)> {
        self.with_region(r, |region| {
            let mut cells: Vec<(CellPos, Arc<[u8]>)> = region
                .chunks
                .iter()
                .map(|(&i, bytes)| {
                    let (i, mask) = (i as i32, (1 << REGION_BITS) - 1);
                    let local = IVec3::new(i & mask, i >> (2 * REGION_BITS), (i >> REGION_BITS) & mask);
                    (CellPos::new(r.0, (r.1 << REGION_BITS) + local), bytes.clone())
                })
                .collect();
            cells.sort_unstable_by_key(|(p, _)| (p.pos.y, p.pos.z, p.pos.x));
            cells
        })
    }

    /// Record a cell's edits. They reach disk at the next [`flush`](Self::flush).
    pub fn put(&self, pos: CellPos, bytes: Vec<u8>) {
        let (r, index) = region_of(pos);
        self.with_region(r, |region| {
            region.chunks.insert(index, Arc::from(bytes));
            region.dirty = true;
        });
    }

    /// Write every region changed since the last flush. Returns how many were written.
    pub fn flush(&self) -> io::Result<usize> {
        let pending: Vec<(RegionPos, Vec<u8>)> = {
            let mut regions = self.regions.lock().unwrap();
            regions
                .iter_mut()
                .filter(|(_, region)| region.dirty)
                .map(|(&r, region)| {
                    region.dirty = false;
                    (r, serialize_region(region))
                })
                .collect()
        };
        let mut result = Ok(pending.len());
        for (r, bytes) in pending {
            let id = self.temp_counter.fetch_add(1, Ordering::Relaxed);
            if let Err(e) = write_atomic(&self.region_path(r), id, &bytes) {
                // Try again at the next flush.
                self.regions.lock().unwrap().entry(r).or_default().dirty = true;
                result = Err(e);
            }
        }
        result
    }
}

/// Read a `key = value` text file. `None` if it does not exist or cannot be read.
pub fn read_kv(path: &Path) -> Option<HashMap<String, String>> {
    let text = fs::read_to_string(path).ok()?;
    Some(
        text.lines()
            .filter_map(|line| line.split_once('='))
            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
            .collect(),
    )
}

/// Write a `key = value` text file atomically.
pub fn write_kv(path: &Path, pairs: &[(&str, String)]) -> io::Result<()> {
    let text: String = pairs.iter().map(|(k, v)| format!("{k} = {v}\n")).collect();
    write_atomic(path, 0, text.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("tungsten-test-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn cells_survive_reopen_and_corruption_is_contained() {
        let dir = temp_dir("store");
        let params = WorldParams { seed: 42, cell_size: 0.25, min_fragment: 0.75 };
        let (store, got) = Store::open(&dir, params).unwrap();
        assert_eq!(got, params);
        let cell = |tier, x, y, z| CellPos::new(tier, IVec3::new(x, y, z));
        // The same position in the other tier is a different cell.
        let (a, b, far) = (cell(0, 3, -2, 17), cell(0, -1, -1, -1), cell(1, 3, -2, 17));
        store.put(a, vec![1, 2, 3]);
        store.put(b, vec![9; 5000]);
        store.put(far, vec![]);
        assert_eq!(store.flush().unwrap(), 3);
        assert_eq!(store.flush().unwrap(), 0);
        store.put(a, vec![4, 5]);
        drop(store);

        // Reopening ignores the requested parameters in favour of the saved ones.
        let (store, got) = Store::open(&dir, WorldParams::default()).unwrap();
        assert_eq!(got, params);
        // The second put to `a` was never flushed.
        assert_eq!(store.get(a).as_deref(), Some(&[1u8, 2, 3][..]));
        assert_eq!(store.get(b).map(|v| v.len()), Some(5000));
        assert_eq!(store.get(far).map(|v| v.len()), Some(0));
        assert!(store.get(cell(0, 3, -2, 18)).is_none());
        let cells: Vec<CellPos> = store.region(region_of(a).0).into_iter().map(|(p, _)| p).collect();
        assert_eq!(cells, vec![a]);
        assert_eq!(store.region(region_of(b).0).len(), 1);
        assert_eq!(store.region(region_of(far).0).len(), 1);
        assert!(store.region((0, IVec3::splat(50))).is_empty());
        drop(store);

        // Flip a byte in the last blob of a region file: only that chunk is lost.
        let ((_, r), _) = region_of(b);
        let path = dir.join("regions").join(format!("r0.{}.{}.{}.tngr", r.x, r.y, r.z));
        let mut bytes = fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 0xff;
        fs::write(&path, &bytes).unwrap();
        let (store, _) = Store::open(&dir, params).unwrap();
        assert!(store.get(b).is_none());
        assert!(store.get(a).is_some());
        let _ = fs::remove_dir_all(&dir);
    }
}
