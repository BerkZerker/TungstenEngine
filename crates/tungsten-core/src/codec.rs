//! Save format of an edited chunk: only the samples edits have touched.
//!
//! Part of the bit-exact contract: the same chunk state always encodes to
//! the same bytes, and decoding is exact.
//!
//! Layout (version 1):
//!   u8      version
//!   varint* run lengths over the touched bitmap, alternating untouched and
//!           touched, starting with untouched, until all samples are covered
//!   rle     fill of each touched sample, in index order
//!   rle     material of each touched sample, in index order
//!   varint  count of awake (still settling) samples, then their indices as
//!           ascending deltas
//! where `rle` is repeated (value u8, run varint) pairs.

use crate::chunk::Chunk;
use crate::coords::CHUNK_VOLUME;
use crate::field::Touched;

const VERSION: u8 = 1;

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(bytes: &[u8], at: &mut usize) -> Result<u32, String> {
    let (mut v, mut shift) = (0u32, 0);
    loop {
        let b = *bytes.get(*at).ok_or("truncated")?;
        *at += 1;
        if shift >= 32 {
            return Err("varint too long".into());
        }
        v |= ((b & 0x7f) as u32) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

fn put_rle(out: &mut Vec<u8>, values: &[u8]) {
    let mut i = 0;
    while i < values.len() {
        let run = values[i..].iter().take_while(|&&v| v == values[i]).count();
        out.push(values[i]);
        put_varint(out, run as u32);
        i += run;
    }
}

fn get_rle(bytes: &[u8], at: &mut usize, count: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let v = *bytes.get(*at).ok_or("truncated")?;
        *at += 1;
        let run = get_varint(bytes, at)? as usize;
        if run == 0 || out.len() + run > count {
            return Err("bad run length".into());
        }
        out.resize(out.len() + run, v);
    }
    Ok(out)
}

#[inline]
fn is_touched(t: &Touched, i: usize) -> bool {
    t[i / 64] >> (i % 64) & 1 == 1
}

/// Encode the touched samples of `chunk`, plus which samples are still settling.
pub fn encode(chunk: &Chunk, touched: &Touched, awake: &[u16]) -> Vec<u8> {
    let mut out = vec![VERSION];
    let (mut fills, mut mats) = (Vec::new(), Vec::new());
    let (mut run, mut state) = (0u32, false);
    for i in 0..CHUNK_VOLUME {
        let t = is_touched(touched, i);
        if t != state {
            put_varint(&mut out, run);
            run = 0;
            state = t;
        }
        run += 1;
        if t {
            let s = chunk.get(i);
            fills.push(s.fill);
            mats.push(s.mat);
        }
    }
    put_varint(&mut out, run);
    put_rle(&mut out, &fills);
    put_rle(&mut out, &mats);

    let mut awake = awake.to_vec();
    awake.sort_unstable();
    awake.dedup();
    put_varint(&mut out, awake.len() as u32);
    let mut prev = 0;
    for a in awake {
        put_varint(&mut out, (a - prev) as u32);
        prev = a;
    }
    out
}

/// What a saved chunk adds to its generated state.
pub struct Decoded {
    pub touched: Touched,
    /// Local indices of samples that were still settling when saved.
    pub awake: Vec<u16>,
}

/// Overlay a saved delta onto a freshly generated chunk.
pub fn apply(chunk: &mut Chunk, bytes: &[u8]) -> Result<Decoded, String> {
    if bytes.first() != Some(&VERSION) {
        return Err(format!("unsupported chunk format {:?}", bytes.first()));
    }
    let mut at = 1;
    let mut touched: Touched = Box::new([0; CHUNK_VOLUME / 64]);
    let (mut i, mut state, mut count) = (0usize, false, 0usize);
    while i < CHUNK_VOLUME {
        let run = get_varint(bytes, &mut at)? as usize;
        if i + run > CHUNK_VOLUME || (run == 0 && i != 0) {
            return Err("bad bitmap run".into());
        }
        if state {
            for k in i..i + run {
                touched[k / 64] |= 1 << (k % 64);
            }
            count += run;
        }
        i += run;
        state = !state;
    }
    let fills = get_rle(bytes, &mut at, count)?;
    let mats = get_rle(bytes, &mut at, count)?;
    let n = get_varint(bytes, &mut at)? as usize;
    let mut awake = Vec::with_capacity(n.min(CHUNK_VOLUME));
    let mut prev = 0u32;
    for _ in 0..n {
        prev += get_varint(bytes, &mut at)?;
        if prev as usize >= CHUNK_VOLUME {
            return Err("awake index out of range".into());
        }
        awake.push(prev as u16);
    }
    if at != bytes.len() {
        return Err("trailing bytes".into());
    }

    if count > 0 {
        let dense = chunk.dense_mut();
        let mut k = 0;
        for i in 0..CHUNK_VOLUME {
            if is_touched(&touched, i) {
                dense.fill[i] = fills[k];
                dense.mat[i] = mats[k];
                k += 1;
            }
        }
    }
    Ok(Decoded { touched, awake })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{Sample, Volume};
    use crate::edit::{EditKind, EditOp, EditRules, FP, apply as apply_edit};
    use crate::worldgen::Generator;
    use crate::{WorldParams, coords::index_xyz};
    use glam::IVec3;

    #[test]
    fn edited_chunk_round_trips_through_generate_and_overlay() {
        let params = WorldParams::default();
        let generator = Generator::new(params);
        let mut vol = Volume::new();
        for y in -2..=1 {
            for z in -1..=1 {
                for x in -1..=1 {
                    let p = IVec3::new(x, y, z);
                    vol.insert(p, generator.generate(p));
                }
            }
        }
        let rules = EditRules::new(&params);
        for tick in 0..80 {
            let op = EditOp {
                kind: if tick == 79 { EditKind::Place } else { EditKind::Break },
                center: [10 * FP, -FP, 12 * FP],
                radius: 5 * FP,
                material: 3,
                tick,
                exclude: None,
            };
            apply_edit(&mut vol, &rules, &op);
        }
        assert!(!vol.touched.is_empty());
        let mut total = 0;
        for (pos, touched) in &vol.touched {
            let bytes = encode(&vol.chunks[pos], touched, &[7, 3, 3, 900]);
            total += bytes.len();
            assert_eq!(bytes, encode(&vol.chunks[pos], touched, &[900, 7, 3]), "encoding is not canonical");
            let mut fresh = generator.generate(*pos);
            let decoded = apply(&mut fresh, &bytes).unwrap();
            assert_eq!(decoded.awake, vec![3, 7, 900]);
            assert!(decoded.touched[..] == touched[..]);
            assert_eq!(fresh.content_hash(0), vol.chunks[pos].content_hash(0), "chunk {pos} differs after reload");
        }
        // A 2.5 m crater costs kilobytes, not the 65 KB of a raw chunk.
        assert!(total < 24_000, "delta of {total} bytes");
    }

    #[test]
    fn rejects_corrupt_data_without_panicking() {
        let mut chunk = Chunk::AIR;
        let mut touched: Touched = Box::new([0; CHUNK_VOLUME / 64]);
        let i = index_xyz(4, 5, 6);
        touched[i / 64] |= 1 << (i % 64);
        chunk.dense_mut().fill[i] = 200;
        chunk.dense_mut().mat[i] = 2;
        let good = encode(&chunk, &touched, &[]);
        let mut fresh = Chunk::AIR;
        apply(&mut fresh, &good).unwrap();
        assert_eq!(fresh.get(i), Sample { fill: 200, mat: 2 });
        for cut in 0..good.len() {
            assert!(apply(&mut Chunk::AIR.clone(), &good[..cut]).is_err(), "accepted truncation at {cut}");
        }
        let mut bad = good.clone();
        bad[0] = 9;
        assert!(apply(&mut Chunk::AIR.clone(), &bad).is_err());
        bad = good.clone();
        bad.push(0);
        assert!(apply(&mut Chunk::AIR.clone(), &bad).is_err());
    }
}
