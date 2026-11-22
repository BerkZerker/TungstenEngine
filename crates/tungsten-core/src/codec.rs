//! Byte encodings for saves: variable-length integers, run-length arrays and
//! overlays of changed samples.
//!
//! Part of the bit-exact contract: the same state always encodes to the same
//! bytes, and decoding is exact.

pub fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

pub fn get_varint(bytes: &[u8], at: &mut usize) -> Result<u64, String> {
    let (mut v, mut shift) = (0u64, 0);
    loop {
        let b = *bytes.get(*at).ok_or("truncated")?;
        *at += 1;
        if shift >= 64 {
            return Err("varint too long".into());
        }
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
        shift += 7;
    }
}

/// Signed integers, small magnitudes first.
pub fn put_signed(out: &mut Vec<u8>, v: i64) {
    put_varint(out, ((v << 1) ^ (v >> 63)) as u64);
}

pub fn get_signed(bytes: &[u8], at: &mut usize) -> Result<i64, String> {
    let v = get_varint(bytes, at)?;
    Ok((v >> 1) as i64 ^ -((v & 1) as i64))
}

pub fn get_u8(bytes: &[u8], at: &mut usize) -> Result<u8, String> {
    let b = *bytes.get(*at).ok_or("truncated")?;
    *at += 1;
    Ok(b)
}

/// Repeated (value, run length) pairs.
pub fn put_rle(out: &mut Vec<u8>, values: &[u8]) {
    let mut i = 0;
    while i < values.len() {
        let run = values[i..].iter().take_while(|&&v| v == values[i]).count();
        out.push(values[i]);
        put_varint(out, run as u64);
        i += run;
    }
}

pub fn get_rle(bytes: &[u8], at: &mut usize, count: usize) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let v = get_u8(bytes, at)?;
        let run = get_varint(bytes, at)? as usize;
        if run == 0 || out.len() + run > count {
            return Err("bad run length".into());
        }
        out.resize(out.len() + run, v);
    }
    Ok(out)
}

/// The samples of `now` that differ from `base`, as bytes. Both are the same
/// box of samples in the same order. Empty if nothing differs.
///
/// Layout:
///   varint  number of changed samples
///   varint* run lengths, alternating unchanged and changed, starting with
///           unchanged, up to and including the last changed run
///   rle     fill of each changed sample, in order
///   rle     material of each changed sample, in order
pub fn overlay_encode(base: (&[u8], &[u8]), now: (&[u8], &[u8])) -> Vec<u8> {
    let n = base.0.len();
    let differs = |i: usize| base.0[i] != now.0[i] || base.1[i] != now.1[i];
    let (mut runs, mut fills, mats) = (Vec::new(), Vec::new(), &mut Vec::new());
    let (mut run, mut state) = (0u64, false);
    for i in 0..n {
        let d = differs(i);
        if d != state {
            put_varint(&mut runs, run);
            run = 0;
            state = d;
        }
        run += 1;
        if d {
            fills.push(now.0[i]);
            mats.push(now.1[i]);
        }
    }
    if fills.is_empty() {
        return Vec::new();
    }
    if state {
        put_varint(&mut runs, run);
    }
    let mut out = Vec::with_capacity(runs.len() + fills.len() / 4 + 8);
    put_varint(&mut out, fills.len() as u64);
    out.extend_from_slice(&runs);
    put_rle(&mut out, &fills);
    put_rle(&mut out, mats);
    out
}

/// Call `set(index, fill, material)` for every sample an overlay changes,
/// in ascending index order. `len` is the number of samples it was made over.
pub fn overlay_apply(bytes: &[u8], len: usize, mut set: impl FnMut(usize, u8, u8)) -> Result<(), String> {
    if bytes.is_empty() {
        return Ok(());
    }
    let mut at = 0;
    let count = get_varint(bytes, &mut at)? as usize;
    if count == 0 || count > len {
        return Err("bad overlay count".into());
    }
    // Runs first, so the values that follow them can be found.
    let mut runs = Vec::new();
    let (mut covered, mut changed, mut state) = (0usize, 0usize, false);
    while changed < count {
        let run = get_varint(bytes, &mut at)? as usize;
        if covered + run > len || (run == 0 && covered != 0) {
            return Err("bad overlay run".into());
        }
        if state {
            runs.push((covered, run));
            changed += run;
        }
        covered += run;
        state = !state;
    }
    if changed != count {
        return Err("overlay runs do not add up".into());
    }
    let fills = get_rle(bytes, &mut at, count)?;
    let mats = get_rle(bytes, &mut at, count)?;
    if at != bytes.len() {
        return Err("trailing bytes in overlay".into());
    }
    let mut k = 0;
    for (start, run) in runs {
        for i in start..start + run {
            set(i, fills[k], mats[k]);
            k += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_round_trip() {
        let mut out = Vec::new();
        let values = [0i64, 1, -1, 127, -128, 300_000, -9_000_000_000, i64::MAX, i64::MIN];
        for &v in &values {
            put_signed(&mut out, v);
        }
        put_varint(&mut out, u64::MAX);
        let mut at = 0;
        for &v in &values {
            assert_eq!(get_signed(&out, &mut at), Ok(v));
        }
        assert_eq!(get_varint(&out, &mut at), Ok(u64::MAX));
        assert_eq!(at, out.len());
        assert!(get_varint(&out, &mut at).is_err());
    }

    #[test]
    fn overlays_round_trip_and_reject_damage() {
        let n = 5000;
        let base: (Vec<u8>, Vec<u8>) = ((0..n).map(|i| (i % 7) as u8).collect(), vec![1; n]);
        assert!(overlay_encode((&base.0, &base.1), (&base.0, &base.1)).is_empty());
        for pattern in [vec![0usize], vec![n - 1], vec![0, 1, 2, 900, 901, 4999], (100..3000).collect()] {
            let mut now = base.clone();
            for &i in &pattern {
                now.0[i] = 200;
                now.1[i] = (i % 3) as u8 + 1;
            }
            let bytes = overlay_encode((&base.0, &base.1), (&now.0, &now.1));
            let mut out = base.clone();
            let mut seen = Vec::new();
            overlay_apply(&bytes, n, |i, f, m| {
                seen.push(i);
                out.0[i] = f;
                out.1[i] = m;
            })
            .unwrap();
            assert!(out == now && seen.windows(2).all(|w| w[0] < w[1]));
            assert!(bytes.len() < pattern.len() * 2 + 32, "{} bytes for {} samples", bytes.len(), pattern.len());
            // Cut short or made for a smaller box: an error, never a panic.
            for cut in 0..bytes.len() {
                assert!(overlay_apply(&bytes[..cut], n, |_, _, _| {}).is_err() || cut == 0);
            }
            let _ = overlay_apply(&bytes, 10, |i, _, _| assert!(i < 10));
        }
    }
}
