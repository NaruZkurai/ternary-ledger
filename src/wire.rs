//! Ledger wire format — send/receive the current token-compression ledger.
//!
//! A compact, versioned binary layout that mirrors the fork's `tcomp`
//! checkpoint token-map block (`id / frequency / viability / pattern / code`),
//! extended with the layer config so a peer (a llama.cpp endpoint or the pi
//! harness) can reconstruct the ledger without prior context.
//!
//! Layout (little-endian; varint = unsigned LEB128):
//!   magic        u32  "TLDG" (0x4744_4C54)
//!   version      u8   = 1
//!   n_vocab      u32
//!   n_window     varint
//!   n_code       varint
//!   n_entries    varint
//!   per entry:
//!     id                varint
//!     frequency         varint
//!     viability         f32
//!     pattern_len       varint  == n_window
//!     pattern           u32 x pattern_len
//!     code_len          varint  == n_code
//!     code              u8 x code_len  (each in {-1,0,+1} stored as signed byte)
//!   trailer       u32  = 0x0000_4C44 ("LD\0\0") for integrity
//!
//! @module ternary-ledger/wire

use crate::ledger::{GeneratedToken, Ledger, LedgerConfig, TernaryCode};

pub const MAGIC: u32 = 0x4744_4C54u32; // "TLDG"
pub const VERSION: u8 = 1;
pub const TRAILER: u32 = 0x0000_4C44u32;

#[derive(Debug)]
pub struct WireError(pub String);

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "wire: {}", self.0)
    }
}

impl std::error::Error for WireError {}

// --- encoding helpers ---

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_f32(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if v == 0 {
            break;
        }
    }
}

/// Serialize the full ledger (config + entries) into the binary wire format.
pub fn encode(ledger: &Ledger) -> Vec<u8> {
    let cfg = ledger.config();
    let mut out = Vec::new();
    put_u32(&mut out, MAGIC);
    out.push(VERSION);
    put_u32(&mut out, cfg.n_vocab);
    put_varint(&mut out, cfg.n_window as u64);
    put_varint(&mut out, cfg.n_code as u64);
    let entries = ledger.entries();
    put_varint(&mut out, entries.len() as u64);
    for t in &entries {
        put_entry(&mut out, t);
    }
    put_u32(&mut out, TRAILER);
    out
}

// Shared per-entry byte layout, used by both the full-ledger encoder and the
// single-entry `/mint_e_token/` payload: `{eid, oid..., formula}`.
fn put_entry(out: &mut Vec<u8>, t: &GeneratedToken) {
    put_varint(out, t.id as u64); // eid
    put_varint(out, t.frequency as u64);
    put_f32(out, t.viability);
    put_varint(out, t.pattern.len() as u64); // oid tuple length
    for &p in &t.pattern {
        put_u32(out, p); // oid...
    }
    put_varint(out, t.code.v.len() as u64); // formula (ternary code) length
    for &c in &t.code.v {
        out.push(c as u8); // formula values in {-1,0,+1}
    }
}

/// Encode a single etoken entry — the `/mint_e_token/` payload
/// `{eid: {oid..., formula}}`. A server that replied "unknown etoken in ledger"
/// receives one of these to mint the etoken on the fly.
pub fn encode_entry(t: &GeneratedToken) -> Vec<u8> {
    let mut out = Vec::new();
    put_entry(&mut out, t);
    out
}

/// Decode a single etoken entry from [`encode_entry`]. Returns it verbatim —
/// no ledger, no config, just the one `{eid, oid..., formula}` record.
pub fn decode_entry(bytes: &[u8]) -> Result<GeneratedToken, WireError> {
    let mut r = Reader { b: bytes, pos: 0 };
    let eid = r.varint()? as u32;
    let frequency = r.varint()? as u32;
    let viability = r.f32()?;
    let pattern_len = r.varint()? as usize;
    let mut oid = Vec::with_capacity(pattern_len);
    for _ in 0..pattern_len {
        oid.push(r.u32()?);
    }
    let code_len = r.varint()? as usize;
    let mut cv = Vec::with_capacity(code_len);
    for _ in 0..code_len {
        cv.push(r.u8()? as i8);
    }
    Ok(GeneratedToken { id: eid, pattern: oid, code: TernaryCode::new(cv), frequency, viability })
}

// --- decoding helpers ---

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32, WireError> {
        if self.pos + 4 > self.b.len() {
            return Err(WireError("truncated u32".into()));
        }
        let v = u32::from_le_bytes([self.b[self.pos], self.b[self.pos + 1], self.b[self.pos + 2], self.b[self.pos + 3]]);
        self.pos += 4;
        Ok(v)
    }
    fn u8(&mut self) -> Result<u8, WireError> {
        if self.pos >= self.b.len() {
            return Err(WireError("truncated u8".into()));
        }
        let v = self.b[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, WireError> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn varint(&mut self) -> Result<u64, WireError> {
        let mut result = 0u64;
        let mut shift = 0;
        loop {
            let b = self.u8()?;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(WireError("varint too long".into()));
            }
        }
        Ok(result)
    }
}

/// Decode a wire-format ledger back into a [`Ledger`].
pub fn decode(bytes: &[u8]) -> Result<Ledger, WireError> {
    let mut r = Reader { b: bytes, pos: 0 };
    if r.u32()? != MAGIC {
        return Err(WireError("bad magic".into()));
    }
    let version = r.u8()?;
    if version != VERSION {
        return Err(WireError(format!("unsupported version {version}")));
    }
    let n_vocab = r.u32()?;
    let n_window = r.varint()? as usize;
    let n_code = r.varint()? as usize;
    let n_entries = r.varint()? as usize;
    let mut ledger = Ledger::new(LedgerConfig { n_vocab, n_window, n_code, ..LedgerConfig::default() });
    let mut entries = Vec::with_capacity(n_entries);
    for _ in 0..n_entries {
        let id = r.varint()? as u32;
        let frequency = r.varint()? as u32;
        let viability = r.f32()?;
        let pattern_len = r.varint()? as usize;
        let mut pattern = Vec::with_capacity(pattern_len);
        for _ in 0..pattern_len {
            pattern.push(r.u32()?);
        }
        let code_len = r.varint()? as usize;
        let mut cv = Vec::with_capacity(code_len);
        for _ in 0..code_len {
            cv.push(r.u8()? as i8);
        }
        entries.push(GeneratedToken { id, pattern, code: TernaryCode::new(cv), frequency, viability });
    }
    ledger.load_entries(entries);
    if r.u32()? != TRAILER {
        return Err(WireError("bad trailer".into()));
    }
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let mut led = Ledger::new(LedgerConfig { n_vocab: 1000, n_window: 4, n_code: 8, ..LedgerConfig::default() });
        led.load_entries(vec![
            GeneratedToken { id: 400_000, pattern: vec![10, 20, 30, 40], code: TernaryCode::new(vec![1, -1, 0, 1, 1, 0, -1, 1]), frequency: 2, viability: 0.87 },
        ]);
        let bytes = encode(&led);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.config().n_vocab, 1000);
        assert_eq!(back.len(), 1);
        assert!(back.contains(400_000));
        let e = back.get_by_id(400_000).unwrap();
        assert_eq!(e.pattern, vec![10, 20, 30, 40]);
        assert_eq!(e.code.dim(), 8);
        assert!((e.viability - 0.87).abs() < 1e-6);
    }

    #[test]
    fn single_entry_roundtrip() {
        // The `/mint_e_token/` payload: {eid: {oid..., formula}}
        let t = GeneratedToken { id: 777_001, pattern: vec![1, 2, 3], code: TernaryCode::new(vec![1, 0, -1, 1, -1, 0, 0, 1]), frequency: 1, viability: 0.6 };
        let bytes = encode_entry(&t);
        let back = decode_entry(&bytes).unwrap();
        assert_eq!(back.id, 777_001);
        assert_eq!(back.pattern, vec![1, 2, 3]);
        assert_eq!(back.code.v, vec![1, 0, -1, 1, -1, 0, 0, 1]);
        assert!((back.viability - 0.6).abs() < 1e-6);
    }
}
