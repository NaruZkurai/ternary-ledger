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
        put_varint(&mut out, t.id as u64);
        put_varint(&mut out, t.frequency as u64);
        put_f32(&mut out, t.viability);
        put_varint(&mut out, t.pattern.len() as u64);
        for &p in &t.pattern {
            put_u32(&mut out, p);
        }
        put_varint(&mut out, t.code.v.len() as u64);
        for &c in &t.code.v {
            out.push(c as u8);
        }
    }
    put_u32(&mut out, TRAILER);
    out
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
        ledger
            .add_raw(GeneratedToken { id, pattern, code: TernaryCode::new(cv), frequency, viability });
    }
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
        let w: Vec<u32> = vec![10, 20, 30, 40];
        led.register_pattern(&w);
        led.set_entry(led.lookup_pattern(&w).unwrap(), TernaryCode::new(vec![1, -1, 0, 1, 1, 0, -1, 1]), 0.87);
        let bytes = encode(&led);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.config().n_vocab, 1000);
        assert_eq!(back.len(), 1);
        let e = back.entries();
        assert_eq!(e[0].pattern, w);
        assert_eq!(e[0].code.dim(), 8);
        assert!((e[0].viability - 0.87).abs() < 1e-6);
    }
}
