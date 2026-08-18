//! Dynamic compressed-token ledger.
//!
//! Mirrors the fork's `tcomp` "generated token map" as a clean Rust module:
//! a set of *minted* custom tokens (ids >= `n_vocab`), each mathematically
//! derived from a raw token window (its `pattern`) and a compact `ternary_code`
//! that the GPU can decode back into that window on the fly. This shortens the
//! on-the-wire context: instead of sending an `n_window`-long sequence, the
//! harness sends one ledger-backed token id.
//!
//! The ledger is *dynamic*: minting happens as patterns recur, viabilities
//! update, and stale/low-viability entries can be pruned. It is also
//! serializable — the current full ledger can be sent to / received from a
//! llama.cpp endpoint at any time via the compact binary wire format in
//! [`crate::wire`].
//!
//! @module ternary-ledger/ledger

use std::collections::HashMap;

/// A ternary code: `n_code` values each in `{-1, 0, +1}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TernaryCode {
    pub v: Vec<i8>,
}

impl TernaryCode {
    pub fn new(v: Vec<i8>) -> TernaryCode {
        TernaryCode { v }
    }
    pub fn dim(&self) -> usize {
        self.v.len()
    }
}

/// One minted custom-token ledger entry.
///
/// `id` >= n_vocab. `pattern` is the raw token window this token decodes to;
/// `code` is its ternary encoding (the thing the GPU/sampler decodes on the fly).
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedToken {
    pub id: u32,
    pub pattern: Vec<u32>,
    pub code: TernaryCode,
    pub frequency: u32,
    pub viability: f32,
}

/// Configuration for the ledger's dynamic minting/pruning behaviour.
#[derive(Debug, Clone)]
pub struct LedgerConfig {
    /// Base of the minted id space (== the model's `n_vocab`).
    pub n_vocab: u32,
    /// Window length each custom token compresses.
    pub n_window: usize,
    /// Code dimension.
    pub n_code: usize,
    /// Viability floor below which an entry is pruned.
    pub min_viability: f32,
    /// Hard cap on the number of minted tokens.
    pub max_entries: usize,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        LedgerConfig {
            n_vocab: 248_320,
            n_window: 8,
            n_code: 64,
            min_viability: 0.05,
            max_entries: 24_000,
        }
    }
}

/// FNV-1a hash used to key a window pattern without a str dependency.
pub fn pattern_hash(pattern: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &t in pattern {
        let bytes = t.to_le_bytes();
        for &b in &bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

/// The mutable, dynamic token-compression ledger.
#[derive(Debug)]
pub struct Ledger {
    cfg: LedgerConfig,
    /// pattern hash -> entry
    map: HashMap<u64, GeneratedToken>,
    next_id: u32,
}

impl Ledger {
    pub fn new(cfg: LedgerConfig) -> Ledger {
        let next_id = cfg.n_vocab;
        Ledger { cfg, map: HashMap::new(), next_id }
    }

    pub fn config(&self) -> &LedgerConfig {
        &self.cfg
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Look up a minted token for a raw window, if one exists.
    pub fn lookup_pattern(&self, pattern: &[u32]) -> Option<u32> {
        let key = pattern_hash(pattern);
        self.map.get(&key).map(|t| t.id)
    }

    /// Mint or refresh an entry for the given window. If the pattern already has
    /// a token, bumps its frequency. Otherwise a new id (`>= n_vocab`) is minted
    /// with a placeholder viability that a decoder would later refine.
    pub fn register_pattern(&mut self, pattern: &[u32]) -> u32 {
        if pattern.len() != self.cfg.n_window {
            return u32::MAX; // caller must supply exact-window patterns
        }
        let key = pattern_hash(pattern);
        if let Some(tok) = self.map.get_mut(&key) {
            tok.frequency = tok.frequency.saturating_add(1);
            return tok.id;
        }
        if self.map.len() >= self.cfg.max_entries {
            // evict the lowest-viability entry to stay within the cap.
            self.evict_lowest_viability();
        }
        let id = self.next_id;
        self.next_id += 1;
        // A default "cold" code: zeros are revocable; a decoder/PP server that
        // actually trains the layer would replace this with the learned code.
        let code = TernaryCode::new(vec![0; self.cfg.n_code]);
        self.map.insert(
            key,
            GeneratedToken {
                id,
                pattern: pattern.to_vec(),
                code,
                frequency: 1,
                viability: 0.5,
            },
        );
        id
    }

    /// Update the ternary code + viability for an existing minted id.
    /// Returns false if the id is not in the ledger.
    pub fn set_entry(&mut self, id: u32, code: TernaryCode, viability: f32) -> bool {
        for (_k, tok) in self.map.iter_mut() {
            if tok.id == id {
                tok.code = code;
                tok.viability = viability;
                return true;
            }
        }
        false
    }

    /// All current minted entries, in stable (sorted-by-id) order.
    pub fn entries(&self) -> Vec<GeneratedToken> {
        let mut v: Vec<GeneratedToken> = self.map.values().cloned().collect();
        v.sort_by_key(|t| t.id);
        v
    }

    /// Insert a raw entry verbatim (used by the wire decoder / receiving a
    /// ledger). Replaces any entry with the same id.
    pub fn add_raw(&mut self, tok: GeneratedToken) {
        let key = pattern_hash(&tok.pattern);
        if tok.id >= self.next_id {
            self.next_id = tok.id + 1;
        }
        self.map.insert(key, tok);
    }

    fn evict_lowest_viability(&mut self) {
        let mut worst: Option<u64> = None;
        let mut worst_v = f32::MAX;
        for (k, t) in &self.map {
            if t.viability < worst_v {
                worst_v = t.viability;
                worst = Some(*k);
            }
        }
        if let Some(k) = worst {
            self.map.remove(&k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mints_and_reuses() {
        let mut l = Ledger::new(LedgerConfig::default());
        let win: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let id1 = l.register_pattern(&win);
        assert!(id1 >= l.config().n_vocab);
        let id2 = l.register_pattern(&win); // same pattern -> reuse
        assert_eq!(id1, id2);
        assert_eq!(l.lookup_pattern(&win), Some(id1));
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn evicts_below_cap() {
        let mut cfg = LedgerConfig::default();
        cfg.max_entries = 2;
        let mut l = Ledger::new(cfg);
        l.register_pattern(&[1, 1, 1, 1, 1, 1, 1, 1]);
        l.register_pattern(&[2, 2, 2, 2, 2, 2, 2, 2]);
        // make the first entry low viability, then a third forces eviction
        let first = l.entries()[0].id;
        l.set_entry(first, TernaryCode::new(vec![0; 64]), 0.01);
        let w3: Vec<u32> = vec![3, 3, 3, 3, 3, 3, 3, 3];
        l.register_pattern(&w3);
        assert_eq!(l.len(), 2);
        assert_eq!(l.lookup_pattern(&w3), Some(l.entries().last().unwrap().id));
    }
}
