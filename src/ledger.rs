//! Dynamic compressed-token ledger.
//!
//! Mirrors the fork's `tcomp` "generated token map" as a clean Rust module:
//! a set of *minted* custom tokens (ids >= `n_vocab`), each mathematically
//! derived from a raw token window (its `pattern`) and a compact `ternary_code`
//! that the GPU can decode back into that window on the fly. This shortens the
//! on-the-wire context: instead of sending an `n_window`-long sequence, the
//! harness sends one ledger-backed token id.
//!
//! The ledger is an **array of formulas for the minted custom etokens**. It is
//! fed from two directions, sharing one entry type and one wire format:
//!
//! * **External** — the etoken definitions are authored outside the crate and
//!   handed in via [`Ledger::load_entries`] (or `POST /ledger` over HTTP).
//!   Entries keep whatever `id >= n_vocab` was assigned externally; no id is
//!   auto-derived.
//! * **Internal** — the crate derives an etoken itself from a raw window via
//!   [`Ledger::mint_pattern`], producing the same [`GeneratedToken`] record.
//!
//! Either way each entry is `{id, pattern, code, frequency, viability}`, exactly
//! mirroring the fork's `tcomp::generated_token`, and serializes through the
//! `TLDG` wire format in [`crate::wire`] so the current ledger can be sent to /
//! received from a llama.cpp endpoint at any time.
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

    /// --- EXTERNAL path ---
    ///
    /// Load an externally-authored array of etoken definitions wholesale,
    /// replacing the previous contents. Each entry keeps whatever `id >=
    /// n_vocab` it was given; no id is auto-assigned. This is the "give / create
    /// externally" path — the ledger is an input table of formulas, and is what
    /// `POST /ledger` deserializes into.
    pub fn load_entries(&mut self, entries: Vec<GeneratedToken>) {
        self.map.clear();
        for t in entries {
            let key = pattern_hash(&t.pattern);
            if t.id >= self.next_id {
                self.next_id = t.id + 1;
            }
            self.map.insert(key, t);
        }
    }

    /// --- INTERNAL path ---
    ///
    /// Derive an etoken from a raw window and add it to the ledger, returning
    /// its minted id (`>= n_vocab`). On repeat windows the existing entry's
    /// frequency is bumped and its id returned (mint is idempotent per pattern).
    /// The caller supplies the ternary `code` and `viability` for the pattern —
    /// the internal producer of those is a host trainer that fed the layer; this
    /// method only marshals the window + code into a ledger record.
    pub fn mint_pattern(&mut self, pattern: &[u32], code: TernaryCode, viability: f32) -> u32 {
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
        self.map.insert(
            key,
            GeneratedToken {
                id,
                pattern: pattern.to_vec(),
                code,
                frequency: 1,
                viability,
            },
        );
        id
    }

    /// All current minted entries, in stable (sorted-by-id) order.
    pub fn entries(&self) -> Vec<GeneratedToken> {
        let mut v: Vec<GeneratedToken> = self.map.values().cloned().collect();
        v.sort_by_key(|t| t.id);
        v
    }

    /// Look up an entry by minted id, if present.
    pub fn get_by_id(&self, id: u32) -> Option<&GeneratedToken> {
        self.map.values().find(|t| t.id == id)
    }

    /// Recognition check: does the ledger already know this minted id?
    ///
    /// This is what lets a server answer "unknown etoken in ledger" for an
    /// id >= n_vocab it has not yet seen, versus "recognized" for one already
    /// minted.
    pub fn contains(&self, id: u32) -> bool {
        self.map.values().any(|t| t.id == id)
    }

    /// --- DYNAMIC path (e.g. `/mint_e_token/`) ---
    ///
    /// Register an etoken with an **externally-specified** minted id — the
    /// caller chose `eid` (id >= n_vocab), not this crate. `eid` maps to the
    /// original token(s) it represents (`oid` tuple) and its ternary `formula`.
    ///
    /// This is the on-the-fly mint a server performs when it receives
    /// `{eid: {oid..., formula}}` after replying "unknown etoken in ledger".
    ///
    /// Returns false if `eid` is already minted or is below n_vocab (not an
    /// etoken id). Callers should keep the LEDGER KEYED BY ID model in mind:
    /// `oid...` is the pattern the etoken decodes back to; the server stores
    /// the etoken in KV *as the etoken*, using `formula` (the code) as its KV
    /// value.
    pub fn mint_external(&mut self, eid: u32, oid: Vec<u32>, formula: TernaryCode) -> bool {
        if eid < self.cfg.n_vocab {
            return false; // not an etoken id
        }
        if self.contains(eid) {
            return false; // already minted
        }
        if self.map.len() >= self.cfg.max_entries {
            self.evict_lowest_viability();
        }
        let key = pattern_hash(&oid);
        if eid >= self.next_id {
            self.next_id = eid + 1;
        }
        self.map.insert(
            key,
            GeneratedToken {
                id: eid,
                pattern: oid,
                code: formula,
                frequency: 1,
                viability: 0.5,
            },
        );
        true
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

    fn code(v: &[i8]) -> TernaryCode {
        TernaryCode::new(v.to_vec())
    }

    #[test]
    fn internal_mint_and_reuse() {
        let mut l = Ledger::new(LedgerConfig::default());
        let win: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let id1 = l.mint_pattern(&win, code(&[1, -1, 0, 1, 1, 0, -1, 1]), 0.8);
        assert!(id1 >= l.config().n_vocab);
        let id2 = l.mint_pattern(&win, code(&[1, -1, 0, 1, 1, 0, -1, 1]), 0.8); // same pattern -> reuse
        assert_eq!(id1, id2);
        assert_eq!(l.lookup_pattern(&win), Some(id1));
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn external_load_keeps_ids() {
        let mut l = Ledger::new(LedgerConfig { n_vocab: 1000, n_window: 4, n_code: 8, ..LedgerConfig::default() });
        l.load_entries(vec![
            GeneratedToken { id: 100_500, pattern: vec![10, 20, 30, 40], code: code(&[1, 0, -1, 1, 1, -1, 0, 1]), frequency: 3, viability: 0.9 },
            GeneratedToken { id: 777_010, pattern: vec![1, 1, 2, 2], code: code(&[0, 0, 1, -1, 1, 1, 0, -1]), frequency: 1, viability: 0.4 },
        ]);
        // Externally-assigned ids are preserved verbatim.
        assert!(l.get_by_id(100_500).is_some());
        assert!(l.get_by_id(777_010).is_some());
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn dynamic_mint_external_and_contains() {
        let mut l = Ledger::new(LedgerConfig { n_vocab: 1000, n_window: 3, n_code: 8, ..LedgerConfig::default() });
        // Id not in ledger yet -> "unknown etoken in ledger"
        assert!(!l.contains(400_000));
        // Mint it on the fly: {eid: {oid..., formula}}
        let ok = l.mint_external(400_000, vec![5, 6, 7], code(&[1, -1, 0, 1, 1, 0, -1, 1]));
        assert!(ok);
        assert!(l.contains(400_000));
        assert_eq!(l.get_by_id(400_000).unwrap().pattern, vec![5, 6, 7]);
        // Duplicate mint rejected; id below n_vocab rejected.
        assert!(!l.mint_external(400_000, vec![8, 9, 10], code(&[0; 8])));
        assert!(!l.mint_external(900, vec![1, 2, 3], code(&[0; 8])));
        assert_eq!(l.len(), 1);
    }

    #[test]
    fn evicts_below_cap() {
        let mut cfg = LedgerConfig::default();
        cfg.max_entries = 2;
        let mut l = Ledger::new(cfg);
        l.mint_pattern(&[1, 1, 1, 1, 1, 1, 1, 1], code(&[0; 64]), 0.5);
        l.mint_pattern(&[2, 2, 2, 2, 2, 2, 2, 2], code(&[0; 64]), 0.99);
        // make the first entry low viability, then a third forces eviction
        let first = l.entries()[0].id;
        let low = l.get_by_id(first).unwrap().clone();
        l.load_entries(vec![low, l.entries()[1].clone()]);
        l.evict_lowest_viability();
        assert_eq!(l.len(), 1);
        assert!(!l.contains(first));
    }
}
