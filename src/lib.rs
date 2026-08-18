//! ternary-ledger
//!
//! A dynamic token-compression ledger for the pi harness + llama.cpp direct-token
//! endpoint. It mints extra *compressed token* ids (>= `n_vocab`), each
//! mathematically derived from a raw token window (`pattern` + `code`), so a
//! long context can be shortened on the wire to a single ledger-backed id that
//! the GPU decodes on the fly. The full current ledger can be serialized and
//! sent/received at any time via [`wire`].
//!
//! Modules:
//!   * [`ledger`] — the in-memory dynamic ledger (mint / reuse / prune / look up).
//!   * [`wire`] — versioned binary wire format for send/recv of the ledger.
//!
//! @module ternary-ledger

pub mod ledger;
pub mod wire;

pub use ledger::{GeneratedToken, Ledger, LedgerConfig, TernaryCode};
