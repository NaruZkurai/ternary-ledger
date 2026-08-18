//! ternary-ledger
//!
//! A dynamic token-compression ledger for the pi harness + llama.cpp direct-token
//! endpoint. It mints extra *compressed token* ids (>= `n_vocab`), each a
//! formula `{oid..., formula}` that the GPU decodes on the fly, so a long
//! context can be shortened on the wire to a single ledger-backed id.
//!
//! The ledger is an **array of formulas for the minted etokens**, fed from two
//! directions sharing one entry type and one wire format:
//!
//! * **External** — author entries and hand them in via [`Ledger::load_entries`]
//!   (or `POST /ledger` over HTTP). Externally-chosen ids are kept verbatim.
//! * **Internal** — derive an etoken from a raw window via
//!   [`Ledger::mint_pattern`].
//! * **Dynamic** — the recognition + mint handshake: [`Ledger::contains`] answers
//!   "unknown etoken in ledger", and [`Ledger::mint_external`] mints an
//!   externally-chosen id on the fly from `{eid: {oid..., formula}}`, i.e. the
//!   `/mint_e_token/` path.
//!
//! The full ledger serializes via [`wire::encode`]/[`wire::decode`]; a single
//! `/mint_e_token/` payload serializes via [`wire::encode_entry`]/
//! [`wire::decode_entry`].
//!
//! Modules:
//!   * [`ledger`] — the in-memory dynamic ledger (external / internal / dynamic).
//!   * [`wire`] — versioned binary wire format for the full ledger and single
//!     e_token entry.
//!
//! @module ternary-ledger

pub mod ledger;
pub mod wire;

pub use ledger::{GeneratedToken, Ledger, LedgerConfig, TernaryCode};
pub use wire::{decode, decode_entry, encode, encode_entry};
