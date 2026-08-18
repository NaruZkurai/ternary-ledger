# ternary-ledger

A Rust module for **dynamic compressed-token sequences** for the pi harness.

Instead of sending an `n_window`-long raw token sequence, the harness mints a
custom token (id `>= n_vocab`) whose `ternary_code` the GPU can decode back into
that window **on the fly** during KV compute. This shortens the on-the-wire
context — the harness sends one ledger-backed id, not the whole chunk.

The ledger is *dynamic*: patterns are minted as they recur, viabilities update,
and low-viability / stale entries get pruned. It is fully serializable so the
**current ledger can be sent to / received from a llama.cpp endpoint at any
time** through the compact binary wire format in `src/wire.rs`.

This is the clean Rust replacement for the fork's imperfect C++ `tools/tcomp`
experiment (which had no server wiring, no sampler hook, and no defined wire
format).

## Crate layout

```
src/lib.rs       public API re-exports
src/ledger.rs    TernaryCode, GeneratedToken, Ledger, LedgerConfig
src/wire.rs      binary encode/decode (send/recv the whole ledger)
examples/ledger_http.rs  minimal pure-std HTTP endpoint demo
```

## Core types

- `TernaryCode { v: Vec<i8> }` — `n_code` values, each in `{-1, 0, +1}`.
- `GeneratedToken { id, pattern, code, frequency, viability }` — one minted
  entry. `id >= n_vocab`, `pattern` is the raw window it decodes to.
- `LedgerConfig { n_vocab, n_window, n_code, min_viability, max_entries }`.
- `Ledger` — `new`, `len`, `is_empty`, `register_pattern`, `lookup_pattern`,
  `set_entry`, `entries`, `add_raw`, `evict_lowest_viability`.

## Wire format (`src/wire.rs`)

Magic + version + config + entries, with LEB128 varints for counts/ints:

| field            | encoding                     |
|------------------|------------------------------|
| `MAGIC`          | `0x4744_4C54` (ASCII `TLDG`) |
| `VERSION`        | `u8` = 1                     |
| `n_vocab`        | varint                       |
| `n_window`       | varint                       |
| `n_code`         | varint                       |
| `max_entries`    | varint                       |
| `min_viability`  | `f32`                        |
| entry count      | varint                       |
| per entry        | varint id, varint pattern len + `u32[]`, code dim + `i8[]`, varint frequency, `f32` viability |
| `TRAILER`        | `0x0000_4C44`                |

`encode(&Ledger) -> Vec<u8>`, `decode(&[u8]) -> Result<Ledger, Error>`.

## Send / receive the ledger over HTTP

`examples/ledger_http.rs` is a dependency-free endpoint on `127.0.0.1:8791`:

```
cargo run --example ledger_http
curl  http://127.0.0.1:8791/ledger -o ledger.bin      # receive
curl -X POST --data-binary @ledger.bin http://127.0.0.1:8791/ledger  # send
```

GET returns the binary ledger; POST replaces the in-memory ledger (validated by
`wire::decode`, `400` on a bad payload). This is the exact contract a llama.cpp
server endpoint should host.

## Roadmap: llama.cpp endpoint + on-the-fly GPU decode

The crate now delivers the RPC side: **mint, prune, serialize, send/recv**. Two
pieces remain, in the C++ fork (`llama-direct-token-input`), and they are the
hard ML parts — do not rush them blindly:

1. **Host the endpoint in the server.** Expose `GET /ledger` / `POST /ledger`
   (and optionally accept minted ids in `POST /v1/chat_pretokenized`) by linking
   this crate (Rust → C ABI) or porting the ledger logic. The fork currently has
   **no** `/ledger` route and **no** tcomp wiring, so this is new surface.

2. **Decode the minted chunk in the GPU/KV path.** This is the open design
   question the subagent flagged as the incomplete core of the user's attempt:
   a minted id `>= n_vocab` cannot enter the real `[0, n_vocab)` token stream.
   Options to resolve:
   - **Expand the effective id space** so `n_vocab..n_vocab+max_entries` are
     legal in the sampler/KV, mapping each back to `pattern` on decode; or
   - **Keep chunks at the KV/ledger level** (separate from the vocab id stream) —
     the harness sends the ledger-backed chunk id, the server expands it to the
     window *before* KV embedding, so the GPU never sees an out-of-range id.

   Either way the "ternary code the GPU decodes on the fly" must become a real
   sampler / KV-embedding hook in the server — it is not there today.

## Tests

```
cargo test   # mints_and_reuses, evicts_below_cap, wire roundtrip
```

## Status

Core + wire + HTTP demo complete and verified end-to-end (GET/POST roundtrip,
115-byte `TLDG` payload). Fork endpoint wiring and GPU decode hook = open design
(see Roadmap).
