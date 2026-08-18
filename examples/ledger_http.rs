//! End-to-end demo: mint compressed tokens, send/recv the ledger over a
//! minimal pure-std HTTP endpoint, and run the unknown-etoken handshake.
//!
//! Endpoints a llama.cpp server would host:
//!   GET  /ledger           -> binary ledger bytes ("receive the current ledger")
//!   POST /ledger           -> upload a ledger, replaces the in-memory one ("send")
//!   GET  /check/{id}       -> "recognized" or "unknown etoken in ledger"
//!   POST /mint_e_token/    -> mint an etoken on the fly from {eid: {oid..., formula}}
//!
//! Run:  cargo run --example ledger_http
//! Curl:
//!   curl http://127.0.0.1:8791/ledger -o ledger.bin
//!   curl -X POST --data-binary @ledger.bin http://127.0.0.1:8791/ledger
//!   curl http://127.0.0.1:8791/check/400000          # unknown etoken in ledger
//!   printf '\x20\x0c\x00\x00...' | curl -X POST --data-binary @- http://127.0.0.1:8791/mint_e_token/
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use ternary_ledger::{Ledger, LedgerConfig, TernaryCode, wire};

const ADDR: &str = "0.0.0.0:8791";

fn read_request(stream: &mut TcpStream) -> (String, String, Vec<u8>) {
    let mut raw = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if raw.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let head = String::from_utf8_lossy(&raw);
    let first = head.lines().next().unwrap_or("");
    let method = first.split_whitespace().next().unwrap_or("").to_string();
    let path = first.split_whitespace().nth(1).unwrap_or("").to_string();
    let body_start = head.find("\r\n\r\n").map(|i| i + 4).unwrap_or(raw.len());
    let body = raw[body_start..].to_vec();
    (method, path, body)
}

fn respond(stream: &mut TcpStream, status: &str, body: &[u8]) {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let mut out = head.into_bytes();
    out.extend_from_slice(body);
    let _ = stream.write_all(&out);
}

fn handle(led: Arc<Mutex<Ledger>>, mut stream: TcpStream) {
    let (method, path, body) = read_request(&mut stream);

    if method == "GET" && path == "/ledger" {
        let bytes = { let l = led.lock().unwrap(); wire::encode(&l) };
        respond(&mut stream, "200 OK", &bytes);
        return;
    }
    if method == "POST" && path == "/ledger" {
        match wire::decode(&body) {
            Ok(incoming) => {
                let mut l = led.lock().unwrap();
                *l = incoming;
                respond(&mut stream, "200 OK", b"ok");
            }
            Err(e) => respond(&mut stream, "400 Bad Request", format!("bad ledger: {e}").as_bytes()),
        }
        return;
    }
    // GET /check/{id} — recognition check.
    if method == "GET" && path.starts_with("/check/") {
        let id: u32 = path.trim_start_matches("/check/").parse().unwrap_or(0);
        let known = { let l = led.lock().unwrap(); l.contains(id) };
        if known {
            respond(&mut stream, "200 OK", b"recognized");
        } else {
            respond(&mut stream, "404 Not Found", b"unknown etoken in ledger");
        }
        return;
    }
    // POST /mint_e_token/ — mint an externally-chosen etoken on the fly.
    if method == "POST" && path.starts_with("/mint_e_token") {
        match wire::decode_entry(&body) {
            Ok(t) => {
                let minted = {
                    let mut l = led.lock().unwrap();
                    l.mint_external(t.id, t.pattern.clone(), t.code.clone())
                };
                if minted {
                    respond(&mut stream, "200 OK", b"minted");
                } else {
                    respond(&mut stream, "409 Conflict", b"eid already minted or below n_vocab");
                }
            }
            Err(e) => respond(&mut stream, "400 Bad Request", format!("bad entry: {e}").as_bytes()),
        }
        return;
    }
    respond(&mut stream, "404 Not Found", b"not found");
}

fn main() {
    // Build a small ledger via the internal path.
    let mut led = Ledger::new(LedgerConfig { n_vocab: 1000, n_window: 4, n_code: 8, ..LedgerConfig::default() });
    for w in [vec![10, 20, 30, 40], vec![1, 1, 2, 2], vec![5, 6, 7, 8]] {
        led.mint_pattern(&w, TernaryCode::new(vec![1, -1, 0, 1, 0, -1, 1, 0]), 0.81);
    }
    let shared = Arc::new(Mutex::new(led));
    let count = shared.lock().unwrap().len();
    let listener = TcpListener::bind(ADDR).expect("bind");
    println!("ledger endpoint listening on http://{ADDR}/ledger  ({count} minted tokens)");
    for stream in listener.incoming() {
        if let Ok(s) = stream {
            handle(shared.clone(), s);
        }
    }
}

