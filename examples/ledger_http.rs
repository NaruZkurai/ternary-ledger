//! End-to-end demo: mint compressed tokens, then send/recv the ledger over a
//! minimal pure-std HTTP endpoint.
//!
//! This demonstrates the wire path a llama.cpp endpoint would host:
//!   GET  /ledger  -> binary ledger bytes  ("receive the current ledger")
//!   POST /ledger  -> upload a ledger, replaces the in-memory one ("send")
//!
//! Run:  cargo run --example ledger_http
//! Then curl:
//!   curl http://127.0.0.1:8791/ledger -o ledger.bin
//!   curl -X POST --data-binary @ledger.bin http://127.0.0.1:8791/ledger
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use ternary_ledger::{Ledger, LedgerConfig, TernaryCode, wire};

const ADDR: &str = "127.0.0.1:8791";

fn handle(led: Arc<Mutex<Ledger>>, mut stream: TcpStream) {
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
    let method = first.split_whitespace().next().unwrap_or("");
    let path = first.split_whitespace().nth(1).unwrap_or("");
    let body_start = head.find("\r\n\r\n").map(|i| i + 4).unwrap_or(raw.len());

    let mut response = Vec::new();
    let mut status = "200 OK";
    match (method, path) {
        ("GET", "/ledger") => {
            let bytes = {
                let l = led.lock().unwrap();
                wire::encode(&l)
            };
            response.extend_from_slice(&format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/octet-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                bytes.len()
            ).as_bytes());
            response.extend_from_slice(&bytes);
        }
        ("POST", "/ledger") => {
            let body = &raw[body_start..];
            match wire::decode(body) {
                Ok(incoming) => {
                    let mut l = led.lock().unwrap();
                    *l = incoming;
                    response.extend_from_slice(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
                }
                Err(e) => {
                    status = "400 Bad Request";
                    let msg = format!("bad ledger: {e}");
                    response.extend_from_slice(&format!(
                        "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{msg}",
                        msg.len()
                    ).as_bytes());
                }
            }
        }
        _ => {
            response.extend_from_slice(b"HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found");
        }
    }
    let _ = stream.write_all(&response);
}

fn main() {
    // Build a small dynamic ledger.
    let mut led = Ledger::new(LedgerConfig { n_vocab: 1000, n_window: 4, n_code: 8, ..LedgerConfig::default() });
    for w in [vec![10, 20, 30, 40], vec![1, 1, 2, 2], vec![5, 6, 7, 8]] {
        let id = led.register_pattern(&w);
        led.set_entry(id, TernaryCode::new(vec![1, -1, 0, 1, 0, -1, 1, 0]), 0.81);
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
