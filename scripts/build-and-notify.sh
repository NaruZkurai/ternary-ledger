#!/usr/bin/env bash
# build-and-notify.sh
#
# Builds BOTH pi-harness Rust crates in the background, auto-resolving
# dependencies, and notifies the user when done:
#
#   /nzk/git/pithagoras/rust/ternary-ledger    (ledger / mint_e_token endpoint)
#   /nzk/git/pithagoras/rust/gigatoken-addon   (local tokenizer addon, nightly)
#
# Behavior:
#   * ternary-ledger : cargo fetch (auto-resolve) + cargo build --release
#                      --example ledger_http
#   * gigatoken-addon: ./build.sh (stages vendored gigatoken + builds cdylib)
#   * notifies when done via notify-send (desktop) + systemd-notify (service):
#       "<crate>: building..." at start, then "built" or "BUILD FAILED"
#   * writes an aggregate status file consumed by the "built vs building" UI:
#       ledger=, ledger_path=, addon=, addon_path=, ts=
#   * logs a journald tag so a background service is observable.
#
# Usage: build-and-notify.sh [status-file]
#   default status-file: /tmp/pi-crates-build-state

set -uo pipefail

TL_DIR="/nzk/git/pithagoras/rust/ternary-ledger"
GTA_DIR="/nzk/git/pithagoras/rust/gigatoken-addon"
STATUS_FILE="${1:-/tmp/pi-crates-build-state}"
TL_BIN_REL="target/release/examples/ledger_http"
GTA_SO_REL="target/release/libgigatoken_addon.so"

APP_NAME="pi-crates"
TAG="pi-crates-build"

write_status() {
  local ledger="$1" tl_path="$2" addon="$3" gta_path="$4"
  local tmp; tmp="$(mktemp)" # atomic write, no torn reads
  printf 'ledger=%s\nledger_path=%s\naddon=%s\naddon_path=%s\nts=%d\n' \
    "$ledger" "$tl_path" "$addon" "$gta_path" "$(date +%s)" > "$tmp"
  mv -f "$tmp" "$STATUS_FILE"
}

notify() {
  local urgency="$1" summary="$2" body="$3"
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -a "$APP_NAME" -u "$urgency" "$summary" "$body" 2>/dev/null || true
  command -v systemd-notify >/dev/null 2>&1 && \
    systemd-notify --status="$summary: $body" 2>/dev/null || true
  logger -t "$TAG" "$summary: $body" 2>/dev/null || true
}

# build_one <name> <dir> <cmd...> then, from the caller, inspect the saved var.
build_one() {
  local name="$1" dir="$2"; shift 2
  notify normal "$name: building..." "resolving + compiling (background)"
  if (cd "$dir" && "$@"); then
    B_RES=ok
  else
    B_RES=fail
  fi
}

main() {
  write_status "building" "" "building" ""

  # --- ternary-ledger: auto-resolve deps then build ---
  build_one "ternary-ledger" "$TL_DIR" bash -c 'cargo fetch >/dev/null 2>&1; cargo build --release --example ledger_http'
  local tl_state="failed" tl_path=""
  if [ "$B_RES" = ok ]; then
    tl_state="built"; tl_path="$TL_DIR/$TL_BIN_REL"
    notify normal "ternary-ledger: built" "$tl_path"
  else
    notify critical "ternary-ledger: BUILD FAILED" "see target/ in $TL_DIR"
  fi

  # --- gigatoken-addon: stage vendored gigatoken + build ---
  build_one "gigatoken-addon" "$GTA_DIR" ./build.sh
  local gta_state="failed" gta_path=""
  if [ "$B_RES" = ok ]; then
    gta_state="built"; gta_path="$GTA_DIR/$GTA_SO_REL"
    notify normal "gigatoken-addon: built" "$gta_path"
  else
    notify critical "gigatoken-addon: BUILD FAILED" "see target/ in $GTA_DIR"
  fi

  write_status "$tl_state" "$tl_path" "$gta_state" "$gta_path"
  logger -t "$TAG" "done: ledger=$tl_state addon=$gta_state" 2>/dev/null || true
}

main "$@"

