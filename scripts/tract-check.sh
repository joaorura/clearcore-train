#!/usr/bin/env bash
# Build/test/run do tools/tract-check com a toolchain persistente (Rust 1.90.0 em ~/.rustup).
# O target fica no checkout principal (compartilhado entre worktrees), nunca em /tmp.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT="${CCTRAIN_ROOT:-$HERE}"
# Toolchain fixada por variável: vale em qualquer cwd (o rust-toolchain.toml só vale dentro de tools/tract-check).
export RUSTUP_TOOLCHAIN=1.90.0
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$CARGO_HOME/bin:$PATH"
export CARGO_TARGET_DIR="$TARGET_ROOT/tools/tract-check/target"
for d in "$RUSTUP_HOME" "$CARGO_HOME" "$CARGO_TARGET_DIR"; do
  case "$d" in /tmp/*) echo "ERRO: $d em /tmp" >&2; exit 1;; esac
done
mkdir -p "$CARGO_TARGET_DIR"
fstype="$(findmnt -no FSTYPE --target "$CARGO_TARGET_DIR")"
[ "$fstype" != "tmpfs" ] || { echo "ERRO: CARGO_TARGET_DIR em tmpfs" >&2; exit 1; }
# O commit fixado do Clearcore ainda não está publicado: a URL pública é redirecionada ao checkout local.
export CARGO_NET_GIT_FETCH_WITH_CLI=true
if [ -n "${CCTRAIN_CLEARCORE_ROOT:-}" ]; then
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${CCTRAIN_CLEARCORE_ROOT}.insteadOf"
  export GIT_CONFIG_VALUE_0="https://github.com/joaorura/clearcore.git"
fi
MANIFEST="$HERE/tools/tract-check/Cargo.toml"
cmd="${1:-}"; shift || true
case "$cmd" in
  build|test)
    if [ ! -x "$CARGO_TARGET_DIR/release/tract-check" ]; then
      avail_kb=$(df --output=avail -k "$CARGO_TARGET_DIR" | tail -1)
      [ "$avail_kb" -ge $((14 * 1024 * 1024)) ] || { echo "ERRO: < 14 GiB livres (target ~3-4 GiB + reserva 10 GiB)" >&2; exit 1; }
    fi
    lock_flag="--locked"; [ -f "$HERE/tools/tract-check/Cargo.lock" ] || lock_flag=""
    cargo "$cmd" --release $lock_flag --manifest-path "$MANIFEST" "$@"
    ;;
  run) exec "$CARGO_TARGET_DIR/release/tract-check" "$@" ;;
  *) echo "uso: scripts/tract-check.sh build|test|run [args]" >&2; exit 2 ;;
esac
