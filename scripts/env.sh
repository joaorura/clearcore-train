# shellcheck shell=bash
# Ambiente comum das tarefas: source "$HOME/orca/projects/clearcore-train/scripts/env.sh"
# Sem caminhos pessoais: tudo deriva de $HOME ou de variáveis exportadas pelo orquestrador.
export CCTRAIN_ROOT="${CCTRAIN_ROOT:-$HOME/orca/projects/clearcore-train}"
export CCTRAIN_WT="${CCTRAIN_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$CCTRAIN_WT/build"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$CCTRAIN_ROOT/.venv}"
export CCTRAIN_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export CCTRAIN_TRACT_CHECK_BIN="${CCTRAIN_TRACT_CHECK_BIN:-$CCTRAIN_ROOT/tools/tract-check/target/release/tract-check}"
export CCTRAIN_DATA_DIR="${CCTRAIN_DATA_DIR:-$CCTRAIN_ROOT/data}"
export CCTRAIN_RUNS_DIR="${CCTRAIN_RUNS_DIR:-$CCTRAIN_ROOT/runs}"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PYTHONPATH="$CCTRAIN_WT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
if [ -z "${CCTRAIN_CLEARCORE_ROOT:-}" ]; then
  echo "AVISO: CCTRAIN_CLEARCORE_ROOT não definido (checkout do Clearcore com o commit 70dc85c)" >&2
fi
