#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPELINE_HOME="${PIPELINE_HOME:-$HOME/auto-editing-workflow}"
CONFIG="${CONFIG:-$PROJECT_DIR/config/config.dev.yaml}"

VENV="${VENV:-}"
if [[ -z "$VENV" ]]; then
  for candidate in "$PIPELINE_HOME/.venv" "$PROJECT_DIR/.venv"; do
    if [[ -x "$candidate/bin/python" ]]; then VENV="$candidate"; break; fi
  done
fi
if [[ -z "$VENV" || ! -x "$VENV/bin/python" ]]; then
  echo "virtual environment not found (looked in $PIPELINE_HOME/.venv and $PROJECT_DIR/.venv)" >&2
  echo "create it with: python3 -m venv $PIPELINE_HOME/.venv && $PIPELINE_HOME/.venv/bin/pip install -r $PROJECT_DIR/requirements.txt" >&2
  exit 1
fi

cd "$PROJECT_DIR"
COMMAND="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi
exec "$VENV/bin/python" -m photo_pipeline "$COMMAND" --dev --config "$CONFIG" "$@"
