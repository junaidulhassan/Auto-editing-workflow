#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

candidates=(
  "${PYTHON:-}"
  "${CONDA_PREFIX:-}/bin/python"
  "$HOME/miniconda3/envs/photo-env/bin/python"
  "$HOME/auto-editing-workflow/.venv/bin/python"
  "$PROJECT_DIR/.venv/bin/python"
  "$(command -v python3 || true)"
)
PY=""
for candidate in "${candidates[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c "import cv2, watchdog, yaml, PIL, numpy" 2>/dev/null; then
    PY="$candidate"
    break
  fi
done
if [[ -z "$PY" ]]; then
  echo "No Python with the project requirements found." >&2
  echo "Run: conda activate photo-env && pip install -r $PROJECT_DIR/requirements.txt" >&2
  exit 1
fi
echo "using $PY"
exec "$PY" -m photo_pipeline run --dev --config "$PROJECT_DIR/config/config.local-test.yaml" "$@"
