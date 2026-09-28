#!/usr/bin/env bash
set -euo pipefail

YUNET_NAME="face_detection_yunet_2023mar.onnx"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URLS=(
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
  "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MEDIAPIPE_NAME="face_landmarker.task"
MEDIAPIPE_SHA256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MEDIAPIPE_URLS=(
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--dir DIR] [--with-mediapipe] [--force]

Downloads the offline face models once and verifies their SHA-256 checksums.
  --dir DIR          target folder (default: \$MODELS_DIR or /opt/photo-pipeline/models)
  --with-mediapipe   also fetch the optional MediaPipe face landmarker
  --force            re-download even if a verified file exists
USAGE
}

MODELS_DIR="${MODELS_DIR:-/opt/photo-pipeline/models}"
WITH_MEDIAPIPE=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir) MODELS_DIR="$2"; shift 2 ;;
    --with-mediapipe) WITH_MEDIAPIPE=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

command -v curl >/dev/null || { echo "curl is required" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 1; }
mkdir -p "$MODELS_DIR"

verify() {
  local file="$1" expected="$2"
  [[ -f "$file" ]] && [[ "$(sha256sum "$file" | awk '{print $1}')" == "$expected" ]]
}

fetch() {
  local name="$1" expected="$2"; shift 2
  local target="$MODELS_DIR/$name"
  if [[ $FORCE -eq 0 ]] && verify "$target" "$expected"; then
    echo "ok       $target (already present, checksum verified)"
    return 0
  fi
  local tmp
  tmp="$(mktemp "$MODELS_DIR/.${name}.XXXXXX")"
  for url in "$@"; do
    echo "fetch    $url"
    if curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$tmp" "$url"; then
      if verify "$tmp" "$expected"; then
        chmod 0644 "$tmp"
        mv -f "$tmp" "$target"
        echo "ok       $target (sha256 verified)"
        return 0
      fi
      echo "warning  checksum mismatch from $url" >&2
    fi
  done
  rm -f "$tmp"
  echo "error    could not download a verified copy of $name" >&2
  return 1
}

fetch "$YUNET_NAME" "$YUNET_SHA256" "${YUNET_URLS[@]}"
if [[ $WITH_MEDIAPIPE -eq 1 ]]; then
  fetch "$MEDIAPIPE_NAME" "$MEDIAPIPE_SHA256" "${MEDIAPIPE_URLS[@]}"
fi
echo "models stored in $MODELS_DIR; the pipeline now runs fully offline"
