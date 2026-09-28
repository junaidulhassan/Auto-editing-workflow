#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

HOST="${FTP_HOST:-127.0.0.1}"
PORT="${FTP_PORT:-21}"
USER_NAME="${FTP_USER:-camera}"
REMOTE_DIR="/"
JOBS=1
GENERATE=0
REPEAT=1
TMP_RENAME=0
FTPS=0
LOCAL_DIR=""

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options] [FILE|DIR ...]

Uploads photos to the local FTP server exactly like a camera would.
With no files it uses tests/fixtures/* or generates synthetic JPEGs.

  -H HOST          FTP host (default: \$FTP_HOST or 127.0.0.1)
  -P PORT          FTP port (default: \$FTP_PORT or 21)
  -u USER          FTP user (default: \$FTP_USER or camera)
  -r DIR           remote folder inside the chroot (default: /)
  -j N             parallel uploads to simulate a burst (default: 1)
  -g N             generate N synthetic 6000x4000 JPEGs with ImageMagick
  -n N             repeat the upload N times with unique names (default: 1)
  --tmp-rename     upload as NAME.tmp and rename when done (some cameras do this)
  --ftps           use explicit FTP over TLS
  --local DIR      skip FTP and drop the files into DIR (atomic rename)
  -h               help

Password: \$FTP_PASSWORD (from the environment or .env) or an interactive prompt.
USAGE
}

FILES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -H) HOST="$2"; shift 2 ;;
    -P) PORT="$2"; shift 2 ;;
    -u) USER_NAME="$2"; shift 2 ;;
    -r) REMOTE_DIR="$2"; shift 2 ;;
    -j) JOBS="$2"; shift 2 ;;
    -g) GENERATE="$2"; shift 2 ;;
    -n) REPEAT="$2"; shift 2 ;;
    --tmp-rename) TMP_RENAME=1; shift ;;
    --ftps) FTPS=1; shift ;;
    --local) LOCAL_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *) FILES+=("$1"); shift ;;
  esac
done

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/camera-sim.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT

collect() {
  local item="$1"
  if [[ -d "$item" ]]; then
    find "$item" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.cr2' -o -iname '*.cr3' -o -iname '*.nef' -o -iname '*.arw' -o -iname '*.raf' -o -iname '*.dng' \) | sort
  elif [[ -f "$item" ]]; then
    echo "$item"
  else
    echo "not found: $item" >&2
  fi
}

SOURCES=()
for item in "${FILES[@]}"; do
  while IFS= read -r f; do [[ -n "$f" ]] && SOURCES+=("$f"); done < <(collect "$item")
done

if [[ $GENERATE -gt 0 ]]; then
  IM="$(command -v magick || command -v convert || true)"
  [[ -n "$IM" ]] || { echo "ImageMagick is required for -g" >&2; exit 1; }
  for i in $(seq 1 "$GENERATE"); do
    out="$STAGE/SIM_$(date +%H%M%S)_$i.jpg"
    "$IM" -size 1500x1000 "plasma:fractal" -blur 0x1 -resize 6000x4000! -quality 92 "$out"
    SOURCES+=("$out")
  done
fi

if [[ ${#SOURCES[@]} -eq 0 && -d "$PROJECT_DIR/tests/fixtures" ]]; then
  while IFS= read -r f; do [[ -n "$f" ]] && SOURCES+=("$f"); done < <(collect "$PROJECT_DIR/tests/fixtures")
fi
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "no photos to upload: pass files, use -g N, or run scripts/fetch_test_images.sh" >&2
  exit 1
fi

UPLOADS=()
for round in $(seq 1 "$REPEAT"); do
  for src in "${SOURCES[@]}"; do
    base="$(basename "$src")"
    stem="${base%.*}"
    ext="${base##*.}"
    name="${stem}_$(date +%Y%m%d%H%M%S)_${round}_$RANDOM.${ext}"
    cp "$src" "$STAGE/$name"
    UPLOADS+=("$STAGE/$name")
  done
done

echo "uploading ${#UPLOADS[@]} photo(s)"
START=$(date +%s%N)

if [[ -n "$LOCAL_DIR" ]]; then
  mkdir -p "$LOCAL_DIR"
  for f in "${UPLOADS[@]}"; do
    name="$(basename "$f")"
    cp "$f" "$LOCAL_DIR/.$name.part"
    mv "$LOCAL_DIR/.$name.part" "$LOCAL_DIR/$name"
    echo "  dropped $LOCAL_DIR/$name"
  done
else
  command -v lftp >/dev/null || { echo "lftp is required (sudo apt install lftp)" >&2; exit 1; }
  if [[ -z "${FTP_PASSWORD:-}" ]]; then
    read -r -s -p "FTP password for $USER_NAME@$HOST: " FTP_PASSWORD
    echo
  fi
  export LFTP_PASSWORD="$FTP_PASSWORD"
  SCRIPT="$STAGE/.lftp"
  {
    echo "set cmd:fail-exit true"
    echo "set net:max-retries 2"
    echo "set net:timeout 20"
    echo "set net:reconnect-interval-base 2"
    echo "set ftp:passive-mode true"
    if [[ $FTPS -eq 1 ]]; then
      echo "set ftp:ssl-force true"
      echo "set ftp:ssl-protect-data true"
      echo "set ssl:verify-certificate no"
    else
      echo "set ftp:ssl-allow false"
    fi
    echo "open --env-password -u \"$USER_NAME\" -p $PORT $HOST"
    echo "cd \"$REMOTE_DIR\""
    if [[ $TMP_RENAME -eq 1 ]]; then
      for f in "${UPLOADS[@]}"; do
        name="$(basename "$f")"
        echo "put \"$f\" -o \"$name.tmp\""
        echo "mv \"$name.tmp\" \"$name\""
      done
    else
      printf 'mput -P %s' "$JOBS"
      for f in "${UPLOADS[@]}"; do printf ' "%s"' "$f"; done
      echo
    fi
    echo "bye"
  } > "$SCRIPT"
  lftp -f "$SCRIPT"
fi

END=$(date +%s%N)
awk -v s="$START" -v e="$END" 'BEGIN { printf "done in %.2fs; watch the pipeline log to see processing\n", (e - s) / 1e9 }'
