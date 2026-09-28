#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_ROOT=/opt/photo-pipeline
APP_DIR="$APP_ROOT/app"
VENV_DIR="$APP_ROOT/venv"
MODELS_DIR="$APP_ROOT/models"
CONFIG_DIR=/etc/photo-pipeline
STATE_DIR=/var/lib/photo-pipeline
LOG_DIR=/var/log/photo-pipeline
PHOTOS_ROOT=/srv/photos
GALLERY_DIR=/var/www/gallery/photos
SERVICE_USER=photopipe
SERVICE_GROUP=photos

SKIP_APT=0
SKIP_PIP=0
SKIP_MODELS=0
WITH_MEDIAPIPE=0
ENABLE=1

usage() {
  cat <<USAGE
Usage: sudo $(basename "$0") [--skip-apt] [--skip-pip] [--skip-models] [--with-mediapipe] [--no-enable]

Production install: system packages, service user, /srv/photos folders, venv in $VENV_DIR,
offline models, config in $CONFIG_DIR and the systemd unit. Run scripts/setup_ftp.sh afterwards.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt) SKIP_APT=1; shift ;;
    --skip-pip) SKIP_PIP=1; shift ;;
    --skip-models) SKIP_MODELS=1; shift ;;
    --with-mediapipe) WITH_MEDIAPIPE=1; shift ;;
    --no-enable) ENABLE=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
export DEBIAN_FRONTEND=noninteractive

if [[ $SKIP_APT -eq 0 ]]; then
  apt-get update
  mapfile -t PACKAGES < <(grep -Ev '^\s*(#|$)' "$PROJECT_DIR/apt-packages.txt")
  apt-get install -y "${PACKAGES[@]}"
fi

PYTHON=""
for candidate in python3.13 python3.12 python3.11; do
  if command -v "$candidate" >/dev/null; then PYTHON="$(command -v "$candidate")"; break; fi
done
if [[ -z "$PYTHON" ]]; then
  if [[ $SKIP_APT -eq 0 ]]; then
    apt-get install -y python3.11 python3.11-venv python3.11-dev
    PYTHON="$(command -v python3.11)"
  else
    echo "Python 3.11+ not found (Ubuntu 22.04: apt install python3.11 python3.11-venv)" >&2
    exit 1
  fi
fi
echo "using $PYTHON"

groupadd -f "$SERVICE_GROUP"
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --home-dir "$STATE_DIR" --shell /usr/sbin/nologin -g "$SERVICE_GROUP" "$SERVICE_USER"
fi
usermod -aG "$SERVICE_GROUP" "$SERVICE_USER"

install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 2775 "$PHOTOS_ROOT"
for sub in incoming originals processed failed work; do
  install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 2775 "$PHOTOS_ROOT/$sub"
done
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 0750 "$STATE_DIR" "$LOG_DIR"
install -d -o "$SERVICE_USER" -g "$SERVICE_GROUP" -m 2775 "$GALLERY_DIR"
chmod o+rx /var/www /var/www/gallery "$GALLERY_DIR" 2>/dev/null || true
install -d -o root -g "$SERVICE_GROUP" -m 0750 "$CONFIG_DIR"
install -d -o root -g root -m 0755 "$APP_ROOT" "$MODELS_DIR"

rsync -a --delete \
  --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'models' \
  --exclude 'tests/fixtures' --exclude '__pycache__' --exclude '.env' \
  "$PROJECT_DIR/" "$APP_DIR/"
chown -R root:root "$APP_DIR"
chmod -R go-w "$APP_DIR"

if [[ $SKIP_PIP -eq 0 ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  "$VENV_DIR/bin/pip" install --upgrade pip wheel
  "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
  if [[ $WITH_MEDIAPIPE -eq 1 ]]; then
    "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements-optional.txt" || echo "warning: mediapipe install failed; landmark refinement stays off" >&2
  fi
fi

if [[ $SKIP_MODELS -eq 0 ]]; then
  MODEL_ARGS=(--dir "$MODELS_DIR")
  [[ $WITH_MEDIAPIPE -eq 1 ]] && MODEL_ARGS+=(--with-mediapipe)
  "$APP_DIR/scripts/download_models.sh" "${MODEL_ARGS[@]}"
fi

if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
  install -o root -g "$SERVICE_GROUP" -m 0640 "$APP_DIR/config/config.yaml" "$CONFIG_DIR/config.yaml"
  echo "installed $CONFIG_DIR/config.yaml"
else
  echo "kept existing $CONFIG_DIR/config.yaml"
fi
if [[ ! -f "$CONFIG_DIR/photo-pipeline.env" ]]; then
  grep -Ev '^(PIPELINE_HOME|PIPELINE_INCOMING|FTP_)' "$APP_DIR/.env.example" > "$CONFIG_DIR/photo-pipeline.env"
  chown root:"$SERVICE_GROUP" "$CONFIG_DIR/photo-pipeline.env"
  chmod 0640 "$CONFIG_DIR/photo-pipeline.env"
  echo "installed $CONFIG_DIR/photo-pipeline.env (put publish credentials here)"
fi

install -m 0644 "$APP_DIR/systemd/90-photo-pipeline.conf" /etc/sysctl.d/90-photo-pipeline.conf
sysctl --system >/dev/null

install -m 0644 "$APP_DIR/systemd/photo-pipeline.service" /etc/systemd/system/photo-pipeline.service
systemctl daemon-reload

(cd "$APP_DIR" && sudo -u "$SERVICE_USER" "$VENV_DIR/bin/python" -m photo_pipeline check --config "$CONFIG_DIR/config.yaml") || true

if [[ $ENABLE -eq 1 ]]; then
  systemctl enable --now photo-pipeline.service
  systemctl --no-pager status photo-pipeline.service || true
fi

cat <<INFO

Installed. Next steps:
  1. sudo $APP_DIR/scripts/setup_ftp.sh            (camera FTP account, firewall)
  2. edit $CONFIG_DIR/config.yaml and $CONFIG_DIR/photo-pipeline.env if needed
  3. sudo systemctl restart photo-pipeline && journalctl -u photo-pipeline -f
INFO
