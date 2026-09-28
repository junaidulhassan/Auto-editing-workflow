#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FTP_USER="camera"
FTP_ROOT="/srv/photos/incoming"
FTP_GROUP="photos"
DIR_OWNER=""
LISTEN_PORT=21
PASV_MIN=40000
PASV_MAX=40100
PASV_ADDRESS=""
ALLOW_FROM=""
USE_UFW=1
USE_FTPS=0
DEV_USER=""

usage() {
  cat <<USAGE
Usage: sudo $(basename "$0") [options]

Installs and configures vsftpd so a camera can upload into the pipeline's incoming folder.
  --user NAME          FTP login for the camera (default: camera)
  --root DIR           upload folder / chroot (default: /srv/photos/incoming)
  --group NAME         shared group with the pipeline (default: photos)
  --owner NAME         owner of the upload folder (default: photopipe if it exists, else root)
  --port N             control port (default: 21)
  --pasv-min N         passive range start (default: 40000)
  --pasv-max N         passive range end (default: 40100)
  --pasv-address IP    public/LAN IP to advertise in passive mode (default: auto)
  --allow-from CIDR    only allow FTP from this network (default: detected LAN subnet)
  --dev-user NAME      add a developer account to the shared group (dev mode)
  --ftps               enable explicit FTPS (TLS) with a self-signed certificate
  --no-ufw             do not touch the firewall
Password: \$FTP_PASSWORD, or a random one is generated and printed once.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) FTP_USER="$2"; shift 2 ;;
    --root) FTP_ROOT="$2"; shift 2 ;;
    --group) FTP_GROUP="$2"; shift 2 ;;
    --owner) DIR_OWNER="$2"; shift 2 ;;
    --port) LISTEN_PORT="$2"; shift 2 ;;
    --pasv-min) PASV_MIN="$2"; shift 2 ;;
    --pasv-max) PASV_MAX="$2"; shift 2 ;;
    --pasv-address) PASV_ADDRESS="$2"; shift 2 ;;
    --allow-from) ALLOW_FROM="$2"; shift 2 ;;
    --dev-user) DEV_USER="$2"; shift 2 ;;
    --ftps) USE_FTPS=1; shift ;;
    --no-ufw) USE_UFW=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }

export DEBIAN_FRONTEND=noninteractive
if ! command -v vsftpd >/dev/null; then
  apt-get update
  apt-get install -y vsftpd
fi
if [[ $USE_UFW -eq 1 ]] && ! command -v ufw >/dev/null; then
  apt-get install -y ufw
fi

SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -z "$ALLOW_FROM" ]]; then
  IFACE_CIDR="$(ip -o -f inet addr show scope global | awk '{print $4}' | head -n1)"
  if [[ -n "$IFACE_CIDR" ]]; then
    ALLOW_FROM="$(python3 -c 'import ipaddress,sys; print(ipaddress.ip_interface(sys.argv[1]).network)' "$IFACE_CIDR")"
  fi
fi

groupadd -f "$FTP_GROUP"
if [[ -z "$DIR_OWNER" ]]; then
  if id photopipe >/dev/null 2>&1; then DIR_OWNER="photopipe"; else DIR_OWNER="root"; fi
fi

NOLOGIN="$(command -v nologin || echo /usr/sbin/nologin)"
if id "$FTP_USER" >/dev/null 2>&1; then
  usermod -d "$FTP_ROOT" -g "$FTP_GROUP" -s "$NOLOGIN" "$FTP_USER"
else
  useradd -M -d "$FTP_ROOT" -g "$FTP_GROUP" -s "$NOLOGIN" -c "Camera FTP upload" "$FTP_USER"
fi

GENERATED=0
if [[ -z "${FTP_PASSWORD:-}" ]]; then
  FTP_PASSWORD="$(openssl rand -base64 24 | tr -dc 'A-Za-z0-9' | head -c 20)"
  GENERATED=1
fi
echo "$FTP_USER:$FTP_PASSWORD" | chpasswd

mkdir -p "$FTP_ROOT"
chown "$DIR_OWNER:$FTP_GROUP" "$FTP_ROOT"
chmod 2775 "$FTP_ROOT"
if [[ -n "$DEV_USER" ]]; then
  usermod -aG "$FTP_GROUP" "$DEV_USER"
  echo "added $DEV_USER to group $FTP_GROUP (log out and back in for it to apply)"
fi

check_dir="$(dirname "$FTP_ROOT")"
while [[ "$check_dir" != "/" ]]; do
  if ! sudo -u "$FTP_USER" test -x "$check_dir" 2>/dev/null; then
    echo "warning: $FTP_USER cannot traverse $check_dir; uploads will fail until it is accessible (chmod o+x or use /srv)" >&2
  fi
  check_dir="$(dirname "$check_dir")"
done

echo "$FTP_USER" > /etc/vsftpd.photos.userlist
chmod 644 /etc/vsftpd.photos.userlist
install -m 644 "$PROJECT_DIR/ftp/vsftpd-photos.pam" /etc/pam.d/vsftpd-photos

SSL_BLOCK="ssl_enable=NO"
if [[ $USE_FTPS -eq 1 ]]; then
  CERT=/etc/ssl/certs/vsftpd-photos.pem
  KEY=/etc/ssl/private/vsftpd-photos.key
  if [[ ! -f "$CERT" || ! -f "$KEY" ]]; then
    openssl req -x509 -nodes -newkey rsa:2048 -days 3650 -subj "/CN=${SERVER_IP:-photo-pipeline}" -keyout "$KEY" -out "$CERT"
    chmod 600 "$KEY"
  fi
  SSL_BLOCK="ssl_enable=YES
rsa_cert_file=$CERT
rsa_private_key_file=$KEY
allow_anon_ssl=NO
force_local_logins_ssl=YES
force_local_data_ssl=YES
require_ssl_reuse=NO
ssl_ciphers=HIGH"
fi

PASV_LINE=""
if [[ -n "$PASV_ADDRESS" ]]; then
  PASV_LINE="pasv_address=$PASV_ADDRESS"
fi

if [[ -f /etc/vsftpd.conf ]]; then
  cp /etc/vsftpd.conf "/etc/vsftpd.conf.bak.$(date +%Y%m%d%H%M%S)"
fi
LISTEN_PORT="$LISTEN_PORT" PASV_MIN="$PASV_MIN" PASV_MAX="$PASV_MAX" PASV_LINE="$PASV_LINE" SSL_BLOCK="$SSL_BLOCK" \
  python3 - "$PROJECT_DIR/ftp/vsftpd.conf.template" /etc/vsftpd.conf <<'PY'
import os
import sys

template, target = sys.argv[1], sys.argv[2]
text = open(template).read()
values = {
    "@LISTEN_PORT@": os.environ["LISTEN_PORT"],
    "@PASV_MIN@": os.environ["PASV_MIN"],
    "@PASV_MAX@": os.environ["PASV_MAX"],
    "@PASV_ADDRESS_LINE@": os.environ["PASV_LINE"],
    "@SSL_BLOCK@": os.environ["SSL_BLOCK"],
}
for key, value in values.items():
    text = text.replace(key, value)
with open(target, "w") as handle:
    handle.write("\n".join(line for line in text.splitlines() if line.strip()) + "\n")
PY
chmod 644 /etc/vsftpd.conf
mkdir -p /var/run/vsftpd/empty

systemctl enable vsftpd >/dev/null
systemctl restart vsftpd
systemctl --no-pager --lines=0 status vsftpd || true

if [[ $USE_UFW -eq 1 ]]; then
  SOURCE="${ALLOW_FROM:-any}"
  ufw allow OpenSSH >/dev/null
  if [[ "$SOURCE" == "any" ]]; then
    ufw allow "$LISTEN_PORT/tcp"
    ufw allow "$PASV_MIN:$PASV_MAX/tcp"
  else
    ufw allow from "$SOURCE" to any port "$LISTEN_PORT" proto tcp
    ufw allow from "$SOURCE" to any port "$PASV_MIN:$PASV_MAX" proto tcp
  fi
  ufw --force enable
  ufw status numbered
fi

cat <<INFO

================ Camera FTP settings ================
Server / host     : ${PASV_ADDRESS:-$SERVER_IP}
Port              : $LISTEN_PORT
Protocol          : $( [[ $USE_FTPS -eq 1 ]] && echo "FTPS (explicit TLS)" || echo "FTP" )
Username          : $FTP_USER
Password          : $( [[ $GENERATED -eq 1 ]] && echo "$FTP_PASSWORD   (generated; store it now)" || echo "(the FTP_PASSWORD you provided)" )
Passive mode      : ON (ports $PASV_MIN-$PASV_MAX)
Target folder     : /   (the camera is jailed in $FTP_ROOT)
Allowed from      : ${ALLOW_FROM:-any}
=====================================================
Test: FTP_PASSWORD='...' $PROJECT_DIR/scripts/simulate_camera.sh -H 127.0.0.1 -u $FTP_USER
INFO
