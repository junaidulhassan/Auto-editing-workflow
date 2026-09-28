#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$PROJECT_DIR/tests/fixtures}"
mkdir -p "$DEST"

download() {
  local url="$1" name="$2"
  if [[ -s "$DEST/$name" ]]; then
    echo "ok       $DEST/$name"
    return
  fi
  echo "fetch    $url"
  curl -fL --retry 3 --connect-timeout 20 -o "$DEST/$name.tmp" "$url"
  mv -f "$DEST/$name.tmp" "$DEST/$name"
}

download "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg" "face.jpg"
download "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/messi5.jpg" "small_face.jpg"
download "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/cv/cascadeandhog/images/addams-family.png" "group.png"
echo "test images stored in $DEST (add your own side-profile photo as $DEST/profile.jpg to test real profiles)"
