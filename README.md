# Photo Auto-Edit Pipeline

Camera → FTP → auto edit → export → website gallery, fully automated and headless.

A camera uploads photos over FTP to a Linux machine. A background service picks up each photo the moment its upload
finishes (inotify `close_write`), edits it automatically, exports it at the target resolution and publishes it to the
website. Nobody needs to be at the machine.

```
camera ──FTP──> /srv/photos/incoming ──inotify──> queue ──> N workers
                                                              │
      originals/  <── archived untouched ─────────────────────┤
      failed/     <── bad files + .error.txt ─────────────────┤
                                                              ▼
         decode (RAW via RawTherapee/darktable) → resize → white balance → exposure → CLAHE contrast
         → face detect (YuNet) → skin mask → edge-preserving skin smoothing (+ SSIM safety check)
         → JPEG export (sRGB, EXIF oriented) → processed/ → publish (local | rsync/sftp | HTTPS API)
                                                              │
                                    persistent retry queue <──┘ (if the network or site is down)
```

Measured on a normal laptop CPU: a 24 MP JPEG is processed in about **0.9 s**, including face retouching. That breaks
down as decode 50–80 ms, edits about 300 ms, face detection and retouch about 170 ms, and export about 50 ms. RAW files
also pay the RawTherapee or darktable development time.

---

## 1. Project layout

```
auto-editing-workflow/
├── README.md
├── requirements.txt              core Python deps (opencv-python-headless, numpy, pillow, watchdog, pyyaml, requests, pytest)
├── requirements-optional.txt     mediapipe (optional landmark masks)
├── apt-packages.txt              Ubuntu packages
├── .env.example                  credential/env template (copy to .env)
├── pytest.ini
├── config/
│   ├── config.yaml               production config (/srv paths)
│   └── config.dev.yaml           dev config (~/auto-editing-workflow paths, console logging)
├── photo_pipeline/
│   ├── __main__.py               python -m photo_pipeline ...
│   ├── cli.py                    run | process | check | status | retry-publish | init-dirs
│   ├── config.py                 YAML + defaults + ${ENV:-default} expansion + .env + validation
│   ├── logging_setup.py          rotating file log + console (dev) / journald (prod)
│   ├── service.py                wires watcher, queue, workers, publisher; signals; sd_notify
│   ├── watcher.py                inotify close_write / moved_to + startup scan + safety sweep
│   ├── workqueue.py              de-duplicating thread-safe queue + worker pool
│   ├── processor.py              per-photo flow: verify → hash → claim → archive → render → publish
│   ├── validation.py             file filters, size-stability check, JPEG/PNG/RAW integrity checks
│   ├── state.py                  SQLite: processed photos (by SHA-256) + persistent publish queue
│   ├── loader.py                 JPEG/PNG decode, EXIF orientation, ICC → sRGB, RAW loading
│   ├── raw.py                    rawtherapee-cli / darktable-cli wrappers
│   ├── metadata.py               EXIF keep / minimal / strip, GPS + MakerNote stripping
│   ├── exporter.py               resize + atomic JPEG/WebP/PNG export
│   ├── colorspace.py             sRGB/linear LUTs, luma helpers
│   ├── checks.py                 environment self-test
│   ├── editing/
│   │   ├── white_balance.py      gray-world on near-neutral pixels, gain-limited
│   │   ├── exposure.py           percentile auto-levels, soft highlight shoulder, midtone gamma
│   │   ├── contrast.py           CLAHE on L* only
│   │   └── auto_edit.py          step runner
│   ├── faces/
│   │   ├── detector.py           YuNet (cv2.FaceDetectorYN) + thread-safe model pool
│   │   ├── landmarks.py          optional MediaPipe Face Landmarker (478 points)
│   │   ├── geometry.py           face/landmark geometry, profile detection
│   │   ├── masks.py              face region, feature exclusion, adaptive skin model, feathering
│   │   ├── ssim.py               SSIM for the safety check
│   │   └── retouch.py            frequency-separation smoothing with bilateral/guided/gaussian base
│   └── publishers/
│       ├── base.py               Publisher interface + transient/permanent errors
│       ├── local.py              copy to a folder + manifest.json (default)
│       ├── ssh.py                rsync or sftp over SSH with key auth
│       ├── https.py              upload to an existing website/CMS API
│       ├── factory.py            backend selection
│       └── manager.py            inline retries, exponential backoff, persistent queue thread
├── scripts/
│   ├── install.sh                production install (packages, user, /srv, venv, models, systemd)
│   ├── setup_ftp.sh              vsftpd + camera user + chroot + passive ports + ufw
│   ├── download_models.sh        YuNet (+ MediaPipe) download with SHA-256 verification
│   ├── run_dev.sh                run any CLI command in dev mode
│   ├── simulate_camera.sh        upload photos to the local FTP server with lftp (or drop locally)
│   └── fetch_test_images.sh      sample face images for the tests
├── ftp/
│   ├── vsftpd.conf.template
│   └── vsftpd-photos.pam
├── systemd/
│   ├── photo-pipeline.service
│   └── 90-photo-pipeline.conf    inotify sysctl limits
├── models/                       (dev) downloaded models go here or in ~/auto-editing-workflow/models
└── tests/
    ├── conftest.py
    ├── test_config.py            config, env expansion, validation
    ├── test_editing.py           white balance, exposure, contrast
    ├── test_io.py                filters, integrity checks, loader, EXIF, exporter
    ├── test_faces.py             single / none / multiple / small / downscaled / profile faces, thread safety
    ├── test_retouch.py           masks, non-face pixels untouched, skin tone preserved, safety fallback
    ├── test_pipeline.py          end-to-end, duplicates, failures, recovery, outage queue, burst, inotify
    └── test_publishers.py        local, HTTPS (live test server), retry queue, backoff
```

## 2. Assumptions (all configurable)

| Assumption | Default | Where to change |
|---|---|---|
| Website delivery method is unknown | `local` folder copy (`/var/www/gallery/photos`) | `publish.backend` (`local`, `ssh`/`rsync`/`sftp`, `https`) |
| Camera model is unknown | Supports FTP upload in **passive** mode | `scripts/setup_ftp.sh --pasv-*` |
| Output | Long edge 2048 px, JPEG q90, sRGB, EXIF kept (GPS stripped) | `export.*` |
| Workers | 2 parallel workers, 2 OpenCV threads each | `workers.*` |
| Retouching | On, subtle strength 0.35, YuNet ellipse masks | `retouch.*` |
| Network | Only the publish step uses the network | — |

---

## 3. Development setup (home directory, foreground)

Dev mode uses `config/config.dev.yaml`: every path lives under `~/auto-editing-workflow` (override with `PIPELINE_HOME`),
logs go to the console, and publishing copies into `~/auto-editing-workflow/data/gallery`.

```bash
sudo apt update && sudo apt install -y $(grep -v '^#' apt-packages.txt)

# Python env: conda ...
conda create -n photo-env python=3.12 -y && conda activate photo-env
# ... or venv
python3 -m venv ~/auto-editing-workflow/.venv && source ~/auto-editing-workflow/.venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-optional.txt          # optional, only for MediaPipe landmark masks

./scripts/download_models.sh --dir ~/auto-editing-workflow/models   # add --with-mediapipe if wanted
cp .env.example .env                                                 # optional, for credentials
```

Run it:

```bash
# venv in ~/auto-editing-workflow/.venv is found automatically; for conda pass VENV
VENV="$CONDA_PREFIX" ./scripts/run_dev.sh check
VENV="$CONDA_PREFIX" ./scripts/run_dev.sh run

# or directly
python -m photo_pipeline check --dev
python -m photo_pipeline run --dev
```

Drop a photo into `~/auto-editing-workflow/data/incoming/` and watch the console. The results appear in `data/processed/`
and `data/gallery/`, the untouched original goes to `data/originals/YYYY/MM/DD/`, and anything broken goes to
`data/failed/`.

The dev folders:

```
~/auto-editing-workflow/
├── models/        face_detection_yunet_2023mar.onnx
└── data/  incoming/ originals/ processed/ failed/ work/ gallery/ logs/ state.db
```

## 4. Production setup (systemd, /srv paths, 24/7)

```bash
sudo ./scripts/install.sh                   # add --with-mediapipe for landmark masks
sudo /opt/photo-pipeline/app/scripts/setup_ftp.sh
```

`install.sh` does the following:
* Installs the apt packages. On Ubuntu 22.04 it also installs Python 3.11, because 22.04 ships 3.10.
* Creates the `photos` group and the `photopipe` system user.
* Creates `/srv/photos/{incoming,originals,processed,failed,work}` owned by `photopipe:photos` with mode `2775`.
* Copies the app to `/opt/photo-pipeline/app` and builds a venv in `/opt/photo-pipeline/venv`.
* Downloads and verifies the models into `/opt/photo-pipeline/models`.
* Installs `/etc/photo-pipeline/config.yaml` and `/etc/photo-pipeline/photo-pipeline.env`. It never overwrites existing
  copies.
* Raises the inotify limits and installs, enables and starts `photo-pipeline.service`.

Manual service commands:

```bash
sudo cp systemd/photo-pipeline.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now photo-pipeline
sudo systemctl status photo-pipeline
journalctl -u photo-pipeline -f                     # live log (journal)
tail -f /var/log/photo-pipeline/pipeline.log        # rotating file log
sudo systemctl restart photo-pipeline               # safe at any time
```

The unit uses `Restart=always` and `Type=notify`: the service reports ready only after models are loaded and the
incoming folder has been scanned. It runs sandboxed with `ProtectSystem=strict`, so the service can write only to
`/srv/photos`, `/var/lib/photo-pipeline`, `/var/log/photo-pipeline` and `/var/www/gallery`. **If you move the local
publish folder elsewhere, add it to `ReadWritePaths=` in the unit.**

Status and maintenance:

```bash
cd /opt/photo-pipeline/app
sudo -u photopipe /opt/photo-pipeline/venv/bin/python -m photo_pipeline status --config /etc/photo-pipeline/config.yaml
sudo -u photopipe /opt/photo-pipeline/venv/bin/python -m photo_pipeline retry-publish --config /etc/photo-pipeline/config.yaml
sudo -u photopipe /opt/photo-pipeline/venv/bin/python -m photo_pipeline check --config /etc/photo-pipeline/config.yaml
```

---

## 5. FTP server and camera configuration

```bash
sudo ./scripts/setup_ftp.sh                               # FTP, LAN-only firewall, random password
sudo FTP_PASSWORD='choose-one' ./scripts/setup_ftp.sh     # fixed password
sudo ./scripts/setup_ftp.sh --ftps                        # explicit TLS (if the camera supports FTPS)
sudo ./scripts/setup_ftp.sh --allow-from 192.168.1.0/24 --pasv-address 192.168.1.50
```

The script sets up:
* vsftpd with a dedicated local user `camera`. The user has no shell (`nologin`) and is on an allow-list
  (`/etc/vsftpd.photos.userlist`).
* A chroot jail in `/srv/photos/incoming`, so the camera sees `/` and nothing else. Uploads are created group-writable
  (`umask 002`), so the pipeline user can move them.
* A passive port range of 40000–40100.
* ufw rules for 21/tcp and 40000:40100/tcp, allowed only from your LAN subnet. OpenSSH stays allowed.
* A printed block with the camera settings.

Enter these values once in the camera's FTP settings:

| Camera setting | Value |
|---|---|
| Server / host | the machine's LAN IP (`hostname -I`) |
| Port | `21` |
| Protocol | FTP (or FTPS if you used `--ftps`) |
| Username | `camera` |
| Password | printed by `setup_ftp.sh` (or your `FTP_PASSWORD`) |
| Passive mode | **ON** |
| Target folder / directory | `/` |
| Auto-transfer after shooting | ON (camera-specific menu) |

**Security:** plain FTP sends the password and photos unencrypted. Keep it on a trusted LAN or a dedicated Wi-Fi
network (the script's firewall rules default to LAN-only), or use `--ftps` if the camera supports it. The camera account
cannot leave its upload folder or log in to a shell. No secrets are stored in the repository.

### FTP in dev mode

vsftpd can't write into a home directory that has `750` permissions. For dev testing, point FTP at `/srv/photos/incoming`
and tell the dev pipeline to watch that folder:

```bash
sudo ./scripts/setup_ftp.sh --owner "$USER" --dev-user "$USER"      # then log out/in once (group membership)
PIPELINE_INCOMING=/srv/photos/incoming VENV="$CONDA_PREFIX" ./scripts/run_dev.sh run
```

---

## 6. Testing

### Unit and integration tests

```bash
./scripts/fetch_test_images.sh        # sample face / group photos into tests/fixtures (once)
python -m pytest                      # 63 tests
```

The face tests need the YuNet model. They look in `$PHOTO_PIPELINE_MODELS`, `./models`,
`~/auto-editing-workflow/models` and `/opt/photo-pipeline/models`, and they are skipped, not failed, when it is missing.
To test a real side profile, put a photo at `tests/fixtures/profile.jpg`.

What the tests cover:
* **Edits:** white balance removes a cast and respects its gain limit; exposure brightens dark images without blowing
  highlights; CLAHE raises local contrast.
* **Detection:** exactly one face on a portrait, none on a landscape, two in a composite, several in a group photo.
  Small faces in a large frame and faces in 12 MP frames that get downscaled for detection are found, and their
  coordinates map back correctly. Side profiles don't crash, and 6 threads can share the detector pool.
* **Retouch:** no pixel outside the skin mask ever changes. The eyes and mouth stay out of the mask. The mean skin colour
  stays within ΔE < 1.5. Detail in the skin is reduced. Strength 0 is an exact no-op, and the SSIM/difference safety
  check falls back to weaker strengths.
* **Pipeline:** the full flow from incoming to gallery, the manifest, duplicate uploads, corrupt files going to
  `failed/`, truncated uploads waiting, recovery after a crash, publishing during an outage through the persistent queue,
  bursts across workers, and inotify firing only on `close_write`/rename (not on `.part` writes).

### Simulating a camera upload

```bash
# via the local FTP server (vsftpd must be set up)
FTP_PASSWORD='...' ./scripts/simulate_camera.sh ~/Pictures/test.jpg
FTP_PASSWORD='...' ./scripts/simulate_camera.sh -g 10 -j 4           # 10 generated 24 MP JPEGs, 4 parallel uploads (burst)
FTP_PASSWORD='...' ./scripts/simulate_camera.sh --tmp-rename DIR/    # upload as .tmp then rename, like some cameras
FTP_PASSWORD='...' ./scripts/simulate_camera.sh --ftps photo.jpg

# without FTP, drop straight into the dev incoming folder
./scripts/simulate_camera.sh --local ~/auto-editing-workflow/data/incoming -g 5
```

With curl or plain lftp:

```bash
curl -T photo.jpg --ftp-pasv ftp://127.0.0.1/ --user camera:PASSWORD
lftp -u camera -e "set ftp:passive-mode true; put photo.jpg; bye" 127.0.0.1
```

### Processing single files for tuning

```bash
python -m photo_pipeline process --dev photo.jpg --out /tmp/look --compare --save-masks
python -m photo_pipeline process --dev photo.jpg --out /tmp/look --retouch-strength 0.6 --method guided
python -m photo_pipeline process --dev photo.jpg --out /tmp/look --no-edit       # retouch only
```

This prints the timings, the edit decisions (gains, black/white points, gamma) and the per-face retouch report
(strength used, SSIM, mask coverage). With `--save-masks` it also writes `<name>_mask.png`, where white areas are
smoothed and black areas are untouched. With `--compare` it writes `<name>_compare.jpg`: BEFORE | AFTER side by side,
plus a third SMOOTHED AREA panel (green overlay) when combined with `--save-masks`.

---

## 7. How the automatic edits work

| Step | Method | Main settings (config key) |
|---|---|---|
| White balance | Gray-world in linear light, using near-neutral pixels only (falls back to all mid-tones at half strength). Gains are clamped and blended. | `edit.white_balance.strength` (0.6), `max_gain` (1.25) |
| Exposure | Percentile auto-levels (0.3 % / 99.7 %) with a limited black point, gain capped by `max_gain`, a soft highlight shoulder so nothing new clips, and a midtone gamma toward `target_midtone` | `edit.exposure.*` |
| Contrast | CLAHE on L\* only (colour untouched), blended | `edit.contrast.clip_limit` (1.6), `strength` (0.6) |
| RAW | `rawtherapee-cli` with the *Auto-Matched Curve* profile (falls back to its default profile, then to `darktable-cli`) → 16-bit TIFF, then the steps above | `raw.*` |

The image is resized to the export size before editing, so each step works on about 2.8 MP instead of the full sensor
resolution. The full-resolution original is always archived untouched.

### Face and skin retouching (likeness-preserving)

No generative models are used. ML only locates faces; every pixel change is a classical filter.

1. **Detection:** OpenCV **YuNet** runs on the CPU. The model is loaded once at startup, with one instance per worker
   held in a pool so threads never share an instance. It returns boxes plus 5 landmarks. Faces below the confidence
   threshold (`retouch.detector.score_threshold`) or smaller than `min_face_px` are ignored. Images with no faces are
   passed through untouched.
2. **Face region:** an ellipse built from the eye line and mouth, rotated with head roll. The eyes, eyebrows, nostrils
   and lips are cut out. With `retouch.landmarks.enabled: true` and mediapipe installed, a precise 478-point MediaPipe
   face oval with polygon cut-outs is used instead.
3. **Skin model:** an adaptive YCrCb model. It samples each face's own cheeks, nose bridge and forehead, so it adapts to
   every skin tone instead of relying on fixed "skin colour" limits. It falls back to classic Cr/Cb ranges when sampling
   isn't possible. Hair, beard, background and eyes fall outside the model.
4. **Mask:** the face region is combined with the skin model and cleaned with morphology. It is then eroded and feathered
   with a Gaussian blur, so the soft edge stays inside the skin with no visible seam.
5. **Smoothing (frequency separation):** L\* is split into a base layer (`bilateral`, `guided` or `gaussian`) and
   detail. The mid-frequency blemish band is reduced by `strength`, and the finest detail (pores and texture) is kept
   (`texture_keep` 0.9). Colour blotches get mild chroma smoothing. Nothing is warped, reshaped, slimmed or enlarged.
6. **Skin tone lock:** after smoothing, the mask-weighted mean change in L\*a\*b\* is subtracted, so the skin is never
   lightened, darkened or shifted in hue. It is only smoothed.
7. **Safety check:** within each face box, the SSIM and mean pixel difference against the original are measured. If
   either exceeds its limit (`retouch.safety`), the strength is reduced (×`fallback_factor`) and tried again. Below
   `min_strength` the face is left untouched.
8. **Profiles:** strongly turned faces (landmark geometry check) get half strength and a conservative mask
   (`profile_policy: reduce`), or are skipped (`profile_policy: skip`).

Pixels outside the mask are bit-for-bit identical to the input, and the tests check this.

## 8. Tuning the edit strength

All settings are in `config.yaml` (dev: `config/config.dev.yaml`, production: `/etc/photo-pipeline/config.yaml`). Restart
the service after editing. Use `process --save-masks` on a few typical shots first.

| Goal | Change |
|---|---|
| Stronger / weaker skin smoothing | `retouch.strength` 0.2 (very light) … 0.35 (default) … 0.6 (noticeable) |
| Keep more pore texture | `retouch.texture_keep` → 1.0 |
| Smooth larger blotches | `retouch.smooth_radius` 0.02 → 0.03, `sigma_color` 7 → 10 |
| Less colour smoothing of redness | `retouch.chroma_smoothing` 0.5 → 0.2 |
| Faces missed / false faces | `retouch.detector.score_threshold` lower (0.6) / higher (0.85) |
| Tiny faces in group shots | `retouch.detector.min_face_px` lower, `max_side` higher |
| Safety check too strict | lower `retouch.safety.min_ssim`, raise `max_mean_abs_diff` |
| Turn off retouching | `retouch.enabled: false` |
| Warmer/cooler images left alone | lower `edit.white_balance.strength` |
| Less brightening | lower `edit.exposure.strength` or `max_gain` |
| Punchier / flatter | raise / lower `edit.contrast.clip_limit` (1.2–2.5) |
| Output size / quality | `export.long_edge`, `export.quality`, `export.format` |
| Privacy | `export.metadata: minimal` or `strip`; `strip_gps` is on by default |
| Throughput for big bursts | `workers.count` = CPU cores ÷ 2, `workers.opencv_threads` 1–2 |

## 9. Publishing backends

Credentials come from the environment: `.env` in dev, `/etc/photo-pipeline/photo-pipeline.env` in production. Every
config string supports `${VAR}` and `${VAR:-default}`.

**local (default):** copies the file into `publish.local.target_dir` atomically (temp file + rename) and keeps a
`manifest.json` (newest first, with size and SHA-256) that a gallery page can read.

**ssh (rsync or sftp, key auth):**
```yaml
publish:
  backend: rsync            # or sftp
```
```bash
PUBLISH_SSH_HOST=web.example.com
PUBLISH_SSH_USER=deploy
PUBLISH_SSH_REMOTE_DIR=/var/www/site/gallery
PUBLISH_SSH_KEY=/var/lib/photo-pipeline/.ssh/id_ed25519
```
```bash
sudo -u photopipe ssh-keygen -t ed25519 -N '' -f /var/lib/photo-pipeline/.ssh/id_ed25519
sudo -u photopipe ssh-copy-id -i /var/lib/photo-pipeline/.ssh/id_ed25519 deploy@web.example.com
```
With sftp, the file is uploaded under a temporary name and renamed, so the website never serves a half-written photo.
If the server's rsync is older than 3.2.3, set `publish.ssh.mkpath: false`.

**https (existing website/CMS API; we are the client):**
```yaml
publish:
  backend: https
  https:
    method: POST
    upload_mode: multipart        # or raw (body = file bytes)
    file_field: file
    form_fields: {filename: "{filename}", sha256: "{sha256}", album: "events"}
    auth: bearer                  # none | bearer | basic | header
    response_url_field: url       # JSON field (dotted path) holding the published URL
```
```bash
PUBLISH_HTTPS_URL=https://example.com/api/photos
PUBLISH_HTTPS_TOKEN=...
```
Each request carries an `Idempotency-Key: <sha256>` header, so a retried upload can be de-duplicated by the server.
HTTP 5xx, 408 and 429 responses and network errors are retried. Other 4xx responses are logged as permanent failures.

**Retries:** each photo gets `inline_attempts` quick tries (1 s, 2 s, …). If those fail, it goes into the SQLite publish
queue, which survives restarts and reboots. A background thread retries with exponential backoff and jitter
(`base_delay_s` 30 s, doubling up to `max_delay_s` 30 min; `max_attempts: 0` means retry forever). Processing never waits
for the network. Every success and failure is logged per photo.

## 10. Reliability behaviour

* **Only complete uploads are processed.** Triggers are `IN_CLOSE_WRITE` and `IN_MOVED_TO` (rename after upload), never
  create. Before processing, the file's size and mtime must also be stable for `stability_ms`, the image must open, and a
  JPEG must have its end marker. Truncated uploads are left alone for `incomplete_grace_s` (the camera may resume) and
  then moved to `failed/`.
* **Temp files are ignored:** `.tmp`, `.part`, `.partial`, `.filepart`, dotfiles and `~` lock files. Accepted formats
  are JPG, JPEG, PNG, CR2, CR3, NEF, ARW, RAF and DNG, in any letter case.
* **Never processed twice:** files are keyed by SHA-256 in SQLite. Re-uploads go to `originals/_duplicates/` (policy
  `originals.duplicates`), and the queue also de-duplicates paths already waiting.
* **Originals are never modified.** They are moved to `originals/YYYY/MM/DD/` before editing starts.
* **One bad file never stops the pipeline.** Every error is caught per photo. The file goes to `failed/` with a
  `.error.txt` traceback, and the workers keep going.
* **Safe to restart at any time.** On startup, photos interrupted mid-edit are resumed from `originals/`, photos
  processed but not published are re-queued, and files already in `incoming/` are processed. A sweep every
  `sweep_interval_s` catches anything a missed inotify event left behind.

## 11. Troubleshooting

| Symptom | Check |
|---|---|
| Nothing happens on upload | `journalctl -u photo-pipeline -f`; confirm the file landed in `/srv/photos/incoming` (`ls -la`); run `photo_pipeline check`. |
| Camera cannot connect | Passive mode ON in the camera; `sudo ufw status`; `sudo tail /var/log/vsftpd.log`; the camera must be in the allowed subnet (`--allow-from`). Behind NAT, set `--pasv-address`. |
| `530 Login incorrect` | Reset with `sudo FTP_PASSWORD=... ./scripts/setup_ftp.sh`; the user must be in `/etc/vsftpd.photos.userlist`. |
| `553 Could not create file` | Upload folder permissions: `ls -ld /srv/photos/incoming` should be `drwxrwsr-x photopipe photos`. |
| `500 OOPS: vsftpd: refusing to run with writable root` | The template sets `allow_writeable_chroot=YES`; re-run `setup_ftp.sh`. |
| "face retouching disabled: model file not found" | `./scripts/download_models.sh --dir <paths.models>` |
| RAW files go to `failed/` | `sudo apt install rawtherapee darktable`; check `rawtherapee-cli -v`; see `.error.txt`. |
| Photos stuck in "publish_pending" | `photo_pipeline status`; fix the network or credentials, then `photo_pipeline retry-publish` (or wait for the backoff). |
| `Permission denied` writing the gallery | Add the folder to `ReadWritePaths=` in the unit and make it writable by `photopipe`. |
| Service keeps restarting | `systemctl status photo-pipeline`; `journalctl -u photo-pipeline -b`; `photo_pipeline check`. |
| Retouch too strong or visible | Lower `retouch.strength`; inspect `--save-masks` output. |
| Slow processing | Log lines show `decode/edit/retouch/export` ms per photo; raise `workers.count` on multi-core CPUs; RAW development dominates for RAW files. |
| inotify watch limit errors | `cat /proc/sys/fs/inotify/max_user_watches`; `install.sh` sets 524288 via `/etc/sysctl.d/90-photo-pipeline.conf`. |

Useful commands:

```bash
python -m photo_pipeline status --dev                # counts per status + last photos with errors
python -m photo_pipeline check --dev                 # folders, models, converters, publisher config
sqlite3 ~/auto-editing-workflow/data/state.db 'select id,source_name,status,error from photos order by id desc limit 20;'
```
