#!/usr/bin/env python3
"""Smoke Spotter — a Waggle/Sage edge-AI wildfire-smoke detection plugin.

Doctrine: **move the work, not the data.** This plugin reads frames from a node
camera (or RTSP / file:// clip), runs a cheap CPU change-gate, then a cheap
smoke-haze heuristic, and escalates to the node's bundled vision-language model
ONLY for frames that both *changed* and look *hazy*. It publishes tiny
measurements to Beehive and uploads a snapshot ONLY on an alert.

Pipeline (cheap -> expensive, escalate only on threshold):
  1. Camera frame   — waggle.data.vision.Camera(device).snapshot()
  2. CHANGE-GATE    — downscaled grayscale mean-abs-diff vs previous frame
  3. SMOKE SCREEN   — grey-haze / low-contrast / upper-band desaturation blend
  4. VLM ESCALATION — only if changed AND smoke_score >= vlm_threshold
  5. PUBLISH        — env.smoke.score / env.smoke.alert / env.smoke.vlm
  6. UPLOAD         — one snapshot, ONLY on alert

Published measurements (per processed frame):
  env.smoke.score   float[0..1]   the heuristic smoke screen
  env.smoke.changed int(0|1)      did the frame change vs the previous one
  env.smoke.alert   int(0|1)      heuristic-or-VLM raised an alert
  env.smoke.vlm     str           VLM label: "smoke" | "clear" | "unknown"
                                  (only published when the VLM was invoked)
  env.smoke.status  json          retained node liveness self-report (periodic)
  upload            <snapshot>    only on alert (meta: camera, smoke_score, vlm)

Interop with the Sage reference detector (iperezx/wildfire-smoke-detection /
SmokeyNet): it shares the env.smoke. namespace but only emits
env.smoke.tile_probs (a STRINGIFIED list of per-tile probs, not a scalar) and
env.smoke.certainty (binary-classifier float). Our names are deliberately
DISJOINT from those, so both can run on one node and be joined in
sage_data_client by (meta.vsn, timestamp); our score/alert are real numeric
scalars (not a stringified list). See ecr/sage.yaml metadata.interop.

NODE-HARDENING (this file is the long-running service; modelled on the
RadioTower ADS-B service — load_config / preflight / publish_status / a
retry-gate that self-heals + a --post dry-run bring-up seam):

  * ``load_config()`` — never-crash config from env + argparse, malformed
    values degrade to safe defaults (never raises).
  * ``preflight_camera()`` / ``open_camera_with_retry()`` — a camera that
    enumerates slowly retries with backoff instead of crashing the node.
  * ``status_payload()`` + ``publish_status()`` — a retained env.smoke.status
    so the node is visibly alive even with no smoke to report.
  * ``--post`` / ``SMOKE_POST`` — OFF by default: LOG the exact measurement
    payload it WOULD publish (safe bring-up); ON: publish for real. Mirrors
    TowerWatch firewatch_vision's gated --post.

pywaggle is imported LAZILY so this file imports without it. Offline development
uses ``dev_run.py`` (no node, no pywaggle, no API required).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

# Local detection logic — pure, no node/VLM/network deps. ``cameras`` +
# ``ingest_normalize`` are the offline-first ingest modules (stdlib-only,
# injectable fetch/subprocess) wired into the live path below.
try:
    from app import smoke as smoke_mod
    from app import vlm as vlm_mod
    from app import cameras as cameras_mod
    from app import ingest_normalize as ingest_mod
except ImportError:  # when run as `python3 main.py` from inside app/
    import smoke as smoke_mod  # type: ignore
    import vlm as vlm_mod  # type: ignore
    import cameras as cameras_mod  # type: ignore
    import ingest_normalize as ingest_mod  # type: ignore

LOG = logging.getLogger("smoke_spotter")

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (env-overridable; see build_parser for the env var names)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CAMERA = "bottom_camera"          # node data-shim id; or rtsp:// / file://
DEFAULT_CHANGE_THRESHOLD = smoke_mod.DEFAULT_CHANGE_THRESHOLD  # 8.0
DEFAULT_SMOKE_ALERT = 0.75                # smoke_score at/above which alert=true
DEFAULT_VLM_THRESHOLD = 0.6               # smoke_score at/above which we escalate
DEFAULT_INTERVAL = 60.0                   # seconds between frames
DEFAULT_VLM_KIND = "auto"                 # apple | node | ollama | none | auto
DEFAULT_VLM_TIMEOUT = 60.0

# Camera preflight / retry-gate (the slow-enumeration self-heal seam).
DEFAULT_CAMERA_RETRY_S = 10.0             # backoff between camera open attempts
DEFAULT_CAMERA_MAX_RETRY_S = 120.0       # backoff cap
DEFAULT_CAMERA_ATTEMPTS = 0               # 0 == retry forever (never give up)

# How often to emit the retained status self-report (independent of frame cadence).
DEFAULT_STATUS_INTERVAL_S = 300.0

# ── Ingest path (http(s) still / rtsp-or-hls stream / ALERTCalifornia discovery) ─
# When ``camera`` is a URL (http(s)://, rtsp://, *.m3u8) or the keyword
# ``discover`` it is resolved through app/ingest_normalize (transport precedence)
# instead of the node Camera shim, and one frame is grabbed per ``interval``:
# a still-JPEG via stdlib urllib, or a single ffmpeg keyframe for a stream. This
# is the seam that lets the plugin watch a public ALERTCalifornia / WIFIRE camera
# (or any RTSP PTZ cam) — pywaggle is OPTIONAL on this path (absent -> dry-run).
DEFAULT_INGEST_TIMEOUT_S = 15.0          # per-frame still-fetch / ffmpeg deadline
DEFAULT_RELAY_BASE = ""                   # optional ingest relay (HLS/jpeg proxy)
# Discovery centre + radius for the ``discover`` camera keyword (mirrors the
# cameras module defaults — UC Davis, the hackathon home node, 50-mile radius).
DEFAULT_LAT = cameras_mod.DAVIS_LAT
DEFAULT_LON = cameras_mod.DAVIS_LON
DEFAULT_RADIUS_MI = cameras_mod.DEFAULT_RADIUS_MI

# ── Tiled + rolling-baseline cascade (the static-camera + localized-plume fix) ──
# These drive the DEFAULT detection path now. The benchmark proved the global
# change-gate + global smoke_score are blind to a static camera with a slow,
# localized plume (frame-to-frame diff stays flat; a small plume is averaged out).
# The tiled detector splits the frame into a grid, keeps a per-tile rolling EMA
# background (so slow growth accumulates), and takes the MAX haze over tiles.
DEFAULT_GRID_ROWS = smoke_mod.DEFAULT_GRID_ROWS            # 4
DEFAULT_GRID_COLS = smoke_mod.DEFAULT_GRID_COLS            # 4
DEFAULT_BASELINE = "ema"                                   # ema | frame
DEFAULT_BASELINE_ALPHA = smoke_mod.DEFAULT_BASELINE_ALPHA  # 0.05
DEFAULT_TILE_CHANGE_THRESHOLD = smoke_mod.DEFAULT_TILE_CHANGE_THRESHOLD  # 8.0
DEFAULT_TILE_SMOKE_ALERT = smoke_mod.DEFAULT_TILE_SMOKE_ALERT            # 0.55
DEFAULT_TILED = True   # use the tiled cascade by default (the fix)

# PERIODIC FORCED ESCALATION ("heartbeat"): even with NO tile trip, escalate to
# the VLM every SMOKE_HEARTBEAT_S seconds so a static scene is still confirmed
# by the heavy path on a fixed cadence. This BOUNDS misses: the absolute worst
# case for a missed plume is one heartbeat interval, not "forever". 0 disables it.
DEFAULT_HEARTBEAT_S = 600.0

# Self-report measurement name. Beehive treats this as the node-liveness channel.
STATUS_NAME = "env.smoke.status"

# Status states (mirrors the ADS-B service's polling/waiting/no_receiver/stopped).
ST_WATCHING = "watching"        # camera up, pipeline running, no smoke to report
ST_NO_CAMERA = "no_camera"      # preflight/open failing; retrying with backoff
ST_ESCALATING = "escalating"    # an alert just fired (VLM invoked / smoke seen)
ST_STARTING = "starting"
ST_STOPPED = "stopped"


# ─────────────────────────────────────────────────────────────────────────────
# Never-crash config loader (env + argparse, malformed -> safe defaults)
# ─────────────────────────────────────────────────────────────────────────────

def _env_str(name: str, default: str) -> str:
    """Env string with a default. Empty/whitespace-only -> default."""
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _coerce_float(value: Any, default: float, *, positive: bool = False) -> float:
    """Coerce anything to float, falling back to default on garbage.

    With ``positive`` a non-positive (<=0) or non-finite value also degrades to
    the default — interval/threshold/timeout values must stay sane on a node.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
        return default
    if positive and f <= 0.0:
        return default
    return f


def _coerce_int(value: Any, default: int, *, minimum: Optional[int] = None) -> int:
    try:
        i = int(float(value))  # tolerate "3", "3.0", 3.0
    except (TypeError, ValueError):
        return default
    if minimum is not None and i < minimum:
        return default
    return i


def _coerce_bool(value: Any, default: bool) -> bool:
    """Tolerant truthiness for env strings: 1/true/yes/on, 0/false/no/off."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on", "y", "t"):
        return True
    if s in ("0", "false", "no", "off", "n", "f", ""):
        return False
    return default


def _env_float(name: str, default: float, *, positive: bool = False) -> float:
    return _coerce_float(os.environ.get(name), default, positive=positive)


_VLM_CHOICES = ("auto", "apple", "node", "ollama", "none")


@dataclass
class SmokeConfig:
    """Resolved Smoke Spotter runtime config.

    Built by :func:`load_config` from env + CLI; every field is validated, so a
    node started with a garbled environment still comes up on safe defaults
    rather than crashing (same never-crash convention as the ADS-B service's
    ``load_config``).
    """

    camera: str = DEFAULT_CAMERA
    interval_s: float = DEFAULT_INTERVAL
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD
    smoke_alert: float = DEFAULT_SMOKE_ALERT
    vlm_threshold: float = DEFAULT_VLM_THRESHOLD
    vlm_kind: str = DEFAULT_VLM_KIND
    vlm_timeout_s: float = DEFAULT_VLM_TIMEOUT
    # Camera preflight / retry-gate.
    camera_retry_s: float = DEFAULT_CAMERA_RETRY_S
    camera_max_retry_s: float = DEFAULT_CAMERA_MAX_RETRY_S
    camera_attempts: int = DEFAULT_CAMERA_ATTEMPTS   # 0 == forever
    # Status self-report cadence.
    status_interval_s: float = DEFAULT_STATUS_INTERVAL_S
    # Ingest path (used when ``camera`` is a URL / ``discover`` keyword).
    ingest_timeout_s: float = DEFAULT_INGEST_TIMEOUT_S
    relay_base: str = DEFAULT_RELAY_BASE
    lat: float = DEFAULT_LAT
    lon: float = DEFAULT_LON
    radius_mi: float = DEFAULT_RADIUS_MI
    # Ember: aggregate per-frame verdicts into an evidence Event (operational
    # plane) and replay closed candidates into lessons at loop exit (learning
    # plane). Pure/local; no network. Off -> the loop behaves exactly as before.
    ember: bool = True
    # Tiled + rolling-baseline cascade (the static-camera + localized-plume fix).
    tiled: bool = DEFAULT_TILED
    grid_rows: int = DEFAULT_GRID_ROWS
    grid_cols: int = DEFAULT_GRID_COLS
    baseline: str = DEFAULT_BASELINE
    baseline_alpha: float = DEFAULT_BASELINE_ALPHA
    tile_change_threshold: float = DEFAULT_TILE_CHANGE_THRESHOLD
    tile_smoke_alert: float = DEFAULT_TILE_SMOKE_ALERT
    # Periodic forced VLM escalation ('heartbeat'); 0 disables. Bounds misses.
    heartbeat_s: float = DEFAULT_HEARTBEAT_S
    # Publishing gate: OFF until on a node (dry-run logs the payload instead).
    post: bool = False
    verbose: bool = False

    @property
    def is_file(self) -> bool:
        return isinstance(self.camera, str) and self.camera.startswith("file://")

    def as_dict(self) -> dict:
        return {
            "camera": self.camera,
            "interval_s": self.interval_s,
            "change_threshold": self.change_threshold,
            "smoke_alert": self.smoke_alert,
            "vlm_threshold": self.vlm_threshold,
            "vlm_kind": self.vlm_kind,
            "vlm_timeout_s": self.vlm_timeout_s,
            "camera_retry_s": self.camera_retry_s,
            "camera_max_retry_s": self.camera_max_retry_s,
            "camera_attempts": self.camera_attempts,
            "status_interval_s": self.status_interval_s,
            "ingest_timeout_s": self.ingest_timeout_s,
            "relay_base": self.relay_base,
            "lat": self.lat,
            "lon": self.lon,
            "radius_mi": self.radius_mi,
            "ember": self.ember,
            "tiled": self.tiled,
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "baseline": self.baseline,
            "baseline_alpha": self.baseline_alpha,
            "tile_change_threshold": self.tile_change_threshold,
            "tile_smoke_alert": self.tile_smoke_alert,
            "heartbeat_s": self.heartbeat_s,
            "post": self.post,
        }


def load_config(argv: Optional[list] = None) -> SmokeConfig:
    """Resolve config from env + argparse into a validated SmokeConfig.

    NEVER raises on bad config: malformed/missing values silently degrade to
    safe defaults (a node started with a typo'd env var or a bad CLI float must
    still come up watching). Precedence: explicit CLI flag > env var > default.

    ``argv`` is the argument list (defaults to ``sys.argv[1:]``). argparse's own
    type-coercion failures (e.g. ``--interval abc``) would normally ``SystemExit``;
    we parse leniently (all numeric flags are strings here) and coerce ourselves
    so even a bad CLI value falls back to default instead of killing the process.
    """
    parser = build_parser()
    # parse_known_args so stray/unknown args never abort node bring-up.
    try:
        args, _unknown = parser.parse_known_args(argv)
    except SystemExit:
        # --help/-h or an argparse-level error: fall back to a pure-env config
        # rather than letting the service die during startup.
        args = argparse.Namespace()

    def pick(attr: str, env: str, default):
        """CLI value if the user set it, else env, else default (as raw strings)."""
        cli = getattr(args, attr, None)
        if cli is not None:
            return cli
        return os.environ.get(env, default)

    cfg = SmokeConfig()

    # --- camera (string; never numeric) ---
    cam = pick("camera", "SMOKE_CAMERA", DEFAULT_CAMERA)
    cam = str(cam).strip() if cam is not None else ""
    cfg.camera = cam or DEFAULT_CAMERA

    # --- numeric thresholds / intervals (positive-only where it matters) ---
    cfg.interval_s = _coerce_float(
        pick("interval", "SMOKE_INTERVAL", DEFAULT_INTERVAL),
        DEFAULT_INTERVAL, positive=True,
    )
    cfg.change_threshold = _coerce_float(
        pick("change_threshold", "SMOKE_CHANGE_THRESHOLD", DEFAULT_CHANGE_THRESHOLD),
        DEFAULT_CHANGE_THRESHOLD,
    )
    if cfg.change_threshold < 0:
        cfg.change_threshold = DEFAULT_CHANGE_THRESHOLD
    cfg.smoke_alert = _clamp01_or(
        pick("smoke_alert", "SMOKE_ALERT", DEFAULT_SMOKE_ALERT), DEFAULT_SMOKE_ALERT
    )
    cfg.vlm_threshold = _clamp01_or(
        pick("vlm_threshold", "SMOKE_VLM_THRESHOLD", DEFAULT_VLM_THRESHOLD),
        DEFAULT_VLM_THRESHOLD,
    )
    cfg.vlm_timeout_s = _coerce_float(
        pick("vlm_timeout", "SMOKE_VLM_TIMEOUT", DEFAULT_VLM_TIMEOUT),
        DEFAULT_VLM_TIMEOUT, positive=True,
    )

    # --- vlm backend (constrained to the known set) ---
    vk = str(pick("vlm", "SMOKE_VLM", DEFAULT_VLM_KIND) or "").strip().lower()
    cfg.vlm_kind = vk if vk in _VLM_CHOICES else DEFAULT_VLM_KIND

    # --- camera preflight / retry-gate ---
    cfg.camera_retry_s = _coerce_float(
        pick("camera_retry", "SMOKE_CAMERA_RETRY_S", DEFAULT_CAMERA_RETRY_S),
        DEFAULT_CAMERA_RETRY_S, positive=True,
    )
    cfg.camera_max_retry_s = _coerce_float(
        pick("camera_max_retry", "SMOKE_CAMERA_MAX_RETRY_S", DEFAULT_CAMERA_MAX_RETRY_S),
        DEFAULT_CAMERA_MAX_RETRY_S, positive=True,
    )
    if cfg.camera_max_retry_s < cfg.camera_retry_s:
        cfg.camera_max_retry_s = cfg.camera_retry_s
    cfg.camera_attempts = _coerce_int(
        pick("camera_attempts", "SMOKE_CAMERA_ATTEMPTS", DEFAULT_CAMERA_ATTEMPTS),
        DEFAULT_CAMERA_ATTEMPTS, minimum=0,
    )

    # --- status cadence ---
    cfg.status_interval_s = _coerce_float(
        pick("status_interval", "SMOKE_STATUS_INTERVAL_S", DEFAULT_STATUS_INTERVAL_S),
        DEFAULT_STATUS_INTERVAL_S, positive=True,
    )

    # --- ingest path (url/discover camera): timeout, relay, discovery geo ---
    cfg.ingest_timeout_s = _coerce_float(
        pick("ingest_timeout", "SMOKE_INGEST_TIMEOUT_S", DEFAULT_INGEST_TIMEOUT_S),
        DEFAULT_INGEST_TIMEOUT_S, positive=True,
    )
    relay = pick("relay_base", "SMOKE_RELAY_BASE", DEFAULT_RELAY_BASE)
    cfg.relay_base = str(relay).strip() if relay is not None else DEFAULT_RELAY_BASE
    cfg.lat = _coerce_float(pick("lat", "SMOKE_LAT", DEFAULT_LAT), DEFAULT_LAT)
    cfg.lon = _coerce_float(pick("lon", "SMOKE_LON", DEFAULT_LON), DEFAULT_LON)
    cfg.radius_mi = _coerce_float(
        pick("radius", "SMOKE_RADIUS_MI", DEFAULT_RADIUS_MI),
        DEFAULT_RADIUS_MI, positive=True,
    )
    cfg.ember = _coerce_bool(os.environ.get("SMOKE_EMBER"), True)

    # --- tiled + rolling-baseline cascade (the static-camera + plume fix) ---
    tiled_cli = getattr(args, "no_tiled", None)
    if tiled_cli:  # --no-tiled given -> opt back into the global-only path
        cfg.tiled = False
    else:
        cfg.tiled = _coerce_bool(os.environ.get("SMOKE_TILED"), DEFAULT_TILED)
    cfg.grid_rows = _coerce_int(
        pick("grid_rows", "SMOKE_GRID_ROWS", DEFAULT_GRID_ROWS),
        DEFAULT_GRID_ROWS, minimum=1,
    )
    cfg.grid_cols = _coerce_int(
        pick("grid_cols", "SMOKE_GRID_COLS", DEFAULT_GRID_COLS),
        DEFAULT_GRID_COLS, minimum=1,
    )
    bl = str(pick("baseline", "SMOKE_BASELINE", DEFAULT_BASELINE) or "").strip().lower()
    cfg.baseline = bl if bl in ("ema", "frame") else DEFAULT_BASELINE
    cfg.baseline_alpha = _coerce_float(
        pick("baseline_alpha", "SMOKE_BASELINE_ALPHA", DEFAULT_BASELINE_ALPHA),
        DEFAULT_BASELINE_ALPHA, positive=True,
    )
    if not (0.0 < cfg.baseline_alpha <= 1.0):  # EMA weight must be in (0,1]
        cfg.baseline_alpha = DEFAULT_BASELINE_ALPHA
    cfg.tile_change_threshold = _coerce_float(
        pick("tile_change_threshold", "SMOKE_TILE_CHANGE_THRESHOLD",
             DEFAULT_TILE_CHANGE_THRESHOLD),
        DEFAULT_TILE_CHANGE_THRESHOLD,
    )
    if cfg.tile_change_threshold < 0:
        cfg.tile_change_threshold = DEFAULT_TILE_CHANGE_THRESHOLD
    cfg.tile_smoke_alert = _clamp01_or(
        pick("tile_smoke_alert", "SMOKE_TILE_ALERT", DEFAULT_TILE_SMOKE_ALERT),
        DEFAULT_TILE_SMOKE_ALERT,
    )

    # --- periodic forced VLM escalation ('heartbeat'); 0 disables ---
    cfg.heartbeat_s = _coerce_float(
        pick("heartbeat", "SMOKE_HEARTBEAT_S", DEFAULT_HEARTBEAT_S),
        DEFAULT_HEARTBEAT_S,
    )
    if cfg.heartbeat_s < 0:  # negative is meaningless; 0 = disabled is allowed
        cfg.heartbeat_s = DEFAULT_HEARTBEAT_S

    # --- publish gate: --post OR SMOKE_POST (default OFF / dry-run) ---
    post_cli = getattr(args, "post", None)
    if post_cli:  # store_true -> True only when the flag is given
        cfg.post = True
    else:
        cfg.post = _coerce_bool(os.environ.get("SMOKE_POST"), False)

    cfg.verbose = bool(getattr(args, "verbose", False))
    return cfg


def _clamp01_or(value: Any, default: float) -> float:
    """Coerce to a [0,1] float; garbage -> default, out-of-range -> clamped."""
    f = _coerce_float(value, default)
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


# ─────────────────────────────────────────────────────────────────────────────
# Lazy pywaggle import (so this module imports without it installed)
# ─────────────────────────────────────────────────────────────────────────────

def _import_waggle():
    """Return (Plugin, Camera) or raise a clear, actionable error."""
    try:
        from waggle.plugin import Plugin  # type: ignore
        from waggle.data.vision import Camera  # type: ignore
    except ImportError as exc:  # pragma: no cover - only when pywaggle absent
        raise SystemExit(
            "pywaggle is not installed. For the node plugin run:\n"
            "    pip install -U 'pywaggle[all]'\n"
            "For OFFLINE development without a node, use dev_run.py instead:\n"
            "    python3 app/dev_run.py --camera file://clip.mp4\n"
            f"(import error: {exc})"
        )
    return Plugin, Camera


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_snapshot(sample, suffix: str = ".jpg") -> Optional[str]:
    """Persist an ImageSample to a temp file for upload. Returns path or None."""
    try:
        fd, path = tempfile.mkstemp(prefix="smoke_", suffix=suffix)
        os.close(fd)
        sample.save(path)  # ImageSample.save uses cv2.imwrite under the hood
        return path
    except Exception as exc:
        LOG.warning("could not save snapshot for upload: %s", exc)
        return None


def _encode_for_vlm(sample) -> Optional[bytes]:
    """Encode an ImageSample to JPEG bytes for the VLM. Returns None on failure.

    A sample may expose its own ``encode_jpeg()`` (the offline ``DevSample`` does)
    — prefer it so the dev runner and the node share THIS exact escalation path.
    Otherwise prefer cv2 (already a pywaggle dep) and fall back to a saved temp file.
    """
    enc = getattr(sample, "encode_jpeg", None)
    if callable(enc):
        try:
            b = enc()
            if b:
                return bytes(b)
        except Exception as exc:  # noqa: BLE001 — fall through to the cv2 path
            LOG.debug("sample.encode_jpeg failed, trying cv2: %s", exc)
    try:
        import cv2  # type: ignore

        # ImageSample.data is RGB by default; cv2 wants BGR for imencode.
        data = sample.format.format_to_cv2(sample.data)
        ok, buf = cv2.imencode(".jpg", data)
        if ok:
            return bytes(buf.tobytes())
    except Exception as exc:
        LOG.debug("cv2 encode failed, falling back to temp file: %s", exc)
    path = _save_snapshot(sample)
    if path is None:
        return None
    try:
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# INGEST PATH — URL / ALERTCalifornia-discovery frame source (no node Camera)
# ─────────────────────────────────────────────────────────────────────────────
#
# When ``camera`` is a URL or the ``discover`` keyword the plugin grabs frames
# itself (resolved through app/ingest_normalize) instead of the node Camera shim:
# a still-JPEG over stdlib urllib, or one ffmpeg keyframe for an rtsp/hls stream.
# Both side-effecting seams (network fetch, ffmpeg subprocess) are INJECTABLE so
# the whole path unit-tests with fakes — NOTHING here touches the network or
# spawns a process at import time (offline-first, same discipline as cameras.py /
# ingest_normalize.py). pywaggle is OPTIONAL here (absent -> dry-run only).

# Camera specs that route through the ingest resolver rather than the node Camera.
_INGEST_URL_SCHEMES = ("http://", "https://", "rtsp://", "rtsps://")
_DISCOVER_KEYWORDS = ("discover", "alertca", "alertcalifornia")


def is_ingest_camera(camera: Any) -> bool:
    """True iff ``camera`` should be resolved via the ingest path (URL/discover).

    A URL scheme (http/https/rtsp) or an explicit ``discover`` keyword routes
    through app/ingest_normalize; a bare node-shim id (``bottom_camera``) or a
    ``file://`` clip stays on the node Camera path. Never raises.
    """
    if not isinstance(camera, str):
        return False
    c = camera.strip().lower()
    if c.startswith(_INGEST_URL_SCHEMES):
        return True
    # ``discover`` / ``discover:3`` / ``alertca`` -> ALERTCalifornia discovery.
    head = c.split(":", 1)[0]
    return head in _DISCOVER_KEYWORDS


def _decode_jpeg_to_rgb(data: bytes):
    """Decode encoded image bytes -> a full-res numpy RGB array, or None.

    cv2 first (already a pywaggle dep on the node), then Pillow. Returns None if
    neither is available or the bytes don't decode — the caller then hands the raw
    bytes through as ``.data`` (smoke.* decodes bytes too, degrading to score 0.0
    only if nothing can decode). Mirrors dev_run._load_image_file's decode order
    so the ingest, dev, and node paths produce the same ``.data`` contract.
    """
    if not data:
        return None
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        arr = np.frombuffer(data, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return bgr[:, :, ::-1].copy()  # BGR -> RGB to match the node contract
    except Exception:  # noqa: BLE001 — fall through to Pillow
        pass
    try:
        import io

        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(img, dtype=np.uint8)
    except Exception:  # noqa: BLE001 — no decoder; caller falls back to raw bytes
        return None


class IngestSample:
    """Offline frame sample for the ingest path (matches process_frame's contract).

    Carries exactly what process_frame / _encode_for_vlm / _save_snapshot read:
    ``.data`` (numpy RGB array, or the raw bytes if no decoder is present),
    ``.timestamp`` (ns), ``.encode_jpeg()`` (the ORIGINAL bytes, so the VLM and
    snapshot get the source image with no re-encode), and ``.save(path)`` (writes
    those bytes for the alert snapshot upload). Same shape as dev_run.DevSample.
    """

    __slots__ = ("data", "timestamp", "_jpeg")

    def __init__(self, jpeg_bytes: bytes, timestamp: int):
        decoded = _decode_jpeg_to_rgb(jpeg_bytes)
        self.data = decoded if decoded is not None else jpeg_bytes
        self.timestamp = int(timestamp)
        self._jpeg = jpeg_bytes

    def encode_jpeg(self) -> Optional[bytes]:
        return self._jpeg

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            fh.write(self._jpeg)


def _urllib_fetch_bytes(url: str, *, timeout: float) -> bytes:
    """Default still-fetch seam: GET ``url`` -> raw bytes, with the CDN headers.

    The ALERTCalifornia frame CDN 403/404s without a browser-ish User-Agent AND
    the exact Referer (see cameras.py gotcha #2), so we reuse those. Raises on any
    transport error; the ingest loop catches it and retries on the next interval.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": cameras_mod.USER_AGENT,
            "Referer": cameras_mod.CDN_REFERER,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def _run_ffmpeg_keyframe(argv: list, *, timeout: float) -> Optional[bytes]:
    """Default stream-grab seam: run the ffmpeg argv, return the keyframe bytes.

    ``argv`` comes from ingest_normalize.one_frame_command (output ``pipe:1`` ->
    the still is written to stdout). Returns the captured bytes, or None on a
    non-zero exit / no output. Raises subprocess.TimeoutExpired on overrun (the
    ingest loop catches it). Imported lazily so the module needs no subprocess at
    import and tests inject a fake runner instead of spawning ffmpeg.
    """
    import subprocess

    proc = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)  # noqa: S603
    if proc.returncode != 0:
        err = (proc.stderr or b"")[:200].decode("utf-8", errors="replace")
        LOG.warning("ffmpeg keyframe grab failed (rc=%d): %s", proc.returncode, err)
        return None
    return proc.stdout or None


def _camera_spec_to_dict(camera: str) -> dict:
    """Map a bare camera URL string to the dict ingest_normalize.normalize_camera
    expects, classifying by scheme/suffix:

      * ``*.m3u8`` (http)   -> ``hls_url``  (an already-republished HLS playlist)
      * other http(s)       -> ``jpeg_url`` (a still-image URL)
      * ``rtsp(s)://``      -> ``rtsp_url`` (a raw RTSP stream)
    """
    c = camera.strip()
    low = c.lower()
    if low.startswith(("rtsp://", "rtsps://")):
        return {"rtsp_url": c}
    if low.endswith(".m3u8"):
        return {"hls_url": c}
    return {"jpeg_url": c}


def _discover_first_camera(
    cfg: SmokeConfig,
    *,
    fetch_text: Callable[[str], str] = cameras_mod._urllib_fetch,
) -> Optional[dict]:
    """Discover the nearest public ALERTCalifornia camera and return it as a
    normalize_camera-ready dict (``{jpeg_url, id, label, lat, lon}``), or None.

    Parses an optional limit from a ``discover:N`` spec (only the nearest is used
    as the live source; N is logged for visibility). ``fetch_text`` is injectable
    so discovery unit-tests with canned ArcGIS JSON and never hits the network.
    Never raises: any discovery failure -> None (the caller logs + bails out).
    """
    cams = cameras_mod.discover_cameras(
        center=(cfg.lat, cfg.lon), radius_mi=cfg.radius_mi, limit=1, fetch=fetch_text,
    )
    if not cams:
        LOG.error(
            "discover: no ALERTCalifornia cameras within %.0f mi of (%.4f, %.4f)",
            cfg.radius_mi, cfg.lat, cfg.lon,
        )
        return None
    cam = cams[0]
    LOG.info("discover: using %s (%s) at (%.4f, %.4f) — %s",
             cam.get("camera_id"), cam.get("name"), cam.get("lat"),
             cam.get("lon"), cam.get("image_url"))
    return {
        "id": cam.get("camera_id"),
        "label": cam.get("name"),
        "jpeg_url": cam.get("image_url"),
        "lat": cam.get("lat"),
        "lon": cam.get("lon"),
    }


def resolve_ingest_source(
    cfg: SmokeConfig,
    *,
    fetch_text: Callable[[str], str] = cameras_mod._urllib_fetch,
) -> dict:
    """Resolve ``cfg.camera`` to a normalized ingest camera (transport + url).

    ``discover``/``alertca`` -> nearest ALERTCalifornia camera (via fetch_text);
    a URL string -> classified by scheme. Returns the normalize_camera dict
    (``transport: 'none'`` if nothing usable). Never raises.
    """
    head = cfg.camera.strip().lower().split(":", 1)[0]
    if head in _DISCOVER_KEYWORDS:
        spec = _discover_first_camera(cfg, fetch_text=fetch_text)
        if spec is None:
            return ingest_mod.normalize_camera({}, relay_base=cfg.relay_base)
    else:
        spec = _camera_spec_to_dict(cfg.camera)
    return ingest_mod.normalize_camera(spec, relay_base=cfg.relay_base)


def ingest_frames(
    cfg: SmokeConfig,
    stop: dict,
    *,
    fetch_bytes: Callable[..., bytes] = _urllib_fetch_bytes,
    run_ffmpeg: Callable[..., Optional[bytes]] = _run_ffmpeg_keyframe,
    fetch_text: Callable[[str], str] = cameras_mod._urllib_fetch,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> Iterator[IngestSample]:
    """Yield one IngestSample per ``interval`` from the resolved ingest source.

    Resolves the source ONCE (so discovery / normalization happens a single time),
    then loops: grab a frame (still fetch or ffmpeg keyframe), wrap it in an
    IngestSample, yield, sleep ``interval`` (in small interruptible slices so a
    stop signal is honoured promptly). A failed/empty grab is logged and skipped —
    the loop retries on the next interval rather than crashing the service (the
    same never-die contract as the node path's _live_snapshots). All side-effecting
    seams are injected so tests drive this with fakes and no real I/O.
    """
    norm = resolve_ingest_source(cfg, fetch_text=fetch_text)
    plan = ingest_mod.grab_command(norm)
    kind = plan["kind"]
    if kind == "none":
        LOG.error("ingest: no usable frame source from camera=%r (transport=%s)",
                  cfg.camera, norm.get("transport"))
        return
    LOG.info("ingest: transport=%s kind=%s url=%s interval=%.1fs",
             norm.get("transport"), kind, plan.get("url") or "(stream)",
             cfg.interval_s)

    while not stop.get("flag"):
        data: Optional[bytes] = None
        try:
            if kind == "still":
                data = fetch_bytes(plan["url"], timeout=cfg.ingest_timeout_s)
            else:  # ffmpeg keyframe for an rtsp / hls stream
                data = run_ffmpeg(plan["argv"], timeout=cfg.ingest_timeout_s)
        except Exception as exc:  # noqa: BLE001 — transient grab failure, retry
            LOG.warning("ingest grab failed (%s): %s — retrying after interval",
                        kind, exc)
        if data:
            yield IngestSample(data, int(clock() * 1e9))
        else:
            LOG.warning("ingest: empty frame (%s) — retrying after interval", kind)
        # Interruptible interval sleep (honour a stop signal within ~0.5s).
        slept = 0.0
        while slept < cfg.interval_s and not stop.get("flag"):
            step = min(0.5, cfg.interval_s - slept)
            sleep(step)
            slept += step


# ─────────────────────────────────────────────────────────────────────────────
# VLM call watchdog (a HARD wall-clock deadline around the escalation call)
# ─────────────────────────────────────────────────────────────────────────────

def detect_with_deadline(backend, image_bytes: bytes, deadline_s: float):
    """Call ``backend.detect(image_bytes)`` under a HARD wall-clock deadline.

    The VLM call is the only blocking I/O in the per-frame loop. The backend sets
    a urllib socket timeout, but that does NOT reliably fire on a half-stalled
    streaming response (the offline eval harness, app/vlm_eval.py, observed a
    local model hang for ~50 min with the socket timeout never tripping). On the
    long-running node that would wedge frame processing, the heartbeat, AND the
    status self-report behind one stuck model — the node goes dark with no alert.

    So we run the call in a single-use worker thread and abandon it if it overruns
    ``deadline_s``, returning ``unknown`` (a timed-out abstention — the same
    fail-safe the degrade-gracefully doctrine already uses). The orphaned thread
    is left to die with the process; one wedged call costs one frame, not the node.
    Mirrors the watchdog in app/vlm_eval.py (kept here so the SHIPPED path, not
    just the eval, is protected — the exact gap Codex flagged).
    """
    executor = ThreadPoolExecutor(max_workers=1)
    fut = executor.submit(backend.detect, image_bytes)
    try:
        result = fut.result(timeout=deadline_s)
        executor.shutdown(wait=False)
        return result
    except FuturesTimeout:
        executor.shutdown(wait=False)
        LOG.warning("VLM call exceeded %.0fs watchdog — reporting unknown",
                    deadline_s)
        return vlm_mod.VLMResult(smoke=None, detail="watchdog-timeout",
                                 backend=getattr(backend, "name", "?"))


# ─────────────────────────────────────────────────────────────────────────────
# Camera preflight + retry-gate (the slow-enumeration self-heal seam)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PreflightResult:
    """Outcome of probing the camera/frame source. Never carries an exception."""

    ok: bool
    camera: str
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "camera": self.camera, "error": self.error}


def preflight_camera(camera: str, opener: Callable[[str], Any]) -> PreflightResult:
    """Probe the camera/frame source for readiness. NEVER raises.

    A node camera can enumerate slowly (USB/CSI bring-up, driver load), so an
    open failure here is an expected transient state — we return ok=False with
    the reason and let the retry-gate decide. ``opener`` is the Camera factory
    (injected so this is unit-testable without pywaggle); we open and
    immediately release a context-managed handle just to confirm it works.

    file:// sources are clips, not live cameras — they're always "ready"
    (dev_run handles those offline), so we don't probe them here.
    """
    if isinstance(camera, str) and camera.startswith("file://"):
        return PreflightResult(ok=True, camera=camera)
    try:
        handle = opener(camera)
        # Camera is a context manager on the node; release immediately if so.
        close = getattr(handle, "__exit__", None)
        enter = getattr(handle, "__enter__", None)
        if enter is not None and close is not None:
            enter()
            close(None, None, None)
        else:  # pragma: no cover - non-cm handle (defensive)
            closer = getattr(handle, "close", None)
            if callable(closer):
                closer()
    except Exception as exc:  # noqa: BLE001 — slow/absent camera is expected
        return PreflightResult(ok=False, camera=camera, error=str(exc))
    return PreflightResult(ok=True, camera=camera)


def open_camera_with_retry(
    cfg: SmokeConfig,
    opener: Callable[[str], Any],
    *,
    stop: Optional[dict] = None,
    on_state: Optional[Callable[[str, PreflightResult], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Optional[Any]:
    """Preflight-gate the camera with exponential backoff; return an open handle.

    Mirrors the ADS-B service's ``_preflight_gate``: a camera that isn't ready
    at startup (slow enumeration) is retried with backoff instead of crashing
    the node. Returns the open Camera handle on success, or None if stopped /
    out of attempts. ``on_state`` is called with (state, preflight) on each
    transition so the caller can self-report (no_camera while waiting).
    """
    stop = stop if stop is not None else {"flag": False}
    backoff = cfg.camera_retry_s
    attempt = 0
    while not stop.get("flag"):
        attempt += 1
        pf = preflight_camera(cfg.camera, opener)
        if pf.ok:
            try:
                handle = opener(cfg.camera)
            except Exception as exc:  # noqa: BLE001 — lost the race; back off
                pf = PreflightResult(ok=False, camera=cfg.camera, error=str(exc))
            else:
                LOG.info("camera ready: %s (attempt %d)", cfg.camera, attempt)
                return handle
        if on_state is not None:
            on_state(ST_NO_CAMERA, pf)
        if cfg.camera_attempts and attempt >= cfg.camera_attempts:
            LOG.error(
                "camera %s not ready after %d attempts (%s) — giving up",
                cfg.camera, attempt, pf.error,
            )
            return None
        LOG.warning(
            "camera %s not ready (%s); retry %d in %.0fs",
            cfg.camera, pf.error, attempt, backoff,
        )
        sleep(backoff)
        backoff = min(cfg.camera_max_retry_s, backoff * 2.0)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Status self-report (retained env.smoke.status — "the node is alive")
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Counters:
    frames: int = 0
    changed: int = 0
    escalated: int = 0
    alerts: int = 0
    heartbeats: int = 0                            # forced periodic escalations
    last_frame_monotonic: Optional[float] = None  # time.monotonic() of last frame
    last_heartbeat_monotonic: Optional[float] = None  # last heartbeat escalation
    last_max_tile_score: float = 0.0              # most recent tiled max score
    started_monotonic: float = field(default_factory=time.monotonic)


def status_payload(
    cfg: SmokeConfig,
    state: str,
    counters: Counters,
    *,
    vlm_backend: str = "none",
    now: Callable[[], float] = time.monotonic,
    preflight: Optional[PreflightResult] = None,
) -> dict:
    """Build the retained status self-report body. Pure — no I/O, never raises.

    Shape (so a node is visibly alive with no smoke to report):
        state            watching | no_camera | escalating | starting | stopped
        last_frame_age_s seconds since the last processed frame (None if none)
        vlm              backend name (none/apple/node/ollama)
        frames/alerts    monotonic counters since start
        post             whether real publishing is enabled (dry-run if False)
        detector         the active cascade (tiled grid / baseline) + heartbeat
        last_max_tile_score / last_heartbeat_age_s  liveness of the tiled path
    """
    last_age: Optional[float] = None
    if counters.last_frame_monotonic is not None:
        last_age = round(max(0.0, now() - counters.last_frame_monotonic), 1)
    hb_age: Optional[float] = None
    if cfg.heartbeat_s and counters.last_heartbeat_monotonic is not None:
        hb_age = round(max(0.0, now() - counters.last_heartbeat_monotonic), 1)
    body = {
        "service": "smoke_spotter",
        "state": state,
        "camera": cfg.camera,
        "last_frame_age_s": last_age,
        "vlm": vlm_backend,
        "post": cfg.post,
        "uptime_s": round(max(0.0, now() - counters.started_monotonic), 1),
        "interval_s": cfg.interval_s,
        # Which cascade is running + the localized-plume signal's last value.
        "detector": {
            "tiled": cfg.tiled,
            "grid": [cfg.grid_rows, cfg.grid_cols] if cfg.tiled else None,
            "baseline": cfg.baseline if cfg.tiled else None,
            "baseline_alpha": cfg.baseline_alpha if cfg.tiled else None,
            "heartbeat_s": cfg.heartbeat_s,
        },
        "last_max_tile_score": round(float(counters.last_max_tile_score), 4),
        "last_heartbeat_age_s": hb_age,
        "heartbeats": counters.heartbeats,
        "counters": {
            # Kept backward-compatible (a strict downstream parser checks this
            # exact set); heartbeat count is surfaced at the top level instead.
            "frames": counters.frames,
            "changed": counters.changed,
            "escalated": counters.escalated,
            "alerts": counters.alerts,
        },
    }
    if preflight is not None:
        body["camera_preflight"] = preflight.as_dict()
    return body


def publish_status(
    plugin,
    cfg: SmokeConfig,
    state: str,
    counters: Counters,
    *,
    vlm_backend: str = "none",
    preflight: Optional[PreflightResult] = None,
) -> dict:
    """Emit the retained status self-report. Best-effort: never crashes the loop.

    Honours the --post gate exactly like the measurement path: with publishing
    OFF we LOG the status payload we WOULD publish; ON, we publish for real.
    Returns the payload (so tests/dry-run can assert its shape offline).
    """
    body = status_payload(cfg, state, counters, vlm_backend=vlm_backend,
                          preflight=preflight)
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True)
    if not cfg.post or plugin is None:
        LOG.info("[dry-run] WOULD publish %s = %s", STATUS_NAME, encoded)
        return body
    try:
        plugin.publish(STATUS_NAME, encoded, meta={"camera": cfg.camera})
    except Exception as exc:  # noqa: BLE001 — status is best-effort
        LOG.warning("status publish failed: %s", exc)
    return body


# ─────────────────────────────────────────────────────────────────────────────
# Core per-frame processing (shared shape with dev_run for parity)
# ─────────────────────────────────────────────────────────────────────────────

def build_detector(cfg: SmokeConfig):
    """Build the stateful detector for the configured cascade.

    Returns a ``smoke.TiledDetector`` (the default — fixes the static-camera +
    localized-plume blind spot) or, when ``cfg.tiled`` is off, the original
    global ``smoke.ChangeGate`` (kept for backward-compat / A-B).
    """
    if cfg.tiled:
        return smoke_mod.TiledDetector(
            rows=cfg.grid_rows,
            cols=cfg.grid_cols,
            baseline=cfg.baseline,
            alpha=cfg.baseline_alpha,
            tile_change_threshold=cfg.tile_change_threshold,
            tile_smoke_alert=cfg.tile_smoke_alert,
        )
    return smoke_mod.ChangeGate(threshold=cfg.change_threshold)


def process_frame(
    sample,
    detector,
    backend,
    *,
    smoke_alert: float,
    vlm_threshold: float,
    force_escalate: bool = False,
    vlm_deadline_s: Optional[float] = None,
) -> dict:
    """Run the cheap cascade -> (maybe) VLM for one frame. Returns a verdict dict.

    Drives EITHER the TILED + rolling-baseline detector (a ``TiledDetector`` —
    the default fix for the static-camera + localized-plume blind spot) OR the
    legacy global ``ChangeGate`` (back-compat), dispatched by ``detector`` type.

    ``force_escalate`` is the periodic HEARTBEAT: when True we escalate to the
    VLM even if no tile tripped, so a static scene is still confirmed on cadence
    (bounds the worst-case miss to one heartbeat interval). Does NO publishing or
    uploading — the caller owns side effects (reused by the offline dev runner).

    ``vlm_deadline_s`` (when set) wraps the VLM call in a HARD wall-clock watchdog
    so a wedged model can't block the loop (see ``detect_with_deadline``); ``None``
    calls the backend directly (the default, so unit tests of the cascade stay
    synchronous). The node loop passes a real deadline.
    """
    frame = sample.data  # numpy RGB array (HxWx3 uint8)
    verdict = {
        "ts": int(getattr(sample, "timestamp", 0) or 0),
        "changed": False,
        "diff": 0.0,
        "smoke_score": 0.0,
        "subscores": {},
        "alert": False,
        "escalated": False,
        "vlm": "skipped",
        "vlm_detail": None,
        # tiled extras (always present; 0/None on the global path)
        "tiled": isinstance(detector, smoke_mod.TiledDetector),
        "max_tile_score": 0.0,
        "tile_bbox": None,
        "any_tile_alert": False,
        "heartbeat": bool(force_escalate),
    }

    if isinstance(detector, smoke_mod.TiledDetector):
        tv = detector.update(frame)
        verdict["changed"] = bool(tv["changed"])
        verdict["diff"] = float(tv["max_tile_change"])
        verdict["max_tile_score"] = float(tv["max_tile_score"])
        verdict["tile_bbox"] = tv["bbox"]
        verdict["any_tile_alert"] = bool(tv["any_tile_alert"])
        verdict["subscores"] = {
            "grid": tv["grid"],
            "max_tile_score": tv["max_tile_score"],
            "mean_tile_score": tv["mean_tile_score"],
            "n_tiles_alert": tv["n_tiles_alert"],
            "warming_up": tv["warming_up"],
        }
        # Image-level smoke_score = the MAX over tiles (localized plume drives it).
        # Post-fix this max is taken over tiles that ALERTED (persistent + risen
        # above their own haze baseline), so a clear hazy scene reads ~0.
        score = float(tv["max_tile_score"])
        verdict["smoke_score"] = score
        # Gate on the genuine persistent tile alert AND the image-score bar —
        # identical to benchmark.py's would_alert. (Codex audit fix: the old
        # `or score>=smoke_alert` let a merely-CHANGED cloud tile's fallback
        # score raise an alert with no rise/persist gate — a false positive the
        # benchmark never saw because it used the stricter AND.)
        alert = bool(tv["any_tile_alert"]) and score >= smoke_alert
        # Escalate ONLY on a genuine persistent tile alert (moved + risen above
        # baseline + persisted) crossing the VLM threshold, OR on the heartbeat.
        # Keying off ``any_tile_alert`` (not raw ``changed``) is the false-positive
        # fix: drifting cirrus moves tiles every frame but does not produce a
        # persistent self-relative haze rise, so it no longer escalates.
        should_escalate = (
            (bool(tv["any_tile_alert"]) and score >= vlm_threshold)
            or force_escalate
        )
    else:
        # Legacy GLOBAL path (back-compat): whole-frame change-gate + smoke_score.
        changed, diff = detector.update(frame)
        verdict["changed"] = bool(changed)
        verdict["diff"] = round(float(diff), 4)
        if not changed and not force_escalate:
            return verdict  # cheap path wins — no smoke screen, no VLM
        subs = smoke_mod.smoke_subscores(frame)
        score = subs["score"]
        verdict["smoke_score"] = score
        verdict["subscores"] = subs
        verdict["max_tile_score"] = score  # parity field
        alert = score >= smoke_alert
        should_escalate = (changed and score >= vlm_threshold) or force_escalate

    if should_escalate:
        verdict["escalated"] = True
        img_bytes = _encode_for_vlm(sample)
        if img_bytes is not None:
            if vlm_deadline_s and vlm_deadline_s > 0:
                res = detect_with_deadline(backend, img_bytes, vlm_deadline_s)
            else:
                res = backend.detect(img_bytes)
            verdict["vlm"] = res.label  # smoke | clear | unknown
            verdict["vlm_detail"] = res.detail
            if res.smoke is True:
                alert = True
        else:
            verdict["vlm"] = "unknown"
    verdict["alert"] = bool(alert)
    return verdict


# ─────────────────────────────────────────────────────────────────────────────
# Measurement publishing (honours the --post dry-run gate)
# ─────────────────────────────────────────────────────────────────────────────

def publish_measurements(
    plugin,
    cfg: SmokeConfig,
    verdict: dict,
    sample,
) -> list:
    """Publish (or, in dry-run, LOG) the per-frame measurements + alert upload.

    With ``cfg.post`` OFF (the default until on a node) this performs NO
    plugin.publish / plugin.upload_file calls — it logs the exact payload it
    WOULD send so bring-up can review it (mirrors firewatch_vision's --post).
    Returns the list of (name, value, meta) tuples it published/would-publish.
    """
    ts = verdict["ts"] or None  # ns since epoch (from the frame)
    cam_meta = {"camera": cfg.camera}
    pending = [
        ("env.smoke.changed", int(verdict["changed"]), cam_meta),
        ("env.smoke.score", float(verdict["smoke_score"]), cam_meta),
        ("env.smoke.alert", int(verdict["alert"]), cam_meta),
    ]
    # Tiled cascade extras: the localized-plume signal (max over tiles) + which
    # tile won (bbox as "y0,x0,y1,x1" str meta). Only on the tiled path.
    if verdict.get("tiled"):
        bbox = verdict.get("tile_bbox")
        tile_meta = {"camera": cfg.camera}
        if bbox:
            tile_meta["bbox"] = ",".join(str(int(v)) for v in bbox)
        pending.append(
            ("env.smoke.max_tile_score", float(verdict.get("max_tile_score", 0.0)),
             tile_meta)
        )
    if verdict["escalated"]:
        pending.append((
            "env.smoke.vlm", str(verdict["vlm"]),
            {"camera": cfg.camera, "detail": verdict["vlm_detail"] or ""},
        ))

    if not cfg.post or plugin is None:
        for name, value, meta in pending:
            LOG.info("[dry-run] WOULD publish %s = %r meta=%s", name, value, meta)
        if verdict["alert"]:
            LOG.info("[dry-run] WOULD upload snapshot (alert) meta=%s",
                     {"smoke_score": str(verdict["smoke_score"]),
                      "vlm": str(verdict["vlm"])})
        return pending

    for name, value, meta in pending:
        try:
            plugin.publish(name, value, meta=meta, timestamp=ts)
        except Exception as exc:  # noqa: BLE001 — one bad publish must not kill us
            LOG.warning("publish %s failed: %s", name, exc)

    if verdict["alert"]:
        path = _save_snapshot(sample)
        if path is not None:
            try:
                # keep=True: pywaggle's Uploader defaults to keep=False, which
                # DELETES our source file itself — our finally: os.remove(path)
                # would then hit OSError. keep=True leaves cleanup to us (and is
                # a no-op on deletion in the PYWAGGLE_LOG_DIR dev path where the
                # real uploader is disabled). meta values must be str (str(...)).
                plugin.upload_file(
                    path,
                    meta={
                        "camera": cfg.camera,
                        "smoke_score": str(verdict["smoke_score"]),
                        "vlm": str(verdict["vlm"]),
                    },
                    timestamp=ts,
                    keep=True,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("snapshot upload failed: %s", exc)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
    return pending


# ─────────────────────────────────────────────────────────────────────────────
# Heartbeat: periodic forced VLM escalation (bounds misses on a static scene)
# ─────────────────────────────────────────────────────────────────────────────

def heartbeat_due(
    cfg: SmokeConfig,
    last_heartbeat_monotonic: Optional[float],
    *,
    now: float,
) -> bool:
    """Is a forced (heartbeat) VLM escalation due?

    True when heartbeats are enabled (``cfg.heartbeat_s > 0``) AND either none has
    fired yet (the FIRST frame escalates so a static scene is confirmed at start)
    OR at least ``heartbeat_s`` has elapsed since the last one. Pure / no I/O so
    the cadence is unit-testable with an injected clock.
    """
    if not cfg.heartbeat_s or cfg.heartbeat_s <= 0:
        return False
    if last_heartbeat_monotonic is None:
        return True
    return (now - last_heartbeat_monotonic) >= cfg.heartbeat_s


# ─────────────────────────────────────────────────────────────────────────────
# Shared runtime setup + one-frame handler (used by BOTH the node + ingest loops
# so a frame is processed/published/counted identically no matter the source)
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_runtime(cfg: SmokeConfig):
    """Build the VLM backend + detector and log the chosen cascade. Returns
    ``(backend, backend_name, detector)``. Shared by the node + ingest paths."""
    deps = smoke_mod.deps_available()
    if not deps["numpy"]:
        LOG.warning("numpy unavailable — smoke_score will be 0.0 (no escalation)")

    backend = vlm_mod.build_backend(cfg.vlm_kind, timeout=cfg.vlm_timeout_s)
    backend_name = getattr(backend, "name", "none")
    LOG.info("VLM backend: %s", backend_name)
    if not cfg.post:
        LOG.warning(
            "SMOKE_POST is OFF — DRY-RUN: measurements will be LOGGED, not "
            "published. Set --post (or SMOKE_POST=1) on a node to publish."
        )

    detector = build_detector(cfg)
    if cfg.tiled:
        LOG.info(
            "tiled cascade: grid=%dx%d baseline=%s alpha=%.3f tile_change>=%.1f "
            "tile_alert>=%.2f heartbeat=%.0fs",
            cfg.grid_rows, cfg.grid_cols, cfg.baseline, cfg.baseline_alpha,
            cfg.tile_change_threshold, cfg.tile_smoke_alert, cfg.heartbeat_s,
        )
    else:
        LOG.info("global cascade (tiled OFF): change_threshold=%.1f heartbeat=%.0fs",
                 cfg.change_threshold, cfg.heartbeat_s)
    return backend, backend_name, detector


def _install_stop_handlers() -> dict:
    """Install SIGINT/SIGTERM handlers that flip a shared stop flag (graceful
    shutdown after the current frame). Returns the ``{'flag': False}`` cell."""
    stop = {"flag": False}

    def _handle(signum, _frame):
        LOG.info("signal %s received — shutting down after current frame", signum)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)
    return stop


def _handle_frame(
    sample,
    *,
    cfg: SmokeConfig,
    detector,
    backend,
    counters: Counters,
    plugin,
    status_cb: Callable[[str], None],
    last_status: list,
    tracker=None,
) -> dict:
    """Process + publish ONE frame, update counters, and self-report. The single
    per-frame code path both the node and ingest loops call, so the two sources
    can never drift in how a frame is scored, escalated, published, or counted.
    Returns the verdict dict.
    """
    t0 = time.time()
    # HEARTBEAT: force a VLM escalation on cadence even with no tile trip, so a
    # static scene is still confirmed (bounds the miss to one interval).
    force = heartbeat_due(
        cfg, counters.last_heartbeat_monotonic, now=time.monotonic()
    )
    verdict = process_frame(
        sample, detector, backend,
        smoke_alert=cfg.smoke_alert,
        vlm_threshold=cfg.vlm_threshold,
        force_escalate=force,
        # Hard wall-clock backstop ABOVE the backend's own socket timeout, for
        # the half-stalled-stream case urllib won't catch.
        vlm_deadline_s=cfg.vlm_timeout_s + 30.0,
    )

    publish_measurements(plugin, cfg, verdict, sample)

    # Ember operational plane: aggregate the verdict into the candidate Event
    # graph (pure/local; never raises; the learning plane replays it at loop exit).
    if tracker is not None:
        tracker.feed(verdict, submission_time=time.time())

    counters.frames += 1
    counters.changed += int(verdict["changed"])
    counters.escalated += int(verdict["escalated"])
    counters.alerts += int(verdict["alert"])
    counters.last_max_tile_score = float(verdict.get("max_tile_score", 0.0))
    counters.last_frame_monotonic = time.monotonic()
    if force:
        counters.heartbeats += 1
        counters.last_heartbeat_monotonic = counters.last_frame_monotonic

    LOG.info(
        "frame#%d changed=%s smoke=%.3f tile=%.3f vlm=%s alert=%s hb=%s (%.0fms)",
        counters.frames, verdict["changed"], verdict["smoke_score"],
        verdict.get("max_tile_score", 0.0), verdict["vlm"],
        verdict["alert"], force, (time.time() - t0) * 1000.0,
    )

    # Self-report: escalating on an alert, else periodic "watching".
    if verdict["alert"] or verdict["escalated"]:
        status_cb(ST_ESCALATING)
    elif (time.monotonic() - last_status[0]) >= cfg.status_interval_s:
        status_cb(ST_WATCHING)
    return verdict


def _make_tracker(cfg: SmokeConfig):
    """Build the Ember operational EventTracker (or None if disabled / unavailable).
    Lazy-imported so the loop carries no extra weight when ember is off; never
    raises (a failure just disables aggregation, the loop runs normally)."""
    if not getattr(cfg, "ember", False):
        return None
    try:
        from app.sensornet.operational import EventTracker
        return EventTracker(sensor_id=str(cfg.camera), lat=cfg.lat, lon=cfg.lon)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("ember: could not start the evidence tracker (%s) — disabled", exc)
        return None


def _finish_learning(cfg: SmokeConfig, tracker) -> None:
    """Learning plane, run ONCE after the operational loop exits: close any open
    candidate, replay all closed events into lessons, log what was missing. Never
    raises (learning failures must not affect the operational shutdown)."""
    if tracker is None:
        return
    from app.sensornet.planes import PlaneViolation
    try:
        tracker.close()                       # force-close the open candidate
        events = tracker.drain_closed_events()  # hand off + clear (bounded; no re-replay)
        if not events:
            return
        from app.sensornet.learning import run_learning
        lp = run_learning(events)             # enters the LEARNING plane
        s = lp.summary()
        LOG.info("ember learning: replayed %d event(s), %d lesson(s); missing=%s",
                 s["events_replayed"], s["total_lessons"], s["missing_by_kind"])
    except PlaneViolation:
        raise                                 # a plane-invariant breach is a bug: surface it
    except Exception as exc:  # noqa: BLE001 — a learning failure must not break shutdown
        LOG.warning("ember learning pass failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# The plugin loop (long-running service: preflight-gated, self-healing)
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: SmokeConfig) -> int:
    # An http(s)/rtsp URL or a ``discover`` keyword takes the ingest path (the
    # plugin grabs frames itself via app/ingest_normalize); pywaggle is optional
    # there. A bare node-shim id / file:// clip takes the node Camera path below.
    if is_ingest_camera(cfg.camera):
        return _run_ingest(cfg)

    Plugin, Camera = _import_waggle()
    backend, backend_name, detector = _prepare_runtime(cfg)
    stop = _install_stop_handlers()
    counters = Counters()
    last_status = [0.0]  # mutable cell so the inner helper can update it
    tracker = _make_tracker(cfg)   # Ember operational evidence aggregation

    try:
        with Plugin() as plugin:
            def _status(state: str, preflight: Optional[PreflightResult] = None) -> None:
                publish_status(plugin, cfg, state, counters,
                              vlm_backend=backend_name, preflight=preflight)
                last_status[0] = time.monotonic()

            _status(ST_STARTING)

            # --- camera preflight + retry-gate: self-heal a slow-enumerating cam ---
            camera = open_camera_with_retry(
                cfg, Camera, stop=stop, on_state=lambda st, pf: _status(st, pf)
            )
            if camera is None:
                LOG.error("no camera — exiting")
                _status(ST_STOPPED)
                return 0 if stop["flag"] else 1

            with camera:
                LOG.info("smoke-spotter running on camera=%s interval=%.1fs post=%s",
                         cfg.camera, cfg.interval_s, cfg.post)
                _status(ST_WATCHING)

                if cfg.is_file:
                    frame_iter = camera.stream()
                else:
                    frame_iter = _live_snapshots(camera, cfg.interval_s, stop)

                for sample in frame_iter:
                    if stop["flag"]:
                        break
                    _handle_frame(
                        sample, cfg=cfg, detector=detector, backend=backend,
                        counters=counters, plugin=plugin, status_cb=_status,
                        last_status=last_status, tracker=tracker,
                    )

            _status(ST_STOPPED)

        LOG.info("done: %d frames | %d changed | %d escalated | %d alerts",
                 counters.frames, counters.changed, counters.escalated, counters.alerts)
        return 0
    finally:
        # learning plane runs even if the loop raised, so a candidate is never
        # left open/unreplayed (codex review). Runs in OPERATIONAL (loop exit).
        _finish_learning(cfg, tracker)


def _run_ingest(cfg: SmokeConfig) -> int:
    """Run the watch loop against a URL / discovered camera via the ingest path.

    Same cheap->expensive cascade + publish contract as the node loop (it reuses
    _prepare_runtime + _handle_frame), but the frame source is ingest_frames (an
    http still or an ffmpeg keyframe) and pywaggle is OPTIONAL: when it is not
    importable we run with ``plugin=None`` (dry-run only — publish_measurements /
    publish_status already log the payload they WOULD publish in that case). This
    is what lets the plugin watch a public ALERTCalifornia camera off-node.
    """
    backend, backend_name, detector = _prepare_runtime(cfg)
    stop = _install_stop_handlers()
    counters = Counters()
    last_status = [0.0]
    tracker = _make_tracker(cfg)   # Ember operational evidence aggregation

    # pywaggle is optional on the ingest path: a real Plugin() on a node, else a
    # null context yielding None (dry-run). _import_waggle would SystemExit; here
    # we degrade instead so the path runs anywhere. We catch NOT just ImportError
    # (pywaggle absent) but ANY Plugin() construction failure (installed but the
    # node runtime is unavailable / misconfigured) — a crash here would violate
    # the degrade-gracefully + offline-first contract. Both cases fall back to a
    # null context (dry-run), each with its own distinct, honest warning.
    plugin_cm = contextlib.nullcontext(None)
    have_pywaggle = False
    try:
        from waggle.plugin import Plugin  # type: ignore
        plugin_cm = Plugin()
        have_pywaggle = True
    except ImportError:
        LOG.warning("pywaggle not installed — ingest runs in DRY-RUN (no publish). "
                    "Install 'pywaggle[all]' on a node to publish for real.")
    except Exception as exc:  # noqa: BLE001 — installed but Plugin() init failed
        LOG.warning("pywaggle present but Plugin() init failed (%s) — DRY-RUN "
                    "(no publish). Off-node or misconfigured node runtime.", exc)
    if have_pywaggle and not cfg.post:
        LOG.info("pywaggle present but SMOKE_POST is OFF — dry-run.")

    # An ExitStack so a Plugin().__enter__() failure (installed but the node
    # runtime can't actually start the messaging loop) also degrades to dry-run
    # instead of crashing — not just the Plugin() construction failure above
    # (codex review).
    try:
        with contextlib.ExitStack() as stack:
            try:
                plugin = stack.enter_context(plugin_cm)
            except Exception as exc:  # noqa: BLE001 — __enter__ failed -> dry-run
                LOG.warning("pywaggle Plugin().__enter__ failed (%s) — DRY-RUN "
                            "(no publish).", exc)
                plugin = None
                have_pywaggle = False

            def _status(state: str, preflight: Optional[PreflightResult] = None) -> None:
                publish_status(plugin, cfg, state, counters,
                              vlm_backend=backend_name, preflight=preflight)
                last_status[0] = time.monotonic()

            _status(ST_STARTING)
            LOG.info("smoke-spotter (ingest) camera=%s interval=%.1fs post=%s pywaggle=%s",
                     cfg.camera, cfg.interval_s, cfg.post, have_pywaggle)
            _status(ST_WATCHING)

            for sample in ingest_frames(cfg, stop):
                if stop["flag"]:
                    break
                _handle_frame(
                    sample, cfg=cfg, detector=detector, backend=backend,
                    counters=counters, plugin=plugin, status_cb=_status,
                    last_status=last_status, tracker=tracker,
                )

            _status(ST_STOPPED)

        LOG.info("done (ingest): %d frames | %d changed | %d escalated | %d alerts",
                 counters.frames, counters.changed, counters.escalated, counters.alerts)
        return 0
    finally:
        # learning plane runs even if the loop raised (codex review).
        _finish_learning(cfg, tracker)


def _live_snapshots(camera, interval: float, stop: dict):
    """Yield ``snapshot()`` samples forever on ``interval`` (live cam / RTSP)."""
    while not stop["flag"]:
        try:
            yield camera.snapshot()
        except Exception as exc:  # transient camera errors shouldn't kill us
            LOG.warning("snapshot failed: %s — retrying after interval", exc)
        # Sleep in small slices so a stop signal is honoured promptly.
        slept = 0.0
        while slept < interval and not stop["flag"]:
            step = min(0.5, interval - slept)
            time.sleep(step)
            slept += step


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    """argparse parser. Numeric flags are parsed as raw strings and coerced in
    ``load_config`` so a bad CLI value degrades to a default instead of aborting
    node bring-up. Defaults are None so we can tell "user set it" from "env/default".
    """
    p = argparse.ArgumentParser(
        description="Smoke Spotter — Waggle wildfire-smoke detection plugin",
    )
    p.add_argument(
        "--camera", default=None,
        help="frame source: a node data-shim id (bottom_camera) or file://clip.mp4 "
             "(node Camera path), OR an http(s):// still / rtsp:// stream / *.m3u8 "
             "playlist / the keyword 'discover' (ingest path — grabs frames itself, "
             f"pywaggle optional) (env SMOKE_CAMERA; default: {DEFAULT_CAMERA})",
    )
    p.add_argument(
        "--interval", default=None,
        help=f"seconds between frames for a live camera (env SMOKE_INTERVAL; "
             f"default: {DEFAULT_INTERVAL})",
    )
    th = p.add_argument_group(
        "thresholds (calibrated vs FIgLib — docs/research/threshold-calibration.md; "
        "defaults held pending Pauma dynamic-scene cross-validation)")
    th.add_argument(
        "--change-threshold", dest="change_threshold", default=None,
        help="mean-abs grayscale diff (0..255) to count a frame as changed "
             "(env SMOKE_CHANGE_THRESHOLD)",
    )
    th.add_argument(
        "--smoke-alert", dest="smoke_alert", default=None,
        help="smoke_score [0..1] at/above which alert=true (env SMOKE_ALERT)",
    )
    th.add_argument(
        "--vlm-threshold", dest="vlm_threshold", default=None,
        help="smoke_score [0..1] at/above which to escalate to the VLM "
             "(env SMOKE_VLM_THRESHOLD)",
    )
    v = p.add_argument_group("VLM (optional, degrades to 'unknown')")
    v.add_argument(
        "--vlm", default=None,
        help="VLM backend: auto|apple|node|ollama|none "
             "(env SMOKE_VLM; default: auto)",
    )
    v.add_argument(
        "--vlm-timeout", dest="vlm_timeout", default=None,
        help="VLM request timeout seconds (env SMOKE_VLM_TIMEOUT)",
    )
    cam = p.add_argument_group("camera preflight / retry-gate")
    cam.add_argument(
        "--camera-retry", dest="camera_retry", default=None,
        help="seconds to wait before retrying a not-ready camera "
             f"(env SMOKE_CAMERA_RETRY_S; default {DEFAULT_CAMERA_RETRY_S}, "
             "doubles up to the cap)",
    )
    cam.add_argument(
        "--camera-max-retry", dest="camera_max_retry", default=None,
        help="backoff cap in seconds (env SMOKE_CAMERA_MAX_RETRY_S; "
             f"default {DEFAULT_CAMERA_MAX_RETRY_S})",
    )
    cam.add_argument(
        "--camera-attempts", dest="camera_attempts", default=None,
        help="max camera-open attempts; 0 = retry forever "
             f"(env SMOKE_CAMERA_ATTEMPTS; default {DEFAULT_CAMERA_ATTEMPTS})",
    )
    st = p.add_argument_group("status self-report")
    st.add_argument(
        "--status-interval", dest="status_interval", default=None,
        help="seconds between retained env.smoke.status self-reports "
             f"(env SMOKE_STATUS_INTERVAL_S; default {DEFAULT_STATUS_INTERVAL_S})",
    )
    ing = p.add_argument_group(
        "ingest source (when --camera is an http(s)/rtsp URL or 'discover')")
    ing.add_argument(
        "--ingest-timeout", dest="ingest_timeout", default=None,
        help="per-frame still-fetch / ffmpeg-keyframe deadline in seconds "
             f"(env SMOKE_INGEST_TIMEOUT_S; default {DEFAULT_INGEST_TIMEOUT_S})",
    )
    ing.add_argument(
        "--relay-base", dest="relay_base", default=None,
        help="optional ingest relay base URL (republished HLS / jpeg proxy) "
             "(env SMOKE_RELAY_BASE; default none)",
    )
    ing.add_argument(
        "--lat", dest="lat", default=None,
        help="discovery centre latitude for --camera discover "
             f"(env SMOKE_LAT; default {DEFAULT_LAT} = UC Davis)",
    )
    ing.add_argument(
        "--lon", dest="lon", default=None,
        help="discovery centre longitude for --camera discover "
             f"(env SMOKE_LON; default {DEFAULT_LON})",
    )
    ing.add_argument(
        "--radius", dest="radius", default=None,
        help="discovery radius in miles for --camera discover "
             f"(env SMOKE_RADIUS_MI; default {DEFAULT_RADIUS_MI})",
    )
    td = p.add_argument_group(
        "tiled cascade (DEFAULT — fixes the static-camera + localized-plume miss)")
    td.add_argument(
        "--no-tiled", dest="no_tiled", action="store_true", default=None,
        help="disable the tiled+rolling-baseline cascade and use the legacy "
             "global change-gate + smoke_score path (env SMOKE_TILED=0)",
    )
    td.add_argument(
        "--grid-rows", dest="grid_rows", default=None,
        help=f"tile grid rows (env SMOKE_GRID_ROWS; default {DEFAULT_GRID_ROWS})",
    )
    td.add_argument(
        "--grid-cols", dest="grid_cols", default=None,
        help=f"tile grid cols (env SMOKE_GRID_COLS; default {DEFAULT_GRID_COLS})",
    )
    td.add_argument(
        "--baseline", dest="baseline", default=None,
        help="per-tile baseline mode: ema (rolling, accumulates slow growth) or "
             f"frame (immediate prev-frame diff) (env SMOKE_BASELINE; "
             f"default {DEFAULT_BASELINE})",
    )
    td.add_argument(
        "--baseline-alpha", dest="baseline_alpha", default=None,
        help="EMA background smoothing in (0,1]; smaller = slower background = "
             f"slow plumes accumulate longer (env SMOKE_BASELINE_ALPHA; "
             f"default {DEFAULT_BASELINE_ALPHA})",
    )
    td.add_argument(
        "--tile-change-threshold", dest="tile_change_threshold", default=None,
        help="per-tile mean-abs grey change vs baseline to count a tile changed "
             f"(env SMOKE_TILE_CHANGE_THRESHOLD; default {DEFAULT_TILE_CHANGE_THRESHOLD})",
    )
    td.add_argument(
        "--tile-alert", dest="tile_smoke_alert", default=None,
        help="per-tile haze score [0..1] at/above which a tile alerts "
             f"(env SMOKE_TILE_ALERT; default {DEFAULT_TILE_SMOKE_ALERT})",
    )
    td.add_argument(
        "--heartbeat", dest="heartbeat", default=None,
        help="seconds between FORCED periodic VLM escalations even with no tile "
             "trip, so a static scene is still confirmed (0 disables; "
             f"env SMOKE_HEARTBEAT_S; default {DEFAULT_HEARTBEAT_S})",
    )
    g = p.add_argument_group("publish gate (SAFE bring-up)")
    g.add_argument(
        "--post", action="store_true", default=None,
        help="actually publish to Beehive. OFF by default (env SMOKE_POST=1): "
             "dry-run LOGS the exact payload it WOULD publish instead.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return p


def main(argv: Optional[list] = None) -> int:
    cfg = load_config(argv)
    logging.basicConfig(
        level=logging.DEBUG if cfg.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
