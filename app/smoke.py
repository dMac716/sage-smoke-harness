#!/usr/bin/env python3
"""Pure-CPU smoke detection logic for the Smoke Spotter Waggle plugin.

This module is the cheap "decide where to look harder" brain. It has **no**
node, VLM, or network dependencies and is fully offline + unit-testable.

Doctrine (Sage/Waggle): *move the work, not the data.* The expensive
vision-language model is only ever invoked for the handful of frames whose
cheap CPU screen says "something changed AND it looks hazy." Everything in
this file is that cheap screen.

Two GLOBAL primitives (the original screen), both on small downscaled thumbnails:

  1. ``ChangeGate`` / ``frame_diff`` — downscaled grayscale mean-absolute
     per-pixel difference vs the previous frame. Unchanged frames short-circuit
     the whole pipeline (no smoke heuristic, no VLM).

  2. ``smoke_score`` — a transparent, UNCALIBRATED first-pass screen in [0, 1]
     blending three sub-scores (grey-haze fraction / reduced contrast /
     upper-band desaturation). It exists to *rank* frames for escalation, not
     to declare fire.

DESIGN-FLAW FIX (the static-camera + slow-localized-plume blind spot)
─────────────────────────────────────────────────────────────────────
The offline Pauma benchmark (docs/research/benchmark-vs-smokeynet.md) proved the
GLOBAL screen above is *blind* to the canonical fire-watch-tower scenario:

  (a) the change-gate compares frame-to-frame, so on a STATIC tripod camera the
      whole-frame diff stays below threshold and the gate NEVER opens; and
  (b) ``smoke_score`` is GLOBAL (whole-frame), so a small LOCALIZED plume is
      diluted by the unchanged majority of the scene and never moves the score.

The fix keeps everything cheap (numpy/PIL only, no ML) and adds two primitives
that operate PER TILE on a grid, while leaving the global API above untouched:

  3. ``TILED detection`` (``tiled_subscores`` / ``tile_grid``) — split the frame
     into a grid; compute a per-tile haze sub-score. The image-level score is the
     MAX over tiles (not the mean), so one hazy tile drives the verdict. Exposes
     ``max_tile_score``, the winning tile ``bbox``, and ``any_tile_alert``.

  4. ``ROLLING BASELINE`` (``TiledDetector``) — a per-tile EMA background instead
     of a pure frame-to-frame diff. SLOW growth on a static camera ACCUMULATES
     against the background and eventually exceeds the change threshold, where a
     frame-to-frame diff would stay flat. The immediate-diff mode is still
     available (``baseline="frame"``).

Inputs are accepted as numpy arrays (e.g. ``waggle`` ``ImageSample.data``, which
is RGB ``HxWx3`` uint8), PIL ``Image`` objects, or raw encoded image bytes
(JPEG/PNG). Everything degrades gracefully if numpy or Pillow is missing.

Ported from the operator's TowerWatch ``tools/firewatch_vision/producer.py``.
"""
from __future__ import annotations

import io
import logging
from typing import Any, List, Optional, Tuple

LOG = logging.getLogger("smoke_spotter.smoke")

# Both gates work on a DOWNSCALE x DOWNSCALE thumbnail. Small + fixed so the
# screen is O(1) regardless of source resolution.
DOWNSCALE = 64

# changed-gate default: mean-abs grayscale diff (0..255 scale) at/above which a
# frame counts as "changed". NOT calibrated — overridable by the caller.
DEFAULT_CHANGE_THRESHOLD = 8.0

# smoke_score sub-score blend weights (sum to 1.0). Documented inline below.
_W_SATURATION = 0.40  # grey-haze fraction (dominant signal)
_W_CONTRAST = 0.35    # reduced contrast
_W_UPPER = 0.25       # upper-band desaturation shift

# ── Tiled-detection defaults (the static-camera + localized-plume fix) ──
# Grid shape: split the DOWNSCALE thumbnail into GRID_ROWS x GRID_COLS tiles.
# 4x4 = 16 tiles is a good cheap default: a tile is ~16x16 px of the 64x64 thumb,
# which localizes a plume to ~1/16 of the frame so it isn't washed out globally.
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLS = 4

# A target tile EDGE in thumbnail pixels; callers may pass ``tile_px`` instead of
# explicit rows/cols and the grid is sized to roughly this many pixels per tile.
DEFAULT_TILE_PX = 16

# Rolling-baseline EMA smoothing. The per-tile background is updated as
#   bg <- (1 - alpha) * bg + alpha * cur
# Smaller alpha = slower background = slow plume growth accumulates LONGER before
# being absorbed into the background (so a creeping plume still trips). 0.05 means
# the background tracks at ~1/20th of the per-frame rate.
DEFAULT_BASELINE_ALPHA = 0.05

# Per-tile change threshold (0..255 mean-abs grey diff vs the tile's background).
# A localized plume only has to move ONE tile this far, not the whole frame, so
# this can be comparable to the global default and still be far more sensitive.
DEFAULT_TILE_CHANGE_THRESHOLD = 8.0

# Per-tile smoke sub-score at/above which that tile is considered "alerting".
DEFAULT_TILE_SMOKE_ALERT = 0.55

# ── Self-relative haze-RISE gate (the false-positive fix) ──────────────────────
# The original tiled alert tripped on ABSOLUTE per-tile haze (>= tile_smoke_alert)
# plus motion. On a real fire-watch scene that BACKFIRES: a clear hazy distant
# sky / muted terrain already scores ~0.4-0.8 on the grey-haze cue WITH ZERO FIRE
# (the cue is not smoke-specific), and drifting cirrus supplies the motion — so
# the global-cue alert fires on essentially every frame, including clear pre-fire
# frames (verified on Pauma: 20/24 frames, 17/20 winning tiles in the sky band).
#
# The fix screens on how much a tile's haze RISES above that tile's OWN rolling
# baseline, not its absolute value. A clear (even hazy) tile sits near its own
# baseline -> ~0 rise -> no alert; smoke appearing in a tile lifts it well above
# its history. ``DEFAULT_HAZE_RISE_ALERT`` is that minimum rise. Set to 0.0 to
# disable the rise gate and fall back to pure absolute-haze behaviour.
DEFAULT_HAZE_RISE_ALERT = 0.20

# Require a tile to satisfy (moved AND hazy-rise) for this many CONSECUTIVE frames
# before it counts as alerting. Drifting clouds flick a tile on for a frame or two
# then move on; a real plume grows and PERSISTS in place. 1 = no persistence
# requirement (single-frame trip, the original behaviour).
DEFAULT_PERSIST_FRAMES = 2

# EMA smoothing for the per-tile HAZE baseline (separate from the gray-background
# alpha). Same intent: a slow background so a rising plume keeps diverging from
# its own haze history before being absorbed.
DEFAULT_HAZE_BASELINE_ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency probes (never hard-fail at import time)
# ─────────────────────────────────────────────────────────────────────────────

def _try_import_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except Exception:  # pragma: no cover - environment dependent
        return None


def _try_import_pil():
    try:
        from PIL import Image  # type: ignore

        return Image
    except Exception:  # pragma: no cover - environment dependent
        return None


_NP = _try_import_numpy()
_PIL = _try_import_pil()


def deps_available() -> dict:
    """Report which optional deps are present (handy for dev/diagnostics)."""
    return {"numpy": _NP is not None, "pillow": _PIL is not None}


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

def clamp01(x: float) -> float:
    """Clamp to [0, 1]; NaN -> 0.0."""
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return 0.0
    if xf != xf:  # NaN
        return 0.0
    return 0.0 if xf < 0.0 else (1.0 if xf > 1.0 else xf)


def _is_numpy_array(obj: Any) -> bool:
    return _NP is not None and isinstance(obj, _NP.ndarray)


def _is_pil_image(obj: Any) -> bool:
    # _PIL is the PIL.Image *module*; the base class is _PIL.Image.
    return _PIL is not None and isinstance(obj, _PIL.Image)


# ─────────────────────────────────────────────────────────────────────────────
# Frame normalization: anything -> small RGB numpy thumbnail (HxWx3 uint8)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_bytes_to_rgb_array(data: bytes, size: int):
    """Decode encoded image bytes -> size x size x3 uint8 RGB array, or None."""
    if _PIL is None or _NP is None:
        return None
    try:
        img = _PIL.open(io.BytesIO(data)).convert("RGB").resize((size, size))
    except Exception as exc:
        LOG.warning("image decode failed: %s", exc)
        return None
    return _NP.asarray(img, dtype=_NP.uint8)


def _pil_to_rgb_array(img, size: int):
    if _NP is None:
        return None
    try:
        img = img.convert("RGB").resize((size, size))
    except Exception as exc:
        LOG.warning("PIL resize failed: %s", exc)
        return None
    return _NP.asarray(img, dtype=_NP.uint8)


def _array_to_rgb_thumb(arr, size: int):
    """Normalize an arbitrary numpy image array to size x size x3 uint8 RGB.

    Accepts HxW (gray), HxWx3 (RGB), or HxWx4 (RGBA). Uses PIL for the resize
    when available; otherwise falls back to numpy nearest-neighbour subsampling.
    """
    if _NP is None:
        return None
    a = arr
    # Squeeze a singleton channel dim, e.g. HxWx1.
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    # Grayscale -> stack to 3 channels.
    if a.ndim == 2:
        a = _NP.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    else:
        LOG.warning("unsupported array shape %r for smoke thumb", getattr(arr, "shape", None))
        return None

    # Normalize dtype to uint8 in 0..255.
    if a.dtype != _NP.uint8:
        af = a.astype("float32")
        mx = float(af.max()) if af.size else 0.0
        if mx <= 1.0:  # looks like 0..1 floats
            af = af * 255.0
        a = _NP.clip(af, 0, 255).astype(_NP.uint8)

    # Resize. Prefer PIL for quality; otherwise nearest-neighbour subsample.
    if _PIL is not None:
        try:
            img = _PIL.fromarray(a).resize((size, size))
            return _NP.asarray(img, dtype=_NP.uint8)
        except Exception:
            pass
    h, w = a.shape[0], a.shape[1]
    if h == 0 or w == 0:
        return None
    ys = (_NP.linspace(0, h - 1, size)).astype(_NP.int64)
    xs = (_NP.linspace(0, w - 1, size)).astype(_NP.int64)
    return a[_NP.ix_(ys, xs)].astype(_NP.uint8)


def to_rgb_thumb(frame: Any, size: int = DOWNSCALE):
    """Coerce a frame (numpy array | PIL image | encoded bytes) to a small RGB
    uint8 numpy thumbnail (size x size x 3). Returns None if it can't.
    """
    if frame is None:
        return None
    if _is_numpy_array(frame):
        return _array_to_rgb_thumb(frame, size)
    if _is_pil_image(frame):
        return _pil_to_rgb_array(frame, size)
    if isinstance(frame, (bytes, bytearray)):
        return _decode_bytes_to_rgb_array(bytes(frame), size)
    LOG.warning("unsupported frame type %r", type(frame))
    return None


def _rgb_thumb_to_gray(thumb):
    """size x size x3 uint8 RGB -> size x size float32 luma (Rec. 601)."""
    if thumb is None or _NP is None:
        return None
    r = thumb[:, :, 0].astype("float32")
    g = thumb[:, :, 1].astype("float32")
    b = thumb[:, :, 2].astype("float32")
    return 0.299 * r + 0.587 * g + 0.114 * b


def to_gray_thumb(frame: Any, size: int = DOWNSCALE):
    """Coerce a frame to a size x size float32 grayscale thumbnail, or None."""
    return _rgb_thumb_to_gray(to_rgb_thumb(frame, size))


# ─────────────────────────────────────────────────────────────────────────────
# CHANGED-GATE (cheap, CPU)
# ─────────────────────────────────────────────────────────────────────────────

def frame_diff(prev: Any, cur: Any, size: int = DOWNSCALE) -> float:
    """Mean absolute per-pixel grayscale difference (0..255 scale).

    Returns 255.0 when there is no previous frame (treat as fully changed) so a
    camera's first frame always escalates. Returns 0.0 if the current frame
    can't be decoded (conservative: nothing to compare, do not falsely alert).
    """
    cur_g = to_gray_thumb(cur, size)
    if cur_g is None:
        # Last-ditch: if both are raw bytes and PIL is missing, fall back to
        # byte (in)equality so we never silently skip a changed frame.
        if isinstance(prev, (bytes, bytearray)) and isinstance(cur, (bytes, bytearray)):
            return 0.0 if bytes(prev) == bytes(cur) else 255.0
        return 0.0
    if prev is None:
        return 255.0
    prev_g = to_gray_thumb(prev, size)
    if prev_g is None:
        return 255.0
    if _NP is not None:
        return float(_NP.mean(_NP.abs(cur_g - prev_g)))
    # Pure-python fallback (numpy absent but somehow we have lists).
    flat_c: List[float] = list(cur_g)
    flat_p: List[float] = list(prev_g)
    n = min(len(flat_c), len(flat_p)) or 1
    return sum(abs(flat_c[i] - flat_p[i]) for i in range(n)) / float(n)


class ChangeGate:
    """Stateful change detector: remembers the previous frame's gray thumbnail.

    Usage::

        gate = ChangeGate(threshold=8.0)
        changed, diff = gate.update(frame)   # first call -> always changed

    The gate stores only a tiny (size x size) thumbnail, never the full frame.
    """

    def __init__(self, threshold: float = DEFAULT_CHANGE_THRESHOLD,
                 size: int = DOWNSCALE):
        self.threshold = float(threshold)
        self.size = int(size)
        self._prev_gray = None  # type: Optional[Any]

    def reset(self) -> None:
        self._prev_gray = None

    def update(self, frame: Any) -> Tuple[bool, float]:
        """Feed a frame; return (changed, diff). Updates internal prev frame."""
        cur_g = to_gray_thumb(frame, self.size)
        if cur_g is None:
            return False, 0.0
        if self._prev_gray is None:
            self._prev_gray = cur_g
            return True, 255.0
        if _NP is not None:
            diff = float(_NP.mean(_NP.abs(cur_g - self._prev_gray)))
        else:  # pragma: no cover - numpy effectively required for arrays
            a = list(cur_g)
            b = list(self._prev_gray)
            n = min(len(a), len(b)) or 1
            diff = sum(abs(a[i] - b[i]) for i in range(n)) / float(n)
        self._prev_gray = cur_g
        return (diff >= self.threshold), diff


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE HEURISTIC (cheap, CPU)
# ─────────────────────────────────────────────────────────────────────────────
#
# HONEST DISCLAIMER: this is a transparent FIRST-PASS SCREEN, NOT a calibrated
# detector. It decides which frames are worth the expensive VLM call, not
# whether there is fire.
#
# Smoke (and wildfire haze) tends to look like:
#   - LOW SATURATION    : smoke desaturates the scene toward grey.
#   - GREY-MIDTONE HAZE : lots of pixels clustered near mid-grey, low colour.
#   - REDUCED CONTRAST  : haze flattens local contrast (low grayscale stddev).
#   - UPPER-REGION SHIFT: smoke first appears in the sky / upper image band;
#                         a desaturation shift up top is suggestive.
#
# We combine three normalized [0,1] sub-scores with fixed, auditable weights.

def smoke_subscores(frame: Any, size: int = DOWNSCALE) -> dict:
    """Return the individual smoke sub-scores (for diagnostics/calibration).

    Keys: saturation, contrast, upper, score. All in [0, 1]. Returns all-zeros
    if numpy is unavailable or the frame can't be decoded.
    """
    zero = {"saturation": 0.0, "contrast": 0.0, "upper": 0.0, "score": 0.0}
    if _NP is None:
        return zero
    thumb = to_rgb_thumb(frame, size)
    if thumb is None:
        return zero

    rgb = thumb.astype("float32")
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    mx = _NP.maximum(_NP.maximum(r, g), b)
    mn = _NP.minimum(_NP.minimum(r, g), b)
    sat = (mx - mn) / 255.0  # per-pixel saturation proxy, [0,1]
    lum = 0.299 * r + 0.587 * g + 0.114 * b  # [0,255]

    # --- sub-score 1: grey-haze fraction (low saturation, midtone luma) ---
    haze_mask = (sat < 0.12) & (lum >= 60.0) & (lum <= 210.0)
    s_saturation = float(haze_mask.mean())  # already [0,1]

    # --- sub-score 2: reduced contrast (low grayscale stddev) ---
    std_g = float(lum.std())
    # Clear scenes have high contrast (std often > 50 on 0..255). Map LOW std to
    # HIGH haze: std 60+ -> 0; std 10 -> ~0.83.
    s_contrast = clamp01((60.0 - std_g) / 60.0)

    # --- sub-score 3: upper-band desaturation relative to lower band ---
    upper_cut = max(1, size // 3)  # top third = "sky" band
    up = float(sat[:upper_cut, :].mean())
    lo = float(sat[upper_cut:, :].mean()) if upper_cut < size else up
    s_upper = clamp01((lo - up) * 3.0)

    score = clamp01(
        _W_SATURATION * s_saturation
        + _W_CONTRAST * s_contrast
        + _W_UPPER * s_upper
    )
    return {
        "saturation": round(s_saturation, 4),
        "contrast": round(s_contrast, 4),
        "upper": round(s_upper, 4),
        "score": round(score, 4),
    }


def smoke_score(frame: Any, size: int = DOWNSCALE) -> float:
    """Transparent, UNCALIBRATED smoke screen in [0, 1]. 0.0 if undecodable."""
    return smoke_subscores(frame, size)["score"]


# ─────────────────────────────────────────────────────────────────────────────
# Verdict assembly (no I/O — pure)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    cur: Any,
    prev: Any = None,
    *,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    smoke_alert: float = 0.75,
    vlm_threshold: float = 0.6,
    size: int = DOWNSCALE,
) -> dict:
    """Run the cheap pipeline for one frame pair; return a verdict dict.

    The verdict is the same contract used end-to-end (dev runner, plugin, and
    eventually Beehive)::

        { changed, diff, smoke_score, subscores, alert, escalate }

    ``escalate`` is True when the frame changed AND smoke_score >= vlm_threshold
    (i.e. the caller should invoke the VLM). ``alert`` here is the heuristic-only
    verdict; the plugin may upgrade it from the VLM result.
    """
    diff = frame_diff(prev, cur, size)
    changed = bool(diff >= change_threshold)
    verdict = {
        "changed": changed,
        "diff": round(float(diff), 4),
        "smoke_score": 0.0,
        "subscores": {"saturation": 0.0, "contrast": 0.0, "upper": 0.0, "score": 0.0},
        "alert": False,
        "escalate": False,
    }
    if not changed:
        return verdict
    subs = smoke_subscores(cur, size)
    score = subs["score"]
    verdict["smoke_score"] = score
    verdict["subscores"] = subs
    verdict["alert"] = bool(score >= smoke_alert)
    verdict["escalate"] = bool(score >= vlm_threshold)
    return verdict


# ═════════════════════════════════════════════════════════════════════════════
# TILED DETECTION + ROLLING BASELINE  (the static-camera + localized-plume fix)
# ═════════════════════════════════════════════════════════════════════════════
#
# Everything below is NEW and additive — the global ChangeGate / smoke_score API
# above is untouched. The tiled path solves two failures the global path can't:
#
#   * STATIC CAMERA  : per-tile rolling EMA background, not frame-to-frame diff,
#                      so a slow plume accumulates against its background.
#   * LOCALIZED PLUME: per-tile haze sub-scores, image score = MAX over tiles, so
#                      one hazy tile drives the verdict instead of being averaged
#                      away by the unchanged majority of the scene.
#
# Still cheap: a few numpy reductions over a 64x64 thumbnail split into a grid.


def _resolve_grid(size: int, rows: Optional[int], cols: Optional[int],
                  tile_px: Optional[int]) -> Tuple[int, int]:
    """Pick (rows, cols) for a ``size``x``size`` thumbnail.

    Precedence: explicit rows/cols win; else size to ~``tile_px`` per tile; else
    the 4x4 default. Always clamped to at least 1x1 and at most ``size``.
    """
    if rows is not None or cols is not None:
        r = rows if rows is not None else DEFAULT_GRID_ROWS
        c = cols if cols is not None else DEFAULT_GRID_COLS
    elif tile_px is not None and tile_px > 0:
        r = c = max(1, int(round(size / float(tile_px))))
    else:
        r, c = DEFAULT_GRID_ROWS, DEFAULT_GRID_COLS
    r = max(1, min(int(r), size))
    c = max(1, min(int(c), size))
    return r, c


def tile_grid(size: int = DOWNSCALE, *, rows: Optional[int] = None,
              cols: Optional[int] = None, tile_px: Optional[int] = None) -> list:
    """Return the list of tile slices as (r, c, y0, y1, x0, x1) over a thumbnail.

    Pure geometry helper (no frame needed) so callers can map a winning tile
    index back to thumbnail-pixel coordinates. Tiles tile the whole frame with
    near-equal sizes (the last row/col absorbs any remainder).
    """
    r, c = _resolve_grid(size, rows, cols, tile_px)
    ys = [round(i * size / r) for i in range(r + 1)]
    xs = [round(j * size / c) for j in range(c + 1)]
    out = []
    for i in range(r):
        for j in range(c):
            out.append((i, j, ys[i], ys[i + 1], xs[j], xs[j + 1]))
    return out


def _tile_haze_subscore(rgb_tile) -> dict:
    """Per-tile haze sub-score in [0,1] from a small RGB uint8/float tile.

    Reuses the SAME haze cues as the global ``smoke_subscores`` (grey-haze
    fraction + reduced contrast), minus the upper-band term (a single tile has no
    meaningful sky/ground split). Returns saturation/contrast/score in [0,1].
    """
    zero = {"saturation": 0.0, "contrast": 0.0, "score": 0.0}
    if _NP is None or rgb_tile is None or rgb_tile.size == 0:
        return zero
    rgb = rgb_tile.astype("float32")
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    mx = _NP.maximum(_NP.maximum(r, g), b)
    mn = _NP.minimum(_NP.minimum(r, g), b)
    sat = (mx - mn) / 255.0
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    haze_mask = (sat < 0.12) & (lum >= 60.0) & (lum <= 210.0)
    s_saturation = float(haze_mask.mean())
    std_g = float(lum.std())
    s_contrast = clamp01((60.0 - std_g) / 60.0)
    # Re-weight the two tile-local cues to sum to 1.0 (the global blend drops the
    # upper-band term here): saturation still dominates.
    score = clamp01(0.55 * s_saturation + 0.45 * s_contrast)
    return {
        "saturation": round(s_saturation, 4),
        "contrast": round(s_contrast, 4),
        "score": round(score, 4),
    }


def tiled_subscores(
    frame: Any,
    size: int = DOWNSCALE,
    *,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    tile_px: Optional[int] = None,
    tile_smoke_alert: float = DEFAULT_TILE_SMOKE_ALERT,
) -> dict:
    """STATELESS tiled haze screen for a single frame.

    Splits the frame into a grid and computes a haze sub-score per tile, then
    reports the MAX over tiles so a single localized hazy tile drives the
    image-level score (a global mean would wash it out).

    Returns::

        {
          grid: [rows, cols],
          max_tile_score: float,           # image-level = max over tiles
          mean_tile_score: float,          # for contrast/diagnostics
          bbox: [y0, x0, y1, x1] | None,   # winning tile (thumbnail px) or None
          tile_index: int | None,          # flat index of the winning tile
          any_tile_alert: bool,            # any tile score >= tile_smoke_alert
          n_tiles_alert: int,
          tiles: [ {row,col,score,...}, ... ],
        }

    All-zeros (and bbox None) if numpy is unavailable or the frame can't decode.
    """
    r, c = _resolve_grid(size, rows, cols, tile_px)
    empty = {
        "grid": [r, c],
        "max_tile_score": 0.0,
        "mean_tile_score": 0.0,
        "bbox": None,
        "tile_index": None,
        "any_tile_alert": False,
        "n_tiles_alert": 0,
        "tiles": [],
    }
    if _NP is None:
        return empty
    thumb = to_rgb_thumb(frame, size)
    if thumb is None:
        return empty

    grid = tile_grid(size, rows=r, cols=c)
    tiles = []
    scores = []
    best = -1.0
    best_idx = None
    best_bbox = None
    n_alert = 0
    for idx, (ti, tj, y0, y1, x0, x1) in enumerate(grid):
        sub = _tile_haze_subscore(thumb[y0:y1, x0:x1])
        s = sub["score"]
        scores.append(s)
        alert = bool(s >= tile_smoke_alert)
        if alert:
            n_alert += 1
        tiles.append({
            "row": ti, "col": tj, "score": s,
            "saturation": sub["saturation"], "contrast": sub["contrast"],
            "bbox": [y0, x0, y1, x1], "alert": alert,
        })
        if s > best:
            best = s
            best_idx = idx
            best_bbox = [y0, x0, y1, x1]

    max_score = max(0.0, best)
    mean_score = float(sum(scores) / len(scores)) if scores else 0.0
    return {
        "grid": [r, c],
        "max_tile_score": round(max_score, 4),
        "mean_tile_score": round(mean_score, 4),
        "bbox": best_bbox,
        "tile_index": best_idx,
        "any_tile_alert": bool(n_alert > 0),
        "n_tiles_alert": int(n_alert),
        "tiles": tiles,
    }


class TiledDetector:
    """Stateful tiled change-gate with a PER-TILE ROLLING BASELINE.

    This is the static-camera fix. Instead of comparing each frame only to the
    immediately preceding one (which a static camera makes ~zero), it keeps a
    slow per-tile EMA background and measures each tile against THAT. A plume that
    grows slowly over many frames keeps diverging from its (lagging) background,
    so its per-tile change accumulates and eventually crosses the threshold even
    though no single frame-to-frame step is large.

    Two baseline modes:
      * ``baseline="ema"``  (default) — rolling EMA background per tile.
      * ``baseline="frame"`` — immediate previous-frame diff per tile (the old
        frame-to-frame behaviour, but tiled). Available for parity/benchmarking.

    The detector trips a tile when BOTH:
      * its change vs the baseline >= ``tile_change_threshold`` (something moved
        in that tile), AND
      * its haze sub-score >= ``tile_smoke_alert`` (it looks like smoke).

    ``update(frame)`` returns a verdict dict (see ``_verdict``); it stores only
    tiny per-tile gray means + the per-tile background thumbnail, never the frame.
    """

    def __init__(
        self,
        *,
        size: int = DOWNSCALE,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        tile_px: Optional[int] = None,
        baseline: str = "ema",
        alpha: float = DEFAULT_BASELINE_ALPHA,
        tile_change_threshold: float = DEFAULT_TILE_CHANGE_THRESHOLD,
        tile_smoke_alert: float = DEFAULT_TILE_SMOKE_ALERT,
        haze_rise_alert: float = DEFAULT_HAZE_RISE_ALERT,
        persist_frames: int = DEFAULT_PERSIST_FRAMES,
        haze_alpha: float = DEFAULT_HAZE_BASELINE_ALPHA,
    ):
        # size<=0 would create empty tile slices (.mean() on empty) and break
        # the PIL-absent resize fallback (linspace(..., size)); clamp to >=1.
        self.size = max(1, int(size))
        self.rows, self.cols = _resolve_grid(self.size, rows, cols, tile_px)
        self.baseline = baseline if baseline in ("ema", "frame") else "ema"
        # alpha must stay in (0,1]; anything outside (or non-numeric) -> default.
        try:
            a = float(alpha)
        except (TypeError, ValueError):
            a = float("nan")
        self.alpha = a if (a == a and 0.0 < a <= 1.0) else DEFAULT_BASELINE_ALPHA
        self.tile_change_threshold = float(tile_change_threshold)
        self.tile_smoke_alert = float(tile_smoke_alert)
        # Self-relative haze-rise gate (the false-positive fix). >= 0; clamp neg.
        try:
            hr = float(haze_rise_alert)
        except (TypeError, ValueError):
            hr = DEFAULT_HAZE_RISE_ALERT
        self.haze_rise_alert = hr if (hr == hr and hr >= 0.0) else 0.0
        # Persistence (consecutive frames a tile must hold the condition). >= 1.
        try:
            pf = int(persist_frames)
        except (TypeError, ValueError):
            pf = DEFAULT_PERSIST_FRAMES
        self.persist_frames = pf if pf >= 1 else 1
        # Haze-baseline EMA alpha, in (0,1].
        try:
            ha = float(haze_alpha)
        except (TypeError, ValueError):
            ha = float("nan")
        self.haze_alpha = (ha if (ha == ha and 0.0 < ha <= 1.0)
                           else DEFAULT_HAZE_BASELINE_ALPHA)
        self._bg_gray = None      # type: Optional[Any]  rolling per-pixel gray bg
        self._bg_haze = None      # type: Optional[Any]  rolling per-TILE haze bg
        self._persist = None      # type: Optional[Any]  per-tile consec-active run
        self._grid = tile_grid(self.size, rows=self.rows, cols=self.cols)

    def reset(self) -> None:
        self._bg_gray = None
        self._bg_haze = None
        self._persist = None

    def _empty_verdict(self) -> dict:
        return {
            "grid": [self.rows, self.cols],
            "baseline": self.baseline,
            "changed": False,
            "max_tile_change": 0.0,
            "max_tile_score": 0.0,
            "mean_tile_score": 0.0,
            "bbox": None,
            "tile_index": None,
            "any_tile_alert": False,
            "n_tiles_alert": 0,
            "warming_up": True,
            "tiles": [],
        }

    def update(self, frame: Any) -> dict:
        """Feed a frame; return the tiled verdict and advance the baseline.

        First frame seeds the baseline and reports warming_up=True (no trip).
        """
        if _NP is None:
            return self._empty_verdict()
        thumb = to_rgb_thumb(frame, self.size)
        if thumb is None:
            return self._empty_verdict()
        cur_gray = _rgb_thumb_to_gray(thumb)  # size x size float32

        n_tiles = len(self._grid)
        warming = False
        if self._bg_gray is None:
            # Seed the background with the first frame; nothing to compare yet.
            self._bg_gray = cur_gray.astype("float32")
            warming = True
        if self._bg_haze is None:
            self._bg_haze = _NP.zeros(n_tiles, dtype="float32")
        if self._persist is None:
            self._persist = _NP.zeros(n_tiles, dtype="int64")

        # Per-tile CHANGE = mean-abs(current gray - baseline gray) within tile.
        diff_map = _NP.abs(cur_gray - self._bg_gray)

        cur_haze = _NP.zeros(n_tiles, dtype="float32")
        # Per-tile mask of which tiles changed this frame; the haze baseline is
        # FROZEN on changing tiles (don't fold a suspected plume into its own
        # "normal" history, or a slow ramp would absorb itself and never trip).
        changed_mask = _NP.zeros(n_tiles, dtype=bool)
        tiles = []
        best_score = -1.0
        best_idx = None
        best_bbox = None
        max_change = 0.0
        n_alert = 0
        score_sum = 0.0
        for idx, (ti, tj, y0, y1, x0, x1) in enumerate(self._grid):
            tile_change = float(diff_map[y0:y1, x0:x1].mean())
            haze = _tile_haze_subscore(thumb[y0:y1, x0:x1])
            score = haze["score"]
            cur_haze[idx] = score
            score_sum += score
            changed_tile = tile_change >= self.tile_change_threshold
            # SELF-RELATIVE haze cue: how much this tile's haze rose above its OWN
            # rolling baseline. On a warming frame the baseline isn't trained yet
            # so the rise is meaningless (seeded to the current value -> 0).
            haze_rise = (0.0 if warming
                         else float(score - float(self._bg_haze[idx])))
            # CHROMA-AWARE activity gate. Wildfire smoke desaturates a scene toward
            # grey, so a localized plume can lift a tile's HAZE sub-score while its
            # grayscale LUMA barely moves (verified: a plume tile reading
            # score=0.99 / haze_rise=0.55 but luma change 2.6 < 8.0). A pure
            # grayscale-motion gate misses exactly that slow chroma plume. So a
            # tile counts as "moved" on luma motion OR a meaningful self-relative
            # haze rise. The haze cue itself (a grey-haze fraction) responds to the
            # desaturation, so this needs no new colour math.
            haze_moved = bool((not warming) and haze_rise >= self.haze_rise_alert)
            suspect_tile = bool(changed_tile or haze_moved)
            # FREEZE the haze baseline on suspect tiles (luma OR haze movement), not
            # just grayscale-changed ones. Without this, a chroma plume that doesn't
            # move luma is treated as "unchanged", so its rising haze gets EMA-folded
            # straight into its own baseline and the rise we detect on vanishes.
            changed_mask[idx] = suspect_tile
            # A tile is "active" this frame iff it moved (luma OR haze) AND its haze
            # rose enough above its own history AND its absolute haze clears the
            # floor. The absolute-haze floor keeps the synthetic/contract behaviour;
            # the self-relative rise + persistence gates (unchanged) are what kill
            # the clear-frame / drifting-cirrus false positives on real scenes.
            active = bool(
                (not warming)
                and suspect_tile
                and score >= self.tile_smoke_alert
                and haze_rise >= self.haze_rise_alert
            )
            # Persistence: count consecutive active frames per tile; a tile only
            # ALERTS once it has held "active" for persist_frames in a row.
            if active:
                self._persist[idx] += 1
            else:
                self._persist[idx] = 0
            alert = bool(self._persist[idx] >= self.persist_frames)
            if alert:
                n_alert += 1
            if tile_change > max_change:
                max_change = tile_change
            tiles.append({
                "row": ti, "col": tj,
                "change": round(tile_change, 4),
                "score": score,
                "haze_rise": round(haze_rise, 4),
                "saturation": haze["saturation"],
                "contrast": haze["contrast"],
                "bbox": [y0, x0, y1, x1],
                "changed": bool(changed_tile),
                "active": active,
                "persist": int(self._persist[idx]),
                "alert": alert,
            })
            # The winning tile (for bbox / image-level score) is the highest haze
            # score among tiles that ALERTED; else the highest among tiles that at
            # least moved; else the global max haze tile (so bbox stays meaningful
            # while warming up or before any alert).
            if alert:
                ranked = score + 2000.0
            elif (not warming) and suspect_tile:
                ranked = score
            else:
                ranked = score - 1000.0
            if ranked > best_score:
                best_score = ranked
                best_idx = idx
                best_bbox = [y0, x0, y1, x1]

        # Image-level score = MAX haze over tiles that ALERTED (a genuine,
        # localized, persistent, risen-above-baseline plume drives the verdict).
        # If nothing alerted, fall back to the max over tiles that merely MOVED
        # (diagnostic / softer escalate signal). A clear scene -> both are 0.
        alert_scores = [t["score"] for t in tiles if t["alert"]]
        moved_scores = [t["score"] for t in tiles if t["changed"]]
        if alert_scores:
            max_tile_score = max(alert_scores)
        elif moved_scores:
            max_tile_score = max(moved_scores)
        else:
            max_tile_score = 0.0
        mean_tile_score = score_sum / len(tiles) if tiles else 0.0

        verdict = {
            "grid": [self.rows, self.cols],
            "baseline": self.baseline,
            "changed": bool(max_change >= self.tile_change_threshold),
            "max_tile_change": round(max_change, 4),
            "max_tile_score": round(max(0.0, max_tile_score), 4),
            "mean_tile_score": round(mean_tile_score, 4),
            "bbox": best_bbox,
            "tile_index": best_idx,
            "any_tile_alert": bool(n_alert > 0),
            "n_tiles_alert": int(n_alert),
            "warming_up": warming,
            "tiles": tiles,
        }

        # Advance the baselines AFTER scoring this frame.
        if self.baseline == "ema":
            if not warming:
                self._bg_gray = ((1.0 - self.alpha) * self._bg_gray
                                 + self.alpha * cur_gray)
        else:  # "frame": baseline is simply the previous frame
            self._bg_gray = cur_gray.astype("float32")
        # Per-tile haze baseline. Seed on the warming frame, then EMA-track ONLY
        # the tiles that did NOT change this frame. This is what makes the alert
        # SELF-RELATIVE without self-absorbing: a stable (even hazy) clear tile
        # tracks its own history (so it never alerts), while a tile with an active
        # plume has its baseline FROZEN at the pre-plume level, so the haze rise
        # keeps accumulating across a slow ramp instead of being averaged in.
        if warming:
            self._bg_haze = cur_haze.copy()
        else:
            updated = ((1.0 - self.haze_alpha) * self._bg_haze
                       + self.haze_alpha * cur_haze)
            # Freeze changing tiles at their prior baseline; update the rest.
            self._bg_haze = _NP.where(changed_mask, self._bg_haze, updated)
        return verdict


def evaluate_tiled(
    detector: "TiledDetector",
    frame: Any,
    *,
    vlm_threshold: float = 0.6,
) -> dict:
    """Convenience: run a TiledDetector for one frame and add an ``escalate`` flag.

    ``escalate`` is True when a tile has ALERTED (moved + risen above its own haze
    baseline + persisted) AND its haze score crosses ``vlm_threshold`` (so the
    caller should fire the VLM). It deliberately keys off the SAME persistent,
    self-relative alert as ``any_tile_alert`` — escalation is "softer" only via a
    lower *score* bar, never by skipping the persistence / haze-rise gates that
    suppress drifting-cloud false positives. ``alert`` mirrors ``any_tile_alert``.
    Pure — no I/O.
    """
    v = detector.update(frame)
    v["alert"] = bool(v["any_tile_alert"])
    v["escalate"] = bool(v["any_tile_alert"]
                         and v["max_tile_score"] >= vlm_threshold)
    return v
