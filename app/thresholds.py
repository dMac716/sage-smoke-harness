#!/usr/bin/env python3
"""Single source of truth for every Smoke Spotter detection threshold + a
free-text VLM-reply whitelist normalizer.

WHY THIS MODULE EXISTS
──────────────────────
The detection knobs (change-gate threshold, smoke-alert level, VLM-escalation
threshold, the haze sub-score blend weights, and the whole tiled + rolling-
baseline cascade) are currently DUPLICATED: ``app/smoke.py`` defines the cheap-
detector defaults, ``app/main.py`` re-states many of them as ``DEFAULT_*`` knobs,
and ``app/dev_run.py`` advertises them again in its CLI help. Three copies drift.

This module centralizes the canonical VALUES in one auditable leaf so the gate,
the heuristic, and the VLM can never silently disagree. It is a **LEAF**: it
imports nothing from ``app/`` (no ``smoke``, no ``main``) — so ``smoke.py`` and
``main.py`` can re-export FROM here without an import cycle. The values below are
mirrored from today's ``smoke.py`` exactly; ``tests/test_thresholds.py`` asserts
they stay equal so the single-source-of-truth can't quietly fork.

Ported in spirit from the operator's RadioTower ``services/eas_keywords.py`` — a
tiny, transparent, whitelist/centralization module. That file centralizes the
EAS keyword set so "what counts as an emergency" is decided in ONE place instead
of three that drift; this file does the same for "what counts as smoke" (the
numeric thresholds) plus a prose-to-boolean whitelist for the VLM. We adapt the
discipline (auditable whitelist, one decision point, case-insensitive matching),
NOT the EAS semantics.

Doctrine (Sage/Waggle): *move the work, not the data.* These thresholds govern
the cheap edge screen that decides which frames are worth the expensive VLM call.

Everything here is stdlib-only (``re``, ``typing``, ``dataclasses`` not needed)
and never raises: the prose normalizer returns ``None`` (ambiguous) rather than
throwing on any odd input.
"""
from __future__ import annotations

import re
from typing import Optional

# ═════════════════════════════════════════════════════════════════════════════
# DETECTION THRESHOLDS — the canonical values (mirror app/smoke.py exactly)
# ═════════════════════════════════════════════════════════════════════════════
#
# Each constant below is the ONE true default. ``smoke.py`` /  ``main.py`` should
# re-export from here; ``tests/test_thresholds.py`` pins each to the value
# ``smoke.py`` currently uses so the two can't drift apart.

# ── Global change-gate ──────────────────────────────────────────────────────
# Both the global and tiled gates work on a DOWNSCALE x DOWNSCALE grayscale
# thumbnail. Small + fixed so the screen is O(1) regardless of source resolution.
DOWNSCALE = 64

# changed-gate default: mean-abs grayscale diff (0..255 scale) at/above which a
# whole frame counts as "changed". NOT calibrated — overridable by the caller.
# Mirrors smoke.DEFAULT_CHANGE_THRESHOLD.
CHANGE_THRESHOLD = 8.0

# ── Verdict thresholds (smoke_score is in [0, 1]) ───────────────────────────
# smoke_score at/above which the heuristic alone raises ``alert``. Mirrors
# main.DEFAULT_SMOKE_ALERT (smoke.evaluate's ``smoke_alert`` default).
SMOKE_ALERT = 0.75

# smoke_score at/above which we ESCALATE a changed frame to the VLM. Lower than
# SMOKE_ALERT: we ask the model on "maybe" frames before declaring an alert.
# Mirrors main.DEFAULT_VLM_THRESHOLD (smoke.evaluate's ``vlm_threshold`` default).
VLM_THRESHOLD = 0.6

# ── Global smoke_score sub-score blend weights (sum to 1.0) ──────────────────
# The global ``smoke_subscores`` blends three normalized [0,1] cues with these
# fixed, auditable weights. Mirror smoke._W_SATURATION / _W_CONTRAST / _W_UPPER.
W_SATURATION = 0.40  # grey-haze fraction (the dominant signal)
W_CONTRAST = 0.35    # reduced contrast (haze flattens local contrast)
W_UPPER = 0.25       # upper-band desaturation shift (smoke appears up top first)

# ── Haze-cue constants (the per-pixel "looks like smoke" thresholds) ─────────
# A pixel counts toward the grey-haze fraction when it is BOTH low-saturation and
# in the mid-luma band (smoke desaturates the scene toward mid-grey). These mirror
# the literals inside smoke.smoke_subscores / smoke._tile_haze_subscore.
HAZE_SAT_MAX = 0.12     # per-pixel saturation proxy must be BELOW this (grey)
HAZE_LUM_MIN = 60.0     # ... and luma (0..255) at/above this ...
HAZE_LUM_MAX = 210.0    # ... and at/below this (mid-tones, not black/blown-out).

# Reduced-contrast cue: map LOW grayscale stddev to HIGH haze via
#   s_contrast = clamp01((CONTRAST_STD_REF - std) / CONTRAST_STD_REF)
# Clear scenes have high contrast (std often > 50 on 0..255); std 60+ -> 0,
# std 10 -> ~0.83. Mirrors the ``/ 60.0`` map in smoke.py.
CONTRAST_STD_REF = 60.0

# ═════════════════════════════════════════════════════════════════════════════
# TILED DETECTION DEFAULTS — the static-camera + localized-plume fix
# ═════════════════════════════════════════════════════════════════════════════
#
# The tiled cascade is the DEFAULT detection path. It splits the thumbnail into a
# grid, keeps a per-tile rolling EMA background (so slow growth on a static camera
# accumulates), and takes the MAX haze over tiles (so a small localized plume
# isn't averaged away). All values mirror smoke.DEFAULT_* exactly.

# Grid shape: split the DOWNSCALE thumbnail into GRID_ROWS x GRID_COLS tiles.
# 4x4 = 16 tiles is a good cheap default (~16x16 px per tile of the 64x64 thumb),
# localizing a plume to ~1/16 of the frame so it isn't washed out globally.
GRID_ROWS = 4
GRID_COLS = 4

# A target tile EDGE in thumbnail pixels; callers may pass ``tile_px`` instead of
# explicit rows/cols and the grid is sized to roughly this many pixels per tile.
TILE_PX = 16

# Rolling-baseline EMA smoothing for the per-tile GRAY background:
#   bg <- (1 - alpha) * bg + alpha * cur
# Smaller alpha = slower background = a slow plume accumulates LONGER before being
# absorbed. 0.05 tracks at ~1/20th of the per-frame rate.
BASELINE_ALPHA = 0.05

# Per-tile change threshold (0..255 mean-abs grey diff vs the tile's background).
# A localized plume only has to move ONE tile this far, not the whole frame.
TILE_CHANGE_THRESHOLD = 8.0

# Per-tile haze sub-score at/above which that tile is considered "alerting".
TILE_SMOKE_ALERT = 0.55

# Self-relative haze-RISE gate (the false-positive fix). A tile must have its haze
# rise at least this far above its OWN rolling baseline to count — a clear (even
# hazy) tile sits near its baseline (~0 rise) and never alerts; smoke lifts a tile
# well above its history. 0.0 disables the rise gate.
#
# CALIBRATION (2026-06-26, docs/research/threshold-calibration.md): this is the
# BINDING gate, and 0.20 is tuned for DYNAMIC scenes (kills drifting-cirrus FPs on
# the Pauma clip, docs/lessons/0004). On STATIC fire-watch cameras (FIgLib) the
# observed plume rise is only ~0.07–0.11, so at 0.20 the gate NEVER fires (0/432
# operating points) — static-camera detection then rests entirely on the VLM
# heartbeat. Lowering to ~0.02 turns the gate into a high-recall screen on FIgLib
# (R=0.93, F1=0.77 ≈ SmokeyNet) whose FPs the high-precision VLM confirmer filters
# (docs/lessons/0003), but reintroduces dynamic-scene FPs — so the default is left
# at 0.20 pending a Pauma cross-validation. A static-camera deployment can opt into
# ~0.02. One global value can't serve both regimes (the cue isn't smoke-specific).
HAZE_RISE_ALERT = 0.20

# Require a tile to satisfy (moved AND hazy-rise) for this many CONSECUTIVE frames
# before it alerts. Drifting clouds flick a tile on for a frame or two; a real
# plume grows and PERSISTS in place. 1 = no persistence requirement.
PERSIST_FRAMES = 2

# EMA smoothing for the per-tile HAZE baseline (separate from the gray-background
# alpha). Same intent: a slow background so a rising plume keeps diverging.
HAZE_BASELINE_ALPHA = 0.05

# ═════════════════════════════════════════════════════════════════════════════
# SERVICE CADENCE DEFAULTS (main.py loop knobs that are also thresholds)
# ═════════════════════════════════════════════════════════════════════════════

# Periodic FORCED VLM escalation ("heartbeat"): even with NO tile trip, escalate
# every HEARTBEAT_S seconds so a static scene is still confirmed by the heavy path
# on a fixed cadence. Bounds the worst-case miss to one interval. 0 disables it.
# Mirrors main.DEFAULT_HEARTBEAT_S.
HEARTBEAT_S = 600.0


# ═════════════════════════════════════════════════════════════════════════════
# VLM FREE-TEXT REPLY WHITELIST  (prose -> boolean smoke verdict)
# ═════════════════════════════════════════════════════════════════════════════
#
# ``app/vlm.py`` asks the model for strict JSON
#   {"smoke": true|false, "detail": "<sentence>"}
# and parses ``smoke`` directly. But small edge VLMs (moondream / Florence-2 on a
# node) routinely IGNORE the format instruction and answer in prose:
#   "No, there is no smoke in this image, just clouds over the hills."
# When the strict-JSON parse fails, the caller can fall back to this normalizer to
# salvage a verdict from the prose instead of always degrading to ``unknown``.
#
# This is the SAME whitelist discipline as RadioTower's ``eas_keywords.is_eas_text``
# — an auditable phrase list, matched case-insensitively, decided in ONE place.
# We adapt it for a THREE-way verdict (smoke / no-smoke / ambiguous) with explicit
# negation handling, because "no smoke" must map to False, not be missed.
#
# The prompt in vlm.py warns the model to distinguish real wildfire smoke from
# benign LOOK-ALIKES (clouds / fog / lens haze); we honour that here — a reply
# whose only weather word is "clouds"/"fog" and that names no fire/smoke -> False.

# POSITIVE phrases — clear evidence of wildfire smoke / fire. Matched as whole
# words (so "fire" won't fire on "fireplace"-style substrings, and "smoke" won't
# match "smokestack"). Order doesn't matter; any hit (absent negation) -> True.
POSITIVE_SMOKE_TERMS = (
    "smoke",
    "smoky",
    "smokey",
    "wildfire",
    "wildfires",
    "fire",
    "fires",
    "flame",
    "flames",
    "plume",
    "plumes",
    "blaze",
    "burning",
    "ablaze",
)

# NEGATIVE phrases — clear evidence there is NO wildfire smoke. Two kinds:
#   (a) explicit negations of the positive terms ("no smoke", "no fire", ...), and
#   (b) benign LOOK-ALIKES the prompt warns about, when named ALONE (cloud, fog,
#       lens haze) — a clear sky / cloudy / foggy scene with no fire/smoke.
# Multi-word phrases are matched as substrings (case-insensitive); single words as
# whole words. Any negative hit, with NO surviving positive evidence -> False.
NEGATIVE_SMOKE_PHRASES = (
    # explicit negations of smoke / fire
    "no smoke",
    "no visible smoke",
    "not smoke",
    "no signs of smoke",
    "no sign of smoke",
    "without smoke",
    "smoke-free",
    "smoke free",
    "no fire",
    "no visible fire",
    "not a fire",
    "no flames",
    "nothing",
    # explicit "all clear" wording
    "clear sky",
    "clear skies",
    "all clear",
    "clear",
    # benign look-alikes the prompt warns about (cloud / fog / lens haze)
    "cloud",
    "clouds",
    "cloudy",
    "overcast",
    "fog",
    "foggy",
    "mist",
    "misty",
    "lens haze",
    "lens flare",
    "haze",  # haze ALONE (the prompt's own "lens haze" caveat) is benign
    "hazy",
)

# Negation cues that flip an otherwise-positive sentence to "no smoke". We scan a
# small window of words BEFORE a positive term for one of these. This catches
# phrasings the fixed NEGATIVE_SMOKE_PHRASES list can't enumerate, e.g.
# "there isn't any smoke", "I don't see fire".
_NEGATION_CUES = (
    "no",
    "not",
    "n't",
    "without",
    "never",
    "none",
    "free of",
    "absence of",
    "lacks",
    "lacking",
)

# How many words before a positive term we scan for a negation cue.
_NEGATION_WINDOW = 4

# Pre-split a reply into lowercase word tokens (letters/digits), discarding
# punctuation. Apostrophes are kept so "isn't" -> "isn't" (the "n't" cue matches).
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list:
    """Lowercase, punctuation-tolerant word tokens of ``text``.

    "No visible smoke!!!" -> ['no', 'visible', 'smoke']. Apostrophes survive so
    contractions ("isn't") keep their "n't" so a negation cue can match.
    """
    return _WORD_RE.findall(text.lower())


def _has_word(tokens: list, word: str) -> bool:
    """Whole-word membership (already-lowercased ``word``)."""
    return word in tokens


def _negated_before(tokens: list, idx: int) -> bool:
    """Is there a negation cue within ``_NEGATION_WINDOW`` words before ``tokens[idx]``?

    Scans the preceding window for any token that IS a negation cue. The lone
    CONTRACTION cue ("n't") instead matches as a SUFFIX so "isn't" / "doesn't"
    flip the verdict. Word cues ("no", "not", "none", "never", ...) must match as
    WHOLE WORDS — matching them as suffixes wrongly negated benign words that
    merely END in those letters ("casino"/"piano" -> "no", "cannot" -> "not"),
    so a real "...smoke..." after such a word was silently lost. Lets us flip
    "there is no smoke" / "I don't see smoke" to a negative verdict.
    """
    lo = max(0, idx - _NEGATION_WINDOW)
    for t in tokens[lo:idx]:
        for cue in _NEGATION_CUES:
            if " " in cue:
                continue  # phrase cues are handled by NEGATIVE_SMOKE_PHRASES
            if "'" in cue:
                # contraction suffix cue ("n't"): match the tail of "isn't" etc.
                if t.endswith(cue):
                    return True
            elif t == cue:  # word cue: WHOLE-WORD match only (no substring/suffix)
                return True
    return False


def vlm_text_to_smoke(text: str) -> Optional[bool]:
    """Whitelist-match a FREE-TEXT VLM reply into a boolean smoke verdict.

    For the fallback path in ``vlm.py`` when strict-JSON parsing of the model's
    answer fails: salvage a verdict from prose instead of always reporting
    ``unknown``.

    Returns:
      * ``True``  — the prose clearly indicates wildfire smoke / fire / a plume.
      * ``False`` — it clearly indicates clear / no-smoke (an explicit negation
                    like "no smoke" / "not smoke" / "smoke-free", OR a scene
                    described only as a benign look-alike — clouds / fog / lens
                    haze — with no fire/smoke).
      * ``None``  — ambiguous, empty, or no whitelist phrase matched at all.

    Decision order (transparent + auditable):
      1. Empty / non-string -> None.
      2. Find POSITIVE evidence: a positive term NOT immediately negated.
      3. Find NEGATIVE evidence: an explicit no-smoke phrase, OR a benign
         look-alike named alone.
      4. positive-only -> True; negative-only -> False; both / neither -> resolve:
         - both present (e.g. "no smoke, just clouds, but a fire on the ridge"):
           a SURVIVING (non-negated) positive wins -> True; otherwise False.
         - neither -> None.

    Never raises; any odd input degrades to None.
    """
    if not text or not isinstance(text, str):
        return None
    if not _tokens(text):
        return None
    lower = text.lower()

    # --- 2. NEGATIVE evidence FIRST, and BLANK each matched phrase's span ------
    # We detect the explicit negative phrases up front and erase their text span
    # from ``lower`` BEFORE scanning for positives. This is what makes phrases
    # that embed a positive term self-consistent: "no smoke" and "smoke-free"
    # carry their own "smoke", so erasing the phrase span stops that "smoke" from
    # later registering as a SURVIVING positive. Whole-word benign terms (cloud /
    # fog) carry no positive term so they need no span erasure — they just set the
    # negative flag.
    negative = False
    for phrase in NEGATIVE_SMOKE_PHRASES:
        if " " in phrase or "-" in phrase:
            # multi-word / hyphenated phrases ("no smoke", "smoke-free",
            # "no visible smoke", "lens haze") match as substrings; erase each
            # occurrence so any positive term inside it can't survive below.
            if phrase in lower:
                negative = True
                lower = lower.replace(phrase, " ")
        else:
            # single benign/clear words match as whole words ("clouds", "fog").
            if _has_word(_tokens(lower), phrase):
                negative = True

    # --- 3. surviving POSITIVE evidence (a positive term NOT in a negated span
    #        and NOT immediately preceded by a negation cue) ---------------------
    # Re-tokenize the (now negative-span-stripped) text. Anything left that is a
    # positive term and is not negated by a nearby cue is genuine smoke evidence.
    tokens = _tokens(lower)
    positive = False
    negated_positive = False
    for term in POSITIVE_SMOKE_TERMS:
        for i, tok in enumerate(tokens):
            if tok != term:
                continue
            if _negated_before(tokens, i):
                negated_positive = True
            else:
                positive = True
                break
        if positive:
            break

    # --- 4. resolve ---
    if positive:
        # A surviving, non-negated mention of smoke/fire/plume wins even if the
        # reply also says "no clouds" or mentions a benign term — the model saw
        # smoke. (Mixed "no smoke ... but a fire" -> the un-negated 'fire' -> True.)
        return True
    if negative or negated_positive:
        # Explicit no-smoke phrasing, a negated positive ("no smoke"/"isn't fire"),
        # or a benign-only scene (clouds/fog/haze) -> clearly NOT smoke.
        return False
    return None  # nothing in the whitelist matched -> ambiguous


# ═════════════════════════════════════════════════════════════════════════════
# Convenience helpers (provenance / diagnostics)
# ═════════════════════════════════════════════════════════════════════════════

def is_smoke_alert(smoke_score: float, *, smoke_alert: float = SMOKE_ALERT) -> bool:
    """Heuristic alert decision: is ``smoke_score`` at/above the alert level?

    The single canonical comparison so the gate, the dev runner, and the plugin
    can't apply a subtly different one. NaN / garbage -> False (never alert on
    undecodable input).
    """
    try:
        s = float(smoke_score)
    except (TypeError, ValueError):
        return False
    if s != s:  # NaN
        return False
    return s >= float(smoke_alert)


def thresholds_summary() -> dict:
    """Return ALL canonical thresholds as a flat dict (provenance / diagnostics).

    Handy for stamping a run's ``provenance.json`` or a status self-report so the
    exact thresholds a run used are recorded alongside its outputs.
    """
    return {
        "downscale": DOWNSCALE,
        "change_threshold": CHANGE_THRESHOLD,
        "smoke_alert": SMOKE_ALERT,
        "vlm_threshold": VLM_THRESHOLD,
        "w_saturation": W_SATURATION,
        "w_contrast": W_CONTRAST,
        "w_upper": W_UPPER,
        "haze_sat_max": HAZE_SAT_MAX,
        "haze_lum_min": HAZE_LUM_MIN,
        "haze_lum_max": HAZE_LUM_MAX,
        "contrast_std_ref": CONTRAST_STD_REF,
        "grid_rows": GRID_ROWS,
        "grid_cols": GRID_COLS,
        "tile_px": TILE_PX,
        "baseline_alpha": BASELINE_ALPHA,
        "tile_change_threshold": TILE_CHANGE_THRESHOLD,
        "tile_smoke_alert": TILE_SMOKE_ALERT,
        "haze_rise_alert": HAZE_RISE_ALERT,
        "persist_frames": PERSIST_FRAMES,
        "haze_baseline_alpha": HAZE_BASELINE_ALPHA,
        "heartbeat_s": HEARTBEAT_S,
    }


__all__ = [
    # global thresholds
    "DOWNSCALE",
    "CHANGE_THRESHOLD",
    "SMOKE_ALERT",
    "VLM_THRESHOLD",
    # blend weights
    "W_SATURATION",
    "W_CONTRAST",
    "W_UPPER",
    # haze cues
    "HAZE_SAT_MAX",
    "HAZE_LUM_MIN",
    "HAZE_LUM_MAX",
    "CONTRAST_STD_REF",
    # tiled defaults
    "GRID_ROWS",
    "GRID_COLS",
    "TILE_PX",
    "BASELINE_ALPHA",
    "TILE_CHANGE_THRESHOLD",
    "TILE_SMOKE_ALERT",
    "HAZE_RISE_ALERT",
    "PERSIST_FRAMES",
    "HAZE_BASELINE_ALPHA",
    "HEARTBEAT_S",
    # whitelist + helpers
    "POSITIVE_SMOKE_TERMS",
    "NEGATIVE_SMOKE_PHRASES",
    "vlm_text_to_smoke",
    "is_smoke_alert",
    "thresholds_summary",
]
