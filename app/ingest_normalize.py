#!/usr/bin/env python3
"""Camera transport normalization for the Smoke Spotter ingest path.

Sage / Waggle PTZ cameras are **not** all the same shape on the wire. A node
camera (or a public ALERTCalifornia / WIFIRE feed proxied through a relay) may
expose any of: an RTSP stream, an HLS playlist, or a single still-JPEG URL — and
some of those only reachable through a relay/proxy host (NAT, auth, or
transcode). The rest of the pipeline (``smoke.detect`` on a single frame) wants
exactly ONE answer: *what transport do I use, and what URL do I point at?*

This module is that resolver. Given a raw, possibly-garbled camera dict it picks
ONE canonical ``transport`` + ``playback_url`` by a fixed precedence and returns
a normalized dict. It then offers a tiny ``one_frame_command`` helper that builds
(but never runs) the ``ffmpeg`` argv to grab a single keyframe from an RTSP/HLS
source — JPEG sources need no command, they are a direct still fetch.

Doctrine (Sage/Waggle): *move the work, not the data.* We resolve the cheapest
single-frame ingest at the edge; we never pull whole streams to the cloud, and
for a still-JPEG camera we fetch exactly one image, not a video.

OFFLINE-FIRST (hard rule): this module performs **no** network I/O and runs
**no** subprocess by itself. It only *constructs* URLs and argv lists. A caller
that wants to actually fetch a JPEG still may pass its own ``fetch`` callable to
``fetch_still`` — the default raises rather than touching the network, so the
module imports and unit-tests cleanly with nothing installed and no node.

Ported from the operator's TowerWatch ``services/emergency_ops/video_bridge.py``
(the ``normalize_camera`` transport-precedence ladder + the ``build_*`` manifest
shapes). The precedence is mirrored faithfully and documented per branch below.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

LOG = logging.getLogger("smoke_spotter.ingest")

# Transport names — the canonical vocabulary the rest of the pipeline switches on.
#   rtsp_transformed : an RTSP source republished as HLS through a relay host.
#   jpeg_proxy       : a still-JPEG source fetched through a relay/proxy host.
#   jpeg             : a directly-reachable still-JPEG URL (no relay).
#   rtsp             : a directly-reachable RTSP stream (no relay).
#   none             : nothing usable (empty / garbled / unknown camera).
TRANSPORT_RTSP_TRANSFORMED = "rtsp_transformed"
TRANSPORT_JPEG_PROXY = "jpeg_proxy"
TRANSPORT_JPEG = "jpeg"
TRANSPORT_RTSP = "rtsp"
TRANSPORT_NONE = "none"

# Sources that are a continuous stream (RTSP or relay-republished HLS) need a
# decode step to extract a single frame; JPEG sources are already one image.
_STREAM_TRANSPORTS = (TRANSPORT_RTSP, TRANSPORT_RTSP_TRANSFORMED)
_STILL_TRANSPORTS = (TRANSPORT_JPEG, TRANSPORT_JPEG_PROXY)

# Default ffmpeg single-keyframe knobs. ``-frames:v 1`` grabs exactly one decoded
# video frame; ``-rtsp_transport tcp`` is the reliable default for RTSP over the
# public internet (UDP is frequently dropped by NAT/firewalls). NOT executed here.
DEFAULT_FFMPEG_BIN = "ffmpeg"
DEFAULT_RTSP_TRANSPORT = "tcp"


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers (never raise on garbled input)
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    """Compact UTC ISO-8601 timestamp (``...Z``), matching the source manifests."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _clean_str(value: Any) -> str:
    """Coerce any value to a stripped string; non-strings/None become ``""``.

    The whole point is tolerance: a camera dict may carry ``None``, ints, or junk
    where a URL/id should be. We never raise — we just treat it as absent.
    """
    if value is None:
        return ""
    try:
        return str(value).strip()
    except Exception:  # pragma: no cover - str() on a pathological object
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Transport resolution (the ported precedence ladder)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_camera(camera: Any, relay_base: str = "") -> dict:
    """Resolve ONE canonical ``transport`` + ``playback_url`` for a camera.

    Mirrors TowerWatch ``video_bridge.normalize_camera`` precedence exactly. The
    ladder is ordered most-capable / most-constrained first so a relay always
    wins over a bare URL when both an id and a relay base are present:

      1. **rtsp + relay**  → ``rtsp_transformed`` : an RTSP stream is too heavy /
         not directly reachable, so the relay republishes it as HLS. We point at
         the relay's ``/hls/<id>/index.m3u8`` playlist.
      2. **jpeg + relay**  → ``jpeg_proxy``       : a still-JPEG fetched through
         the relay's ``/jpeg/<id>`` proxy (auth / NAT / rate-limit shielding).
      3. **bare jpeg**     → ``jpeg``             : a directly-reachable still URL.
      4. **bare rtsp**     → ``rtsp``             : a directly-reachable stream.
      5. **none**          → ``none``             : nothing usable.

    Branches 1 and 2 require BOTH a camera ``id`` AND a non-empty ``relay_base``
    (you can't build a relay URL without an id to key it). If a relay is
    configured but the id is missing, we fall THROUGH to the bare branches so a
    usable direct URL is still chosen rather than dropped.

    Returns a NEW dict that copies the input's keys (id/label/lat/lon/etc. are
    preserved for the manifest builders) and overwrites ``transport`` +
    ``playback_url`` + ``source``. Never raises: a non-dict / empty / garbled
    input yields ``{transport: 'none', playback_url: ''}``.
    """
    # Tolerate a non-dict (None, list, str, ...) — start from an empty cam.
    cam: dict = dict(camera) if isinstance(camera, dict) else {}

    cam_id = _clean_str(cam.get("id"))
    relay = _clean_str(relay_base).rstrip("/")
    rtsp_url = _clean_str(cam.get("rtsp_url"))
    jpeg_url = _clean_str(cam.get("jpeg_url"))
    # ``hls_url`` is accepted as an already-republished playlist: if the camera
    # arrives pre-transformed (relay handed us the HLS URL directly) we honour it
    # without needing the relay base to reconstruct the path.
    hls_url = _clean_str(cam.get("hls_url"))
    # Relay flags let a caller force the relay branch even when it can't be
    # inferred (e.g. id present but caller knows the direct URL is unreachable).
    relay_flag = bool(cam.get("relay") or cam.get("use_relay"))

    if rtsp_url and cam_id and relay:
        # (1) rtsp + relay -> republish as HLS through the relay.
        cam["transport"] = TRANSPORT_RTSP_TRANSFORMED
        cam["playback_url"] = f"{relay}/hls/{cam_id}/index.m3u8"
        cam["source"] = rtsp_url
    elif rtsp_url and hls_url:
        # (1b) caller already supplied a transformed HLS playlist for an RTSP
        # source — treat it as rtsp_transformed and point at the given playlist.
        cam["transport"] = TRANSPORT_RTSP_TRANSFORMED
        cam["playback_url"] = hls_url
        cam["source"] = rtsp_url
    elif jpeg_url and cam_id and relay:
        # (2) jpeg + relay -> fetch the still through the relay's jpeg proxy.
        cam["transport"] = TRANSPORT_JPEG_PROXY
        cam["playback_url"] = f"{relay}/jpeg/{cam_id}"
        cam["source"] = jpeg_url
    elif jpeg_url and relay_flag and cam_id and not relay:
        # (2b) relay explicitly requested but no relay base configured: we cannot
        # build a proxy URL, so fall back to the bare still rather than dropping
        # the camera. (Documented so the degraded behaviour is intentional.)
        cam["transport"] = TRANSPORT_JPEG
        cam["playback_url"] = jpeg_url
        cam["source"] = jpeg_url
    elif jpeg_url:
        # (3) bare jpeg -> a directly-reachable still URL.
        cam["transport"] = TRANSPORT_JPEG
        cam["playback_url"] = jpeg_url
        cam["source"] = jpeg_url
    elif rtsp_url:
        # (4) bare rtsp -> a directly-reachable stream.
        cam["transport"] = TRANSPORT_RTSP
        cam["playback_url"] = rtsp_url
        cam["source"] = rtsp_url
    elif hls_url:
        # (4b) only an HLS playlist given (no rtsp/jpeg) -> treat it as a stream.
        cam["transport"] = TRANSPORT_RTSP_TRANSFORMED
        cam["playback_url"] = hls_url
        cam["source"] = hls_url
    else:
        # (5) nothing usable.
        cam["transport"] = TRANSPORT_NONE
        cam["playback_url"] = ""
        cam["source"] = ""
    return cam


def is_stream(normalized: Any) -> bool:
    """True iff the normalized camera is a continuous stream (needs a decode)."""
    if not isinstance(normalized, dict):
        return False
    return _clean_str(normalized.get("transport")) in _STREAM_TRANSPORTS


def is_still(normalized: Any) -> bool:
    """True iff the normalized camera is a single still JPEG (direct fetch)."""
    if not isinstance(normalized, dict):
        return False
    return _clean_str(normalized.get("transport")) in _STILL_TRANSPORTS


# ─────────────────────────────────────────────────────────────────────────────
# Single-keyframe command construction (built, NEVER executed)
# ─────────────────────────────────────────────────────────────────────────────

def one_frame_command(
    normalized: Any,
    *,
    output: str = "pipe:1",
    ffmpeg_bin: str = DEFAULT_FFMPEG_BIN,
    rtsp_transport: str = DEFAULT_RTSP_TRANSPORT,
) -> Optional[List[str]]:
    """Build the ffmpeg argv to grab ONE keyframe from a stream source.

    Returns the command as a **list** (never a shell string — no shell injection,
    no quoting hazards; the caller hands it straight to ``subprocess`` IF and when
    it chooses to). Shape::

        [ffmpeg, -loglevel, error, -rtsp_transport, tcp, -i, <url>,
         -frames:v, 1, -f, image2, -q:v, 2, -y, <output>]

    * ``-rtsp_transport tcp`` is emitted ONLY for ``rtsp`` (raw RTSP); a relay's
      ``rtsp_transformed`` HLS playlist is plain HTTP and takes no RTSP flag.
    * ``-frames:v 1`` is the actual "single keyframe" knob.
    * ``output`` defaults to ``pipe:1`` so the still streams to stdout — the
      caller can capture the bytes without a temp file. ``-f image2`` forces a
      JPEG-family still encoder regardless of the (stdout) extension.

    Returns ``None`` for JPEG sources — those are a direct still fetch (see
    :func:`fetch_still`), not an ffmpeg decode — and ``None`` for ``none`` /
    unusable input. Never raises and never runs ffmpeg.
    """
    if not isinstance(normalized, dict):
        return None
    transport = _clean_str(normalized.get("transport"))
    url = _clean_str(normalized.get("playback_url"))
    if transport not in _STREAM_TRANSPORTS or not url:
        # JPEG (direct fetch) or nothing usable -> no decode command.
        return None

    bin_ = _clean_str(ffmpeg_bin) or DEFAULT_FFMPEG_BIN
    argv: List[str] = [bin_, "-loglevel", "error"]
    # Raw RTSP benefits from a forced transport; the transformed-HLS playlist is
    # ordinary HTTP and must NOT carry an RTSP flag.
    if transport == TRANSPORT_RTSP:
        transport_mode = _clean_str(rtsp_transport) or DEFAULT_RTSP_TRANSPORT
        argv += ["-rtsp_transport", transport_mode]
    argv += [
        "-i", url,
        "-frames:v", "1",   # exactly one decoded frame
        "-f", "image2",     # still-image (JPEG-family) muxer
        "-q:v", "2",        # high-quality JPEG
        "-y",               # overwrite output without prompting
        _clean_str(output) or "pipe:1",
    ]
    return argv


def still_url(normalized: Any) -> Optional[str]:
    """The direct still-image URL to GET for a JPEG source, else ``None``.

    For ``jpeg`` / ``jpeg_proxy`` this is the ``playback_url`` (the proxy or the
    bare URL respectively). For stream / none transports there is no single still
    URL — the caller must use :func:`one_frame_command` instead.
    """
    if not is_still(normalized):
        return None
    url = _clean_str(normalized.get("playback_url"))
    return url or None


def _no_network_fetch(url: str, *, timeout: float) -> bytes:
    """Default fetch seam: refuses to touch the network (offline-first)."""
    raise RuntimeError(
        "fetch_still requires an explicit fetch callable; the default performs "
        "no network I/O (offline-first). Pass fetch=<your urllib fetcher>."
    )


def fetch_still(
    normalized: Any,
    *,
    fetch: Callable[..., bytes] = _no_network_fetch,
    timeout: float = 10.0,
) -> Optional[bytes]:
    """Fetch the single still image for a JPEG source via an injectable seam.

    This is the ONLY function that *could* touch the network, and only if the
    caller passes its own ``fetch(url, timeout=...) -> bytes`` callable. The
    default seam raises rather than reaching out, so the module stays offline-safe
    and unit-testable with a fake fetcher. Returns ``None`` (no exception) for a
    non-JPEG / unusable camera, and ``None`` if the fetch itself fails.
    """
    url = still_url(normalized)
    if url is None:
        return None
    try:
        data = fetch(url, timeout=timeout)
    except Exception as exc:
        LOG.warning("still fetch failed for %s: %s", url, exc)
        return None
    return data if isinstance(data, (bytes, bytearray)) else None


def grab_command(
    normalized: Any,
    **kwargs: Any,
) -> dict:
    """Unified single-frame plan for ANY transport (pure; runs nothing).

    Returns a small descriptor the caller can act on without re-branching::

        { transport, kind: 'ffmpeg'|'still'|'none',
          argv: list|None, url: str|None }

    * stream  -> ``kind='ffmpeg'``, ``argv`` set, ``url`` None.
    * jpeg    -> ``kind='still'``,  ``url`` set,  ``argv`` None.
    * none    -> ``kind='none'``,   both None.
    """
    transport = (_clean_str(normalized.get("transport"))
                 if isinstance(normalized, dict) else TRANSPORT_NONE)
    if is_stream(normalized):
        return {
            "transport": transport,
            "kind": "ffmpeg",
            "argv": one_frame_command(normalized, **kwargs),
            "url": None,
        }
    if is_still(normalized):
        return {
            "transport": transport,
            "kind": "still",
            "argv": None,
            "url": still_url(normalized),
        }
    return {"transport": TRANSPORT_NONE, "kind": "none", "argv": None, "url": None}


# ─────────────────────────────────────────────────────────────────────────────
# Manifest builders (shapes ported from video_bridge.build_*; pure)
# ─────────────────────────────────────────────────────────────────────────────

def build_ingest_card(normalized: Any) -> dict:
    """One ingest card for a normalized camera (mirrors ``build_video_cards``).

    Never raises: a non-dict yields a ``none``-transport placeholder card.
    """
    cam: dict = normalized if isinstance(normalized, dict) else {}
    return {
        "id": cam.get("id"),
        "label": cam.get("label") or cam.get("id"),
        "group": cam.get("group") or "smoke",
        "transport": cam.get("transport") or TRANSPORT_NONE,
        "playback_url": cam.get("playback_url") or "",
        "source": cam.get("source") or "",
    }


def build_ingest_manifest(
    cameras: Any,
    relay_base: str = "",
) -> dict:
    """Normalize a list of cameras and emit one ingest manifest.

    Mirrors ``video_bridge.build_worldmonitor_manifest``: an ``ok`` flag, a UTC
    timestamp, a count, and a ``streams`` list — here each entry also carries the
    resolved single-frame ``kind`` (ffmpeg / still / none) so a consumer knows how
    to grab a frame without re-deriving it. Skips non-dict entries silently;
    never raises on a garbled list (or a non-list).
    """
    streams: List[dict] = []
    for cam in (cameras or []) if isinstance(cameras, (list, tuple)) else []:
        if not isinstance(cam, dict):
            continue
        norm = normalize_camera(cam, relay_base=relay_base)
        plan = grab_command(norm)
        streams.append({
            "id": norm.get("id"),
            "label": norm.get("label") or norm.get("id"),
            "transport": norm.get("transport") or TRANSPORT_NONE,
            "playback_url": norm.get("playback_url") or "",
            "source": norm.get("source") or "",
            "kind": plan["kind"],
            "lat": norm.get("lat"),
            "lon": norm.get("lon"),
        })
    return {
        "ok": True,
        "generated_at_utc": _utc_now(),
        "count": len(streams),
        "streams": streams,
    }
