#!/usr/bin/env python3
"""Public ALERTCalifornia camera discovery for the Smoke Spotter plugin.

This module answers one question — *which public outdoor cameras are near me?* —
so the offline dev loop (and, once node access is granted, the plugin) has a
catalogue of real wildfire-watch frame sources to point at. It is the lean,
stdlib-only port of the operator's TowerWatch ``firewatch_cameras.py`` scraper,
with the Azure-Blob + Bearer-ingest halves stripped entirely: we DISCOVER and
locate cameras, we do not upload frames or POST to any private ingest.

Source of truth: the **public ALERTCalifornia ArcGIS FeatureServer** (no auth) —
UC San Diego's WIFIRE Lab camera index. We query it, project each feature to a
flat ``{camera_id, name, lat, lon, image_url, attribution}`` dict, filter by a
haversine radius around a centre (default UC Davis), sort by distance, and cap.

Doctrine (Sage/Waggle): *move the work, not the data.* This module returns small
metadata (camera id + coords + a CDN URL), never image bytes — the heavy frame
fetch is somebody else's job and happens at most one camera at a time.

OFFLINE-FIRST (hard rule). The HTTP fetch is INJECTABLE: ``discover_cameras``
takes a ``fetch`` callable defaulting to a stdlib-``urllib`` fetcher, so tests
pass a fake that returns canned ArcGIS JSON and the whole module runs and tests
with **no network**. Nothing here ever raises on a bad upstream response — a
garbled / short / empty / non-JSON payload (or a ``fetch`` that throws) yields an
empty list, never an exception.

Two CDN gotchas are baked in as comments at their call sites and must not be
"simplified" away:

  1. The latest-frame CDN URL is keyed on the camera **slug** (e.g.
     ``Axis-AtlasPeakWest``) parsed out of the ArcGIS ``imageURL`` field — NOT on
     the human ``cameraName`` (``"Atlas Peak 1"``). Building the URL from the name
     produced global 404s in TowerWatch (commit ``4dc2d7f9``).
  2. The CDN returns a non-200 unless the request carries a browser-ish
     ``User-Agent`` **and** a ``Referer: https://cameras.alertcalifornia.org/``.

Ported from the operator's TowerWatch
``containerapps/scrapers/scrapers/firewatch_cameras.py`` (ArcGIS query, slug
regex, frame-CDN pattern, haversine, polite headers, public-domain attribution).
"""
from __future__ import annotations

import json
import logging
import math
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Tuple

LOG = logging.getLogger("smoke_spotter.cameras")

# ALERTCalifornia ArcGIS feature service — the public source of truth for camera
# locations + the canonical ``imageURL`` slug. No auth required. (Carried over
# verbatim from TowerWatch; verified upstream 2026-05-14.)
CAMERA_INDEX_URL = (
    "https://services8.arcgis.com/X84q166Srnyl4JMV/arcgis/rest/services/"
    "ALERTCalifornia_Camera_Feed/FeatureServer/0/query"
    "?where=1%3D1&outFields=cameraName,positionPan,viewTime,county,state,imageURL"
    "&f=geojson&resultRecordCount=2000"
)

# Latest-frame CDN template. The ``{slug}`` placeholder is the canonical camera
# slug, NOT the human cameraName — see ``_IMAGE_URL_SLUG_RE`` below and the
# module docstring's gotcha #1 (cameraName -> global 404s, TowerWatch 4dc2d7f9).
FRAME_URL_TEMPLATE = (
    "https://cameras.alertcalifornia.org/public-camera-data/{slug}/latest-frame.jpg"
)

# The canonical slug lives between ``/public-camera-data/`` and the
# ``/latest-(frame|thumb).jpg`` segment of the ArcGIS ``imageURL`` field, e.g.
# ``.../public-camera-data/Axis-AtlasPeakWest/latest-frame.jpg`` -> the slug is
# ``Axis-AtlasPeakWest``. cameraName ("Atlas Peak 1") does NOT map 1:1 to it.
_IMAGE_URL_SLUG_RE = re.compile(
    r"/public-camera-data/([^/]+)/latest-(?:frame|thumb)\.jpg",
    re.IGNORECASE,
)

# Polite + CDN-required request headers. The ALERTCalifornia frame CDN returns a
# non-200 (403/404) without BOTH a browser-ish User-Agent and this exact Referer
# — see the module docstring's gotcha #2. We also send these on the ArcGIS query
# to stay a good citizen (the FeatureServer is public but rate-aware).
USER_AGENT = "SmokeSpotter/0.1 (Sage/Waggle edge plugin; camera discovery)"
CDN_REFERER = "https://cameras.alertcalifornia.org/"

# Public-domain license + attribution carried over from TowerWatch. These ride
# along on every discovered camera so downstream consumers always have provenance.
LICENSE = "Public domain (UC San Diego ALERT California)"
ATTRIBUTION = "ALERT California / UC San Diego WIFIRE Lab"

# Discovery defaults. Centre on UC Davis (the hackathon's home node); 50 mi keeps
# the catalogue to the local foothills/valley without pulling the whole state.
DAVIS_LAT = 38.5449
DAVIS_LON = -121.7405
DEFAULT_RADIUS_MI = 50.0

# Earth radius in miles (mean) for the haversine. Matches TowerWatch's constant.
_EARTH_RADIUS_MI = 3958.7613

# urllib timeout for the (real) index fetch.
_HTTP_TIMEOUT_S = 30.0


# ─────────────────────────────────────────────────────────────────────────────
# Geo + URL helpers (pure, tested directly)
# ─────────────────────────────────────────────────────────────────────────────

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two WGS84 coordinates.

    Pure stdlib ``math``. Returns ``inf`` if any coordinate is non-numeric so a
    bad feature sorts/filters to the bottom rather than raising.
    """
    try:
        phi1 = math.radians(float(lat1))
        phi2 = math.radians(float(lat2))
        dphi = math.radians(float(lat2) - float(lat1))
        dlmb = math.radians(float(lon2) - float(lon1))
    except (TypeError, ValueError):
        return float("inf")
    a = (math.sin(dphi / 2.0) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2)
    return 2.0 * _EARTH_RADIUS_MI * math.asin(math.sqrt(a))


def frame_url(slug: str) -> str:
    """Build the latest-frame CDN URL from a camera SLUG (never the cameraName).

    The slug is the value parsed out of ArcGIS ``imageURL`` (e.g.
    ``Axis-AtlasPeakWest``). Using the human-readable ``cameraName`` here instead
    of the slug is exactly the bug that produced global 404s in TowerWatch
    (commit ``4dc2d7f9``) — so callers MUST pass the slug.
    """
    return FRAME_URL_TEMPLATE.format(slug=slug)


def slug_from_image_url(image_url: str) -> Optional[str]:
    """Extract the canonical camera slug from an ArcGIS ``imageURL``, or None."""
    if not isinstance(image_url, str) or not image_url:
        return None
    m = _IMAGE_URL_SLUG_RE.search(image_url)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP fetch (injectable so tests run fully offline)
# ─────────────────────────────────────────────────────────────────────────────

def _urllib_fetch(url: str, *, timeout: float = _HTTP_TIMEOUT_S) -> str:
    """Default fetcher: GET ``url`` with polite headers, return the body text.

    Stdlib ``urllib`` only. Sends the browser-ish User-Agent + the CDN Referer so
    the same fetcher works for both the ArcGIS query and (if ever reused) the
    frame CDN. Raises on any transport error — ``discover_cameras`` catches it and
    degrades to an empty list, so callers never see the exception.
    """
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/geo+json",
            "User-Agent": USER_AGENT,
            # CDN gotcha #2: the frame CDN 403/404s without this exact Referer.
            "Referer": CDN_REFERER,
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read()
    if isinstance(raw, (bytes, bytearray)):
        return bytes(raw).decode("utf-8", errors="replace")
    return str(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Feature projection (pure; never raises on a garbled feature)
# ─────────────────────────────────────────────────────────────────────────────

def _project_feature(feat: object, center: Tuple[float, float]) -> Optional[dict]:
    """Project one ArcGIS GeoJSON feature -> a camera dict (or None to skip).

    Skips silently (returns None) on any missing/garbled field — a bad feature
    must never abort the whole discovery. Returns a dict with an internal
    ``_distance_mi`` key (stripped before yield) used for sort + radius filter.
    """
    if not isinstance(feat, dict):
        return None
    props = feat.get("properties") or {}
    geom = feat.get("geometry") or {}
    if not isinstance(props, dict) or not isinstance(geom, dict):
        return None
    if geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates") or []
    if not isinstance(coords, (list, tuple)) or len(coords) < 2:
        return None
    # GeoJSON is [lon, lat] order.
    try:
        lon = float(coords[0])
        lat = float(coords[1])
    except (TypeError, ValueError):
        return None

    name = props.get("cameraName")
    if not isinstance(name, str) or not name:
        return None

    # The CDN URL is keyed on the slug parsed from imageURL, never on cameraName
    # (gotcha #1). No parseable slug -> we can't build a working frame URL -> skip.
    slug = slug_from_image_url(props.get("imageURL"))
    if not slug:
        return None

    dist = haversine_miles(center[0], center[1], lat, lon)
    return {
        "camera_id": slug,
        "name": name,
        "lat": lat,
        "lon": lon,
        "image_url": frame_url(slug),
        "attribution": ATTRIBUTION,
        "_distance_mi": dist,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def discover_cameras(
    center: Tuple[float, float] = (DAVIS_LAT, DAVIS_LON),
    radius_mi: float = DEFAULT_RADIUS_MI,
    limit: Optional[int] = None,
    *,
    fetch: Callable[[str], str] = _urllib_fetch,
) -> List[dict]:
    """Discover public ALERTCalifornia cameras near ``center``.

    Queries the ArcGIS FeatureServer (via the injectable ``fetch``), projects each
    feature, filters to within ``radius_mi`` (haversine) of ``center``, sorts by
    ascending distance, and caps at ``limit`` (if given).

    Returns a list of dicts::

        {camera_id, name, lat, lon, image_url, attribution}

    where ``image_url`` is the slug-keyed latest-frame CDN URL.

    OFFLINE / never-raise contract: ``fetch`` is injectable (default = stdlib
    urllib). ANY failure — fetch raises, body isn't JSON, the payload is short /
    empty / the wrong shape, individual features are garbled — degrades to ``[]``;
    this function never propagates an exception.
    """
    # 1. Fetch the index. A raising fetch (no network, timeout, 500) -> [].
    try:
        body = fetch(CAMERA_INDEX_URL)
    except Exception as exc:  # noqa: BLE001 — never let a transport error escape
        LOG.warning("camera index fetch failed: %s", exc)
        return []

    # 2. Parse JSON. Garbage / truncated / non-text -> [].
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as exc:
        LOG.warning("camera index not valid JSON: %s", exc)
        return []

    # 3. Validate the shape. Anything but a dict with a ``features`` list -> [].
    if not isinstance(payload, dict):
        LOG.warning("camera index payload not an object (%s)", type(payload).__name__)
        return []
    features = payload.get("features")
    if not isinstance(features, list):
        LOG.warning("camera index has no features list")
        return []

    # 4. Project + radius-filter. Each feature is guarded individually so one bad
    #    record can't sink the batch. Coerce ``radius_mi`` ONCE, defensively: a
    #    non-numeric radius (e.g. an unparsed config string) must degrade to the
    #    default rather than raise inside the loop — the never-raise contract
    #    covers caller-supplied scalars too, exactly like ``limit`` below.
    try:
        radius = float(radius_mi)
    except (TypeError, ValueError):
        LOG.warning("bad radius_mi %r; using default %.1f", radius_mi, DEFAULT_RADIUS_MI)
        radius = DEFAULT_RADIUS_MI
    if radius != radius:  # NaN -> default (NaN comparisons are always False)
        radius = DEFAULT_RADIUS_MI
    cams: List[dict] = []
    for feat in features:
        try:
            cam = _project_feature(feat, center)
        except Exception as exc:  # noqa: BLE001 — defensive; projection is pure
            LOG.debug("skipping unprojectable feature: %s", exc)
            continue
        if cam is None:
            continue
        if cam["_distance_mi"] > radius:
            continue
        cams.append(cam)

    # 5. Distance sort (nearest first), then cap.
    cams.sort(key=lambda c: c["_distance_mi"])
    if limit is not None:
        try:
            n = int(limit)
        except (TypeError, ValueError):
            n = None
        if n is not None and n >= 0:
            cams = cams[:n]

    # 6. Strip the internal distance key — the public contract is exactly the six
    #    documented fields.
    for c in cams:
        c.pop("_distance_mi", None)
    return cams
