#!/usr/bin/env python3
"""Pluggable vision-language-model (VLM) backend for the Smoke Spotter plugin.

The cheap CPU screen in ``smoke.py`` decides *which* frames are worth a second
look; this module performs that second look with a VLM — but only ever as an
optional escalation. Everything here degrades gracefully: if no backend is
available the plugin still runs offline and emits ``smoke=unknown``.

Backends (selected by name or auto-probed):

  * ``"apple"``  — an Apple-Silicon premium escalation tier: a local Apple
                   Foundation Models / Core AI HTTP sidecar (WWDC-2026) running a
                   full VLM on the Neural Engine at sub-watt. Address comes from
                   env; degrades gracefully to ``unknown`` if the sidecar is
                   absent (most nodes are cheap Linux/Jetson and won't have one).
  * ``"node"``   — a model bundled on the Waggle node (served over a local
                   OpenAI-compatible / Ollama-style HTTP endpoint, typically the
                   node's on-board inference container). Address comes from env.
  * ``"ollama"`` — a local Ollama daemon (great for offline dev on a laptop).
  * ``"none"``   — explicit no-op; always returns ``unknown``.
  * ``"auto"``   — try apple, then node, then ollama, then fall back to none.

A backend is anything with ``.detect(image_bytes, prompt=None) -> VLMResult``.
``smoke`` is ``True`` / ``False`` / ``None`` (None == "unknown / unavailable").

No third-party VLM library is imported; HTTP is done with the stdlib so the file
imports cleanly with nothing installed.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

# Free-text VLM-reply whitelist (prose -> bool) for the strict-JSON fallback.
# thresholds is a stdlib-only LEAF (imports nothing from app/), so this can't
# cycle. Dual import so this file works imported as ``app.vlm`` or top-level ``vlm``.
try:
    from app import thresholds as thresholds_mod
except ImportError:  # pragma: no cover - only when run from inside app/
    import thresholds as thresholds_mod  # type: ignore

LOG = logging.getLogger("smoke_spotter.vlm")

DEFAULT_PROMPT = (
    "You are a wildfire smoke spotter looking at a single outdoor camera frame. "
    "Is there visible wildfire smoke or an active fire (not clouds, fog, or "
    "lens haze)? Answer strictly as JSON: "
    '{"smoke": true|false, "detail": "<one short sentence>"}.'
)


class VLMResult:
    """Outcome of a VLM call.

    ``smoke`` is True / False, or None when undetermined (backend unavailable,
    timed out, or returned an unparseable answer). ``detail`` is a short human
    string; ``backend`` records which backend produced it.
    """

    __slots__ = ("smoke", "detail", "backend")

    def __init__(self, smoke: Optional[bool], detail: Optional[str],
                 backend: str = "none"):
        self.smoke = smoke
        self.detail = detail
        self.backend = backend

    @property
    def label(self) -> str:
        if self.smoke is True:
            return "smoke"
        if self.smoke is False:
            return "clear"
        return "unknown"

    def as_dict(self) -> dict:
        return {"smoke": self.smoke, "detail": self.detail,
                "backend": self.backend, "label": self.label}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"VLMResult(label={self.label!r}, backend={self.backend!r})"


UNKNOWN = VLMResult(smoke=None, detail=None, backend="none")


# ─────────────────────────────────────────────────────────────────────────────
# Response parsing (pure — no I/O, unit-testable offline)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_smoke_field(value) -> Optional[bool]:
    """Strictly interpret a JSON ``smoke`` field into True / False / None.

    The old code did ``bool(value)``, which silently turns the STRINGS
    ``"false"`` / ``"no"`` / ``"unknown"`` (all non-empty -> truthy) into
    ``True`` — a malformed-but-plausible model reply could thus fabricate a smoke
    alert. We parse strictly instead:

      * a real ``bool``           -> itself (the only fully-trusted form);
      * a number ``1`` / ``0``    -> True / False (other numbers -> None);
      * a string                  -> the prose whitelist (``"false"``/``"no smoke"``
                                     -> False, ``"yes"``/``"smoke"`` -> True, else None);
      * anything else / ``None``  -> None (undetermined).

    ``isinstance(True, int)`` is True in Python, so the bool check MUST come first.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        return thresholds_mod.vlm_text_to_smoke(value)
    return None


def _result_from_ollama_response(raw: Optional[str], backend_name: str) -> VLMResult:
    """Turn an Ollama ``response`` string into a VLMResult. Pure, never raises.

    The model is asked for strict JSON ``{"smoke": bool, "detail": str}`` but
    small edge VLMs routinely answer in PROSE instead. Decision order:

      1. Try to parse ``raw`` as JSON; read a strict ``smoke`` field from it.
      2. If that yields no verdict (missing key, unparseable JSON, or an
         ambiguous string), SALVAGE one from the prose via the whitelist
         (``thresholds.vlm_text_to_smoke``) instead of always degrading to
         ``unknown`` — "No, just clouds over the ridge" -> False, not unknown.
      3. ``detail`` is the model's ``detail`` field if present, else the raw prose
         (truncated), so the verdict stays explainable downstream.
    """
    raw = raw or ""
    parsed = None
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except (ValueError, json.JSONDecodeError):
        parsed = None  # prose, not JSON — handled by the whitelist fallback below

    smoke_bool: Optional[bool] = None
    detail: Optional[str] = None
    if isinstance(parsed, dict):
        # The model returned JSON. Read a strict verdict from the ``smoke`` field.
        if "smoke" in parsed:
            smoke_bool = _parse_smoke_field(parsed.get("smoke"))
        d = parsed.get("detail")
        if d:
            detail = str(d)[:240]
        # If the JSON gave no usable verdict, salvage from the DETAIL prose ONLY —
        # NOT the raw envelope: the JSON text literally contains the key "smoke",
        # which would fool the whitelist into a false True.
        if smoke_bool is None and detail:
            smoke_bool = thresholds_mod.vlm_text_to_smoke(detail)
    else:
        # The response was PROSE, not JSON (small VLMs ignore the format ask) —
        # salvage a verdict from the whole reply.
        smoke_bool = thresholds_mod.vlm_text_to_smoke(raw)
        if raw.strip():
            detail = raw.strip()[:240]

    return VLMResult(smoke=smoke_bool, detail=detail, backend=backend_name)


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

class NoneBackend:
    """Always-unknown backend. The safe offline default."""

    name = "none"
    available = True

    def detect(self, image_bytes: bytes, prompt: Optional[str] = None) -> VLMResult:
        return VLMResult(smoke=None, detail=None, backend="none")


class OllamaStyleBackend:
    """VLM over an Ollama-compatible HTTP API (``/api/generate``).

    Used for BOTH the node-bundled model and a local Ollama daemon — they speak
    the same wire format; only the host/model differ. Probes ``/api/tags`` at
    construction; if the endpoint is unreachable, ``available`` is False and the
    plugin treats this backend as absent.
    """

    def __init__(self, *, name: str, host: str, model: str,
                 timeout: float = 60.0, probe_timeout: float = 3.0):
        self.name = name
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.available = self._probe(probe_timeout)
        if self.available:
            LOG.info("VLM backend %r ready (%s @ %s)", name, model, self.host)
        else:
            LOG.info("VLM backend %r unavailable at %s — will return unknown",
                     name, self.host)

    def _probe(self, probe_timeout: float) -> bool:
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=probe_timeout) as resp:  # noqa: S310
                resp.read()
            return True
        except Exception as exc:
            LOG.debug("VLM probe failed for %s: %s", self.host, exc)
            return False

    def detect(self, image_bytes: bytes, prompt: Optional[str] = None) -> VLMResult:
        if not self.available:
            return VLMResult(smoke=None, detail=None, backend=self.name)
        body = json.dumps({
            "model": self.model,
            "prompt": prompt or DEFAULT_PROMPT,
            "images": [base64.b64encode(image_bytes).decode("ascii")],
            "stream": False,
            "format": "json",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                out = json.loads(resp.read().decode("utf-8"))
            # Strict-JSON parse with a prose-whitelist fallback (pure helper).
            return _result_from_ollama_response(out.get("response"), self.name)
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            LOG.warning("VLM call failed (%s): %s", self.name, exc)
            return VLMResult(smoke=None, detail=None, backend=self.name)


class AppleFMBackend(OllamaStyleBackend):
    """Premium VLM tier on an Apple-Silicon node (WWDC-2026 frameworks).

    Talks to a small local HTTP sidecar that fronts Apple's on-device VLM —
    the **Foundation Models API** / new **Core AI** framework — which runs a
    full-scale model on the Neural Engine against Apple Silicon's unified
    memory, at sub-watt. On a Sage fleet of cheap Linux/Jetson nodes, an
    Apple-Silicon node can therefore serve as a *premium escalation tier*:
    richer, more explainable smoke verdicts for the small fraction of frames
    the cheap gate flags.

    The sidecar (operator-owned, see ``~/Repos/AppleSiliconVision``) is expected
    to expose the same Ollama-style ``/api/generate`` + ``/api/tags`` wire format
    as the other backends, so this reuses :class:`OllamaStyleBackend` verbatim.
    If the sidecar is not running — which is the case on every non-Apple node —
    the probe fails, ``available`` is False, and ``detect`` returns ``unknown``:
    the plugin degrades gracefully exactly like a missing node/ollama backend.

    Configured via ``SMOKE_VLM_APPLE_HOST`` (default ``http://127.0.0.1:8080``)
    and ``SMOKE_VLM_APPLE_MODEL`` (default ``apple-fm-vlm``).
    """

    def __init__(self, *, host: str, model: str, timeout: float = 60.0,
                 probe_timeout: float = 3.0):
        super().__init__(name="apple", host=host, model=model,
                         timeout=timeout, probe_timeout=probe_timeout)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def build_backend(
    kind: str = "auto",
    *,
    apple_host: Optional[str] = None,
    apple_model: Optional[str] = None,
    node_host: Optional[str] = None,
    node_model: Optional[str] = None,
    ollama_host: Optional[str] = None,
    ollama_model: Optional[str] = None,
    timeout: float = 60.0,
):
    """Build a VLM backend by name. Always returns a usable object (never None).

    ``kind``: ``apple`` | ``node`` | ``ollama`` | ``none`` | ``auto``.
    Hosts/models default from env vars so node operators can configure without
    code changes:

      SMOKE_VLM_APPLE_HOST   (default http://127.0.0.1:8080)
      SMOKE_VLM_APPLE_MODEL  (default apple-fm-vlm)
      SMOKE_VLM_NODE_HOST    (default http://127.0.0.1:8000)
      SMOKE_VLM_NODE_MODEL   (default smoke-vlm)
      SMOKE_VLM_OLLAMA_HOST  (default http://127.0.0.1:11434)
      SMOKE_VLM_OLLAMA_MODEL (default moondream)

    The Ollama default is ``moondream`` — one of Sage's edge-optimized VLMs;
    ``florence`` (Microsoft Florence-2) is a good alternative escalation model.
    The ``apple`` backend is an optional Apple-Silicon premium tier (Foundation
    Models / Core AI sidecar) and is tried first in ``auto``; it is absent on
    cheap Linux/Jetson nodes and simply degrades to the next backend.
    """
    kind = (kind or "auto").lower()
    apple_host = apple_host or _env("SMOKE_VLM_APPLE_HOST", "http://127.0.0.1:8080")
    apple_model = apple_model or _env("SMOKE_VLM_APPLE_MODEL", "apple-fm-vlm")
    node_host = node_host or _env("SMOKE_VLM_NODE_HOST", "http://127.0.0.1:8000")
    node_model = node_model or _env("SMOKE_VLM_NODE_MODEL", "smoke-vlm")
    ollama_host = ollama_host or _env("SMOKE_VLM_OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model = ollama_model or _env("SMOKE_VLM_OLLAMA_MODEL", "moondream")

    def _apple():
        return AppleFMBackend(host=apple_host, model=apple_model, timeout=timeout)

    def _node():
        return OllamaStyleBackend(name="node", host=node_host,
                                  model=node_model, timeout=timeout)

    def _ollama():
        return OllamaStyleBackend(name="ollama", host=ollama_host,
                                  model=ollama_model, timeout=timeout)

    if kind == "none":
        return NoneBackend()
    if kind == "apple":
        b = _apple()
        return b if b.available else NoneBackend()
    if kind == "node":
        b = _node()
        return b if b.available else NoneBackend()
    if kind == "ollama":
        b = _ollama()
        return b if b.available else NoneBackend()

    # auto: apple -> node -> ollama -> none
    for builder in (_apple, _node, _ollama):
        b = builder()
        if b.available:
            return b
    LOG.info("no VLM backend reachable — escalation will report unknown")
    return NoneBackend()
