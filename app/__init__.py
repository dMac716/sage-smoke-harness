"""Smoke Spotter — Waggle/Sage edge-AI wildfire-smoke detection plugin.

Modules:
  smoke            — pure-CPU change-gate + tiled rolling-baseline detector (no node/VLM deps)
  vlm              — pluggable VLM backend (node | ollama | none), degrades gracefully
  thresholds       — single source of truth for detection thresholds + the VLM prose whitelist
  cameras          — public ALERTCalifornia camera discovery (stdlib urllib, no auth)
  ingest_normalize — RTSP / HLS / JPEG transport normalization to one frame path
  main             — the pywaggle plugin loop (lazy pywaggle import)
  dev_run          — offline runner (no node, no API, no pywaggle); shares main's cascade
  query            — the READ side: query env.smoke.* back out of Beehive (lazy sage-data-client)
"""

__all__ = ["smoke", "vlm", "thresholds", "cameras", "ingest_normalize"]
__version__ = "0.2.0"
