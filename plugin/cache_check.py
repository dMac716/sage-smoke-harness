#!/usr/bin/env python3
"""B2 persistence gate — does a hostPath volume survive across jobs?

Reads a marker under $CACHE_DIR (the pluginSpec.volume mount), publishes what
it found, then writes/updates the marker. The gate: a SECOND job fire reports
`cache.marker_found=1` carrying the FIRST fire's run id — proof the node-local
cache persists across pods, which is what makes multi-GB VLM caches viable
(pull once, reuse every job).

Publishes: cache.header, cache.marker_found (0/1 + prior run/count/age),
cache.marker_written, cache.exit. Read-only besides its own marker file.
"""
import json
import os
import time
from pathlib import Path

from waggle.plugin import Plugin

RUN_ID = os.environ.get("RUN_ID", f"cache-check-{int(time.time())}")
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache"))
MARKER = CACHE_DIR / "b2-persistence-marker.json"


def main() -> int:
    with Plugin() as plugin:
        def pub(name, value, meta=None):
            plugin.publish(name, value, meta={"run_id": RUN_ID, **(meta or {})})

        pub("cache.header", 1, meta={"cache_dir": str(CACHE_DIR)})
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 — an unwritable mount is the finding
            pub("cache.mount_unwritable", str(exc)[:200])
            return 3

        if MARKER.exists():
            try:
                prior = json.loads(MARKER.read_text())
                pub("cache.marker_found", 1, meta={
                    "prior_run": str(prior.get("run_id")),
                    "prior_count": str(prior.get("count")),
                    "age_s": f"{time.time() - float(prior.get('ts', 0)):.0f}"})
                count = int(prior.get("count", 0)) + 1
            except Exception as exc:  # noqa: BLE001
                pub("cache.marker_corrupt", str(exc)[:120])
                count = 1
        else:
            pub("cache.marker_found", 0)
            count = 1

        MARKER.write_text(json.dumps({"run_id": RUN_ID, "ts": time.time(),
                                      "count": count}))
        pub("cache.marker_written", count)
        # free-space visibility for future model-cache sizing
        st = os.statvfs(CACHE_DIR)
        pub("cache.exit", 1, meta={
            "free_gb": f"{st.f_bavail * st.f_frsize / 1e9:.1f}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
