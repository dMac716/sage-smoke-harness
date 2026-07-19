#!/usr/bin/env python3
"""C6 — the Beehive-tail insurance archiver (droplet timer).

Everything the harness publishes on a node survives crashes ONLY through
Beehive. This archiver tails the public data API for OUR plugin's
measurement names on OUR nodes and appends them to a local NDJSON archive —
so the crash-safe record also lives on infrastructure we own, watermarked
and idempotent (re-runs never duplicate: dedup on (timestamp,name,vsn,value)
within the overlap window).

Runs every 15 min via sage-beehive-archiver.timer. Zero credentials needed
for our own published measurements (public query API); Basic auth optional
via BEEHIVE_USER/BEEHIVE_PASS env for protected data.

Usage:
  python3 tools/beehive_archiver.py --out /srv/observatory/alertwest/harness_beehive \
      --vsn "H0.*" --names "run.*|detect|probe|champion|endphase|harness|detect.*|inference_ns"
  # --probe sys.uptime : one-shot plumbing check against system telemetry
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

API = "https://data.sagecontinuum.org/api/v1/query"
OVERLAP_MIN = 5          # re-query overlap so a slow publish never slips through


def query(start: str, vsn: str, names: str, timeout: float = 60,
          plugin: str = "") -> list:
    filt = {"vsn": vsn, "name": names}
    if plugin:
        # plugin-meta scoping — the precise cut: name prefixes like env.* are
        # the SHARED ontology (first run without this archived 89k rows of
        # other plugins' bird counts). Verified the API accepts it.
        filt["plugin"] = plugin
    body = {"start": start, "filter": filt}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    user, pw = os.environ.get("BEEHIVE_USER"), os.environ.get("BEEHIVE_PASS")
    if user and pw:
        req.add_header("Authorization", "Basic "
                       + base64.b64encode(f"{user}:{pw}".encode()).decode())
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r.read().decode().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vsn", default="H0.*")
    # GOTCHA (measured 2026-07-18): the data API 500s on ^/$ anchors and
    # parenthesized groups in the name filter — only unanchored patterns with
    # .* and | alternation are safe.
    ap.add_argument("--names",
                    default="run.*|detect.*|probe.*|champion.*|endphase.*|"
                            "harness.*|window.*|env.*|log.*|event.*|"
                            "inference_ns|sys.harness.*")
    ap.add_argument("--plugin", default=".*smoke-harness.*",
                    help="plugin-meta regex scope (empty = no plugin filter)")
    ap.add_argument("--harness-ingest", default="",
                    help="also POST new rows to this /harness/ingest URL "
                         "(the Beehive-tail leg of dual-publish: node runs "
                         "appear on the live page ~2min behind real time)")
    ap.add_argument("--probe", default="",
                    help="one-shot: query this name for -30min and print count "
                         "(plumbing check; nothing archived)")
    args = ap.parse_args()

    if args.probe:
        start = (datetime.now(timezone.utc) - timedelta(minutes=30)
                 ).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = query(start, args.vsn, args.probe)   # unanchored (API 500s on ^$)
        print(f"probe {args.probe}: {len(rows)} records since {start}")
        return 0 if rows else 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wm_path = out_dir / "watermark.json"
    archive = out_dir / "archive.ndjson"

    wm = {}
    if wm_path.exists():
        try:
            wm = json.loads(wm_path.read_text())
        except json.JSONDecodeError:
            wm = {}
    last = wm.get("last_ts")
    start_dt = (datetime.fromisoformat(last.replace("Z", "+00:00"))
                - timedelta(minutes=OVERLAP_MIN)) if last else \
        datetime.now(timezone.utc) - timedelta(days=7)
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = query(start, args.vsn, args.names, plugin=args.plugin)
    # dedup within the overlap window against the archive tail
    seen = set()
    if archive.exists():
        with open(archive, "rb") as fh:
            try:
                fh.seek(-2_000_000, 2)
            except OSError:
                fh.seek(0)
            for line in fh.read().decode(errors="replace").splitlines():
                try:
                    r = json.loads(line)
                    seen.add((r.get("timestamp"), r.get("name"),
                              (r.get("meta") or {}).get("vsn"), str(r.get("value"))))
                except json.JSONDecodeError:
                    continue
    new = [r for r in rows
           if (r.get("timestamp"), r.get("name"),
               (r.get("meta") or {}).get("vsn"), str(r.get("value"))) not in seen]
    if new:
        with open(archive, "a") as fh:
            for r in new:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        if args.harness_ingest:
            # group by run: meta.run_id if the harness published one, else a
            # stable per-(vsn,job) synthetic id — one live-page card per job
            by_run = {}
            for r in new:
                m = r.get("meta") or {}
                rid = m.get("run_id") or f"beehive-{m.get('vsn','?')}-{m.get('job','job')}"
                by_run.setdefault(rid, []).append(
                    {"name": r.get("name"), "value": r.get("value"),
                     "meta": m, "ts": r.get("timestamp")})
            for rid, recs in by_run.items():
                try:
                    req = urllib.request.Request(
                        args.harness_ingest,
                        data=json.dumps({"run_id": rid, "records": recs}).encode(),
                        headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=15).read()
                except Exception as exc:  # noqa: BLE001 — live leg is best-effort
                    print(f"[archiver] harness-ingest failed for {rid}: {exc}")
    max_ts = max((r.get("timestamp") for r in rows), default=last)
    if max_ts:
        wm_path.write_text(json.dumps({"last_ts": max_ts,
                                       "updated": datetime.now(timezone.utc).isoformat()}))
    print(f"[archiver] queried since {start}: {len(rows)} rows, {len(new)} new, "
          f"watermark {max_ts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
