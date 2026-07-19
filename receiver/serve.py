#!/usr/bin/env python3
"""Standalone harness receiver — the "bring your own redundancy" server.

Run this on ANY host that sits on YOUR tailnet (a cheap VM, a droplet, a spare
Pi) and it gives you the same live-feed + KILL + bundle-serving the reference
setup has — no observatory portal required. The plugin's `regime` / tailnet
modes point at it via BUNDLE_BASE / HARNESS_RECEIVER.

Endpoints (same contract the plugin speaks):
  POST /harness/ingest            live measurements (dual-publish live leg)
  GET  /harness/api/runs          list runs
  GET  /harness/api/tail/<run>    byte-offset tail of a run
  GET  /harness/control/<run>     poll the soft-kill flag
  POST /harness/kill/<run>        set the kill flag
  GET  /harness/                  live page
  GET  /bundles/<name>/current.json + /<file>   serve frame/model bundles

Setup for a teammate wanting the same redundancy (nothing here breaks the
reference tailnet — keys are tailnet-scoped):
  1. Mint YOUR OWN Tailscale auth key (ephemeral+reusable) on YOUR tailnet.
  2. Run this on a host joined to YOUR tailnet:
        pip install flask
        DATA_ROOT=./harness-data python3 receiver/serve.py --port 8777
     then `tailscale serve 8777` (or bind it however your tailnet reaches it).
  3. Drop frame bundles under $DATA_ROOT/bundles/<name>/ (current.json + tgz).
  4. Submit the plugin with YOUR env:
        TS_AUTHKEY=<your key>  BUNDLE_BASE=http://<your-host-magicdns>
     (HARNESS_RECEIVER defaults to BUNDLE_BASE).

Tailnet-only by design — do not expose this on a public interface.
"""
import argparse
import os
from pathlib import Path

from flask import Flask

import bundles as bundles_mod
import harness_live


def create_app(root: str) -> Flask:
    app = Flask(__name__)
    Path(root).mkdir(parents=True, exist_ok=True)
    app.config["OBSERVATORY_ROOT"] = root      # the blueprints' only dependency
    app.config["MAX_CONTENT_LENGTH"] = 8_000_000
    harness_live.register(app)
    bundles_mod.register(app)
    return app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("DATA_ROOT", "./harness-data"))
    ap.add_argument("--host", default="127.0.0.1")   # tailnet-only; front with `tailscale serve`
    ap.add_argument("--port", type=int, default=8777)
    args = ap.parse_args()
    app = create_app(args.root)
    print(f"harness receiver on {args.host}:{args.port}, data root {args.root}")
    app.run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
