#!/usr/bin/env python3
"""Pull-if-newer bundle client — models-and-data-are-DATA distribution.

The droplet (or any HTTP host we control) serves versioned bundles:

    GET <base>/bundles/<name>/current.json
        -> {"name", "version", "sha256", "size", "file"}
    GET <base>/bundles/<name>/<file>          (a .tar.gz)

This client keeps a persistent local cache and pulls ONLY when the served
version differs from the cached one (so a warm node never re-downloads), then
sha256-verifies before an atomic swap. Used for regime frame packs and BigML
champion bundles; ollama VLM blobs use ollama's own cache (Baseline 2).

Usage:
  python3 plugin/bundle_pull.py --base http://<host> --name proof-frames \
      --cache ~/.cache/smoke-harness/bundles
  # prints the extracted bundle dir on stdout; exit 0 = cache is current.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


def fetch_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pull(base: str, name: str, cache_root: Path, timeout: float = 30.0) -> Path:
    base = base.rstrip("/")
    cache = cache_root / name
    cache.mkdir(parents=True, exist_ok=True)
    state_path = cache / "state.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}

    cur = fetch_json(f"{base}/bundles/{name}/current.json", timeout)
    target = cache / "current"
    if (state.get("version") == cur["version"]
            and state.get("sha256") == cur["sha256"] and target.is_dir()):
        print(f"[bundle] {name} v{cur['version']} already cached", file=sys.stderr)
        return target

    url = f"{base}/bundles/{name}/{cur['file']}"
    print(f"[bundle] pulling {name} v{cur['version']} ({cur.get('size', '?')}B) "
          f"from {url}", file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=cache) as td:
        tar_path = Path(td) / cur["file"]
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tar_path, "wb") as out:
            shutil.copyfileobj(r, out)
        got = sha256_file(tar_path)
        if got != cur["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {name}: got {got[:12]} "
                               f"want {cur['sha256'][:12]} — refusing")
        extract = Path(td) / "extract"
        extract.mkdir()
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract, filter="data")   # no path traversal
        new = cache / f"v-{cur['version']}"
        if new.exists():
            shutil.rmtree(new)
        shutil.move(str(extract), str(new))
    # atomic-ish swap: repoint 'current' symlink, then persist state
    tmp_link = cache / ".current.tmp"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    tmp_link.symlink_to(new.name)
    tmp_link.replace(target)
    state_path.write_text(json.dumps({"version": cur["version"],
                                      "sha256": cur["sha256"]}))
    # prune old versions (keep current only — bundles are re-pullable)
    for d in cache.glob("v-*"):
        if d != new:
            shutil.rmtree(d, ignore_errors=True)
    return target


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cache", default="~/.cache/smoke-harness/bundles")
    args = ap.parse_args()
    out = pull(args.base, args.name, Path(args.cache).expanduser())
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
