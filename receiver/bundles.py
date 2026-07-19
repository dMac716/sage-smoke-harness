"""Versioned bundle distribution — the droplet as the harness's source of
truth for models-and-data-as-DATA (regime frame packs, BigML champion
bundles). Read-only: bundles are placed by the operator (scp/rsync into
``<OBSERVATORY_ROOT>/bundles/<name>/``); this blueprint only serves them.

Layout per bundle:
    bundles/<name>/current.json   {"name","version","sha256","size","file"}
    bundles/<name>/<file>         the .tar.gz current.json points at

Client: plugin/bundle_pull.py (pull-if-newer + sha256 verify + atomic swap).
Reachability: tailnet-only via tailscale serve, same as the rest of the
portal — no public surface.
"""
from __future__ import annotations

import re
from pathlib import Path

from flask import Blueprint, current_app, jsonify, send_file

bp = Blueprint("bundles", __name__, url_prefix="/bundles")

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_FILE_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}\.tar\.gz$")


def _root() -> Path:
    return Path(current_app.config["OBSERVATORY_ROOT"]) / "bundles"


@bp.get("/")
def index():
    out = []
    root = _root()
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if d.is_dir() and (d / "current.json").exists():
                out.append(d.name)
    return jsonify({"ok": True, "bundles": out})


@bp.get("/<name>/current.json")
def current(name: str):
    if not _NAME_RE.match(name):
        return jsonify({"ok": False, "error": "bad name"}), 400
    p = _root() / name / "current.json"
    if not p.exists():
        return jsonify({"ok": False, "error": "no such bundle"}), 404
    return send_file(p, mimetype="application/json", max_age=30)


@bp.get("/<name>/<file>")
def blob(name: str, file: str):
    if not _NAME_RE.match(name) or not _FILE_RE.match(file):
        return jsonify({"ok": False, "error": "bad path"}), 400
    p = _root() / name / file
    if not p.exists():
        return jsonify({"ok": False, "error": "no such file"}), 404
    return send_file(p, mimetype="application/gzip", max_age=3600,
                     as_attachment=True, download_name=file)


def register(app) -> None:
    app.register_blueprint(bp)
