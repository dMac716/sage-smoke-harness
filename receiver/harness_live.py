"""Live test-harness feed — receiver + live view + KILL (plan C2+C3, rung R0.5).

The in-job harness (plugin/harness_sink.py) dual-publishes: every measurement
goes to Beehive via pywaggle (crash-safe insurance) AND here over the tailnet
(live, ours). This module is the "here": an ingest endpoint that appends each
record to an NDJSON file per run, a control endpoint the harness polls for the
kill flag, and a plain live page to watch a run as it happens.

Reachability/auth: the portal is served tailnet-only (tailscale serve → this
gunicorn); nothing public. Persistence is append-only NDJSON under
``<OBSERVATORY_ROOT>/harness_runs/`` — the run file IS the archive, no DB.

Honesty: this live feed is a CONVENIENCE view; the crash-safe record of a Sage
run is the Beehive-published stream. Losing the live feed loses nothing.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template_string, request

bp = Blueprint("harness_live", __name__, url_prefix="/harness")

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_MAX_BATCH = 1000
_MAX_VALUE_LEN = 65536  # one traceback fits; nobody streams video through this


def _runs_dir() -> Path:
    d = Path(current_app.config["OBSERVATORY_ROOT"]) / "harness_runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.match(run_id or ""):
        raise ValueError(f"bad run_id: {run_id!r}")
    return run_id


@bp.post("/ingest")
def ingest():
    body = request.get_json(force=True, silent=True) or {}
    try:
        run_id = _safe_run_id(str(body.get("run_id", "")))
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    records = body.get("records") or []
    if not isinstance(records, list) or len(records) > _MAX_BATCH:
        return jsonify({"ok": False, "error": "records must be a list <= %d" % _MAX_BATCH}), 400
    received_at = time.time()
    out = []
    for rec in records:
        if not isinstance(rec, dict) or "name" not in rec:
            continue
        val = rec.get("value")
        if isinstance(val, str) and len(val) > _MAX_VALUE_LEN:
            val = val[:_MAX_VALUE_LEN] + "…[truncated]"
        out.append(json.dumps({
            "name": str(rec["name"])[:128],
            "value": val,
            "meta": rec.get("meta") or {},
            "ts": rec.get("ts"),
            "received_at": received_at,
        }, separators=(",", ":"), default=str))
    if out:
        with open(_runs_dir() / f"{run_id}.ndjson", "a", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
    return jsonify({"ok": True, "n": len(out),
                    "kill": (_runs_dir() / f"{run_id}.KILL").exists()})


@bp.get("/control/<run_id>")
def control(run_id: str):
    try:
        run_id = _safe_run_id(run_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True, "kill": (_runs_dir() / f"{run_id}.KILL").exists()})


@bp.post("/kill/<run_id>")
def kill(run_id: str):
    """Set the kill flag. The harness polls /control and aborts; on Sage the
    independent paths (sesctl rm, k3s limits) still exist — this is the soft one."""
    try:
        run_id = _safe_run_id(run_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    flag = _runs_dir() / f"{run_id}.KILL"
    flag.write_text(json.dumps({"killed_at": time.time(),
                                "by": request.remote_addr}))
    return jsonify({"ok": True, "kill": True})


@bp.get("/api/runs")
def api_runs():
    runs = []
    for p in sorted(_runs_dir().glob("*.ndjson"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        rid = p.stem
        runs.append({
            "run_id": rid,
            "bytes": p.stat().st_size,
            "mtime": p.stat().st_mtime,
            "killed": (p.parent / f"{rid}.KILL").exists(),
        })
    return jsonify({"ok": True, "runs": runs[:100]})


@bp.get("/api/tail/<run_id>")
def api_tail(run_id: str):
    """Byte-offset tail so the page only ever pulls what's new."""
    try:
        run_id = _safe_run_id(run_id)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    path = _runs_dir() / f"{run_id}.ndjson"
    if not path.exists():
        return jsonify({"ok": False, "error": "no such run"}), 404
    try:
        offset = max(0, int(request.args.get("offset", "0")))
    except ValueError:
        offset = 0
    size = path.stat().st_size
    lines = []
    if offset < size:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            chunk = fh.read(min(size - offset, 2_000_000))
        # only complete lines; leave a partial trailing line for next poll
        upto = chunk.rfind("\n")
        if upto >= 0:
            lines = [json.loads(l) for l in chunk[:upto].splitlines() if l.strip()]
            offset += upto + 1
    return jsonify({"ok": True, "records": lines, "offset": offset,
                    "kill": (path.parent / f"{run_id}.KILL").exists()})


_PAGE = """<!doctype html>
<title>Harness — live runs</title>
<style>
 body { font: 14px/1.45 -apple-system, system-ui, sans-serif; margin: 1.2rem; color:#222; }
 h1 { font-size: 1.2rem; } code { background:#f2f2f2; padding:0 3px; }
 select, button { font: inherit; padding: .25rem .6rem; }
 #killbtn { background:#b91c1c; color:#fff; border:none; border-radius:4px; cursor:pointer; }
 #killbtn[disabled] { background:#ccc; cursor:default; }
 .row { display:flex; gap:2rem; flex-wrap:wrap; margin-top:1rem; }
 .col { flex:1 1 420px; min-width:380px; }
 #console { background:#111; color:#ddd; font:12px/1.4 ui-monospace,monospace;
            padding:.6rem; height:340px; overflow-y:auto; white-space:pre-wrap; }
 table { border-collapse:collapse; width:100%; font-size:12.5px; }
 td,th { border-bottom:1px solid #e5e5e5; padding:2px 6px; text-align:left; }
 tr.alert td { background:#fee2e2; }
 .stat { display:inline-block; margin-right:1.2rem; }
 .stat b { font-size:1.15rem; }
 #status.killed { color:#b91c1c; font-weight:600; }
 .note { color:#666; font-size:12.5px; max-width:60rem; }
</style>
<h1>Test-harness live feed</h1>
<p class="note">Live view of in-job harness runs (dual-published over the tailnet).
This page is a convenience view; the crash-safe record is the published
measurement stream (Beehive on Sage, <code>data.ndjson</code> locally).
KILL sets a flag the harness polls — on Sage, <code>sesctl rm</code> and k3s
limits remain the independent hard paths.</p>
<div>
 run: <select id="runsel"></select>
 <button onclick="loadRuns()">refresh list</button>
 <button id="killbtn" onclick="killRun()">KILL RUN</button>
 <span id="status" class="stat"></span>
</div>
<div class="row">
 <div class="col">
  <div><span class="stat">frames <b id="s_frames">0</b></span>
       <span class="stat">alerts <b id="s_alerts">0</b></span>
       <span class="stat">errors <b id="s_errs">0</b></span>
       <span class="stat">last rss <b id="s_rss">–</b> MB</span></div>
  <h3>frames</h3>
  <div style="max-height:340px;overflow-y:auto">
  <table id="frames"><tr><th>event</th><th>#</th><th>diff</th><th>smoke</th>
  <th>tile</th><th>alert</th></tr></table></div>
 </div>
 <div class="col">
  <h3>console</h3>
  <div id="console"></div>
 </div>
</div>
<script>
let run = null, offset = 0, frames = {}, stats = {frames:0, alerts:0, errs:0};
async function loadRuns() {
  const r = await (await fetch('/harness/api/runs')).json();
  const sel = document.getElementById('runsel');
  const cur = sel.value;
  sel.innerHTML = '';
  for (const it of r.runs) {
    const o = document.createElement('option');
    o.value = it.run_id;
    o.textContent = it.run_id + (it.killed ? '  [KILLED]' : '');
    sel.appendChild(o);
  }
  if (cur && [...sel.options].some(o => o.value === cur)) sel.value = cur;
  if (sel.value !== run) switchRun(sel.value);
}
function switchRun(rid) {
  run = rid; offset = 0; frames = {}; stats = {frames:0, alerts:0, errs:0};
  document.getElementById('console').textContent = '';
  document.querySelectorAll('#frames tr:not(:first-child)').forEach(t => t.remove());
  updStats();
}
document.getElementById('runsel').addEventListener('change',
  e => switchRun(e.target.value));
function updStats() {
  s_frames.textContent = stats.frames; s_alerts.textContent = stats.alerts;
  s_errs.textContent = stats.errs;
}
function addFrame(m, rec) {
  const key = (m.event || '') + '#' + (m.frame || '');
  let row = frames[key];
  if (!row) {
    row = frames[key] = {tr: document.createElement('tr'), d: {}};
    const tb = document.getElementById('frames');
    row.tr.innerHTML = `<td>${(m.event||'').slice(0,18)}</td><td>${m.frame||''}</td>
      <td class=d></td><td class=s></td><td class=t></td><td class=a></td>`;
    tb.appendChild(row.tr);
    row.tr.scrollIntoView({block:'nearest'});
    stats.frames++;
  }
  if (rec.name === 'detect.diff') row.tr.querySelector('.d').textContent = (+rec.value).toFixed(2);
  if (rec.name === 'detect.smoke_score') row.tr.querySelector('.s').textContent = (+rec.value).toFixed(3);
  if (rec.name === 'detect.max_tile_score') row.tr.querySelector('.t').textContent = (+rec.value).toFixed(3);
  if (rec.name === 'detect.alert' && +rec.value) {
    row.tr.classList.add('alert');
    row.tr.querySelector('.a').textContent = 'ALERT';
    stats.alerts++;
  }
}
async function poll() {
  if (!run) return;
  try {
    const r = await (await fetch(`/harness/api/tail/${run}?offset=${offset}`)).json();
    if (!r.ok) return;
    offset = r.offset;
    const con = document.getElementById('console');
    for (const rec of r.records) {
      const m = rec.meta || {};
      if (rec.name === 'log.console')
        con.textContent += `[${m.level||''}] ${rec.value}\\n`;
      else if (rec.name.startsWith('detect.')) addFrame(m, rec);
      else if (rec.name === 'harness.traceback') {
        stats.errs++;
        con.textContent += `--- TRACEBACK (${m.event||''} f${m.frame||''}) ---\\n${rec.value}\\n`;
      }
      else if (rec.name === 'sys.harness.maxrss_kb')
        s_rss.textContent = (rec.value/1024).toFixed(0);
      else if (rec.name === 'run.exit')
        con.textContent += `=== RUN EXIT ${JSON.stringify(m)} ===\\n`;
    }
    if (r.records.length) con.scrollTop = con.scrollHeight;
    const st = document.getElementById('status');
    st.textContent = r.kill ? 'KILL FLAG SET' : 'live';
    st.className = r.kill ? 'stat killed' : 'stat';
    updStats();
  } catch (e) { /* transient */ }
}
async function killRun() {
  if (!run || !confirm(`Set the kill flag for ${run}?`)) return;
  await fetch(`/harness/kill/${run}`, {method: 'POST'});
}
loadRuns();
setInterval(poll, 2000);
setInterval(loadRuns, 15000);
</script>
"""


@bp.get("/")
def page():
    return render_template_string(_PAGE)


def register(app) -> None:
    app.register_blueprint(bp)
