#!/usr/bin/env python3
"""Bounded ollama END-PHASE for the template job (Apple post-run
diagnostic-window pattern; plan P4, off-node proof).

After the detection/harness phase completes, the job does NOT die silently:
 (1) probe ollama (on-node: bring it up; here: OLLAMA_URL must answer),
 (2) ANALYZE the run's published results (data.ndjson) into compact stats,
 (3) ask the model for a concise digest and publish it back over the same
     dual path (move-the-work-not-the-data: consumers read a digest, not the
     full stream),
 (4) check the agent-bus INBOX for queued requests: messages -> serve them
     inside the HARD window budget; empty -> relinquish immediately,
 (5) publish window.closing, tear down, exit.

HONESTY GUARDRAIL: the digest and any bus answers are CANDIDATE conveniences,
never the record — the full measurement stream is already archived crash-safe
(Beehive on Sage, ndjson locally, droplet mirror). `candidate != truth`
applies to the edge LLM too; every published digest carries candidate=true.

Good-citizen bounds: ONE wall-clock budget covers the whole window (model
probe, digest, bus serving). Every model call gets min(remaining, cap) as its
timeout; when the budget is spent we close the window no matter what is left.

Local proof:
  PYWAGGLE_LOG_DIR=/tmp/endphase RUN_ID=endphase-001 \
    .venv/bin/python plugin/harness_endphase.py \
      --results /tmp/r0real/data.ndjson --window-budget-s 90 \
      --bus-root ~/.agent-orchestration/bus --serve-inbox Sage_CampProject
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from waggle.plugin import Plugin

from harness_sink import LiveSink

RUN_ID = os.environ.get("RUN_ID", f"endphase-{int(time.time())}")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
RECEIVER = os.environ.get("HARNESS_RECEIVER", "").strip()


class Window:
    """The single hard budget for the whole end-phase."""
    def __init__(self, budget_s: float):
        self.t0 = time.monotonic()
        self.budget_s = budget_s

    def remaining(self) -> float:
        return max(0.0, self.budget_s - (time.monotonic() - self.t0))

    def expired(self) -> bool:
        return self.remaining() <= 0


def ollama_generate(model: str, prompt: str, timeout_s: float,
                    num_predict: int = 320) -> dict:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps({"model": model, "prompt": prompt, "stream": False,
                         "options": {"num_predict": num_predict}}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=max(1.0, timeout_s)) as resp:
        out = json.loads(resp.read())
    out["_wall_s"] = time.monotonic() - t0
    return out


def analyze_results(ndjson_path: Path) -> dict:
    """Compact, model-free stats over the run's published stream — this is the
    payload the model summarizes, and it is small by construction."""
    events: dict = {}
    lat_ms, errors, exit_meta = [], 0, None
    for line in ndjson_path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        name, meta = r.get("name", ""), r.get("meta", {}) or {}
        ev = meta.get("event")
        if name == "detect.smoke_score" and ev:
            e = events.setdefault(ev, {"camera": meta.get("camera", "?"),
                                       "scores": [], "alerts": 0})
            e["scores"].append(float(r["value"]))
        elif name == "detect.alert" and ev and int(r.get("value") or 0):
            events.setdefault(ev, {"camera": meta.get("camera", "?"),
                                   "scores": [], "alerts": 0})["alerts"] += 1
        elif name == "inference_ns":
            lat_ms.append(float(r["value"]) / 1e6)
        elif name == "harness.traceback":
            errors += 1
        elif name == "run.exit":
            exit_meta = meta
    for e in events.values():
        s = e.pop("scores")
        e.update(n_frames=len(s),
                 score_first=round(s[0], 3) if s else None,
                 score_max=round(max(s), 3) if s else None,
                 score_last=round(s[-1], 3) if s else None)
    return {
        "run_exit": exit_meta,
        "events": events,
        "errors": errors,
        "latency_ms_median": round(statistics.median(lat_ms), 1) if lat_ms else None,
        "n_inferences": len(lat_ms),
    }


def read_inbox(bus_root: Path, project: str, limit: int = 5) -> list:
    inbox = bus_root / project / "inbox"
    if not inbox.is_dir():
        return []
    msgs = []
    for p in sorted(inbox.glob("*.json")):
        try:
            m = json.loads(p.read_text())
        except Exception:
            continue
        if m.get("type") in ("request", "question"):
            msgs.append((p, m))
        if len(msgs) >= limit:
            break
    return msgs


def write_reply(bus_root: Path, req: dict, answer: str) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    to = req.get("from", "unknown")
    rid = f"{ts}__from-Sage_CampProject__reply__{os.getpid()}"
    out = bus_root / to / "inbox"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{rid}.json"
    path.write_text(json.dumps({
        "schema": "agent-fleet/message@1",
        "id": rid, "from": "Sage_CampProject", "to": to, "type": "reply",
        "in_reply_to": req.get("id"),
        "subject": f"endphase answer: {req.get('subject', '')[:80]}",
        "body": answer,
        "provenance": {"run_id": RUN_ID, "candidate": True,
                       "note": "edge-LLM answer during bounded end-phase window; "
                               "candidate != truth — verify against archived data"},
    }, indent=1))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="run data.ndjson to digest")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--window-budget-s", type=float, default=120.0)
    ap.add_argument("--bus-root", default=os.path.expanduser("~/.agent-orchestration/bus"))
    ap.add_argument("--serve-inbox", default="",
                    help="project inbox to serve (empty = skip bus phase)")
    ap.add_argument("--max-requests", type=int, default=3)
    args = ap.parse_args()

    win = Window(args.window_budget_s)
    sink = LiveSink(RECEIVER, RUN_ID) if RECEIVER else None

    with Plugin() as plugin:
        def pub(name, value, meta=None):
            m = {"run_id": RUN_ID, **(meta or {})}
            plugin.publish(name, value, meta=m)
            if sink:
                sink.publish(name, value, m)

        served = 0
        digest_ok = False
        try:
            pub("endphase.header", 1, meta={
                "model": args.model, "budget_s": str(args.window_budget_s),
                "results": args.results})

            # (1) ollama up?
            try:
                with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags",
                                            timeout=min(10, win.remaining())) as r:
                    models = [m["name"] for m in json.loads(r.read()).get("models", [])]
                pub("endphase.ollama_up", 1, meta={"n_models": str(len(models))})
            except Exception as exc:
                pub("endphase.ollama_up", 0, meta={"error": str(exc)[:200]})
                return 3  # no model window possible; finally still closes clean

            # (2) analyze (model-free, cheap)
            stats = analyze_results(Path(args.results))
            pub("endphase.stats", json.dumps(stats, separators=(",", ":")))

            # (3) digest — bounded by remaining window
            if not win.expired():
                prompt = (
                    "You are the on-node post-run analyst for a wildfire-smoke "
                    "detection test harness. Summarize this run in <=6 short "
                    "lines for an operator: outcome, per-event score trajectory "
                    "(rising scores across a series suggest developing smoke), "
                    "alerts, errors, latency. Be factual; do not invent data.\n\n"
                    + json.dumps(stats, indent=1))
                try:
                    out = ollama_generate(args.model, prompt,
                                          timeout_s=min(90, win.remaining()))
                    pub("endphase.digest", out.get("response", "").strip(), meta={
                        "candidate": "true", "model": args.model,
                        "wall_s": f"{out['_wall_s']:.1f}",
                        "load_ms": str(int(out.get("load_duration", 0) / 1e6)),
                        "eval_tokens": str(out.get("eval_count", 0))})
                    digest_ok = True
                except Exception as exc:
                    pub("endphase.digest_failed", str(exc)[:200])

            # (4) bus inbox: serve or relinquish
            if args.serve_inbox and not win.expired():
                msgs = read_inbox(Path(args.bus_root), args.serve_inbox,
                                  args.max_requests)
                if not msgs:
                    pub("endphase.bus_empty", 1)
                for path, m in msgs:
                    if win.expired():
                        pub("endphase.bus_window_expired", 1)
                        break
                    q = (f"Question from fleet peer {m.get('from')} — subject: "
                         f"{m.get('subject')}\n\n{m.get('body', '')[:4000]}\n\n"
                         "Answer concisely using ONLY the run stats below; if "
                         "the stats cannot answer it, say so explicitly.\n\n"
                         + json.dumps(stats, indent=1))
                    try:
                        out = ollama_generate(args.model, q,
                                              timeout_s=min(60, win.remaining()))
                        reply = write_reply(Path(args.bus_root), m,
                                            out.get("response", "").strip())
                        served += 1
                        pub("endphase.bus_served", served, meta={
                            "request": m.get("id", "?"), "reply": reply.name,
                            "candidate": "true"})
                    except Exception as exc:
                        pub("endphase.bus_serve_failed", str(exc)[:200],
                            meta={"request": m.get("id", "?")})
        finally:
            # (5) close the window NO MATTER WHAT — signal both edges
            pub("window.closing", 1, meta={
                "elapsed_s": f"{time.monotonic() - win.t0:.1f}",
                "budget_s": str(args.window_budget_s),
                "digest": str(int(digest_ok)), "bus_served": str(served),
                "status": "expired" if win.expired() else "relinquished"})
            if sink:
                sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
