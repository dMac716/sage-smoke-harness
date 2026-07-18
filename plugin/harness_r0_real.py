#!/usr/bin/env python3
"""R0-REAL — harness VIABILITY proof: the R0 capture spine wrapped around the
REAL Smoke Spotter cascade over REAL regime frames, entirely off-Sage.

Where ``harness_r0.py`` proved the capture spine with a stub detector and
synthetic frames, this runs the exact shipped pipeline (``app.main.build_detector``
+ ``app.main.process_frame`` — TiledDetector + rolling baseline, same code path
as the node) over real single-camera TIME SERIES from the observatory event
archive (``events/evt:*/frames/``). Feeding the temporal cascade a jumble of
unrelated stills would violate its semantics — every frame would look
"changed" — so the unit of work here is an EVENT (one camera, chronological
frames), not a bag of images.

Scope honesty: this proves the harness can orchestrate the real detector on
real frames and capture everything crash-safe. It is NOT a detector-quality
claim (no labels are scored here), and the paired BigML baseline/champion
evaluation is a separate, later integration.

Run (from repo root, venv python, mirrored to local ndjson):
  PYWAGGLE_LOG_DIR=/tmp/r0real .venv/bin/python plugin/harness_r0_real.py \
      --max-events 2 --min-frames 6
Gate: data.ndjson holds header/exit, per-frame verdicts + latency, console log,
resource samples, and a traceback for any frame that fails.
"""
import argparse
import json
import logging
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

from waggle.plugin import Plugin

from harness_sink import LiveSink

from app import dev_run as dev_mod
from app import main as main_mod
from app import smoke as smoke_mod
from app import vlm as vlm_mod

RUN_ID = os.environ.get("RUN_ID", f"r0-real-{int(time.time())}")
DEFAULT_EVENTS_ROOT = "/Volumes/Transcend/Observatory/alertwest/events"
# dual-publish: set HARNESS_RECEIVER (e.g. http://<your-receiver-host>) to mirror every
# measurement to the droplet live feed + enable the soft-kill flag. Empty = off.
RECEIVER = os.environ.get("HARNESS_RECEIVER", "").strip()


class PublishLogHandler(logging.Handler):
    """Capture-all: every log record ships as a measurement (crash-safe on Sage,
    data.ndjson locally)."""
    def __init__(self, pub):
        super().__init__(level=logging.DEBUG)
        self.pub = pub

    def emit(self, record):
        try:
            self.pub("log.console", self.format(record),
                     meta={"level": record.levelname, "run_id": RUN_ID})
        except Exception:
            pass  # observability must never crash the run


def resource_sampler(pub, stop_evt, period=2.0):
    n = 0
    while not stop_evt.is_set():
        ru = resource.getrusage(resource.RUSAGE_SELF)
        pub("sys.harness.maxrss_kb", ru.ru_maxrss // 1024,
            meta={"run_id": RUN_ID})
        pub("sys.harness.heartbeat", n, meta={"run_id": RUN_ID})
        n += 1
        stop_evt.wait(period)


def pick_events(root: Path, min_frames: int, max_events: int, explicit):
    """Explicit event ids win; otherwise the first N events with a big-enough
    frame series (sorted for determinism)."""
    if explicit:
        return [root / e for e in explicit]
    picked = []
    for evt in sorted(root.iterdir()):
        frames = evt / "frames"
        if not frames.is_dir():
            continue
        n = sum(1 for p in frames.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if n >= min_frames:
            picked.append(evt)
        if len(picked) >= max_events:
            break
    return picked


def event_camera(evt_dir: Path) -> str:
    try:
        meta = json.loads((evt_dir / "metadata.json").read_text())
        return str(meta.get("camera_id", "unknown"))
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events-root", default=DEFAULT_EVENTS_ROOT)
    ap.add_argument("--event", action="append", default=[],
                    help="explicit event dir name (repeatable), e.g. evt:0123c2aa7f2a47eb")
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--max-events", type=int, default=2)
    ap.add_argument("--frame-budget-s", type=float, default=10.0,
                    help="per-frame wall-clock watchdog budget")
    ap.add_argument("--limit-frames", type=int, default=0,
                    help="stop after N frames total (R3 one-frame gate: 1)")
    ap.add_argument("--frame-delay-s", type=float, default=0.0,
                    help="sleep between frames (simulate node capture cadence; "
                         "also makes the live feed / KILL humanly observable)")
    ap.add_argument("--vlm", default="none",
                    help="VLM backend (none|ollama) — none keeps R0 fully offline")
    ap.add_argument("--fatal", action="store_true",
                    help="with --inject-fault: RE-RAISE after publishing the "
                         "traceback — uncaught crash kills the pod; the "
                         "in-block publish is flushed by Plugin.__exit__ "
                         "(Baseline-1 pattern, proven off-node)")
    ap.add_argument("--inject-fault", action="store_true",
                    help="raise on the 3rd frame of the first event to re-prove "
                         "traceback capture (OFF by default for a genuine run)")
    args = ap.parse_args()

    log = logging.getLogger("harness.r0real")
    log.setLevel(logging.DEBUG)

    root = Path(args.events_root)
    events = pick_events(root, args.min_frames, args.max_events, args.event)
    if not events:
        print(f"no usable events under {root}", file=sys.stderr)
        return 2

    deps = smoke_mod.deps_available()
    backend = vlm_mod.build_backend(args.vlm, timeout=30)
    cfg = main_mod.SmokeConfig(camera="file://harness-r0-real", vlm_kind=args.vlm)

    sink = LiveSink(RECEIVER, RUN_ID) if RECEIVER else None

    with Plugin() as plugin:
        def pub(name, value, meta=None):
            plugin.publish(name, value, meta=meta)   # crash-safe insurance path
            if sink:
                sink.publish(name, value, meta)      # live path (best-effort)

        log.addHandler(PublishLogHandler(pub))

        pub("run.header", 1, meta={
            "run_id": RUN_ID, "hardware": "local-dev", "phase": "A-capture-all",
            "detector": "app.main tiled cascade", "vlm": args.vlm,
            "numpy": str(deps.get("numpy")), "n_events": str(len(events)),
            "events": ",".join(e.name for e in events)})
        log.info(f"R0-REAL {RUN_ID}: {len(events)} event series, vlm={args.vlm}")

        stop = threading.Event()
        sampler = threading.Thread(target=resource_sampler, args=(pub, stop),
                                   daemon=True)
        sampler.start()

        ok = err = alerts = escalated = total = overruns = 0
        killed = False

        def kill_requested() -> bool:
            return bool(sink and sink.kill_event.is_set())

        try:
            for ei, evt in enumerate(events):
                if killed or (args.limit_frames and total >= args.limit_frames):
                    break
                camera = event_camera(evt)
                # fresh detector per event — rolling baseline must not leak
                # across cameras/series
                detector = main_mod.build_detector(cfg)
                frames = list(dev_mod.iter_frames_dir(str(evt / "frames")))
                pub("event.header", len(frames), meta={
                    "run_id": RUN_ID, "event": evt.name, "camera": camera})
                log.info(f"{evt.name} ({camera}): {len(frames)} frames")

                for fi, sample in enumerate(frames):
                    if kill_requested():
                        killed = True
                        log.warning("KILL flag received — aborting run cleanly")
                        pub("harness.killed", 1, meta={
                            "run_id": RUN_ID, "event": evt.name,
                            "frame": str(fi)})
                        break
                    fmeta = {"run_id": RUN_ID, "event": evt.name,
                             "camera": camera, "frame": str(fi)}
                    t_frame = time.monotonic()
                    try:
                        if args.inject_fault and ei == 0 and fi == 2:
                            raise ValueError(
                                "injected fault (traceback-capture re-proof)")
                        with plugin.timeit("inference_ns"):
                            v = main_mod.process_frame(
                                sample, detector, backend,
                                smoke_alert=cfg.smoke_alert,
                                vlm_threshold=cfg.vlm_threshold)
                        elapsed = time.monotonic() - t_frame
                        if elapsed > args.frame_budget_s:
                            overruns += 1
                            pub("harness.watchdog.overrun", elapsed,
                                           meta=fmeta)
                        pub("detect.smoke_score", v["smoke_score"], meta=fmeta)
                        pub("detect.diff", v["diff"], meta=fmeta)
                        pub("detect.max_tile_score",
                                       v.get("max_tile_score", 0.0), meta=fmeta)
                        pub("detect.alert", int(v["alert"]), meta=fmeta)
                        pub("harness.frame_verdict",
                                       json.dumps(v, default=str,
                                                  separators=(",", ":")),
                                       meta=fmeta)
                        alerts += int(v["alert"])
                        escalated += int(v["escalated"])
                        ok += 1
                        log.info(
                            f"{evt.name} f{fi:03d} changed={int(v['changed'])} "
                            f"diff={v['diff']:.2f} smoke={v['smoke_score']:.3f} "
                            f"tile={v.get('max_tile_score', 0.0):.3f} "
                            f"{'ALERT' if v['alert'] else ''}")
                    except Exception as e:
                        pub("harness.traceback", traceback.format_exc(),
                                       meta={**fmeta, "error": type(e).__name__,
                                             "fatal": str(int(args.fatal))})
                        log.error(f"{evt.name} f{fi:03d} FAILED: {e}")
                        err += 1
                        if args.fatal:
                            # uncaught from here: the run DIES. run.exit still
                            # publishes from finally, then __exit__ flushes all.
                            raise
                    total += 1
                    if args.limit_frames and total >= args.limit_frames:
                        log.info(f"frame limit {args.limit_frames} reached")
                        killed = killed or False
                        break
                    if args.frame_delay_s > 0:
                        time.sleep(args.frame_delay_s)
        finally:
            stop.set()
            sampler.join(timeout=3)
            status = ("killed" if killed
                      else "clean" if err == 0 else "with-errors")
            pub("run.exit", 1, meta={
                "run_id": RUN_ID, "frames": str(total), "ok": str(ok),
                "err": str(err), "alerts": str(alerts),
                "escalated": str(escalated), "watchdog_overruns": str(overruns),
                "status": status})
            log.info(f"R0-REAL done [{status}]: frames={total} ok={ok} err={err} "
                     f"alerts={alerts} escalated={escalated} overruns={overruns}")
            if sink:
                sink.close()
                log.info(f"live sink: sent={sink.sent} dropped={sink.dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
