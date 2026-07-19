#!/usr/bin/env python3
"""lora-capture — RTL-SDR 433-ISM decode -> pywaggle publish (docs/lora-share).

Spawns ``rtl_433 -F json`` as a subprocess and turns every decoded over-the-air
packet (AcuRite weather stations, sensor remotes, TPMS, LoRa-OOK beacons...)
into pywaggle measurements so participants' RF traffic lands on Beehive,
queryable + self-serve via the existing harness tiers. RECEIVE/decode only —
we never transmit. One dongle, no custom RF code.

Ontology per decoded packet:
  lora.rtl433.raw                       compact JSON of the whole packet
                                        (provenance: exactly what rtl_433 said)
  lora.rtl433.<model>.<field>           one measurement per numeric field
                                        (temperature_C, humidity, wind_avg_km_h,
                                        pressure_kPa, battery_ok, rssi, ...)
  meta on every publish: model, device_id, channel (when present), run_id.

Run bookkeeping (same crash-safe shape as the harness spine):
  lora.capture.run.header / lora.capture.run.exit   with counters + status
  lora.capture.heartbeat                            periodic liveness
  lora.capture.traceback                            published INSIDE the
      with-Plugin block before re-raise (lesson 0111: an excepthook after
      Plugin shutdown never flushes — the traceback must ride the live plugin).

Bounds (this is a camp plugin — it must always come home):
  --duration-s     hard wall-clock watchdog; the subprocess is terminated at
                   the deadline even if it is blocking us on a read.
  --limit-packets  stop after N published packets.

Dedupe: 433 MHz devices repeat each burst 2-3x back-to-back; identical
(model, id, payload) packets within --dedupe-window-s are dropped (volatile
radio fields like rssi/snr/time are excluded from the identity).

pywaggle is OPTIONAL: if it is not importable (or Plugin() fails off-node)
the run degrades to dry-run — every publish prints as one ndjson line on
stdout — so the whole path is testable with nothing but Python.

Usage (node):
  python3 plugin/lora_capture.py --duration-s 3600
Usage (dev, no SDR — canned packets through the identical path):
  python3 plugin/lora_capture.py --rtl433-cmd "cat canned.jsonl" --duration-s 5
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback

RUN_ID = os.environ.get("RUN_ID", f"lora-capture-{int(time.time())}")

DEFAULT_RTL433_CMD = "rtl_433 -F json"
# Fields that identify/annotate rather than measure — meta, not measurements.
META_FIELDS = {"time", "model", "id", "channel", "mic", "mod", "type",
               "message_type", "subtype"}
# Radio-conditions fields that vary between the 2-3 repeats of one burst;
# excluded from the dedupe identity (but still published as measurements).
VOLATILE_FIELDS = {"time", "rssi", "snr", "noise", "freq", "freq1", "freq2"}


def slug(s: str) -> str:
    """Lowercase [a-z0-9_] ontology token: 'Acurite-5n1' -> 'acurite_5n1'."""
    out = "".join(c if c.isalnum() else "_" for c in str(s).strip().lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def parse_line(line: str):
    """One rtl_433 JSON line -> dict with a model, else None (skip, no crash)."""
    try:
        pkt = json.loads(line)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(pkt, dict) or not pkt.get("model"):
        return None
    return pkt


def measurements(pkt: dict):
    """Numeric fields -> [(name, value)] under lora.rtl433.<model>.<field>."""
    model = slug(pkt.get("model", "unknown"))
    out = []
    for key, val in pkt.items():
        if key in META_FIELDS:
            continue
        if isinstance(val, bool):
            val = int(val)
        if isinstance(val, (int, float)):
            out.append((f"lora.rtl433.{model}.{slug(key)}", val))
    return out


def packet_meta(pkt: dict) -> dict:
    meta = {"model": str(pkt.get("model", "unknown")),
            "device_id": str(pkt.get("id", "")),
            "run_id": RUN_ID}
    if "channel" in pkt:
        meta["channel"] = str(pkt["channel"])
    return meta


class PacketDeduper:
    """Drop identical (model,id,payload) packets seen within window_s."""

    def __init__(self, window_s: float = 2.0, max_keys: int = 4096):
        self.window_s = window_s
        self.max_keys = max_keys
        self._last = {}  # identity -> last-seen monotonic ts

    @staticmethod
    def identity(pkt: dict) -> str:
        stable = {k: v for k, v in pkt.items() if k not in VOLATILE_FIELDS}
        return json.dumps(stable, sort_keys=True, separators=(",", ":"),
                          default=str)

    def is_dupe(self, pkt: dict, now: float) -> bool:
        key = self.identity(pkt)
        last = self._last.get(key)
        self._last[key] = now
        if len(self._last) > self.max_keys:  # bound memory on a long run
            cutoff = now - self.window_s
            self._last = {k: t for k, t in self._last.items() if t >= cutoff}
        return last is not None and (now - last) < self.window_s


def _plugin_context():
    """(context manager yielding plugin-or-None, mode string).

    pywaggle importable + Plugin() constructs -> live publish; otherwise
    degrade to dry-run (plugin=None), never crash — offline-first contract.
    """
    try:
        from waggle.plugin import Plugin
    except ImportError:
        return contextlib.nullcontext(None), "dry-run"
    try:
        return Plugin(), "pywaggle"
    except Exception as e:  # off-node / misconfigured runtime
        print(f"[lora_capture] Plugin() failed ({e}); dry-run", file=sys.stderr)
        return contextlib.nullcontext(None), "dry-run"


def make_pub(plugin):
    def pub(name, value, meta=None):
        if plugin is not None:
            plugin.publish(name, value, meta=meta or {})
        else:
            print(json.dumps({"name": name, "value": value,
                              "meta": meta or {}},
                             separators=(",", ":"), default=str))
    return pub


def _signal_group(proc, sig):
    """Signal the decoder's whole process group (a shell wrapper's children
    inherit our stdout pipe; leaving them alive would block the reader)."""
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, sig)
            return
    proc.send_signal(sig)


def _terminate(proc):
    """Clean subprocess teardown: TERM the group, brief wait, then KILL."""
    if proc.poll() is None:
        _signal_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _signal_group(proc, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=3)


def capture(args, pub) -> int:
    """The bounded capture loop. Publishes run.header/run.exit itself."""
    counters = {"packets": 0, "published": 0, "deduped": 0, "malformed": 0}
    expired = threading.Event()   # watchdog fired
    stop = threading.Event()      # loop finished -> stop helper threads
    status = "eof"                # subprocess ended on its own
    rc = 1

    pub("lora.capture.run.header", 1, meta={
        "run_id": RUN_ID, "cmd": args.rtl433_cmd,
        "duration_s": str(args.duration_s),
        "limit_packets": str(args.limit_packets),
        "dedupe_window_s": str(args.dedupe_window_s)})

    try:
        proc = subprocess.Popen(
            shlex.split(args.rtl433_cmd), stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            start_new_session=hasattr(os, "setsid"))
    except FileNotFoundError:
        msg = (f"rtl_433 launch failed: {shlex.split(args.rtl433_cmd)[0]!r} "
               f"not found (install rtl_433, or point --rtl433-cmd at it)")
        print(f"[lora_capture] {msg}", file=sys.stderr)
        pub("lora.capture.run.exit", 1, meta={
            "run_id": RUN_ID, "status": "rtl433-missing", "error": msg,
            **{k: str(v) for k, v in counters.items()}})
        return 2

    def watchdog():
        # Hard time budget: terminating the subprocess also unblocks a
        # reader stuck on a silent radio.
        if not stop.wait(args.duration_s):
            expired.set()
            _terminate(proc)

    def heartbeat():
        n = 0
        while not stop.wait(args.heartbeat_s):
            pub("lora.capture.heartbeat", n, meta={
                "run_id": RUN_ID,
                **{k: str(v) for k, v in counters.items()}})
            n += 1

    threads = [threading.Thread(target=watchdog, daemon=True),
               threading.Thread(target=heartbeat, daemon=True)]
    for t in threads:
        t.start()

    deduper = PacketDeduper(window_s=args.dedupe_window_s)
    try:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            pkt = parse_line(line)
            if pkt is None:
                counters["malformed"] += 1
                continue
            counters["packets"] += 1
            if deduper.is_dupe(pkt, time.monotonic()):
                counters["deduped"] += 1
                continue
            meta = packet_meta(pkt)
            pub("lora.rtl433.raw",
                json.dumps(pkt, separators=(",", ":"), default=str),
                meta=meta)
            for name, value in measurements(pkt):
                pub(name, value, meta=meta)
            counters["published"] += 1
            if (args.limit_packets > 0
                    and counters["published"] >= args.limit_packets):
                status = "limit-reached"
                break
    finally:
        stop.set()
        _terminate(proc)
        for t in threads:
            t.join(timeout=3)
        if expired.is_set():
            status = "watchdog-expired"
        proc_rc = proc.poll()
        # A run that captured what it was asked to (or ran its full budget)
        # is a success; only an unasked-for subprocess death is a failure.
        rc = 0 if status in ("limit-reached", "watchdog-expired") \
            or proc_rc == 0 else 1
        pub("lora.capture.run.exit", 1, meta={
            "run_id": RUN_ID, "status": status, "proc_rc": str(proc_rc),
            **{k: str(v) for k, v in counters.items()}})
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rtl433-cmd", default=DEFAULT_RTL433_CMD,
                    help="decoder command emitting JSON lines on stdout "
                         "(tests/dev: 'cat canned.jsonl')")
    ap.add_argument("--duration-s", type=float, default=3600.0,
                    help="hard wall-clock budget (watchdog terminates the "
                         "subprocess at the deadline)")
    ap.add_argument("--limit-packets", type=int, default=0,
                    help="stop after N published packets (0 = no limit)")
    ap.add_argument("--dedupe-window-s", type=float, default=2.0,
                    help="identical-packet suppression window")
    ap.add_argument("--heartbeat-s", type=float, default=10.0,
                    help="liveness heartbeat period")
    args = ap.parse_args(argv)

    ctx, mode = _plugin_context()
    with ctx as plugin:
        pub = make_pub(plugin)
        try:
            print(f"[lora_capture] {RUN_ID} mode={mode} "
                  f"cmd={args.rtl433_cmd!r}", file=sys.stderr)
            return capture(args, pub)
        except Exception:
            # Crash-safety (lesson 0111): the traceback must publish INSIDE
            # the with-Plugin block; an excepthook after shutdown never
            # flushes. Re-raise so it is still a real crash.
            pub("lora.capture.traceback", traceback.format_exc(),
                meta={"run_id": RUN_ID})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
