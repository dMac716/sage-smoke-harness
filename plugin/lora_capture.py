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
radio fields like rssi/snr/time are excluded from the identity). The window
rearms on last PUBLISH, not last sighting, so a static periodic beacon still
comes home once per window instead of being suppressed forever.

Publish mode is EXPLICIT (never inferred from importability alone): pywaggle
publish is used only when we are clearly on-node — WAGGLE_APP_ID or any
WAGGLE_PLUGIN_* env present (the node runtime always injects these), or
PYWAGGLE_LOG_DIR set (pywaggle's file-mirror dev mode), or --publish forces
it. Anywhere else — even with pywaggle importable — the run is dry-run:
every publish prints as one ndjson line on stdout. Off-node, Plugin()
constructs happily against an unreachable broker and silently DROPS the whole
queue at __exit__ with rc=0; importability is not deployment.

Operator stop (SIGTERM/SIGINT) is handled: the decoder's process group is
killed, run.exit publishes with status=killed + counters INSIDE the
with-Plugin block, and the process exits 0. The decoder's stderr is tailed
(last 2 KB) and attached to run.exit when the decoder dies, so missing-dongle
diagnostics come home; a decoder that dies <5 s after launch twice in a row
ends the run with status=decoder-fault and rc 0 (no crash-loop spam on a
permanent fault — lesson 0111).

Usage (node):
  python3 plugin/lora_capture.py --duration-s 3600
Usage (dev, no SDR — canned packets through the identical path):
  python3 plugin/lora_capture.py --rtl433-cmd "cat canned.jsonl" --duration-s 5
"""
from __future__ import annotations

import argparse
import collections
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

# Bounded decoder-stderr tail attached to run.exit when the decoder dies.
STDERR_TAIL_CHARS = 2048
# Crash-loop guard: a decoder death sooner than this after launch is "fast"...
FAST_EXIT_S = 5.0
# ...and this many consecutive fast deaths = permanent fault -> stop, rc 0.
FAST_EXIT_MAX = 2

DEFAULT_RTL433_CMD = "rtl_433 -F json"
# Fields that identify/annotate rather than measure — meta, not measurements.
META_FIELDS = {"time", "model", "id", "channel", "mic", "mod", "type",
               "message_type", "subtype"}
# Radio-conditions fields that vary between the 2-3 repeats of one burst;
# excluded from the dedupe identity (but still published as measurements).
VOLATILE_FIELDS = {"time", "rssi", "snr", "noise", "freq", "freq1", "freq2"}


def slug(s: str) -> str:
    """Lowercase [a-z0-9_] ontology token: 'Acurite-5n1' -> 'acurite_5n1'.

    ASCII-only on purpose: pywaggle rejects publish names outside
    [a-z0-9._]; Unicode isalnum() (e.g. 'température') would crash publish.
    """
    out = "".join(c if (c.isascii() and c.isalnum()) else "_"
                  for c in str(s).strip().lower())
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
    """Drop identical (model,id,payload) packets seen within window_s.

    The window rearms on the last PUBLISH, not the last sighting: a static
    1 Hz beacon (identical payload forever) must still publish once every
    window_s. Rearming on sighting would push the timestamp forward on every
    dropped repeat and suppress it for the life of the run.
    """

    def __init__(self, window_s: float = 2.0, max_keys: int = 4096):
        self.window_s = window_s
        self.max_keys = max_keys
        self._last = {}  # identity -> monotonic ts of last PUBLISHED copy

    @staticmethod
    def identity(pkt: dict) -> str:
        stable = {k: v for k, v in pkt.items() if k not in VOLATILE_FIELDS}
        return json.dumps(stable, sort_keys=True, separators=(",", ":"),
                          default=str)

    def is_dupe(self, pkt: dict, now: float) -> bool:
        key = self.identity(pkt)
        last = self._last.get(key)
        dupe = last is not None and (now - last) < self.window_s
        if not dupe:  # this copy will publish -> rearm the window from it
            self._last[key] = now
        if len(self._last) > self.max_keys:  # bound memory on a long run
            cutoff = now - self.window_s
            self._last = {k: t for k, t in self._last.items() if t >= cutoff}
        return dupe


def _on_node() -> bool:
    """True only when the Waggle node runtime is clearly present.

    Signals: WAGGLE_APP_ID or any WAGGLE_PLUGIN_* env (injected by the node
    scheduler), or PYWAGGLE_LOG_DIR (pywaggle's explicit file-mirror mode,
    where publishes land on disk rather than in a broker queue).
    """
    if os.environ.get("WAGGLE_APP_ID") or os.environ.get("PYWAGGLE_LOG_DIR"):
        return True
    return any(k.startswith("WAGGLE_PLUGIN_") for k in os.environ)


def _plugin_context(force_publish: bool = False):
    """(context manager yielding plugin-or-None, mode string).

    Publish mode is EXPLICIT: pywaggle only when --publish forces it or the
    node runtime env is present (_on_node). Otherwise dry-run — even when
    pywaggle imports fine. Off-node, Plugin() constructs against an
    unreachable broker, queues every publish, then silently drops the whole
    queue at __exit__ with rc=0 — a telemetry black-hole, never acceptable.
    On-node, PYWAGGLE_LOG_DIR file mirroring keeps working: it lives inside
    Plugin itself and this function does not touch it.
    """
    if not (force_publish or _on_node()):
        return contextlib.nullcontext(None), "dry-run"
    try:
        from waggle.plugin import Plugin
    except ImportError:
        if force_publish:
            print("[lora_capture] --publish requested but pywaggle is not "
                  "importable; dry-run", file=sys.stderr)
        return contextlib.nullcontext(None), "dry-run"
    try:
        return Plugin(), "pywaggle"
    except Exception as e:  # misconfigured runtime
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


def _drain_stderr(stream, tail):
    """Reader thread: keep a bounded tail of the decoder's stderr so its
    dying words (missing dongle, USB error) can ride home on run.exit."""
    with contextlib.suppress(Exception):
        for chunk in iter(lambda: stream.read(256), ""):
            tail.extend(chunk)


def capture(args, pub) -> int:
    """The bounded capture loop. Publishes run.header/run.exit itself."""
    counters = {"packets": 0, "published": 0, "deduped": 0, "malformed": 0,
                "publish_errors": 0, "decoder_restarts": 0}
    expired = threading.Event()   # watchdog fired
    stop = threading.Event()      # loop finished -> stop helper threads
    killed = {"sig": None}        # operator SIGTERM/SIGINT received
    procbox = {"proc": None}      # current decoder (swapped across restarts)
    stderr_tail = collections.deque(maxlen=STDERR_TAIL_CHARS)
    exit_extra = {}               # extra run.exit meta (error text)
    status = "eof"                # decoder ended on its own
    rc_forced = None
    rc = 1

    pub("lora.capture.run.header", 1, meta={
        "run_id": RUN_ID, "cmd": args.rtl433_cmd,
        "duration_s": str(args.duration_s),
        "limit_packets": str(args.limit_packets),
        "dedupe_window_s": str(args.dedupe_window_s)})

    def watchdog():
        # Hard time budget: terminating the subprocess also unblocks a
        # reader stuck on a silent radio.
        if not stop.wait(args.duration_s):
            expired.set()
            p = procbox["proc"]
            if p is not None:
                _terminate(p)

    def heartbeat():
        n = 0
        while not stop.wait(args.heartbeat_s):
            pub("lora.capture.heartbeat", n, meta={
                "run_id": RUN_ID,
                **{k: str(v) for k, v in counters.items()}})
            n += 1

    def _on_signal(signum, frame):
        # Operator stop: mark it, TERM the decoder's group — the closed pipe
        # unblocks the reader, and run.exit flushes inside the Plugin block.
        killed["sig"] = signum
        p = procbox["proc"]
        if p is not None:
            with contextlib.suppress(Exception):
                _signal_group(p, signal.SIGTERM)

    prev_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(ValueError, OSError):
                prev_handlers[s] = signal.signal(s, _on_signal)

    threads = [threading.Thread(target=watchdog, daemon=True),
               threading.Thread(target=heartbeat, daemon=True)]
    for t in threads:
        t.start()

    deduper = PacketDeduper(window_s=args.dedupe_window_s)
    fast_exits = 0
    try:
        while True:  # decoder launch/relaunch loop
            if killed["sig"] is not None or expired.is_set():
                break
            started = time.monotonic()
            try:
                proc = subprocess.Popen(
                    shlex.split(args.rtl433_cmd), stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, text=True, bufsize=1,
                    start_new_session=hasattr(os, "setsid"))
            except FileNotFoundError:
                msg = (f"rtl_433 launch failed: "
                       f"{shlex.split(args.rtl433_cmd)[0]!r} not found "
                       f"(install rtl_433, or point --rtl433-cmd at it)")
                print(f"[lora_capture] {msg}", file=sys.stderr)
                status = "rtl433-missing"
                exit_extra["error"] = msg
                rc_forced = 2
                break
            procbox["proc"] = proc
            t_err = threading.Thread(target=_drain_stderr,
                                     args=(proc.stderr, stderr_tail),
                                     daemon=True)
            t_err.start()
            threads.append(t_err)

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
                try:
                    pub("lora.rtl433.raw",
                        json.dumps(pkt, separators=(",", ":"), default=str),
                        meta=meta)
                    for name, value in measurements(pkt):
                        pub(name, value, meta=meta)
                except Exception as e:
                    # One hostile packet (bad name/value for the publisher)
                    # must never kill the run: count it, log it, move on.
                    counters["publish_errors"] += 1
                    print(f"[lora_capture] publish failed for "
                          f"model={meta.get('model')!r}: {e}",
                          file=sys.stderr)
                    continue
                counters["published"] += 1
                if (args.limit_packets > 0
                        and counters["published"] >= args.limit_packets):
                    status = "limit-reached"
                    break

            # Decoder stdout closed (its death) or we broke out ourselves.
            _terminate(proc)
            if (status == "limit-reached" or killed["sig"] is not None
                    or expired.is_set()):
                break
            if proc.poll() == 0:
                break  # clean EOF (canned input finished) -> status "eof"
            # Unexpected nonzero decoder death: crash-loop guard, then
            # relaunch within the same bounded run.
            if time.monotonic() - started < FAST_EXIT_S:
                fast_exits += 1
                if fast_exits >= FAST_EXIT_MAX:
                    # Permanent fault (missing dongle, bad driver): report
                    # via telemetry and stop cleanly — rc 0, no crash-loop
                    # spam for the scheduler to amplify (lesson 0111).
                    status = "decoder-fault"
                    break
            else:
                fast_exits = 0
            counters["decoder_restarts"] += 1
    finally:
        stop.set()
        p = procbox["proc"]
        if p is not None:
            _terminate(p)
        for s, h in prev_handlers.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(s, h)
        for t in threads:
            t.join(timeout=3)
        if expired.is_set():
            status = "watchdog-expired"
        if killed["sig"] is not None:
            status = "killed"
            exit_extra["signal"] = str(killed["sig"])
        proc_rc = p.poll() if p is not None else None
        # A run that captured what it was asked to, ran its full budget, was
        # stopped by the operator, or cleanly reported a permanent decoder
        # fault is a success; only an unexplained decoder death is a failure.
        rc = 0 if status in ("limit-reached", "watchdog-expired", "killed",
                             "decoder-fault") or proc_rc == 0 else 1
        exit_meta = {
            "run_id": RUN_ID, "status": status, "proc_rc": str(proc_rc),
            **exit_extra, **{k: str(v) for k, v in counters.items()}}
        tail = "".join(stderr_tail).strip()
        if tail and status in ("eof", "decoder-fault") and proc_rc != 0:
            # The decoder died: its last words are the diagnostics.
            exit_meta["stderr_tail"] = tail[-STDERR_TAIL_CHARS:]
        pub("lora.capture.run.exit", 1, meta=exit_meta)
    return rc_forced if rc_forced is not None else rc


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
    ap.add_argument("--publish", action="store_true",
                    help="force pywaggle publish mode; without this flag "
                         "pywaggle is used only when the node runtime env "
                         "(WAGGLE_APP_ID / WAGGLE_PLUGIN_* / "
                         "PYWAGGLE_LOG_DIR) is present")
    args = ap.parse_args(argv)

    ctx, mode = _plugin_context(force_publish=args.publish)
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
