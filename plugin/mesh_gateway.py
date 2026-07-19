#!/usr/bin/env python3
"""mesh-gateway — forward Meshtastic mesh traffic to owners via Beehive.

One Meshtastic device on USB at our dorm Sage node hears EVERY participant's
packets on the shared channel (a mesh propagates all traffic), so this gateway
publishes each received packet to Beehive under ``mesh.<kind>`` with sender
metadata + raw payload provenance — participants self-serve their own node's
data downstream (archiver / Pages report / receiver are 100% reuse; see
sage-smoke-harness docs/meshtastic-forward.md).

Interop (told to participants once): same region (US915), same channel name +
PSK, compatible firmware. The channel name and PSK come from env/args ONLY
(``MESH_CHANNEL_NAME`` / ``MESH_CHANNEL_PSK``) — they are captured at camp when
the first unit is flashed and are NEVER hardcoded or published/logged (only a
``psk_set`` boolean ships in run.header). Good-citizen: RECEIVE + forward only.

The ``meshtastic`` pip lib is OPTIONAL (import-guarded, like pywaggle):
  --source serial   real mode — packet callbacks from a serial-attached node
  --source cmd      any subprocess emitting one JSON packet dict per line
  --source fake     JSON-lines packet file through the SAME handler (dev/tests)

Run (dry-run degrades to stdout JSON lines when pywaggle is absent):
  python3 plugin/mesh_gateway.py --source fake --packets pkts.jsonl \
      --duration-s 30 --dry-run
  MESH_CHANNEL_NAME=camp MESH_CHANNEL_PSK=... python3 plugin/mesh_gateway.py \
      --source serial --duration-s 3600

Crash-safety (Baseline 1 / lesson 0111): any traceback is published INSIDE the
with-Plugin block so it flushes off-node via __exit__. --duration-s is a hard
watchdog bound with clean subprocess teardown on expiry. SIGTERM/SIGINT (pod
stop) flush ``run.exit status=killed`` and tear the source down cleanly.
Config-gate failures DECLINE visibly (run.header + mesh.gateway.config_error,
exit 0) instead of crash-looping invisibly, and repeated fast failures trip a
crash-loop breaker that parks one cycle before retrying.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import json
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

try:  # pywaggle is OPTIONAL — absent/failing -> dry-run degrade (plugin=None)
    from waggle.plugin import Plugin  # type: ignore
except ImportError:
    Plugin = None


def _on_node() -> bool:
    """True only when the Waggle node runtime is clearly present.

    Signals: WAGGLE_APP_ID or any WAGGLE_PLUGIN_* env (injected by the node
    scheduler), or PYWAGGLE_LOG_DIR (pywaggle's explicit file-mirror mode).
    Importability of pywaggle is NOT a signal — off-node it queues publishes
    to an unreachable broker and drops them silently at exit.
    """
    if os.environ.get("WAGGLE_APP_ID") or os.environ.get("PYWAGGLE_LOG_DIR"):
        return True
    return any(k.startswith("WAGGLE_PLUGIN_") for k in os.environ)

try:  # meshtastic pip lib is OPTIONAL — only --source serial needs it
    import meshtastic.serial_interface as _mesh_serial  # type: ignore
    from pubsub import pub as _pubsub  # type: ignore  (pypubsub, meshtastic dep)
except ImportError:
    _mesh_serial = None
    _pubsub = None

log = logging.getLogger("mesh.gateway")

# portnum -> the "mesh.<kind>" measurement family (design doc naming)
PORTNUM_NAMES = {
    "TEXT_MESSAGE_APP": "msg",
    "TELEMETRY_APP": "telemetry",
    "POSITION_APP": "position",
    "NODEINFO_APP": "nodeinfo",
}
RAW_META_MAX = 1024  # bound the raw-provenance blob per packet
STDERR_TAIL_LINES = 12   # bounded stderr tail kept from a dying cmd source
STDERR_LINE_MAX = 300
STDERR_TAIL_MAX = 2000

# crash-loop breaker (lesson 0111: permanent faults must not crash-loop)
FAILURE_STATUSES = frozenset({"crashed", "source-died", "serial-unavailable"})
FAST_FAIL_S = 30.0    # a failing run shorter than this counts as a fast failure
MAX_FAST_FAILS = 5    # consecutive fast failures before declining one cycle


def _state_path() -> str:
    return os.environ.get(
        "MESH_GW_STATE",
        os.path.join(tempfile.gettempdir(), "mesh_gateway_state.json"))


def _load_fast_fails(path: str) -> int:
    with contextlib.suppress(Exception):
        with open(path, encoding="utf-8") as fh:
            return int(json.load(fh).get("fast_fails", 0))
    return 0


def _store_fast_fails(path: str, n: int) -> None:
    with contextlib.suppress(Exception):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"fast_fails": n, "ts": time.time()}, fh)


def _install_signal_handlers(stop: threading.Event) -> dict:
    """SIGTERM (pod stop) / SIGINT -> set the stop flag so the run winds down
    cleanly and run.exit status=killed flushes INSIDE the with-Plugin block."""
    prev = {}

    def _on_signal(signum, frame):  # noqa: ARG001 — signal handler signature
        log.info("signal %s received — stopping cleanly", signum)
        stop.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(ValueError):  # not the main thread (tests)
            prev[sig] = signal.signal(sig, _on_signal)
    return prev


def _restore_signal_handlers(prev: dict) -> None:
    for sig, handler in prev.items():
        with contextlib.suppress(Exception):
            signal.signal(sig, handler)


def _bounded_raw(packet, limit: int = RAW_META_MAX) -> tuple:
    """Raw provenance as ALWAYS-parseable JSON — never truncate mid-JSON.
    Oversize packets get their string/bytes payload leaves shrunk until the
    serialized form fits. Returns (json_str, truncated_bool)."""
    blob = json.dumps(packet, default=str, separators=(",", ":"))
    if len(blob) <= limit:
        return blob, False

    def _shrink(obj, cap):
        if isinstance(obj, dict):
            return {k: _shrink(v, cap) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_shrink(v, cap) for v in obj]
        if isinstance(obj, (bytes, bytearray)):
            obj = obj.decode("utf-8", "replace")
        elif not isinstance(obj, str):
            obj = obj if isinstance(obj, (int, float, bool, type(None))) else str(obj)
        if isinstance(obj, str) and len(obj) > cap:
            return obj[:cap] + "..."
        return obj

    for cap in (256, 64, 16, 4):
        blob = json.dumps(_shrink(packet, cap), default=str,
                          separators=(",", ":"))
        if len(blob) <= limit:
            return blob, True
    # pathological breadth (thousands of keys): keep a parseable envelope
    return json.dumps({"raw_dropped": "structure-too-large"}), True


def _node_id(val) -> str:
    """Meshtastic node id as the canonical '!hex8' string ('' if unusable)."""
    if isinstance(val, str) and val:
        return val
    if isinstance(val, int):
        return f"!{val & 0xFFFFFFFF:08x}"
    return ""


class MeshGateway:
    """Stateful packet handler: extract -> publish, count per sender, and
    survive ANY malformed packet (log + count, never crash the gateway)."""

    def __init__(self, pub, run_id: str):
        self.pub = pub
        self.run_id = run_id
        self.t0 = time.monotonic()
        self.per_sender: dict[str, int] = {}
        self.published = 0
        self.malformed = 0
        self.malformed_reasons: dict[str, int] = {}

    # ── packet path ─────────────────────────────────────────────────────────
    def handle_packet(self, packet) -> bool:
        """Forward one mesh packet dict. Returns True iff published."""
        try:
            return self._handle(packet)
        except Exception as exc:  # noqa: BLE001 — gateway must outlive any packet
            self.note_malformed(f"handler-{type(exc).__name__}")
            log.warning("packet handler error (%s) — packet dropped",
                        type(exc).__name__)
            return False

    def _handle(self, packet) -> bool:
        if not isinstance(packet, dict):
            self.note_malformed("not-a-dict")
            return False
        from_id = _node_id(packet.get("fromId") or packet.get("from"))
        if not from_id:
            self.note_malformed("no-sender")
            return False
        decoded = packet.get("decoded")
        if not isinstance(decoded, dict):
            # encrypted-for-others / undecodable frames arrive without a usable
            # decoded block — count, don't forward, don't crash
            self.note_malformed("no-decoded")
            return False

        portnum = str(decoded.get("portnum", "UNKNOWN"))
        kind = PORTNUM_NAMES.get(portnum, "other")
        value = self._extract_value(kind, decoded)
        # raw payload provenance (bounded but ALWAYS parseable JSON — payload
        # fields are shrunk before serializing, never truncated mid-JSON)
        raw, truncated = _bounded_raw(packet)
        meta = {
            "from_id": from_id,
            "to_id": _node_id(packet.get("toId") or packet.get("to")) or "^all",
            "channel": str(packet.get("channel", 0)),
            "snr": str(packet.get("rxSnr", "")),
            "rssi": str(packet.get("rxRssi", "")),
            "portnum": portnum,
            "raw": raw,
            "raw_truncated": "1" if truncated else "0",
        }
        self.pub(f"mesh.{kind}", value, meta=meta)
        self.per_sender[from_id] = self.per_sender.get(from_id, 0) + 1
        self.published += 1
        return True

    @staticmethod
    def _extract_value(kind: str, decoded: dict) -> str:
        if kind == "msg":
            text = decoded.get("text")
            if text is None:
                payload = decoded.get("payload", b"")
                text = (payload.decode("utf-8", "replace")
                        if isinstance(payload, (bytes, bytearray))
                        else str(payload))
            return str(text)
        if kind == "telemetry":
            return json.dumps(decoded.get("telemetry") or {}, default=str,
                              separators=(",", ":"))
        if kind == "position":
            return json.dumps(decoded.get("position") or {}, default=str,
                              separators=(",", ":"))
        return json.dumps(decoded, default=str, separators=(",", ":"))

    def note_malformed(self, reason: str) -> None:
        self.malformed += 1
        self.malformed_reasons[reason] = self.malformed_reasons.get(reason, 0) + 1

    # ── periodic gateway stats ──────────────────────────────────────────────
    def publish_stats(self) -> None:
        self.pub("mesh.gateway.stats", json.dumps({
            "per_sender": self.per_sender,
            "published": self.published,
            "malformed": self.malformed,
            "malformed_reasons": self.malformed_reasons,
            "uptime_s": round(time.monotonic() - self.t0, 1),
        }, separators=(",", ":")), meta={"n_senders": str(len(self.per_sender))})

    def summary(self) -> dict:
        return {"published": str(self.published),
                "malformed": str(self.malformed),
                "senders": str(len(self.per_sender))}


# ── packet sources (all feed the SAME handler) ──────────────────────────────

def _pump_lines(stream, gw: MeshGateway) -> None:
    """One JSON packet dict per line -> handler. '' (EOF) ends the pump."""
    for line in iter(stream.readline, ""):
        line = line.strip()
        if not line:
            continue
        try:
            pkt = json.loads(line)
        except json.JSONDecodeError:
            gw.note_malformed("bad-json-line")
            continue
        gw.handle_packet(pkt)


def run_cmd_source(gw: MeshGateway, argv: list, deadline: float,
                   popen=subprocess.Popen, poll_s: float = 0.2,
                   term_grace_s: float = 3.0,
                   stop: threading.Event | None = None) -> dict:
    """Subprocess emitting JSON packet lines on stdout. Handles the process
    dying (finding, not crash — with a bounded stderr tail in the report) and
    tears it down cleanly on watchdog expiry or stop signal."""
    proc = popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                 text=True, bufsize=1)
    pump = threading.Thread(target=_pump_lines, args=(proc.stdout, gw),
                            daemon=True)
    pump.start()
    err_tail: collections.deque = collections.deque(maxlen=STDERR_TAIL_LINES)
    err_pump = None
    err_stream = getattr(proc, "stderr", None)
    if err_stream is not None:
        def _pump_err():
            for line in iter(err_stream.readline, ""):
                err_tail.append(line.rstrip("\n")[:STDERR_LINE_MAX])
        err_pump = threading.Thread(target=_pump_err, daemon=True)
        err_pump.start()
    try:
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                return {"status": "killed"}
            rc = proc.poll()
            if rc is not None:
                pump.join(timeout=1.0)  # drain whatever it wrote before dying
                if err_pump is not None:
                    err_pump.join(timeout=1.0)
                log.warning("cmd source exited rc=%s", rc)
                return {"status": "source-died", "returncode": rc,
                        "stderr_tail": "\n".join(err_tail)[-STDERR_TAIL_MAX:]}
            time.sleep(poll_s)
        return {"status": "watchdog"}
    finally:
        if proc.poll() is None:  # watchdog/stop/exception path: clean teardown
            proc.terminate()
            try:
                proc.wait(timeout=term_grace_s)
            except Exception:
                proc.kill()
                with contextlib.suppress(Exception):
                    proc.wait(timeout=term_grace_s)  # reap — no zombie child
        pump.join(timeout=1.0)
        if err_pump is not None:
            err_pump.join(timeout=1.0)


def run_serial_source(gw: MeshGateway, serial_port: str, deadline: float,
                      stop: threading.Event | None = None) -> dict:
    """Real mode: meshtastic lib packet callbacks from a serial-attached node.
    Availability is checked in main() before we get here."""
    try:
        iface = _mesh_serial.SerialInterface(devPath=serial_port or None)
    except SystemExit as exc:
        # meshtastic our_exit() sys.exit()s when no device is attached —
        # convert to a finding so run.exit still flushes (no silent death)
        log.error("meshtastic SerialInterface unavailable (SystemExit: %s)",
                  exc.code)
        return {"status": "serial-unavailable",
                "returncode": exc.code if exc.code is not None else ""}

    def on_receive(packet, interface=None):  # pubsub callback signature
        gw.handle_packet(packet)

    _pubsub.subscribe(on_receive, "meshtastic.receive")
    try:
        while time.monotonic() < deadline:
            if stop is not None and stop.is_set():
                return {"status": "killed"}
            if stop is not None:
                stop.wait(0.5)
            else:
                time.sleep(0.5)
        return {"status": "watchdog"}
    finally:
        with contextlib.suppress(Exception):
            _pubsub.unsubscribe(on_receive, "meshtastic.receive")
        with contextlib.suppress(Exception):
            iface.close()


def run_fake_source(gw: MeshGateway, packets_path: str, deadline: float,
                    stop: threading.Event | None = None) -> dict:
    """Test/dev mode: JSON-lines packet file through the same handler."""
    with open(packets_path, encoding="utf-8") as fh:
        for line in fh:
            if stop is not None and stop.is_set():
                return {"status": "killed"}
            if time.monotonic() >= deadline:
                return {"status": "watchdog"}
            line = line.strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except json.JSONDecodeError:
                gw.note_malformed("bad-json-line")
                continue
            gw.handle_packet(pkt)
    return {"status": "eof"}


# ── entrypoint ──────────────────────────────────────────────────────────────

def _stats_loop(gw: MeshGateway, stop: threading.Event, period_s: float) -> None:
    while not stop.wait(period_s):
        try:
            gw.publish_stats()
        except Exception:  # noqa: BLE001 — observability must never crash the run
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", choices=("serial", "cmd", "fake"), default="serial")
    ap.add_argument("--serial-port", default=os.environ.get("MESH_SERIAL_PORT", ""),
                    help="serial device path ('' = meshtastic lib auto-detect)")
    ap.add_argument("--cmd", default="",
                    help="--source cmd: command emitting one JSON packet per line")
    ap.add_argument("--packets", default="",
                    help="--source fake: JSON-lines packet file")
    ap.add_argument("--channel", default=None,
                    help="mesh channel NAME (default: $MESH_CHANNEL_NAME; "
                         "captured at camp — never hardcoded)")
    ap.add_argument("--psk", default=None,
                    help="mesh channel PSK (default: $MESH_CHANNEL_PSK; "
                         "never published or logged)")
    ap.add_argument("--region", default=os.environ.get("MESH_REGION", "US915"))
    ap.add_argument("--duration-s", type=float, default=300.0,
                    help="hard watchdog bound on the whole run")
    ap.add_argument("--stats-period-s", type=float, default=60.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print measurements as JSON lines (no pywaggle)")
    ap.add_argument("--publish", action="store_true",
                    help="force pywaggle publish even without node-runtime env")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_id = os.environ.get("RUN_ID", f"mesh-gw-{int(time.time())}")
    channel = (args.channel if args.channel is not None
               else os.environ.get("MESH_CHANNEL_NAME", ""))
    psk = (args.psk if args.psk is not None
           else os.environ.get("MESH_CHANNEL_PSK", ""))

    # config gates — evaluated NOW, but DECLINED inside the Plugin block so
    # the failure is visible off-node (lesson 0111: publish-nothing-then-
    # crash-loop is invisible; a clean declining run is not)
    config_missing, config_msg = "", ""
    if args.source == "serial":
        if not channel or not psk:
            config_missing = "channel/psk"
            config_msg = ("serial mode needs the mesh channel name + PSK "
                          "(--channel/--psk or MESH_CHANNEL_NAME/"
                          "MESH_CHANNEL_PSK) — captured at camp when flashing "
                          "the first unit; never hardcoded.")
        elif _mesh_serial is None or _pubsub is None:
            config_missing = "meshtastic-lib"
            config_msg = ("the 'meshtastic' pip lib is not installed "
                          "(pip install meshtastic) — required for "
                          "--source serial.")
    elif args.source == "cmd" and not args.cmd:
        config_missing = "--cmd"
        config_msg = "--source cmd requires --cmd"
    elif args.source == "fake" and not args.packets:
        config_missing = "--packets"
        config_msg = "--source fake requires --packets"

    # Publish mode is EXPLICIT (same contract as lora_capture): pywaggle only
    # when --publish forces it or the node runtime env is clearly present.
    # Off-node, Plugin() constructs fine against an unreachable broker, queues
    # every publish, then silently drops the queue at __exit__ with rc=0 — a
    # telemetry black-hole. Importability alone must never select pywaggle.
    plugin_cm = contextlib.nullcontext()
    if (Plugin is not None and not args.dry_run
            and (args.publish or _on_node())):
        try:
            plugin_cm = Plugin()
        except Exception:  # misconfigured runtime -> dry-run degrade
            log.warning("Plugin() init failed — degrading to dry-run")
            plugin_cm = contextlib.nullcontext()
    elif Plugin is not None and not args.dry_run:
        log.info("pywaggle importable but no node-runtime env and no "
                 "--publish — running dry-run (no silent broker queue)")

    state_path = _state_path()
    fast_fails = _load_fast_fails(state_path)
    stop = threading.Event()
    prev_handlers = _install_signal_handlers(stop)  # SIGTERM/SIGINT -> stop

    rc = 0
    t_start = time.monotonic()
    try:
        with plugin_cm as plugin:  # nullcontext yields None (dry-run)
            def pub(name, value, meta=None):
                m = {k: str(v) for k, v in (meta or {}).items()}
                m["run_id"] = run_id
                if plugin is not None:
                    plugin.publish(name, value, meta=m)
                else:
                    print(json.dumps({"name": name, "value": value, "meta": m},
                                     default=str), flush=True)

            gw = MeshGateway(pub, run_id)
            pub("run.header", 1, meta={
                "mode": "mesh-gateway", "source": args.source,
                "region": args.region, "channel": channel,
                "psk_set": str(bool(psk)).lower(),  # NEVER the PSK itself
                "duration_s": args.duration_s,
                "dry_run": str(plugin is None).lower()})
            log.info("mesh-gateway %s: source=%s region=%s duration=%.0fs",
                     run_id, args.source, args.region, args.duration_s)

            if config_missing:
                # clean, VISIBLE, non-looping decline — not a crash-loop
                pub("mesh.gateway.config_error", config_msg,
                    meta={"missing": config_missing})
                pub("run.exit", 1, meta={"status": "config-error",
                                         "missing": config_missing,
                                         **gw.summary()})
                print(f"mesh-gateway: {config_msg}", file=sys.stderr)
                log.error("config error (%s) — declining cleanly",
                          config_missing)
                return 0

            deadline = time.monotonic() + args.duration_s

            if fast_fails >= MAX_FAST_FAILS:
                # crash-loop breaker: N consecutive fast failures — decline
                # this cycle (parked, interruptible), then reset and retry
                pub("mesh.gateway.crashloop",
                    f"{fast_fails} consecutive fast failures — parking one "
                    f"cycle before retrying",
                    meta={"consecutive_fast_fails": fast_fails})
                log.error("crash-loop breaker tripped (%d fast failures) — "
                          "parking %.0fs", fast_fails, args.duration_s)
                _store_fast_fails(state_path, 0)
                stop.wait(max(0.0, deadline - time.monotonic()))
                pub("run.exit", 1, meta={"status": "crashloop-declined",
                                         **gw.summary()})
                return 0

            # last-resort hard stop if a source wedges past watchdog + grace
            failsafe = threading.Timer(args.duration_s + 60.0, os._exit,
                                       args=(3,))
            failsafe.daemon = True
            failsafe.start()
            stats_t = threading.Thread(target=_stats_loop,
                                       args=(gw, stop, args.stats_period_s),
                                       daemon=True)
            stats_t.start()

            result = {"status": "crashed"}
            try:
                if args.source == "serial":
                    result = run_serial_source(gw, args.serial_port, deadline,
                                               stop=stop)
                elif args.source == "cmd":
                    result = run_cmd_source(gw, shlex.split(args.cmd),
                                            deadline, stop=stop)
                else:
                    result = run_fake_source(gw, args.packets, deadline,
                                             stop=stop)
            except (Exception, SystemExit) as exc:  # noqa: BLE001
                # crash-safety (lesson 0111): publish the traceback INSIDE the
                # with-Plugin block so it flushes off-node via __exit__
                pub("mesh.gateway.traceback", traceback.format_exc(),
                    meta={"error": type(exc).__name__})
                log.error("mesh-gateway crashed: %s", exc)
                rc = 1
            finally:
                stop.set()
                stats_t.join(timeout=1.0)
                failsafe.cancel()
                with contextlib.suppress(Exception):
                    gw.publish_stats()  # final stats snapshot, best-effort
                status = "crashed" if rc else result.get("status", "unknown")
                # crash-loop accounting: consecutive fast failures accumulate;
                # any clean end (eof/watchdog/killed) resets the counter
                if (status in FAILURE_STATUSES
                        and time.monotonic() - t_start < FAST_FAIL_S):
                    fast_fails += 1
                else:
                    fast_fails = 0
                _store_fast_fails(state_path, fast_fails)
                extra = {k: str(result[k])
                         for k in ("returncode", "stderr_tail") if k in result}
                pub("run.exit", 1, meta={"status": status, **extra,
                                         **gw.summary()})
                log.info("mesh-gateway done [%s]: published=%d malformed=%d "
                         "senders=%d", status, gw.published, gw.malformed,
                         len(gw.per_sender))
    finally:
        _restore_signal_handlers(prev_handlers)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
