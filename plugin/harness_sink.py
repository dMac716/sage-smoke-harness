"""LiveSink — the harness's second publish path (dual-publish, plan C1→C2).

``plugin.publish`` (pywaggle → Beehive) stays the crash-safe insurance; this
sink mirrors every measurement to our droplet receiver (portal
``/harness/ingest``) over the tailnet for the live view, and polls
``/harness/control/<run_id>`` for the soft-kill flag.

Design constraints (measurement must not perturb the measured):
- ``publish()`` NEVER blocks and NEVER raises — it enqueues; a background
  thread batches and POSTs. On failure records are dropped and counted
  (insurance path still has them).
- kill polling runs on its own thread; the run loop just checks
  ``sink.kill_event.is_set()`` between frames.

stdlib-only (urllib) so the plugin image needs nothing extra.
"""
from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request


class LiveSink:
    def __init__(self, base_url: str, run_id: str,
                 batch_max: int = 200, flush_s: float = 1.0,
                 control_poll_s: float = 2.0, timeout_s: float = 5.0):
        self.base = base_url.rstrip("/")
        self.run_id = run_id
        self.batch_max = batch_max
        self.flush_s = flush_s
        self.control_poll_s = control_poll_s
        self.timeout_s = timeout_s
        self.kill_event = threading.Event()
        self.dropped = 0
        self.sent = 0
        self._q: "queue.Queue[dict]" = queue.Queue(maxsize=10_000)
        self._stop = threading.Event()
        self._sender = threading.Thread(target=self._send_loop, daemon=True)
        self._control = threading.Thread(target=self._control_loop, daemon=True)
        self._sender.start()
        self._control.start()

    # -- the publish mirror ---------------------------------------------------
    def publish(self, name: str, value, meta: dict | None = None) -> None:
        try:
            self._q.put_nowait({"name": name, "value": value,
                                "meta": meta or {}, "ts": time.time()})
        except queue.Full:
            self.dropped += 1

    # -- internals ------------------------------------------------------------
    def _post(self, path: str, payload: dict | None) -> dict:
        data = json.dumps(payload, default=str).encode() if payload is not None else b""
        req = urllib.request.Request(
            f"{self.base}{path}", data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read() or b"{}")

    def _drain(self) -> list:
        batch = []
        while len(batch) < self.batch_max:
            try:
                batch.append(self._q.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send_loop(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            batch = self._drain()
            if batch:
                try:
                    r = self._post("/harness/ingest",
                                   {"run_id": self.run_id, "records": batch})
                    self.sent += len(batch)
                    if r.get("kill"):
                        self.kill_event.set()
                except Exception:
                    self.dropped += len(batch)
            else:
                self._stop.wait(self.flush_s)

    def _control_loop(self) -> None:
        while not self._stop.is_set():
            try:
                r = self._post(f"/harness/control/{self.run_id}", None)
                if r.get("kill"):
                    self.kill_event.set()
            except Exception:
                pass  # live path is best-effort; hard kill paths are elsewhere
            self._stop.wait(self.control_poll_s)

    def close(self, flush_timeout_s: float = 10.0) -> None:
        deadline = time.time() + flush_timeout_s
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.1)
        self._stop.set()
        self._sender.join(timeout=5)
        self._control.join(timeout=2)
