#!/usr/bin/env python3
"""C4 — the egress boundary probe (rung R2). Answers ONE question from inside
a real node job: can this pod reach the public internet, and how?

Every result is PUBLISHED via pywaggle — the crash-safe Beehive path that
works regardless of egress — so even a fully-sandboxed pod reports its own
sandboxing. Probes (all read-only GETs, tiny, operator-owned or Sage-owned):

  dns        resolve a public hostname
  sage_api   GET https://data.sagecontinuum.org  (Sage's own data API)
  operator   GET $PROBE_URL (default: the operator's public health endpoint)

Publishes: probe.dns_ok, probe.sage_api_ok, probe.operator_ok (+ latency_ms
and error details in meta), probe.exit. No listening ports, no writes, no
retries — one pass, seconds of runtime, then exit.
"""
import os
import urllib.error
import socket
import time
import urllib.request

from waggle.plugin import Plugin

RUN_ID = os.environ.get("RUN_ID", f"egress-probe-{int(time.time())}")
PROBE_URL = os.environ.get("PROBE_URL", "https://tower-watch.com/api/ingest/health")
TIMEOUT_S = float(os.environ.get("PROBE_TIMEOUT_S", "15"))


def probe_dns(host: str = "data.sagecontinuum.org") -> dict:
    t0 = time.monotonic()
    try:
        addr = socket.getaddrinfo(host, 443)[0][4][0]
        return {"ok": 1, "ms": round((time.monotonic() - t0) * 1e3, 1),
                "addr": addr}
    except Exception as exc:
        return {"ok": 0, "ms": round((time.monotonic() - t0) * 1e3, 1),
                "error": str(exc)[:200]}


def probe_get(url: str) -> dict:
    """ANY HTTP response (even 404) proves egress — the server answered.
    Only failure to get a response at all means the path is blocked."""
    t0 = time.monotonic()
    req = urllib.request.Request(url, headers={
        "User-Agent": "sage-smoke-harness-egress-probe/0.0.2"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            code = r.status
            r.read(2048)
        return {"ok": 1, "code": code,
                "ms": round((time.monotonic() - t0) * 1e3, 1)}
    except urllib.error.HTTPError as exc:
        return {"ok": 1, "code": exc.code, "note": "http-error-but-reached",
                "ms": round((time.monotonic() - t0) * 1e3, 1)}
    except Exception as exc:
        return {"ok": 0, "ms": round((time.monotonic() - t0) * 1e3, 1),
                "error": str(exc)[:200]}


def main() -> int:
    with Plugin() as plugin:
        def pub(name, result):
            meta = {"run_id": RUN_ID,
                    **{k: str(v) for k, v in result.items() if k != "ok"}}
            plugin.publish(name, int(result["ok"]), meta=meta)

        plugin.publish("probe.header", 1, meta={
            "run_id": RUN_ID, "operator_url": PROBE_URL})
        pub("probe.dns_ok", probe_dns())
        pub("probe.sage_api_ok", probe_get("https://data.sagecontinuum.org"))
        pub("probe.operator_ok", probe_get(PROBE_URL))
        plugin.publish("probe.exit", 1, meta={"run_id": RUN_ID})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
