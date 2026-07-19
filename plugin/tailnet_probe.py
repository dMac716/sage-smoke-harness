#!/usr/bin/env python3
"""R5a — tailnet-join probe. Proves userspace-Tailscale works INSIDE a pod:
tailscaled (userspace networking, NO TUN, cannot become all-traffic) joins the
tailnet with the ephemeral job-secret key, and the pod reaches OUR droplet over
the tailnet — the real-time dual-publish leg the Beehive-tail fallback stood in
for. launch.sh brings the tunnel up and sets HTTP_PROXY to tailscaled's own
outbound proxy (which resolves MagicDNS), so this script just makes a plain GET.
All results publish crash-safe; the ephemeral node auto-removes on exit.
"""
import os
import time
import urllib.request

from waggle.plugin import Plugin

RUN_ID = os.environ.get("RUN_ID", f"tailnet-probe-{int(time.time())}")
RECEIVER = os.environ.get("HARNESS_RECEIVER",
                          "")  # set HARNESS_RECEIVER — no default (BYO)


def main() -> int:
    with Plugin() as plugin:
        def pub(name, value, meta=None):
            plugin.publish(name, value, meta={"run_id": RUN_ID, **(meta or {})})

        ts_ip = os.environ.get("TS_IP", "")
        pub("tailnet.header", 1, meta={"receiver": RECEIVER})
        pub("tailnet.joined", 1 if ts_ip else 0, meta={"ts_ip": ts_ip or "none"})

        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(f"{RECEIVER}/harness/api/runs",
                                        timeout=25) as r:
                code, body = r.status, r.read()
            pub("tailnet.droplet_ok", 1 if 200 <= code < 400 else 0, meta={
                "code": str(code), "ms": f"{(time.monotonic()-t0)*1e3:.0f}",
                "bytes": str(len(body))})
        except Exception as exc:  # noqa: BLE001 — an unreachable droplet is THE finding
            pub("tailnet.droplet_ok", 0, meta={
                "error": str(exc)[:200],
                "ms": f"{(time.monotonic()-t0)*1e3:.0f}"})
        pub("tailnet.exit", 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
