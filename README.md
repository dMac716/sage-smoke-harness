# sage-smoke-harness

A **self-instrumenting test-harness plugin** for [Sage/Waggle](https://sagecontinuum.org)
edge nodes. It runs a wildfire-smoke detection cascade (cheap tiled
change-gate → smoke heuristic → optional VLM escalation) over single-camera
frame series with **total observability**: every log line, per-frame verdict,
inference latency, resource sample, and traceback is published as a
measurement (crash-safe via Beehive), optionally dual-published live to an
operator-owned receiver, with a poll-based soft-kill flag.

This is a *validation harness*, not a production detector claim. It exists to
measure detectors honestly on real edge silicon.

## Components

| File | Role |
|---|---|
| `plugin/harness_r0_real.py` | Capture spine + the shipped cascade over event series |
| `plugin/harness_sink.py` | Never-blocking dual-publish sink + soft-kill poll |
| `plugin/harness_endphase.py` | Bounded post-run window: stats → local-LLM digest → agent-bus serve → relinquish (hard wall-clock budget) |
| `plugin/make_synthetic_events.py` | Build-time synthetic proof series (see data policy) |
| `app/` | The detection cascade (tiled change-gate, smoke heuristic, VLM backends) |

## Design rules

- **Unit of work = one event series** (one camera, chronological frames,
  fresh detector state per series). The cascade is temporal; a bag of
  unrelated stills would be meaningless.
- **Dual-publish**: `plugin.publish` → Beehive is the crash-safe record;
  the live sink is a best-effort convenience and loses nothing if it dies.
- **Kill paths**: soft flag (polled), plus the platform's `sesctl rm` and k8s
  resource limits — the harness never runs without a tested kill.
- **Bounded end-phase**: one hard wall-clock budget covers digest + request
  serving; when it expires the window closes regardless. LLM output is always
  stamped `candidate=true` — a convenience digest, never the record.

## Data policy (why the frames are synthetic)

This repository and image contain **no third-party camera imagery**. The
baked-in series are deterministic synthetic scenes generated at build time —
enough to prove plumbing (capture → publish → kill → latency) with zero
egress on first run. Real regime frames are distributed as **versioned
bundles from the operator's own infrastructure** and pulled at runtime
(`--events-root`), keeping data licensing and distribution under operator
control.

## Run locally (no node required)

```bash
pip install "pywaggle==0.56.3" numpy Pillow
python3 plugin/make_synthetic_events.py --out /tmp/events
PYWAGGLE_LOG_DIR=/tmp/harness python3 plugin/harness_r0_real.py \
    --events-root /tmp/events --min-frames 5
# every measurement mirrors to /tmp/harness/data.ndjson
```

Optional live feed + kill: set `HARNESS_RECEIVER=http://<your-receiver-host>`
(a small Flask blueprint implementing `/harness/ingest`, `/harness/control`,
`/harness/kill`).
