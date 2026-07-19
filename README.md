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

## Bring your own redundancy (for other participants)

The plugin image has **no hardcoded infrastructure** — receiver host, bundle
host, and auth key are all env. To get the same crash-safe + real-time
dual-publish redundancy on **your own** tailnet:

1. **Mint your own Tailscale key** (ephemeral + reusable) on your tailnet.
   A key is tailnet-scoped — it only joins *your* tailnet, so this never
   touches anyone else's setup.
2. **Run the receiver** (`receiver/serve.py`) on a host joined to your tailnet
   (`pip install flask; DATA_ROOT=./harness-data python3 receiver/serve.py`),
   then expose it tailnet-only (`tailscale serve 8777`).
3. **Put frame bundles** under `$DATA_ROOT/bundles/<name>/` (a `current.json`
   manifest + a `.tar.gz`; see `plugin/bundle_pull.py` for the format).
4. **Submit** with your env — same image, your values:
   ```
   HARNESS_MODE=regime  TS_AUTHKEY=<your key>
   BUNDLE_BASE=http://<your-host-magicdns>   # HARNESS_RECEIVER defaults to this
   ```

Everything is env-parameterised, so your job and the reference job run the
**same image** with different config — neither can break the other.

## Zero-infra redundancy — just a GitHub repo (for everyone else)

No droplet, no tailnet, no key. Every Sage node run publishes crash-safe to
**Beehive** (the public data API), so you can archive your runs with only a
GitHub repo:

1. **Fork this repo.**
2. Set repo **variable** `HARNESS_PLUGIN` to your plugin image regex (e.g.
   `.*yourname/yourplugin.*`), optionally `HARNESS_VSN` (default `H0.*`).
3. **Enable Actions.** `.github/workflows/archive-beehive.yml` runs every
   30 min, tails Beehive for your measurements, and commits them under
   `harness-archive/` — watermarked + de-duped (`tools/beehive_archiver.py`).

That's the whole setup. It's *redundancy* (lags one interval), not a live
feed — for real-time, run `receiver/serve.py` on a tailnet host (previous
section). Three tiers, pick what you have:

| You have… | Use | Gets you |
|---|---|---|
| just a GitHub repo | this Action | crash-safe archive **+ a GitHub Pages report** you can browse, ~30-min lag |
| a tailnet host | `receiver/serve.py` + your key | real-time live feed + KILL + bundles |
| nothing | (Beehive itself) | your data is already safe on Beehive; query it anytime |
