# Smoke-detection test harness (validation instrument)

## Science motivation

Field precision of camera smoke detectors collapses under domain transfer
(models trained on curated benchmarks lose most of their precision on real
field cameras). Before any detector claim is trustworthy on Sage hardware, it
must be measured — same frames, same silicon, honest capture of every
verdict, latency, and failure.

This plugin is that measuring instrument. It is not a production detector; it
is the harness that runs detectors under a validation regime and publishes
everything needed to audit the run afterwards, including its own crashes.

## What it measures per run

- Per-frame: cascade verdict (change gate, smoke score, tile scores, alert),
  full verdict JSON, inference latency (`timeit`).
- Per-run: resource samples on a dedicated thread, console log stream,
  crash-safe tracebacks, header/exit records with counts and status.
- Optional bounded end-phase: a local-LLM digest of the run (always labeled
  `candidate` — the published measurement stream remains the record).

## Data

First-run series are synthetic (generated at image build; plumbing proof
only). Real evaluation frames are versioned bundles distributed from the
operator's infrastructure at runtime — no third-party imagery ships in this
image.

## Outputs (ontology)

`run.header`, `event.header`, `detect.smoke_score`, `detect.diff`,
`detect.max_tile_score`, `detect.alert`, `harness.frame_verdict`,
`inference_ns`, `log.console`, `sys.harness.*`, `harness.traceback`,
`harness.killed`, `run.exit`, `endphase.*`, `window.closing`.
