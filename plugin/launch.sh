#!/bin/sh
# Mode dispatch for the single plugin image (keeps one ECR app; the job spec
# selects behavior via HARNESS_MODE):
#   harness (default) — capture spine + cascade over $EVENTS_ROOT
#   probe             — the R2 egress boundary probe (seconds, read-only)
#   endphase          — bounded post-run digest window (needs OLLAMA_URL)
set -eu
MODE="${HARNESS_MODE:-harness}"
case "$MODE" in
  probe)
    exec python3 /app/plugin/egress_probe.py ;;
  cache-check)
    exec python3 /app/plugin/cache_check.py ;;
  endphase)
    exec python3 /app/plugin/harness_endphase.py "$@" ;;
  harness|*)
    exec python3 /app/plugin/harness_r0_real.py \
        --events-root "${EVENTS_ROOT:-/app/data/events}" \
        --min-frames "${MIN_FRAMES:-5}" --max-events "${MAX_EVENTS:-2}" "$@" ;;
esac
