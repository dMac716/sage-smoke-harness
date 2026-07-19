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
  tailnet-probe|tailnet)
    : "${TS_AUTHKEY:?TS_AUTHKEY required for tailnet mode}"
    # userspace networking: no TUN, no host route change — cannot become
    # all-traffic. tailscaled's own outbound HTTP proxy resolves MagicDNS,
    # so the app needs no pod DNS changes (--accept-dns=false).
    SOCK=/tmp/tailscaled.sock
    /usr/local/bin/tailscaled --tun=userspace-networking \
        --socks5-server=localhost:1055 \
        --outbound-http-proxy-listen=localhost:1080 \
        --state=/tmp/ts.state --socket="$SOCK" >/tmp/tailscaled.log 2>&1 &
    for i in $(seq 1 30); do
      /usr/local/bin/tailscale --socket="$SOCK" status >/dev/null 2>&1 && break
      sleep 1
    done
    /usr/local/bin/tailscale --socket="$SOCK" up --authkey="$TS_AUTHKEY" \
        --hostname="sage-harness-${WAGGLE_NODE_ID:-${HOSTNAME:-probe}}" \
        --accept-routes=false --accept-dns=false || true
    export TS_IP="$(/usr/local/bin/tailscale --socket="$SOCK" ip -4 2>/dev/null | head -1)"
    export HTTP_PROXY=http://localhost:1080 HTTPS_PROXY=http://localhost:1080 \
           http_proxy=http://localhost:1080 https_proxy=http://localhost:1080
    exec python3 /app/plugin/tailnet_probe.py ;;
  endphase)
    exec python3 /app/plugin/harness_endphase.py "$@" ;;
  harness|*)
    exec python3 /app/plugin/harness_r0_real.py \
        --events-root "${EVENTS_ROOT:-/app/data/events}" \
        --min-frames "${MIN_FRAMES:-5}" --max-events "${MAX_EVENTS:-2}" "$@" ;;
esac
