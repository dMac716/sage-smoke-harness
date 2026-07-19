#!/bin/sh
# Mode dispatch for the single plugin image (keeps one ECR app; the job spec
# selects behavior via HARNESS_MODE):
#   harness (default) — capture spine + cascade over $EVENTS_ROOT (bundled)
#   probe             — R2 egress boundary probe (seconds, read-only)
#   cache-check       — B2 hostPath persistence marker
#   tailnet-probe     — R5a userspace-Tailscale join + reach droplet
#   regime            — R5c full regime: tunnel up -> pull real frames over the
#                       tailnet -> run cascade streaming live to the droplet
#   endphase          — bounded post-run digest window (needs OLLAMA_URL)
#   lora-capture      — RTL-SDR rtl_433 -> Beehive RF capture (docs/lora-share)
#   mesh-gateway      — Meshtastic USB node -> Beehive gateway
#                       (docs/meshtastic-forward; channel secrets via job env)
set -eu
MODE="${HARNESS_MODE:-harness}"

# Bring up userspace-Tailscale (no TUN, no host route — cannot become
# all-traffic) and export HTTP_PROXY = tailscaled's outbound proxy, which
# resolves MagicDNS so callers need no pod DNS changes. Sets TS_IP.
ts_up() {
    : "${TS_AUTHKEY:?TS_AUTHKEY required for tailnet mode}"
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
}

case "$MODE" in
  probe)
    exec python3 /app/plugin/egress_probe.py ;;
  cache-check)
    exec python3 /app/plugin/cache_check.py ;;
  tailnet-probe|tailnet)
    ts_up
    exec python3 /app/plugin/tailnet_probe.py ;;
  regime)
    # R5c — real frames over the tailnet + live-stream to the droplet.
    ts_up
    : "${BUNDLE_BASE:?BUNDLE_BASE required (droplet MagicDNS base URL)}"
    BN="${BUNDLE_NAME:-proof-frames}"
    python3 /app/plugin/bundle_pull.py --base "$BUNDLE_BASE" --name "$BN" \
        --cache /tmp/bundles
    export HARNESS_RECEIVER="${HARNESS_RECEIVER:-$BUNDLE_BASE}"
    exec python3 /app/plugin/harness_r0_real.py \
        --events-root "/tmp/bundles/$BN/current/events" \
        --min-frames "${MIN_FRAMES:-5}" --max-events "${MAX_EVENTS:-3}" "$@" ;;
  endphase)
    exec python3 /app/plugin/harness_endphase.py "$@" ;;
  lora-capture)
    # RECEIVE/decode only — env passthrough for the capture bounds; extra job
    # spec args append (e.g. ["--dry-run"]).
    exec python3 /app/plugin/lora_capture.py \
        --rtl433-cmd "${RTL433_CMD:-rtl_433 -F json}" \
        --duration-s "${DURATION_S:-3600}" \
        --limit-packets "${LIMIT_PACKETS:-0}" \
        --dedupe-window-s "${DEDUPE_WINDOW_S:-2.0}" \
        --heartbeat-s "${HEARTBEAT_S:-10.0}" "$@" ;;
  mesh-gateway)
    # MESH_CHANNEL_NAME / MESH_CHANNEL_PSK arrive as job-spec env (job-secret
    # pattern — never baked into the image, never placed on argv; the script
    # reads them from the environment itself and never logs the PSK).
    exec python3 /app/plugin/mesh_gateway.py \
        --source "${MESH_SOURCE:-serial}" \
        --duration-s "${DURATION_S:-3600}" \
        --stats-period-s "${STATS_PERIOD_S:-60}" "$@" ;;
  harness|*)
    exec python3 /app/plugin/harness_r0_real.py \
        --events-root "${EVENTS_ROOT:-/app/data/events}" \
        --min-frames "${MIN_FRAMES:-5}" --max-events "${MAX_EVENTS:-2}" "$@" ;;
esac
