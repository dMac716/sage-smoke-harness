# sage-smoke-harness plugin image.
#
# Self-contained first run: synthetic proof series are generated AT BUILD TIME
# (no third-party imagery in this repo or image; real regime frames are pulled
# at runtime as versioned bundles from the operator's own infrastructure).
#
# Base note: waggle/plugin-base:1.1.1-ml ships Python 3.6.9 (too old for this
# codebase, and 1.6GB of unused torch). pywaggle is a plain pip package — a
# modern slim base + pywaggle is ~90MB, faster to pull on the node, and
# WES-compatible (pywaggle finds the node broker via WAGGLE_* env, which WES
# injects; PYWAGGLE_LOG_DIR mirrors locally for off-node runs).
FROM python:3.11-slim-bookworm

RUN pip install --no-cache-dir "pywaggle==0.56.3" numpy Pillow

# RF capture + mesh gateway (lora-capture / mesh-gateway modes; see
# docs/lora-share.md + docs/meshtastic-forward.md):
# - rtl-433: Debian bookworm ships rtl-433 22.11-1 on both amd64 and arm64
#   (verified at packages.debian.org/bookworm/rtl-433) — decodes 433/868/915
#   ISM traffic from an RTL-SDR dongle. If a future base image drops the apt
#   package, build from source instead:
#     apt-get install -y cmake build-essential librtlsdr-dev libusb-1.0-0-dev
#     git clone https://github.com/merbanan/rtl_433 && cmake -B build rtl_433
#     cmake --build build && cmake --install build
# - meshtastic: serial/protobuf API for the Meshtastic gateway; small,
#   mostly pure-python deps (protobuf, pyserial, pypubsub) — no BLE extras.
RUN apt-get update && apt-get install -y --no-install-recommends rtl-433 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir meshtastic \
    && rtl_433 -V

# Tailscale static binaries for the userspace-networking dual-publish leg
# (tailnet-probe / R5 modes). Fetched by arch from the official static index
# so this works for both the arm64 node and amd64 builds. No TUN device or
# host route change is ever used — see plugin/launch.sh.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && arch="$(uname -m)" && case "$arch" in \
         x86_64) a=amd64 ;; aarch64|arm64) a=arm64 ;; *) a="$arch" ;; esac \
    && tgz="$(curl -fsSL 'https://pkgs.tailscale.com/stable/?mode=json' \
             | python3 -c "import json,sys;print(json.load(sys.stdin)['Tarballs']['$a'])")" \
    && curl -fsSL "https://pkgs.tailscale.com/stable/$tgz" -o /tmp/ts.tgz \
    && tar -xzf /tmp/ts.tgz -C /tmp \
    && mv /tmp/tailscale_*/tailscale /tmp/tailscale_*/tailscaled /usr/local/bin/ \
    && rm -rf /tmp/ts.tgz /tmp/tailscale_* \
    && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* \
    && tailscale --version

WORKDIR /app
COPY app/ /app/app/
COPY plugin/ /app/plugin/
RUN python3 /app/plugin/make_synthetic_events.py --out /app/data/events

ENV PYTHONUNBUFFERED=1
# Default: run the bundled synthetic series through the full capture spine +
# shipped cascade; minutes of work, then exit clean. Override --events-root to
# a runtime-pulled bundle for real regime frames.
# Mode dispatch (HARNESS_MODE=harness|probe|endphase); extra args append to
# the harness CLI (e.g. job spec args: ["--limit-frames","1"]).
ENTRYPOINT ["/bin/sh", "/app/plugin/launch.sh"]
