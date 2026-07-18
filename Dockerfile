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

WORKDIR /app
COPY app/ /app/app/
COPY plugin/ /app/plugin/
RUN python3 /app/plugin/make_synthetic_events.py --out /app/data/events

ENV PYTHONUNBUFFERED=1
# Default: run the bundled synthetic series through the full capture spine +
# shipped cascade; minutes of work, then exit clean. Override --events-root to
# a runtime-pulled bundle for real regime frames.
ENTRYPOINT ["python3", "/app/plugin/harness_r0_real.py", \
            "--events-root", "/app/data/events", \
            "--min-frames", "5", "--max-events", "2"]
