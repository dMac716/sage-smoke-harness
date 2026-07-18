#!/usr/bin/env python3
"""Offline dev runner for the Smoke Spotter plugin — NO node, NO API, NO pywaggle.

Runs the exact same cheap->expensive pipeline as the deployed plugin
(``app/main.py``) against a local clip or a directory of frames, then prints the
per-frame verdicts. This is how you develop and demo the plugin without Sage
developer access or a live node.

Sources:
  * A video clip:   --camera file://clip.mp4   (or a bare path: clip.mp4)
  * An image dir:   --frames-dir ./frames/      (*.jpg/*.jpeg/*.png, sorted)

Frame decoding order of preference (all optional, graceful fallback):
  1. opencv (cv2)        — best; handles video + images
  2. Pillow (PIL)        — images only (a dir of stills)
The VLM defaults to ``none`` here so the runner is fully offline; pass
``--vlm ollama`` to exercise a local model if you have one.

Examples:
  python3 app/dev_run.py --frames-dir data/sample_frames
  python3 app/dev_run.py --camera file://data/clip.mp4
  python3 app/dev_run.py --camera file://data/clip.mp4 --json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Iterator, Optional

try:
    from app import main as main_mod
    from app import smoke as smoke_mod
    from app import vlm as vlm_mod
except ImportError:  # run directly from inside app/
    import main as main_mod  # type: ignore
    import smoke as smoke_mod  # type: ignore
    import vlm as vlm_mod  # type: ignore

LOG = logging.getLogger("smoke_spotter.dev")

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# A tiny offline ImageSample stand-in (matches the fields main.process_frame
# reads: .data numpy array, .timestamp ns, and an encoder for the VLM path).
# ─────────────────────────────────────────────────────────────────────────────

class DevSample:
    def __init__(self, data, timestamp: int, jpeg_bytes: Optional[bytes] = None):
        self.data = data            # numpy RGB array (HxWx3) or PIL fallback
        self.timestamp = timestamp  # ns since epoch
        self._jpeg = jpeg_bytes

    def encode_jpeg(self) -> Optional[bytes]:
        if self._jpeg is not None:
            return self._jpeg
        return _encode_jpeg(self.data)


def _encode_jpeg(data) -> Optional[bytes]:
    """Encode a frame (numpy array or PIL image) to JPEG bytes, or None."""
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if isinstance(data, np.ndarray):
            arr = data
            if arr.ndim == 3 and arr.shape[2] == 3:  # RGB -> BGR for cv2
                arr = arr[:, :, ::-1]
            ok, buf = cv2.imencode(".jpg", arr)
            if ok:
                return bytes(buf.tobytes())
    except Exception:
        pass
    try:
        import io
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore

        if isinstance(data, np.ndarray):
            img = Image.fromarray(data.astype("uint8"))
        else:
            img = data
        bio = io.BytesIO()
        img.convert("RGB").save(bio, format="JPEG")
        return bio.getvalue()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Frame sources
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(camera: str) -> str:
    if camera.startswith("file://"):
        return camera[len("file://"):]
    return camera


def iter_video(path: str, every: int = 1) -> Iterator[DevSample]:
    """Yield frames from a video clip using cv2. Raises if cv2 is missing."""
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise SystemExit(
            f"reading a video clip requires opencv (cv2): {exc}\n"
            "Install it (`pip install opencv-python`) or use --frames-dir with "
            "a directory of still images instead."
        )
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    delta_ns = int(1e9 / fps) if 0 < fps < 1000 else 0
    base_ns = int(Path(path).stat().st_mtime_ns) if os.path.exists(path) else 0
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if idx % max(1, every) == 0:
                rgb = bgr[:, :, ::-1].copy()  # BGR -> RGB to match the node
                yield DevSample(data=rgb, timestamp=base_ns + idx * delta_ns)
            idx += 1
    finally:
        cap.release()


def iter_frames_dir(root: str) -> Iterator[DevSample]:
    """Yield frames from a directory of still images (sorted by name)."""
    files = sorted(
        p for p in Path(root).glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f"no images ({sorted(_IMAGE_SUFFIXES)}) found in {root!r}")
    for p in files:
        sample = _load_image_file(p)
        if sample is not None:
            yield sample


def _load_image_file(p: Path) -> Optional[DevSample]:
    ts = int(p.stat().st_mtime_ns)
    raw = p.read_bytes()
    # Prefer decoding to a numpy RGB array so .data matches the node contract.
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        bgr = cv2.imread(str(p))
        if bgr is not None:
            return DevSample(data=bgr[:, :, ::-1].copy(), timestamp=ts,
                             jpeg_bytes=raw)
    except Exception:
        pass
    try:
        import io
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return DevSample(data=np.asarray(img, dtype=np.uint8), timestamp=ts,
                         jpeg_bytes=raw)
    except Exception as exc:
        LOG.warning("could not decode %s: %s", p, exc)
        # Last resort: hand the raw bytes through; smoke.py can decode bytes.
        return DevSample(data=raw, timestamp=ts, jpeg_bytes=raw)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline — runs the EXACT shipped cascade (no node side effects)
# ─────────────────────────────────────────────────────────────────────────────
#
# Codex review flagged that this runner used to drive the legacy global
# ``ChangeGate`` while the deployed plugin (app/main.py) ships the TILED +
# rolling-baseline detector — so the offline loop validated a DIFFERENT detector
# than we deploy, defeating the whole point of offline-first. We now build the
# detector with ``main.build_detector`` and score with ``main.process_frame``, so
# the dev loop, the tests, and the node all exercise ONE code path.


def _config_from_args(args: argparse.Namespace) -> "main_mod.SmokeConfig":
    """Map the dev CLI to a real ``SmokeConfig`` (the same object main.py uses)."""
    return main_mod.SmokeConfig(
        camera=args.camera or "file://dev",
        change_threshold=args.change_threshold,
        smoke_alert=args.smoke_alert,
        vlm_threshold=args.vlm_threshold,
        vlm_kind=args.vlm,
        vlm_timeout_s=args.vlm_timeout,
        # Default to the shipped TILED cascade; --no-tiled opts into the legacy
        # global change-gate path for A/B comparison (parity with main --no-tiled).
        tiled=not args.no_tiled,
    )


def process_sample(sample: DevSample, detector, backend, *,
                   smoke_alert: float, vlm_threshold: float,
                   force_escalate: bool = False) -> dict:
    """Score one frame through the shipped cascade. Thin named seam over
    ``main.process_frame`` so the offline runner and the node share it verbatim.
    """
    return main_mod.process_frame(
        sample, detector, backend,
        smoke_alert=smoke_alert, vlm_threshold=vlm_threshold,
        force_escalate=force_escalate,
    )


def run(args: argparse.Namespace) -> int:
    deps = smoke_mod.deps_available()
    LOG.info("deps: numpy=%s pillow=%s", deps["numpy"], deps["pillow"])
    if not deps["numpy"]:
        print("WARNING: numpy unavailable — smoke_score will be 0.0 "
              "(install numpy + Pillow for real detection)", file=sys.stderr)

    backend = vlm_mod.build_backend(args.vlm, timeout=args.vlm_timeout)
    LOG.info("VLM backend: %s", getattr(backend, "name", "none"))

    cfg = _config_from_args(args)
    detector = main_mod.build_detector(cfg)
    det_desc = (f"tiled {cfg.grid_rows}x{cfg.grid_cols}/{cfg.baseline}"
                if cfg.tiled else "global change-gate")

    if args.frames_dir:
        source = iter_frames_dir(args.frames_dir)
        src_desc = f"frames-dir {args.frames_dir}"
    else:
        source = iter_video(_resolve_path(args.camera), every=args.every)
        src_desc = f"clip {args.camera}"
    print(f"smoke-spotter (offline) | source={src_desc} | "
          f"detector={det_desc} | vlm={getattr(backend, 'name', 'none')}")

    n = n_changed = n_alert = n_escalated = 0

    for i, sample in enumerate(source):
        v = process_sample(
            sample, detector, backend,
            smoke_alert=cfg.smoke_alert,
            vlm_threshold=cfg.vlm_threshold,
        )
        n += 1
        n_changed += int(v["changed"])
        n_alert += int(v["alert"])
        n_escalated += int(v["escalated"])
        if args.json:
            print(json.dumps(v, separators=(",", ":")))
        else:
            mark = "ALERT" if v["alert"] else ("esc" if v["escalated"] else "")
            print(
                f"  frame {i:04d}  changed={int(v['changed'])} "
                f"diff={v['diff']:6.2f}  smoke={v['smoke_score']:.3f}  "
                f"tile={v.get('max_tile_score', 0.0):.3f}  "
                f"vlm={v['vlm']:<7} {mark}"
            )
        if args.limit and n >= args.limit:
            break

    print(f"summary: {n} frames | {n_changed} changed | "
          f"{n_escalated} escalated | {n_alert} alerts")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline runner for Smoke Spotter (no node/API/pywaggle)",
    )
    src = p.add_argument_group("source (pick one)")
    src.add_argument("--camera", default=None,
                     help="file://clip.mp4 (or a bare video path)")
    src.add_argument("--frames-dir", default=None,
                     help="directory of still images (*.jpg/*.png), sorted")
    src.add_argument("--every", type=int, default=1,
                     help="for video: process every Nth frame (default: 1)")

    th = p.add_argument_group("thresholds")
    th.add_argument("--change-threshold", type=float,
                    default=smoke_mod.DEFAULT_CHANGE_THRESHOLD)
    th.add_argument("--smoke-alert", type=float, default=0.75)
    th.add_argument("--vlm-threshold", type=float, default=0.6)
    th.add_argument(
        "--no-tiled", dest="no_tiled", action="store_true",
        help="use the legacy GLOBAL change-gate path instead of the shipped "
             "tiled + rolling-baseline cascade (parity with app.main --no-tiled)")

    v = p.add_argument_group("VLM")
    v.add_argument("--vlm", default="none",
                   choices=["none", "ollama", "node", "apple", "auto"],
                   help="VLM backend (default: none — fully offline). "
                        "'apple'/'node'/'ollama' need a reachable HTTP sidecar; "
                        "they degrade to unknown offline. Parity with app.main "
                        "main._VLM_CHOICES / vlm.build_backend.")
    v.add_argument("--vlm-timeout", type=float, default=60.0)

    p.add_argument("--limit", type=int, default=0, help="stop after N frames")
    p.add_argument("--json", action="store_true",
                   help="emit one JSON verdict per line")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.camera and not args.frames_dir:
        build_parser().error("provide --camera file://clip.mp4 or --frames-dir DIR")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
