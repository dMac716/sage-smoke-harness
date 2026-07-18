#!/usr/bin/env python3
"""Generate SYNTHETIC proof event series for the harness image.

Two deterministic (seeded) single-camera series in the same on-disk shape the
harness expects (``events/<id>/frames/*.jpg`` + ``metadata.json``):

  synthetic-plume  — static hillside scene with a soft gray plume that grows
                     frame over frame (exercises the change gate + smoke
                     heuristic + escalation path),
  synthetic-static — the identical scene with sensor noise only (negative;
                     exercises the no-change path).

WHY SYNTHETIC: the public plugin repo ships NO third-party camera imagery —
real regime frames are distributed as versioned bundles from the operator's
own infrastructure and pulled at runtime (see README). These frames exist so
the FIRST on-node run needs zero egress and zero third-party data; they prove
PLUMBING (capture, publish, kill, latency), never detection quality.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

W, H, N_FRAMES = 640, 480, 10


def base_scene(rng) -> np.ndarray:
    """Sky gradient over a textured hillside — static across the series."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    horizon = int(H * 0.55)
    sky_t = np.linspace(0, 1, horizon)[:, None]
    img[:horizon, :, 0] = 150 + 60 * sky_t
    img[:horizon, :, 1] = 180 + 40 * sky_t
    img[:horizon, :, 2] = 230 - 10 * sky_t
    hill = rng.uniform(0, 18, (H - horizon, W, 1)).astype(np.float32)
    img[horizon:, :, 0] = 90 + hill[:, :, 0]
    img[horizon:, :, 1] = 110 + hill[:, :, 0] * 1.2
    img[horizon:, :, 2] = 70 + hill[:, :, 0] * 0.6
    return img


def add_plume(img: np.ndarray, step: int) -> np.ndarray:
    """Soft gray plume rising from a fixed hillside point, growing with step."""
    out = img.copy()
    cx, base_y = int(W * 0.62), int(H * 0.58)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # plume core drifts up and right as it grows
    size = 8 + step * 7
    cy = base_y - step * 12
    d2 = ((xx - cx - step * 3) ** 2) / (2.5 * size**2) + ((yy - cy) ** 2) / (size**2)
    alpha = np.clip(np.exp(-d2) * (0.25 + 0.06 * step), 0, 0.85)[:, :, None]
    gray = np.full_like(out, 185.0)
    return out * (1 - alpha) + gray * alpha


def write_series(root: Path, name: str, plume: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scene = base_scene(rng)
    fdir = root / name / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for i in range(N_FRAMES):
        frame = add_plume(scene, i) if plume else scene
        noisy = np.clip(frame + rng.normal(0, 2.0, frame.shape), 0, 255)
        Image.fromarray(noisy.astype(np.uint8)).save(
            fdir / f"20260101T{i:02d}0000Z_{i:04d}.jpg", quality=88)
    (root / name / "metadata.json").write_text(json.dumps({
        "camera_id": name,
        "synthetic": True,
        "purpose": "plumbing proof only — never detection-quality evidence",
        "generator": "plugin/make_synthetic_events.py",
        "seed": seed, "n_frames": N_FRAMES,
    }, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/events")
    args = ap.parse_args()
    root = Path(args.out)
    write_series(root, "synthetic-plume", plume=True, seed=7)
    write_series(root, "synthetic-static", plume=False, seed=7)
    print(f"wrote 2 synthetic series under {root}")


if __name__ == "__main__":
    main()
