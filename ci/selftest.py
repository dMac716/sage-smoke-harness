#!/usr/bin/env python3
"""CI selftest: run the harness over the synthetic series and assert the
capture manifest + cascade behavior. This is the same gate the image build is
judged by locally — plumbing AND discrimination (plume alerts, static stays
silent)."""
import collections
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED = {"run.header", "event.header", "detect.smoke_score", "detect.diff",
            "detect.max_tile_score", "detect.alert", "harness.frame_verdict",
            "inference_ns", "log.console", "run.exit"}


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        events = Path(td) / "events"
        out = Path(td) / "out"
        out.mkdir()
        subprocess.run([sys.executable, str(root / "plugin/make_synthetic_events.py"),
                        "--out", str(events)], check=True)
        env = {**os.environ, "PYWAGGLE_LOG_DIR": str(out), "RUN_ID": "ci-selftest"}
        subprocess.run([sys.executable, str(root / "plugin/harness_r0_real.py"),
                        "--events-root", str(events), "--min-frames", "5",
                        "--max-events", "2"], check=True, env=env, timeout=600)

        names = collections.Counter()
        scores: dict = {}
        exit_meta = None
        for line in (out / "data.ndjson").read_text().splitlines():
            r = json.loads(line)
            names[r["name"]] += 1
            if r["name"] == "detect.smoke_score":
                scores.setdefault(r["meta"]["event"], []).append(float(r["value"]))
            if r["name"] == "run.exit":
                exit_meta = r["meta"]

        missing = REQUIRED - set(names)
        assert not missing, f"capture manifest incomplete: {missing}"
        assert exit_meta and exit_meta["status"] == "clean", f"bad exit: {exit_meta}"
        assert int(exit_meta["err"]) == 0, f"errors in run: {exit_meta}"
        plume, static = scores["synthetic-plume"], scores["synthetic-static"]
        assert max(plume) > 0.7, f"plume never scored (max {max(plume)})"
        assert plume[-1] > plume[0], "plume scores did not rise across the series"
        assert max(static) < 0.2, f"static series scored smoke (max {max(static)})"
        assert int(exit_meta["alerts"]) >= 1, "no alerts on the plume series"
        print(f"selftest OK: manifest={dict(names)}")
        print(f"plume max={max(plume):.3f} static max={max(static):.3f} "
              f"alerts={exit_meta['alerts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
