"""Reproducible synthetic preview; these are NOT measured ShinOne results."""
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .coverage_influence_report import generate_report, KP_COLS
except ImportError:
    from coverage_influence_report import generate_report, KP_COLS


def run(output=None):
    output = Path(output or Path(__file__).parent / "coverage_plots" / "influence_demo")
    rng = np.random.default_rng(42)
    centers = np.array([[-1.7, -1.0, .2], [1.6, -.8, 1.0], [.0, 1.8, -1.3]])
    ref = pd.DataFrame(np.concatenate([rng.normal(c, .42, size=(200, 3)) for c in centers]), columns=KP_COLS)
    ref["pattern_id"] = [f"REF_{i:04d}" for i in range(len(ref))]
    ref["region"] = np.repeat(["left", "right", "upper"], 200)
    ref["cover_status"] = "gap"
    parts = []
    for name, center, count in [("A", centers[0], 32), ("B", centers[1], 16),
                                ("C", centers[0]+.12, 16), ("D", [7, 7, 7], 5)]:
        part = pd.DataFrame(rng.normal(center, .3, size=(count, 3)), columns=KP_COLS)
        part["gauge_name"] = [f"{name}_{i:03d}" for i in range(count)]
        parts.append(part)
    sample = pd.concat(parts, ignore_index=True)
    ref["kpca_frame_id"] = sample["kpca_frame_id"] = "synthetic-demo-seed-42"
    output.mkdir(parents=True, exist_ok=True)
    ref.to_csv(output / "demo_reference.csv", index=False)
    sample.to_csv(output / "demo_sample.csv", index=False)
    return generate_report(ref, sample, .45, output, bins=18,
        title="SYNTHETIC DEMO · Reference coverage & group influence",
        provenance={"data": "synthetic KP coordinates, NOT measured data", "seed": 42})


if __name__ == "__main__":
    run()
