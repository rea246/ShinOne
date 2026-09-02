import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("7.select_representatives.py")
SPEC = importlib.util.spec_from_file_location("representative_selection", MODULE_PATH)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def _record(cluster, count, dispersion):
    return {
        "h0_label": 0,
        "coarse_group": "H0_0",
        "topology_cluster": cluster,
        "final_cluster": f"H0_0_T{cluster}",
        "pattern_count": count,
        "centroid_distance_p95": dispersion,
    }


class RepresentativeSelectionTest(unittest.TestCase):
    def test_stage2_labels_require_matching_population_fingerprint(self):
        rows = np.array([2, 4], dtype=np.int64)
        h0_labels = np.array([0, -1], dtype=np.int32)
        topology_labels = np.array([3, 1], dtype=np.int32)
        with tempfile.TemporaryDirectory() as out_dir:
            labels_path = Path(out_dir, "topology_labels.npz")
            np.savez(
                labels_path,
                rows=rows,
                h0_labels=h0_labels,
                topology_labels=topology_labels,
                population_fingerprint=np.asarray("matching-hash"),
            )
            original_path = selection.STAGE2_LABELS_PATH
            selection.STAGE2_LABELS_PATH = labels_path
            try:
                actual = selection._load_stage2_labels(
                    rows, h0_labels, "matching-hash"
                )
                np.testing.assert_array_equal(actual, topology_labels)
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    selection._load_stage2_labels(
                        rows, h0_labels, "different-hash"
                    )
            finally:
                selection.STAGE2_LABELS_PATH = original_path

    def test_quota_is_exact_deterministic_and_rare_safe(self):
        records = [
            _record(0, 50, 1.0),
            _record(1, 20, 2.0),
            _record(2, 2, 0.2),
        ]
        first = selection.allocate_community_quotas(
            records, budget=20, minimum_per_community=3
        )
        second = selection.allocate_community_quotas(
            records, budget=20, minimum_per_community=3
        )

        self.assertEqual(first, second)
        self.assertEqual(sum(row["allocated_quota"] for row in first), 20)
        self.assertEqual(first[2]["allocated_quota"], 2)
        for row in first:
            self.assertGreaterEqual(row["allocated_quota"], row["minimum_quota"])
            self.assertLessEqual(row["allocated_quota"], row["pattern_count"])

    def test_quota_rejects_more_communities_than_budget(self):
        records = [_record(cluster, 2, 1.0) for cluster in range(4)]
        with self.assertRaisesRegex(ValueError, "community"):
            selection.allocate_community_quotas(
                records, budget=3, minimum_per_community=1
            )

    def test_representatives_are_unique_and_include_core_and_boundary(self):
        embedding = np.array(
            [[0.0], [0.1], [0.2], [0.3], [1.0], [2.0], [5.0], [10.0]],
            dtype=np.float32,
        )
        selected, backend = selection.select_community_representatives(
            embedding, quota=5, use_cuda=False
        )

        indices = [row["local_index"] for row in selected]
        reasons = [row["selection_reason"] for row in selected]
        self.assertEqual(len(indices), 5)
        self.assertEqual(len(set(indices)), 5)
        self.assertEqual(reasons[0], "centroid_nearest")
        self.assertIn("core_farthest", reasons)
        self.assertIn("boundary_farthest", reasons)
        self.assertTrue(backend.startswith("numpy_cpu"))

    def test_output_files_preserve_exact_manifest(self):
        manifest = []
        for index in range(3):
            manifest.append({
                "pattern_key": f"pattern-{index}",
                "global_row": index,
                "h0_label": 0,
                "topology_cluster": 0,
                "final_cluster": "H0_0_T0",
                "selection_rank": index + 1,
                "global_selection_rank": index + 1,
                "selection_reason": (
                    "centroid_nearest" if index == 0 else "core_farthest"
                ),
                "distance_to_centroid": float(index),
                "distance_to_nearest_prior": float(index),
            })
        quota = selection.allocate_community_quotas(
            [_record(0, 3, 1.0)], budget=3, minimum_per_community=3
        )

        with tempfile.TemporaryDirectory() as out_dir:
            original_out_dir = selection.OUT_DIR
            selection.OUT_DIR = Path(out_dir)
            try:
                selection._write_outputs(manifest, quota, {"selected_patterns": 3})
                with Path(out_dir, "representative_1000.csv").open(
                    "r", newline="", encoding="utf-8"
                ) as fp:
                    rows = list(csv.DictReader(fp))
                self.assertEqual(
                    [row["pattern_key"] for row in rows],
                    ["pattern-0", "pattern-1", "pattern-2"],
                )
                with np.load(Path(out_dir, "representative_1000.npz")) as labels:
                    np.testing.assert_array_equal(labels["rows"], [0, 1, 2])
                metadata = json.loads(
                    Path(
                        out_dir, "representative_selection_metadata.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["selected_patterns"], 3)
            finally:
                selection.OUT_DIR = original_out_dir


if __name__ == "__main__":
    unittest.main()
