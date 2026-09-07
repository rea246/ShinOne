import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "topology_analysis", Path(__file__).with_name("7.analysis.py")
)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def make_stage6_fixture(directory):
    rng = np.random.default_rng(4)
    n = 80
    features = {name: rng.normal(size=(n, 8)).astype(np.float32)
                for name in analysis.FEATURE_BLOCKS}
    keys = [f"pattern-{i}" for i in range(n)]
    # Shuffled population rows test global-row vs array-position mapping.
    rows = rng.permutation(n).astype(np.int64)
    h0 = np.where(rows < 40, 0, -1)
    labels = (rows % 40) // 10
    fingerprint = analysis.population_fingerprint(keys, rows)
    np.savez(Path(directory) / "topology_labels.npz", rows=rows,
             h0_labels=h0, topology_labels=labels,
             population_fingerprint=np.asarray(fingerprint))
    metadata = {
        "feature_blocks": list(analysis.FEATURE_BLOCKS),
        "feature_dimensions": {name: 8 for name in analysis.FEATURE_BLOCKS},
        "block_weights": {name: 1.0 for name in analysis.FEATURE_BLOCKS},
        "population_fingerprint": fingerprint,
        "representative_patterns": 8, "final_communities": 8,
        "selected_resolution": 9.0,
    }
    analysis.write_json(Path(directory) / "topology_run_metadata.json", metadata)
    write_representatives(directory, keys, offset=5)
    return {"features": features, "keys": keys}, rows, metadata


def write_representatives(directory, keys, offset):
    with (Path(directory) / "topology_representatives.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["global_row", "pattern_key", "h0_label", "topology_cluster"])
        for row in range(offset, len(keys), 10):
            writer.writerow([row, keys[row], 0 if row < 40 else -1, (row % 40)//10])


class AnalysisTest(unittest.TestCase):
    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_ref_sampling_is_unique_deterministic_and_order_independent(self):
        rows = np.arange(60)
        first = analysis.sample_reference_rows(rows, 20, 0)
        second = analysis.sample_reference_rows(rows[::-1], 20, 0)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(np.unique(first)), 20)
        self.assertFalse(np.array_equal(first, analysis.sample_reference_rows(rows, 20, 1)))
        np.testing.assert_array_equal(analysis.sample_reference_rows(rows[:5], 20, 0), rows[:5])
        with self.assertRaisesRegex(ValueError, "unique"):
            analysis.sample_reference_rows([1, 1, 2], 3, 0)

    def test_reference_manifest_validates_identity_and_scope(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ref.csv"
            with path.open("w", newline="") as fp:
                writer = csv.writer(fp)
                writer.writerows([["pattern_key"], ["b"], ["d"]])
            rows = analysis.reference_population_rows(np.arange(4), ["a", "b", "c", "d"], path)
            np.testing.assert_array_equal(rows, [1, 3])
            with self.assertRaisesRegex(ValueError, "missing"):
                analysis.reference_population_rows(np.arange(3), ["a", "b", "c", "d"], path)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                analysis.reference_population_rows(np.arange(4), ["b", "b", "c", "d"], path)

    def test_stage6_manifest_and_cache_rows_match(self):
        with tempfile.TemporaryDirectory() as temp:
            cache, rows, _ = make_stage6_fixture(temp)
            with mock.patch.multiple(analysis, STAGE6_DIR=Path(temp),
                                     read_feature_cache=mock.Mock(return_value=cache)):
                inputs = analysis.load_stage6_inputs()
                self.assertEqual(len(inputs["feature_names"]), 40)
                np.testing.assert_array_equal(inputs["rows"][inputs["rep_positions"]], inputs["rep_rows"])
                recovered = analysis.gather_raw_embeddings(inputs["features"], inputs["rep_rows"])
                self.assertEqual(recovered.shape, (8, 40))
                np.testing.assert_array_equal(recovered[:, :8], cache["features"]["h0"][inputs["rep_rows"]])
                # A changed key order makes stale Stage-6 inputs fail loudly.
                cache["keys"][0], cache["keys"][1] = cache["keys"][1], cache["keys"][0]
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    analysis.load_stage6_inputs()

    def test_stage6_rejects_wrong_representative_key(self):
        with tempfile.TemporaryDirectory() as temp:
            cache, _, _ = make_stage6_fixture(temp)
            wrong_keys = [key + "-wrong" for key in cache["keys"]]
            write_representatives(temp, wrong_keys, 5)
            with mock.patch.multiple(analysis, STAGE6_DIR=Path(temp),
                                     read_feature_cache=mock.Mock(return_value=cache)):
                with self.assertRaisesRegex(ValueError, "key/label mismatch"):
                    analysis.load_stage6_inputs()

    def test_missing_population_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            analysis.positions_for_rows(np.array([3, 1, 5]), np.array([2]))

    def test_kpca_matches_sklearn_kernel_and_transform(self):
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import KernelPCA
        from threadpoolctl import threadpool_limits

        rng = np.random.default_rng(9)
        reference = rng.normal(size=(65, 40))
        query = rng.normal(size=(9, 40))
        with threadpool_limits(limits=2):
            frame = analysis.fit_reference_frame(reference, np.full(40, 1/np.sqrt(8)),
                                                 30, 3, 0, gamma=0.1)
            expected = KernelPCA(n_components=3, kernel="rbf", gamma=0.1,
                                 eigen_solver="arpack", random_state=0).fit(frame["landmarks"])
            expected_query = expected.transform(analysis.normalize_embeddings(query, frame))
            actual_query = analysis.transform_raw_embeddings(query, frame, block_size=4)
            np.testing.assert_allclose(frame["eigenvalues"], expected.eigenvalues_, rtol=1e-8, atol=1e-9)
            # Distances are invariant to arbitrary eigenvector sign choices.
            np.testing.assert_allclose(pdist(actual_query), pdist(expected_query), rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(analysis.transform_raw_embeddings(query, frame, block_size=1),
                                       actual_query, rtol=1e-9, atol=1e-10)

    def test_frame_uses_only_reference_scaler_and_preserves_block_weights(self):
        reference = np.random.default_rng(0).normal(size=(20, 5))
        weights = np.array([1, 2, 3, 4, 5], dtype=float)
        frame = analysis.fit_reference_frame(reference, weights, 10, 3, 0)
        normalized = analysis.normalize_embeddings(reference, frame)
        np.testing.assert_allclose(normalized.mean(0), 0, atol=1e-12)
        np.testing.assert_allclose(normalized.std(0), weights)
        center_before = frame["center"].copy()
        analysis.transform_raw_embeddings(np.full((2, 5), 100.0), frame)
        np.testing.assert_array_equal(frame["center"], center_before)

    def test_degenerate_reference_and_nonfinite_input_fail(self):
        with self.assertRaisesRegex(ValueError, "degenerate"):
            analysis.fit_reference_frame(np.ones((20, 4)), np.ones(4), 10, 3, 0)
        with self.assertRaisesRegex(ValueError, "invalid REF"):
            analysis.fit_reference_frame(np.full((20, 4), np.nan), np.ones(4), 10, 3, 0)

    def test_radius_excludes_self_and_is_nominal_budget_based(self):
        from scipy.spatial.distance import cdist
        reference = np.arange(21, dtype=float)[:, None]
        radius, k = analysis.representation_radius(reference, budget=10)
        self.assertEqual(k, 3)  # ceil(21/10), not round and not actual B size.
        matrix = cdist(reference, reference)
        expected = np.median(np.sort(matrix, axis=1)[:, k])
        self.assertEqual(radius, expected)
        repeated = np.zeros((30, 3))
        self.assertEqual(analysis.representation_radius(repeated, 10)[0], 0)

    def test_coverage_includes_boundary_and_retains_identical_ref_rows(self):
        ref = np.array([[0.0], [0.0], [1.0], [2.0], [3.0]])
        sample = np.array([[0.0], [3.0]])
        result = analysis.evaluate_coverage(ref, sample, radius=1.0)
        np.testing.assert_array_equal(result["covered"], [True]*5)
        result = analysis.evaluate_coverage(ref, sample, radius=.5)
        np.testing.assert_array_equal(result["covered"], [True, True, False, False, True])
        self.assertEqual(len(result["distances"]), 5)

    def test_nearest_distance_blocks_and_ties(self):
        ref = np.array([[0.0], [1.0], [2.0]])
        sample = np.array([[0.0], [2.0]])
        d, idx = analysis.nearest_distances(ref, sample, block_size=1)
        np.testing.assert_array_equal(d, [0, 1, 0])
        np.testing.assert_array_equal(idx, [0, 0, 1])
        with self.assertRaisesRegex(ValueError, "invalid"):
            analysis.nearest_distances(ref, np.empty((0, 1)))

    def test_saved_frame_transform_and_corruption_detection(self):
        raw = np.random.default_rng(8).normal(size=(30, 5))
        frame = analysis.fit_reference_frame(raw, np.ones(5), 15, 3, 0)
        frame["frame_id"] = np.asarray(analysis.arrays_fingerprint(frame))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "frame.npz"
            analysis.save_npz(path, frame)
            loaded = analysis.load_reference_frame(path)
            np.testing.assert_allclose(analysis.transform_raw_embeddings(raw, loaded),
                                       analysis.transform_raw_embeddings(raw, frame))
            frame["scale"][0] *= 2
            analysis.save_npz(path, frame)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                analysis.load_reference_frame(path)

    def test_projection_bundle_rejects_other_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "projection.npz"
            analysis.save_npz(path, {"frame_id": np.asarray("frame-A")})
            with self.assertRaisesRegex(ValueError, "different"):
                analysis.load_projection_bundle(path, {"frame_id": np.asarray("frame-B")})

    def test_integration_outputs_and_stage8_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage6 = root / "stage6"
            stage6.mkdir()
            cache, _, _ = make_stage6_fixture(stage6)
            with mock.patch.multiple(analysis, STAGE6_DIR=stage6,
                                     read_feature_cache=mock.Mock(return_value=cache),
                                     N_REF=60, N_LANDMARKS=24, N_COMPONENTS=3,
                                     RADIUS_BUDGET=8, KERNEL_BACKEND="numpy_cpu",
                                     CPU_THREADS=2, MAKE_PLOTS=True, TRANSFORM_BLOCK=20):
                summary = analysis.main(root / "analysis")
                frame = analysis.load_reference_frame(root / "analysis" / "reference_frame.npz")
                ref = analysis.load_projection_bundle(root / "analysis" / "reference_sample.npz", frame)
                sample = analysis.load_projection_bundle(root / "analysis" / "topology_sample.npz", frame)
                self.assertEqual(summary["ref_sample_count"], 60)
                self.assertEqual(summary["topology_representative_count"], 8)
                self.assertEqual(summary["covered_ref_count"]+summary["uncovered_ref_count"], 60)
                self.assertFalse(summary["hdr_filter_applied"])
                self.assertEqual(ref["raw_embeddings"].shape, (60, 40))
                self.assertEqual(sample["raw_embeddings"].shape, (8, 40))
                projected = analysis.transform_raw_embeddings(sample["raw_embeddings"], frame)
                np.testing.assert_allclose(projected, sample["kpca_coordinates"], atol=1e-10)
                with (root / "analysis" / "uncovered_kpca_gap_patterns.csv").open() as fp:
                    gap_rows = list(csv.DictReader(fp))
                self.assertEqual(len(gap_rows), summary["uncovered_ref_count"])
                self.assertTrue(all(row["cover_status"] == "gap" for row in gap_rows))
                for name in ("kpca_coverage_2d.png", "kpca_coverage_3d.png", "kpca_coverage_distance_cdf.png"):
                    self.assertGreater((root / "analysis" / name).stat().st_size, 1000)
                # Change B representatives: REF sample/frame/radius stay unchanged.
                write_representatives(stage6, cache["keys"], offset=4)
                with mock.patch.object(analysis, "MAKE_PLOTS", False):
                    rerun = analysis.main(root / "second-analysis")
                self.assertEqual(summary["frame_id"], rerun["frame_id"])
                self.assertEqual(summary["radius"], rerun["radius"])
                self.assertNotEqual(summary["stage6_representative_fingerprint"],
                                    rerun["stage6_representative_fingerprint"])

    def test_run_logging_success_and_failure_preserve_latest_success(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "OUT_DIR", Path(temp)):
            with mock.patch.object(analysis, "main", return_value={"frame_id":"success"}):
                self.assertEqual(analysis.run_with_log(), 0)
            with (Path(temp) / "latest_run.json").open() as fp:
                pointer = json.load(fp)
            with mock.patch.object(analysis, "main", side_effect=RuntimeError("intentional failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(analysis.run_with_log(), 1)
            with (Path(temp) / "latest_run.json").open() as fp:
                self.assertEqual(json.load(fp), pointer)
            logs = list(Path(temp).glob("run_*/analysis.log"))
            self.assertEqual(len(logs), 2)
            self.assertTrue(any("intentional failure" in p.read_text() for p in logs))

    def test_cuda_kernel_matches_cpu_when_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA device is unavailable")
        rng = np.random.default_rng(0)
        x, y = rng.normal(size=(12, 40)), rng.normal(size=(9, 40))
        np.testing.assert_allclose(analysis.rbf_kernel(x, y, .1, "torch_gpu"),
                                   analysis.rbf_kernel(x, y, .1), rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
