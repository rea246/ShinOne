import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
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


def make_gap_fixture(points, radius):
    points = np.asarray(points, dtype=float)
    rows = np.arange(10, 10+len(points))
    raw = np.pad(points, ((0, 0), (0, 40-points.shape[1])))
    distances = np.linalg.norm(points-points[0], axis=1)
    reference = {
        "frame_id": np.asarray("test-frame"), "global_rows": rows,
        "pattern_keys": np.asarray([f"p-{i}" for i in rows]),
        "h0_labels": np.zeros(len(rows), dtype=int), "topology_labels": np.zeros(len(rows), dtype=int),
        "raw_embeddings": raw, "normalized_embeddings": raw,
        "kpca_coordinates": points, "distance_coordinates": points,
        "feature_names": np.asarray([f"feature-{i}" for i in range(40)]),
        "nearest_topology_distances": distances, "covered_by_topology": distances <= radius,
        "nearest_topology_global_rows": np.full(len(rows), rows[0]),
    }
    sample = {"frame_id": reference["frame_id"], "global_rows": rows[:1],
              "pattern_keys": reference["pattern_keys"][:1]}
    return reference, sample


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

    def test_heatmaps_preserve_counts_full_dimension_labels_and_empty_cells(self):
        # Same 2D position, different third coordinates and original 3D decisions.
        points = np.array([[0, 0, 0], [0, 0, 4], [0, 0, 2], [1, 1, 1], [2, 2, 2]])
        coverage = {"covered": np.array([True, False, True, False, True]),
                    "distances": np.array([.1, 2.0, .2, 1.0, .3])}
        grids = analysis.coverage_heatmap_grids(points, coverage, bins=2)
        self.assertEqual([grid["pair"] for grid in grids], [(0, 1), (0, 2), (1, 2)])
        for grid in grids:
            self.assertEqual(grid["ref_counts"].sum(), 5)
            self.assertEqual(grid["gap_counts"].sum(), 2)
            self.assertAlmostEqual(np.nansum(grid["mean_distance"]*grid["ref_counts"]), 3.6)
            reconstructed = np.zeros((2, 2), dtype=int)
            np.add.at(reconstructed, tuple(grid["ref_bin_indices"].T), 1)
            np.testing.assert_array_equal(reconstructed, grid["ref_counts"])
        first = grids[0]
        self.assertEqual(first["ref_counts"][0, 0], 3)
        self.assertAlmostEqual(first["gap_fraction"][0, 0], 1/3)
        self.assertEqual(first["ref_counts"][1, 1], 2)  # Includes the maximum-edge point.
        self.assertEqual(first["gap_fraction"][1, 1], .5)
        self.assertTrue(np.isnan(first["gap_fraction"][0, 1]))
        self.assertTrue(np.isnan(first["mean_distance"][0, 1]))
        changed_labels = {**coverage, "covered": ~coverage["covered"]}
        changed = analysis.coverage_heatmap_grids(points, changed_labels, bins=2)
        for before, after in zip(grids, changed):
            np.testing.assert_array_equal(before["x_edges"], after["x_edges"])
            np.testing.assert_allclose(before["mean_distance"], after["mean_distance"], equal_nan=True)

    def test_heatmap_plots_handle_constant_axes_and_all_covered_or_all_gap(self):
        reference = {"kpca_coordinates": np.zeros((4, 2))}
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "HEATMAP_BINS", 4):
            for covered in (True, False):
                coverage = {"covered": np.full(4, covered, dtype=bool),
                            "distances": np.full(4, 0.0 if covered else 1.0)}
                analysis.plot_coverage_heatmaps(Path(temp), reference, coverage, .5)
                for name in ("kpca_coverage_2d_heatmap.png", "kpca_nearest_distance_2d_heatmap.png"):
                    self.assertGreater((Path(temp) / name).stat().st_size, 1000)

    def test_full_gap_candidates_use_any_pair_union_with_exact_bin_edges(self):
        points = np.array([[0., 0, 0], [.1, .1, .1], [0., 0, 1],
                           [1., 1, 0], [.5, .5, 0], [1., 1, 0]])
        distances = np.linalg.norm(points, axis=1)
        coverage = {"covered": distances <= .05, "distances": distances}
        result = analysis.fully_uncovered_bin_candidates(points, coverage, 2)
        np.testing.assert_array_equal(result["eligible_mask"], [False, False, True, True, True, True])
        # The hidden-axis gap is mixed in KP1/KP2 but eligible in the other pairs.
        self.assertEqual(result["bin_ids_by_ref"][2, 0], "")
        self.assertTrue((result["bin_ids_by_ref"][2, 1:] != "").all())
        first_pair = {b["bin_id"]: b for b in result["bins"] if b["pair"] == "KP1/KP2"}
        self.assertEqual(len(first_pair), 1)
        bin_id = result["bin_ids_by_ref"][3, 0]
        self.assertEqual(result["bin_ids_by_ref"][4, 0], bin_id)  # Internal edge goes right.
        self.assertEqual(first_pair[bin_id]["ref_count"], 3)  # Duplicate/max-edge points retained.
        self.assertTrue(first_pair[bin_id]["x_upper_inclusive"])
        self.assertTrue(all(b["ref_count"] == b["gap_count"] > 0 for b in result["bins"]))
        perm = np.array([5, 2, 0, 4, 1, 3])
        shuffled = analysis.fully_uncovered_bin_candidates(
            points[perm], {name: value[perm] for name, value in coverage.items()}, 2)
        np.testing.assert_array_equal(shuffled["bin_ids_by_ref"], result["bin_ids_by_ref"][perm])
        self.assertEqual(shuffled["bins"], result["bins"])
        # The candidate filter must also reach the exported real representatives
        # and member list, while the original five gap decisions remain intact.
        ref, sample = make_gap_fixture(points, .05)
        before = analysis.arrays_fingerprint(ref)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "HEATMAP_BINS", 2):
            diagnostics = analysis.summarize_uncovered_reference(ref, sample, .05)
            summary = diagnostics["summary"]
            self.assertEqual(summary["uncovered_ref_count"], 5)
            self.assertEqual(summary["eligible_ref_count"], 4)
            self.assertEqual(summary["excluded_mixed_bin_uncovered_ref_count"], 1)
            self.assertAlmostEqual(sum(g["fraction_of_uncovered_ref"] for g in summary["groups"]), .8)
            self.assertAlmostEqual(sum(g["fraction_of_eligible_ref"] for g in summary["groups"]), 1.)
            analysis.write_gap_diagnostics(temp, ref, diagnostics)
            with (Path(temp)/"uncovered_gap_members.csv").open() as fp:
                members = list(csv.DictReader(fp))
            self.assertEqual(sorted(int(row["global_row"]) for row in members), [12, 13, 14, 15])
            self.assertTrue(all(row["fully_uncovered_bin_ids"] for row in members))
            with (Path(temp)/"uncovered_gap_bins.csv").open() as fp:
                exported_bins = {row["bin_id"]: row for row in csv.DictReader(fp)}
            for row in members:
                for bin_id in row["fully_uncovered_bin_ids"].split(";"):
                    self.assertEqual(exported_bins[bin_id]["ref_count"], exported_bins[bin_id]["gap_count"])
            self.assertEqual(analysis.arrays_fingerprint(ref), before)

    def test_full_gap_bins_reject_rounded_100_percent_and_keep_singletons(self):
        points = np.full((10_001, 3), .01)
        points[0], points[-1] = 0., 1.
        distances = np.linalg.norm(points, axis=1)
        result = analysis.fully_uncovered_bin_candidates(
            points, {"covered": distances <= .001, "distances": distances}, 2)
        self.assertEqual(result["eligible_mask"].sum(), 1)
        self.assertTrue(result["eligible_mask"][-1])
        self.assertEqual(len(result["bins"]), 3)
        self.assertTrue(all(b["ref_count"] == 1 for b in result["bins"]))

    def test_gap_candidates_empty_with_only_mixed_bins_and_when_all_covered(self):
        ref, sample = make_gap_fixture([[0., 0, 0], [.01, .01, .01], [1., 1, 1]], .001)
        # Add a covered point in every occupied bin, leaving one real gap.
        sample = {"frame_id": ref["frame_id"], "global_rows": ref["global_rows"][[0, -1]],
                  "pattern_keys": ref["pattern_keys"][[0, -1]]}
        ref["covered_by_topology"][-1] = True
        ref["nearest_topology_distances"][-1] = 0.
        ref["nearest_topology_global_rows"][-1] = ref["global_rows"][-1]
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "HEATMAP_BINS", 2):
            result = analysis.summarize_uncovered_reference(ref, sample, .001)
            self.assertEqual(result["summary"]["uncovered_ref_count"], 1)
            self.assertEqual(result["summary"]["eligible_ref_count"], 0)
            self.assertEqual(result["summary"]["diagnostic_representative_count"], 0)
            analysis.write_gap_diagnostics(temp, ref, result)
            analysis.plot_gap_diagnostics(temp, ref, result)
            analysis.write_html_report(temp, {"frame_id": "test-frame"}, result["summary"])
            report = (Path(temp)/"analysis_report.html").read_text()
            self.assertIn("미커버 REF는 있지만 100% 미커버 bin이 없어", report)
            with (Path(temp)/"uncovered_gap_bins.csv").open() as fp:
                self.assertEqual(list(csv.DictReader(fp)), [])
            with np.load(Path(temp)/"uncovered_gap_representatives.npz", allow_pickle=False) as bundle:
                self.assertEqual(bundle["fully_uncovered_bin_ids"].shape, (0, 3))
            ref["covered_by_topology"][:] = True
            ref["nearest_topology_distances"][:] = 0.
            empty = analysis.summarize_uncovered_reference(ref, sample, 0.)
            self.assertEqual(empty["summary"]["uncovered_ref_count"], 0)
            self.assertEqual(empty["summary"]["fully_uncovered_bin_count"], 0)

    def test_gap_plot_marks_only_the_100_percent_pairs_of_each_representative(self):
        import matplotlib.pyplot as plt
        ref, sample = make_gap_fixture([[0., 0, 0], [0., 0, 1]], .1)
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(analysis, "HEATMAP_BINS", 2):
            result = analysis.summarize_uncovered_reference(ref, sample, .1)
            with mock.patch.object(plt, "close"):
                analysis.plot_gap_diagnostics(temp, ref, result)
                fig = plt.gcf()
                self.assertEqual(len(fig.axes[0].collections[1].get_offsets()), 0)
                for ax in fig.axes[1:3]:
                    np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[0., 1.]])
            plt.close(fig)

    def test_gap_radius_cover_is_bounded_order_independent_and_keeps_duplicates(self):
        from scipy.spatial.distance import cdist
        points = np.array([[0.0], [1.0], [1.0], [4.0], [5.0]])
        rows = np.array([7, 2, 13, 6, 10])
        existing = np.array([2.0, 3.0, 3.0, 6.0, 7.0])
        selected, labels, distances = analysis.radius_cover_representatives(points, rows, existing, 1.0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(labels), 5)
        self.assertLessEqual(distances.max(), 1.0)
        actual = cdist(points, points[selected])[np.arange(len(points)), labels]
        np.testing.assert_allclose(distances, actual)
        self.assertEqual(labels[1], labels[2])
        perm = np.array([4, 2, 0, 3, 1])
        other, other_labels, other_distances = analysis.radius_cover_representatives(
            points[perm], rows[perm], existing[perm], 1.0)
        np.testing.assert_array_equal(rows[selected], rows[perm][other])
        np.testing.assert_array_equal(labels[perm], other_labels)
        np.testing.assert_allclose(distances[perm], other_distances)

    def test_gap_groups_use_hidden_kpca_axes_without_unbounded_chaining(self):
        points = np.zeros((6, 3))
        points[:, 2] = np.arange(6) * .9
        selected, labels, distances = analysis.radius_cover_representatives(
            points, np.arange(6), np.arange(6)+2.0, 1.0)
        self.assertGreaterEqual(len(selected), 3)
        self.assertLessEqual(distances.max(), 1.0)
        self.assertEqual(len(np.unique(labels)), len(selected))

    def test_gap_radius_cover_handles_empty_zero_radius_and_ties(self):
        selected, labels, distances = analysis.radius_cover_representatives(
            np.empty((0, 3)), np.empty(0, dtype=int), np.empty(0), 0)
        self.assertEqual(len(selected)+len(labels)+len(distances), 0)
        points = np.array([[0.0, 0], [0.0, 0], [2.0, 0]])
        selected, labels, distances = analysis.radius_cover_representatives(
            points, np.array([10, 2, 5]), np.ones(3), 0)
        self.assertEqual(selected[0], 1)  # Equal existing distance: smallest global row.
        self.assertEqual(len(selected), 2)
        np.testing.assert_array_equal(distances, np.zeros(3))
        self.assertEqual(labels[0], labels[1])

    def test_gap_outputs_include_only_uncovered_real_patterns_and_handle_no_gaps(self):
        points = np.array([[0., 0, 0], [2., 0, 0], [2.2, 0, 0], [6., 0, 0], [6.2, 0, 0]])
        distances = points[:, 0]
        reference = {
            "frame_id": np.asarray("test-frame"), "global_rows": np.arange(10, 15),
            "pattern_keys": np.asarray([f"p-{i}" for i in range(10, 15)]),
            "h0_labels": np.array([0, 0, 0, -1, -1]), "topology_labels": np.zeros(5, dtype=int),
            "raw_embeddings": np.pad(points, ((0, 0), (0, 37))),
            "normalized_embeddings": np.pad(points, ((0, 0), (0, 37))),
            "kpca_coordinates": points, "distance_coordinates": points,
            "feature_names": np.asarray([f"feature-{i}" for i in range(40)]),
            "nearest_topology_distances": distances, "covered_by_topology": distances <= 1,
            "nearest_topology_global_rows": np.full(5, 10),
        }
        sample = {"frame_id":reference["frame_id"], "global_rows":np.array([10]),
                  "pattern_keys":np.asarray(["p-10"])}
        with tempfile.TemporaryDirectory() as temp:
            before = analysis.arrays_fingerprint(reference)
            diagnostics = analysis.summarize_uncovered_reference(reference, sample, 1.0)
            self.assertEqual(len(diagnostics["summary"]["groups"]), 2)
            self.assertEqual(sum(g["gap_ref_count"] for g in diagnostics["summary"]["groups"]), 4)
            self.assertTrue((distances[diagnostics["representative_indices"]] > 1).all())
            analysis.write_gap_diagnostics(Path(temp), reference, diagnostics)
            reps = analysis.load_projection_bundle(Path(temp)/"uncovered_gap_representatives.npz", sample)
            self.assertEqual(reps["raw_embeddings"].shape, (2, 40))
            self.assertNotIn(10, reps["global_rows"])
            self.assertFalse(reps["covered_by_topology"].any())
            self.assertTrue((reps["nearest_topology_distances"] > 1).all())
            self.assertTrue((reps["fully_uncovered_bin_ids"] != "").any(axis=1).all())
            with (Path(temp)/"uncovered_gap_members.csv").open() as fp:
                members = list(csv.DictReader(fp))
            self.assertEqual(len(members), 4)
            self.assertTrue(all(float(row["distance_to_gap_representative"]) <= 1 for row in members))
            self.assertEqual(analysis.arrays_fingerprint(reference), before)
            # A run with no uncovered REF still exports valid empty files/report.
            covered_reference = {**reference, "covered_by_topology":np.ones(5, dtype=bool)}
            empty = analysis.summarize_uncovered_reference(covered_reference, sample, 10.0)
            analysis.write_gap_diagnostics(Path(temp), covered_reference, empty)
            self.assertEqual(empty["summary"]["diagnostic_representative_count"], 0)
            with (Path(temp)/"uncovered_gap_representatives.csv").open() as fp:
                self.assertEqual(len(list(csv.DictReader(fp))), 0)
            analysis.plot_gap_diagnostics(Path(temp), covered_reference, empty)
            analysis.write_html_report(Path(temp), {"frame_id":"test-frame"}, empty["summary"])
            self.assertIn("현재 R에서 미커버 REF가 없습니다", (Path(temp)/"analysis_report.html").read_text())

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
                for name in ("kpca_coverage_2d.png", "kpca_coverage_3d.png", "kpca_coverage_distance_cdf.png",
                             "kpca_coverage_2d_heatmap.png", "kpca_nearest_distance_2d_heatmap.png",
                             "kpca_uncovered_representatives_2d.png"):
                    self.assertGreater((root / "analysis" / name).stat().st_size, 1000)
                report = (root / "analysis" / "analysis_report.html").read_text(encoding="utf-8")
                self.assertIn('src="data:image/png;base64,', report)
                self.assertIn(f'{summary["coverage_fraction"]:.2%}', report)
                self.assertIn(summary["frame_id"], report)
                self.assertEqual(report.count('src="data:image/png;base64,'), 6)
                with (root / "analysis" / "uncovered_gap_summary.json").open() as fp:
                    gaps = json.load(fp)
                self.assertEqual(gaps["uncovered_ref_count"], summary["uncovered_ref_count"])
                self.assertEqual(gaps["assigned_uncovered_ref_count"], gaps["eligible_ref_count"])
                self.assertEqual(sum(g["gap_ref_count"] for g in gaps["groups"]), gaps["eligible_ref_count"])
                self.assertEqual(gaps["eligible_ref_count"]+gaps["excluded_mixed_bin_uncovered_ref_count"],
                                 summary["uncovered_ref_count"])
                for group in gaps["groups"]:
                    self.assertIn(group["representative_pattern_key"], report)
                    self.assertGreater(group["representative_distance_to_existing"], summary["radius"])
                    self.assertLessEqual(group["max_member_distance_to_gap_representative"], summary["radius"])
                    self.assertTrue(group["representative_100pct_bins"])
                    self.assertTrue(all(b["gap_count"] == b["ref_count"] > 0
                                        for b in group["representative_100pct_bins"]))
                # An existing run can be viewed again without changing its analysis.
                data_paths = [root / "analysis" / name for name in
                              ("reference_frame.npz", "reference_sample.npz", "topology_sample.npz")]
                data_paths.append(root / "analysis" / "coverage_summary.json")
                data_paths.extend((root / "analysis").glob("kpca_*scatter.csv"))
                original_bytes = {path: path.read_bytes() for path in data_paths}
                analysis.write_json(root / "latest_run.json",
                                     {"run_directory": "analysis", "frame_id": summary["frame_id"]})
                with mock.patch.multiple(analysis, OUT_DIR=root,
                                         fit_reference_frame=mock.Mock(side_effect=AssertionError("replot must not fit")),
                                         nearest_distances=mock.Mock(side_effect=AssertionError("replot must not query NN"))):
                    with mock.patch.object(sys, "argv", ["7.analysis.py", "--replot", "latest"]):
                        self.assertEqual(analysis.cli(), 0)
                for path, original in original_bytes.items():
                    self.assertEqual(path.read_bytes(), original)
                self.assertIn("[Replot] DONE", (root / "analysis" / "replot.log").read_text())
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
