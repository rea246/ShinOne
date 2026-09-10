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

from test_analysis import make_gap_fixture, make_stage6_fixture


SPEC = importlib.util.spec_from_file_location("compare_coverage", Path(__file__).with_name("8.compare_coverage.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def comparison_fixture():
    points = [[0, 0, 0], [.1, .1, .1], [.2, .2, .2], [1, 1, 0], [1, 1, .1], [0, 1, 1]]
    reference, _ = make_gap_fixture(points, .5)
    a, b = np.array([.2, 2, 2, .1, .2, .2]), np.array([2, 2, .2, 2, 2, .2])
    values = compare.classify_coverage(a, b, .5)
    values.update(distance_A=a, distance_B=b,
        nearest_A_pattern_keys=np.full(6, "A-key"), nearest_B_pattern_keys=np.full(6, "B-key"),
        nearest_A_global_rows=np.full(6, 100), nearest_B_global_rows=np.full(6, 200),
        is_A_member=np.zeros(6, dtype=bool), is_B_member=np.zeros(6, dtype=bool))
    return reference, values


class CompareCoverageTest(unittest.TestCase):
    def setUp(self):
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def test_b_keys_preserve_exact_names_and_audit_duplicates_missing_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "B.txt"
            path.write_text("\ufeffpattern_key\n0001\nGauge_A\n\n0001\n", encoding="utf-8")
            keys, counts, content = compare.read_b_keys(path)
            self.assertEqual(keys, ["0001", "Gauge_A"])
            self.assertEqual(counts, {"0001": 2, "Gauge_A": 1})
            self.assertEqual(content, path.read_bytes())
            report = Path(temp) / "lookup.csv"
            rows = compare.lookup_b_rows(["unused", "Gauge_A", "0001"], keys, counts, report)
            np.testing.assert_array_equal(rows, [2, 1])
            with report.open() as fp:
                audit = list(csv.DictReader(fp))
            self.assertEqual(audit[0]["input_occurrences"], "2")
            with self.assertRaisesRegex(ValueError, "missing"):
                compare.lookup_b_rows(["gauge_a", "0001"], keys, counts, report)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                compare.lookup_b_rows(["Gauge_A", "0001", "0001"], keys, counts, report)
            with report.open() as fp:
                self.assertEqual(next(csv.DictReader(fp))["status"], "ambiguous")
            path.write_text("pattern_key\n\n")
            with self.assertRaisesRegex(ValueError, "no pattern keys"):
                compare.read_b_keys(path)

    def test_four_states_are_disjoint_directional_and_include_radius_boundary(self):
        result = compare.classify_coverage([.5, .5, 1., 1.], [.5, 1., .5, 1.], .5)
        for i, name in enumerate(compare.STATES):
            np.testing.assert_array_equal(result[name], np.arange(4) == i)
        swapped = compare.classify_coverage([.5, 1., .5, 1.], [.5, .5, 1., 1.], .5)
        np.testing.assert_array_equal(result["A_only"], swapped["B_only"])
        zero = compare.classify_coverage([0, 1, 0], [0, 0, 1], 0)
        np.testing.assert_array_equal(zero["A_only"], [False, False, True])
        np.testing.assert_array_equal(sum(result[name] for name in compare.STATES), np.ones(4))

    def test_bins_use_all_ref_denominator_and_diagnostics_exclude_mixed_bins(self):
        reference, values = comparison_fixture()
        grids = compare.comparison_heatmap_grids(reference, values, 2)
        first = grids[0]
        self.assertEqual(first["ref_counts"][0, 0], 3)
        self.assertAlmostEqual(first["A_only_fraction"][0, 0], 1/3)  # Not 1/1 A-covered REF.
        self.assertEqual(first["A_only_fraction"][1, 1], 1.)
        self.assertTrue(np.isnan(first["A_only_fraction"][1, 0]))
        for grid in grids:
            np.testing.assert_array_equal(sum(grid[name+"_count"] for name in compare.STATES), grid["ref_counts"])
            fractions = sum(grid[name+"_fraction"] for name in compare.STATES)
            np.testing.assert_allclose(fractions[grid["ref_counts"] > 0], 1.)
        before = compare.stage7.arrays_fingerprint(reference)
        diagnostics = compare.select_a_only_representatives(reference, values, .5, 2, 1.)
        self.assertEqual(diagnostics["summary"]["A_only_ref_count"], 3)
        self.assertEqual(diagnostics["summary"]["eligible_ref_count"], 2)
        self.assertEqual(diagnostics["summary"]["excluded_mixed_bin_A_only_ref_count"], 1)
        np.testing.assert_array_equal(diagnostics["eligible_indices"], [3, 4])
        self.assertEqual(diagnostics["summary"]["representative_count"], 1)
        self.assertLessEqual(diagnostics["member_distances"].max(), .5)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            compare.write_diagnostics(root, reference, values, diagnostics)
            with (root / "a_only_members.csv").open() as fp:
                members = list(csv.DictReader(fp))
            self.assertEqual(sorted(int(row["global_row"]) for row in members), [13, 14])
            with np.load(root / "a_only_representatives.npz", allow_pickle=False) as saved:
                self.assertEqual(saved["raw_embeddings"].shape, (1, 40))
                self.assertTrue(saved["covered_by_A"].all())
                self.assertFalse(saved["covered_by_B"].any())
            compare.write_bin_statistics(root / "bins.csv", grids)
            with (root / "bins.csv").open() as fp:
                bins = list(csv.DictReader(fp))
            empty = [row for row in bins if row["ref_count"] == "0"]
            self.assertTrue(empty)
            self.assertTrue(all(row["A_only_fraction"] == "" for row in empty))
        self.assertEqual(before, compare.stage7.arrays_fingerprint(reference))

    def test_all_mixed_or_no_a_only_bins_produce_empty_valid_diagnostics(self):
        reference, values = comparison_fixture()
        # Make the previously pure cells mixed: one A-only plus one neither.
        values["distance_A"][4] = 2.
        values.update(compare.classify_coverage(values["distance_A"], values["distance_B"], .5))
        diagnostics = compare.select_a_only_representatives(reference, values, .5, 2, 1.)
        self.assertEqual(diagnostics["summary"]["A_only_ref_count"], 2)
        self.assertEqual(diagnostics["summary"]["eligible_ref_count"], 0)
        with tempfile.TemporaryDirectory() as temp:
            compare.write_diagnostics(Path(temp), reference, values, diagnostics)
            with (Path(temp) / "a_only_representatives.csv").open() as fp:
                self.assertEqual(list(csv.DictReader(fp)), [])
        values.update(compare.classify_coverage(values["distance_A"], values["distance_A"], .5))
        empty = compare.select_a_only_representatives(reference, values, .5, 2, 1.)
        self.assertEqual(empty["summary"]["A_only_ref_count"], 0)
        self.assertEqual(empty["summary"]["representative_count"], 0)

    def test_representative_markers_appear_only_in_qualifying_axis_pairs(self):
        from matplotlib.axes import Axes
        reference, _ = make_gap_fixture([[0., 0, 0], [0., 0, 1]], .5)
        values = compare.classify_coverage([.1, .1], [.1, 1.], .5)
        values.update(distance_A=np.array([.1, .1]), distance_B=np.array([.1, 1.]),
            nearest_A_pattern_keys=np.full(2, "A"), nearest_B_pattern_keys=np.full(2, "B"),
            nearest_A_global_rows=np.full(2, 1), nearest_B_global_rows=np.full(2, 2),
            is_A_member=np.zeros(2, dtype=bool), is_B_member=np.zeros(2, dtype=bool))
        diagnostics = compare.select_a_only_representatives(reference, values, .5, 2, 1.)
        grids = compare.comparison_heatmap_grids(reference, values, 2)
        calls, scatter = [], Axes.scatter

        def capture(ax, x, y, *args, **kwargs):
            calls.append((np.asarray(x).copy(), np.asarray(y).copy()))
            return scatter(ax, x, y, *args, **kwargs)

        with tempfile.TemporaryDirectory() as temp, mock.patch.object(Axes, "scatter", capture):
            compare.plot_comparison(Path(temp), reference, values, grids, diagnostics,
                {"A_count": 1, "B_count": 1, "radius": .5, "kpc_components": 3, "gap_preview_groups": 20})
        self.assertEqual([len(x) for x, y in calls], [0, 1, 1])
        for x, y in calls[1:]:
            np.testing.assert_array_equal(x, [0.])
            np.testing.assert_array_equal(y, [1.])

    def test_full_pipeline_preserves_stage7_and_replots_without_cache_or_new_distances(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage6 = root / "stage6"
            stage6.mkdir()
            cache, _, _ = make_stage6_fixture(stage6)
            source = root / "stage7"
            with mock.patch.multiple(compare.stage7, STAGE6_DIR=stage6, N_REF=60, N_LANDMARKS=24,
                    N_COMPONENTS=3, RADIUS_BUDGET=8, MAKE_PLOTS=False, CPU_THREADS=2,
                    KERNEL_BACKEND="numpy_cpu", read_feature_cache=mock.Mock(return_value=cache)):
                baseline = compare.stage7.main(source)
            frame = compare.stage7.load_reference_frame(source / "reference_frame.npz")
            reference = compare.stage7.load_projection_bundle(source / "reference_sample.npz", frame)
            sample_a = compare.stage7.load_projection_bundle(source / "topology_sample.npz", frame)
            original = {path: path.read_bytes() for path in source.iterdir() if path.is_file()}
            unseen = np.setdiff1d(np.arange(80), np.union1d(reference["global_rows"], sample_a["global_rows"]))
            b_rows = np.r_[sample_a["global_rows"][:2], unseen[:10]]
            path = root / "B.txt"
            path.write_text("pattern_key\n"+"\n".join(cache["keys"][i] for i in np.r_[b_rows, b_rows[0]])+"\n")
            output = root / "stage8"
            with mock.patch.multiple(compare.stage7,
                    read_feature_cache=mock.Mock(return_value=cache),
                    fit_reference_frame=mock.Mock(side_effect=AssertionError("must not fit")),
                    sample_reference_rows=mock.Mock(side_effect=AssertionError("must not sample")),
                    representation_radius=mock.Mock(side_effect=AssertionError("must not change R"))):
                with mock.patch.multiple(compare, MAKE_PLOTS=True, CPU_THREADS=2, KERNEL_BACKEND="numpy_cpu"):
                    self.assertEqual(compare.run_with_log(path, source, root/"cache.pt", output), 0)
                pointer = json.loads((output / "latest_run.json").read_text())
                run = output / pointer["run_directory"]
                summary = json.loads((run / "comparison_summary.json").read_text())
                self.assertEqual(summary["B_count"], len(b_rows))
                self.assertEqual(summary["B_duplicate_count"], 1)
                self.assertEqual(summary["B_projection"]["reused_A_rows"], 2)
                self.assertEqual(summary["B_projection"]["transformed_rows"], len(b_rows)-2)
                self.assertEqual(summary["frame_id"], baseline["frame_id"])
                self.assertEqual(summary["radius"], baseline["radius"])
                self.assertEqual(summary["outcomes"]["coverage_A"], baseline["coverage_fraction"])
                self.assertEqual(sum(summary["outcomes"]["counts"].values()), 60)
                saved = compare.stage7.load_projection_bundle(run / "reference_comparison.npz", frame)
                b = compare.stage7.load_projection_bundle(run / "b_sample.npz", frame)
                np.testing.assert_array_equal(b["global_rows"], b_rows)
                np.testing.assert_array_equal(b["raw_embeddings"], compare.stage7.gather_raw_embeddings(cache["features"], b_rows))
                np.testing.assert_array_equal(saved["covered_by_A"], reference["covered_by_topology"])
                self.assertEqual(summary["nonmember_outcomes"]["ref_count"], int((~saved["excluded_from_nonmember_comparison"]).sum()))
                for name, mask in (("a_only_ref_patterns.csv", saved["A_only"]),
                                   ("b_uncovered_ref_patterns.csv", ~saved["covered_by_B"])):
                    with (run / name).open() as fp:
                        self.assertEqual(len(list(csv.DictReader(fp))), int(mask.sum()))
                self.assertEqual((run / "comparison_report.html").read_text().count('src="data:image/png;base64,'), 4)
                saved_paths = [run / name for name in ("reference_frame.npz", "reference_comparison.npz", "a_sample.npz",
                    "b_sample.npz", "comparison_summary.json", "ref_comparison.csv", "coverage_bins.csv")]
                saved_bytes = {p: p.read_bytes() for p in saved_paths}
                with mock.patch.multiple(compare.stage7,
                        read_feature_cache=mock.Mock(side_effect=AssertionError("replot must not load cache")),
                        nearest_distances=mock.Mock(side_effect=AssertionError("replot must not query NN")),
                        transform_raw_embeddings=mock.Mock(side_effect=AssertionError("replot must not transform"))):
                    self.assertEqual(compare.replot_saved_run("latest", output), 0)
                for p, data in {**original, **saved_bytes}.items():
                    self.assertEqual(p.read_bytes(), data)
                # Exact A=B control, including a changed key order: no artificial exclusive regions.
                path.write_text("\n".join(sample_a["pattern_keys"][::-1])+"\n")
                with mock.patch.object(compare, "MAKE_PLOTS", False):
                    equal = compare.main(root/"equal", path, source, root/"cache.pt")
                self.assertEqual(equal["outcomes"]["counts"]["A_only"], 0)
                self.assertEqual(equal["outcomes"]["counts"]["B_only"], 0)
                # Failed lookup keeps the last successful run, with an audit and log.
                pointer_before = (output / "latest_run.json").read_bytes()
                path.write_text("missing-gauge\n")
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(compare.run_with_log(path, source, root/"cache.pt", output), 1)
                self.assertEqual((output / "latest_run.json").read_bytes(), pointer_before)
                # Same names under changed model values must fail the baseline compatibility check.
                row = int(reference["global_rows"][0])
                cache["features"]["h1"][row, 0] += 1.
                with self.assertRaisesRegex(ValueError, "embeddings differ from saved REF"):
                    compare.extract_b_embeddings(root/"cache.pt", [cache["keys"][b_rows[0]]],
                        {cache["keys"][b_rows[0]]: 1}, reference, sample_a, frame, root/"mismatch.csv")


if __name__ == "__main__":
    unittest.main()
