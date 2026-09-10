"""Run: python -m unittest discover -s coverage_analysis -p test_coverage_influence.py -v"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import coverage_domain_kpca as pipeline
from coverage_influence_report import analyze, heatmap_bins, KP_COLS
from group_reinforce_nnvote import nearest_votes, group_labels


class CoverageInfluenceCheck(unittest.TestCase):
    def setUp(self):
        self.ref = pd.DataFrame([[0, 0, 0], [0, 0, 0], [1, 0, 0], [2, 1, 0], [6, 3, 2]], columns=KP_COLS)
        self.sample = pd.DataFrame([[0, 0, 0], [0, 0, 0], [2, 1, 0], [20, 20, 20]], columns=KP_COLS)
        self.sample["gauge_name"] = ["A_1", "B_1", "C_1", "D_1"]

    def test_votes_coverage_ablation_and_bins_against_brute_force(self):
        result = analyze(self.ref, self.sample, .1, topk=3)
        scale = self.ref[KP_COLS].to_numpy().std(axis=0)
        raw_d = np.linalg.norm((self.ref[KP_COLS].to_numpy()[:, None] - self.sample[KP_COLS].to_numpy()) / scale, axis=2)
        hit = raw_d <= .1
        self.assertTrue(np.array_equal(result["hits"], hit))
        np.testing.assert_allclose(result["votes"].sum(axis=1), 1)
        self.assertAlmostEqual(result["ranking"].total_score.sum(), len(self.ref))
        np.testing.assert_allclose(result["samples"].influence_score.sum(), len(self.ref))
        for i, group in enumerate(result["groups"]):
            row = result["ranking"].set_index("group").loc[group]
            loss = hit.any(axis=1).mean() - np.delete(hit, i, axis=1).any(axis=1).mean()
            self.assertAlmostEqual(row.unique_loss_pp, loss*100)
        ranks = result["ranking"].set_index("group")
        self.assertEqual(ranks.loc["A", "unique_loss_pp"], 0)
        self.assertEqual(ranks.loc["C", "unique_loss_pp"], 20)
        self.assertEqual(ranks.loc["D", "total_score"], 0)
        self.assertAlmostEqual(result["ranking"].gain_in_vote_order_pp.sum(), result["summary"]["coverage_pct"])
        for bins in (2, 7):
            for grid in heatmap_bins(result, bins):
                n = grid["target_count"]
                self.assertEqual(n.sum(), len(self.ref))
                self.assertTrue(np.isnan(grid["coverage"][n == 0]).all())
                self.assertAlmostEqual(np.nansum(grid["coverage"] * n), hit.any(axis=1).sum())
                np.testing.assert_allclose(np.nansum(np.stack(grid["group_vote"]), axis=0)[n > 0], 1)

    def test_full_dimension_decision_and_target_scale(self):
        sample = self.sample.iloc[[0]].copy()
        ref = pd.DataFrame([[0, 0, 0], [0, 0, 5]], columns=KP_COLS)
        result = analyze(ref, sample, 0)
        self.assertEqual(result["summary"]["coverage_pct"], 50)
        self.assertEqual(heatmap_bins(result, 2)[0]["coverage"][1, 1], .5)
        subset = analyze(self.ref, self.sample, .1, target=self.ref.iloc[[0, 1]])
        np.testing.assert_allclose(subset["summary"]["ystd"], self.ref[KP_COLS].std(ddof=0))
        self.assertEqual(subset["summary"]["n_target"], 2)
        ref4 = pd.DataFrame([[0, 0, 0, 0], [0, 0, 0, 5]], columns=KP_COLS+["KP4"])
        sample4 = ref4.iloc[[0]].assign(gauge_name="A_0")
        result4 = analyze(ref4, sample4, 0, kp_cols=KP_COLS+["KP4"])
        self.assertEqual(result4["summary"]["coverage_pct"], 50)

    def test_frame_invalid_empty_and_group_partition(self):
        ref, sample = self.ref.assign(kpca_frame_id="frame1"), self.sample.assign(kpca_frame_id="frame2")
        with self.assertRaisesRegex(ValueError, "same kpca_frame_id"):
            analyze(ref, sample, 1)
        for bad in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                analyze(self.ref, self.sample, bad)
        with self.assertRaisesRegex(ValueError, "subset"):
            analyze(self.ref, self.sample, .1, target=self.ref.iloc[[0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "No target"):
            analyze(self.ref.assign(cover_status="out_of_domain"), self.sample, .1)
        d, idx, weights = nearest_votes([[0, 0]], [[0, 0]], topk=10)
        self.assertEqual(d.shape, (1, 1))
        self.assertEqual(weights[0, 0], 1)
        with self.assertRaises(ValueError):
            nearest_votes([[0, 0]], [[0, 0]], topk=0)
        np.testing.assert_array_equal(group_labels(pd.Series(["AB_x", "B_y", "C_z"]), ["A", "B"], "_"), ["A", "B", "other"])
        dirty = pd.concat([self.sample, pd.DataFrame([[np.nan, 0, 0, "E_bad"]], columns=self.sample.columns)], ignore_index=True)
        clean = analyze(self.ref, dirty, .1)
        self.assertEqual(clean["summary"]["dropped_nonfinite"]["sample"], 1)

    def test_end_to_end_kpca_pipeline_and_offline_report(self):
        rng = np.random.default_rng(13)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref = pd.DataFrame(rng.normal(size=(50, 3)), columns=["f1", "f2", "f3"])
            sample = ref.iloc[:8].copy()
            sample["gauge_name"] = ["A_x"]*4 + ["<B>_y"]*4
            ref.to_csv(root / "reference.csv", index=False)
            sample.to_csv(root / "sample.csv", index=False)
            with patch.multiple(pipeline, REF_PATH=str(root / "reference.csv"), SAMPLE_PATH=str(root / "sample.csv"),
                    OUTPUT_DIR=str(root / "out"), KEEP_COLS=[], FEATURE_COLS=["f1", "f2", "f3"],
                    N_REF=50, N_LANDMARKS=25, USE_CACHE=False, HEATMAP_BINS=4, N_GAP_GROUPS=2), redirect_stdout(io.StringIO()):
                summary = pipeline.run()
            output = root / "out" / "influence"
            influence = json.loads((output / "coverage_influence_summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(influence["coverage_pct"], summary["kpca_true_coverage_pct"], places=2)
            self.assertEqual(influence["radius"], summary["repr_radius_kpca"])
            self.assertEqual(influence["frame_id"], summary["kpca_frame_id"])
            html = (output / "coverage_influence_report.html").read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("&lt;B&gt;", html)
            self.assertNotIn('src="http', html)
            self.assertNotIn("<B>", html)
            self.assertTrue((output / "coverage_influence_bins.csv").is_file())


if __name__ == "__main__":
    unittest.main()
