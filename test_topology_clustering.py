import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("6.topology_clustering.py")
SPEC = importlib.util.spec_from_file_location("topology_clustering", MODULE_PATH)
topology = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topology)


class TopologyClusteringTest(unittest.TestCase):
    def test_community_ids_are_deterministic_and_largest_first(self):
        labels = topology.renumber_communities([7, 2, 7, 9, 2, 7])
        np.testing.assert_array_equal(labels, [0, 1, 0, 2, 1, 0])

    def test_nonself_edges_survive_ties_and_missing_self(self):
        query = np.array([0, 1], dtype=np.int32)
        neighbors = np.array([[2, 0, 1], [0, 2, 3]], dtype=np.int32)
        distances = np.array([[0.0, 0.0, 1.0], [0.2, 0.3, 0.4]], dtype=np.float32)
        src, dst, _ = topology._nonself_edges(query, neighbors, distances, k=2)
        np.testing.assert_array_equal(src, [0, 0, 1, 1])
        np.testing.assert_array_equal(dst, [2, 1, 0, 2])

    def test_similarity_decreases_with_distance(self):
        distance = np.array([0.1, 1.0], dtype=np.float32)
        src = np.array([0, 0], dtype=np.int32)
        dst = np.array([1, 1], dtype=np.int32)
        scale = np.ones(2, dtype=np.float32)
        for mode in ("local_gaussian", "inverse_distance"):
            weights = topology.edge_weights(mode, distance, src, dst, scale)
            self.assertGreater(weights[0], weights[1])
        np.testing.assert_array_equal(
            topology.edge_weights("binary", distance, src, dst, scale), [1.0, 1.0]
        )

    def test_final_cluster_names_preserve_rare(self):
        self.assertEqual(topology.final_cluster_id(3, 2), "H0_3_T2")
        self.assertEqual(topology.final_cluster_id(-1, 2), "RARE_T2")

    def test_outputs_preserve_rows_labels_and_schema(self):
        summaries = topology._community_summary(-1, np.array([0, 1]), total_count=2)
        with tempfile.TemporaryDirectory() as out_dir:
            original_out_dir = topology.OUT_DIR
            topology.OUT_DIR = out_dir
            try:
                topology.write_outputs(
                    keys=["a", "b"], rows=np.array([0, 1]),
                    h0_labels=np.array([-1, -1]), topology_labels=np.array([0, 1]),
                    summaries=summaries, diagnostics=[]
                )
                with np.load(Path(out_dir, "topology_labels.npz")) as labels:
                    np.testing.assert_array_equal(labels["rows"], [0, 1])
                    np.testing.assert_array_equal(labels["h0_labels"], [-1, -1])
                header = Path(out_dir, "topology_assignments.csv").read_text(
                    encoding="utf-8"
                ).splitlines()[0]
                self.assertEqual(
                    header,
                    "pattern_key,original_h0_label,coarse_group,topology_cluster,final_cluster",
                )
            finally:
                topology.OUT_DIR = original_out_dir


if __name__ == "__main__":
    unittest.main()
