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

    def test_torch_exact_backend_matches_sklearn_across_blocks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed in this test environment")
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(123)
        embedding = rng.normal(size=(23, 7)).astype(np.float32)
        original_query_block = topology.TORCH_GPU_QUERY_BLOCK
        original_reference_block = topology.TORCH_GPU_REFERENCE_BLOCK
        topology.TORCH_GPU_QUERY_BLOCK = 4
        topology.TORCH_GPU_REFERENCE_BLOCK = 6
        try:
            backend = topology._TorchExactKNN(embedding, torch, device="cpu")
            actual_distances, actual_neighbors = backend.kneighbors(
                embedding, n_neighbors=6, return_distance=True
            )
        finally:
            topology.TORCH_GPU_QUERY_BLOCK = original_query_block
            topology.TORCH_GPU_REFERENCE_BLOCK = original_reference_block

        expected = NearestNeighbors(
            n_neighbors=6, metric="euclidean", algorithm="brute"
        ).fit(embedding)
        expected_distances, expected_neighbors = expected.kneighbors(embedding)
        np.testing.assert_array_equal(actual_neighbors, expected_neighbors)
        np.testing.assert_allclose(
            actual_distances, expected_distances, rtol=1e-5, atol=1e-5
        )

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

    def test_cugraph_labels_restore_vertex_order_and_isolates(self):
        labels = topology._labels_from_vertex_partitions(
            vertices=np.array([3, 0, 1]),
            partitions=np.array([8, 4, 4]),
            total_vertices=4,
        )
        np.testing.assert_array_equal(labels, [0, 0, 2, 1])

    def test_cugraph_adapter_passes_symmetric_csr_and_restores_labels(self):
        from scipy.sparse import csr_matrix

        adjacency = csr_matrix(
            np.array([
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ], dtype=np.float32)
        )

        class FakeSeries:
            def __init__(self, values):
                self.values = np.asarray(values)

            def to_numpy(self):
                return self.values

        class FakeCudf:
            Series = FakeSeries

        class FakeDevice:
            def __init__(self, index):
                self.index = index

            def use(self):
                self.used = True

        class FakeStream:
            def synchronize(self):
                return None

        class FakeCuda:
            Device = FakeDevice

            @staticmethod
            def get_current_stream():
                return FakeStream()

        class FakeCupy:
            cuda = FakeCuda()

        class FakeGraph:
            def __init__(self, directed):
                self.directed = directed

            def from_cudf_adjlist(
                self, offsets, indices, weights, renumber, symmetrize
            ):
                np.testing.assert_array_equal(offsets.values, [0, 1, 3, 4, 4])
                np.testing.assert_array_equal(indices.values, [1, 0, 2, 1])
                np.testing.assert_allclose(weights.values, [1.0, 1.0, 2.0, 2.0])
                self.options = (renumber, symmetrize)

        class FakeCugraph:
            Graph = FakeGraph

            @staticmethod
            def leiden(graph, max_iter, resolution, random_state):
                assert graph.options == (False, False)
                assert max_iter == topology.CUGRAPH_MAX_ITERATIONS
                assert resolution == topology.LEIDEN_RESOLUTION
                assert random_state == topology.RANDOM_SEED
                return {
                    "vertex": FakeSeries([2, 0, 1, 3]),
                    "partition": FakeSeries([5, 5, 5, 9]),
                }, 0.75

        labels = topology._run_cugraph_leiden(
            adjacency, FakeCugraph(), FakeCudf(), FakeCupy()
        )
        np.testing.assert_array_equal(labels, [0, 0, 0, 1])

    def test_community_summary_measures_normalized_space_dispersion(self):
        labels = np.array([0, 0, 1], dtype=np.int32)
        embedding = np.array([[0.0], [2.0], [10.0]], dtype=np.float32)
        summaries = topology._community_summary(
            0, labels, total_count=3, embedding=embedding
        )

        self.assertAlmostEqual(summaries[0]["centroid_distance_mean"], 1.0)
        self.assertAlmostEqual(summaries[0]["centroid_distance_p95"], 1.0)
        self.assertAlmostEqual(summaries[0]["centroid_distance_max"], 1.0)
        self.assertAlmostEqual(summaries[1]["centroid_distance_p95"], 0.0)

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
                    self.assertEqual(
                        labels["population_fingerprint"].item(),
                        topology.population_fingerprint(["a", "b"], [0, 1]),
                    )
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
