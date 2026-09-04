import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("6.topology_clustering.py")
SPEC = importlib.util.spec_from_file_location("topology_clustering", MODULE_PATH)
topology = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topology)


class TopologyClusteringTest(unittest.TestCase):
    def test_persistent_run_log_captures_console_output(self):
        observed_paths = []

        def fake_main(run_log_path=None):
            observed_paths.append(run_log_path)
            print("pipeline checkpoint", flush=True)

        with tempfile.TemporaryDirectory() as out_dir, mock.patch.multiple(
            topology,
            OUT_DIR=out_dir,
            main=mock.Mock(side_effect=fake_main),
        ):
            self.assertEqual(topology.run_with_persistent_log(), 0)
            self.assertEqual(len(observed_paths), 1)
            log_path = Path(observed_paths[0])
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("pipeline checkpoint", content)
            self.assertIn("[Run log] CLOSED exit_code=0", content)

    def test_stratified_sample_allocation_is_exact_and_retains_groups(self):
        sizes = np.array([2, 8, 90], dtype=np.int64)
        allocation = topology.allocate_stratified_sample_counts(sizes, 20)
        self.assertEqual(int(allocation.sum()), 20)
        self.assertTrue((allocation >= 1).all())
        self.assertTrue((allocation <= sizes).all())
        np.testing.assert_array_equal(
            topology.allocate_stratified_sample_counts(sizes, 100), sizes
        )

    def test_resolution_selection_accepts_symmetric_five_percent_band(self):
        resolution, in_tolerance = topology.choose_resolution(
            {0.5: 940, 0.7: 1_030, 1.0: 1_100},
            target=1_000,
            tolerance=0.05,
        )
        self.assertEqual(resolution, 0.7)
        self.assertTrue(in_tolerance)

    def test_resolution_search_expands_and_interpolates(self):
        expanded = topology.next_resolution_candidate(
            {0.5: 60, 1.0: 130},
            target=1_000,
            minimum_resolution=1.0e-4,
            maximum_resolution=4_096.0,
        )
        self.assertEqual(expanded, 2.0)

        interpolated = topology.next_resolution_candidate(
            {8.0: 800, 16.0: 1_200},
            target=1_000,
            minimum_resolution=1.0e-4,
            maximum_resolution=4_096.0,
        )
        self.assertGreater(interpolated, 8.0)
        self.assertLess(interpolated, 16.0)

    def test_majority_vote_is_deterministic_on_ties(self):
        labels = np.array([
            [2, 1, 2, 1, 3],
            [4, 4, 7, 8, 9],
        ], dtype=np.int32)
        np.testing.assert_array_equal(
            topology.majority_vote_labels(labels), [1, 4]
        )

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

    def test_integrated_representatives_are_real_centroid_nearest_members(self):
        labels = np.array([0, 0, 1, 1], dtype=np.int32)
        embedding = np.array([[0.0], [2.0], [10.0], [14.0]], dtype=np.float32)
        summaries, representatives = topology.summarize_and_select_representatives(
            h0_label=0,
            local_labels=labels,
            total_count=4,
            embedding=embedding,
            global_rows=np.arange(4, dtype=np.int64),
            keys=["p0", "p1", "p2", "p3"],
        )

        self.assertEqual(len(summaries), 2)
        self.assertEqual(
            [row["pattern_key"] for row in representatives], ["p0", "p2"]
        )
        self.assertTrue(all(
            row["selection_reason"] == "centroid_nearest_full_population"
            for row in representatives
        ))

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

    def test_integrated_outputs_include_representatives_and_metadata(self):
        labels = np.array([0, 0], dtype=np.int32)
        summaries = topology._community_summary(
            0, labels, total_count=2, embedding=np.array([[0.0], [2.0]])
        )
        representatives = [{
            "pattern_key": "a",
            "global_row": 0,
            "h0_label": 0,
            "topology_cluster": 0,
            "final_cluster": "H0_0_T0",
            "representative_rank": 1,
            "community_pattern_count": 2,
            "selection_reason": "centroid_nearest_full_population",
            "distance_to_centroid": 1.0,
        }]
        with tempfile.TemporaryDirectory() as out_dir:
            original_out_dir = topology.OUT_DIR
            topology.OUT_DIR = out_dir
            try:
                topology.write_outputs(
                    keys=["a", "b"],
                    rows=np.array([0, 1]),
                    h0_labels=np.array([0, 0]),
                    topology_labels=labels,
                    summaries=summaries,
                    diagnostics=[],
                    representatives=representatives,
                    metadata={"selected_resolution": 0.7},
                    sample_rows=np.array([0]),
                )
                self.assertTrue(
                    Path(out_dir, "topology_representatives.csv").exists()
                )
                self.assertTrue(
                    Path(out_dir, "topology_run_metadata.json").exists()
                )
                with np.load(Path(out_dir, "topology_labels.npz")) as saved:
                    np.testing.assert_array_equal(saved["sample_rows"], [0])
                    self.assertAlmostEqual(float(saved["selected_resolution"]), 0.7)
            finally:
                topology.OUT_DIR = original_out_dir

    def test_integrated_main_runs_sample_assign_and_representative_pipeline(self):
        from scipy.sparse import csr_matrix

        keys = [f"p{index}" for index in range(6)]
        rows = np.arange(6, dtype=np.int64)
        h0_labels = np.array([0, 0, 0, 0, -1, -1], dtype=np.int32)
        dims = {name: 1 for name in topology.FEATURE_BLOCKS}

        class FakeIndex:
            _fit_method = "fake_exact"

            def __init__(self, references):
                self.references = np.asarray(references, dtype=np.float32)

            def kneighbors(self, queries, n_neighbors, return_distance=True):
                queries = np.asarray(queries, dtype=np.float32)
                distances = np.abs(
                    queries[:, None, 0] - self.references[None, :, 0]
                )
                neighbors = np.argsort(distances, axis=1)[:, :n_neighbors]
                selected = np.take_along_axis(distances, neighbors, axis=1)
                return selected.astype(np.float32), neighbors.astype(np.int64)

        def fake_embedding(_features, _scalers, global_rows, _dims):
            return np.asarray(global_rows, dtype=np.float32)[:, None]

        def fake_build_knn(embedding):
            n = len(embedding)
            return (
                csr_matrix((n, n), dtype=np.float32),
                FakeIndex(embedding),
                min(1, max(0, n - 1)),
            )

        def fake_leiden_resolutions(adjacency, resolutions):
            n = adjacency.shape[0]
            results = {}
            for resolution in resolutions:
                if float(resolution) >= 3.0 and n >= 3:
                    results[float(resolution)] = np.arange(n, dtype=np.int32)
                elif float(resolution) == 1.0 and n >= 3:
                    results[float(resolution)] = np.array(
                        [0, 0, 1], dtype=np.int32
                    )
                else:
                    results[float(resolution)] = np.zeros(n, dtype=np.int32)
            return results

        with tempfile.TemporaryDirectory() as out_dir, mock.patch.multiple(
            topology,
            OUT_DIR=out_dir,
            SAMPLE_BUDGET=4,
            ASSIGN_NEIGHBORS=3,
            COMMUNITY_TARGET=4,
            COMMUNITY_TOLERANCE=0.05,
            LEIDEN_RESOLUTION_CANDIDATES=(0.5, 1.0),
            load_inputs=mock.Mock(
                return_value=({}, keys, rows, h0_labels, dims)
            ),
            fit_block_scalers=mock.Mock(return_value={}),
            make_normalized_embedding=mock.Mock(side_effect=fake_embedding),
            build_knn_graph=mock.Mock(side_effect=fake_build_knn),
            run_leiden_resolutions=mock.Mock(
                side_effect=fake_leiden_resolutions
            ),
            _make_exact_knn_index=mock.Mock(
                side_effect=lambda embedding, _neighbors: FakeIndex(embedding)
            ),
        ):
            topology.main()
            metadata = json.loads(
                Path(out_dir, "topology_run_metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(metadata["selected_resolution"], 3.0)
            self.assertEqual(metadata["resolution_search_rounds"], 1)
            self.assertEqual(metadata["final_communities"], 4)
            self.assertEqual(metadata["representative_patterns"], 4)
            with np.load(Path(out_dir, "topology_labels.npz")) as saved:
                self.assertTrue((saved["topology_labels"] >= 0).all())
                self.assertEqual(len(saved["sample_rows"]), 4)


if __name__ == "__main__":
    unittest.main()
