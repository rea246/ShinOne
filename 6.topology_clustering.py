"""
Stage 2 hierarchical topology clustering.

Reads the full Stage-1 H0 labels and the cached hierarchical GNN blocks, then
runs an independent kNN + Leiden partition inside every dense H0 group and the
preserved HDBSCAN noise/rare group.

Required in the research environment:
    pip install python-igraph leidenalg

Existing pipeline dependencies (numpy, scipy, scikit-learn, torch) are reused.
FAISS GPU is used for exact kNN when available; otherwise an exact PyTorch CUDA
backend uses the existing torch dependency before falling back to CPU search.
Representative sampling is intentionally separated into 7.select_representatives.py.
"""

import csv
import gc
import hashlib
import os
import time

import numpy as np


# ============================================================================
# CONFIG
# ============================================================================
_here = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_here, "hkeys_features.pt")
H0_LABELS_PATH = os.path.join(_here, "h0_clustering_out", "h0_labels_full.npz")
OUT_DIR = os.path.join(_here, "topology_clustering_out")

K_NEIGHBORS = 20
LEIDEN_RESOLUTION = 1.0
EDGE_WEIGHT_MODE = "local_gaussian"  # local_gaussian | inverse_distance | binary
MUTUAL_KNN = False
RANDOM_SEED = 42

# Bounds peak RAM without changing the exact kNN definition.
NORMALIZE_BLOCK = 250_000
KNN_QUERY_BLOCK = 10_000

# Exact search backend priority in auto mode:
# FAISS GPU -> PyTorch CUDA -> FAISS CPU (OpenMP) -> sklearn CPU (all cores).
KNN_BACKEND = "auto"  # auto | faiss_gpu | torch_gpu | faiss_cpu | sklearn
GPU_DEVICE = 0

# PyTorch CUDA is the dependency-free GPU fallback when FAISS GPU is absent.
# It keeps exact full-population L2 search while bounding the temporary distance
# matrix. These values use about 1 GiB for one 2,048 x 131,072 float32 tile.
TORCH_GPU_QUERY_BLOCK = 2_048
TORCH_GPU_REFERENCE_BLOCK = 131_072

DIAGNOSTIC_ANCHORS = 5
DIAGNOSTIC_NEIGHBORS = 5

FEATURE_BLOCKS = ("h0", "h1", "h2", "h3", "edge")
BLOCK_WEIGHTS = {name: 1.0 for name in FEATURE_BLOCKS}


def _import_or_exit(module_name, install_name=None):
    try:
        return __import__(module_name)
    except ImportError as exc:
        package = install_name or module_name
        raise SystemExit(
            f"필수 패키지 '{module_name}'가 없습니다. 연구 환경에 설치하세요: "
            f"pip install {package}"
        ) from exc


def coarse_group_name(h0_label):
    return "rare" if int(h0_label) < 0 else f"H0_{int(h0_label)}"


def final_cluster_id(h0_label, topology_label):
    prefix = "RARE" if int(h0_label) < 0 else f"H0_{int(h0_label)}"
    return f"{prefix}_T{int(topology_label)}"


def _update_population_fingerprint(digest, global_row, key):
    key_bytes = str(key).encode("utf-8")
    digest.update(int(global_row).to_bytes(8, "little", signed=True))
    digest.update(len(key_bytes).to_bytes(8, "little", signed=False))
    digest.update(key_bytes)


def population_fingerprint(keys, rows):
    """Hash ordered global-row/key pairs to detect stale Stage-2 outputs."""
    digest = hashlib.sha256()
    for global_row in rows:
        _update_population_fingerprint(digest, global_row, keys[int(global_row)])
    return digest.hexdigest()


def renumber_communities(labels):
    """Return deterministic IDs: largest community first, then old ID."""
    labels = np.asarray(labels, dtype=np.int64)
    ids, counts = np.unique(labels, return_counts=True)
    order = sorted(zip(ids.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))
    mapping = {old: new for new, (old, _) in enumerate(order)}
    return np.fromiter((mapping[int(v)] for v in labels), dtype=np.int32, count=len(labels))


def edge_weights(mode, distances, src, dst, local_scale):
    """Convert normalized Euclidean distance to positive similarity."""
    distances = np.asarray(distances, dtype=np.float32)
    if mode == "binary":
        return np.ones(len(distances), dtype=np.float32)
    if mode == "inverse_distance":
        return (1.0 / (1.0 + distances)).astype(np.float32, copy=False)
    if mode == "local_gaussian":
        # Self-tuning Gaussian: each endpoint's kth-neighbor radius supplies
        # the local scale, so dense and rare regions are not forced to share
        # one arbitrary global bandwidth.
        denom = local_scale[src] * local_scale[dst]
        denom = np.maximum(denom, np.finfo(np.float32).eps)
        exponent = np.minimum((distances * distances) / denom, 50.0)
        return np.exp(-exponent).astype(np.float32, copy=False)
    raise ValueError(
        f"EDGE_WEIGHT_MODE={mode!r}; choose local_gaussian, inverse_distance, or binary"
    )


def _validate_config():
    if K_NEIGHBORS < 1:
        raise ValueError("K_NEIGHBORS must be >= 1")
    if LEIDEN_RESOLUTION <= 0:
        raise ValueError("LEIDEN_RESOLUTION must be > 0")
    if NORMALIZE_BLOCK < 1 or KNN_QUERY_BLOCK < 1:
        raise ValueError("block sizes must be >= 1")
    valid_backends = {"auto", "faiss_gpu", "torch_gpu", "faiss_cpu", "sklearn"}
    if KNN_BACKEND not in valid_backends:
        raise ValueError(f"KNN_BACKEND must be one of {sorted(valid_backends)}")
    if GPU_DEVICE < 0:
        raise ValueError("GPU_DEVICE must be >= 0")
    if TORCH_GPU_QUERY_BLOCK < 1 or TORCH_GPU_REFERENCE_BLOCK < 1:
        raise ValueError("PyTorch GPU block sizes must be >= 1")
    if set(BLOCK_WEIGHTS) != set(FEATURE_BLOCKS):
        raise ValueError(f"BLOCK_WEIGHTS keys must be {FEATURE_BLOCKS}")
    if any(BLOCK_WEIGHTS[name] <= 0 for name in FEATURE_BLOCKS):
        raise ValueError("all BLOCK_WEIGHTS must be > 0")
    edge_weights(EDGE_WEIGHT_MODE, np.array([], dtype=np.float32),
                 np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                 np.ones(1, dtype=np.float32))


def load_inputs():
    torch = _import_or_exit("torch")
    if not os.path.exists(CACHE_PATH):
        raise FileNotFoundError(f"feature cache 없음: {CACHE_PATH} (먼저 4번 실행)")
    if not os.path.exists(H0_LABELS_PATH):
        raise FileNotFoundError(f"Stage-1 전량 label 없음: {H0_LABELS_PATH} (먼저 4번 실행)")

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    features, keys = cache["features"], cache["keys"]
    missing = [name for name in FEATURE_BLOCKS if name not in features]
    if missing:
        raise KeyError(f"feature cache에 필요한 block 없음: {missing}")

    dims = {}
    total_rows = None
    for name in FEATURE_BLOCKS:
        tensor = features[name]
        if tensor.ndim != 2 or tensor.shape[1] < 1:
            raise ValueError(f"features[{name!r}] shape 오류: {tuple(tensor.shape)}")
        total_rows = tensor.shape[0] if total_rows is None else total_rows
        if tensor.shape[0] != total_rows:
            raise ValueError(f"feature row 수 불일치: {name}={tensor.shape[0]}, expected={total_rows}")
        dims[name] = int(tensor.shape[1])
    if len(keys) != total_rows:
        raise ValueError(f"keys={len(keys)}개, feature rows={total_rows}개로 불일치")

    stage1 = np.load(H0_LABELS_PATH)
    rows = np.asarray(stage1["rows"], dtype=np.int64)
    h0_labels = np.asarray(stage1["labels"])
    if rows.ndim != 1 or h0_labels.ndim != 1 or len(rows) != len(h0_labels):
        raise ValueError("h0_labels_full.npz의 rows/labels는 같은 길이의 1D 배열이어야 함")
    if len(rows) == 0:
        raise ValueError("Stage-1 label이 비어 있음")
    if rows.min() < 0 or rows.max() >= total_rows:
        raise IndexError("Stage-1 rows가 feature cache 범위를 벗어남")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Stage-1 rows에 중복 global row가 있음")
    if not np.all(np.equal(h0_labels, h0_labels.astype(np.int64))):
        raise ValueError("Stage-1 labels는 정수여야 함")
    if (h0_labels < -1).any():
        raise ValueError("Stage-1 labels는 dense ID(>=0) 또는 rare/noise(-1)여야 함")

    return features, keys, rows, h0_labels.astype(np.int32), dims


def _take_feature_rows(tensor, rows):
    torch = _import_or_exit("torch")
    index = torch.from_numpy(np.asarray(rows, dtype=np.int64))
    return tensor.index_select(0, index).float().numpy()


def fit_block_scalers(features, rows):
    """Fit one population-level StandardScaler per hierarchy block."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("scikit-learn 필요: pip install scikit-learn") from exc

    scalers = {}
    print("\n[Normalization] Stage-1 전체 population 기준 block별 StandardScaler",
          flush=True)
    for name in FEATURE_BLOCKS:
        scaler = StandardScaler()
        for start in range(0, len(rows), NORMALIZE_BLOCK):
            block_rows = rows[start:start + NORMALIZE_BLOCK]
            values = _take_feature_rows(features[name], block_rows)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} block에 NaN/Inf가 있음 (rows {start}:{start + len(values)})")
            scaler.partial_fit(values)
        scalers[name] = scaler
        print(f"  {name:<4} dim={features[name].shape[1]}", flush=True)
    return scalers


def make_normalized_embedding(features, scalers, global_rows, dims):
    """Materialize one coarse group's block-normalized full embedding."""
    started = time.time()
    total_dim = sum(dims.values())
    total_rows = len(global_rows)
    embedding_mib = total_rows * total_dim * np.dtype(np.float32).itemsize / (1024 ** 2)
    print(
        f"  [Embedding] START rows={total_rows:,}, total_dim={total_dim:,}, "
        f"estimated_memory={embedding_mib:,.1f} MiB",
        flush=True,
    )
    result = np.empty((total_rows, total_dim), dtype=np.float32)
    for start in range(0, total_rows, NORMALIZE_BLOCK):
        end = min(start + NORMALIZE_BLOCK, total_rows)
        parts = []
        for name in FEATURE_BLOCKS:
            values = _take_feature_rows(features[name], global_rows[start:end])
            values = scalers[name].transform(values).astype(np.float32, copy=False)
            # Equal total contribution per block if future cache dimensions differ.
            values *= np.sqrt(BLOCK_WEIGHTS[name] / dims[name])
            parts.append(values)
        result[start:end] = np.concatenate(parts, axis=1)
        print(
            f"  [Embedding] {end:,}/{total_rows:,} "
            f"({end / total_rows:.1%}), elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    print(f"  [Embedding] DONE elapsed={time.time() - started:.1f}s", flush=True)
    return result


def _nonself_edges(query_ids, neighbor_ids, distances, k):
    """Flatten the first k non-self neighbors per query, including tie cases."""
    keep = neighbor_ids != query_ids[:, None]
    keep &= np.cumsum(keep, axis=1) <= k
    row_pos, col_pos = np.where(keep)
    return (query_ids[row_pos].astype(np.int32, copy=False),
            neighbor_ids[row_pos, col_pos].astype(np.int32, copy=False),
            distances[row_pos, col_pos].astype(np.float32, copy=False))


def _optional_import(module_name):
    try:
        return __import__(module_name), None
    except (ImportError, OSError) as exc:
        return None, exc


class _FaissExactKNN:
    """sklearn-like exact L2 adapter backed by FAISS CPU or GPU."""

    def __init__(self, embedding, faiss, use_gpu, gpu_device=0):
        self._faiss = faiss
        self._resources = None
        self._n_samples = len(embedding)
        dim = embedding.shape[1]
        index = faiss.IndexFlatL2(dim)

        if use_gpu:
            self._resources = faiss.StandardGpuResources()
            clone_options = faiss.GpuClonerOptions()
            clone_options.useFloat16 = False
            index = faiss.index_cpu_to_gpu(
                self._resources, int(gpu_device), index, clone_options
            )
            self._fit_method = (
                f"faiss_gpu_exact(device={gpu_device}, float32, exhaustive)"
            )
        else:
            cpu_threads = max(1, os.cpu_count() or 1)
            if hasattr(faiss, "omp_set_num_threads"):
                faiss.omp_set_num_threads(cpu_threads)
            self._fit_method = (
                f"faiss_cpu_exact(OpenMP_threads={cpu_threads}, exhaustive)"
            )

        self._index = index
        self._index.add(np.ascontiguousarray(embedding, dtype=np.float32))

    def kneighbors(self, queries, n_neighbors, return_distance=True):
        if n_neighbors > self._n_samples:
            raise ValueError("n_neighbors exceeds fitted population")
        query_array = np.ascontiguousarray(queries, dtype=np.float32)
        squared_distances, neighbors = self._index.search(query_array, n_neighbors)
        if not return_distance:
            return neighbors
        # IndexFlatL2 returns squared L2. The original sklearn path returns L2,
        # so sqrt is required to preserve the graph weighting definition.
        squared_distances = np.asarray(squared_distances, dtype=np.float32)
        np.maximum(squared_distances, 0.0, out=squared_distances)
        np.sqrt(squared_distances, out=squared_distances)
        return squared_distances, neighbors


class _TorchExactKNN:
    """Exact blocked L2 search on CUDA, retaining only top-k per query."""

    def __init__(self, embedding, torch, device):
        self._torch = torch
        self._device = torch.device(device)
        self._n_samples = len(embedding)
        self._vectors = torch.from_numpy(
            np.ascontiguousarray(embedding, dtype=np.float32)
        ).to(self._device)
        self._norms = (self._vectors * self._vectors).sum(dim=1)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            device_name = torch.cuda.get_device_name(self._device)
            self._fit_method = (
                f"torch_cuda_exact(device={self._device.index}, {device_name}, "
                "float32, tf32=False, exhaustive)"
            )
        else:
            self._fit_method = "torch_cpu_exact(float32, exhaustive)"

    def kneighbors(self, queries, n_neighbors, return_distance=True):
        if n_neighbors > self._n_samples:
            raise ValueError("n_neighbors exceeds fitted population")

        torch = self._torch
        query_array = np.ascontiguousarray(queries, dtype=np.float32)
        result_distances = np.empty(
            (len(query_array), n_neighbors), dtype=np.float32
        )
        result_neighbors = np.empty(
            (len(query_array), n_neighbors), dtype=np.int64
        )

        cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
        previous_tf32 = None
        if self._device.type == "cuda" and cuda_matmul is not None:
            previous_tf32 = cuda_matmul.allow_tf32
            cuda_matmul.allow_tf32 = False

        try:
            with torch.inference_mode():
                for query_start in range(0, len(query_array), TORCH_GPU_QUERY_BLOCK):
                    query_end = min(
                        query_start + TORCH_GPU_QUERY_BLOCK, len(query_array)
                    )
                    query_tensor = torch.from_numpy(
                        query_array[query_start:query_end]
                    ).to(self._device)
                    query_norms = (query_tensor * query_tensor).sum(
                        dim=1, keepdim=True
                    )
                    query_count = len(query_tensor)
                    best_distances = torch.full(
                        (query_count, n_neighbors),
                        float("inf"),
                        dtype=torch.float32,
                        device=self._device,
                    )
                    best_neighbors = torch.full(
                        (query_count, n_neighbors),
                        -1,
                        dtype=torch.int64,
                        device=self._device,
                    )

                    for reference_start in range(
                            0, self._n_samples, TORCH_GPU_REFERENCE_BLOCK):
                        reference_end = min(
                            reference_start + TORCH_GPU_REFERENCE_BLOCK,
                            self._n_samples,
                        )
                        references = self._vectors[reference_start:reference_end]
                        squared_distances = (
                            query_norms
                            + self._norms[reference_start:reference_end].unsqueeze(0)
                        )
                        squared_distances.addmm_(
                            query_tensor, references.T, beta=1.0, alpha=-2.0
                        )
                        squared_distances.clamp_min_(0.0)

                        block_k = min(n_neighbors, reference_end - reference_start)
                        block_distances, block_neighbors = torch.topk(
                            squared_distances,
                            block_k,
                            dim=1,
                            largest=False,
                            sorted=True,
                        )
                        block_neighbors += reference_start
                        merged_distances = torch.cat(
                            (best_distances, block_distances), dim=1
                        )
                        merged_neighbors = torch.cat(
                            (best_neighbors, block_neighbors), dim=1
                        )
                        best_distances, selection = torch.topk(
                            merged_distances,
                            n_neighbors,
                            dim=1,
                            largest=False,
                            sorted=True,
                        )
                        best_neighbors = torch.gather(
                            merged_neighbors, dim=1, index=selection
                        )

                    best_distances.sqrt_()
                    result_distances[query_start:query_end] = (
                        best_distances.cpu().numpy()
                    )
                    result_neighbors[query_start:query_end] = (
                        best_neighbors.cpu().numpy()
                    )
        finally:
            if previous_tf32 is not None:
                cuda_matmul.allow_tf32 = previous_tf32

        if return_distance:
            return result_distances, result_neighbors
        return result_neighbors


def _make_exact_knn_index(embedding, query_neighbors):
    """Select the fastest available exhaustive float32 L2 backend."""
    requested = KNN_BACKEND
    faiss, faiss_error = _optional_import("faiss")

    if requested in {"auto", "faiss_gpu"}:
        gpu_ready = (
            faiss is not None
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "index_cpu_to_gpu")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() > GPU_DEVICE
        )
        if gpu_ready:
            try:
                return _FaissExactKNN(
                    embedding, faiss, use_gpu=True, gpu_device=GPU_DEVICE
                )
            except Exception as exc:
                if requested == "faiss_gpu":
                    raise
                print(
                    f"  ⚠ [Exact kNN] FAISS GPU initialization failed: {exc}",
                    flush=True,
                )
        elif requested == "faiss_gpu":
            detail = faiss_error or "FAISS GPU bindings/device not available"
            raise SystemExit(f"KNN_BACKEND='faiss_gpu' 사용 불가: {detail}")

    if requested in {"auto", "torch_gpu"}:
        torch, torch_error = _optional_import("torch")
        gpu_ready = (
            torch is not None
            and torch.cuda.is_available()
            and torch.cuda.device_count() > GPU_DEVICE
        )
        if gpu_ready:
            try:
                return _TorchExactKNN(
                    embedding, torch, device=f"cuda:{GPU_DEVICE}"
                )
            except Exception as exc:
                if requested == "torch_gpu":
                    raise
                print(
                    f"  ⚠ [Exact kNN] PyTorch CUDA initialization failed: {exc}",
                    flush=True,
                )
        elif requested == "torch_gpu":
            detail = torch_error or "CUDA device not available to PyTorch"
            raise SystemExit(f"KNN_BACKEND='torch_gpu' 사용 불가: {detail}")

    if requested in {"auto", "faiss_cpu"} and faiss is not None:
        try:
            return _FaissExactKNN(embedding, faiss, use_gpu=False)
        except Exception as exc:
            if requested == "faiss_cpu":
                raise
            print(
                f"  ⚠ [Exact kNN] FAISS CPU initialization failed: {exc}",
                flush=True,
            )
    elif requested == "faiss_cpu":
        raise SystemExit(f"KNN_BACKEND='faiss_cpu' 사용 불가: {faiss_error}")

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise SystemExit("exact kNN backend 필요: torch, faiss 또는 scikit-learn") from exc
    knn = NearestNeighbors(
        n_neighbors=query_neighbors, metric="euclidean", n_jobs=-1
    )
    knn.fit(embedding)
    return knn


def build_knn_graph(embedding):
    """Build an undirected sparse union/mutual kNN graph in bounded query RAM."""
    try:
        from scipy.sparse import csr_matrix
    except ImportError as exc:
        raise SystemExit("SciPy 필요: pip install scipy") from exc

    n = len(embedding)
    dim = embedding.shape[1]
    if n <= 1:
        print(f"  [Exact kNN] START rows={n:,}, dim={dim:,}, k=0", flush=True)
        print(f"  [Exact kNN] rough distance candidates (N^2)={n * n:,}", flush=True)
        print("  [Exact kNN] DONE (skipped: fewer than 2 rows)", flush=True)
        return csr_matrix((n, n), dtype=np.float32), None, 0

    k = min(K_NEIGHBORS, n - 1)
    query_neighbors = min(n, k + 1)
    distance_candidates = n * n
    print(f"  [Exact kNN] START rows={n:,}, dim={dim:,}, k={k}", flush=True)
    print(
        f"  [Exact kNN] rough distance candidates (N^2)={distance_candidates:,}",
        flush=True,
    )
    if distance_candidates >= 1_000_000_000:
        print(
            "  ⚠ [Exact kNN] N^2 candidates >= 1e9; exact neighbor search may be "
            "the dominant bottleneck",
            flush=True,
        )
    fit_started = time.time()
    knn = _make_exact_knn_index(embedding, query_neighbors)
    print(
        f"  [Exact kNN] fit DONE method={getattr(knn, '_fit_method', 'unknown')}, "
        f"elapsed={time.time() - fit_started:.1f}s",
        flush=True,
    )

    capacity = n * k
    src = np.empty(capacity, dtype=np.int32)
    dst = np.empty(capacity, dtype=np.int32)
    distance = np.empty(capacity, dtype=np.float32)
    local_scale = np.zeros(n, dtype=np.float32)
    cursor = 0
    query_started = time.time()

    for start in range(0, n, KNN_QUERY_BLOCK):
        end = min(start + KNN_QUERY_BLOCK, n)
        block_started = time.time()
        print(f"  [Exact kNN] QUERY START {start:,}:{end:,} / {n:,}", flush=True)
        query_ids = np.arange(start, end, dtype=np.int32)
        dist_block, neighbor_block = knn.kneighbors(
            embedding[start:end], n_neighbors=query_neighbors, return_distance=True
        )
        src_block, dst_block, distance_block = _nonself_edges(
            query_ids, neighbor_block, dist_block, k
        )
        count = len(src_block)
        src[cursor:cursor + count] = src_block
        dst[cursor:cursor + count] = dst_block
        distance[cursor:cursor + count] = distance_block
        np.maximum.at(local_scale, src_block, distance_block)
        cursor += count
        total_elapsed = time.time() - query_started
        completed_rows = end
        average_rows_per_sec = (
            completed_rows / total_elapsed if total_elapsed > 0 else float("inf")
        )
        print(
            f"  [Exact kNN] QUERY DONE {end:,}/{n:,}, {end / n:.1%}, "
            f"block_elapsed={time.time() - block_started:.1f}s, "
            f"total_elapsed={total_elapsed:.1f}s, "
            f"average_rows/sec={average_rows_per_sec:,.1f}",
            flush=True,
        )

    src, dst, distance = src[:cursor], dst[:cursor], distance[:cursor]
    positive = local_scale[local_scale > 0]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    local_scale[local_scale <= 0] = fallback
    graph_started = time.time()
    print("  [Graph assembly] edge weighting START", flush=True)
    weights = edge_weights(EDGE_WEIGHT_MODE, distance, src, dst, local_scale)
    print(
        f"  [Graph assembly] edge weighting DONE elapsed={time.time() - graph_started:.1f}s",
        flush=True,
    )

    csr_started = time.time()
    directed = csr_matrix((weights, (src, dst)), shape=(n, n), dtype=np.float32)
    del src, dst, distance, weights, local_scale
    directed.sum_duplicates()
    print(
        f"  [Graph assembly] CSR creation DONE elapsed={time.time() - csr_started:.1f}s",
        flush=True,
    )
    symmetrize_started = time.time()
    adjacency = directed.minimum(directed.T) if MUTUAL_KNN else directed.maximum(directed.T)
    del directed
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    print(
        f"  [Graph assembly] symmetrization DONE "
        f"elapsed={time.time() - symmetrize_started:.1f}s, "
        f"total_assembly_elapsed={time.time() - graph_started:.1f}s",
        flush=True,
    )
    print(f"  [Graph assembly] undirected edges={adjacency.nnz // 2:,}", flush=True)
    return adjacency.tocsr(), knn, k


def run_leiden(adjacency):
    n = adjacency.shape[0]
    if n <= 1:
        print(
            f"  [Leiden] DONE (skipped: vertices={n:,}, edges=0)",
            flush=True,
        )
        return np.zeros(n, dtype=np.int32)
    if adjacency.nnz == 0:
        print("  ⚠ kNN edge 없음: 각 pattern을 singleton community로 보존", flush=True)
        return np.arange(n, dtype=np.int32)

    igraph = _import_or_exit("igraph", "python-igraph")
    leidenalg = _import_or_exit("leidenalg")
    conversion_started = time.time()
    print("  [Leiden] igraph conversion START", flush=True)
    graph = igraph.Graph.Weighted_Adjacency(
        adjacency, mode="undirected", attr="weight", loops=False
    )
    print(
        f"  [Leiden] igraph conversion DONE elapsed={time.time() - conversion_started:.1f}s, "
        f"vertices={graph.vcount():,}, edges={graph.ecount():,}",
        flush=True,
    )
    partition_started = time.time()
    print("  [Leiden] partition START", flush=True)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(LEIDEN_RESOLUTION),
        n_iterations=-1,
        seed=int(RANDOM_SEED),
    )
    print(
        f"  [Leiden] partition DONE elapsed={time.time() - partition_started:.1f}s",
        flush=True,
    )
    return renumber_communities(partition.membership)


def nearest_neighbor_diagnostics(knn, embedding, local_labels, global_rows,
                                 keys, h0_label, rng):
    started = time.time()
    if knn is None or len(embedding) <= 1 or DIAGNOSTIC_ANCHORS < 1:
        print(
            f"  [Diagnostics] DONE elapsed={time.time() - started:.1f}s (skipped)",
            flush=True,
        )
        return []
    anchors = rng.choice(
        len(embedding), min(DIAGNOSTIC_ANCHORS, len(embedding)), replace=False
    )
    n_query = min(len(embedding), DIAGNOSTIC_NEIGHBORS + 1)
    distances, neighbors = knn.kneighbors(
        embedding[anchors], n_neighbors=n_query, return_distance=True
    )
    rows = []
    for query_row, anchor in enumerate(anchors.tolist()):
        keep = neighbors[query_row] != anchor
        chosen_neighbors = neighbors[query_row][keep][:DIAGNOSTIC_NEIGHBORS]
        chosen_distances = distances[query_row][keep][:DIAGNOSTIC_NEIGHBORS]
        anchor_key = str(keys[int(global_rows[anchor])])
        for rank, (neighbor, distance) in enumerate(
                zip(chosen_neighbors.tolist(), chosen_distances.tolist()), start=1):
            rows.append({
                "coarse_group": coarse_group_name(h0_label),
                "anchor_key": anchor_key,
                "neighbor_rank": rank,
                "neighbor_key": str(keys[int(global_rows[neighbor])]),
                "distance": float(distance),
                "same_final_cluster": bool(local_labels[anchor] == local_labels[neighbor]),
            })
    print(
        f"  [Diagnostics] DONE elapsed={time.time() - started:.1f}s, rows={len(rows):,}",
        flush=True,
    )
    return rows


def _community_summary(h0_label, local_labels, total_count, embedding=None):
    """Summarize communities and their spread in normalized topology space."""
    ids, counts = np.unique(local_labels, return_counts=True)
    coarse_count = len(local_labels)
    rows = []
    for cid, count in zip(ids, counts):
        if embedding is None:
            distance_mean = distance_p95 = distance_max = float("nan")
        else:
            member_positions = np.flatnonzero(local_labels == cid)
            centroid_sum = np.zeros(embedding.shape[1], dtype=np.float64)
            for start in range(0, len(member_positions), NORMALIZE_BLOCK):
                end = min(start + NORMALIZE_BLOCK, len(member_positions))
                centroid_sum += embedding[member_positions[start:end]].sum(
                    axis=0, dtype=np.float64
                )
            centroid = (centroid_sum / len(member_positions)).astype(np.float32)
            distances = np.empty(len(member_positions), dtype=np.float32)
            for start in range(0, len(member_positions), NORMALIZE_BLOCK):
                end = min(start + NORMALIZE_BLOCK, len(member_positions))
                values = embedding[member_positions[start:end]]
                delta = values - centroid
                distances[start:end] = np.sqrt(
                    np.einsum("ij,ij->i", delta, delta)
                )
            distance_mean = float(distances.mean())
            distance_p95 = float(np.percentile(distances, 95))
            distance_max = float(distances.max())

        rows.append({
            "h0_label": int(h0_label),
            "coarse_group": coarse_group_name(h0_label),
            "topology_cluster": int(cid),
            "final_cluster": final_cluster_id(h0_label, cid),
            "pattern_count": int(count),
            "fraction_in_coarse_group": float(count / coarse_count),
            "fraction_in_total": float(count / total_count),
            "centroid_distance_mean": distance_mean,
            "centroid_distance_p95": distance_p95,
            "centroid_distance_max": distance_max,
        })
    return rows


def _print_group_diagnostics(h0_label, n, k, labels, edge_count):
    counts = np.unique(labels, return_counts=True)[1]
    print(f"  Coarse group          : {coarse_group_name(h0_label)}", flush=True)
    print(f"  Number of patterns    : {n:,}", flush=True)
    print(f"  k / undirected edges  : {k} / {edge_count:,}", flush=True)
    print(f"  Leiden communities    : {len(counts):,}", flush=True)
    print("  community size        : "
          f"min={counts.min():,}, median={np.median(counts):,.1f}, "
          f"mean={counts.mean():,.1f}, max={counts.max():,}", flush=True)


def write_outputs(keys, rows, h0_labels, topology_labels, summaries, diagnostics):
    started = time.time()
    print("\n[Output] writing START", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    assignment_path = os.path.join(OUT_DIR, "topology_assignments.csv")
    assignment_tmp = assignment_path + ".tmp"
    fingerprint_digest = hashlib.sha256()
    with open(assignment_tmp, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pattern_key", "original_h0_label", "coarse_group",
                         "topology_cluster", "final_cluster"])
        total_rows = len(rows)
        assignment_started = time.time()
        for written, (row, h0_label, topology_label) in enumerate(
                zip(rows, h0_labels, topology_labels), start=1):
            _update_population_fingerprint(
                fingerprint_digest, row, keys[int(row)]
            )
            writer.writerow([
                str(keys[int(row)]), int(h0_label), coarse_group_name(h0_label),
                int(topology_label), final_cluster_id(h0_label, topology_label),
            ])
            if written % 250_000 == 0 or written == total_rows:
                print(
                    f"  [Output] topology_assignments.csv {written:,}/{total_rows:,} "
                    f"({written / total_rows:.1%}), "
                    f"elapsed={time.time() - assignment_started:.1f}s",
                    flush=True,
                )
    os.replace(assignment_tmp, assignment_path)

    summary_path = os.path.join(OUT_DIR, "topology_cluster_summary.csv")
    summary_tmp = summary_path + ".tmp"
    summary_fields = [
        "h0_label", "coarse_group", "topology_cluster", "final_cluster",
        "pattern_count", "fraction_in_coarse_group", "fraction_in_total",
        "centroid_distance_mean", "centroid_distance_p95", "centroid_distance_max",
    ]
    with open(summary_tmp, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    os.replace(summary_tmp, summary_path)

    diagnostic_path = os.path.join(OUT_DIR, "nearest_neighbor_diagnostics.csv")
    diagnostic_tmp = diagnostic_path + ".tmp"
    diagnostic_fields = ["coarse_group", "anchor_key", "neighbor_rank", "neighbor_key",
                         "distance", "same_final_cluster"]
    with open(diagnostic_tmp, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=diagnostic_fields)
        writer.writeheader()
        writer.writerows(diagnostics)
    os.replace(diagnostic_tmp, diagnostic_path)

    labels_path = os.path.join(OUT_DIR, "topology_labels.npz")
    labels_tmp = labels_path + ".tmp.npz"
    np.savez(
        labels_tmp,
        rows=rows,
        h0_labels=h0_labels,
        topology_labels=topology_labels,
        population_fingerprint=np.asarray(fingerprint_digest.hexdigest()),
    )
    os.replace(labels_tmp, labels_path)

    print("\n[Saved]", flush=True)
    for path in (assignment_path, summary_path, diagnostic_path, labels_path):
        print(f"  {path}", flush=True)
    print(f"[Output] writing DONE elapsed={time.time() - started:.1f}s", flush=True)


def main():
    _validate_config()
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    started = time.time()

    features, keys, rows, h0_labels, dims = load_inputs()
    dense_groups = sorted(int(v) for v in np.unique(h0_labels) if v >= 0)
    coarse_groups = dense_groups + ([-1] if (h0_labels < 0).any() else [])
    rare_count = int((h0_labels < 0).sum())

    print("=" * 72, flush=True)
    print("Stage 2: hierarchical full-topology kNN + Leiden", flush=True)
    print("=" * 72, flush=True)
    print(f"Total patterns             : {len(rows):,}", flush=True)
    print(f"Number of H0 coarse groups : {len(dense_groups)}", flush=True)
    print(f"Rare/noise patterns        : {rare_count:,}", flush=True)
    print(f"Rare fraction              : {rare_count / len(rows):.2%}", flush=True)
    print(f"Feature dimensions         : {dims} (total={sum(dims.values())})", flush=True)
    print(f"k={K_NEIGHBORS}, resolution={LEIDEN_RESOLUTION}, "
          f"edge={EDGE_WEIGHT_MODE}, mutual={MUTUAL_KNN}, seed={RANDOM_SEED}",
          flush=True)
    print(
        f"kNN backend={KNN_BACKEND}, gpu_device={GPU_DEVICE}, "
        f"torch_gpu_tiles={TORCH_GPU_QUERY_BLOCK:,}x{TORCH_GPU_REFERENCE_BLOCK:,}",
        flush=True,
    )

    scalers = fit_block_scalers(features, rows)
    topology_labels = np.full(len(rows), -1, dtype=np.int32)
    summaries, diagnostics = [], []

    total_groups = len(coarse_groups)
    for group_index, h0_label in enumerate(coarse_groups, start=1):
        group_started = time.time()
        positions = np.where(h0_labels == h0_label)[0]
        print("\n" + "-" * 72, flush=True)
        print(
            f"[Group {group_index}/{total_groups}] START "
            f"coarse_group={coarse_group_name(h0_label)}, patterns={len(positions):,}",
            flush=True,
        )
        global_rows = rows[positions]
        embedding = make_normalized_embedding(features, scalers, global_rows, dims)
        adjacency, knn, k = build_knn_graph(embedding)
        local_labels = run_leiden(adjacency)
        topology_labels[positions] = local_labels

        _print_group_diagnostics(
            h0_label, len(positions), k, local_labels, adjacency.nnz // 2
        )
        dispersion_started = time.time()
        print("  [Community dispersion] START", flush=True)
        summaries.extend(
            _community_summary(
                h0_label, local_labels, len(rows), embedding=embedding
            )
        )
        print(
            "  [Community dispersion] DONE "
            f"elapsed={time.time() - dispersion_started:.1f}s",
            flush=True,
        )
        diagnostics.extend(nearest_neighbor_diagnostics(
            knn, embedding, local_labels, global_rows, keys, h0_label, rng
        ))
        del embedding, adjacency, knn, local_labels
        gc.collect()
        print(
            f"[Group {group_index}/{total_groups}] DONE "
            f"coarse_group={coarse_group_name(h0_label)}, "
            f"elapsed={time.time() - group_started:.1f}s",
            flush=True,
        )

    if (topology_labels < 0).any():
        raise RuntimeError("일부 pattern에 Stage-2 community가 배정되지 않음")

    sizes = np.asarray([row["pattern_count"] for row in summaries])
    rare_communities = sum(row["h0_label"] < 0 for row in summaries)
    print("\n" + "=" * 72, flush=True)
    print(f"Total final communities    : {len(summaries):,}", flush=True)
    print(f"Largest final community    : {sizes.max():,}", flush=True)
    print(f"Smallest final community   : {sizes.min():,}", flush=True)
    print(f"Rare-group community count : {rare_communities:,}", flush=True)
    print("=" * 72, flush=True)

    write_outputs(keys, rows, h0_labels, topology_labels, summaries, diagnostics)
    print(f"완료: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
