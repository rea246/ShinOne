"""
Integrated sampled topology clustering and representative extraction.

Reads the full Stage-1 H0 labels and cached hierarchical GNN blocks.  A
deterministic stratified sample is clustered inside each H0 coarse group, the
full population is assigned to the sampled Leiden communities by 5-NN, and one
real centroid-nearest pattern is emitted for every final community.

Required for the default GPU Leiden backend:
    RAPIDS cuGraph, cuDF, and CuPy matching the installed CUDA version

Existing pipeline dependencies (numpy, scipy, scikit-learn, torch) are reused.
FAISS GPU is used for exact search when available; otherwise an exact PyTorch
CUDA backend uses the existing torch dependency before falling back to CPU.
"""

import csv
import gc
import hashlib
import json
import math
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
# Kept as the direct-call default for run_leiden() and compatibility tests.
LEIDEN_RESOLUTION = 1.0
LEIDEN_RESOLUTION_CANDIDATES = (
    0.25, 0.35, 0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 2.00,
)
LEIDEN_RESOLUTION_SEARCH_ROUNDS = 12
LEIDEN_RESOLUTION_MIN = 1.0e-4
LEIDEN_RESOLUTION_MAX = 4_096.0
LEIDEN_BACKEND = "cugraph"  # cugraph | leidenalg
CUGRAPH_MAX_ITERATIONS = 100
CPU_LEIDEN_ITERATIONS = 2
EDGE_WEIGHT_MODE = "local_gaussian"  # local_gaussian | inverse_distance | binary
MUTUAL_KNN = False
RANDOM_SEED = 42

# Sampled Stage-2 and integrated representative extraction.
SAMPLE_BUDGET = 100_000
ASSIGN_NEIGHBORS = 5
COMMUNITY_TARGET = 1_000
COMMUNITY_TOLERANCE = 0.05

# Bounds peak RAM for each exact sample-search or assignment query.
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


def allocate_stratified_sample_counts(group_sizes, sample_budget):
    """Allocate an exact proportional sample while retaining every H0 group."""
    sizes = np.asarray(group_sizes, dtype=np.int64)
    if sizes.ndim != 1 or len(sizes) == 0 or (sizes < 1).any():
        raise ValueError("group_sizes must be a non-empty 1D array of positive values")
    if sample_budget < 1:
        raise ValueError("sample_budget must be >= 1")

    total = int(sizes.sum())
    budget = min(int(sample_budget), total)
    if budget < len(sizes):
        raise ValueError(
            f"sample budget {budget:,}개로 H0 group {len(sizes):,}개를 "
            "각각 한 번 이상 sampling할 수 없음"
        )
    if budget == total:
        return sizes.copy()

    allocation = np.ones(len(sizes), dtype=np.int64)
    capacity = sizes - 1
    remaining = budget - len(sizes)
    if remaining:
        ideal = remaining * capacity.astype(np.float64) / capacity.sum()
        additions = np.floor(ideal).astype(np.int64)
        allocation += additions
        leftover = remaining - int(additions.sum())
        if leftover:
            order = sorted(
                range(len(sizes)),
                key=lambda index: (
                    -(ideal[index] - additions[index]),
                    -int(capacity[index]),
                    index,
                ),
            )
            for index in order:
                if leftover == 0:
                    break
                if allocation[index] < sizes[index]:
                    allocation[index] += 1
                    leftover -= 1

    if int(allocation.sum()) != budget or (allocation > sizes).any():
        raise RuntimeError("stratified sample allocation failed")
    return allocation


def choose_resolution(community_counts, target, tolerance):
    """Choose the sampled partition closest to target, preferring tolerance."""
    if target < 1 or not 0 <= tolerance < 1:
        raise ValueError("target must be >= 1 and tolerance must be in [0, 1)")
    if not community_counts:
        raise ValueError("community_counts must not be empty")
    lower = target * (1.0 - tolerance)
    upper = target * (1.0 + tolerance)
    in_tolerance = [
        resolution for resolution, count in community_counts.items()
        if lower <= count <= upper
    ]
    candidates = in_tolerance or list(community_counts)
    selected = min(
        candidates,
        key=lambda resolution: (
            abs(community_counts[resolution] - target),
            -float(resolution),
        ),
    )
    return float(selected), bool(in_tolerance)


def next_resolution_candidate(
        community_counts, target, minimum_resolution, maximum_resolution):
    """Propose one unseen resolution that moves community count toward target."""
    if not community_counts:
        raise ValueError("community_counts must not be empty")
    if target < 1:
        raise ValueError("target must be >= 1")
    if minimum_resolution <= 0 or maximum_resolution <= minimum_resolution:
        raise ValueError("invalid resolution search bounds")

    ordered = sorted(
        (float(resolution), int(count))
        for resolution, count in community_counts.items()
    )
    if any(count == target for _, count in ordered):
        return None

    crossings = []
    for left, right in zip(ordered, ordered[1:]):
        if (left[1] - target) * (right[1] - target) < 0:
            crossings.append((left, right))

    if crossings:
        (left_resolution, left_count), (right_resolution, right_count) = min(
            crossings,
            key=lambda pair: (
                abs(math.log(pair[1][0] / pair[0][0])),
                abs(pair[0][1] - target) + abs(pair[1][1] - target),
            ),
        )
        fraction = (target - left_count) / (right_count - left_count)
        candidate = math.exp(
            math.log(left_resolution)
            + fraction * math.log(right_resolution / left_resolution)
        )
        if not left_resolution < candidate < right_resolution:
            candidate = math.sqrt(left_resolution * right_resolution)
    elif max(count for _, count in ordered) < target:
        candidate = ordered[-1][0] * 2.0
    elif min(count for _, count in ordered) > target:
        candidate = ordered[0][0] / 2.0
    else:
        below = min(
            ((resolution, count) for resolution, count in ordered if count < target),
            key=lambda item: target - item[1],
        )
        above = min(
            ((resolution, count) for resolution, count in ordered if count > target),
            key=lambda item: item[1] - target,
        )
        candidate = math.sqrt(below[0] * above[0])

    candidate = min(max(candidate, minimum_resolution), maximum_resolution)
    candidate = float(f"{candidate:.12g}")
    if any(
        math.isclose(candidate, resolution, rel_tol=1.0e-10, abs_tol=0.0)
        for resolution, _ in ordered
    ):
        return None
    return candidate


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
    if (
        not LEIDEN_RESOLUTION_CANDIDATES
        or any(value <= 0 for value in LEIDEN_RESOLUTION_CANDIDATES)
        or len(set(LEIDEN_RESOLUTION_CANDIDATES))
        != len(LEIDEN_RESOLUTION_CANDIDATES)
    ):
        raise ValueError("LEIDEN_RESOLUTION_CANDIDATES must be unique and > 0")
    if LEIDEN_RESOLUTION_SEARCH_ROUNDS < 0:
        raise ValueError("LEIDEN_RESOLUTION_SEARCH_ROUNDS must be >= 0")
    if (
        LEIDEN_RESOLUTION_MIN <= 0
        or LEIDEN_RESOLUTION_MAX <= LEIDEN_RESOLUTION_MIN
    ):
        raise ValueError("invalid adaptive Leiden resolution bounds")
    if LEIDEN_BACKEND not in {"cugraph", "leidenalg"}:
        raise ValueError("LEIDEN_BACKEND must be cugraph or leidenalg")
    if CUGRAPH_MAX_ITERATIONS < 1 or CPU_LEIDEN_ITERATIONS < 1:
        raise ValueError("Leiden iteration limits must be >= 1")
    if NORMALIZE_BLOCK < 1 or KNN_QUERY_BLOCK < 1:
        raise ValueError("block sizes must be >= 1")
    if SAMPLE_BUDGET < 1 or ASSIGN_NEIGHBORS < 1:
        raise ValueError("SAMPLE_BUDGET and ASSIGN_NEIGHBORS must be >= 1")
    if COMMUNITY_TARGET < 1 or not 0 <= COMMUNITY_TOLERANCE < 1:
        raise ValueError(
            "COMMUNITY_TARGET must be >= 1 and COMMUNITY_TOLERANCE in [0, 1)"
        )
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


def _run_leidenalg_partition(graph, leidenalg, resolution):
    partition_started = time.time()
    print(
        f"  [Leiden][CPU] partition START resolution={resolution:g}, "
        f"iterations={CPU_LEIDEN_ITERATIONS}",
        flush=True,
    )
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        n_iterations=int(CPU_LEIDEN_ITERATIONS),
        seed=int(RANDOM_SEED),
    )
    labels = renumber_communities(partition.membership)
    print(
        f"  [Leiden][CPU] partition DONE "
        f"resolution={resolution:g}, elapsed={time.time() - partition_started:.1f}s, "
        f"communities={len(np.unique(labels)):,}",
        flush=True,
    )
    return labels


def _igraph_from_csr(adjacency, igraph):
    conversion_started = time.time()
    print("  [Leiden][CPU] igraph conversion START", flush=True)
    graph = igraph.Graph.Weighted_Adjacency(
        adjacency, mode="undirected", attr="weight", loops=False
    )
    print(
        f"  [Leiden][CPU] igraph conversion DONE "
        f"elapsed={time.time() - conversion_started:.1f}s, "
        f"vertices={graph.vcount():,}, edges={graph.ecount():,}",
        flush=True,
    )
    return graph


def _run_leidenalg_cpu(adjacency, resolution=None):
    """Bounded CPU implementation retained for explicit fallback/validation."""
    resolution = LEIDEN_RESOLUTION if resolution is None else float(resolution)
    igraph = _import_or_exit("igraph", "python-igraph")
    leidenalg = _import_or_exit("leidenalg")
    graph = _igraph_from_csr(adjacency, igraph)
    labels = _run_leidenalg_partition(graph, leidenalg, resolution)
    del graph
    return labels


def _require_cugraph_stack():
    modules = {}
    failures = []
    for module_name in ("cugraph", "cudf", "cupy"):
        module, error = _optional_import(module_name)
        if module is None:
            failures.append(f"{module_name}: {error}")
        else:
            modules[module_name] = module
    if failures:
        raise SystemExit(
            "LEIDEN_BACKEND='cugraph'에 RAPIDS cuGraph/cuDF/CuPy가 필요합니다. "
            "CUDA 버전에 맞춰 설치하세요 (https://docs.rapids.ai/install/). "
            + "; ".join(failures)
        )

    cupy = modules["cupy"]
    try:
        device_count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise SystemExit(f"cuGraph CUDA device 확인 실패: {exc}") from exc
    if device_count <= GPU_DEVICE:
        raise SystemExit(
            f"LEIDEN_BACKEND='cugraph': GPU_DEVICE={GPU_DEVICE} 사용 불가 "
            f"(CUDA devices={device_count})"
        )
    return modules["cugraph"], modules["cudf"], cupy


def _cudf_series_to_numpy(series, cupy):
    if hasattr(series, "to_cupy"):
        return np.asarray(cupy.asnumpy(series.to_cupy()))
    return np.asarray(series.to_numpy())


def _labels_from_vertex_partitions(vertices, partitions, total_vertices):
    """Restore cuGraph's vertex-labelled result, preserving any isolates."""
    vertices = np.asarray(vertices, dtype=np.int64)
    partitions = np.asarray(partitions, dtype=np.int64)
    if vertices.ndim != 1 or partitions.ndim != 1:
        raise ValueError("cuGraph vertex/partition output must be 1D")
    if len(vertices) != len(partitions):
        raise ValueError("cuGraph vertex/partition lengths differ")
    if len(vertices) and (
        vertices.min() < 0 or vertices.max() >= total_vertices
    ):
        raise ValueError("cuGraph returned an out-of-range vertex ID")
    if len(np.unique(vertices)) != len(vertices):
        raise ValueError("cuGraph returned duplicate vertex IDs")
    if len(partitions) and (partitions < 0).any():
        raise ValueError("cuGraph returned a negative partition ID")

    labels = np.full(total_vertices, -1, dtype=np.int64)
    labels[vertices] = partitions
    isolates = np.flatnonzero(labels < 0)
    if len(isolates):
        next_partition = int(partitions.max()) + 1 if len(partitions) else 0
        labels[isolates] = np.arange(
            next_partition, next_partition + len(isolates), dtype=np.int64
        )
    return renumber_communities(labels)


def _cugraph_from_csr(adjacency, cugraph, cudf, cupy):
    """Transfer one symmetric weighted CSR and return a reusable GPU graph."""
    total_vertices = adjacency.shape[0]
    if (
        total_vertices > np.iinfo(np.int32).max
        or adjacency.nnz > np.iinfo(np.int32).max
    ):
        graph_index_dtype = np.int64
    else:
        graph_index_dtype = np.int32

    cupy.cuda.Device(GPU_DEVICE).use()
    conversion_started = time.time()
    print(
        f"  [Leiden][cuGraph] CSR transfer START device={GPU_DEVICE}, "
        f"vertices={total_vertices:,}, undirected_edges={adjacency.nnz // 2:,}",
        flush=True,
    )
    offsets = cudf.Series(
        np.asarray(adjacency.indptr, dtype=graph_index_dtype)
    )
    indices = cudf.Series(
        np.asarray(adjacency.indices, dtype=graph_index_dtype)
    )
    weights = cudf.Series(np.asarray(adjacency.data, dtype=np.float32))
    graph = cugraph.Graph(directed=False)
    graph.from_cudf_adjlist(
        offsets,
        indices,
        weights,
        renumber=False,
        symmetrize=False,
    )
    cupy.cuda.get_current_stream().synchronize()
    print(
        f"  [Leiden][cuGraph] CSR transfer DONE "
        f"elapsed={time.time() - conversion_started:.1f}s",
        flush=True,
    )
    del offsets, indices, weights
    return graph


def _run_cugraph_partition(
        graph, total_vertices, resolution, cugraph, cupy):
    partition_started = time.time()
    print(
        f"  [Leiden][cuGraph] partition START max_iter={CUGRAPH_MAX_ITERATIONS}, "
        f"resolution={resolution:g}, seed={RANDOM_SEED}, "
        f"version={getattr(cugraph, '__version__', 'unknown')}",
        flush=True,
    )
    parts, modularity_score = cugraph.leiden(
        graph,
        max_iter=int(CUGRAPH_MAX_ITERATIONS),
        resolution=float(resolution),
        random_state=int(RANDOM_SEED),
    )
    cupy.cuda.get_current_stream().synchronize()
    vertices = _cudf_series_to_numpy(parts["vertex"], cupy)
    partitions = _cudf_series_to_numpy(parts["partition"], cupy)
    labels = _labels_from_vertex_partitions(
        vertices, partitions, total_vertices
    )
    print(
        f"  [Leiden][cuGraph] partition DONE "
        f"resolution={resolution:g}, elapsed={time.time() - partition_started:.1f}s, "
        f"modularity={float(modularity_score):.8f}, "
        f"communities={len(np.unique(labels)):,}",
        flush=True,
    )
    del parts, vertices, partitions
    return labels


def _run_cugraph_leiden(
        adjacency, cugraph, cudf, cupy, resolution=None):
    """Run one weighted undirected Leiden partition from a symmetric CSR."""
    resolution = LEIDEN_RESOLUTION if resolution is None else float(resolution)
    graph = _cugraph_from_csr(adjacency, cugraph, cudf, cupy)
    labels = _run_cugraph_partition(
        graph, adjacency.shape[0], resolution, cugraph, cupy
    )
    del graph
    gc.collect()
    return labels


def run_leiden_resolutions(adjacency, resolutions):
    """Evaluate resolution candidates while reusing one converted graph."""
    resolutions = tuple(float(value) for value in resolutions)
    if not resolutions or any(value <= 0 for value in resolutions):
        raise ValueError("resolutions must contain positive values")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must be unique")

    n = adjacency.shape[0]
    if n <= 1:
        print(
            f"  [Leiden] DONE (skipped: vertices={n:,}, edges=0)",
            flush=True,
        )
        return {
            resolution: np.zeros(n, dtype=np.int32)
            for resolution in resolutions
        }
    if adjacency.nnz == 0:
        print("  ⚠ kNN edge 없음: 각 pattern을 singleton community로 보존", flush=True)
        return {
            resolution: np.arange(n, dtype=np.int32)
            for resolution in resolutions
        }

    if LEIDEN_BACKEND == "cugraph":
        cugraph, cudf, cupy = _require_cugraph_stack()
        graph = _cugraph_from_csr(adjacency, cugraph, cudf, cupy)
        results = {
            resolution: _run_cugraph_partition(
                graph, n, resolution, cugraph, cupy
            )
            for resolution in resolutions
        }
        del graph
    else:
        igraph = _import_or_exit("igraph", "python-igraph")
        leidenalg = _import_or_exit("leidenalg")
        graph = _igraph_from_csr(adjacency, igraph)
        results = {
            resolution: _run_leidenalg_partition(
                graph, leidenalg, resolution
            )
            for resolution in resolutions
        }
        del graph
    gc.collect()
    return results


def run_leiden(adjacency, resolution=None):
    resolution = LEIDEN_RESOLUTION if resolution is None else float(resolution)
    return run_leiden_resolutions(adjacency, (resolution,))[resolution]


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


def majority_vote_labels(neighbor_labels):
    """Vectorized uniform kNN vote; ties resolve to the smallest label."""
    labels = np.asarray(neighbor_labels, dtype=np.int32)
    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError("neighbor_labels must be a non-empty 2D array")
    ordered = np.sort(labels, axis=1)
    best = ordered[:, 0].copy()
    best_count = np.zeros(len(ordered), dtype=np.int16)
    for column in range(ordered.shape[1]):
        candidate = ordered[:, column]
        count = np.sum(ordered == candidate[:, None], axis=1)
        replace = count > best_count
        best[replace] = candidate[replace]
        best_count[replace] = count[replace]
    return best


def assign_full_population(
        sample_embedding, sample_labels, full_embedding,
        sample_local_positions):
    """Assign every group member to sampled communities with exact 5-NN."""
    sample_embedding = np.ascontiguousarray(sample_embedding, dtype=np.float32)
    sample_labels = np.asarray(sample_labels, dtype=np.int32)
    sample_local_positions = np.asarray(sample_local_positions, dtype=np.int64)
    if len(sample_embedding) != len(sample_labels):
        raise ValueError("sample embedding/label lengths differ")
    if len(sample_local_positions) != len(sample_labels):
        raise ValueError("sample positions/label lengths differ")
    if len(full_embedding) < len(sample_embedding):
        raise ValueError("sample cannot be larger than full population")

    if (
        len(sample_embedding) == len(full_embedding)
        and np.array_equal(sample_local_positions, np.arange(len(full_embedding)))
    ):
        print("  [Assignment] skipped: entire group was sampled", flush=True)
        return sample_labels.copy(), None, "identity_all_sampled"

    neighbors = min(ASSIGN_NEIGHBORS, len(sample_embedding))
    fit_started = time.time()
    knn = _make_exact_knn_index(sample_embedding, neighbors)
    backend = getattr(knn, "_fit_method", "unknown")
    print(
        f"  [Assignment] index READY references={len(sample_embedding):,}, "
        f"k={neighbors}, method={backend}, elapsed={time.time() - fit_started:.1f}s",
        flush=True,
    )

    labels = np.empty(len(full_embedding), dtype=np.int32)
    started = time.time()
    for start in range(0, len(full_embedding), KNN_QUERY_BLOCK):
        end = min(start + KNN_QUERY_BLOCK, len(full_embedding))
        block_started = time.time()
        print(
            f"  [Assignment] QUERY START {start:,}:{end:,} / "
            f"{len(full_embedding):,}",
            flush=True,
        )
        _, neighbor_ids = knn.kneighbors(
            full_embedding[start:end],
            n_neighbors=neighbors,
            return_distance=True,
        )
        labels[start:end] = majority_vote_labels(sample_labels[neighbor_ids])
        total_elapsed = time.time() - started
        print(
            f"  [Assignment] QUERY DONE {end:,}/{len(full_embedding):,}, "
            f"{end / len(full_embedding):.1%}, "
            f"block_elapsed={time.time() - block_started:.1f}s, "
            f"total_elapsed={total_elapsed:.1f}s, "
            f"average_rows/sec={end / total_elapsed:,.1f}",
            flush=True,
        )

    # Sample vertices define the learned communities and retain their own labels.
    labels[sample_local_positions] = sample_labels
    labels = renumber_communities(labels)
    print(
        f"  [Assignment] DONE elapsed={time.time() - started:.1f}s, "
        f"communities={len(np.unique(labels)):,}",
        flush=True,
    )
    return labels, knn, backend


def sampled_assignment_diagnostics(
        knn, full_embedding, full_labels, full_global_rows,
        sample_labels, sample_global_rows, keys, h0_label, rng):
    """Inspect full-population anchors against their sampled reference neighbors."""
    started = time.time()
    if knn is None or DIAGNOSTIC_ANCHORS < 1:
        print(
            f"  [Diagnostics] DONE elapsed={time.time() - started:.1f}s (skipped)",
            flush=True,
        )
        return []
    anchors = rng.choice(
        len(full_embedding), min(DIAGNOSTIC_ANCHORS, len(full_embedding)),
        replace=False,
    )
    n_query = min(len(sample_global_rows), DIAGNOSTIC_NEIGHBORS + 1)
    distances, neighbors = knn.kneighbors(
        full_embedding[anchors], n_neighbors=n_query, return_distance=True
    )
    rows = []
    for query_row, anchor in enumerate(anchors.tolist()):
        anchor_global_row = int(full_global_rows[anchor])
        rank = 0
        for neighbor, distance in zip(
                neighbors[query_row].tolist(), distances[query_row].tolist()):
            neighbor_global_row = int(sample_global_rows[neighbor])
            if neighbor_global_row == anchor_global_row:
                continue
            rank += 1
            rows.append({
                "coarse_group": coarse_group_name(h0_label),
                "anchor_key": str(keys[anchor_global_row]),
                "neighbor_rank": rank,
                "neighbor_key": str(keys[neighbor_global_row]),
                "distance": float(distance),
                "same_final_cluster": bool(
                    full_labels[anchor] == sample_labels[neighbor]
                ),
            })
            if rank == DIAGNOSTIC_NEIGHBORS:
                break
    print(
        f"  [Diagnostics] DONE elapsed={time.time() - started:.1f}s, "
        f"rows={len(rows):,}",
        flush=True,
    )
    return rows


def summarize_and_select_representatives(
        h0_label, local_labels, total_count, embedding, global_rows, keys):
    """Compute full-population statistics and one real centroid-nearest member."""
    started = time.time()
    labels = np.asarray(local_labels, dtype=np.int32)
    ids = np.unique(labels)
    if not np.array_equal(ids, np.arange(len(ids), dtype=ids.dtype)):
        raise ValueError("local community labels must be contiguous from zero")
    community_count = len(ids)
    counts = np.bincount(labels, minlength=community_count).astype(np.int64)

    centroid_sums = np.empty(
        (community_count, embedding.shape[1]), dtype=np.float64
    )
    for dimension in range(embedding.shape[1]):
        centroid_sums[:, dimension] = np.bincount(
            labels,
            weights=embedding[:, dimension],
            minlength=community_count,
        )
    centroids = (centroid_sums / counts[:, None]).astype(np.float32)
    del centroid_sums

    distances = np.empty(len(embedding), dtype=np.float32)
    distance_sums = np.zeros(community_count, dtype=np.float64)
    distance_max = np.zeros(community_count, dtype=np.float32)
    for start in range(0, len(embedding), NORMALIZE_BLOCK):
        end = min(start + NORMALIZE_BLOCK, len(embedding))
        block_labels = labels[start:end]
        delta = embedding[start:end] - centroids[block_labels]
        block_distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        distances[start:end] = block_distances
        distance_sums += np.bincount(
            block_labels,
            weights=block_distances,
            minlength=community_count,
        )
        np.maximum.at(distance_max, block_labels, block_distances)

    order = np.argsort(labels, kind="stable")
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    summaries = []
    representatives = []
    for community in range(community_count):
        members = order[boundaries[community]:boundaries[community + 1]]
        member_distances = distances[members]
        representative_local = int(members[np.argmin(member_distances)])
        final_cluster = final_cluster_id(h0_label, community)
        summaries.append({
            "h0_label": int(h0_label),
            "coarse_group": coarse_group_name(h0_label),
            "topology_cluster": community,
            "final_cluster": final_cluster,
            "pattern_count": int(counts[community]),
            "fraction_in_coarse_group": float(
                counts[community] / len(labels)
            ),
            "fraction_in_total": float(counts[community] / total_count),
            "centroid_distance_mean": float(
                distance_sums[community] / counts[community]
            ),
            "centroid_distance_p95": float(
                np.percentile(member_distances, 95)
            ),
            "centroid_distance_max": float(distance_max[community]),
        })
        global_row = int(global_rows[representative_local])
        representatives.append({
            "pattern_key": str(keys[global_row]),
            "global_row": global_row,
            "h0_label": int(h0_label),
            "topology_cluster": community,
            "final_cluster": final_cluster,
            "community_pattern_count": int(counts[community]),
            "selection_reason": "centroid_nearest_full_population",
            "distance_to_centroid": float(distances[representative_local]),
        })

    print(
        f"  [Summary/representatives] DONE communities={community_count:,}, "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    return summaries, representatives


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
    print(f"  sample k / graph edges: {k} / {edge_count:,}", flush=True)
    print(f"  Leiden communities    : {len(counts):,}", flush=True)
    print("  community size        : "
          f"min={counts.min():,}, median={np.median(counts):,.1f}, "
          f"mean={counts.mean():,.1f}, max={counts.max():,}", flush=True)


def write_outputs(
        keys, rows, h0_labels, topology_labels, summaries, diagnostics,
        representatives=None, metadata=None, sample_rows=None):
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
    labels_payload = {
        "rows": rows,
        "h0_labels": h0_labels,
        "topology_labels": topology_labels,
        "population_fingerprint": np.asarray(fingerprint_digest.hexdigest()),
    }
    if sample_rows is not None:
        labels_payload["sample_rows"] = np.asarray(sample_rows, dtype=np.int64)
    if metadata is not None and "selected_resolution" in metadata:
        labels_payload["selected_resolution"] = np.asarray(
            metadata["selected_resolution"], dtype=np.float64
        )
    np.savez(labels_tmp, **labels_payload)
    os.replace(labels_tmp, labels_path)

    saved_paths = [assignment_path, summary_path, diagnostic_path, labels_path]
    if representatives is not None:
        representative_path = os.path.join(
            OUT_DIR, "topology_representatives.csv"
        )
        representative_tmp = representative_path + ".tmp"
        representative_fields = [
            "pattern_key", "global_row", "h0_label", "topology_cluster",
            "final_cluster", "representative_rank", "community_pattern_count",
            "selection_reason", "distance_to_centroid",
        ]
        with open(representative_tmp, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=representative_fields)
            writer.writeheader()
            writer.writerows(representatives)
        os.replace(representative_tmp, representative_path)

        representative_labels_path = os.path.join(
            OUT_DIR, "topology_representatives.npz"
        )
        representative_labels_tmp = representative_labels_path + ".tmp.npz"
        np.savez(
            representative_labels_tmp,
            rows=np.asarray(
                [row["global_row"] for row in representatives], dtype=np.int64
            ),
            h0_labels=np.asarray(
                [row["h0_label"] for row in representatives], dtype=np.int32
            ),
            topology_labels=np.asarray(
                [row["topology_cluster"] for row in representatives],
                dtype=np.int32,
            ),
        )
        os.replace(representative_labels_tmp, representative_labels_path)
        saved_paths.extend((representative_path, representative_labels_path))

    if metadata is not None:
        metadata = dict(metadata)
        metadata["population_fingerprint"] = fingerprint_digest.hexdigest()
        metadata_path = os.path.join(OUT_DIR, "topology_run_metadata.json")
        metadata_tmp = metadata_path + ".tmp"
        with open(metadata_tmp, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, ensure_ascii=False, indent=2)
        os.replace(metadata_tmp, metadata_path)
        saved_paths.append(metadata_path)

    print("\n[Saved]", flush=True)
    for path in saved_paths:
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

    positions_by_group = {
        h0_label: np.flatnonzero(h0_labels == h0_label)
        for h0_label in coarse_groups
    }
    group_sizes = np.asarray(
        [len(positions_by_group[h0_label]) for h0_label in coarse_groups],
        dtype=np.int64,
    )
    sample_counts = allocate_stratified_sample_counts(group_sizes, SAMPLE_BUDGET)
    actual_sample_count = int(sample_counts.sum())
    sample_resolution_scale = max(1.0, len(rows) / actual_sample_count)
    resolution_candidates = tuple(sorted(set(
        [float(LEIDEN_RESOLUTION)]
        + [
            min(
                max(
                    float(value) * sample_resolution_scale,
                    LEIDEN_RESOLUTION_MIN,
                ),
                LEIDEN_RESOLUTION_MAX,
            )
            for value in LEIDEN_RESOLUTION_CANDIDATES
        ]
    )))
    tolerance_min = int(np.ceil(COMMUNITY_TARGET * (1.0 - COMMUNITY_TOLERANCE)))
    tolerance_max = int(np.floor(COMMUNITY_TARGET * (1.0 + COMMUNITY_TOLERANCE)))

    print("=" * 72, flush=True)
    print("Integrated sampled topology clustering + representatives", flush=True)
    print("=" * 72, flush=True)
    print(f"Total patterns             : {len(rows):,}", flush=True)
    print(f"Sample patterns            : {actual_sample_count:,}", flush=True)
    print(f"Number of H0 coarse groups : {len(dense_groups)}", flush=True)
    print(f"Rare/noise patterns        : {rare_count:,}", flush=True)
    print(f"Rare fraction              : {rare_count / len(rows):.2%}", flush=True)
    print(f"Feature dimensions         : {dims} (total={sum(dims.values())})", flush=True)
    print(
        f"k={K_NEIGHBORS}, assign_k={ASSIGN_NEIGHBORS}, "
        f"edge={EDGE_WEIGHT_MODE}, mutual={MUTUAL_KNN}, seed={RANDOM_SEED}",
        flush=True,
    )
    print(
        f"Community target           : {COMMUNITY_TARGET:,} "
        f"(tolerance={COMMUNITY_TOLERANCE:.1%}, "
        f"range={tolerance_min:,}..{tolerance_max:,})",
        flush=True,
    )
    print(
        "Base resolution candidates  : "
        + ", ".join(f"{value:g}" for value in LEIDEN_RESOLUTION_CANDIDATES),
        flush=True,
    )
    print(f"Sample resolution scale    : {sample_resolution_scale:.6g}", flush=True)
    print(
        "Scaled resolution candidates: "
        + ", ".join(f"{value:g}" for value in resolution_candidates),
        flush=True,
    )
    print(
        f"Adaptive resolution search : rounds={LEIDEN_RESOLUTION_SEARCH_ROUNDS}, "
        f"bounds={LEIDEN_RESOLUTION_MIN:g}..{LEIDEN_RESOLUTION_MAX:g}",
        flush=True,
    )
    print(
        f"kNN backend={KNN_BACKEND}, gpu_device={GPU_DEVICE}, "
        f"torch_gpu_tiles={TORCH_GPU_QUERY_BLOCK:,}x{TORCH_GPU_REFERENCE_BLOCK:,}",
        flush=True,
    )
    print(
        f"Leiden backend={LEIDEN_BACKEND}, "
        f"cugraph_max_iter={CUGRAPH_MAX_ITERATIONS}, "
        f"cpu_iterations={CPU_LEIDEN_ITERATIONS}",
        flush=True,
    )

    scalers = fit_block_scalers(features, rows)
    resolution_counts = {
        float(resolution): 0
        for resolution in resolution_candidates
    }
    sample_models = []
    sample_backends = set()
    total_groups = len(coarse_groups)
    print("\n[Phase 1/2] sampled kNN graphs and Leiden resolution search", flush=True)
    for group_index, (h0_label, sample_count) in enumerate(
            zip(coarse_groups, sample_counts.tolist()), start=1):
        group_started = time.time()
        positions = positions_by_group[h0_label]
        if sample_count == len(positions):
            sample_local_positions = np.arange(len(positions), dtype=np.int64)
        else:
            sample_local_positions = np.sort(
                rng.choice(len(positions), size=sample_count, replace=False)
            ).astype(np.int64, copy=False)
        sample_population_positions = positions[sample_local_positions]
        sample_global_rows = rows[sample_population_positions]
        print("\n" + "-" * 72, flush=True)
        print(
            f"[Group {group_index}/{total_groups}] START "
            f"coarse_group={coarse_group_name(h0_label)}, "
            f"patterns={len(positions):,}, sample={sample_count:,}",
            flush=True,
        )
        sample_embedding = make_normalized_embedding(
            features, scalers, sample_global_rows, dims
        )
        adjacency, sample_knn, k = build_knn_graph(sample_embedding)
        sample_backend = getattr(sample_knn, "_fit_method", "none")
        sample_backends.add(sample_backend)
        edge_count = int(adjacency.nnz // 2)
        labels_by_resolution = run_leiden_resolutions(
            adjacency, resolution_candidates
        )
        group_counts = {}
        for resolution, labels in labels_by_resolution.items():
            count = int(labels.max()) + 1 if len(labels) else 0
            group_counts[resolution] = count
            resolution_counts[resolution] += count
        print(
            "  [Resolution counts] "
            + ", ".join(
                f"{resolution:g}={group_counts[resolution]:,}"
                for resolution in resolution_candidates
            ),
            flush=True,
        )
        sample_models.append({
            "h0_label": int(h0_label),
            "sample_local_positions": sample_local_positions,
            "sample_global_rows": sample_global_rows,
            "sample_embedding": sample_embedding,
            "adjacency": adjacency,
            "labels_by_resolution": labels_by_resolution,
            "sample_k": int(k),
            "sample_edges": edge_count,
            "sample_backend": sample_backend,
        })
        del sample_knn
        gc.collect()
        print(
            f"[Group {group_index}/{total_groups}] DONE "
            f"coarse_group={coarse_group_name(h0_label)}, "
            f"elapsed={time.time() - group_started:.1f}s",
            flush=True,
        )

    selected_resolution, within_tolerance = choose_resolution(
        resolution_counts,
        target=COMMUNITY_TARGET,
        tolerance=COMMUNITY_TOLERANCE,
    )
    resolution_search_rounds = 0
    while (
        not within_tolerance
        and resolution_search_rounds < LEIDEN_RESOLUTION_SEARCH_ROUNDS
    ):
        candidate = next_resolution_candidate(
            resolution_counts,
            target=COMMUNITY_TARGET,
            minimum_resolution=LEIDEN_RESOLUTION_MIN,
            maximum_resolution=LEIDEN_RESOLUTION_MAX,
        )
        if candidate is None:
            break

        resolution_search_rounds += 1
        print(
            f"\n[Adaptive resolution {resolution_search_rounds}/"
            f"{LEIDEN_RESOLUTION_SEARCH_ROUNDS}] START resolution={candidate:g}",
            flush=True,
        )
        candidate_count = 0
        for group_index, model in enumerate(sample_models, start=1):
            labels = run_leiden_resolutions(
                model["adjacency"], (candidate,)
            )[candidate]
            model["labels_by_resolution"][candidate] = labels
            group_count = int(labels.max()) + 1 if len(labels) else 0
            candidate_count += group_count
            print(
                f"  [Adaptive group {group_index}/{total_groups}] "
                f"coarse_group={coarse_group_name(model['h0_label'])}, "
                f"communities={group_count:,}",
                flush=True,
            )
        resolution_counts[candidate] = candidate_count
        selected_resolution, within_tolerance = choose_resolution(
            resolution_counts,
            target=COMMUNITY_TARGET,
            tolerance=COMMUNITY_TOLERANCE,
        )
        print(
            f"[Adaptive resolution {resolution_search_rounds}/"
            f"{LEIDEN_RESOLUTION_SEARCH_ROUNDS}] DONE "
            f"resolution={candidate:g}, communities={candidate_count:,}, "
            f"best={selected_resolution:g}/"
            f"{resolution_counts[selected_resolution]:,}, "
            f"inside_tolerance={within_tolerance}",
            flush=True,
        )

    selected_sample_communities = resolution_counts[selected_resolution]
    print("\n" + "=" * 72, flush=True)
    print("Resolution search result", flush=True)
    for resolution in sorted(resolution_counts):
        marker = "  <-- selected" if resolution == selected_resolution else ""
        print(
            f"  resolution={resolution:g}: "
            f"communities={resolution_counts[resolution]:,}{marker}",
            flush=True,
        )
    if within_tolerance:
        print(
            f"Selected resolution={selected_resolution:g}, "
            f"communities={selected_sample_communities:,} "
            f"(inside {tolerance_min:,}..{tolerance_max:,})",
            flush=True,
        )
    else:
        print(
            f"⚠ No candidate produced {tolerance_min:,}..{tolerance_max:,}; "
            f"selected closest resolution={selected_resolution:g}, "
            f"communities={selected_sample_communities:,}",
            flush=True,
        )
    print("=" * 72, flush=True)

    if not within_tolerance:
        raise RuntimeError(
            f"adaptive resolution search failed to produce "
            f"{tolerance_min:,}..{tolerance_max:,} communities; "
            f"closest={selected_sample_communities:,} at "
            f"resolution={selected_resolution:g}. No representative output written."
        )

    for model in sample_models:
        selected_labels = model["labels_by_resolution"][selected_resolution]
        model["labels_by_resolution"] = {
            selected_resolution: selected_labels
        }
        del model["adjacency"]
    gc.collect()

    topology_labels = np.full(len(rows), -1, dtype=np.int32)
    summaries, diagnostics, representatives = [], [], []
    assignment_backends = set()
    print("\n[Phase 2/2] full-population assignment and representatives", flush=True)
    for group_index, model in enumerate(sample_models, start=1):
        group_started = time.time()
        h0_label = model["h0_label"]
        positions = positions_by_group[h0_label]
        global_rows = rows[positions]
        sample_local_positions = model["sample_local_positions"]
        sample_labels = model["labels_by_resolution"][selected_resolution]
        print("\n" + "-" * 72, flush=True)
        print(
            f"[Assign group {group_index}/{total_groups}] START "
            f"coarse_group={coarse_group_name(h0_label)}, "
            f"patterns={len(positions):,}, "
            f"sample={len(sample_local_positions):,}",
            flush=True,
        )
        full_embedding = make_normalized_embedding(
            features, scalers, global_rows, dims
        )
        local_labels, assignment_knn, assignment_backend = assign_full_population(
            model["sample_embedding"],
            sample_labels,
            full_embedding,
            sample_local_positions,
        )
        assignment_backends.add(assignment_backend)
        topology_labels[positions] = local_labels

        group_summaries, group_representatives = (
            summarize_and_select_representatives(
                h0_label,
                local_labels,
                len(rows),
                full_embedding,
                global_rows,
                keys,
            )
        )
        summaries.extend(group_summaries)
        representatives.extend(group_representatives)
        sample_final_labels = local_labels[sample_local_positions]
        diagnostics.extend(sampled_assignment_diagnostics(
            assignment_knn,
            full_embedding,
            local_labels,
            global_rows,
            sample_final_labels,
            model["sample_global_rows"],
            keys,
            h0_label,
            rng,
        ))
        _print_group_diagnostics(
            h0_label,
            len(positions),
            model["sample_k"],
            local_labels,
            model["sample_edges"],
        )
        del (
            full_embedding, local_labels, assignment_knn,
            sample_final_labels, group_summaries, group_representatives,
            model["sample_embedding"], model["labels_by_resolution"],
        )
        gc.collect()
        print(
            f"[Assign group {group_index}/{total_groups}] DONE "
            f"coarse_group={coarse_group_name(h0_label)}, "
            f"elapsed={time.time() - group_started:.1f}s",
            flush=True,
        )

    if (topology_labels < 0).any():
        raise RuntimeError("일부 pattern에 topology community가 배정되지 않음")
    if len(representatives) != len(summaries):
        raise RuntimeError("community와 representative 수가 다름")
    if len(summaries) != selected_sample_communities:
        raise RuntimeError(
            "sample partition community 수와 full assignment community 수가 다름"
        )
    representative_rows = [row["global_row"] for row in representatives]
    if len(set(representative_rows)) != len(representative_rows):
        raise RuntimeError("representative global_row가 중복됨")
    for rank, representative in enumerate(representatives, start=1):
        representative["representative_rank"] = rank

    sizes = np.asarray([row["pattern_count"] for row in summaries])
    rare_communities = sum(row["h0_label"] < 0 for row in summaries)
    print("\n" + "=" * 72, flush=True)
    print(f"Total final communities    : {len(summaries):,}", flush=True)
    print(f"Largest final community    : {sizes.max():,}", flush=True)
    print(f"Smallest final community   : {sizes.min():,}", flush=True)
    print(f"Rare-group community count : {rare_communities:,}", flush=True)
    print(f"Representative patterns    : {len(representatives):,}", flush=True)
    print(
        f"Target tolerance satisfied : "
        f"{tolerance_min <= len(representatives) <= tolerance_max}",
        flush=True,
    )
    print("=" * 72, flush=True)

    sample_rows = np.concatenate([
        model["sample_global_rows"] for model in sample_models
    ]).astype(np.int64, copy=False)
    if (
        len(sample_rows) != actual_sample_count
        or len(np.unique(sample_rows)) != len(sample_rows)
    ):
        raise RuntimeError("sample row 수가 잘못되었거나 중복됨")
    metadata = {
        "method": "stratified_sample_leiden_full_5nn_assignment",
        "total_patterns": int(len(rows)),
        "sample_budget": int(SAMPLE_BUDGET),
        "actual_sample_patterns": actual_sample_count,
        "assignment_neighbors": int(ASSIGN_NEIGHBORS),
        "community_target": int(COMMUNITY_TARGET),
        "community_tolerance": float(COMMUNITY_TOLERANCE),
        "community_tolerance_min": tolerance_min,
        "community_tolerance_max": tolerance_max,
        "selected_resolution": float(selected_resolution),
        "selected_resolution_inside_tolerance": bool(within_tolerance),
        "sample_resolution_scale": float(sample_resolution_scale),
        "resolution_search_rounds": int(resolution_search_rounds),
        "scaled_initial_resolution_candidates": [
            float(value) for value in resolution_candidates
        ],
        "resolution_community_counts": {
            f"{resolution:g}": int(resolution_counts[resolution])
            for resolution in sorted(resolution_counts)
        },
        "final_communities": int(len(summaries)),
        "representative_patterns": int(len(representatives)),
        "sample_knn_backends": sorted(sample_backends),
        "assignment_knn_backends": sorted(assignment_backends),
        "leiden_backend": LEIDEN_BACKEND,
        "leiden_max_iterations": int(CUGRAPH_MAX_ITERATIONS),
        "k_neighbors": int(K_NEIGHBORS),
        "edge_weight_mode": EDGE_WEIGHT_MODE,
        "mutual_knn": bool(MUTUAL_KNN),
        "random_seed": int(RANDOM_SEED),
        "feature_dimensions": dims,
        "feature_blocks": list(FEATURE_BLOCKS),
        "block_weights": BLOCK_WEIGHTS,
        "elapsed_seconds_before_output": float(time.time() - started),
    }
    write_outputs(
        keys,
        rows,
        h0_labels,
        topology_labels,
        summaries,
        diagnostics,
        representatives=representatives,
        metadata=metadata,
        sample_rows=sample_rows,
    )
    print(f"완료: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
