"""REF-fixed landmark kPCA coverage of the real representatives from Stage 6.

Stage 7 compares REF with the topology representative set (B), not yet with an
external method (A). Stage 8 reuses the saved frame, REF rows and radius,
and calls the topology set A and the external B.txt set B (legacy fields stay).
The 40 inputs are the cached learned h0/h1/h2/h3/edge embeddings, not 21 raw
handcrafted features. Stage-6 selection stays fixed; real diagnostic representatives
come only from nonempty heatmap bins whose sampled REF are all uncovered.

Dependencies: numpy, scipy, matplotlib, threadpoolctl and torch (Stage-4 cache).
RBF kernels use PyTorch CUDA when available, otherwise vectorized SciPy.
Projection and the small landmark ARPACK eigensolve use CPU BLAS threads.
No new RAPIDS package is needed.
"""

import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np


# ============================================================================
# CONFIG
# ============================================================================
_here = Path(__file__).resolve().parent
CACHE_PATH = _here / "hkeys_features.pt"
STAGE6_DIR = _here / "topology_clustering_out"
OUT_DIR = _here / "topology_analysis_out"

# None: all valid Stage-6 population rows are the REF candidate population.
# To use a specific REF group, provide a CSV with a unique pattern_key column.
# Keys must exist in the same cache AND the Stage-6 population.
REF_PATTERN_KEYS_CSV = None
N_REF = 20_000
N_LANDMARKS = 2_000
N_COMPONENTS = 3
RANDOM_SEED = 0
GAMMA = None                  # 1 / (2 * median squared REF-landmark distance)
GAMMA_PAIR_CAP = 2_000
RADIUS_BUDGET = 1_000          # Fixed comparison budget, NOT len(A) or len(B).
R_MULT = 1.0
TRANSFORM_BLOCK = 2_048        # Never materialize a 20K x 20K kernel.
DISTANCE_BLOCK = 2_048
KERNEL_BACKEND = "auto"        # auto | torch_gpu | numpy_cpu
GPU_DEVICE = 0
CPU_THREADS = 8
MAKE_PLOTS = True
HEATMAP_BINS = 50             # Shared by heatmaps AND 100%-gap candidate selection.
GAP_RADIUS_MULTIPLIER = 1.0   # Diagnostic grouping radius = saved coverage R * this.
GAP_PREVIEW_GROUPS = 20       # Labels/table preview only; all groups are exported.

FEATURE_BLOCKS = ("h0", "h1", "h2", "h3", "edge")
SCHEMA_VERSION = 1


def log(message):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def validate_config():
    for name, value, minimum in (
        ("N_REF", N_REF, 2), ("N_LANDMARKS", N_LANDMARKS, 2),
        ("N_COMPONENTS", N_COMPONENTS, 1), ("RADIUS_BUDGET", RADIUS_BUDGET, 1),
        ("GAMMA_PAIR_CAP", GAMMA_PAIR_CAP, 2),
        ("TRANSFORM_BLOCK", TRANSFORM_BLOCK, 1),
        ("DISTANCE_BLOCK", DISTANCE_BLOCK, 1), ("CPU_THREADS", CPU_THREADS, 1),
        ("HEATMAP_BINS", HEATMAP_BINS, 2),
        ("GAP_PREVIEW_GROUPS", GAP_PREVIEW_GROUPS, 1),
    ):
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if N_COMPONENTS >= min(N_REF, N_LANDMARKS):
        raise ValueError("N_COMPONENTS must be smaller than REF/landmark counts")
    if not np.isfinite(R_MULT) or R_MULT <= 0:
        raise ValueError("R_MULT must be finite and > 0")
    if GAMMA is not None and (not np.isfinite(GAMMA) or GAMMA <= 0):
        raise ValueError("GAMMA must be None or finite and > 0")
    if not np.isfinite(GAP_RADIUS_MULTIPLIER) or GAP_RADIUS_MULTIPLIER <= 0:
        raise ValueError("GAP_RADIUS_MULTIPLIER must be finite and > 0")
    if KERNEL_BACKEND not in {"auto", "torch_gpu", "numpy_cpu"}:
        raise ValueError("invalid KERNEL_BACKEND")
    if GPU_DEVICE < 0:
        raise ValueError("GPU_DEVICE must be >= 0")


def integer_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite 1D integer array")
    integers = values.astype(np.int64)
    if not np.array_equal(values, integers):
        raise ValueError(f"{name} must contain integers")
    return integers


def population_fingerprint(keys, rows):
    """The same ordered row/key fingerprint written by Stage 6."""
    digest = hashlib.sha256()
    for row in rows:
        encoded = str(keys[int(row)]).encode("utf-8")
        digest.update(int(row).to_bytes(8, "little", signed=True))
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def read_feature_cache(path):
    import torch

    # This is the trusted cache produced by Stage 4, not a downloaded model.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_stage6_inputs():
    started = time.perf_counter()
    log("[Input] START Stage-6 labels/representatives and cached embeddings")
    cache = read_feature_cache(CACHE_PATH)
    features, keys = cache["features"], cache["keys"]
    with (STAGE6_DIR / "topology_run_metadata.json").open(encoding="utf-8") as fp:
        metadata = json.load(fp)
    if tuple(metadata["feature_blocks"]) != FEATURE_BLOCKS:
        raise ValueError("Stage-6 feature block order differs from h0/h1/h2/h3/edge")
    dimensions = metadata["feature_dimensions"]
    weights = metadata["block_weights"]
    for name in FEATURE_BLOCKS:
        expected = (len(keys), int(dimensions[name]))
        if expected[1] < 1 or tuple(features[name].shape) != expected:
            raise ValueError(f"cached {name} dimensions differ from Stage 6: {expected}")
        if not np.isfinite(weights[name]) or weights[name] <= 0:
            raise ValueError(f"invalid Stage-6 block weight: {name}")

    with np.load(STAGE6_DIR / "topology_labels.npz", allow_pickle=False) as data:
        rows = integer_vector(data["rows"], "Stage-6 rows")
        h0 = integer_vector(data["h0_labels"], "Stage-6 h0_labels")
        labels = integer_vector(data["topology_labels"], "Stage-6 topology_labels")
        fingerprint = str(data["population_fingerprint"].item())
    if (len(rows) < 2 or len(np.unique(rows)) != len(rows)
            or rows.min() < 0 or rows.max() >= len(keys)):
        raise ValueError("invalid/duplicate Stage-6 population rows")
    if h0.shape != rows.shape or labels.shape != rows.shape or (labels < 0).any():
        raise ValueError("invalid Stage-6 label alignment")
    if (h0 < -1).any():
        raise ValueError("invalid Stage-6 H0 label")
    if "total_patterns" in metadata and len(rows) != metadata["total_patterns"]:
        raise ValueError("population count differs from Stage-6 metadata")
    if (fingerprint != metadata["population_fingerprint"]
            or fingerprint != population_fingerprint(keys, rows)):
        raise ValueError("Stage-6 population fingerprint differs from the current cache")

    with (STAGE6_DIR / "topology_representatives.csv").open(
            newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        required = {"global_row", "pattern_key", "h0_label", "topology_cluster"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("invalid Stage-6 representative manifest columns")
        representatives = list(reader)
    if not representatives:
        raise ValueError("Stage-6 representative set is empty")
    rep_rows = np.asarray([int(row["global_row"]) for row in representatives])
    if len(np.unique(rep_rows)) != len(rep_rows):
        raise ValueError("duplicate representative global rows")
    rep_positions = positions_for_rows(rows, rep_rows)
    for record, row, pos in zip(representatives, rep_rows, rep_positions):
        if (str(keys[row]) != record["pattern_key"]
                or int(record["h0_label"]) != h0[pos]
                or int(record["topology_cluster"]) != labels[pos]):
            raise ValueError(f"representative key/label mismatch at global_row={row}")
    if (len(rep_rows) != metadata["representative_patterns"]
            or len(rep_rows) != metadata["final_communities"]):
        raise ValueError("representative count differs from Stage-6 metadata")
    pairs = set(zip(h0[rep_positions].tolist(), labels[rep_positions].tolist()))
    if len(pairs) != len(rep_rows):
        raise ValueError("multiple representatives for the same final community")
    multiplier = np.concatenate([
        np.full(int(dimensions[name]), math.sqrt(weights[name] / dimensions[name]))
        for name in FEATURE_BLOCKS
    ])
    feature_names = [
        f"{name}_{index}" for name in FEATURE_BLOCKS
        for index in range(int(dimensions[name]))
    ]
    log(f"[Input] DONE population={len(rows):,}, representatives={len(rep_rows):,}, "
        f"dimensions={len(feature_names)}, elapsed={time.perf_counter() - started:.1f}s")
    return {
        "features": features, "keys": keys, "rows": rows, "h0_labels": h0,
        "topology_labels": labels, "rep_rows": rep_rows,
        "rep_positions": rep_positions, "metadata": metadata,
        "feature_names": feature_names, "block_multiplier": multiplier,
        "population_fingerprint": fingerprint,
    }


def positions_for_rows(population_rows, requested_rows):
    order = np.argsort(population_rows)
    sorted_rows = population_rows[order]
    positions = np.searchsorted(sorted_rows, requested_rows)
    if (positions >= len(sorted_rows)).any():
        raise ValueError("requested global row is outside Stage-6 population")
    if not np.array_equal(sorted_rows[positions], requested_rows):
        raise ValueError("requested global row is outside Stage-6 population")
    return order[positions]


def reference_population_rows(population_rows, keys, manifest_path=None):
    if manifest_path is None:
        return population_rows.copy()
    with Path(manifest_path).open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        if "pattern_key" not in (reader.fieldnames or []):
            raise ValueError("REF manifest requires a pattern_key column")
        requested = [row["pattern_key"] for row in reader]
    if not requested or any(not key for key in requested):
        raise ValueError("REF manifest has empty keys")
    wanted = set(requested)
    if len(wanted) != len(requested):
        raise ValueError("REF manifest has duplicate pattern keys")
    found = {}
    for row in population_rows:
        key = str(keys[int(row)])
        if key in wanted:
            if key in found:
                raise ValueError(f"ambiguous REF pattern_key in cache: {key}")
            found[key] = int(row)
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"REF keys missing from Stage-6 population: {sorted(missing)[:5]}")
    return np.sort(np.asarray(list(found.values()), dtype=np.int64))


def sample_reference_rows(rows, budget, seed):
    """Uniform without replacement; independent of representative membership."""
    rows = np.sort(integer_vector(rows, "REF rows"))
    if len(rows) < 2 or len(np.unique(rows)) != len(rows) or budget < 2:
        raise ValueError("REF needs >= 2 unique rows and sample budget >= 2")
    if len(rows) <= budget:
        return rows.copy()
    return np.sort(np.random.default_rng(seed).choice(rows, budget, replace=False))


def gather_raw_embeddings(features, rows):
    """Read only selected rows from the unnormalized cached learned blocks."""
    parts = []
    for name in FEATURE_BLOCKS:
        value = features[name]
        if isinstance(value, np.ndarray):
            part = value[rows]
        else:
            import torch
            part = value.index_select(0, torch.from_numpy(rows)).float().numpy()
        parts.append(np.asarray(part, dtype=np.float64))
    raw = np.concatenate(parts, axis=1)
    if not np.isfinite(raw).all():
        raise ValueError("selected raw embedding contains NaN/Inf; no rows silently dropped")
    return raw


def choose_kernel_backend(requested, device):
    if requested == "numpy_cpu":
        return "numpy_cpu"
    try:
        import torch
    except ImportError:
        if requested == "torch_gpu":
            raise RuntimeError("PyTorch CUDA requested but torch is not installed")
        return "numpy_cpu"
    if torch.cuda.is_available() and device < torch.cuda.device_count():
        return "torch_gpu"
    if requested == "torch_gpu":
        raise RuntimeError(f"PyTorch CUDA device {device} is unavailable")
    return "numpy_cpu"


def rbf_kernel(x, landmarks, gamma, backend="numpy_cpu", device=0):
    """Float64 Gaussian kernel, matching sklearn's exp(-gamma * squared L2)."""
    if backend == "torch_gpu":
        import torch
        with torch.no_grad():
            a = torch.as_tensor(x, dtype=torch.float64, device=f"cuda:{device}")
            b = torch.as_tensor(landmarks, dtype=torch.float64, device=f"cuda:{device}")
            d2 = ((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :]
                  - 2.0 * (a @ b.T)).clamp_min_(0)
            result = torch.exp(d2.mul_(-gamma)).cpu().numpy()
        return result
    from scipy.spatial.distance import cdist
    d2 = cdist(x, landmarks, metric="sqeuclidean")
    return np.exp(-gamma * d2)


def median_gamma(landmarks, cap, seed):
    from scipy.spatial.distance import pdist
    indices = np.random.default_rng(seed).choice(
        len(landmarks), min(cap, len(landmarks)), replace=False
    )
    median = float(np.median(pdist(landmarks[indices], "sqeuclidean")))
    if median <= 0:
        log("[kPCA] WARNING zero median pair distance; gamma=1/input_dim")
        return 1.0 / landmarks.shape[1]
    return 1.0 / (2.0 * median)


def fit_reference_frame(raw_ref, multiplier, landmark_count, components, seed,
                        gamma=None, backend="numpy_cpu", device=0):
    """Fit ONLY on REF; save numeric arrays for version-independent transforms."""
    from scipy.sparse.linalg import eigsh

    raw_ref = np.asarray(raw_ref, dtype=np.float64)
    multiplier = np.asarray(multiplier, dtype=np.float64)
    if (raw_ref.ndim != 2 or len(raw_ref) < 2 or not np.isfinite(raw_ref).all()
            or multiplier.shape != (raw_ref.shape[1],)
            or not np.isfinite(multiplier).all() or (multiplier <= 0).any()):
        raise ValueError("invalid REF embeddings/block multiplier")
    m = min(landmark_count, len(raw_ref))
    if not 1 <= components < m:
        raise ValueError("need more REF landmarks than requested kPCA components")
    center, scale = raw_ref.mean(0), raw_ref.std(0)
    scale[scale == 0] = 1.0
    normalized = (raw_ref - center) / scale * multiplier
    landmark_indices = np.sort(
        np.random.default_rng(seed + 1).choice(len(raw_ref), m, replace=False)
    )
    landmarks = normalized[landmark_indices]
    gamma = median_gamma(landmarks, GAMMA_PAIR_CAP, seed + 2) if gamma is None else gamma
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    log(f"[kPCA] FIT START REF={len(raw_ref):,}, dim={raw_ref.shape[1]}, "
        f"landmarks={m:,}, components={components}, gamma={gamma:.8g}, "
        f"kernel_memory={m*m*8/1024**2:.1f} MiB, kernel_backend={backend}")
    started = time.perf_counter()
    kernel = rbf_kernel(landmarks, landmarks, gamma, backend, device)
    row_mean = kernel.mean(0)
    grand_mean = float(row_mean.mean())
    kernel -= row_mean[None, :]
    kernel -= row_mean[:, None]
    kernel += grand_mean
    total_mass = float(np.trace(kernel))
    if not np.isfinite(total_mass) or total_mass <= m * np.finfo(np.float64).eps:
        raise ValueError("REF kernel is degenerate; no nonzero kPCA components")
    log("[kPCA] centered kernel READY; CPU ARPACK eigensolve START")
    eigenvalues, eigenvectors = eigsh(
        kernel, k=components, which="LA", tol=1.0e-9,
        v0=np.random.default_rng(seed + 3).uniform(-1, 1, size=m),
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    if total_mass <= 0 or (eigenvalues <= max(1.0, total_mass) * 1.0e-12).any():
        raise ValueError("REF kernel is degenerate; not enough nonzero kPCA components")
    # Fix arbitrary eigenvector signs for a stable persisted coordinate frame.
    peaks = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[peaks, np.arange(components)])
    eigenvectors *= signs
    frame = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "center": center, "scale": scale, "block_multiplier": multiplier,
        "landmarks": landmarks, "landmark_sample_indices": landmark_indices,
        "gamma": np.asarray(gamma), "kernel_row_mean": row_mean,
        "kernel_grand_mean": np.asarray(grand_mean),
        "eigenvalues": eigenvalues, "eigenvectors": eigenvectors,
        "kernel_total_mass": np.asarray(total_mass),
        "kpc_scale": np.ones(components),
    }
    log(f"[kPCA] FIT DONE elapsed={time.perf_counter() - started:.1f}s, "
        f"retained_landmark_kernel_mass={eigenvalues.sum()/total_mass:.1%}")
    return frame


def normalize_embeddings(raw, frame):
    raw = np.asarray(raw, dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != len(frame["center"])
            or not np.isfinite(raw).all()):
        raise ValueError("raw embedding dimensions/finite values differ from saved frame")
    return (raw - frame["center"]) / frame["scale"] * frame["block_multiplier"]


def transform_raw_embeddings(raw, frame, block_size=TRANSFORM_BLOCK,
                             backend="numpy_cpu", device=0):
    """Public Stage-8 API: transform only; never fit or recalculate scales."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    x = normalize_embeddings(raw, frame)
    projection = frame["eigenvectors"] / np.sqrt(frame["eigenvalues"])[None, :]
    result = np.empty((len(x), projection.shape[1]), dtype=np.float64)
    started = time.perf_counter()
    for start in range(0, len(x), block_size):
        end = min(start + block_size, len(x))
        kernel = rbf_kernel(x[start:end], frame["landmarks"], float(frame["gamma"]),
                            backend, device)
        kernel -= kernel.mean(1)[:, None]
        kernel -= frame["kernel_row_mean"][None, :]
        kernel += float(frame["kernel_grand_mean"])
        result[start:end] = kernel @ projection
        log(f"[kPCA] TRANSFORM {end:,}/{len(x):,} ({end/len(x):.1%}), "
            f"elapsed={time.perf_counter() - started:.1f}s")
    if not np.isfinite(result).all():
        raise ValueError("kPCA projection produced nonfinite coordinates")
    return result


def representation_radius(reference_coordinates, budget, multiplier=1.0):
    """REF-only radius at the nominal budget; include all REF, even rare tails."""
    from scipy.spatial import cKDTree
    reference_coordinates = np.asarray(reference_coordinates, dtype=np.float64)
    n = len(reference_coordinates)
    if n < 2 or budget < 1 or not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid radius inputs")
    k = min(n - 1, max(1, math.ceil(n / budget)))
    # List k returns only the required order statistic, not an n x (k+1) matrix.
    distances, _ = cKDTree(reference_coordinates).query(
        reference_coordinates, k=[k + 1], workers=CPU_THREADS
    )
    radius = float(np.median(distances[:, 0]) * multiplier)
    if radius == 0:
        log("[Radius] WARNING radius=0 from repeated REF coordinates; not inflated")
    return radius, k


def nearest_distances(reference, sample, block_size=DISTANCE_BLOCK):
    from scipy.spatial.distance import cdist
    reference, sample = np.asarray(reference), np.asarray(sample)
    if (reference.ndim != 2 or sample.ndim != 2 or len(sample) == 0
            or reference.shape[1] != sample.shape[1]
            or not np.isfinite(reference).all() or not np.isfinite(sample).all()
            or block_size < 1):
        raise ValueError("invalid nearest-distance inputs")
    nearest = np.empty(len(reference), dtype=np.int64)
    distances = np.empty(len(reference), dtype=np.float64)
    for start in range(0, len(reference), block_size):
        end = min(start + block_size, len(reference))
        matrix = cdist(reference[start:end], sample, "euclidean")
        index = np.argmin(matrix, axis=1)
        nearest[start:end] = index
        distances[start:end] = matrix[np.arange(end-start), index]
        log(f"[Coverage NN] dim={reference.shape[1]}, {end:,}/{len(reference):,}")
    return distances, nearest


def evaluate_coverage(reference_coordinates, sample_coordinates, radius,
                      block_size=DISTANCE_BLOCK):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("coverage radius must be finite and >= 0")
    distances, nearest = nearest_distances(reference_coordinates, sample_coordinates, block_size)
    return {"distances": distances, "nearest_indices": nearest,
            "covered": distances <= radius}


def distance_summary(distances):
    distances = np.asarray(distances)
    if not len(distances):
        return None
    return {"mean": float(distances.mean()), "median": float(np.median(distances)),
            "p95": float(np.percentile(distances, 95)),
            "p99": float(np.percentile(distances, 99)), "max": float(distances.max())}


def radius_cover_representatives(coordinates, global_rows, existing_distances, radius):
    """Deterministic farthest-first cover using real input rows, without an NxN matrix.

    Start at the point farthest from the existing sample. Then select the point
    farthest from the selected diagnostic set until every input is within radius.
    A global-row sort breaks ties independently of the input order. The number
    of representatives is unconstrained; this is not a minimum-cardinality claim.
    """
    from scipy.spatial.distance import cdist

    x = np.asarray(coordinates, dtype=np.float64)
    rows = integer_vector(global_rows, "gap global rows")
    existing = np.asarray(existing_distances, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1] < 1 or len(x) != len(rows)
            or existing.shape != rows.shape or not np.isfinite(x).all()
            or not np.isfinite(existing).all() or (existing < 0).any()
            or len(np.unique(rows)) != len(rows) or not np.isfinite(radius) or radius < 0):
        raise ValueError("invalid gap radius-cover inputs")
    if not len(x):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0)
    order = np.argsort(rows, kind="stable")
    x, existing = x[order], existing[order]
    selected, nearest = [], np.full(len(x), np.inf)
    membership = np.full(len(x), -1, dtype=np.int64)
    next_index = int(np.argmax(existing))
    started = time.perf_counter()
    while True:
        distances = cdist(x, x[next_index:next_index+1], "euclidean")[:, 0]
        closer = distances < nearest
        membership[closer] = len(selected)
        nearest[closer] = distances[closer]
        selected.append(next_index)
        remaining = int(np.count_nonzero(nearest > radius))
        if len(selected) == 1 or len(selected) % 100 == 0 or remaining == 0:
            log(f"[Gap groups] representatives={len(selected):,}, "
                f"within_radius={len(x)-remaining:,}/{len(x):,}, "
                f"elapsed={time.perf_counter()-started:.1f}s")
        if remaining == 0:
            break
        next_index = int(np.argmax(nearest))
    original_membership = np.empty_like(membership)
    original_distances = np.empty_like(nearest)
    original_membership[order] = membership
    original_distances[order] = nearest
    return order[np.asarray(selected)], original_membership, original_distances


def summarize_uncovered_reference(reference, sample, radius):
    """Select diagnostics from 100%-gap bins, preserving all baseline REF/B data."""
    if not np.isfinite(GAP_RADIUS_MULTIPLIER) or GAP_RADIUS_MULTIPLIER <= 0:
        raise ValueError("GAP_RADIUS_MULTIPLIER must be finite and > 0")
    coordinates = np.asarray(reference["distance_coordinates"], dtype=np.float64)
    covered = np.asarray(reference["covered_by_topology"])
    existing = np.asarray(reference["nearest_topology_distances"], dtype=np.float64)
    rows = integer_vector(reference["global_rows"], "REF global rows")
    if (coordinates.ndim != 2 or len(coordinates) != len(rows)
            or covered.dtype != np.dtype(bool) or covered.shape != rows.shape
            or existing.shape != rows.shape or not np.isfinite(existing).all()
            or not np.isfinite(radius) or radius < 0
            or not np.array_equal(covered, existing <= radius)):
        raise ValueError("inconsistent saved REF coverage for gap diagnostics")
    if str(reference["frame_id"].item()) != str(sample["frame_id"].item()):
        raise ValueError("gap diagnostics require REF and sample in the same frame")
    started = time.perf_counter()
    uncovered_count = int((~covered).sum())
    candidates = fully_uncovered_bin_candidates(
        reference["kpca_coordinates"], {"covered": covered, "distances": existing}, HEATMAP_BINS
    )
    gap_indices = np.flatnonzero(candidates["eligible_mask"])
    gap_indices = gap_indices[np.argsort(rows[gap_indices], kind="stable")]
    grouping_radius = radius * GAP_RADIUS_MULTIPLIER
    log(f"[Gap bins] uncovered={uncovered_count:,}, 100%-gap bins={len(candidates['bins']):,}, "
        f"eligible REF={len(gap_indices):,}, excluded mixed-bin gaps={uncovered_count-len(gap_indices):,}; "
        f"grid={HEATMAP_BINS}x{HEATMAP_BINS}, union across displayed pairs")
    log(f"[Gap groups] START eligible={len(gap_indices):,}, radius={grouping_radius:.8g}, "
        f"dimensions={coordinates.shape[1]}; diagnostic representatives only")
    selected, membership, member_distances = radius_cover_representatives(
        coordinates[gap_indices], rows[gap_indices], existing[gap_indices], grouping_radius
    )
    representative_indices = gap_indices[selected]
    counts = np.bincount(membership, minlength=len(selected))
    # Report populous missing regions first, then more distant groups, then row ID.
    ranking = sorted(range(len(selected)), key=lambda i: (
        -int(counts[i]), -float(existing[representative_indices[i]]),
        int(rows[representative_indices[i]])))
    old_to_new = np.empty(len(selected), dtype=np.int64)
    old_to_new[ranking] = np.arange(len(selected))
    membership = old_to_new[membership]
    representative_indices = representative_indices[ranking]
    sample_keys = {int(row): str(key) for row, key in zip(sample["global_rows"], sample["pattern_keys"])}
    bins_by_id = {entry["bin_id"]: entry for entry in candidates["bins"]}
    groups = []
    for label, ref_index in enumerate(representative_indices):
        mask = membership == label
        members = gap_indices[mask]
        nearest_row = int(reference["nearest_topology_global_rows"][ref_index])
        h0_labels, h0_counts = np.unique(reference["h0_labels"][members], return_counts=True)
        groups.append({
            "gap_group": f"G{label+1:04d}", "gap_ref_count": int(len(members)),
            "fraction_of_uncovered_ref": len(members)/uncovered_count,
            "fraction_of_eligible_ref": len(members)/len(gap_indices),
            "fraction_of_sampled_ref": len(members)/len(rows),
            "representative_global_row": int(rows[ref_index]),
            "representative_pattern_key": str(reference["pattern_keys"][ref_index]),
            "representative_h0_label": int(reference["h0_labels"][ref_index]),
            "representative_topology_cluster": int(reference["topology_labels"][ref_index]),
            "representative_100pct_bins": [bins_by_id[bin_id]
                for bin_id in candidates["bin_ids_by_ref"][ref_index] if bin_id],
            "nearest_existing_global_row": nearest_row,
            "nearest_existing_pattern_key": sample_keys[nearest_row],
            "representative_distance_to_existing": float(existing[ref_index]),
            "existing_distance_summary": distance_summary(existing[members]),
            "max_member_distance_to_gap_representative": float(member_distances[mask].max()),
            "h0_composition": {str(int(h0)): int(count) for h0, count in zip(h0_labels, h0_counts)},
        })
    summary = {
        "schema_version": 2, "frame_id": str(reference["frame_id"].item()),
        "method": "deterministic_farthest_first_radius_cover",
        "candidate_selection": "union_of_nonempty_100pct_uncovered_heatmap_bins",
        "heatmap_bins_per_axis": HEATMAP_BINS,
        "heatmap_pairs": candidates["pairs"],
        "fully_uncovered_bin_count": len(candidates["bins"]),
        "fully_uncovered_bins": candidates["bins"],
        "grouping_space": "all REF-standardized kPCA components",
        "coverage_radius": float(radius), "grouping_radius": float(grouping_radius),
        "grouping_radius_multiplier": GAP_RADIUS_MULTIPLIER,
        "ref_sample_count": len(rows), "uncovered_ref_count": uncovered_count,
        "eligible_ref_count": len(gap_indices),
        "excluded_mixed_bin_uncovered_ref_count": uncovered_count-len(gap_indices),
        "diagnostic_representative_count": len(groups),
        "assigned_uncovered_ref_count": len(membership),
        "max_member_distance_to_gap_representative": (
            float(member_distances.max()) if len(member_distances) else None),
        "input_fingerprint": arrays_fingerprint({
            "ref_rows": rows, "ref_coordinates": coordinates,
            "plot_coordinates": reference["kpca_coordinates"],
            "covered": covered, "existing_distances": existing,
            "sample_rows": sample["global_rows"], "sample_keys": sample["pattern_keys"],
        }),
        "scope": "Diagnostic exemplars of sampled REF in 100%-uncovered bins; mixed-bin gaps remain in baseline coverage and the full gap CSV.",
        "groups": groups,
    }
    log(f"[Gap groups] DONE representatives={len(groups):,}, elapsed={time.perf_counter()-started:.1f}s")
    return {"summary": summary, "gap_indices": gap_indices,
            "representative_indices": representative_indices,
            "membership": membership, "member_distances": member_distances,
            "bin_ids_by_ref": candidates["bin_ids_by_ref"]}


def write_gap_diagnostics(run_dir, reference, diagnostics):
    run_dir = Path(run_dir)
    summary, groups = diagnostics["summary"], diagnostics["summary"]["groups"]
    indices = diagnostics["representative_indices"]
    bundle = {"frame_id": reference["frame_id"], "feature_names": reference["feature_names"]}
    for name in ("global_rows", "pattern_keys", "h0_labels", "topology_labels", "raw_embeddings",
                 "normalized_embeddings", "kpca_coordinates", "distance_coordinates"):
        bundle[name] = reference[name][indices]
    bundle["gap_group_ids"] = np.asarray([g["gap_group"] for g in groups], dtype=str)
    bundle["gap_ref_counts"] = np.asarray([g["gap_ref_count"] for g in groups], dtype=np.int64)
    bundle["diagnostic_kind"] = np.asarray("uncovered_REF_in_100pct_gap_bins")
    bundle["diagnostic_schema_version"] = np.asarray(summary["schema_version"])
    bundle["fully_uncovered_bin_ids"] = diagnostics["bin_ids_by_ref"][indices]
    bundle["heatmap_pairs"] = np.asarray(summary["heatmap_pairs"], dtype=str)
    bundle["heatmap_bins_per_axis"] = np.asarray(summary["heatmap_bins_per_axis"])
    bundle["covered_by_topology"] = reference["covered_by_topology"][indices]
    bundle["nearest_topology_global_rows"] = reference["nearest_topology_global_rows"][indices]
    bundle["nearest_topology_distances"] = reference["nearest_topology_distances"][indices]
    bundle["nearest_topology_pattern_keys"] = np.asarray([g["nearest_existing_pattern_key"] for g in groups], dtype=str)
    bundle["coverage_radius"] = np.asarray(summary["coverage_radius"])
    bundle["grouping_radius"] = np.asarray(summary["grouping_radius"])
    save_npz(run_dir / "uncovered_gap_representatives.npz", bundle)
    extras = {
        "gap_group": bundle["gap_group_ids"], "gap_ref_count": bundle["gap_ref_counts"],
        "fraction_of_uncovered_ref": [g["fraction_of_uncovered_ref"] for g in groups],
        "fraction_of_eligible_ref": [g["fraction_of_eligible_ref"] for g in groups],
        "fraction_of_sampled_ref": [g["fraction_of_sampled_ref"] for g in groups],
        "fully_uncovered_bin_ids": [";".join(bin_id for bin_id in ids if bin_id)
                                    for ids in bundle["fully_uncovered_bin_ids"]],
        "nearest_existing_key": [g["nearest_existing_pattern_key"] for g in groups],
        "distance_to_existing": [g["representative_distance_to_existing"] for g in groups],
        "group_existing_distance_p95": [g["existing_distance_summary"]["p95"] for g in groups],
        "max_member_distance_to_gap_representative": [g["max_member_distance_to_gap_representative"] for g in groups],
    }
    write_scatter_csv(run_dir / "uncovered_gap_representatives.csv", bundle, extras)
    with (run_dir / "uncovered_gap_members.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["gap_group", "pattern_key", "global_row", "h0_label", "topology_cluster",
                         "gap_representative_key", "gap_representative_global_row",
                         "distance_to_gap_representative", "distance_to_existing",
                         "fully_uncovered_bin_ids"])
        for ref_index, label, distance in zip(diagnostics["gap_indices"], diagnostics["membership"],
                                               diagnostics["member_distances"]):
            group = groups[label]
            writer.writerow([group["gap_group"], reference["pattern_keys"][ref_index],
                             int(reference["global_rows"][ref_index]), int(reference["h0_labels"][ref_index]),
                             int(reference["topology_labels"][ref_index]), group["representative_pattern_key"],
                             group["representative_global_row"], float(distance),
                             float(reference["nearest_topology_distances"][ref_index]),
                             ";".join(bin_id for bin_id in diagnostics["bin_ids_by_ref"][ref_index] if bin_id)])
    with (run_dir / "uncovered_gap_bins.csv").open("w", newline="", encoding="utf-8") as fp:
        fields = ["bin_id", "pair", "x_bin", "y_bin", "x_min", "x_max", "y_min", "y_max",
                  "x_upper_inclusive", "y_upper_inclusive", "ref_count", "gap_count", "gap_fraction"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["fully_uncovered_bins"])
    write_json(run_dir / "uncovered_gap_summary.json", summary)
    log(f"[Gap groups] SAVED representatives={len(groups):,}, members={len(diagnostics['gap_indices']):,}")


def arrays_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        if name == "frame_id":
            continue
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError("object arrays are not allowed in persisted numeric bundles")
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def load_reference_frame(path):
    """Public Stage-8 API. Load a trusted numeric frame and verify its identity."""
    with np.load(path, allow_pickle=False) as data:
        frame = {name: data[name] for name in data.files}
    if int(frame["schema_version"]) != SCHEMA_VERSION:
        raise ValueError("unsupported kPCA frame schema version")
    if str(frame["frame_id"].item()) != arrays_fingerprint(frame):
        raise ValueError("kPCA frame fingerprint mismatch")
    return frame


def load_projection_bundle(path, frame):
    """REF/B snapshot for Stage 8; never join on rounded kPCA coordinates."""
    with np.load(path, allow_pickle=False) as data:
        bundle = {name: data[name] for name in data.files}
    if str(bundle["frame_id"].item()) != str(frame["frame_id"].item()):
        raise ValueError("projection bundle belongs to a different kPCA frame")
    return bundle


def save_npz(path, payload):
    with Path(path).open("wb") as fp:
        np.savez_compressed(fp, **payload)


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False, allow_nan=False)


def projection_bundle(inputs, rows, raw, projected, frame):
    positions = positions_for_rows(inputs["rows"], rows)
    return {
        "frame_id": frame["frame_id"], "global_rows": rows,
        "pattern_keys": np.asarray([str(inputs["keys"][row]) for row in rows]),
        "h0_labels": inputs["h0_labels"][positions],
        "topology_labels": inputs["topology_labels"][positions],
        "feature_names": frame["feature_names"], "raw_embeddings": raw,
        "normalized_embeddings": normalize_embeddings(raw, frame),
        "kpca_coordinates": projected,
        "distance_coordinates": projected / frame["kpc_scale"],
    }


def write_scatter_csv(path, bundle, extras=None, selection=None):
    extras = extras or {}
    n = len(bundle["global_rows"])
    indices = np.arange(n) if selection is None else np.flatnonzero(selection)
    kpc_columns = [f"KP{i+1}" for i in range(bundle["kpca_coordinates"].shape[1])]
    fields = ["pattern_key", "global_row", "h0_label", "topology_cluster"]
    fields += bundle["feature_names"].tolist() + kpc_columns + list(extras)
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        for i in indices:
            writer.writerow([
                bundle["pattern_keys"][i], int(bundle["global_rows"][i]),
                int(bundle["h0_labels"][i]), int(bundle["topology_labels"][i]),
                *bundle["raw_embeddings"][i].tolist(),
                *bundle["kpca_coordinates"][i].tolist(),
                *[values[i] for values in extras.values()],
            ])
    log(f"[Output] {Path(path).name}: {len(indices):,} rows")


def kpca_plot_pairs(dimensions):
    return [(0, 1), (0, 2), (1, 2)] if dimensions >= 3 else [(0, min(1, dimensions-1))]


def coverage_heatmap_grids(coordinates, coverage, bins):
    """Aggregate existing full-KPC decisions onto REF-only 2D grids.

    Empty cells are unknown (NaN), not zero-gap cells. Repeated REF rows count
    individually, just as in the original coverage calculation. Each pair has
    its own REF extent; no representative coordinates, KDE, or clipping enter.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    covered = np.asarray(coverage["covered"])
    distances = np.asarray(coverage["distances"], dtype=np.float64)
    if (coordinates.ndim != 2 or len(coordinates) == 0 or coordinates.shape[1] < 1
            or not np.isfinite(coordinates).all()
            or covered.shape != (len(coordinates),) or covered.dtype != np.dtype(bool)
            or distances.shape != covered.shape or not np.isfinite(distances).all()
            or (distances < 0).any() or not isinstance(bins, int) or bins < 2):
        raise ValueError("invalid heatmap coordinates/coverage/bins")
    grids = []
    for a, b in kpca_plot_pairs(coordinates.shape[1]):
        counts, x_edges, y_edges = np.histogram2d(
            coordinates[:, a], coordinates[:, b], bins=bins
        )
        gap_counts, _, _ = np.histogram2d(
            coordinates[~covered, a], coordinates[~covered, b], bins=(x_edges, y_edges)
        )
        distance_sums, _, _ = np.histogram2d(
            coordinates[:, a], coordinates[:, b], bins=(x_edges, y_edges), weights=distances
        )
        # Match histogram2d exactly: internal edges go right; the maximum edge
        # belongs to the final bin. Constant axes use histogram2d's expanded edges.
        ref_bin_indices = np.column_stack([
            np.minimum(np.searchsorted(edges, coordinates[:, axis], side="right")-1, bins-1)
            for axis, edges in ((a, x_edges), (b, y_edges))
        ])
        grids.append({
            "pair": (a, b), "x_edges": x_edges, "y_edges": y_edges,
            "ref_bin_indices": ref_bin_indices,
            "ref_counts": counts, "gap_counts": gap_counts,
            "gap_fraction": np.divide(gap_counts, counts,
                                      out=np.full_like(counts, np.nan), where=counts > 0),
            "mean_distance": np.divide(distance_sums, counts,
                                       out=np.full_like(counts, np.nan), where=counts > 0),
        })
    return grids


def fully_uncovered_bin_candidates(coordinates, coverage, bins):
    """Union of real REF rows in nonempty 100%-gap bins of any plotted pair.

    Compare integer counts, not rounded percentages. Keep singleton bins and
    duplicate-coordinate REF rows. This filters diagnostic candidates only.
    """
    grids = coverage_heatmap_grids(coordinates, coverage, bins)
    bin_ids_by_ref = np.full((len(coordinates), len(grids)), "", dtype="U80")
    records, pairs = [], []
    for column, grid in enumerate(grids):
        a, b = grid["pair"]
        pair = f"KP{a+1}/KP{b+1}"
        pairs.append(pair)
        eligible_bins = (grid["ref_counts"] > 0) & (grid["gap_counts"] == grid["ref_counts"])
        lookup = np.full((bins, bins), "", dtype="U80")
        for x, y in np.argwhere(eligible_bins):
            bin_id = f"KP{a+1}-KP{b+1}_X{x+1:03d}_Y{y+1:03d}"
            lookup[x, y] = bin_id
            records.append({
                "bin_id": bin_id, "pair": pair, "x_bin": int(x+1), "y_bin": int(y+1),
                "x_min": float(grid["x_edges"][x]), "x_max": float(grid["x_edges"][x+1]),
                "y_min": float(grid["y_edges"][y]), "y_max": float(grid["y_edges"][y+1]),
                "x_upper_inclusive": bool(x == bins-1), "y_upper_inclusive": bool(y == bins-1),
                "ref_count": int(grid["ref_counts"][x, y]),
                "gap_count": int(grid["gap_counts"][x, y]), "gap_fraction": 1.0,
            })
        x, y = grid["ref_bin_indices"].T
        bin_ids_by_ref[:, column] = lookup[x, y]
    return {"eligible_mask": np.any(bin_ids_by_ref != "", axis=1),
            "bin_ids_by_ref": bin_ids_by_ref, "pairs": pairs, "bins": records}


def plot_coverage_heatmaps(run_dir, reference, coverage, radius):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import PercentFormatter

    started = time.perf_counter()
    run_dir = Path(run_dir)
    coordinates = reference["kpca_coordinates"]
    grids = coverage_heatmap_grids(coordinates, coverage, HEATMAP_BINS)
    log(f"[Heatmap] START grid={HEATMAP_BINS}x{HEATMAP_BINS}, "
        f"REF={len(coordinates):,}; empty cells masked, original coverage labels retained")
    empty_color = "#e6e9ec"
    density_cmap = plt.get_cmap("Blues").copy()
    gap_cmap = plt.get_cmap("YlOrRd").copy()
    density_cmap.set_bad(empty_color)
    gap_cmap.set_bad(empty_color)
    count_norm = LogNorm(vmin=1, vmax=max(2, max(g["ref_counts"].max() for g in grids)))
    gap_norm = Normalize(vmin=0, vmax=1)
    note = (f"Gray: no sampled REF | {HEATMAP_BINS}x{HEATMAP_BINS} bins per panel | "
            f"distances use all {coordinates.shape[1]} scaled KPCs")
    fig, axes = plt.subplots(2, len(grids), figsize=(5*len(grids)+1, 8.4),
                             squeeze=False, constrained_layout=True)
    for column, grid in enumerate(grids):
        a, b = grid["pair"]
        for row in range(2):
            axes[row, column].set(xlabel=f"KP{a+1}", ylabel=f"KP{b+1}",
                                  xlim=grid["x_edges"][[0, -1]], ylim=grid["y_edges"][[0, -1]])
        count_image = axes[0, column].pcolormesh(
            grid["x_edges"], grid["y_edges"], np.ma.masked_equal(grid["ref_counts"].T, 0),
            cmap=density_cmap, norm=count_norm, shading="flat", rasterized=True,
        )
        gap_image = axes[1, column].pcolormesh(
            grid["x_edges"], grid["y_edges"], np.ma.masked_invalid(grid["gap_fraction"].T),
            cmap=gap_cmap, norm=gap_norm, shading="flat", rasterized=True,
        )
        axes[0, column].set_title(f"KP{a+1} / KP{b+1}: REF count")
        axes[1, column].set_title(f"KP{a+1} / KP{b+1}: uncovered fraction")
    fig.colorbar(count_image, ax=axes[0, :].tolist(), label="REF patterns per bin (log scale)", shrink=.9)
    fig.colorbar(gap_image, ax=axes[1, :].tolist(), label="Uncovered / REF in bin",
                 ticks=[0, .25, .5, .75, 1], format=PercentFormatter(xmax=1), shrink=.9)
    fig.suptitle(f"REF density and coverage gaps | Fixed R = {radius:.4g}\n{note}", fontsize=12)
    fig.savefig(run_dir / "kpca_coverage_2d_heatmap.png", dpi=160)
    plt.close(fig)

    # This map uses the saved nearest distances directly and is independent of R.
    max_distance = max(float(np.nanmax(g["mean_distance"])) for g in grids)
    distance_norm = Normalize(vmin=0, vmax=max_distance if max_distance > 0 else 1)
    fig, axes = plt.subplots(1, len(grids), figsize=(5*len(grids)+1, 4.5),
                             squeeze=False, constrained_layout=True)
    for ax, grid in zip(axes[0], grids):
        a, b = grid["pair"]
        distance_image = ax.pcolormesh(
            grid["x_edges"], grid["y_edges"], np.ma.masked_invalid(grid["mean_distance"].T),
            cmap=gap_cmap, norm=distance_norm, shading="flat", rasterized=True,
        )
        ax.set(xlabel=f"KP{a+1}", ylabel=f"KP{b+1}", title=f"KP{a+1} / KP{b+1}",
               xlim=grid["x_edges"][[0, -1]], ylim=grid["y_edges"][[0, -1]])
    fig.colorbar(distance_image, ax=axes[0].tolist(), label="Mean nearest B distance (scaled KPC)", shrink=.9)
    fig.suptitle(f"Nearest-representative distance by region (no coverage threshold)\n{note}", fontsize=12)
    fig.savefig(run_dir / "kpca_nearest_distance_2d_heatmap.png", dpi=160)
    plt.close(fig)
    log(f"[Heatmap] DONE density/gap and nearest-distance maps, elapsed={time.perf_counter()-started:.1f}s")


def plot_coverage(run_dir, reference, sample, coverage, radius):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yr, ys = reference["kpca_coordinates"], sample["kpca_coordinates"]
    covered = coverage["covered"]
    colors = {"covered": "#9caaaf", "gap": "#ce5f3d", "sample": "#156b91"}
    pairs = kpca_plot_pairs(yr.shape[1])
    fig, axes = plt.subplots(1, len(pairs), figsize=(5*len(pairs), 4.5), squeeze=False)
    for ax, (a, b) in zip(axes[0], pairs):
        ax.scatter(yr[covered, a], yr[covered, b], s=3, alpha=.3,
                   color=colors["covered"], label=f"REF covered ({covered.sum():,})")
        ax.scatter(yr[~covered, a], yr[~covered, b], s=5, alpha=.5,
                   color=colors["gap"], label=f"REF gap ({(~covered).sum():,})")
        ax.scatter(ys[:, a], ys[:, b], s=12, alpha=.8, marker="+",
                   color=colors["sample"], label=f"Topology B ({len(ys):,})")
        ax.set(xlabel=f"KP{a+1}", ylabel=f"KP{b+1}")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Sampled REF coverage: {covered.mean():.1%}; "
                 f"decision uses all {yr.shape[1]} scaled KPCs, R={radius:.4g}")
    fig.tight_layout()
    fig.savefig(run_dir / "kpca_coverage_2d.png", dpi=160)
    plt.close(fig)
    if yr.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        for mask, label, color in ((covered, "REF covered", colors["covered"]),
                                   (~covered, "REF gap", colors["gap"])):
            ax.scatter(*yr[mask, :3].T, s=4, alpha=.3, color=color, label=label)
        ax.scatter(*ys[:, :3].T, s=15, marker="+", color=colors["sample"], label="Topology B")
        ax.set(xlabel="KP1", ylabel="KP2", zlabel="KP3", title="REF / topology representatives")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "kpca_coverage_3d.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    distances = np.sort(coverage["distances"])
    ax.plot(distances, np.arange(1, len(distances)+1) / len(distances), color=colors["sample"])
    ax.axvline(radius, color=colors["gap"], linestyle="--", label=f"Fixed R = {radius:.4g}")
    ax.set(xlabel="Nearest topology representative distance (scaled KPC)",
           ylabel="Fraction of sampled REF", ylim=(0, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "kpca_coverage_distance_cdf.png", dpi=160)
    plt.close(fig)
    log("[Plot] DONE coverage projections and distance CDF")
    plot_coverage_heatmaps(run_dir, reference, coverage, radius)


def plot_gap_diagnostics(run_dir, reference, diagnostics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import PercentFormatter

    if not isinstance(GAP_PREVIEW_GROUPS, int) or GAP_PREVIEW_GROUPS < 1:
        raise ValueError("GAP_PREVIEW_GROUPS must be an integer >= 1")
    summary = diagnostics["summary"]
    shown = min(GAP_PREVIEW_GROUPS, summary["diagnostic_representative_count"])
    coordinates = reference["kpca_coordinates"]
    coverage = {"covered": reference["covered_by_topology"],
                "distances": reference["nearest_topology_distances"]}
    grids = coverage_heatmap_grids(coordinates, coverage, summary["heatmap_bins_per_axis"])
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e6e9ec")
    fig, axes = plt.subplots(1, len(grids), figsize=(5*len(grids)+1, 4.8),
                             squeeze=False, constrained_layout=True)
    for column, (ax, grid) in enumerate(zip(axes[0], grids)):
        a, b = grid["pair"]
        mesh = ax.pcolormesh(grid["x_edges"], grid["y_edges"],
                             np.ma.masked_invalid(grid["gap_fraction"].T),
                             cmap=cmap, norm=Normalize(0, 1), shading="flat", rasterized=True)
        preview_indices = diagnostics["representative_indices"][:shown]
        # A representative may qualify in KP1/KP3 but not KP1/KP2. Show it
        # only in the panels where its own bin contains no covered REF.
        labels = np.flatnonzero(diagnostics["bin_ids_by_ref"][preview_indices, column] != "")
        indices = preview_indices[labels]
        ax.scatter(coordinates[indices, a], coordinates[indices, b], marker="D", s=25,
                   color="#182e3a", edgecolors="white", linewidths=.6, zorder=3)
        # Distinct label slots prevent dense-region representatives obscuring IDs.
        ticks = np.linspace(.05, .95, max(17, math.ceil(math.sqrt(shown))+2))
        slots = np.stack(np.meshgrid(ticks, ticks), axis=-1).reshape(-1, 2)
        available = np.ones(len(slots), dtype=bool)
        for label, index in zip(labels, indices):
            number = int(label+1)
            position = np.array([
                (coordinates[index, a]-grid["x_edges"][0])/np.ptp(grid["x_edges"]),
                (coordinates[index, b]-grid["y_edges"][0])/np.ptp(grid["y_edges"]),
            ])
            costs = np.sum((slots-position)**2, axis=1)
            costs[~available] = np.inf
            slot = int(np.argmin(costs))
            available[slot] = False
            ax.annotate(str(number), (coordinates[index, a], coordinates[index, b]),
                        xytext=slots[slot], textcoords="axes fraction", fontsize=8,
                        ha="center", va="center", color="#182e3a",
                        arrowprops={"arrowstyle":"-", "color":"#526b79", "lw":.6},
                        bbox={"facecolor":"white", "edgecolor":"none", "alpha":.9, "pad":.7})
        if len(indices) == 0:
            note = ("No uncovered REF" if summary["uncovered_ref_count"] == 0 else
                    "No 100% uncovered bins" if summary["eligible_ref_count"] == 0 else
                    "No preview representatives in this pair")
            ax.text(.5, .5, note, transform=ax.transAxes,
                    ha="center", va="center", bbox={"facecolor":"white", "alpha":.9})
        ax.set(xlabel=f"KP{a+1}", ylabel=f"KP{b+1}", title=f"KP{a+1} / KP{b+1}",
               xlim=grid["x_edges"][[0, -1]], ylim=grid["y_edges"][[0, -1]])
    fig.colorbar(mesh, ax=axes[0].tolist(), label="Uncovered / REF in bin",
                 ticks=[0, .25, .5, .75, 1], format=PercentFormatter(xmax=1), shrink=.9)
    label_note = f"Labels 1-{shown}: G0001 onward, shown only in qualifying pairs" if shown else "No eligible diagnostic representatives"
    fig.suptitle(f"100% uncovered-bin exemplars | {summary['eligible_ref_count']:,} eligible / "
                 f"{summary['uncovered_ref_count']:,} uncovered REF\n{label_note}", fontsize=12)
    fig.savefig(Path(run_dir) / "kpca_uncovered_representatives_2d.png", dpi=160)
    plt.close(fig)
    log(f"[Gap plot] SAVED top {shown} of {summary['diagnostic_representative_count']:,} groups; qualifying pairs only")


def write_html_report(run_dir, summary, gap_summary=None):
    """Human-readable, offline report. All plot images are embedded in the HTML."""
    import base64
    from html import escape

    run_dir = Path(run_dir)

    def number(value, style="distance"):
        if value is None:
            return "—"
        if style == "count":
            return f"{int(value):,}"
        if style == "percent":
            return f"{float(value):.2%}"
        return f"{float(value):.6g}"

    def table(headers, rows):
        head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
        body = "".join("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row)
                       + "</tr>" for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    card_values = [
        ("REF 표본", number(summary.get("ref_sample_count"), "count")),
        ("대표 패턴 B", number(summary.get("topology_representative_count"), "count")),
        ("Coverage", number(summary.get("coverage_fraction"), "percent")),
        ("미커버 REF", number(summary.get("uncovered_ref_count"), "count")),
        ("고정 반경 R", number(summary.get("radius"))),
        ("학습 landmark", number(summary.get("landmark_count"), "count")),
    ]
    if gap_summary is not None:
        card_values[-1] = ("100% 미커버 bin 진단 대표", number(gap_summary["diagnostic_representative_count"], "count"))
    cards = "".join(f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
                    for label, value in card_values)
    distance_rows = []
    for label, key in (("표준화 kPCA", "nearest_distance_kpca"),
                       ("정규화 원본 embedding", "nearest_distance_40d")):
        values = summary.get(key) or {}
        distance_rows.append([label] + [number(values.get(stat)) for stat in ("mean", "median", "p95", "p99", "max")])
    distance_table = table(["거리 공간", "평균", "중앙값 (R50)", "P95 (R95)", "P99 (R99)", "최댓값"], distance_rows)
    group_rows = [["rare (-1)" if row["h0_label"] == -1 else f'H0_{row["h0_label"]}',
                   number(row["sampled_ref_count"], "count"),
                   number(row["coverage_fraction"], "percent"),
                   number((row.get("nearest_distance") or {}).get("p95"))]
                  for row in summary.get("h0_breakdown", [])]
    group_table = table(["H0 그룹", "REF 표본 수", "Coverage", "kPCA 거리 P95"], group_rows)
    setting_rows = [
        ("REF 모집단 수", number(summary.get("ref_population_count"), "count")),
        ("학습 landmark 수", number(summary.get("landmark_count"), "count")),
        ("원본 embedding 차원", number(summary.get("feature_dimension"), "count")),
        ("kPCA 차원", number(summary.get("kpc_components"), "count")),
        ("반경 산정용 REF 이웃 순위", number(summary.get("radius_k"), "count")),
        ("고정 비교 예산", number(summary.get("radius_budget"), "count")),
        ("REF 표본과 대표 목록의 중복 수", number(summary.get("ref_sample_representative_overlap"), "count")),
        ("대표 자체와의 중복을 제외한 보조 coverage", number(summary.get("coverage_fraction_excluding_representative_self_hits"), "percent")),
        ("RBF gamma", number(summary.get("gamma"))),
        ("유지한 landmark kernel variance mass", number(summary.get("retained_landmark_kernel_mass_fraction"), "percent")),
        ("Sampling seed", number(summary.get("random_seed"), "count")),
    ]
    if gap_summary is not None:
        setting_rows.append(("Heatmap / 진단 후보 격자 수 (축당)",
                             number(gap_summary["heatmap_bins_per_axis"], "count")))
    settings = table(["분석 조건", "값"], setting_rows)
    gap_section = ""
    if gap_summary is not None:
        rows = [[g["gap_group"], g["representative_pattern_key"], g["representative_global_row"],
                 "; ".join(f'{entry["pair"]} [{entry["x_bin"]}, {entry["y_bin"]}], REF {entry["ref_count"]:,}개'
                           for entry in g["representative_100pct_bins"]),
                 number(g["gap_ref_count"], "count"), number(g["fraction_of_uncovered_ref"], "percent"),
                 number(g["fraction_of_sampled_ref"], "percent"),
                 number(g["representative_distance_to_existing"])] for g in gap_summary["groups"]]
        headers = ["그룹", "실제 대표 패턴 ID", "원본 row", "대표가 속한 100% 미커버 bin / REF 수",
                   "그룹의 후보 REF 수", "전체 gap 중 비중",
                   "전체 REF 표본 중 비중", "대표의 기존군 최근접 거리"]
        shown = min(GAP_PREVIEW_GROUPS, len(rows))
        empty_note = ("현재 R에서 미커버 REF가 없습니다." if gap_summary["uncovered_ref_count"] == 0 else
                      "미커버 REF는 있지만 100% 미커버 bin이 없어 진단 대표를 추출하지 않았습니다.")
        preview = table(headers, rows[:shown]) if rows else f"<p>{empty_note}</p>"
        remainder = (f'<details><summary>나머지 {len(rows)-shown:,}개 그룹 보기</summary>'
                     f'{table(headers, rows[shown:])}</details>') if len(rows) > shown else ""
        gap_section = f'''<section><h2>어떤 실제 패턴을 놓쳤는가</h2>
<p><strong>칸 안의 REF가 모두 미커버인 bin에서만 실제 대표를 추립니다.</strong>
세 축 쌍 중 하나라도 조건을 만족하면 후보이며, 같은 REF는 한 번만 셉니다.
비어 있는 칸은 제외합니다. 후보를 모은 뒤 전체 표준화 kPCA 공간에서 반경
{number(gap_summary['grouping_radius'])}으로 묶어 진단 대표를 고릅니다.</p>
<p>전체 미커버 REF {number(gap_summary['uncovered_ref_count'], 'count')}개 중
100% 미커버 bin {number(gap_summary['fully_uncovered_bin_count'], 'count')}개에 속한
고유 후보는 {number(gap_summary['eligible_ref_count'], 'count')}개이며, 진단 대표는
{number(len(rows), 'count')}개입니다. 혼합 bin에만 있는 미커버 REF
{number(gap_summary['excluded_mixed_bin_uncovered_ref_count'], 'count')}개는 후보에서 제외되지만
기존 coverage 통계와 전체 미커버 목록에는 그대로 남습니다.</p>
<p>그룹의 후보 REF 수가 많은 순서입니다. 그림의 숫자 1은 G0001에 해당하며,
각 대표는 자기 bin이 100% 미커버인 축 쌍에만 표시합니다. 번호는 연결선 끝의 실제 위치를 가리킵니다.
bin의 100%는 이번 REF 표본 기준이며, REF가 1개인 칸도 포함하므로 표의 bin별 REF 수를 함께 확인하세요.
기존 대표군과 coverage 값은 유지합니다. 같은 그룹이라는 것만으로 실제 형상이 동일하다는 뜻은 아닙니다.</p>
{preview}{remainder}<p class="muted">전체 대표: uncovered_gap_representatives.csv ·
구성원 연결: uncovered_gap_members.csv · bin 경계/개수: uncovered_gap_bins.csv ·
원본 40D는 대표 CSV/NPZ에 함께 저장됩니다.</p></section>'''
    figures = []
    gap_section_inserted = False
    for filename, title, caption in (
        ("kpca_coverage_2d_heatmap.png", "어디에 REF가 있고, 어디를 놓쳤는가",
         "위: 칸별 REF 수(로그 색상). 아래: 그 칸 REF 중 미커버 비율. 빨갈수록 미커버 비율이 높습니다. "
         "회색은 REF 표본이 없는 칸입니다. REF 수가 적은 칸의 비율은 위쪽 밀도와 함께 확인하세요."),
        ("kpca_uncovered_representatives_2d.png", "미커버 영역과 진단 대표 패턴",
         "100% 미커버 bin에서 추린 상위 그룹의 실제 대표를 해당 bin이 100%인 축 쌍에만 마름모로 표시합니다. "
         "번호는 연결선 끝의 실제 위치를 가리킵니다. "
         "표시 개수는 그림 가독성을 위한 것이며, 나머지 그룹도 아래 표와 파일에 모두 보존합니다."),
        ("kpca_nearest_distance_2d_heatmap.png", "반경을 정하지 않고 보는 대표와의 거리",
         "칸별 REF의 최근접 대표 거리 평균입니다. 빨갈수록 대표에서 멀리 떨어진 영역입니다. "
         "거리 계산은 전체 kPCA 축을 사용하고, 표시만 두 축으로 모았습니다."),
        ("kpca_coverage_distance_cdf.png", "R에 따른 coverage 변화",
         "가로축 거리까지 허용할 때, 세로축만큼의 REF가 커버됩니다. 세로 점선은 이번 실행의 고정 R입니다."),
        ("kpca_coverage_2d.png", "개별 패턴의 위치", "기존 2차원 scatter입니다. Heatmap에서 발견한 영역의 개별 점 분포를 확인합니다."),
        ("kpca_coverage_3d.png", "3차원에서 본 분포", "첫 세 kPCA 축으로 REF와 대표 패턴의 위치를 표시합니다."),
    ):
        path = run_dir / filename
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            figures.append(f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p>'
                           f'<img alt="{escape(title)}" src="data:image/png;base64,{encoded}"></section>')
            if filename == "kpca_uncovered_representatives_2d.png" and gap_section:
                figures.append(gap_section)
                gap_section_inserted = True
    if gap_section and not gap_section_inserted:
        figures.insert(0, gap_section)
    if not figures:
        figures.append("<p>이번 실행에서는 그래프를 생성하지 않았습니다.</p>")
    css = """
    *{box-sizing:border-box} body{margin:0;background:#f2f5f7;color:#213744;font:16px/1.6 system-ui,sans-serif}
    main{max-width:1440px;margin:auto;padding:28px} h1{margin:0 0 8px;font-size:28px} h2{font-size:21px;margin:0 0 12px}
    p{margin:8px 0 16px} .muted{color:#526b79;font-size:14px} .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}
    .card,section{background:white;border:1px solid #d9e2e8;border-radius:12px;padding:22px}
    .card span{display:block;color:#526b79} .card strong{display:block;font-size:28px;color:#156b91}
    section{margin:20px 0} img{display:block;width:100%;height:auto} .table-wrap{overflow-x:auto}
    table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums} th,td{text-align:right;padding:10px 14px;border-bottom:1px solid #e6edf1;white-space:nowrap}
    th{background:#edf3f6} th:first-child,td:first-child{text-align:left} footer{color:#526b79;font-size:13px;overflow-wrap:anywhere}
    @media(max-width:700px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr);gap:10px}.card,section{padding:14px}h1{font-size:24px}.card strong{font-size:24px}}
    """
    html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topology coverage 분석</title><style>{css}</style></head><body><main>
<h1>Topology 대표 패턴 coverage 분석</h1><p class="muted">실행: {escape(run_dir.name)}</p>
<p>Coverage는 가장 가까운 대표 패턴까지의 거리가 R 이내인 REF 표본의 비율입니다.
거리 단위는 REF 축 표준편차로 조정한 kPCA 거리입니다. 전체 모집단의 전수 검사나 공정 길이 단위로 해석하지 않습니다.</p>
<div class="cards">{cards}</div>
{''.join(figures)}
<section><h2>최근접 거리 요약</h2><p>kPCA 행의 R50·R95·R99는 해당 비율의 REF를 커버하는 데 필요한 거리의 표본 분위수입니다.
원본 embedding 행은 별도의 거리 공간에서 계산한 보조 지표입니다.</p>{distance_table}</section>
<section><h2>H0 그룹별 coverage</h2>{group_table}</section>
<section><h2>분석 조건</h2>{settings}</section>
<footer>그림을 포함한 이 보고서는 인터넷 연결 없이 열 수 있습니다.<br>Frame ID: {escape(str(summary.get('frame_id', '—')))}</footer>
</main></body></html>'''
    (run_dir / "analysis_report.html").write_text(html, encoding="utf-8")
    log("[Report] SAVED analysis_report.html (offline, images embedded)")


def replot_saved_run(run):
    """Use saved coverage, add gap diagnostics, and render; do not refit/resample."""
    expected_frame_id = None
    if str(run) == "latest":
        with (Path(OUT_DIR) / "latest_run.json").open(encoding="utf-8") as fp:
            pointer = json.load(fp)
        run_dir = Path(OUT_DIR) / pointer["run_directory"]
        expected_frame_id = pointer["frame_id"]
    else:
        run_dir = Path(run).expanduser().resolve()
    with (run_dir / "replot.log").open("a", encoding="utf-8", buffering=1) as fp:
        with contextlib.redirect_stdout(_Tee(sys.stdout, fp)), contextlib.redirect_stderr(_Tee(sys.stderr, fp)):
            log(f"[Replot] START run={run_dir}")
            try:
                frame = load_reference_frame(run_dir / "reference_frame.npz")
                if expected_frame_id is not None and str(frame["frame_id"].item()) != expected_frame_id:
                    raise ValueError("latest run pointer has a different frame ID")
                reference = load_projection_bundle(run_dir / "reference_sample.npz", frame)
                sample = load_projection_bundle(run_dir / "topology_sample.npz", frame)
                coverage = {"covered": reference["covered_by_topology"],
                            "distances": reference["nearest_topology_distances"]}
                radius = float(frame["radius"])
                if not np.array_equal(coverage["covered"], coverage["distances"] <= radius):
                    raise ValueError("saved coverage labels differ from saved distances/radius")
                with (run_dir / "coverage_summary.json").open(encoding="utf-8") as summary_file:
                    summary = json.load(summary_file)
                if summary["frame_id"] != str(frame["frame_id"].item()):
                    raise ValueError("saved summary belongs to a different kPCA frame")
                diagnostics = summarize_uncovered_reference(reference, sample, radius)
                write_gap_diagnostics(run_dir, reference, diagnostics)
                plot_coverage(run_dir, reference, sample, coverage, radius)
                plot_gap_diagnostics(run_dir, reference, diagnostics)
                write_html_report(run_dir, summary, diagnostics["summary"])
            except Exception:
                traceback.print_exc()
                return 1
            log(f"[Replot] DONE frame_id={str(frame['frame_id'].item())}, R={radius:.8g}")
    return 0


def main(run_dir):
    from threadpoolctl import threadpool_limits

    validate_config()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=CPU_THREADS):
        return analyze(run_dir)


def analyze(run_dir):
    started = time.perf_counter()
    inputs = load_stage6_inputs()
    ref_population = reference_population_rows(inputs["rows"], inputs["keys"], REF_PATTERN_KEYS_CSV)
    ref_rows = sample_reference_rows(ref_population, N_REF, RANDOM_SEED)
    log(f"[REF] population={len(ref_population):,}, sample={len(ref_rows):,}, "
        f"seed={RANDOM_SEED}; uniform sampling, no HDR/outlier/rare exclusion")
    # B membership never changes the REF sample, scaler, landmarks or radius.
    raw_ref = gather_raw_embeddings(inputs["features"], ref_rows)
    raw_sample = gather_raw_embeddings(inputs["features"], inputs["rep_rows"])
    del inputs["features"]
    backend = choose_kernel_backend(KERNEL_BACKEND, GPU_DEVICE)
    frame = fit_reference_frame(raw_ref, inputs["block_multiplier"], N_LANDMARKS,
                                N_COMPONENTS, RANDOM_SEED, GAMMA, backend, GPU_DEVICE)
    projected_ref = transform_raw_embeddings(raw_ref, frame, TRANSFORM_BLOCK, backend, GPU_DEVICE)
    kpc_scale = projected_ref.std(0)
    if (kpc_scale <= np.finfo(np.float64).eps).any():
        raise ValueError("degenerate REF kPCA axis; cannot standardize distance coordinates")
    frame["kpc_scale"] = kpc_scale
    radius, radius_k = representation_radius(projected_ref / kpc_scale, RADIUS_BUDGET, R_MULT)
    frame.update({
        "radius": np.asarray(radius), "radius_k": np.asarray(radius_k),
        "radius_budget": np.asarray(RADIUS_BUDGET), "radius_multiplier": np.asarray(R_MULT),
        "feature_names": np.asarray(inputs["feature_names"]),
        "feature_blocks": np.asarray(FEATURE_BLOCKS),
        "ref_global_rows": ref_rows,
        "landmark_global_rows": ref_rows[frame["landmark_sample_indices"]],
        "population_fingerprint": np.asarray(inputs["population_fingerprint"]),
        "ref_population_fingerprint": np.asarray(population_fingerprint(inputs["keys"], ref_population)),
        "ref_sample_fingerprint": np.asarray(population_fingerprint(inputs["keys"], ref_rows)),
        "ref_embedding_fingerprint": np.asarray(arrays_fingerprint({"raw_ref": raw_ref})),
        "random_seed": np.asarray(RANDOM_SEED),
    })
    frame["frame_id"] = np.asarray(arrays_fingerprint(frame))
    save_npz(run_dir / "reference_frame.npz", frame)
    log(f"[Frame] SAVED frame_id={str(frame['frame_id'])}, R={radius:.8g}, k={radius_k}")

    projected_sample = transform_raw_embeddings(raw_sample, frame, TRANSFORM_BLOCK, backend, GPU_DEVICE)
    reference = projection_bundle(inputs, ref_rows, raw_ref, projected_ref, frame)
    sample = projection_bundle(inputs, inputs["rep_rows"], raw_sample, projected_sample, frame)
    coverage = evaluate_coverage(reference["distance_coordinates"], sample["distance_coordinates"],
                                 radius, DISTANCE_BLOCK)
    # Uncompressed 40D distances are diagnostic, not another threshold/selection rule.
    d40, nn40 = nearest_distances(reference["normalized_embeddings"],
                                 sample["normalized_embeddings"], DISTANCE_BLOCK)
    overlap = np.isin(ref_rows, inputs["rep_rows"])
    reference.update({
        "covered_by_topology": coverage["covered"],
        "nearest_topology_distances": coverage["distances"],
        "nearest_topology_global_rows": sample["global_rows"][coverage["nearest_indices"]],
        "nearest_topology_distances_40d": d40,
        "nearest_topology_global_rows_40d": sample["global_rows"][nn40],
        "is_topology_representative": overlap,
    })
    save_npz(run_dir / "reference_sample.npz", reference)
    save_npz(run_dir / "topology_sample.npz", sample)
    extras = {
        "cover_status": np.where(coverage["covered"], "covered", "gap"),
        "nearest_topology_key": sample["pattern_keys"][coverage["nearest_indices"]],
        "nearest_topology_global_row": reference["nearest_topology_global_rows"],
        "nearest_topology_distance": coverage["distances"],
        "nearest_topology_key_40d": sample["pattern_keys"][nn40],
        "nearest_topology_distance_40d": d40,
        "is_topology_representative": overlap,
    }
    write_scatter_csv(run_dir / "kpca_reference_scatter.csv", reference, extras)
    write_scatter_csv(run_dir / "uncovered_kpca_gap_patterns.csv", reference, extras, ~coverage["covered"])
    write_scatter_csv(run_dir / "kpca_topology_scatter.csv", sample)
    if MAKE_PLOTS:
        plot_coverage(run_dir, reference, sample, coverage, radius)
    diagnostics = summarize_uncovered_reference(reference, sample, radius)
    write_gap_diagnostics(run_dir, reference, diagnostics)
    if MAKE_PLOTS:
        plot_gap_diagnostics(run_dir, reference, diagnostics)

    covered = coverage["covered"]
    h0_summary = []
    for label in np.unique(reference["h0_labels"]):
        mask = reference["h0_labels"] == label
        h0_summary.append({"h0_label": int(label), "sampled_ref_count": int(mask.sum()),
                           "coverage_fraction": float(covered[mask].mean()),
                           "nearest_distance": distance_summary(coverage["distances"][mask])})
    summary = {
        "schema_version": SCHEMA_VERSION, "analysis": "topology_B_vs_sampled_REF",
        "frame_id": str(frame["frame_id"].item()), "reference_source": (
            "Stage-6 valid population" if REF_PATTERN_KEYS_CSV is None else str(REF_PATTERN_KEYS_CSV)),
        "ref_population_count": len(ref_population), "requested_ref_sample_count": N_REF,
        "ref_sample_count": len(ref_rows), "topology_representative_count": len(raw_sample),
        "feature_dimension": raw_ref.shape[1], "feature_blocks": list(FEATURE_BLOCKS),
        "feature_names": inputs["feature_names"], "input_cache": str(CACHE_PATH),
        "population_fingerprint": inputs["population_fingerprint"],
        "stage6_selected_resolution": inputs["metadata"].get("selected_resolution"),
        "stage6_representative_fingerprint": population_fingerprint(inputs["keys"], inputs["rep_rows"]),
        "random_seed": RANDOM_SEED, "landmark_count": len(frame["landmarks"]),
        "kpc_components": N_COMPONENTS, "gamma": float(frame["gamma"]),
        "retained_landmark_kernel_mass_fraction": float(frame["eigenvalues"].sum()/frame["kernel_total_mass"]),
        "kernel_backend": backend, "eigensolver": "scipy_arpack_cpu",
        "radius_budget": RADIUS_BUDGET, "radius": radius, "radius_k": radius_k,
        "radius_multiplier": R_MULT, "hdr_filter_applied": False,
        "covered_ref_count": int(covered.sum()), "uncovered_ref_count": int((~covered).sum()),
        "coverage_fraction": float(covered.mean()), "uncoverage_fraction": float((~covered).mean()),
        "nearest_distance_kpca": distance_summary(coverage["distances"]),
        "nearest_distance_40d": distance_summary(d40),
        "ref_sample_representative_overlap": int(overlap.sum()),
        "coverage_fraction_excluding_representative_self_hits": (
            float(covered[~overlap].mean()) if (~overlap).any() else None),
        "h0_breakdown": h0_summary,
        "scope": "Uniform REF sample estimate in REF-standardized kPCA; not full-population/physical coverage.",
        "stage8_contract": "Reuse reference_frame.npz, reference_sample.npz and topology_sample.npz; transform A only.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(run_dir / "coverage_summary.json", summary)
    write_html_report(run_dir, summary, diagnostics["summary"])
    log(f"[Result] sampled REF covered={covered.sum():,}/{len(covered):,} ({covered.mean():.2%}), "
        f"gaps={(~covered).sum():,}, self_hits={overlap.sum():,}")
    log(f"[Result] DONE elapsed={time.perf_counter() - started:.1f}s; {run_dir}")
    return summary


class _Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.terminal.write(text)
        return len(text)

    def flush(self):
        self.file.flush()
        self.terminal.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


def run_with_log():
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(OUT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "analysis.log").open("x", encoding="utf-8", buffering=1) as fp:
        with contextlib.redirect_stdout(_Tee(sys.stdout, fp)), contextlib.redirect_stderr(_Tee(sys.stderr, fp)):
            log(f"[Run] START log={run_dir / 'analysis.log'}")
            try:
                summary = main(run_dir)
                pointer = Path(OUT_DIR) / f"latest_run_{run_id}.tmp"
                write_json(pointer, {"run_directory": run_id, "frame_id": summary["frame_id"]})
                pointer.replace(Path(OUT_DIR) / "latest_run.json")
            except KeyboardInterrupt:
                log("[Run] INTERRUPTED; partial files/log retained, latest successful run unchanged")
                return 130
            except Exception:
                traceback.print_exc()
                log("[Run] FAILED; partial files/log retained, latest successful run unchanged")
                return 1
            log("[Run] SUCCESS; fixed-frame artifacts ready for Stage 8")
    return 0


def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", metavar="RUN_DIRECTORY_OR_latest",
                        help="redraw plots from saved Stage-7 snapshots without rerunning analysis")
    args = parser.parse_args()
    if args.replot is not None:
        return replot_saved_run(args.replot)
    return run_with_log()


if __name__ == "__main__":
    raise SystemExit(cli())
