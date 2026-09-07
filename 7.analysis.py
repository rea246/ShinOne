"""REF-fixed landmark kPCA coverage of the real representatives from Stage 6.

Stage 7 compares REF with the topology representative set (B), not yet with an
external method (A). Stage 8 must reuse the saved frame, REF rows and radius.
The 40 inputs are the cached learned h0/h1/h2/h3/edge embeddings, not 21 raw
handcrafted features. No clustering or representative selection is rerun.

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
    ):
        if not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if N_COMPONENTS >= min(N_REF, N_LANDMARKS):
        raise ValueError("N_COMPONENTS must be smaller than REF/landmark counts")
    if not np.isfinite(R_MULT) or R_MULT <= 0:
        raise ValueError("R_MULT must be finite and > 0")
    if GAMMA is not None and (not np.isfinite(GAMMA) or GAMMA <= 0):
        raise ValueError("GAMMA must be None or finite and > 0")
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


def plot_coverage(run_dir, reference, sample, coverage, radius):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yr, ys = reference["kpca_coordinates"], sample["kpca_coordinates"]
    covered = coverage["covered"]
    colors = {"covered": "#9caaaf", "gap": "#ce5f3d", "sample": "#156b91"}
    pairs = [(0, 1), (0, 2), (1, 2)] if yr.shape[1] >= 3 else [(0, min(1, yr.shape[1]-1))]
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


if __name__ == "__main__":
    raise SystemExit(run_with_log())
