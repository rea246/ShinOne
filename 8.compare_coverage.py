"""Compare Stage-6 representatives (A) and B.txt pattern keys against saved REF.

Reuse Stage 7's REF, learned feature definition, kPCA frame, axis scales and R.
In Stage 7 the topology set was historically called B; here it is explicitly A.
B.txt supplies the external set. No resampling, fitting or fixed-size truncation.
"""

import contextlib
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np


_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("stage7_coverage", _here / "7.analysis.py")
stage7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage7)
log = stage7.log

B_KEYS_PATH = _here / "B.txt"           # One exact pattern_key per line; optional header.
CACHE_PATH = _here / "hkeys_features.pt"
STAGE7_OUT_DIR = _here / "topology_analysis_out"
STAGE7_RUN = "latest"
OUT_DIR = _here / "coverage_comparison_out"
HEATMAP_BINS = None                    # Inherit saved Stage-7 diagnostic grid; fallback 50.
GAP_RADIUS_MULTIPLIER = 1.0
GAP_PREVIEW_GROUPS = 20                # Display only; every selected representative is saved.
CPU_THREADS = 8
TRANSFORM_BLOCK = 2048
DISTANCE_BLOCK = 2048
KERNEL_BACKEND = "auto"
GPU_DEVICE = 0
MAKE_PLOTS = True
STATES = ("both_covered", "A_only", "B_only", "both_uncovered")
PROJECTION_FIELDS = ("global_rows", "pattern_keys", "raw_embeddings", "normalized_embeddings",
                     "kpca_coordinates", "distance_coordinates")


def resolve_run(run, output_root):
    if str(run) != "latest":
        return Path(run).expanduser().resolve(), None
    with (Path(output_root) / "latest_run.json").open(encoding="utf-8") as fp:
        pointer = json.load(fp)
    return Path(output_root) / pointer["run_directory"], pointer["frame_id"]


def read_b_keys(path):
    content = Path(path).read_bytes()
    lines = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if lines and lines[0] == "pattern_key":
        lines.pop(0)
    if not lines:
        raise ValueError("B.txt contains no pattern keys")
    counts = {}
    for key in lines:
        counts[key] = counts.get(key, 0) + 1
    return list(counts), counts, content


def lookup_b_rows(cache_keys, requested_keys, occurrences, report_path):
    """Exact, case-sensitive lookup; never silently substitute or drop a key."""
    wanted = set(requested_keys)
    matches = {key: [] for key in requested_keys}
    for row, value in enumerate(cache_keys):
        key = str(value)
        if key in wanted:
            matches[key].append(row)
    missing = [key for key, rows in matches.items() if not rows]
    ambiguous = [key for key, rows in matches.items() if len(rows) > 1]
    with Path(report_path).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pattern_key", "input_occurrences", "status", "cache_match_count", "global_rows"])
        for key, rows in matches.items():
            status = "missing" if not rows else "ambiguous" if len(rows) > 1 else "matched"
            writer.writerow([key, occurrences[key], status, len(rows), ";".join(map(str, rows))])
    log(f"[B keys] input={sum(occurrences.values()):,}, unique={len(requested_keys):,}, "
        f"duplicate lines={sum(occurrences.values())-len(requested_keys):,}, "
        f"missing={len(missing):,}, ambiguous={len(ambiguous):,}")
    if missing or ambiguous:
        raise ValueError(f"B key lookup failed: missing={missing[:5]}, ambiguous={ambiguous[:5]}; "
                         f"see {report_path}")
    return np.asarray([matches[key][0] for key in requested_keys], dtype=np.int64)


def validate_projection(bundle, frame, label):
    rows = stage7.integer_vector(bundle["global_rows"], f"{label} rows")
    if not len(rows) or len(np.unique(rows)) != len(rows) or (rows < 0).any():
        raise ValueError(f"{label} has empty/duplicate/invalid rows")
    if bundle["pattern_keys"].shape != rows.shape:
        raise ValueError(f"{label} key/row alignment differs")
    if not np.array_equal(bundle["feature_names"], frame["feature_names"]):
        raise ValueError(f"{label} feature definition differs from saved frame")
    dimension, components = len(frame["center"]), len(frame["kpc_scale"])
    for name in ("raw_embeddings", "normalized_embeddings", "kpca_coordinates", "distance_coordinates"):
        value = np.asarray(bundle[name])
        width = dimension if name.endswith("embeddings") else components
        if value.shape != (len(rows), width) or not np.isfinite(value).all():
            raise ValueError(f"invalid {label} {name}")
    if not np.allclose(bundle["normalized_embeddings"],
                       stage7.normalize_embeddings(bundle["raw_embeddings"], frame), rtol=1e-12, atol=1e-12):
        raise ValueError(f"{label} normalized embeddings differ from saved frame")
    if not np.array_equal(bundle["distance_coordinates"], bundle["kpca_coordinates"] / frame["kpc_scale"]):
        raise ValueError(f"{label} distance axis scaling differs from saved frame")


def load_baseline(run):
    directory, expected_id = resolve_run(run, STAGE7_OUT_DIR)
    frame = stage7.load_reference_frame(directory / "reference_frame.npz")
    frame_id = str(frame["frame_id"].item())
    if expected_id is not None and frame_id != expected_id:
        raise ValueError("Stage-7 latest pointer has a different frame ID")
    radius = float(frame["radius"])
    if not np.isfinite(radius) or radius < 0 or (frame["kpc_scale"] <= 0).any():
        raise ValueError("invalid saved radius or kPCA axis scale")
    reference = stage7.load_projection_bundle(directory / "reference_sample.npz", frame)
    sample_a = stage7.load_projection_bundle(directory / "topology_sample.npz", frame)
    validate_projection(reference, frame, "REF")
    validate_projection(sample_a, frame, "A")
    if not np.array_equal(reference["global_rows"], frame["ref_global_rows"]):
        raise ValueError("REF rows differ from saved frame")
    if stage7.arrays_fingerprint({"raw_ref": reference["raw_embeddings"]}) != str(frame["ref_embedding_fingerprint"].item()):
        raise ValueError("saved REF embedding fingerprint differs")
    with (directory / "coverage_summary.json").open(encoding="utf-8") as fp:
        old_summary = json.load(fp)
    if (old_summary["frame_id"] != frame_id
            or old_summary["ref_sample_count"] != len(reference["global_rows"])
            or old_summary["topology_representative_count"] != len(sample_a["global_rows"])
            or old_summary["radius"] != radius):
        raise ValueError("Stage-7 summary differs from saved snapshots")
    d_a = np.asarray(reference["nearest_topology_distances"])
    covered_a = np.asarray(reference["covered_by_topology"])
    if (d_a.shape != reference["global_rows"].shape or not np.isfinite(d_a).all() or (d_a < 0).any()
            or covered_a.dtype != np.dtype(bool) or not np.array_equal(covered_a, d_a <= radius)
            or int(covered_a.sum()) != old_summary["covered_ref_count"]
            or not np.isin(reference["nearest_topology_global_rows"], sample_a["global_rows"]).all()):
        raise ValueError("saved A coverage/distances/nearest rows are inconsistent")
    bins = HEATMAP_BINS
    diagnostic_path = directory / "uncovered_gap_summary.json"
    if bins is None and diagnostic_path.is_file():
        with diagnostic_path.open(encoding="utf-8") as fp:
            diagnostic = json.load(fp)
        if diagnostic["frame_id"] != frame_id:
            raise ValueError("Stage-7 diagnostic grid belongs to a different frame")
        bins = diagnostic.get("heatmap_bins_per_axis", 50)
    bins = 50 if bins is None else bins
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("HEATMAP_BINS must be None or an integer >= 2")
    log(f"[Baseline] A=Stage-6 topology set ({len(sample_a['global_rows']):,}), "
        f"REF={len(reference['global_rows']):,}, R={radius:.8g}, grid={bins}x{bins}; frame={frame_id}")
    return directory, frame, reference, sample_a, bins


def extract_b_embeddings(cache_path, requested_keys, occurrences, reference, sample_a, frame, report_path):
    started = time.perf_counter()
    cache = stage7.read_feature_cache(cache_path)
    features, keys = cache["features"], cache["keys"]
    if tuple(frame["feature_blocks"]) != stage7.FEATURE_BLOCKS:
        raise ValueError("saved feature blocks differ from h0/h1/h2/h3/edge")
    feature_names = []
    for name in stage7.FEATURE_BLOCKS:
        shape = tuple(features[name].shape)
        if len(shape) != 2 or shape[0] != len(keys) or shape[1] < 1:
            raise ValueError(f"invalid cached {name} shape")
        feature_names.extend(f"{name}_{i}" for i in range(shape[1]))
    if not np.array_equal(np.asarray(feature_names), frame["feature_names"]):
        raise ValueError("cache feature dimensions/order differ from saved Stage-7 frame")
    rows = lookup_b_rows(keys, requested_keys, occurrences, report_path)
    # Verify the actual saved sample values as well as identity: re-encoding the
    # cache under another checkpoint must not silently change the comparison.
    for label, snapshot in (("REF", reference), ("A", sample_a)):
        saved_rows = snapshot["global_rows"]
        if saved_rows.max() >= len(keys) or not np.array_equal(
                np.asarray([str(keys[int(row)]) for row in saved_rows]), snapshot["pattern_keys"]):
            raise ValueError(f"current cache row/key identity differs from saved {label}")
        raw = stage7.gather_raw_embeddings(features, saved_rows)
        if not np.array_equal(raw, snapshot["raw_embeddings"]):
            raise ValueError(f"current cache embeddings differ from saved {label}; use the original cache")
    raw_b = stage7.gather_raw_embeddings(features, rows)
    del cache, features, keys, raw
    gc.collect()
    log(f"[B embeddings] {len(rows):,} x {raw_b.shape[1]} from cache; "
        f"REF/A keys and values verified, elapsed={time.perf_counter()-started:.1f}s")
    return rows, raw_b


def project_b(raw, rows, keys, reference, sample_a, frame, backend):
    """Reuse saved positions for identical rows; transform only unseen B rows."""
    projected = np.empty((len(rows), len(frame["kpc_scale"])), dtype=np.float64)
    saved = {int(row): (reference["kpca_coordinates"][i], "REF")
             for i, row in enumerate(reference["global_rows"])}
    saved.update({int(row): (sample_a["kpca_coordinates"][i], "A")
                  for i, row in enumerate(sample_a["global_rows"])})
    new_indices, reused = [], {"A": 0, "REF": 0}
    for i, row in enumerate(rows):
        if int(row) in saved:
            position, source = saved[int(row)]
            projected[i] = position
            reused[source] += 1
        else:
            new_indices.append(i)
    if new_indices:
        projected[new_indices] = stage7.transform_raw_embeddings(
            raw[new_indices], frame, TRANSFORM_BLOCK, backend, GPU_DEVICE)
    log(f"[B projection] saved A={reused['A']:,}, saved REF={reused['REF']:,}, "
        f"new transforms={len(new_indices):,}; no fit")
    return {"frame_id": frame["frame_id"], "feature_names": frame["feature_names"],
            "global_rows": rows, "pattern_keys": np.asarray(keys), "raw_embeddings": raw,
            "normalized_embeddings": stage7.normalize_embeddings(raw, frame),
            "kpca_coordinates": projected, "distance_coordinates": projected / frame["kpc_scale"]}, {
                "reused_A_rows": reused["A"], "reused_REF_rows": reused["REF"], "transformed_rows": len(new_indices)}


def classify_coverage(distance_a, distance_b, radius):
    d_a, d_b = np.asarray(distance_a), np.asarray(distance_b)
    if (d_a.ndim != 1 or d_a.shape != d_b.shape or not np.isfinite(d_a).all()
            or not np.isfinite(d_b).all() or (d_a < 0).any() or (d_b < 0).any()
            or not np.isfinite(radius) or radius < 0):
        raise ValueError("invalid A/B coverage distances or radius")
    a, b = d_a <= radius, d_b <= radius
    return {"covered_by_A": a, "covered_by_B": b, "both_covered": a & b,
            "A_only": a & ~b, "B_only": ~a & b, "both_uncovered": ~a & ~b}


def evaluate_b(reference, sample_a, sample_b, radius):
    started = time.perf_counter()
    d_b, nn_b = stage7.nearest_distances(reference["distance_coordinates"], sample_b["distance_coordinates"], DISTANCE_BLOCK)
    d40_b, nn40_b = stage7.nearest_distances(reference["normalized_embeddings"], sample_b["normalized_embeddings"], DISTANCE_BLOCK)
    d_a = reference["nearest_topology_distances"]
    result = classify_coverage(d_a, d_b, radius)
    a_keys = {int(row): str(key) for row, key in zip(sample_a["global_rows"], sample_a["pattern_keys"])}
    result.update({
        "distance_A": d_a.copy(), "distance_B": d_b,
        "distance_A_40d": reference["nearest_topology_distances_40d"].copy(), "distance_B_40d": d40_b,
        "nearest_A_global_rows": reference["nearest_topology_global_rows"].copy(),
        "nearest_A_pattern_keys": np.asarray([a_keys[int(row)] for row in reference["nearest_topology_global_rows"]]),
        "nearest_B_global_rows": sample_b["global_rows"][nn_b],
        "nearest_B_pattern_keys": sample_b["pattern_keys"][nn_b],
        "nearest_A_global_rows_40d": reference["nearest_topology_global_rows_40d"].copy(),
        "nearest_B_global_rows_40d": sample_b["global_rows"][nn40_b],
        "is_A_member": np.isin(reference["global_rows"], sample_a["global_rows"]),
        "is_B_member": np.isin(reference["global_rows"], sample_b["global_rows"]),
    })
    result["excluded_from_nonmember_comparison"] = result["is_A_member"] | result["is_B_member"]
    log(f"[Comparison] DONE A={result['covered_by_A'].mean():.2%}, B={result['covered_by_B'].mean():.2%}, "
        f"A-only={result['A_only'].sum():,}, elapsed={time.perf_counter()-started:.1f}s")
    return result


def summarize_outcomes(comparison, mask=None):
    if mask is None:
        mask = np.ones(len(comparison["distance_A"]), dtype=bool)
    n = int(mask.sum())
    counts = {name: int(comparison[name][mask].sum()) for name in STATES}
    return {"ref_count": n, "counts": counts,
            "fractions": {name: value/n if n else None for name, value in counts.items()},
            "coverage_A": float(comparison["covered_by_A"][mask].mean()) if n else None,
            "coverage_B": float(comparison["covered_by_B"][mask].mean()) if n else None}


def comparison_heatmap_grids(reference, comparison, bins):
    coordinates = reference["kpca_coordinates"]
    grids = stage7.coverage_heatmap_grids(coordinates, {
        "covered": comparison["covered_by_B"], "distances": comparison["distance_B"]}, bins)
    for grid in grids:
        for name in STATES:
            counts = np.zeros_like(grid["ref_counts"])
            indices = grid["ref_bin_indices"][comparison[name]]
            np.add.at(counts, tuple(indices.T), 1)
            grid[name+"_count"] = counts
            grid[name+"_fraction"] = np.divide(counts, grid["ref_counts"],
                out=np.full_like(counts, np.nan), where=grid["ref_counts"] > 0)
        grid["A_uncovered_fraction"] = grid["B_only_fraction"] + grid["both_uncovered_fraction"]
        grid["B_uncovered_fraction"] = grid["A_only_fraction"] + grid["both_uncovered_fraction"]
    return grids


def select_a_only_representatives(reference, comparison, radius, bins, multiplier):
    started = time.perf_counter()
    # The generic bin counter aggregates False labels. Here False means the
    # A-covered/B-uncovered event, not a replacement for either coverage label.
    candidates = stage7.fully_uncovered_bin_candidates(reference["kpca_coordinates"],
        {"covered": ~comparison["A_only"], "distances": comparison["distance_B"]}, bins)
    bin_records = [{**{k: v for k, v in item.items() if k not in {"gap_count", "gap_fraction"}},
                    "A_only_count": item["gap_count"], "A_only_fraction": item["gap_fraction"]}
                   for item in candidates["bins"]]
    rows = reference["global_rows"]
    eligible = np.flatnonzero(candidates["eligible_mask"])
    eligible = eligible[np.argsort(rows[eligible], kind="stable")]
    target_count = int(comparison["A_only"].sum())
    log(f"[A-only bins] all A-only REF={target_count:,}, 100% bins={len(bin_records):,}, "
        f"unique candidate REF={len(eligible):,}; denominator=all REF in bin")
    selected, membership, distances = stage7.radius_cover_representatives(
        reference["distance_coordinates"][eligible], rows[eligible], comparison["distance_B"][eligible], radius*multiplier)
    representatives = eligible[selected]
    counts = np.bincount(membership, minlength=len(selected))
    rank = sorted(range(len(selected)), key=lambda i: (-int(counts[i]),
        -float(comparison["distance_B"][representatives[i]]), int(rows[representatives[i]])))
    old_to_new = np.empty(len(selected), dtype=np.int64)
    old_to_new[rank] = np.arange(len(selected))
    membership, representatives = old_to_new[membership], representatives[rank]
    by_id = {item["bin_id"]: item for item in bin_records}
    groups = []
    for label, index in enumerate(representatives):
        members = eligible[membership == label]
        groups.append({
            "group_id": f"AO{label+1:04d}", "member_count": len(members),
            "fraction_of_A_only_ref": len(members)/target_count,
            "fraction_of_sampled_ref": len(members)/len(rows),
            "representative_global_row": int(rows[index]),
            "representative_pattern_key": str(reference["pattern_keys"][index]),
            "representative_bins": [by_id[key] for key in candidates["bin_ids_by_ref"][index] if key],
            "distance_A": float(comparison["distance_A"][index]),
            "distance_B": float(comparison["distance_B"][index]),
            "nearest_A_pattern_key": str(comparison["nearest_A_pattern_keys"][index]),
            "nearest_B_pattern_key": str(comparison["nearest_B_pattern_keys"][index]),
            "nearest_A_global_row": int(comparison["nearest_A_global_rows"][index]),
            "nearest_B_global_row": int(comparison["nearest_B_global_rows"][index]),
            "is_A_member": bool(comparison["is_A_member"][index]),
            "is_B_member": bool(comparison["is_B_member"][index]),
            "max_member_distance": float(distances[membership == label].max()),
        })
    summary = {"schema_version": 1, "frame_id": str(reference["frame_id"].item()),
        "event": "A_covered_and_B_uncovered", "denominator": "all sampled REF in each bin",
        "candidate_selection": "union of nonempty bins with A_only_count == REF_count",
        "method": "deterministic_farthest_first_radius_cover", "heatmap_bins_per_axis": bins,
        "heatmap_pairs": candidates["pairs"], "coverage_radius": radius, "grouping_radius": radius*multiplier,
        "A_only_ref_count": target_count, "eligible_ref_count": len(eligible),
        "excluded_mixed_bin_A_only_ref_count": target_count-len(eligible),
        "representative_count": len(groups), "fully_A_only_bin_count": len(bin_records),
        "fully_A_only_bins": bin_records, "groups": groups}
    log(f"[A-only representatives] DONE {len(groups):,}, elapsed={time.perf_counter()-started:.1f}s")
    return {"summary": summary, "eligible_indices": eligible, "representative_indices": representatives,
            "membership": membership, "member_distances": distances, "bin_ids_by_ref": candidates["bin_ids_by_ref"]}


def write_patterns_csv(path, bundle, extras=None, indices=None):
    extras = extras or {}
    indices = np.arange(len(bundle["global_rows"])) if indices is None else np.asarray(indices)
    metadata = [key for key in ("h0_labels", "topology_labels") if key in bundle]
    fields = ["pattern_key", "global_row", *metadata, *bundle["feature_names"].tolist(),
              *[f"KP{i+1}" for i in range(bundle["kpca_coordinates"].shape[1])], *extras]
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(fields)
        for i in indices:
            writer.writerow([bundle["pattern_keys"][i], int(bundle["global_rows"][i]),
                *[bundle[name][i] for name in metadata], *bundle["raw_embeddings"][i].tolist(),
                *bundle["kpca_coordinates"][i].tolist(), *[value[i] for value in extras.values()]])
    log(f"[Output] {Path(path).name}: {len(indices):,} rows")


def write_diagnostics(run_dir, reference, comparison, diagnostics):
    summary = diagnostics["summary"]
    indices, groups = diagnostics["representative_indices"], summary["groups"]
    bundle = {name: reference[name][indices] for name in PROJECTION_FIELDS}
    bundle.update(frame_id=reference["frame_id"], feature_names=reference["feature_names"],
                  group_ids=np.asarray([g["group_id"] for g in groups], dtype=str),
                  fully_A_only_bin_ids=diagnostics["bin_ids_by_ref"][indices])
    extras = {"group_id": bundle["group_ids"], "member_count": np.asarray([g["member_count"] for g in groups], dtype=int),
              "fully_A_only_bin_ids": np.asarray([";".join(key for key in ids if key)
                                                for ids in bundle["fully_A_only_bin_ids"]], dtype=str)}
    for name in ("distance_A", "distance_B", "nearest_A_pattern_keys", "nearest_B_pattern_keys",
                 "nearest_A_global_rows", "nearest_B_global_rows", "covered_by_A", "covered_by_B",
                 "is_A_member", "is_B_member"):
        extras[name] = comparison[name][indices]
        bundle[name] = extras[name]
    bundle["member_counts"] = extras["member_count"]
    stage7.save_npz(run_dir / "a_only_representatives.npz", bundle)
    write_patterns_csv(run_dir / "a_only_representatives.csv", bundle, extras)
    with (run_dir / "a_only_members.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["group_id", "pattern_key", "global_row", "representative_pattern_key",
                         "distance_to_representative", "distance_A", "distance_B", "fully_A_only_bin_ids"])
        for index, label, distance in zip(diagnostics["eligible_indices"], diagnostics["membership"], diagnostics["member_distances"]):
            writer.writerow([groups[label]["group_id"], reference["pattern_keys"][index], int(reference["global_rows"][index]),
                groups[label]["representative_pattern_key"], float(distance),
                comparison["distance_A"][index], comparison["distance_B"][index],
                ";".join(key for key in diagnostics["bin_ids_by_ref"][index] if key)])
    with (run_dir / "a_only_bins.csv").open("w", newline="", encoding="utf-8") as fp:
        fields = ["bin_id", "pair", "x_bin", "y_bin", "x_min", "x_max", "y_min", "y_max",
                  "x_upper_inclusive", "y_upper_inclusive", "ref_count", "A_only_count", "A_only_fraction"]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["fully_A_only_bins"])
    stage7.write_json(run_dir / "a_only_summary.json", summary)


def write_bin_statistics(path, grids):
    with Path(path).open("w", newline="", encoding="utf-8") as fp:
        fields = ["pair", "x_bin", "y_bin", "x_min", "x_max", "y_min", "y_max", "ref_count"]
        fields += [name+suffix for name in STATES for suffix in ("_count", "_fraction")]
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for grid in grids:
            a, b = grid["pair"]
            for x, y in np.ndindex(grid["ref_counts"].shape):
                n = int(grid["ref_counts"][x, y])
                row = {"pair": f"KP{a+1}/KP{b+1}", "x_bin": x+1, "y_bin": y+1,
                       "x_min": grid["x_edges"][x], "x_max": grid["x_edges"][x+1],
                       "y_min": grid["y_edges"][y], "y_max": grid["y_edges"][y+1], "ref_count": n}
                for name in STATES:
                    row[name+"_count"] = int(grid[name+"_count"][x, y])
                    row[name+"_fraction"] = float(grid[name+"_fraction"][x, y]) if n else ""
                writer.writerow(row)


def plot_comparison(run_dir, reference, comparison, grids, diagnostics, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import PercentFormatter

    started = time.perf_counter()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#e6e9ec")
    columns = len(grids)

    def heatmap(ax, grid, values, norm):
        mesh = ax.pcolormesh(grid["x_edges"], grid["y_edges"],
            np.ma.masked_invalid(values.T), cmap=cmap, norm=norm, shading="flat", rasterized=True)
        a, b = grid["pair"]
        ax.set(xlabel=f"KP{a+1}", ylabel=f"KP{b+1}", title=f"KP{a+1} / KP{b+1}",
               xlim=grid["x_edges"][[0, -1]], ylim=grid["y_edges"][[0, -1]])
        return mesh

    fig, axes = plt.subplots(3, columns, figsize=(5*columns+1, 11), squeeze=False, constrained_layout=True)
    max_count = max(2, max(float(grid["ref_counts"].max()) for grid in grids))
    for row, (field, label) in enumerate((("ref_counts", "REF count"),
                                          ("A_uncovered_fraction", "A uncovered / REF in bin"),
                                          ("B_uncovered_fraction", "B uncovered / REF in bin"))):
        norm = LogNorm(1, max_count) if row == 0 else Normalize(0, 1)
        for ax, grid in zip(axes[row], grids):
            values = grid[field].copy()
            if row == 0:
                values[values == 0] = np.nan
            mesh = heatmap(ax, grid, values, norm)
        options = {} if row == 0 else {"format": PercentFormatter(1), "ticks": [0, .25, .5, .75, 1]}
        fig.colorbar(mesh, ax=axes[row].tolist(), label=label, shrink=.9, **options)
    fig.suptitle(f"Shared REF | A: topology ({summary['A_count']:,}) | B: external ({summary['B_count']:,})\n"
                 f"Fixed R={summary['radius']:.5g}; decisions use all {summary['kpc_components']} scaled KPCs", fontsize=13)
    fig.savefig(run_dir / "ab_ref_coverage_heatmap.png", dpi=160)
    plt.close(fig)

    for with_representatives in (False, True):
        fig, axes = plt.subplots(1, columns, figsize=(5*columns+1, 4.8), squeeze=False, constrained_layout=True)
        limit = summary["gap_preview_groups"]
        preview = diagnostics["representative_indices"][:limit]
        for column, (ax, grid) in enumerate(zip(axes[0], grids)):
            mesh = heatmap(ax, grid, grid["A_only_fraction"], Normalize(0, 1))
            if not with_representatives:
                continue
            labels = np.flatnonzero(diagnostics["bin_ids_by_ref"][preview, column] != "")
            indices = preview[labels]
            coordinates = reference["kpca_coordinates"]
            a, b = grid["pair"]
            ax.scatter(coordinates[indices, a], coordinates[indices, b], marker="D", s=25,
                       color="#182e3a", edgecolors="white", linewidths=.6, zorder=3)
            ticks = np.linspace(.05, .95, max(17, int(np.ceil(np.sqrt(len(preview))))+2))
            slots = np.stack(np.meshgrid(ticks, ticks), axis=-1).reshape(-1, 2)
            available = np.ones(len(slots), dtype=bool)
            for label, index in zip(labels, indices):
                position = np.array([(coordinates[index, a]-grid["x_edges"][0])/np.ptp(grid["x_edges"]),
                                     (coordinates[index, b]-grid["y_edges"][0])/np.ptp(grid["y_edges"])])
                costs = np.sum((slots-position)**2, axis=1)
                costs[~available] = np.inf
                slot = int(np.argmin(costs))
                available[slot] = False
                ax.annotate(str(int(label)+1), (coordinates[index, a], coordinates[index, b]),
                    xytext=slots[slot], textcoords="axes fraction", fontsize=8, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "color": "#526b79", "lw": .6},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": .7})
            if not len(indices):
                note = ("No A-only REF" if not diagnostics["summary"]["A_only_ref_count"] else
                        "No REF in 100% A-only bins" if not diagnostics["summary"]["eligible_ref_count"] else
                        "No preview representatives in this pair")
                ax.text(.5, .5, note, transform=ax.transAxes, ha="center", va="center",
                        bbox={"facecolor": "white", "alpha": .9})
        fig.colorbar(mesh, ax=axes[0].tolist(), label="A covered & B uncovered / REF in bin",
                     ticks=[0, .25, .5, .75, 1], format=PercentFormatter(1), shrink=.9)
        subtitle = (f"Real representatives from 100% bins; labels 1-{len(preview)} map to AO0001 onward"
                    if with_representatives and len(preview) else
                    "No eligible diagnostic representatives" if with_representatives else
                    "Denominator: ALL REF in bin; gray: no sampled REF")
        fig.suptitle(f"A covers / B misses | {comparison['A_only'].sum():,} REF | Fixed R={summary['radius']:.5g}\n"
                     f"{subtitle}", fontsize=12)
        filename = "a_only_representatives_heatmap.png" if with_representatives else "a_only_coverage_heatmap.png"
        fig.savefig(run_dir / filename, dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, key, color in ((f"A: topology ({summary['A_count']:,})", "distance_A", "#156b91"),
                              (f"B: external ({summary['B_count']:,})", "distance_B", "#c35b37")):
        distances = np.sort(comparison[key])
        ax.step(np.r_[0., distances], np.r_[0., np.arange(1, len(distances)+1)/len(distances)],
                where="post", label=label, color=color, linewidth=2)
    ax.axvline(summary["radius"], color="#5d6670", linestyle="--", label=f"Fixed R={summary['radius']:.5g}")
    ax.set(xlabel="Allowed radius in REF-standardized kPCA", ylabel="Covered fraction of the same REF",
           ylim=(0, 1.02), xlim=(0, None), title="Coverage versus radius")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(run_dir / "coverage_radius_trend.png", dpi=160)
    plt.close(fig)
    log(f"[Plots] DONE four comparison figures, elapsed={time.perf_counter()-started:.1f}s")


def write_report(run_dir, summary, diagnostic):
    import base64
    from html import escape

    def number(value, kind="distance"):
        if value is None:
            return "—"
        return f"{int(value):,}" if kind == "count" else f"{value:.2%}" if kind == "percent" else f"{value:.6g}"

    def table(headers, rows):
        header = "".join(f"<th>{escape(str(x))}</th>" for x in headers)
        body = "".join("<tr>"+"".join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
        return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'

    outcome = summary["outcomes"]
    cards = [("A · 기존 topology 대표", number(summary["A_count"], "count")),
             ("B · B.txt 대표", number(summary["B_count"], "count")),
             ("공통 REF 표본", number(outcome["ref_count"], "count")),
             ("A coverage", number(outcome["coverage_A"], "percent")),
             ("B coverage", number(outcome["coverage_B"], "percent")),
             ("고정 반경 R", number(summary["radius"]))]
    cards = "".join(f'<div class="card"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>'
                    for label, value in cards)
    state_names = {"both_covered": "A와 B 모두 커버", "A_only": "A만 커버 · B는 미커버",
                   "B_only": "B만 커버 · A는 미커버", "both_uncovered": "A와 B 모두 미커버"}
    outcome_table = table(["REF 상태", "REF 수", "전체 REF 중 비율"],
        [[state_names[name], number(outcome["counts"][name], "count"), number(outcome["fractions"][name], "percent")]
         for name in STATES])
    aux = summary["nonmember_outcomes"]
    auxiliary_table = table(["보조 비교 REF 수", "A coverage", "B coverage", "A만 커버", "B만 커버"],
        [[number(aux["ref_count"], "count"), number(aux["coverage_A"], "percent"), number(aux["coverage_B"], "percent"),
          number(aux["fractions"]["A_only"], "percent"), number(aux["fractions"]["B_only"], "percent")]])
    distance_rows = []
    for label, name in (("A · 표준화 kPCA", "A_kpca"), ("B · 표준화 kPCA", "B_kpca"),
                        ("A · 정규화 40D embedding", "A_40d"), ("B · 정규화 40D embedding", "B_40d")):
        stats = summary["nearest_distances"][name]
        distance_rows.append([label]+[number(stats[key]) for key in ("mean", "median", "p95", "p99", "max")])
    distance_table = table(["거리 공간 / 대표군", "평균", "중앙값", "P95", "P99", "최댓값"], distance_rows)
    rows = []
    for group in diagnostic["groups"]:
        bins = "; ".join(f'{item["pair"]} [{item["x_bin"]}, {item["y_bin"]}], REF {item["ref_count"]:,}개'
                         for item in group["representative_bins"])
        rows.append([group["group_id"], group["representative_pattern_key"], group["representative_global_row"],
                     number(group["member_count"], "count"), number(group["distance_A"]), number(group["distance_B"]),
                     "A 대표 자체" if group["is_A_member"] else "REF 패턴", bins])
    headers = ["그룹", "실제 패턴 ID", "원본 row", "그룹 REF 수", "A까지 거리", "B까지 거리", "패턴 구분", "100% bin / REF 수"]
    shown = summary["gap_preview_groups"]
    if rows:
        representatives = table(headers, rows[:shown])
        if len(rows) > shown:
            representatives += f'<details><summary>나머지 {len(rows)-shown:,}개 대표 보기</summary>{table(headers, rows[shown:])}</details>'
    else:
        representatives = ("<p>A만 커버하는 REF가 없습니다.</p>" if not diagnostic["A_only_ref_count"] else
                           "<p>A만 커버하는 REF는 있지만 비율이 100%인 bin이 없어 진단 대표는 0개입니다.</p>")
    descriptions = (
        ("coverage_radius_trend.png", "허용 반경에 따른 A·B coverage",
         "같은 반경에서 더 높은 곡선이 더 많은 REF를 커버합니다. 점선은 7번에서 정한 공통 R입니다."),
        ("ab_ref_coverage_heatmap.png", "B와 REF, 그리고 A의 미커버 분포",
         "위: REF 수. 가운데: A 미커버 비율. 아래: B 미커버 비율. 같은 REF 격자와 0–100% 색상 범위를 사용합니다."),
        ("a_only_coverage_heatmap.png", "A는 커버하지만 B는 놓치는 영역",
         "각 칸의 색은 (A 커버 ∩ B 미커버 REF 수) / (그 칸의 전체 REF 수)입니다. "
         "A가 커버한 REF만을 분모로 쓰는 조건부 비율이 아닙니다. 회색은 REF가 없는 칸입니다."),
        ("a_only_representatives_heatmap.png", "100% A-only bin에서 추린 실제 패턴",
         "마름모가 실제 위치이고 번호는 연결선으로 연결됩니다. 대표는 자기 bin의 A-only 비율이 100%인 축 쌍에만 표시합니다."),
    )
    figures = []
    for filename, title, caption in descriptions:
        path = run_dir / filename
        if path.is_file():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            figures.append(f'<section><h2>{escape(title)}</h2><p>{escape(caption)}</p>'
                           f'<img alt="{escape(title)}" src="data:image/png;base64,{encoded}"></section>')
    css = """
    *{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:#213744;font:16px/1.6 system-ui,sans-serif}
    main{max-width:1440px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 8px}h2{font-size:21px;margin:0 0 12px}
    p{margin:8px 0 16px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}
    .card,section{background:white;border:1px solid #d9e2e8;border-radius:12px;padding:22px}section{margin:20px 0}
    .card span{display:block;color:#526b79}.card strong{display:block;font-size:28px;color:#156b91}
    img{display:block;width:100%;height:auto}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
    th,td{text-align:right;padding:10px 14px;border-bottom:1px solid #e6edf1;white-space:nowrap}th{background:#edf3f6}
    th:first-child,td:first-child{text-align:left}footer{color:#526b79;font-size:13px;overflow-wrap:anywhere}
    @media(max-width:700px){main{padding:14px}.cards{grid-template-columns:repeat(2,1fr);gap:10px}.card,section{padding:14px}}
    """
    html = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>A·B·REF coverage 비교</title><style>{css}</style></head>
<body><main><h1>A·B·REF coverage 비교</h1>
<p>A는 6번에서 추린 topology 대표군, B는 B.txt에서 읽은 외부 대표군입니다.
동일한 REF·정규화·kPCA 좌표계·반경으로 비교합니다. 대표 개수 차이도 coverage에 영향을 줄 수 있어 실제 개수를 함께 표시합니다.</p>
<div class="cards">{cards}</div>
<section><h2>같은 REF에서의 네 가지 결과</h2>{outcome_table}
<p>각 REF는 네 상태 중 정확히 하나에 속합니다. 현재 coverage는 전체 표준화 kPCA 공간 기준이며, 그림은 두 축씩 투영합니다.</p></section>
{''.join(figures)}
<section><h2>A가 커버하고 B가 놓친 실제 패턴</h2>
<p>A-only REF {diagnostic['A_only_ref_count']:,}개 중 100% bin {diagnostic['fully_A_only_bin_count']:,}개에 속한 고유 후보
{diagnostic['eligible_ref_count']:,}개를 모아 실제 대표 {diagnostic['representative_count']:,}개를 추렸습니다.
혼합 bin에만 있는 {diagnostic['excluded_mixed_bin_A_only_ref_count']:,}개는 진단 후보에서 제외되지만 전체 A-only 목록에는 남습니다.</p>
<p>100%란 해당 bin에서 관측한 REF 모두를 A가 커버하고 B가 놓쳤다는 뜻입니다. 세 축 쌍 중 하나라도 조건을 만족하면 후보이며
같은 REF는 한 번만 셉니다. REF가 한 개뿐인 bin도 포함하므로 bin별 REF 수를 함께 확인하세요.
후보는 전체 표준화 kPCA 공간에서 반경 {number(diagnostic['grouping_radius'])}으로 묶고, 그룹 REF 수가 많은 순서로 표시합니다.
그림의 숫자 1은 AO0001에 해당합니다. 이 진단 대표를 A 또는 B에 자동 추가하지 않습니다.</p>
{representatives}<p>실제 대표: a_only_representatives.csv · 후보 구성원: a_only_members.csv · bin 경계: a_only_bins.csv<br>
전체 A-only REF: a_only_ref_patterns.csv · B 미커버 REF: b_uncovered_ref_patterns.csv</p></section>
<section><h2>대표 자체와의 매칭을 제외한 공통 보조 비교</h2>
<p>REF 중 A 또는 B에 들어 있는 {summary['ref_member_union_count']:,}개를 양쪽에서 똑같이 제외했습니다.
위의 기본 coverage와 heatmap은 전체 REF를 사용합니다.</p>{auxiliary_table}</section>
<section><h2>REF에서 최근접 대표까지의 거리</h2>{distance_table}
<p>동일 공간의 A·B 행을 비교합니다. 원래 40D 거리는 보조 지표이며 kPCA 거리와 숫자를 직접 비교하지 않습니다.</p></section>
<footer>입력 {summary['B_input_count']:,}줄 → 고유 B {summary['B_count']:,}개 · 중복 {summary['B_duplicate_count']:,}줄<br>
격자: {summary['heatmap_bins_per_axis']} × {summary['heatmap_bins_per_axis']} · Frame: {escape(summary['frame_id'])}<br>
분석 범위는 저장된 REF 표본입니다. 그림은 이 문서에 내장되어 인터넷 없이 열 수 있습니다.</footer></main></body></html>'''
    (run_dir / "comparison_report.html").write_text(html, encoding="utf-8")
    log("[Report] SAVED comparison_report.html (offline, images embedded)")


def validate_config():
    for name, value in (("CPU_THREADS", CPU_THREADS), ("TRANSFORM_BLOCK", TRANSFORM_BLOCK),
                        ("DISTANCE_BLOCK", DISTANCE_BLOCK), ("GAP_PREVIEW_GROUPS", GAP_PREVIEW_GROUPS)):
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if not np.isfinite(GAP_RADIUS_MULTIPLIER) or GAP_RADIUS_MULTIPLIER <= 0:
        raise ValueError("GAP_RADIUS_MULTIPLIER must be finite and > 0")
    if KERNEL_BACKEND not in {"auto", "numpy_cpu", "torch_gpu"} or GPU_DEVICE < 0:
        raise ValueError("invalid kernel backend or GPU device")


def main(run_dir, b_keys_path=B_KEYS_PATH, stage7_run=STAGE7_RUN, cache_path=CACHE_PATH):
    from threadpoolctl import threadpool_limits

    validate_config()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with threadpool_limits(limits=CPU_THREADS):
        source_dir, frame, reference, sample_a, bins = load_baseline(stage7_run)
        if run_dir.resolve() == source_dir.resolve():
            raise ValueError("Stage-8 output must be separate from the Stage-7 source run")
        keys, occurrences, content = read_b_keys(b_keys_path)
        (run_dir / "b_input.txt").write_bytes(content)
        rows, raw_b = extract_b_embeddings(cache_path, keys, occurrences, reference, sample_a, frame,
                                           run_dir / "b_key_lookup.csv")
        backend = stage7.choose_kernel_backend(KERNEL_BACKEND, GPU_DEVICE)
        sample_b, projection_counts = project_b(raw_b, rows, keys, reference, sample_a, frame, backend)
        radius = float(frame["radius"])
        comparison = evaluate_b(reference, sample_a, sample_b, radius)
        diagnostics = select_a_only_representatives(reference, comparison, radius, bins, GAP_RADIUS_MULTIPLIER)
        grids = comparison_heatmap_grids(reference, comparison, bins)
        saved_reference = {name: reference[name] for name in PROJECTION_FIELDS}
        saved_reference.update({name: reference[name] for name in ("h0_labels", "topology_labels") if name in reference})
        saved_reference.update(frame_id=frame["frame_id"], feature_names=frame["feature_names"], **comparison)
        stage7.save_npz(run_dir / "reference_frame.npz", frame)
        stage7.save_npz(run_dir / "reference_comparison.npz", saved_reference)
        stage7.save_npz(run_dir / "a_sample.npz", sample_a)
        stage7.save_npz(run_dir / "b_sample.npz", sample_b)
        write_patterns_csv(run_dir / "b_features.csv", sample_b)
        write_patterns_csv(run_dir / "ref_comparison.csv", reference, comparison)
        write_patterns_csv(run_dir / "a_only_ref_patterns.csv", reference, comparison, np.flatnonzero(comparison["A_only"]))
        write_patterns_csv(run_dir / "b_uncovered_ref_patterns.csv", reference, comparison, np.flatnonzero(~comparison["covered_by_B"]))
        write_diagnostics(run_dir, reference, comparison, diagnostics)
        write_bin_statistics(run_dir / "coverage_bins.csv", grids)
        outcome = summarize_outcomes(comparison)
        summary = {"schema_version": 1, "frame_id": str(frame["frame_id"].item()),
            "analysis": "A_topology_vs_B_external_on_fixed_REF", "source_stage7_run": str(source_dir.resolve()),
            "input_B": str(Path(b_keys_path).resolve()), "input_cache": str(Path(cache_path).resolve()),
            "B_input_sha256": hashlib.sha256(content).hexdigest(),
            "B_input_count": sum(occurrences.values()), "B_duplicate_count": sum(occurrences.values())-len(keys),
            "A_count": len(sample_a["global_rows"]), "B_count": len(rows),
            "feature_dimension": raw_b.shape[1], "kpc_components": len(frame["kpc_scale"]),
            "radius": radius, "radius_budget": int(frame["radius_budget"]), "heatmap_bins_per_axis": bins,
            "gap_radius_multiplier": GAP_RADIUS_MULTIPLIER, "gap_preview_groups": GAP_PREVIEW_GROUPS,
            "kernel_backend": backend, "B_projection": projection_counts, "outcomes": outcome,
            "nonmember_outcomes": summarize_outcomes(comparison, ~comparison["excluded_from_nonmember_comparison"]),
            "ref_A_member_count": int(comparison["is_A_member"].sum()),
            "ref_B_member_count": int(comparison["is_B_member"].sum()),
            "ref_member_union_count": int(comparison["excluded_from_nonmember_comparison"].sum()),
            "nearest_distances": {label: stage7.distance_summary(comparison[name]) for label, name in (
                ("A_kpca", "distance_A"), ("B_kpca", "distance_B"), ("A_40d", "distance_A_40d"), ("B_40d", "distance_B_40d"))},
            "snapshot_fingerprints": {"reference_comparison": stage7.arrays_fingerprint(saved_reference),
                                      "a_sample": stage7.arrays_fingerprint(sample_a),
                                      "b_sample": stage7.arrays_fingerprint(sample_b)},
            "scope": "Full saved REF sample; coverage in all REF-standardized kPCA axes; bin denominator is all REF in bin.",
        }
        if MAKE_PLOTS:
            plot_comparison(run_dir, reference, comparison, grids, diagnostics, summary)
        summary["elapsed_seconds"] = time.perf_counter()-started
        stage7.write_json(run_dir / "comparison_summary.json", summary)
        write_report(run_dir, summary, diagnostics["summary"])
        log(f"[Result] DONE elapsed={time.perf_counter()-started:.1f}s; {run_dir}")
    return summary


def run_with_log(b_keys_path, stage7_run, cache_path, output_root):
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "comparison.log").open("x", encoding="utf-8", buffering=1) as fp:
        with contextlib.redirect_stdout(stage7._Tee(sys.stdout, fp)), contextlib.redirect_stderr(stage7._Tee(sys.stderr, fp)):
            log(f"[Run] START {run_dir}")
            try:
                summary = main(run_dir, b_keys_path, stage7_run, cache_path)
                pointer = Path(output_root) / f"latest_{run_id}.tmp"
                stage7.write_json(pointer, {"run_directory": run_id, "frame_id": summary["frame_id"]})
                pointer.replace(Path(output_root) / "latest_run.json")
            except KeyboardInterrupt:
                log("[Run] INTERRUPTED; partial output retained")
                return 130
            except Exception:
                traceback.print_exc()
                log("[Run] FAILED; see comparison.log and b_key_lookup.csv; latest success retained")
                return 1
            log("[Run] SUCCESS")
    return 0


def replot_saved_run(run, output_root):
    run_dir, expected_id = resolve_run(run, output_root)
    with (run_dir / "replot.log").open("a", encoding="utf-8", buffering=1) as fp:
        with contextlib.redirect_stdout(stage7._Tee(sys.stdout, fp)), contextlib.redirect_stderr(stage7._Tee(sys.stderr, fp)):
            try:
                log(f"[Replot] START {run_dir}")
                frame = stage7.load_reference_frame(run_dir / "reference_frame.npz")
                with (run_dir / "comparison_summary.json").open(encoding="utf-8") as summary_file:
                    summary = json.load(summary_file)
                if (summary["frame_id"] != str(frame["frame_id"].item())
                        or (expected_id is not None and expected_id != summary["frame_id"])
                        or summary["radius"] != float(frame["radius"])):
                    raise ValueError("comparison summary/pointer belongs to a different frame")
                snapshots = {}
                for name, fingerprint in summary["snapshot_fingerprints"].items():
                    bundle = stage7.load_projection_bundle(run_dir / f"{name}.npz", frame)
                    if stage7.arrays_fingerprint(bundle) != fingerprint:
                        raise ValueError(f"saved {name} fingerprint differs")
                    snapshots[name] = bundle
                reference = snapshots["reference_comparison"]
                expected = classify_coverage(reference["distance_A"], reference["distance_B"], summary["radius"])
                if any(not np.array_equal(reference[name], mask) for name, mask in expected.items()):
                    raise ValueError("saved A/B status differs from saved distances and R")
                bins = summary["heatmap_bins_per_axis"]
                diagnostics = select_a_only_representatives(reference, reference, summary["radius"], bins, summary["gap_radius_multiplier"])
                grids = comparison_heatmap_grids(reference, reference, bins)
                write_diagnostics(run_dir, reference, reference, diagnostics)
                plot_comparison(run_dir, reference, reference, grids, diagnostics, summary)
                write_report(run_dir, summary, diagnostics["summary"])
                log("[Replot] DONE; saved frame, distances and comparison snapshots preserved")
            except Exception:
                traceback.print_exc()
                return 1
    return 0


def cli():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-keys", type=Path, default=B_KEYS_PATH)
    parser.add_argument("--stage7-run", default=STAGE7_RUN, help="Stage-7 run directory or latest")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--replot", metavar="RUN_DIRECTORY_OR_latest", help="redraw saved Stage-8 output without cache or GPU")
    args = parser.parse_args()
    if args.replot is not None:
        return replot_saved_run(args.replot, args.out_dir)
    return run_with_log(args.b_keys, args.stage7_run, args.cache, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(cli())
