"""
Stage 3 fixed-budget representative selection.

Consumes the Stage-2 topology labels and selects exactly 1,000 real pattern
keys. Every final topology community receives a minimum quota; remaining slots
are apportioned by community population and normalized-topology dispersion.
Within each community, deterministic centroid-nearest-seeded farthest-point
selection covers both the central 80% and outer 20% of the population.
"""

import csv
import gc
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np


# ============================================================================
# CONFIG
# ============================================================================
REPRESENTATIVE_BUDGET = 1_000
MIN_REPRESENTATIVES_PER_COMMUNITY = 3
CORE_POPULATION_FRACTION = 0.80
SIZE_WEIGHT_EXPONENT = 0.50
DISPERSION_WEIGHT_EXPONENT = 1.00
USE_CUDA_FOR_SELECTION = True
GPU_DEVICE = 0
DISTANCE_BLOCK = 250_000

_here = Path(__file__).resolve().parent
STAGE2_LABELS_PATH = _here / "topology_clustering_out" / "topology_labels.npz"
STAGE2_SUMMARY_PATH = (
    _here / "topology_clustering_out" / "topology_cluster_summary.csv"
)
OUT_DIR = _here / "topology_clustering_out"


def _load_topology_module():
    module_path = _here / "6.topology_clustering.py"
    spec = importlib.util.spec_from_file_location("topology_clustering", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"6번 module을 불러올 수 없음: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_config():
    if REPRESENTATIVE_BUDGET < 1:
        raise ValueError("REPRESENTATIVE_BUDGET must be >= 1")
    if MIN_REPRESENTATIVES_PER_COMMUNITY < 1:
        raise ValueError("MIN_REPRESENTATIVES_PER_COMMUNITY must be >= 1")
    if not 0 < CORE_POPULATION_FRACTION < 1:
        raise ValueError("CORE_POPULATION_FRACTION must be between 0 and 1")
    if SIZE_WEIGHT_EXPONENT < 0 or DISPERSION_WEIGHT_EXPONENT < 0:
        raise ValueError("allocation exponents must be >= 0")
    if GPU_DEVICE < 0 or DISTANCE_BLOCK < 1:
        raise ValueError("GPU_DEVICE must be >= 0 and DISTANCE_BLOCK must be >= 1")


def _load_stage2_labels(
    expected_rows, expected_h0_labels, expected_population_fingerprint
):
    if not STAGE2_LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Stage-2 label 없음: {STAGE2_LABELS_PATH} (먼저 최신 6번 실행)"
        )
    with np.load(STAGE2_LABELS_PATH) as stage2:
        if "population_fingerprint" not in stage2.files:
            raise ValueError(
                "Stage-2 labels에 population fingerprint가 없음; "
                "최신 6번을 다시 실행하세요"
            )
        rows = np.asarray(stage2["rows"], dtype=np.int64)
        h0_labels = np.asarray(stage2["h0_labels"], dtype=np.int32)
        topology_labels = np.asarray(stage2["topology_labels"], dtype=np.int32)
        fingerprint = str(stage2["population_fingerprint"].item())

    if not np.array_equal(rows, expected_rows):
        raise ValueError("Stage-2 rows가 현재 feature/Stage-1 rows와 다름")
    if not np.array_equal(h0_labels, expected_h0_labels):
        raise ValueError("Stage-2 H0 labels가 현재 Stage-1 labels와 다름")
    if fingerprint != expected_population_fingerprint:
        raise ValueError(
            "Stage-2 population fingerprint가 현재 feature key/order와 다름; "
            "6번을 다시 실행하세요"
        )
    if topology_labels.shape != rows.shape or (topology_labels < 0).any():
        raise ValueError("Stage-2 topology label이 누락되었거나 shape이 다름")
    return topology_labels


def _load_community_summary():
    if not STAGE2_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Stage-2 summary 없음: {STAGE2_SUMMARY_PATH} (먼저 최신 6번 실행)"
        )
    required = {
        "h0_label", "coarse_group", "topology_cluster", "final_cluster",
        "pattern_count", "centroid_distance_p95",
    }
    records = []
    with STAGE2_SUMMARY_PATH.open("r", newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Stage-2 summary에 {sorted(missing)} 없음; 최신 6번을 다시 실행하세요"
            )
        for row in reader:
            record = {
                "h0_label": int(row["h0_label"]),
                "coarse_group": row["coarse_group"],
                "topology_cluster": int(row["topology_cluster"]),
                "final_cluster": row["final_cluster"],
                "pattern_count": int(row["pattern_count"]),
                "centroid_distance_p95": float(row["centroid_distance_p95"]),
            }
            if record["pattern_count"] < 1:
                raise ValueError(f"빈 community 발견: {record['final_cluster']}")
            if (
                not np.isfinite(record["centroid_distance_p95"])
                or record["centroid_distance_p95"] < 0
            ):
                raise ValueError(
                    f"유효한 산포가 없는 summary: {record['final_cluster']}"
                )
            expected_coarse = (
                "rare" if record["h0_label"] < 0
                else f"H0_{record['h0_label']}"
            )
            expected_final = (
                "RARE" if record["h0_label"] < 0
                else f"H0_{record['h0_label']}"
            ) + f"_T{record['topology_cluster']}"
            if (
                record["coarse_group"] != expected_coarse
                or record["final_cluster"] != expected_final
            ):
                raise ValueError(
                    f"community ID 불일치: {record['final_cluster']}"
                )
            records.append(record)
    if not records:
        raise ValueError("Stage-2 community summary가 비어 있음")
    return records


def _validate_summary_against_labels(records, h0_labels, topology_labels):
    observed = {}
    for h0_label in np.unique(h0_labels):
        mask = h0_labels == h0_label
        ids, counts = np.unique(topology_labels[mask], return_counts=True)
        for topology_label, count in zip(ids, counts):
            observed[(int(h0_label), int(topology_label))] = int(count)

    summarized = {
        (record["h0_label"], record["topology_cluster"]): record["pattern_count"]
        for record in records
    }
    if len(summarized) != len(records):
        raise ValueError("Stage-2 summary에 중복 community가 있음")
    if observed != summarized:
        missing = sorted(set(observed) - set(summarized))[:5]
        extra = sorted(set(summarized) - set(observed))[:5]
        raise ValueError(
            f"Stage-2 summary/labels 불일치: missing={missing}, extra={extra}"
        )


def allocate_community_quotas(records, budget, minimum_per_community):
    """Deterministically apportion an exact budget with rare-safe floors."""
    if budget < 1 or minimum_per_community < 1:
        raise ValueError("budget and minimum_per_community must be >= 1")
    records = [dict(record) for record in records]
    community_count = len(records)
    total_patterns = sum(record["pattern_count"] for record in records)
    if total_patterns < budget:
        raise ValueError(
            f"candidate pattern {total_patterns:,}개보다 budget {budget:,}개가 큼"
        )
    if community_count > budget:
        raise ValueError(
            f"community {community_count:,}개가 budget {budget:,}개보다 많아 "
            "community당 최소 1개를 보장할 수 없음"
        )

    effective_floor = min(minimum_per_community, max(1, budget // community_count))
    positive_dispersions = [
        record["centroid_distance_p95"]
        for record in records
        if record["centroid_distance_p95"] > 0
    ]
    dispersion_floor = (
        float(np.median(positive_dispersions)) * 0.10
        if positive_dispersions else 1.0
    )

    for record in records:
        count = record["pattern_count"]
        dispersion = max(record["centroid_distance_p95"], dispersion_floor)
        record["minimum_quota"] = min(count, effective_floor)
        record["allocated_quota"] = record["minimum_quota"]
        record["allocation_dispersion"] = dispersion
        record["allocation_weight"] = (
            count ** SIZE_WEIGHT_EXPONENT
            * dispersion ** DISPERSION_WEIGHT_EXPONENT
        )

    remaining = budget - sum(record["allocated_quota"] for record in records)
    while remaining > 0:
        eligible = [
            index for index, record in enumerate(records)
            if record["allocated_quota"] < record["pattern_count"]
        ]
        if not eligible:
            raise RuntimeError("quota capacity가 부족하여 budget을 채울 수 없음")
        total_weight = sum(records[index]["allocation_weight"] for index in eligible)
        if total_weight <= 0:
            weights = {index: 1.0 for index in eligible}
            total_weight = float(len(eligible))
        else:
            weights = {
                index: records[index]["allocation_weight"] for index in eligible
            }

        ideal = {
            index: remaining * weights[index] / total_weight for index in eligible
        }
        granted = 0
        for index in eligible:
            capacity = (
                records[index]["pattern_count"]
                - records[index]["allocated_quota"]
            )
            addition = min(capacity, int(np.floor(ideal[index])))
            records[index]["allocated_quota"] += addition
            remaining -= addition
            granted += addition
        if remaining == 0:
            break

        eligible = [
            index for index in eligible
            if records[index]["allocated_quota"] < records[index]["pattern_count"]
        ]
        order = sorted(
            eligible,
            key=lambda index: (
                -(ideal[index] - np.floor(ideal[index])),
                -records[index]["allocation_weight"],
                records[index]["final_cluster"],
            ),
        )
        for index in order:
            if remaining == 0:
                break
            records[index]["allocated_quota"] += 1
            remaining -= 1
            granted += 1
        if granted == 0:
            raise RuntimeError("quota allocation이 진행되지 않음")

    if sum(record["allocated_quota"] for record in records) != budget:
        raise RuntimeError("allocated quota 합계가 budget과 다름")
    return records


def _centroid_and_squared_distances(embedding):
    centroid_sum = np.zeros(embedding.shape[1], dtype=np.float64)
    for start in range(0, len(embedding), DISTANCE_BLOCK):
        end = min(start + DISTANCE_BLOCK, len(embedding))
        centroid_sum += embedding[start:end].sum(axis=0, dtype=np.float64)
    centroid = (centroid_sum / len(embedding)).astype(np.float32)

    squared_distances = np.empty(len(embedding), dtype=np.float32)
    for start in range(0, len(embedding), DISTANCE_BLOCK):
        end = min(start + DISTANCE_BLOCK, len(embedding))
        delta = embedding[start:end] - centroid
        squared_distances[start:end] = np.einsum("ij,ij->i", delta, delta)
    np.maximum(squared_distances, 0.0, out=squared_distances)
    return centroid, squared_distances


class _NumpyFarthestState:
    def __init__(self, embedding):
        self.embedding = embedding
        self.selected = np.zeros(len(embedding), dtype=bool)
        self.minimum_squared_distance = np.full(
            len(embedding), np.inf, dtype=np.float32
        )
        self.backend = f"numpy_cpu(threads controlled by NumPy, block={DISTANCE_BLOCK:,})"

    def choose(self, pool_mask):
        candidates = pool_mask & ~self.selected
        if not candidates.any():
            return None
        scores = np.where(candidates, self.minimum_squared_distance, -np.inf)
        return int(np.argmax(scores))

    def add(self, index):
        prior = self.minimum_squared_distance[index]
        self.selected[index] = True
        vector = self.embedding[index]
        for start in range(0, len(self.embedding), DISTANCE_BLOCK):
            end = min(start + DISTANCE_BLOCK, len(self.embedding))
            delta = self.embedding[start:end] - vector
            distance = np.einsum("ij,ij->i", delta, delta)
            np.minimum(
                self.minimum_squared_distance[start:end],
                distance,
                out=self.minimum_squared_distance[start:end],
            )
        return 0.0 if not np.isfinite(prior) else float(np.sqrt(max(prior, 0.0)))


class _TorchFarthestState:
    def __init__(self, embedding, torch, device):
        self.torch = torch
        self.device = torch.device(device)
        self.embedding = torch.from_numpy(
            np.ascontiguousarray(embedding, dtype=np.float32)
        ).to(self.device)
        self.selected = torch.zeros(
            len(embedding), dtype=torch.bool, device=self.device
        )
        self.minimum_squared_distance = torch.full(
            (len(embedding),),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        torch.cuda.synchronize(self.device)
        self.backend = f"torch_cuda({torch.cuda.get_device_name(self.device)})"

    def choose(self, pool_mask):
        pool = self.torch.from_numpy(pool_mask).to(self.device)
        candidates = pool & ~self.selected
        if not bool(candidates.any().item()):
            return None
        scores = self.minimum_squared_distance.masked_fill(~candidates, -float("inf"))
        return int(self.torch.argmax(scores).item())

    def add(self, index):
        prior = float(self.minimum_squared_distance[index].item())
        self.selected[index] = True
        distance = ((self.embedding - self.embedding[index]) ** 2).sum(dim=1)
        self.minimum_squared_distance = self.torch.minimum(
            self.minimum_squared_distance, distance
        )
        if not np.isfinite(prior):
            return 0.0
        return float(np.sqrt(max(prior, 0.0)))


def _make_farthest_state(embedding, use_cuda):
    if use_cuda:
        try:
            import torch
        except (ImportError, OSError):
            torch = None
        if (
            torch is not None
            and torch.cuda.is_available()
            and torch.cuda.device_count() > GPU_DEVICE
        ):
            try:
                return _TorchFarthestState(
                    embedding, torch, device=f"cuda:{GPU_DEVICE}"
                )
            except Exception as exc:
                print(
                    f"    ⚠ CUDA representative selection 실패, CPU fallback: {exc}",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
    return _NumpyFarthestState(embedding)


def select_community_representatives(embedding, quota, use_cuda=True):
    """Select actual member indices by centroid seed + core/boundary coverage."""
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    if len(embedding) < 1 or not 1 <= quota <= len(embedding):
        raise ValueError("quota must be between 1 and community size")

    _, centroid_squared_distance = _centroid_and_squared_distances(embedding)
    centroid_nearest = int(np.argmin(centroid_squared_distance))
    threshold = float(
        np.quantile(centroid_squared_distance, CORE_POPULATION_FRACTION)
    )
    core_mask = centroid_squared_distance <= threshold
    boundary_mask = ~core_mask

    if quota == 1 or not boundary_mask.any():
        boundary_target = 0
    else:
        boundary_target = max(
            1, int(round(quota * (1.0 - CORE_POPULATION_FRACTION)))
        )
        boundary_target = min(boundary_target, int(boundary_mask.sum()))
    core_target = quota - boundary_target
    if core_target > int(core_mask.sum()):
        shortage = core_target - int(core_mask.sum())
        core_target -= shortage
        boundary_target += shortage

    state = _make_farthest_state(embedding, use_cuda=use_cuda)
    selected = []

    def append(index, reason):
        distance_to_prior = state.add(index)
        selected.append({
            "local_index": int(index),
            "selection_reason": reason,
            "distance_to_centroid": float(
                np.sqrt(max(centroid_squared_distance[index], 0.0))
            ),
            "distance_to_nearest_prior": distance_to_prior,
        })

    append(centroid_nearest, "centroid_nearest")
    while len(selected) < core_target:
        index = state.choose(core_mask)
        if index is None:
            break
        append(index, "core_farthest")

    boundary_selected = 0
    while boundary_selected < boundary_target:
        index = state.choose(boundary_mask)
        if index is None:
            break
        append(index, "boundary_farthest")
        boundary_selected += 1

    all_mask = np.ones(len(embedding), dtype=bool)
    while len(selected) < quota:
        index = state.choose(all_mask)
        if index is None:
            raise RuntimeError("unique representative candidate가 부족함")
        append(index, "coverage_fill")

    return selected, state.backend


def _write_outputs(manifest, quota_records, metadata):
    started = time.time()
    print("\n[Output] representative writing START", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_DIR / "representative_1000.csv"
    quota_path = OUT_DIR / "representative_quota.csv"
    labels_path = OUT_DIR / "representative_1000.npz"
    metadata_path = OUT_DIR / "representative_selection_metadata.json"

    manifest_fields = [
        "pattern_key", "global_row", "h0_label", "topology_cluster",
        "final_cluster", "selection_rank", "global_selection_rank",
        "selection_reason", "distance_to_centroid", "distance_to_nearest_prior",
    ]
    manifest_tmp = str(manifest_path) + ".tmp"
    with open(manifest_tmp, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest)
    os.replace(manifest_tmp, manifest_path)

    quota_fields = [
        "h0_label", "coarse_group", "topology_cluster", "final_cluster",
        "pattern_count", "centroid_distance_p95", "allocation_dispersion",
        "allocation_weight", "minimum_quota", "allocated_quota",
    ]
    quota_tmp = str(quota_path) + ".tmp"
    with open(quota_tmp, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=quota_fields)
        writer.writeheader()
        writer.writerows(quota_records)
    os.replace(quota_tmp, quota_path)

    labels_tmp = str(labels_path) + ".tmp.npz"
    np.savez(
        labels_tmp,
        rows=np.asarray([row["global_row"] for row in manifest], dtype=np.int64),
        h0_labels=np.asarray([row["h0_label"] for row in manifest], dtype=np.int32),
        topology_labels=np.asarray(
            [row["topology_cluster"] for row in manifest], dtype=np.int32
        ),
    )
    os.replace(labels_tmp, labels_path)

    metadata_tmp = str(metadata_path) + ".tmp"
    with open(metadata_tmp, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    os.replace(metadata_tmp, metadata_path)

    print("\n[Saved]", flush=True)
    for path in (manifest_path, quota_path, labels_path, metadata_path):
        print(f"  {path}", flush=True)
    print(
        f"[Output] representative writing DONE elapsed={time.time() - started:.1f}s",
        flush=True,
    )


def main():
    _validate_config()
    started = time.time()
    topology = _load_topology_module()
    topology._validate_config()

    print("=" * 72, flush=True)
    print("Stage 3: fixed-budget topology representative selection", flush=True)
    print("=" * 72, flush=True)

    features, keys, rows, h0_labels, dims = topology.load_inputs()
    fingerprint_started = time.time()
    print("[Population fingerprint] validation START", flush=True)
    expected_population_fingerprint = topology.population_fingerprint(keys, rows)
    topology_labels = _load_stage2_labels(
        rows, h0_labels, expected_population_fingerprint
    )
    print(
        "[Population fingerprint] validation DONE "
        f"elapsed={time.time() - fingerprint_started:.1f}s",
        flush=True,
    )
    summary_records = _load_community_summary()
    _validate_summary_against_labels(summary_records, h0_labels, topology_labels)
    quota_records = allocate_community_quotas(
        summary_records,
        budget=REPRESENTATIVE_BUDGET,
        minimum_per_community=MIN_REPRESENTATIVES_PER_COMMUNITY,
    )
    quota_by_cluster = {
        (record["h0_label"], record["topology_cluster"]): record
        for record in quota_records
    }

    print(f"Candidate patterns          : {len(rows):,}", flush=True)
    print(f"Final communities           : {len(quota_records):,}", flush=True)
    print(f"Representative budget       : {REPRESENTATIVE_BUDGET:,}", flush=True)
    effective_floor = min(
        MIN_REPRESENTATIVES_PER_COMMUNITY,
        max(1, REPRESENTATIVE_BUDGET // len(quota_records)),
    )
    print(
        "Minimum/community           : "
        f"configured={MIN_REPRESENTATIVES_PER_COMMUNITY}, "
        f"effective={effective_floor} (capped by community size)",
        flush=True,
    )
    print(
        "Allocation weight           : "
        f"count^{SIZE_WEIGHT_EXPONENT} * p95_dispersion^{DISPERSION_WEIGHT_EXPONENT}",
        flush=True,
    )
    print(
        f"Core/boundary selection      : {CORE_POPULATION_FRACTION:.0%}/"
        f"{1.0 - CORE_POPULATION_FRACTION:.0%}",
        flush=True,
    )

    scalers = topology.fit_block_scalers(features, rows)
    manifest = []
    selection_backends = set()
    coarse_groups = sorted(int(value) for value in np.unique(h0_labels) if value >= 0)
    if (h0_labels < 0).any():
        coarse_groups.append(-1)

    for group_index, h0_label in enumerate(coarse_groups, start=1):
        group_started = time.time()
        positions = np.flatnonzero(h0_labels == h0_label)
        global_rows = rows[positions]
        print("\n" + "-" * 72, flush=True)
        print(
            f"[Group {group_index}/{len(coarse_groups)}] "
            f"{topology.coarse_group_name(h0_label)}, patterns={len(positions):,}",
            flush=True,
        )
        embedding = topology.make_normalized_embedding(
            features, scalers, global_rows, dims
        )
        group_topology_labels = topology_labels[positions]

        topology_ids = sorted(
            int(value) for value in np.unique(group_topology_labels)
        )
        for topology_label in topology_ids:
            cluster_key = (int(h0_label), topology_label)
            quota_record = quota_by_cluster[cluster_key]
            quota = quota_record["allocated_quota"]
            member_local = np.flatnonzero(group_topology_labels == topology_label)
            community_embedding = np.ascontiguousarray(embedding[member_local])
            print(
                f"  [{quota_record['final_cluster']}] patterns={len(member_local):,}, "
                f"quota={quota}",
                flush=True,
            )
            community_started = time.time()
            selected, backend = select_community_representatives(
                community_embedding,
                quota=quota,
                use_cuda=USE_CUDA_FOR_SELECTION,
            )
            selection_backends.add(backend)
            for rank, selected_row in enumerate(selected, start=1):
                group_position = member_local[selected_row["local_index"]]
                population_position = positions[group_position]
                global_row = int(rows[population_position])
                manifest.append({
                    "pattern_key": str(keys[global_row]),
                    "global_row": global_row,
                    "h0_label": int(h0_label),
                    "topology_cluster": topology_label,
                    "final_cluster": quota_record["final_cluster"],
                    "selection_rank": rank,
                    "global_selection_rank": len(manifest) + 1,
                    "selection_reason": selected_row["selection_reason"],
                    "distance_to_centroid": selected_row["distance_to_centroid"],
                    "distance_to_nearest_prior": selected_row[
                        "distance_to_nearest_prior"
                    ],
                })
            print(
                f"    selected={len(selected):,}, backend={backend}, "
                f"elapsed={time.time() - community_started:.1f}s",
                flush=True,
            )
            del community_embedding, selected
            gc.collect()

        print(
            f"[Group {group_index}/{len(coarse_groups)}] DONE "
            f"elapsed={time.time() - group_started:.1f}s",
            flush=True,
        )
        del embedding, group_topology_labels
        gc.collect()

    selected_rows = [row["global_row"] for row in manifest]
    selected_keys = [row["pattern_key"] for row in manifest]
    if len(manifest) != REPRESENTATIVE_BUDGET:
        raise RuntimeError(
            f"대표패턴 수 {len(manifest):,} != budget {REPRESENTATIVE_BUDGET:,}"
        )
    if len(set(selected_rows)) != len(selected_rows):
        raise RuntimeError("대표패턴 global_row가 중복됨")
    if len(set(selected_keys)) != len(selected_keys):
        raise RuntimeError("대표패턴 pattern_key가 중복됨")

    metadata = {
        "candidate_patterns": len(rows),
        "representative_budget": REPRESENTATIVE_BUDGET,
        "selected_patterns": len(manifest),
        "final_communities": len(quota_records),
        "minimum_representatives_per_community": (
            MIN_REPRESENTATIVES_PER_COMMUNITY
        ),
        "effective_minimum_floor": effective_floor,
        "core_population_fraction": CORE_POPULATION_FRACTION,
        "size_weight_exponent": SIZE_WEIGHT_EXPONENT,
        "dispersion_weight_exponent": DISPERSION_WEIGHT_EXPONENT,
        "feature_dimensions": dims,
        "feature_blocks": list(topology.FEATURE_BLOCKS),
        "block_weights": topology.BLOCK_WEIGHTS,
        "stage2_labels_path": str(STAGE2_LABELS_PATH),
        "stage2_summary_path": str(STAGE2_SUMMARY_PATH),
        "population_fingerprint": expected_population_fingerprint,
        "selection_backends": sorted(selection_backends),
        "elapsed_seconds": time.time() - started,
    }
    _write_outputs(manifest, quota_records, metadata)
    print(f"완료: {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
