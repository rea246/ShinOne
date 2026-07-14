"""
Dummy Tensor DB Generator
실제 OAS/GG 파일 없이 GNN 전처리기와 동일한 포맷의 더미 데이터셋을 생성합니다.
출력: PREPROCESSED_{name}_part{idx}.pt (실제 전처리기와 동일한 구조)
"""

import os
import gc
import time
import numpy as np
import torch
from typing import List, Tuple, Optional

# ── 파라미터 ────────────────────────────────────────────────────────────────
NUM_GAUGES      = 10_000          # 생성할 게이지 수
NODES_PER_GAUGE = (10, 30)        # 게이지당 박스 수 범위
HALF_WIN_NM     = 256.0           # window_size=0.512um → half=256nm
MAX_DIST_NM     = 300.0           # edge 연결 최대 거리 (nm)
CHUNK_SIZE      = 2_000           # .pt 파일당 게이지 수
OUT_DIR         = "./dummy_dataset"
GDS_NAME        = "DUMMY"

# 반도체 레이아웃에 현실적인 박스 크기 분포 (nm)
# Metal line 패턴: 가로 좁고 세로 긴 형태 or 반대
BOX_WIDTH_RANGE  = (60, 200)      # nm
BOX_HEIGHT_RANGE = (60, 1200)     # nm (라인은 길게)
GRID_PITCH_NM    = 160            # 라인 간격 기준 (pitch)

SEED = 42
# ────────────────────────────────────────────────────────────────────────────


def _gen_nodes_grid(rng: np.random.Generator, n_nodes: int) -> np.ndarray:
    """
    반도체 레이아웃의 격자형 Metal Line 패턴을 모사.
    수직/수평 라인이 섞인 현실적 배치를 생성합니다.
    반환: (n_nodes, 4) float32 — [cx, cy, w, h] in nm
    """
    orientation = rng.choice(['vertical', 'horizontal', 'mixed'])
    nodes = []

    if orientation == 'vertical':
        # 수직 라인 (좁고 긴) — Metal 전형적 패턴
        pitch = rng.uniform(GRID_PITCH_NM * 0.8, GRID_PITCH_NM * 1.3)
        for i in range(n_nodes):
            cx = (i - n_nodes // 2) * pitch + rng.uniform(-10, 10)
            cy = rng.uniform(-HALF_WIN_NM * 0.7, HALF_WIN_NM * 0.7)
            w  = rng.uniform(*BOX_WIDTH_RANGE) * 0.5   # 좁은 폭
            h  = rng.uniform(200, min(BOX_HEIGHT_RANGE[1], HALF_WIN_NM * 1.5))
            nodes.append([cx, cy, w, h])

    elif orientation == 'horizontal':
        pitch = rng.uniform(GRID_PITCH_NM * 0.8, GRID_PITCH_NM * 1.3)
        for i in range(n_nodes):
            cx = rng.uniform(-HALF_WIN_NM * 0.7, HALF_WIN_NM * 0.7)
            cy = (i - n_nodes // 2) * pitch + rng.uniform(-10, 10)
            w  = rng.uniform(200, min(BOX_HEIGHT_RANGE[1], HALF_WIN_NM * 1.5))
            h  = rng.uniform(*BOX_WIDTH_RANGE) * 0.5
            nodes.append([cx, cy, w, h])

    else:  # mixed — 복잡한 레이아웃
        for _ in range(n_nodes):
            cx = rng.uniform(-HALF_WIN_NM * 0.85, HALF_WIN_NM * 0.85)
            cy = rng.uniform(-HALF_WIN_NM * 0.85, HALF_WIN_NM * 0.85)
            if rng.random() > 0.5:
                w, h = rng.uniform(*BOX_WIDTH_RANGE) * 0.5, rng.uniform(150, 600)
            else:
                w, h = rng.uniform(150, 600), rng.uniform(*BOX_WIDTH_RANGE) * 0.5
            nodes.append([cx, cy, w, h])

    # window 범위 클리핑
    arr = np.array(nodes, dtype=np.float32)
    arr[:, 0] = np.clip(arr[:, 0], -HALF_WIN_NM, HALF_WIN_NM)
    arr[:, 1] = np.clip(arr[:, 1], -HALF_WIN_NM, HALF_WIN_NM)
    arr[:, 2] = np.clip(arr[:, 2], 20, HALF_WIN_NM * 2)
    arr[:, 3] = np.clip(arr[:, 3], 20, HALF_WIN_NM * 2)
    return arr


def _box_dist(n1: np.ndarray, n2: np.ndarray) -> float:
    """두 박스 [cx,cy,w,h] 간의 gap 거리 (nm)."""
    l1, r1 = n1[0] - n1[2] / 2, n1[0] + n1[2] / 2
    b1, t1 = n1[1] - n1[3] / 2, n1[1] + n1[3] / 2
    l2, r2 = n2[0] - n2[2] / 2, n2[0] + n2[2] / 2
    b2, t2 = n2[1] - n2[3] / 2, n2[1] + n2[3] / 2

    dx = max(0.0, max(l1, l2) - min(r1, r2))
    dy = max(0.0, max(b1, b2) - min(t1, t2))
    return float(np.hypot(dx, dy))


def _generate_edges(nodes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    실제 전처리기와 동일한 4방향(E/N/W/S) 최근접 이웃 edge 생성.
    반환: raw_edge_index (M,2), raw_edge_attr (M,4)
    """
    centers = nodes[:, :2]
    n = len(centers)
    edges = set()

    for i in range(n):
        diff = centers - centers[i]          # (n, 2)
        dx, dy = diff[:, 0], diff[:, 1]

        masks = {
            'E': (dx > 0) & (dy >= -dx) & (dy < dx),
            'N': (dy > 0) & (dx > -dy) & (dx <= dy),
            'W': (dx < 0) & (dy > dx)  & (dy <= -dx),
            'S': (dy < 0) & (dx >= dy) & (dx < -dy),
        }

        for mask in masks.values():
            mask[i] = False
            candidates = np.where(mask)[0]
            if len(candidates) == 0:
                continue

            # 1차: center 거리 필터
            c_dists = np.hypot(dx[candidates], dy[candidates])
            cand_near = candidates[c_dists <= MAX_DIST_NM + 500.0]
            if len(cand_near) == 0:
                continue

            # 2차: box gap 거리 필터 → 최소 gap 선택
            box_dists = np.array([_box_dist(nodes[i], nodes[j]) for j in cand_near])
            best = cand_near[np.argmin(box_dists)]
            if box_dists[np.argmin(box_dists)] <= MAX_DIST_NM:
                edges.add(tuple(sorted((i, int(best)))))

    if not edges:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, 4), dtype=np.float32)

    raw_ei = np.array(list(edges), dtype=np.int64)
    u, v = raw_ei[:, 0], raw_ei[:, 1]

    box_d  = np.array([_box_dist(nodes[i], nodes[j]) for i, j in zip(u, v)], dtype=np.float32)
    diff_v = centers[v] - centers[u]
    c_dist = np.linalg.norm(diff_v, axis=1).astype(np.float32)
    norms  = np.where(c_dist == 0, 1e-6, c_dist)
    unit_v = diff_v / norms[:, None]

    raw_attr = np.column_stack([box_d, c_dist, unit_v[:, 0], unit_v[:, 1]]).astype(np.float32)
    return raw_ei, raw_attr


def _make_bidirectional(raw_ei: np.ndarray, raw_attr: np.ndarray):
    if raw_ei.size == 0:
        return raw_ei, raw_attr
    u, v = raw_ei[:, 0], raw_ei[:, 1]
    rev_attr = raw_attr.copy()
    rev_attr[:, 2:4] *= -1
    bi_ei   = np.vstack([np.column_stack([u, v]), np.column_stack([v, u])])
    bi_attr = np.vstack([raw_attr, rev_attr])
    return bi_ei.astype(np.int64), bi_attr.astype(np.float32)


def _hop_masks(nodes: np.ndarray, raw_ei: np.ndarray):
    n = len(nodes)
    distances = np.sqrt(nodes[:, 0]**2 + nodes[:, 1]**2)
    center_idx = int(np.argmin(distances))

    adj = [[] for _ in range(n)]
    for s, d in raw_ei:
        adj[s].append(d)
        adj[d].append(s)

    hop0 = {center_idx}
    hop1 = set(adj[center_idx]) - hop0
    hop2 = set()
    for nb in hop1:
        hop2.update(adj[nb])
    hop2 -= (hop0 | hop1)

    h0 = np.zeros(n, dtype=bool); h0[list(hop0)] = True
    h1 = np.zeros(n, dtype=bool); h1[list(hop1)] = True
    h2 = np.zeros(n, dtype=bool); h2[list(hop2)] = True
    h3 = np.ones(n, dtype=bool);  h3[list(hop0 | hop1 | hop2)] = False
    return h0, h1, h2, h3


def _flush(chunk: dict, out_dir: str, chunk_idx: int):
    path = os.path.join(out_dir, f"PREPROCESSED_{GDS_NAME}_part{chunk_idx}.pt")
    torch.save(chunk, path)
    print(f"  [Saved] {path}  ({len(chunk)} gauges)")
    del chunk
    gc.collect()
    return {}


def generate_dummy_gauge_list(n: int, rng: np.random.Generator) -> List[List]:
    """[GDS_NAME, cx_um, cy_um] 형태의 게이지 목록 생성."""
    gauges = []
    for i in range(n):
        name = f"GAUGE_{i:06d}"
        cx = round(rng.uniform(-500.0, 500.0), 4)   # um 단위 (임의 좌표)
        cy = round(rng.uniform(-500.0, 500.0), 4)
        gauges.append([name, cx, cy])
    return gauges


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    gauges = generate_dummy_gauge_list(NUM_GAUGES, rng)
    print(f"[Start] {NUM_GAUGES} dummy gauges → {OUT_DIR}")

    chunk, chunk_idx = {}, 0
    skip = 0
    t0 = time.time()

    for idx, (gname, cx, cy) in enumerate(gauges):
        n_nodes = int(rng.integers(*NODES_PER_GAUGE))
        nodes   = _gen_nodes_grid(rng, n_nodes)

        raw_ei, raw_attr       = _generate_edges(nodes)
        bi_ei,  bi_attr        = _make_bidirectional(raw_ei, raw_attr)
        h0, h1, h2, h3         = _hop_masks(nodes, raw_ei)

        if n_nodes == 0:
            skip += 1
            continue

        chunk[gname] = {
            'coord':      [cx, cy],
            'x':          torch.from_numpy(nodes),
            'edge_index': torch.from_numpy(bi_ei),
            'edge_attr':  torch.from_numpy(bi_attr),
            'hop0_mask':  torch.from_numpy(h0),
            'hop1_mask':  torch.from_numpy(h1),
            'hop2_mask':  torch.from_numpy(h2),
            'hop3_mask':  torch.from_numpy(h3),
        }

        if (idx + 1) % CHUNK_SIZE == 0 or (idx + 1) == NUM_GAUGES:
            elapsed = time.time() - t0
            print(f"  Progress: {(idx+1)/NUM_GAUGES:.1%}  [{idx+1}/{NUM_GAUGES}]  {elapsed:.1f}s")
            chunk = _flush(chunk, OUT_DIR, chunk_idx)
            chunk_idx += 1
            t0 = time.time()

    print(f"\n[Done] {NUM_GAUGES - skip}/{NUM_GAUGES} gauges saved ({skip} skipped)")
    print(f"       Parts: {chunk_idx}  →  {OUT_DIR}/")


if __name__ == "__main__":
    main()
