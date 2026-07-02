import os
import gc
import time
import pickle
import random
import tempfile
import numpy as np
import torch
import klayout.db as db
from scipy.spatial import cKDTree
from multiprocessing import Pool
from typing import Tuple, List, Dict, Optional

# 멀티프로세싱 워커 전역 변수
_worker_ly = None
_worker_top_cell = None
_worker_layer_id = None


def load_layout_pure_db(gds_path: str) -> db.Layout:
    ly = db.Layout()
    ly.read(gds_path)
    return ly


def _make_search_dbox(cx: float, cy: float, half: float, dbu: float) -> db.DBox:
    return db.DBox(
        (cx - half) / dbu,
        (cy - half) / dbu,
        (cx + half) / dbu,
        (cy + half) / dbu
    )


def read_siemens_gg(file_path: str, precision: int) -> List[list]:
    gauge = []
    with open(file_path, "r") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith(("#", ";", "//")):
                continue
            parts = line.split()

            # [Fix BUG-4] 파일 포맷 변동 감지를 위한 Fail-Fast 가드
            if len(parts) < 8:
                print(f" [Format Warning] Line {line_idx} has insufficient tokens ({len(parts)}/8). Skipped.")
                continue

            nname, x1, y1, x2, y2 = parts[1], parts[4], parts[5], parts[6], parts[7]
            cx = round((int(x1) + int(x2)) / 2 / precision, 4)
            cy = round((int(y1) + int(y2)) / 2 / precision, 4)
            gauge.append([nname, cx, cy])
    return gauge


def load_gauge_list(file_path: str, precision: int) -> List[list]:
    """
    게이지 리스트를 읽는다.
    - .gg 확장자: Siemens 텍스트 포맷을 파싱(read_siemens_gg).
    - 그 외(.pkl 등): 외부 스크립트가 이미 [[gauge_name, cx, cy], ...] 형태로 다 만들어
      result.append() 로 쌓은 뒤 pickle.dump 로 그대로 저장해둔 파일 — 그 리스트를 그대로 로드.
      (여러 파일을 합치는 단계가 아니라, 이미 완성된 리스트 1개를 읽는 단계다.)
    """
    if file_path.split(".")[-1] == "gg":
        return read_siemens_gg(file_path, precision)
    with open(file_path, "rb") as f:
        return pickle.load(f)


def shuffle_gauge_list(gauge: List[list], seed: int = 42) -> List[list]:
    """
    이미 완성된 게이지 리스트를 셔플만 한다(다른 파일과 합치지 않음).
    run_large_scale_tiled_vectorization 는 순서대로 chunk_size 개씩 끊어 저장하고,
    train_ver2.py 는 그 청크(파일) 단위로 train/val 을 나눈다. 원본 리스트가 append 된
    순서 그대로(예: 위치/영역별로 뭉쳐서 쌓였다면) 청크를 나누면 특정 청크가 특정
    영역/패턴에 편향돼 val 청크가 전체 분포를 대표하지 못할 수 있다 — 그래서 그래프
    작업(run_large_scale_tiled_vectorization) 에 넘기기 전에 리스트 전체를 한 번 섞는다.
    seed 고정으로 재현 가능하게 섞는다. 원본 list 는 훼손하지 않고 셔플된 복사본을 반환한다.
    """
    shuffled = list(gauge)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _extract_poly_features(
    cell: db.Cell, layer_id: int, cx: float, cy: float, window_size: float, dbu: float, gname: str
) -> Optional[Tuple[np.ndarray, List[db.Box], np.ndarray]]:

    half_win = window_size / 2
    padding = 0.018
    search_dbox = _make_search_dbox(cx, cy, half_win + padding, dbu)
    search_box = db.Box.from_dbox(search_dbox)
    search_region = db.Region(cell.begin_shapes_rec_touching(layer_id, search_box))
    box_region = db.Region([search_box])
    poly_region = search_region & box_region

    # T2T 또는 Bridge 예외 패턴 처리
    if 't2t' in gname.lower() or 'bridge' in gname.lower():
        nodes = [[0.0, 0.0, 0.0, 0.0]]
        ccx_dbu = round(cx / dbu)
        ccy_dbu = round(cy / dbu)  # [BUG FIX] 기존에 cx로 잘못 들어가던 부분 수정
        bboxes = [db.Box(ccx_dbu - 1000, ccy_dbu - 1000, ccx_dbu + 1000, ccy_dbu + 1000)]
        centers = [[0.0, 0.0]]
    else:
        nodes, bboxes, centers = [], [], []

    tg_window_nm = half_win * 1000

    for poly in poly_region.each():
        bbox = poly.bbox()
        bbox_w = round(bbox.width() * dbu * 1000, 4)
        bbox_h = round(bbox.height() * dbu * 1000, 4)
        bbox_cx = round((bbox.center().x * dbu - cx) * 1000, 4)
        bbox_cy = round((bbox.center().y * dbu - cy) * 1000, 4)

        if (-tg_window_nm <= bbox_cx <= tg_window_nm) and (-tg_window_nm <= bbox_cy <= tg_window_nm):
            nodes.append([bbox_cx, bbox_cy, bbox_w, bbox_h])
            bboxes.append(bbox)
            centers.append([bbox_cx, bbox_cy])

    if not nodes:
        return None
    return np.array(nodes, dtype=np.float32), bboxes, np.array(centers, dtype=np.float32)


def _pack_fast_chunk(chunk_dataset: dict) -> Optional[dict]:
    """
    게이지별 dict → 사전 병합 fast 포맷(큰 텐서 5개 + meta + keys).
    train_ver2.FastPreloadedDataset 가 그대로 읽는 포맷이다.
    - edge_index 는 (E,2) 로 생성되므로 (2,E) 로 무조건 정규화 (로더의 모호한 shape 체크 제거).
    - edge_attr 가 비어있는(엣지 0) 게이지는 제외(로더 정책과 동일).
    """
    xs, eis, eas, hms, metas, keys = [], [], [], [], [], []
    x_off = e_off = 0
    for gname, val in chunk_dataset.items():
        ea = val['edge_attr']
        if ea.size(0) == 0:
            continue
        x  = val['x'].to(torch.float16)
        ea = ea.to(torch.float16)
        ei = val['edge_index'].t().contiguous().to(torch.int32)   # (E,2) -> (2,E)
        h0 = val['hop0_mask'].bool(); h1 = val['hop1_mask'].bool()
        h2 = val['hop2_mask'].bool(); h3 = val['hop3_mask'].bool()
        packed = (h0.to(torch.int8) | (h1.to(torch.int8) << 1) |
                  (h2.to(torch.int8) << 2) | (h3.to(torch.int8) << 3))
        n, e = x.size(0), ea.size(0)
        xs.append(x); eis.append(ei); eas.append(ea); hms.append(packed)
        metas.append((x_off, x_off + n, e_off, e_off + e))
        keys.append(gname)
        x_off += n; e_off += e

    if not xs:
        return None

    return {
        'x':          torch.cat(xs, dim=0),
        'edge_index': torch.cat(eis, dim=1),   # (2, totalE) local 인덱스
        'edge_attr':  torch.cat(eas, dim=0),
        'hop_packed': torch.cat(hms, dim=0),
        'meta':       torch.tensor(metas, dtype=torch.int64),   # (G,4): x0,x1,e0,e1
        'keys':       keys,                                     # 게이지 이름 (downstream 추적용)
    }


def _flush_chunk_dataset(chunk_dataset: dict, out_dir: str, gdsname: str, chunk_idx: int) -> dict:
    output_path = os.path.join(out_dir, f"PREPROCESSED_{gdsname}_part{chunk_idx}.pt")
    packed = _pack_fast_chunk(chunk_dataset)
    if packed is None:
        print(f"[Flush] 유효 그래프 없음(엣지 0), 저장 건너뜀: {output_path}")
        del chunk_dataset
        gc.collect()
        return {}
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir)

    try:
        with os.fdopen(tmp_fd, 'wb') as f:
            torch.save(packed, f)
        os.replace(tmp_path, output_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    print(f"[Saved Fast Format (Atomic)]  {output_path}  (graphs={packed['meta'].size(0)})")
    del chunk_dataset, packed
    gc.collect()
    return {}


def _calculate_box_distance(b1: db.Box, b2: db.Box) -> float:
    if b1.right < b2.left: dx = b2.left - b1.right
    elif b2.right < b1.left: dx = b1.left - b2.right
    else: dx = 0.0

    if b1.top < b2.bottom: dy = b2.bottom - b1.top
    elif b2.top < b1.bottom: dy = b1.bottom - b2.top
    else: dy = 0.0
    return np.hypot(dx, dy)


def _generate_edge_index(centers: np.ndarray, bboxes: list, max_dist: float, dbu: float) -> np.ndarray:
    if centers is None or len(centers) < 2:
        return np.empty((0, 2), dtype=int)

    num_nodes = len(centers)
    edges = set()
    tree = cKDTree(centers)

    # Coarse Filtering 마진
    max_pattern_margin = 500.0
    neighbors_list = tree.query_ball_point(centers, r=max_dist + max_pattern_margin)

    for i in range(num_nodes):
        curr_neighbors = np.array(neighbors_list[i])
        curr_neighbors = curr_neighbors[curr_neighbors != i]
        if len(curr_neighbors) == 0:
            continue

        diff_x = centers[curr_neighbors, 0] - centers[i, 0]
        diff_y = centers[curr_neighbors, 1] - centers[i, 1]
        center_dists = np.hypot(diff_x, diff_y)
        valid_candidate_mask = center_dists <= (max_dist + max_pattern_margin)
        if not np.any(valid_candidate_mask):
            continue

        curr_neighbors = curr_neighbors[valid_candidate_mask]
        diff_x = diff_x[valid_candidate_mask]
        diff_y = diff_y[valid_candidate_mask]

        # 4방향(동북서남) 마스킹 처리
        mask_E = (diff_x > 0) & (diff_y >= -diff_x) & (diff_y < diff_x)
        mask_N = (diff_y > 0) & (diff_x > -diff_y) & (diff_x <= diff_y)
        mask_W = (diff_x < 0) & (diff_y > diff_x) & (diff_y <= -diff_x)
        mask_S = (diff_y < 0) & (diff_x >= diff_y) & (diff_x < -diff_y)

        for mask in [mask_E, mask_N, mask_W, mask_S]:
            if np.any(mask):
                masked_indices = curr_neighbors[mask]
                box_dists = [
                    _calculate_box_distance(bboxes[i], bboxes[n_idx]) for n_idx in masked_indices
                ]
                box_dists = np.array(box_dists, dtype=np.float32)
                best_idx = np.argmin(box_dists)

                if box_dists[best_idx] <= max_dist / dbu:
                    edges.add(tuple(sorted((i, masked_indices[best_idx]))))

    if not edges:
        return np.empty((0, 2), dtype=int)
    return np.array(list(edges), dtype=int)


def _calculate_edge_attributes(edge_index: np.ndarray, bboxes: list, centers: np.ndarray, dbu: float) -> np.ndarray:
    if edge_index.size:
        u, v = edge_index.T
        box_dist = np.array([_calculate_box_distance(bboxes[i], bboxes[j]) for i, j in zip(u, v)], dtype=np.float32) * dbu * 1000
        diff_vec = centers[v] - centers[u]
        center_dist = np.linalg.norm(diff_vec, axis=1).astype(np.float32)
        norms = np.where(center_dist == 0, 1e-6, center_dist)
        unit_vec = diff_vec / norms[:, None]
        return np.column_stack((box_dist, center_dist, unit_vec[:, 0], unit_vec[:, 1]))
    return np.empty((0, 4), dtype=np.float32)


def _init_worker_process(gds_path: str, layer_num: int, layer_type: int):
    global _worker_ly, _worker_top_cell, _worker_layer_id
    _worker_ly = db.Layout()
    _worker_ly.read(gds_path)
    _worker_top_cell = _worker_ly.top_cell()
    _worker_layer_id = _worker_ly.layer(layer_num, layer_type)


def _worker_task(args) -> Tuple[str, Optional[dict]]:
    gname, cx, cy, window_size, max_dist = args
    try:
        dbu = _worker_ly.dbu
        poly_feature = _extract_poly_features(_worker_top_cell, _worker_layer_id, cx, cy, window_size, dbu, gname)
        if poly_feature is None:
            return gname, None

        node_matrix, bboxes, centers = poly_feature
        num_nodes = len(node_matrix)

        # [BUG FIX] 기존에 외부 max_dist 인자를 무시하고 250으로 하드코딩되던 부분 제거 혹은 유지 유연화
        # 필요시 이 부분을 완전히 주석처리하거나 유지하십시오. 여기서는 인자를 따르도록 주석 처리합니다.
        # max_dist = 250

        raw_edges = _generate_edge_index(centers, bboxes, max_dist, dbu)
        raw_attrs = _calculate_edge_attributes(raw_edges, bboxes, centers, dbu)

        if raw_edges.size > 0:
            u, v = raw_edges.T
            bi_edge_index = np.vstack([np.column_stack([u, v]), np.column_stack([v, u])])
            raw_attrs_reversed = raw_attrs.copy()
            raw_attrs_reversed[:, 2:4] = -raw_attrs_reversed[:, 2:4]
            bi_edge_attr = np.vstack([raw_attrs, raw_attrs_reversed])
        else:
            bi_edge_index = np.empty((0, 2), dtype=int)
            bi_edge_attr = np.empty((0, 4), dtype=np.float32)

        distances_to_center = np.sqrt(node_matrix[:, 0]**2 + node_matrix[:, 1]**2)
        center_node_idx = int(np.argmin(distances_to_center))

        adj = [[] for _ in range(num_nodes)]
        for src, dst in raw_edges:
            adj[src].append(dst)
            adj[dst].append(src)

        # Hop 기반 이웃 마스크 생성
        hop0 = {center_node_idx}
        hop1 = set()
        for n in hop0:
            hop1.update(adj[n])
        hop1 -= hop0

        hop2 = set()
        for n in hop1:
            hop2.update(adj[n])
        hop2 -= (hop0 | hop1)

        hop0_mask = np.zeros(num_nodes, dtype=bool)
        hop1_mask = np.zeros(num_nodes, dtype=bool)
        hop2_mask = np.zeros(num_nodes, dtype=bool)
        hop3plus_mask = np.ones(num_nodes, dtype=bool)

        hop0_mask[list(hop0)] = True
        hop1_mask[list(hop1)] = True
        hop2_mask[list(hop2)] = True
        hop3plus_mask[list(hop0 | hop1 | hop2)] = False

        return gname, {
            'coord': [cx, cy],
            'x': node_matrix,
            'edge_index': bi_edge_index,
            'edge_attr': bi_edge_attr,
            'hop0_mask': hop0_mask,
            'hop1_mask': hop1_mask,
            'hop2_mask': hop2_mask,
            'hop3_mask': hop3plus_mask
        }

    except Exception as e:
        print(f"[Worker Error] Failed to process gauge '{gname}' at ({cx}, {cy}): {e}")
        return gname, None


def _detect_allocated_cpus() -> int:
    """
    이 프로세스가 실제로 쓸 수 있는 CPU 개수를 판단한다.
    os.cpu_count() 는 호스트의 전체 CPU 개수를 그대로 반환할 뿐, LSF 가 bsub -n 으로
    할당한 슬롯 수는 반영하지 않는다(그 노드에서 cgroup/taskset affinity 를 강제하지
    않는 구성이면 -n 8 로 던져도 24 가 찍힌다). 우선순위:
      1) LSB_DJOB_NUMPROC — LSF 가 bsub -n 값 그대로 넣어주는 환경변수(가장 정확).
      2) os.sched_getaffinity(0) — cgroup/taskset 으로 실제 pin 된 CPU 집합(Linux 전용).
      3) os.cpu_count() — 위 둘 다 없을 때의 최후 폴백(호스트 전체 CPU, 과대추정 위험).
    """
    lsf_slots = os.environ.get("LSB_DJOB_NUMPROC")
    if lsf_slots and lsf_slots.isdigit():
        return int(lsf_slots)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 4


def run_large_scale_tiled_vectorization(
    ggds: str, gauge: list, window_list: list, out_dir: str, gdsname: str,
    layer_num: int, layer_type: int, chunk_size: int, max_dist: float = 300.0
):
    win_loc = window_list[0]
    total_count = len(gauge)
    chunk_idx = 0
    chunk_dataset = {}
    count, skip_count = 0, 0

    avail_cpus = _detect_allocated_cpus()
    num_workers = max(1, avail_cpus - 2)
    print(f"[Parallel] 가동 프로세스 카운트: {num_workers} Cores (allocated={avail_cpus})")

    task_arguments = [(gname, xx, yy, win_loc, max_dist) for gname, xx, yy in gauge]
    start_time = time.time()

    with Pool(processes=num_workers, initializer=_init_worker_process, initargs=(ggds, layer_num, layer_type)) as pool:
        try:
            for gname, result in pool.imap_unordered(_worker_task, task_arguments, chunksize=32):
                count += 1

                if result is None:
                    skip_count += 1
                else:
                    chunk_dataset[gname] = {
                        'coord': result['coord'],
                        'x': torch.from_numpy(result['x']),
                        'edge_index': torch.from_numpy(result['edge_index']),
                        'edge_attr': torch.from_numpy(result['edge_attr']),
                        'hop0_mask': torch.from_numpy(result['hop0_mask']),
                        'hop1_mask': torch.from_numpy(result['hop1_mask']),
                        'hop2_mask': torch.from_numpy(result['hop2_mask']),
                        'hop3_mask': torch.from_numpy(result['hop3_mask'])
                    }

                if count % chunk_size == 0 or count == total_count:
                    elapsed = time.time() - start_time
                    print(f"Progress: {count / total_count:.2%} [{count}/{total_count}] ({elapsed:.2f}s)")
                    start_time = time.time()

                    if chunk_dataset:
                        chunk_dataset = _flush_chunk_dataset(chunk_dataset, out_dir, gdsname, chunk_idx)
                        chunk_idx += 1
        except Exception as pool_err:
            print(f" [Fatal Pool Error] Loop interrupted due to worker lost or exception: {pool_err}")
            raise pool_err

    print(f"Complete: {total_count - skip_count}/{total_count} processed ({skip_count} skipped)")


def main() -> None:
    start_time = time.time()
    ggds = "/user/beol32pi/USERS/HSW/4.study/13.pythonkob/260520_IMG_to_String/0.material/T2T.oas"
    # ggfile: 외부 스크립트가 [gauge_name, cx, cy] 를 result.append() 로 이미 다 쌓아
    # pickle.dump 로 그대로 저장해둔 단일 .pkl(또는 .gg). 여러 파일을 합치는 게 아니라,
    # 이미 완성된 리스트 1개를 읽어서 셔플만 하면 된다.
    ggfile = "/user/beol32pi/USERS/HSW/4.study/13.pythonkob/260520_IMG_to_String/0.material/T2T.gg"
    shuffle_seed = 42
    layer_num = 93
    layer_type = 0

    gdsname = ggds.split("/")[-1]
    title = gdsname.split(".")[0]
    window = [0.512, 0.512]
    precision = 20000
    max_edge_distance_nm = 300.0

    out_dir = os.path.join("./", title)
    os.makedirs(out_dir, exist_ok=True)

    chunk_size = 300000
    # 1) 완성된 리스트 로드 -> 2) 그 리스트를 shuffle -> 3) shuffle 된 리스트로 그래프 작업
    gauge = load_gauge_list(ggfile, precision)
    gauge = shuffle_gauge_list(gauge, seed=shuffle_seed)
    print(f" 게이지 {len(gauge)}개 로드 → shuffle 완료 (seed={shuffle_seed})")

    print(f" {ggds} 멀티 프로세스 그래프 완본 데이터셋 빌더 작동 시작")
    run_large_scale_tiled_vectorization(
        ggds, gauge, window, out_dir, gdsname, layer_num, layer_type, chunk_size, max_dist=max_edge_distance_nm
    )
    print(f"Total Preprocessing Time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
