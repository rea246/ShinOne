#!/usr/bin/env python3
"""B.csv 의 각 행에 input.pkl 의 최근접 게이지를 찾아 붙여 C.csv 를 만든다.

- input.pkl: [[gauge_name, cx, cy], ...] 를 pickle.dump 한 파일
  (preprocessor.py 의 load_gauge_list 와 같은 포맷).
- B.csv: X, Y 컬럼을 가진 CSV. 좌표를 scale 로 나눠 pkl 좌표계로 맞춘다.
- C.csv: B.csv 컬럼 뒤에 gauge_name, gauge_x, gauge_y, dist 를 붙인 파일.

제외되는 행:
- 최근접 게이지가 max_dist 이상 떨어진 행
- 매칭된 gauge_name 이 앞선 행에서 이미 사용된 행

설정은 main() 상단의 변수로 지정한다.
"""

import csv
import pickle
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def load_gauges(pkl_path):
    """pkl 을 이름 리스트와 (N,2) 좌표 배열로 분리."""
    with open(pkl_path, "rb") as f:
        gauge = pickle.load(f)
    if not gauge:
        sys.exit(f"오류: 게이지 리스트가 비어 있습니다: {pkl_path}")
    try:
        names = [g[0] for g in gauge]
        coords = np.array([[float(g[1]), float(g[2])] for g in gauge], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as e:
        sys.exit(f"오류: {pkl_path} 이 [[gauge_name, cx, cy], ...] 형태가 아닙니다 ({e})")
    return names, coords


def nearest_indices(gauge_coords, query_coords):
    """query 각 점의 최근접 게이지 인덱스와 거리. scipy 없으면 브루트포스."""
    if cKDTree is not None:
        tree = cKDTree(gauge_coords)
        dists, idxs = tree.query(query_coords, k=1)
        return np.atleast_1d(idxs), np.atleast_1d(dists)

    idxs = np.empty(len(query_coords), dtype=np.int64)
    dists = np.empty(len(query_coords), dtype=np.float64)
    for i, q in enumerate(query_coords):
        d2 = ((gauge_coords - q) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        idxs[i] = j
        dists[i] = float(np.sqrt(d2[j]))
    return idxs, dists


def load_b_csv(b_csv, x_column, y_column, scale):
    with open(b_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"오류: 입력 파일이 비어 있습니다: {b_csv}")
        for col in (x_column, y_column):
            if col not in reader.fieldnames:
                sys.exit(f"오류: '{col}' 컬럼이 없습니다. 존재하는 컬럼: {', '.join(reader.fieldnames)}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        sys.exit(f"오류: {b_csv} 에 데이터 행이 없습니다.")

    query = np.array(
        [[float(r[x_column]) / scale, float(r[y_column]) / scale] for r in rows],
        dtype=np.float64,
    )
    return fieldnames, rows, query


def print_range_check(gauge_coords, query, scale):
    """단위/스케일 실수를 잡아낼 수 있게 양쪽 좌표 범위를 출력하고 검사."""
    g_min, g_max = gauge_coords.min(axis=0), gauge_coords.max(axis=0)
    q_min, q_max = query.min(axis=0), query.max(axis=0)
    print(f"input.pkl 좌표 범위:  X [{g_min[0]:g}, {g_max[0]:g}], Y [{g_min[1]:g}, {g_max[1]:g}]")
    print(f"B.csv/{scale:g} 좌표 범위: X [{q_min[0]:g}, {q_max[0]:g}], Y [{q_min[1]:g}, {q_max[1]:g}]")

    overlap = all(q_min[a] <= g_max[a] and g_min[a] <= q_max[a] for a in (0, 1))
    g_span = float(max(g_max - g_min))
    q_span = float(max(q_max - q_min))
    ratio = (q_span / g_span) if g_span > 0 and q_span > 0 else 1.0
    if not overlap or ratio < 0.01 or ratio > 100:
        print("[경고] 두 좌표 범위가 겹치지 않거나 크기가 100배 이상 다릅니다. "
              "scale 값(단위 변환)을 확인하세요.", file=sys.stderr)


def inspect_cluster(rows, query, names, gauge_coords, cluster_id, x_column, y_column):
    """지정한 cluster_id 행의 최근접 게이지 top-5 를 브루트포스로 출력."""
    targets = [(i, r) for i, r in enumerate(rows) if r.get("cluster_id", "") == str(cluster_id)]
    if not targets:
        sys.exit(f"오류: cluster_id={cluster_id} 행을 찾지 못했습니다.")
    for i, r in targets:
        q = query[i]
        print(f"\n[inspect] cluster_id={cluster_id}  "
              f"원본 ({r[x_column]}, {r[y_column]}) → 변환 ({q[0]:g}, {q[1]:g})")
        d = np.hypot(gauge_coords[:, 0] - q[0], gauge_coords[:, 1] - q[1])
        for rank, j in enumerate(np.argsort(d)[:5], 1):
            print(f"  {rank}. {names[j]}  ({gauge_coords[j, 0]:g}, {gauge_coords[j, 1]:g})  "
                  f"dist={d[j]:.6g}")


def match(b_csv, input_pkl, output, scale, max_dist, x_column="X", y_column="Y",
          inspect=None):
    names, gauge_coords = load_gauges(input_pkl)
    print(f"게이지 {len(names):,}개 로드: {input_pkl}")

    fieldnames, rows, query = load_b_csv(b_csv, x_column, y_column, scale)
    print_range_check(gauge_coords, query, scale)

    if inspect is not None:
        inspect_cluster(rows, query, names, gauge_coords, inspect, x_column, y_column)
        return

    idxs, dists = nearest_indices(gauge_coords, query)

    extra = ["gauge_name", "gauge_x", "gauge_y", "dist"]
    for col in extra:
        if col in fieldnames:
            sys.exit(f"오류: B.csv 에 이미 '{col}' 컬럼이 있습니다.")

    used_names = set()
    skipped_far = skipped_dup = 0
    kept_dists = []

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra, lineterminator="\n")
        writer.writeheader()
        for row, j, d in zip(rows, idxs, dists):
            if d >= max_dist:
                skipped_far += 1
                continue
            name = names[j]
            if name in used_names:
                skipped_dup += 1
                continue
            used_names.add(name)
            row["gauge_name"] = name
            row["gauge_x"] = repr(float(gauge_coords[j, 0]))
            row["gauge_y"] = repr(float(gauge_coords[j, 1]))
            row["dist"] = f"{d:.6g}"
            writer.writerow(row)
            kept_dists.append(d)

    kept = len(kept_dists)
    print(f"{len(rows)}행 중 {kept}행 저장 "
          f"(거리 {max_dist} 이상 제외: {skipped_far}, 게이지 이름 중복 제외: {skipped_dup})")
    if kept:
        arr = np.array(kept_dists)
        print(f"저장된 행의 거리: 평균 {arr.mean():.4g}, 최대 {arr.max():.4g}")
    print(f"저장 완료: {output}")


def main():
    b_csv = "B.csv"        # extract_cluster_samples_fast.py 결과물
    input_pkl = "input.pkl"  # [[gauge_name, cx, cy], ...] pkl
    output = "C.csv"       # 출력 CSV
    scale = 20000.0        # B.csv 좌표를 pkl 좌표계로 맞추는 나눗셈 값
    max_dist = 0.1         # 이 거리(pkl 좌표계 단위) 이상 떨어진 행은 제외
    x_column = "X"
    y_column = "Y"
    inspect = None         # cluster_id 를 넣으면 그 행의 top-5 만 출력 (검증용)

    match(b_csv, input_pkl, output, scale, max_dist, x_column, y_column, inspect)


if __name__ == "__main__":
    main()
