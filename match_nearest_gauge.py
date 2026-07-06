#!/usr/bin/env python3
"""B.csv 의 각 행에 input.pkl 에서 가장 가까운 게이지를 찾아 붙여 C.csv 를 만든다.

- input.pkl: preprocessor.py 의 load_gauge_list 가 읽는 것과 같은 포맷.
  [[gauge_name, cx, cy], ...] 리스트를 pickle.dump 로 저장한 파일.
- B.csv: extract_cluster_samples_fast.py 의 결과물. X, Y 컬럼을 가진다.
  B.csv 의 좌표는 input.pkl 좌표계와 단위가 달라서 --scale(기본 20000)로
  나눠서 비교한다. (변환: x = X / scale, y = Y / scale)
- C.csv: B.csv 의 모든 컬럼 뒤에 최근접 게이지의
  gauge_name, gauge_x, gauge_y, dist 컬럼을 붙인 파일.
  dist 는 input.pkl 좌표계 기준 유클리드 거리다.

최근접 탐색은 scipy 의 cKDTree 를 사용하고(수백만 게이지도 빠름),
scipy 가 없으면 numpy 브루트포스로 대신한다.

사용법:
    python3 match_nearest_gauge.py B.csv input.pkl -o C.csv
    python3 match_nearest_gauge.py B.csv input.pkl -o C.csv --scale 20000
"""

import argparse
import csv
import pickle
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None


def load_gauges(pkl_path):
    """[[gauge_name, cx, cy], ...] 형태의 pkl 을 이름 리스트와 (N,2) 좌표로 분리."""
    with open(pkl_path, "rb") as f:
        gauge = pickle.load(f)
    if not gauge:
        sys.exit(f"오류: 게이지 리스트가 비어 있습니다: {pkl_path}")
    try:
        names = [g[0] for g in gauge]
        coords = np.array([[float(g[1]), float(g[2])] for g in gauge], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as e:
        sys.exit(
            f"오류: {pkl_path} 이 [[gauge_name, cx, cy], ...] 형태가 아닙니다 ({e})"
        )
    return names, coords


def nearest_indices(gauge_coords, query_coords):
    """query 각 점에 대해 가장 가까운 게이지 인덱스와 거리를 반환."""
    if cKDTree is not None:
        tree = cKDTree(gauge_coords)
        dists, idxs = tree.query(query_coords, k=1)
        return np.atleast_1d(idxs), np.atleast_1d(dists)

    # scipy 폴백: 쿼리(클러스터 수)는 적으므로 행 단위 브루트포스로 충분
    idxs = np.empty(len(query_coords), dtype=np.int64)
    dists = np.empty(len(query_coords), dtype=np.float64)
    for i, q in enumerate(query_coords):
        d2 = ((gauge_coords - q) ** 2).sum(axis=1)
        j = int(np.argmin(d2))
        idxs[i] = j
        dists[i] = float(np.sqrt(d2[j]))
    return idxs, dists


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="B.csv 각 행에 input.pkl 의 최근접 게이지(name, x, y)를 붙여 C.csv 생성."
    )
    parser.add_argument("b_csv", help="B.csv 경로 (X, Y 컬럼 필요)")
    parser.add_argument("input_pkl", help="input.pkl 경로 ([[gauge_name, cx, cy], ...])")
    parser.add_argument(
        "-o", "--output", default="C.csv",
        help="출력 CSV 경로 (기본값: C.csv)",
    )
    parser.add_argument(
        "--scale", type=float, default=20000.0,
        help="B.csv 좌표를 input.pkl 좌표계로 맞추기 위한 나눗셈 값 (기본값: 20000)",
    )
    parser.add_argument("--x-column", default="X", help="B.csv 의 X 컬럼 이름 (기본값: X)")
    parser.add_argument("--y-column", default="Y", help="B.csv 의 Y 컬럼 이름 (기본값: Y)")
    args = parser.parse_args(argv)

    names, gauge_coords = load_gauges(args.input_pkl)
    print(f"게이지 {len(names):,}개 로드: {args.input_pkl}")

    with open(args.b_csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"오류: 입력 파일이 비어 있습니다: {args.b_csv}")
        for col in (args.x_column, args.y_column):
            if col not in reader.fieldnames:
                sys.exit(
                    f"오류: '{col}' 컬럼이 없습니다. "
                    f"존재하는 컬럼: {', '.join(reader.fieldnames)}"
                )
        fieldnames = list(reader.fieldnames)
        rows = list(reader)
    if not rows:
        sys.exit(f"오류: {args.b_csv} 에 데이터 행이 없습니다.")

    query = np.array(
        [[float(r[args.x_column]) / args.scale,
          float(r[args.y_column]) / args.scale] for r in rows],
        dtype=np.float64,
    )

    idxs, dists = nearest_indices(gauge_coords, query)

    extra = ["gauge_name", "gauge_x", "gauge_y", "dist"]
    for col in extra:
        if col in fieldnames:
            sys.exit(f"오류: B.csv 에 이미 '{col}' 컬럼이 있어 덮어쓸 수 없습니다.")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extra, lineterminator="\n")
        writer.writeheader()
        for row, j, d in zip(rows, idxs, dists):
            row["gauge_name"] = names[j]
            row["gauge_x"] = repr(float(gauge_coords[j, 0]))
            row["gauge_y"] = repr(float(gauge_coords[j, 1]))
            row["dist"] = f"{d:.6g}"
            writer.writerow(row)

    print(f"{len(rows)}행 매칭 완료 (평균 거리 {dists.mean():.4g}, 최대 {dists.max():.4g})")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
