#!/usr/bin/env python3
"""cluster_id 별로 대표 행을 하나씩 추출해 CSV로 저장하는 스크립트.

입력 CSV는 G1, G14, ..., cluster_id, X, Y, hashkey 같은 컬럼을 가지며,
cluster_id 컬럼만 있으면 나머지 컬럼 구성은 자유롭다.
표준 라이브러리만 사용하므로 별도 설치 없이 실행 가능하다.

사용법:
    python3 extract_cluster_samples.py input.csv -o output.csv
    python3 extract_cluster_samples.py input.csv -o output.csv --mode random --seed 42
    python3 extract_cluster_samples.py input.csv -o output.csv --key-column cluster_id
"""

import argparse
import csv
import random
import sys


def extract_one_per_cluster(rows, key_column, mode="first", seed=None):
    """행 목록에서 key_column 값별로 한 행씩 골라 반환한다.

    mode:
        first  - 각 클러스터에서 처음 등장한 행 (기본값)
        last   - 각 클러스터에서 마지막으로 등장한 행
        random - 각 클러스터에서 무작위로 한 행
    반환 순서는 클러스터가 처음 등장한 순서를 따른다.
    """
    grouped = {}  # cluster_id -> 선택 로직에 필요한 행(들)
    order = []    # 클러스터 등장 순서 유지

    for row in rows:
        key = row[key_column]
        if key not in grouped:
            order.append(key)
            grouped[key] = [row]
        elif mode == "first":
            pass  # 이미 첫 행을 갖고 있으므로 무시
        elif mode == "last":
            grouped[key] = [row]
        elif mode == "random":
            grouped[key].append(row)

    if mode == "random":
        rng = random.Random(seed)
        return [rng.choice(grouped[key]) for key in order]
    return [grouped[key][0] for key in order]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="CSV에서 cluster_id 별로 행을 하나씩 추출해 저장한다."
    )
    parser.add_argument("input", help="입력 CSV 경로")
    parser.add_argument(
        "-o", "--output", default="cluster_samples.csv",
        help="출력 CSV 경로 (기본값: cluster_samples.csv)",
    )
    parser.add_argument(
        "-k", "--key-column", default="cluster_id",
        help="그룹 기준 컬럼 이름 (기본값: cluster_id)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["first", "last", "random"], default="first",
        help="클러스터 내 행 선택 방식 (기본값: first)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="--mode random 일 때 재현 가능한 결과를 위한 시드",
    )
    args = parser.parse_args(argv)

    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            sys.exit(f"오류: 입력 파일이 비어 있습니다: {args.input}")
        if args.key_column not in reader.fieldnames:
            sys.exit(
                f"오류: '{args.key_column}' 컬럼이 없습니다. "
                f"존재하는 컬럼: {', '.join(reader.fieldnames)}"
            )
        fieldnames = reader.fieldnames
        rows = list(reader)

    selected = extract_one_per_cluster(rows, args.key_column, args.mode, args.seed)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print(f"전체 {len(rows)}행에서 클러스터 {len(selected)}개의 대표 행을 추출했습니다.")
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
