# -*- coding: utf-8 -*-
"""
coverage_bincount.py
====================
[알고리즘 A] BIN COUNT 기반 공간 커버리지 (Grid-based Occupancy)

목적
----
15차원 공간을 '거친(coarse) 격자'로 이산화(digitize)한 뒤,
Reference 가 밟은 고유 격자 집합과 Sample 이 밟은 고유 격자 집합을 비교한다.

    Coverage = len(Set_sam ∩ Set_ref) / len(Set_ref)

차원의 저주 대응
----------------
15차원을 각 축 N 개 Bin 으로 나누면 격자 수가 N^15 개다.
  - N=10 이면 10^15 (1000조) -> 배열로 만들면 즉시 메모리 폭발
따라서 아래 두 가지 전략을 쓴다.
  1) 축당 Bin 을 아주 작게 (기본 4개) 유지해 이산화 자체를 거칠게 한다.
  2) 전체 격자 배열을 만들지 않고, "데이터가 실제로 밟은 격자 좌표(tuple)"만
     set() 에 넣어 관리한다. -> 메모리는 O(고유 격자 수) 로, 데이터가 실제 점유한
     영역에만 비례한다 (Sparse 표현).

Reference(50GB)는 chunksize 스트리밍으로 읽으며 Set_ref 를 '누적' 갱신한다.

실행
----
    python coverage_bincount.py
"""

import argparse
import os

import numpy as np

from common import (
    DummyDataGenerator,
    FEATURE_COLS,
    fit_scaler,
    iter_reference_chunks,
)
import pandas as pd


# =============================================================================
# 1. 격자 좌표 계산 유틸
# =============================================================================
def build_bin_edges(n_bins: int, n_features: int) -> np.ndarray:
    """
    스케일링된 데이터(대략 [0,1] 또는 표준화 범위)를 이산화할 Bin 경계를 만든다.

    Sample 기준 MinMax 스케일이라면 Sample 값은 [0,1] 범위지만,
    Reference 는 그 범위를 벗어날 수 있다(음수/1 초과).
    np.digitize 는 범위를 벗어난 값도 양끝 Bin(0 또는 n_bins-1)으로 안전하게 처리하므로,
    경계는 [0,1] 을 균등 분할한 내부 경계만 사용한다.

    Returns
    -------
    edges : shape (n_features, n_bins - 1) 의 내부 경계 배열.
    """
    # 0~1 을 n_bins 등분한 '내부 경계' (양끝 0,1 은 제외한 n_bins-1 개)
    inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]     # 길이 n_bins-1
    # 모든 축에 동일 경계를 적용 (축별로 다르게 하고 싶으면 여기 확장)
    edges = np.tile(inner, (n_features, 1))
    return edges


def digitize_block(block: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    (n_rows, n_features) 배열을 격자 인덱스 (n_rows, n_features) int8 로 이산화한다.

    각 축마다 np.digitize 를 적용. 결과 인덱스는 0 ~ n_bins-1.
    """
    n_rows, n_features = block.shape
    idx = np.empty((n_rows, n_features), dtype=np.int16)
    for j in range(n_features):
        # digitize: edges[j] 기준으로 각 값이 몇 번째 구간에 속하는지 -> 0..n_bins-1
        idx[:, j] = np.digitize(block[:, j], edges[j])
    return idx


def cells_to_set(idx: np.ndarray) -> set:
    """
    격자 인덱스 배열 (n_rows, n_features) 를 '고유 격자 좌표 tuple 의 set' 으로 변환.

    tuple(row) 형태로 만들어 hashable 하게 하고 set 으로 중복 제거.
    numpy.unique(axis=0) 로 먼저 중복을 줄여 tuple 변환 비용을 낮춘다.
    """
    if idx.shape[0] == 0:
        return set()
    uniq = np.unique(idx, axis=0)          # 먼저 chunk 내부 중복 제거
    return {tuple(row) for row in uniq}    # hashable tuple 집합


# =============================================================================
# 2. Sample 격자 집합 계산
# =============================================================================
def compute_sample_cells(sample_scaled: np.ndarray, edges: np.ndarray) -> set:
    """Sample(1800행)의 고유 격자 집합 Set_sam 을 반환."""
    idx = digitize_block(sample_scaled, edges)
    set_sam = cells_to_set(idx)
    print(f"[BinCount] Sample 고유 격자 수 |Set_sam| = {len(set_sam):,}")
    return set_sam


# =============================================================================
# 3. Reference 격자 집합 계산 (스트리밍 누적)
# =============================================================================
def compute_reference_cells(
    ref_path: str,
    scaler,
    edges: np.ndarray,
    chunksize: int = 200_000,
    fmt: str = None,
) -> set:
    """
    Reference 를 chunk 단위로 읽으며 Set_ref 를 누적한다.
    한 번에 전체를 올리지 않으므로 50GB 파일도 처리 가능.
    """
    set_ref = set()
    total_rows = 0
    for i, chunk in enumerate(iter_reference_chunks(ref_path, scaler, chunksize, fmt)):
        idx = digitize_block(chunk, edges)
        set_ref |= cells_to_set(idx)          # 집합 합집합으로 누적
        total_rows += chunk.shape[0]
        if (i + 1) % 5 == 0:
            print(f"[BinCount] Reference {total_rows:,} rows 처리, "
                  f"현재 |Set_ref| = {len(set_ref):,}")

    print(f"[BinCount] Reference 총 {total_rows:,} rows 처리 완료. "
          f"|Set_ref| = {len(set_ref):,}")
    return set_ref


# =============================================================================
# 4. 커버리지 계산
# =============================================================================
def bincount_coverage(set_sam: set, set_ref: set) -> dict:
    """
    Coverage = |Set_sam ∩ Set_ref| / |Set_ref|

    부가 지표
    ---------
    * intersection : 교집합 크기
    * sample_out_of_ref : Sample 이 밟았지만 Reference 엔 없는 격자 수
                          (Sample 분포가 Reference 밖으로 새는 정도)
    """
    if len(set_ref) == 0:
        raise ValueError("Set_ref 가 비어있다. Reference 데이터를 확인하라.")

    inter = set_sam & set_ref
    coverage = len(inter) / len(set_ref)

    result = {
        "coverage": coverage,
        "intersection": len(inter),
        "n_ref_cells": len(set_ref),
        "n_sam_cells": len(set_sam),
        "sample_out_of_ref": len(set_sam - set_ref),
    }
    return result


# =============================================================================
# 5. 메인 파이프라인
# =============================================================================
def run(
    sample_path: str,
    ref_path: str,
    n_bins: int = 4,
    scale_method: str = "minmax",
    chunksize: int = 200_000,
    fmt: str = None,
):
    # (1) Sample 로드
    sample_df = pd.read_csv(sample_path)

    # (2) Sample 기준 스케일러 학습
    scaler = fit_scaler(sample_df, method=scale_method)
    sample_scaled = scaler.transform(sample_df[FEATURE_COLS].values)

    # (3) 격자 경계 생성 (축당 n_bins 개 -> 이론상 n_bins^15 격자, 실제론 점유분만 관리)
    edges = build_bin_edges(n_bins, n_features=len(FEATURE_COLS))
    print(f"[BinCount] 축당 Bin 수 = {n_bins}, "
          f"이론적 최대 격자 수 = {n_bins}^15 = {n_bins ** 15:,}")

    # (4) Sample / Reference 각각의 고유 격자 집합
    set_sam = compute_sample_cells(sample_scaled, edges)
    set_ref = compute_reference_cells(ref_path, scaler, edges, chunksize, fmt)

    # (5) 커버리지 산출
    result = bincount_coverage(set_sam, set_ref)

    print("\n" + "=" * 60)
    print("  [알고리즘 A] BIN COUNT 기반 공간 커버리지 결과")
    print("=" * 60)
    print(f"  Reference 고유 격자 수 |Set_ref|   : {result['n_ref_cells']:,}")
    print(f"  Sample    고유 격자 수 |Set_sam|   : {result['n_sam_cells']:,}")
    print(f"  교집합 |Set_sam ∩ Set_ref|         : {result['intersection']:,}")
    print(f"  Sample 이 Reference 밖으로 샌 격자  : {result['sample_out_of_ref']:,}")
    print(f"  >>> Coverage                        : {result['coverage']:.4f} "
          f"({result['coverage'] * 100:.2f}%)")
    print("=" * 60)
    return result


# =============================================================================
# 6. 엔트리포인트 (더미 데이터 자동 생성 후 실행)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BIN COUNT 기반 공간 커버리지")
    parser.add_argument("--sample", default="dummy_sample.csv")
    parser.add_argument("--ref", default="dummy_reference.csv")
    parser.add_argument("--fmt", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--n_bins", type=int, default=4, help="축당 Bin 수 (3~5 권장)")
    parser.add_argument("--scale", default="minmax", choices=["minmax", "standard"])
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--ref_rows", type=int, default=2_000_000,
                        help="더미 Reference 총 행 수")
    parser.add_argument("--gen", action="store_true",
                        help="더미 데이터를 새로 생성한다")
    args = parser.parse_args()

    # 더미 데이터가 없거나 --gen 이면 생성
    if args.gen or not (os.path.exists(args.sample) and os.path.exists(args.ref)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample_csv(args.sample, n_rows=1800)
        gen.make_reference(args.ref, total_rows=args.ref_rows, fmt=args.fmt)

    run(
        sample_path=args.sample,
        ref_path=args.ref,
        n_bins=args.n_bins,
        scale_method=args.scale,
        chunksize=args.chunksize,
        fmt=args.fmt,
    )
