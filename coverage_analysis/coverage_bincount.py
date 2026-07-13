# -*- coding: utf-8 -*-
"""
coverage_bincount.py
====================
[알고리즘 A] BIN COUNT 기반 공간 커버리지 (Grid-based Occupancy)

15차원 공간을 거친 격자로 이산화한 뒤, 데이터가 실제 밟은 고유 격자 집합을 비교한다.
    Coverage = |Set_sam ∩ Set_ref| / |Set_ref|

차원의 저주 대응
    * 축당 Bin 을 3~5개로 작게 유지 → 이산화를 거칠게.
    * N^15 전체 격자 배열을 만들지 않고, '실제 점유된 격자' 만 set 으로 관리(Sparse).
      → 메모리는 실제 점유 격자 수에만 비례.

Reference(50GB)는 chunk 스트리밍으로 읽으며 Set_ref 를 누적한다.
Parquet 이면 row group 을 프로세스에 분배해 멀티코어로 병렬 집계한다(실측 2~4x).

실행:  python coverage_bincount.py
"""

import os

import numpy as np
import pandas as pd

from common import (
    DummyDataGenerator, FEATURE_COLS, N_WORKERS,
    fit_scaler, parallel_reduce_reference, set_union,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
# =============================================================================
SAMPLE_PATH  = "dummy_sample.csv"
REF_PATH     = "dummy_reference.csv"
FMT          = "csv"        # 'csv' 또는 'parquet'
SCALE_METHOD = "minmax"     # 'minmax' 또는 'standard'
N_BINS       = 4            # 축당 Bin 수 (3~5 권장)
CHUNKSIZE    = 200_000      # Reference 한 번에 읽을 행 수
N_WORKERS_A  = N_WORKERS    # chunk 병렬 처리 스레드 수

GEN_DUMMY    = True         # True 면 더미 데이터를 새로 생성
REF_ROWS     = 2_000_000    # 더미 Reference 총 행 수


# =============================================================================
# 격자화 유틸
# =============================================================================
# 모든 축이 동일한 내부 경계를 공유하므로, 15축을 하나의 배열 호출로 이산화한다.
_INNER_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]   # 길이 N_BINS-1


def cells_of(block):
    """
    스케일링된 (n_rows, 15) 블록 → 그 블록이 점유한 '고유 격자' 집합.

    - np.digitize 는 배열 전체에 적용되어 각 값을 0..N_BINS-1 로 이산화 (범위 밖 값도 안전).
    - 격자 좌표(15개 int16)를 .tobytes() 로 hashable key 화 → set 으로 중복 제거.
      (교집합 계산에는 동일성만 필요하므로 좌표를 tuple 로 복원할 필요 없음)
    """
    if block.shape[0] == 0:
        return set()
    idx = np.digitize(block, _INNER_EDGES).astype(np.int16)   # (n_rows, 15)
    uniq = np.unique(idx, axis=0)                             # chunk 내부 중복 선제거
    return {row.tobytes() for row in uniq}


def reference_cells(ref_path, scaler):
    """Reference 를 map(cells_of)→reduce(union) 로 집계해 Set_ref 를 구한다."""
    set_ref = parallel_reduce_reference(
        ref_path, scaler, cells_of, set_union, set,
        n_workers=N_WORKERS_A, chunksize=CHUNKSIZE, fmt=FMT,
    )
    print(f"[BinCount] Reference 처리 완료, |Set_ref| = {len(set_ref):,}")
    return set_ref


# =============================================================================
# 메인 파이프라인
# =============================================================================
def run():
    # (1) Sample 로드 → (2) Sample 기준 스케일러 학습 → 변환
    sample_df = pd.read_csv(SAMPLE_PATH)
    scaler = fit_scaler(sample_df, SCALE_METHOD)
    set_sam = cells_of(scaler.transform(sample_df[FEATURE_COLS].values))
    print(f"[BinCount] 축당 Bin={N_BINS} (이론상 {N_BINS}^15={N_BINS**15:,} 격자), "
          f"|Set_sam|={len(set_sam):,}")

    # (3) Reference 고유 격자 누적
    set_ref = reference_cells(REF_PATH, scaler)
    if not set_ref:
        raise ValueError("Set_ref 가 비어있다. Reference 데이터를 확인하라.")

    # (4) 커버리지
    inter = len(set_sam & set_ref)
    coverage = inter / len(set_ref)

    print("\n" + "=" * 56)
    print("  [알고리즘 A] BIN COUNT 공간 커버리지")
    print("=" * 56)
    print(f"  |Set_ref|            : {len(set_ref):,}")
    print(f"  |Set_sam|            : {len(set_sam):,}")
    print(f"  |Set_sam ∩ Set_ref|  : {inter:,}")
    print(f"  >>> Coverage         : {coverage:.4f} ({coverage*100:.2f}%)")
    print("=" * 56)
    return coverage


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
