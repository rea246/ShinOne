# -*- coding: utf-8 -*-
"""
coverage_bincount.py
====================
[알고리즘 A] BIN COUNT 기반 공간 커버리지 (Grid-based Occupancy) — PCA 축소 버전

차원의 저주 해결
    N차원 전체 축을 그대로 격자화하면 고차원 Sparsity 로 Sample∩Reference 교집합이
    0 이 되어 Coverage 가 0% 로 수렴한다. 이를 막기 위해:

    * Sample 로 PCA 를 학습해, '누적 설명분산 VAR_THRESHOLD(기본 90%)' 를 넘는
      최소 주성분 수 K 만 남긴다 (common.fit_pca_var). 이 PCA 를 Reference 에도 공유.
    * 축소된 PC 공간에서 격자화한다. 각 PC 축은 스케일이 다르므로, Sample 이 변환된
      PC 좌표의 축별 [min,max] 로 '축마다 개별' Bin 경계를 만든다 (common.pc_bin_edges).

    Coverage = |Set_sam ∩ Set_ref| / |Set_ref|   (K차원 PC 격자 위에서)

부가 출력
    * 선택된 주성분 차원 수 K 로그.
    * 상위 2개 주성분(2D)으로도 커버리지를 계산하고, 2D PCA 격자 커버리지 그림을 저장.

Reference(50GB)는 chunk 스트리밍으로 읽으며, K차원·2차원 격자 집합을 '한 번의 패스'로
동시에 누적한다. Parquet 이면 row group 을 프로세스에 분배해 병렬 집계(구조 유지).

실행:  python coverage_bincount.py
"""

import functools
import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 환경용 백엔드
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from common import (
    DummyDataGenerator, N_WORKERS,
    fit_scaler, read_feature_matrix, fit_pca_var, pc_bin_edges,
    reservoir_sample_reference, parallel_reduce_reference,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
#   ※ '읽을 컬럼 범위'(FEATURE_COL_IDX)와 헤더 유무(HAS_HEADER)는 common.py 상단에서 설정한다.
#     (Sample/Reference/모든 스크립트가 동일 컬럼 선택을 공유해야 하므로 한 곳에 둔다)
# =============================================================================
SAMPLE_PATH   = "dummy_sample.csv"
REF_PATH      = "dummy_reference.csv"
FMT           = "csv"        # 'csv' 또는 'parquet'
SCALE_METHOD  = "standard"   # PCA 는 표준화(standard)와 궁합이 좋다
VAR_THRESHOLD = 0.90         # PCA 로 보존할 누적 설명분산 (90%)
N_BINS        = 4            # PC 축당 Bin 수 (3~5 권장)
CHUNKSIZE     = 200_000      # Reference 한 번에 읽을 행 수
N_WORKERS_A   = N_WORKERS    # Parquet row-group 병렬 프로세스 수

OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000     # 2D 그림용 Reference 표본 수

GEN_DUMMY     = True         # True 면 더미 데이터를 새로 생성
REF_ROWS      = 2_000_000    # 더미 Reference 총 행 수


# =============================================================================
# 격자화 유틸
# =============================================================================
def cells_of(block, pca, edges):
    """
    스케일링된 (n_rows, N_FEATURES) 블록 → PCA 축소 후 점유한 '고유 격자' 집합.

    1) pca.transform 으로 (n_rows, K) 축소 매트릭스(reduced_block) 생성.
    2) 각 PC 축마다 개별 경계 edges[j] 로 np.digitize → 격자 인덱스(범위 밖 값도 안전).
    3) 격자 좌표(K개 int16)를 .tobytes() 로 hashable key 화 → set 으로 중복 제거.
    """
    if block.shape[0] == 0:
        return set()
    reduced = pca.transform(block)                       # (n_rows, K)
    idx = np.empty(reduced.shape, dtype=np.int16)
    for j in range(reduced.shape[1]):                    # 축마다 개별 경계로 digitize
        idx[:, j] = np.digitize(reduced[:, j], edges[j])
    uniq = np.unique(idx, axis=0)                        # chunk 내부 중복 선제거
    return {row.tobytes() for row in uniq}


def cells_of_dual(block, pca_k, edges_k, pca2, edges2):
    """K차원 격자 집합과 2차원 격자 집합을 '한 블록에서 동시에' 계산 (스트리밍 1패스용)."""
    return (cells_of(block, pca_k, edges_k), cells_of(block, pca2, edges2))


# ---- (K차원, 2차원) 두 집합을 함께 누적하는 map-reduce 헬퍼 (pickle 가능해야 함) ----
def _pair_sets():
    return (set(), set())


def _union_pair(acc, part):
    acc[0].update(part[0])       # in-place 갱신 (튜플 컨테이너라도 안전)
    acc[1].update(part[1])
    return acc


# =============================================================================
# 2D PCA 격자 커버리지 시각화
# =============================================================================
def plot_2d_coverage(sample_pc2, ref_pc2, ref_set2, sam_set2, edges2, coverage2, out_path):
    """
    상위 2개 주성분 평면에서 N_BINS×N_BINS 격자 점유를 그린다.
        - 하늘색 칸 : Reference 가 밟은 칸 (커버리지 분모)
        - 주황 칸   : Sample 이 함께 밟은 칸 (분자, 교집합)
        - 점        : Sample 위치(참고용 오버레이)
    """
    inter = ref_set2 & sam_set2

    def decode(keys):                                    # tobytes → (i, j) 인덱스
        return [tuple(np.frombuffer(k, dtype=np.int16)) for k in keys]

    # 점유 행렬 M[i_x, j_y]: 0=빈칸, 1=Reference, 2=교집합
    M = np.zeros((N_BINS, N_BINS))
    for i, j in decode(ref_set2):
        if 0 <= i < N_BINS and 0 <= j < N_BINS:
            M[i, j] = 1
    for i, j in decode(inter):
        if 0 <= i < N_BINS and 0 <= j < N_BINS:
            M[i, j] = 2

    # 그림용 셀 경계 = [데이터 최소] + 내부경계(edges2) + [데이터 최대]
    allx = np.concatenate([sample_pc2[:, 0], ref_pc2[:, 0]])
    ally = np.concatenate([sample_pc2[:, 1], ref_pc2[:, 1]])
    xb = np.concatenate([[allx.min()], edges2[0], [allx.max()]])
    yb = np.concatenate([[ally.min()], edges2[1], [ally.max()]])

    cmap = ListedColormap(["#FFFFFF", "#C6D9F0", "#DD8452"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8.5, 7.5))
    plt.pcolormesh(xb, yb, M.T, cmap=cmap, norm=norm, edgecolors="#BBBBBB", linewidth=0.6)
    plt.scatter(sample_pc2[:, 0], sample_pc2[:, 1], s=8, c="#7A2E12", alpha=0.45, label="Sample")

    handles = [Patch(facecolor="#C6D9F0", edgecolor="#BBBBBB", label="Reference cell"),
               Patch(facecolor="#DD8452", edgecolor="#BBBBBB", label="Covered (Sample∩Ref)")]
    plt.legend(handles=handles, loc="upper right")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"2D PCA Grid Coverage = {coverage2*100:.2f}%  "
              f"(inter {len(inter)} / ref {len(ref_set2)} cells, {N_BINS}x{N_BINS})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[BinCount] 2D 커버리지 그림 저장: {out_path}")


# =============================================================================
# 메인 파이프라인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # (1) Sample 로드 → (2) Sample 기준 스케일러 학습 → 스케일링
    sample = read_feature_matrix(SAMPLE_PATH, fmt="csv")
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    # (3) Sample 기준 PCA: 누적 설명분산 90% 를 넘는 최소 주성분 수 K 를 동적 결정
    pca_k, K, cumv = fit_pca_var(sample_scaled, VAR_THRESHOLD)
    print(f"[BinCount] 전체 변이의 {VAR_THRESHOLD*100:.0f}%를 보존하기 위해 선택된 "
          f"최적의 주성분 차원 수: {K}개  (상위 {K}개 실제 보존 {cumv*100:.2f}%)")

    # (4) 2D 검증용 PCA(상위 2개 주성분) + 축별 Bin 경계 (Sample PC 좌표 min/max 기준)
    pca2 = PCA(n_components=2).fit(sample_scaled)
    sample_pc_k = pca_k.transform(sample_scaled)
    sample_pc2  = pca2.transform(sample_scaled)
    edges_k = pc_bin_edges(sample_pc_k, N_BINS)          # K개 축, 축마다 개별 경계
    edges2  = pc_bin_edges(sample_pc2,  N_BINS)          # 2개 축

    # (5) Sample 격자 집합 (K차원 / 2차원)
    set_sam_k = cells_of(sample_scaled, pca_k, edges_k)
    set_sam_2 = cells_of(sample_scaled, pca2,  edges2)

    # (6) Reference: 한 번의 스트리밍 패스로 K차원·2차원 격자를 동시 집계 (병렬 유지)
    map_fn = functools.partial(cells_of_dual, pca_k=pca_k, edges_k=edges_k,
                               pca2=pca2, edges2=edges2)
    set_ref_k, set_ref_2 = parallel_reduce_reference(
        REF_PATH, scaler, map_fn, _union_pair, _pair_sets,
        n_workers=N_WORKERS_A, chunksize=CHUNKSIZE, fmt=FMT,
    )
    if not set_ref_k:
        raise ValueError("Set_ref 가 비어있다. Reference 데이터를 확인하라.")

    # (7) 커버리지 (K차원 / 2차원)
    inter_k = len(set_sam_k & set_ref_k)
    inter_2 = len(set_sam_2 & set_ref_2)
    cov_k = inter_k / len(set_ref_k)
    cov_2 = inter_2 / len(set_ref_2)

    # (8) 2D PCA 격자 커버리지 그림 (Reference 는 그림용으로만 downsample)
    ref_pts = reservoir_sample_reference(REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    plot_2d_coverage(sample_pc2, pca2.transform(ref_pts), set_ref_2, set_sam_2,
                     edges2, cov_2, os.path.join(OUTPUT_DIR, "bincount_2d_coverage.png"))

    print("\n" + "=" * 60)
    print("  [알고리즘 A] BIN COUNT 공간 커버리지 (PCA 축소)")
    print("=" * 60)
    print(f"  보존 분산 기준          : {VAR_THRESHOLD*100:.0f}%  →  선택 주성분 K = {K}")
    print(f"  [K={K}차원]  |Set_ref|={len(set_ref_k):,}  |Set_sam|={len(set_sam_k):,}  "
          f"∩={inter_k:,}")
    print(f"  >>> Coverage (K차원)    : {cov_k:.4f} ({cov_k*100:.2f}%)")
    print(f"  [2차원]   |Set_ref|={len(set_ref_2):,}  |Set_sam|={len(set_sam_2):,}  "
          f"∩={inter_2:,}")
    print(f"  >>> Coverage (2차원)    : {cov_2:.4f} ({cov_2*100:.2f}%)")
    print("=" * 60)
    return {"K": K, "coverage_k": cov_k, "coverage_2d": cov_2}


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
