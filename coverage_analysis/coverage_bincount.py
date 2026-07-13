# -*- coding: utf-8 -*-
"""
coverage_bincount.py
====================
[알고리즘 A] BIN COUNT 기반 공간 커버리지 (Grid-based Occupancy) — PCA 축소 + 진단 시각화

수학적 무결성 (차원의 저주)
    N차원 전체 축을 그대로 격자화하면 고차원 Sparsity 로 Sample∩Reference 가 0 이 되어
    Coverage 가 0% 로 수렴한다. → Sample 로 PCA 를 학습해 '누적 설명분산 VAR_THRESHOLD(90%)'
    를 넘는 최소 주성분 수 K 만 남기고(그 PCA 를 Reference 에도 공유), 축소된 PC 공간에서
    격자화한다. PC 축마다 스케일이 다르므로 Sample PC 좌표의 축별 [min,max] 로 개별 Bin 경계.

한 번의 스트리밍 패스로 아래를 '동시에' 집계한다 (parallel_reduce_reference, 메모리 절약 유지):
    (1) K차원 PCA 격자          → 최종 Coverage
    (2) 상위 2개 PC 격자         → 전통적 2D PCA 커버리지  (bincount_2d_coverage.png)
    (3) 커스텀 2D 격자           → 원본 피처 1개(X) + 나머지 (N-1)차원 PCA(n=1)(Y)
                                   (custom_2d_coverage.png)  ← 축의 물리적 의미 보존
    (4) 축별 1D 점유 행렬        → 21개 피처 각각의 독립 커버리지 (per_axis_coverage.png)

Parquet 이면 row group 을 프로세스에 분배해 병렬 집계한다(구조 유지).
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
    DummyDataGenerator, N_FEATURES, N_WORKERS,
    fit_scaler, read_feature_matrix, fit_pca_var, pc_bin_edges,
    reservoir_sample_reference, parallel_reduce_reference,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
#   ※ '읽을 컬럼 범위'(FEATURE_COL_IDX)와 헤더 유무(HAS_HEADER)는 common.py 상단에서 설정한다.
#     (Sample/Reference/모든 스크립트가 동일 컬럼 선택을 공유해야 하므로 한 곳에 둔다)
# =============================================================================
SAMPLE_PATH    = "dummy_sample.csv"
REF_PATH       = "dummy_reference.csv"
FMT            = "csv"        # 'csv' 또는 'parquet'
SCALE_METHOD   = "standard"   # PCA 는 표준화(standard)와 궁합이 좋다
VAR_THRESHOLD  = 0.90         # PCA 로 보존할 누적 설명분산 (90%)
N_BINS         = 4            # PC/커스텀 축당 Bin 수 (고차원 격자용, 3~5 권장)
RAW_FEATURE_IDX = 0           # 커스텀 2D 에서 X축에 '원본 그대로' 살릴 핵심 피처 (선택 열 기준 0번)
PER_AXIS_BINS  = 20           # 축별 1D 독립 커버리지용 Bin 수 (1D 는 차원의 저주가 없어 더 촘촘히)
CHUNKSIZE      = 200_000      # Reference 한 번에 읽을 행 수
N_WORKERS_A    = N_WORKERS    # Parquet row-group 병렬 프로세스 수

OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000      # 2D 그림용 Reference 표본 수

GEN_DUMMY      = True         # True 면 더미 데이터를 새로 생성
REF_ROWS       = 2_000_000    # 더미 Reference 총 행 수


# =============================================================================
# 격자화 유틸
# =============================================================================
def cells_of(block, pca, edges):
    """
    스케일링된 (n_rows, N_FEATURES) 블록 → PCA 축소 후 점유한 '고유 격자' 집합.

    1) pca.transform 으로 (n_rows, K) 축소 매트릭스 생성.
    2) 각 PC 축마다 개별 경계 edges[j] 로 np.digitize (범위 밖 값도 안전).
    3) 격자 좌표(int16)를 .tobytes() 로 hashable key 화 → set 으로 중복 제거.
    """
    if block.shape[0] == 0:
        return set()
    reduced = pca.transform(block)
    idx = np.empty(reduced.shape, dtype=np.int16)
    for j in range(reduced.shape[1]):
        idx[:, j] = np.digitize(reduced[:, j], edges[j])
    uniq = np.unique(idx, axis=0)
    return {row.tobytes() for row in uniq}


def cells_of_custom(block, raw_dim, others, pca_rest, ex, ey):
    """
    커스텀 2D 격자: X = 원본 피처(raw_dim, scaled) 그대로,
                    Y = 나머지 (N-1)차원을 PCA(n=1)로 압축한 PC1.
    각 축을 개별 경계(ex, ey)로 digitize → (ix, iy) 좌표를 tobytes 해시 set 으로 반환.
    """
    if block.shape[0] == 0:
        return set()
    x = block[:, raw_dim]
    y = pca_rest.transform(block[:, others])[:, 0]
    ix = np.digitize(x, ex).astype(np.int16)
    iy = np.digitize(y, ey).astype(np.int16)
    uniq = np.unique(np.stack([ix, iy], axis=1), axis=0)
    return {row.tobytes() for row in uniq}


def axis_occupancy(block, raw_edges):
    """
    축별 1D 점유 행렬 (N_FEATURES, PER_AXIS_BINS) bool 반환.
    각 축을 '타 차원 무시하고' 독립적으로 PER_AXIS_BINS 개 Bin 으로 digitize 하여,
    데이터가 밟은 Bin 을 True 로 표시한다. (chunk 간에는 OR 로 누적)
    """
    occ = np.zeros((block.shape[1], PER_AXIS_BINS), dtype=bool)
    for a in range(block.shape[1]):
        occ[a, np.unique(np.digitize(block[:, a], raw_edges[a]))] = True
    return occ


# =============================================================================
# 스트리밍 1패스용 map / reduce / init  (모두 pickle 가능해야 프로세스 병렬 가능)
#   누적기 = [set_KD, set_2D, set_custom, occ_matrix]
# =============================================================================
def _project_all(block, proj):
    """한 chunk 에서 4종(K차원/2D/커스텀2D/축별점유) 부분결과를 동시에 계산."""
    return [
        cells_of(block, proj["pca_k"], proj["edges_k"]),
        cells_of(block, proj["pca2"], proj["edges2"]),
        cells_of_custom(block, proj["raw_dim"], proj["others"],
                        proj["pca_rest"], proj["ex_c"], proj["ey_c"]),
        axis_occupancy(block, proj["raw_edges"]),
    ]


def _init_all():
    return [set(), set(), set(), np.zeros((N_FEATURES, PER_AXIS_BINS), dtype=bool)]


def _reduce_all(acc, part):
    acc[0].update(part[0])
    acc[1].update(part[1])
    acc[2].update(part[2])
    acc[3] |= part[3]                 # 축별 점유는 bool OR 로 누적
    return acc


# =============================================================================
# 시각화 1: 2D 격자 커버리지 (PCA 2D / 커스텀 2D 공용)
# =============================================================================
def plot_2d_coverage(sample_xy, ref_xy, ref_set, sam_set, edges2, coverage,
                     out_path, xlabel, ylabel, title):
    """
    2D 평면에서 N_BINS×N_BINS 격자 점유를 그린다.
        하늘색 칸 = Reference 가 밟은 칸(분모), 주황 칸 = Sample 도 밟은 칸(분자=교집합),
        점 = Sample 위치 오버레이.
    """
    inter = ref_set & sam_set

    def decode(keys):
        return [tuple(np.frombuffer(k, dtype=np.int16)) for k in keys]

    M = np.zeros((N_BINS, N_BINS))
    for i, j in decode(ref_set):
        if 0 <= i < N_BINS and 0 <= j < N_BINS:
            M[i, j] = 1
    for i, j in decode(inter):
        if 0 <= i < N_BINS and 0 <= j < N_BINS:
            M[i, j] = 2

    allx = np.concatenate([sample_xy[:, 0], ref_xy[:, 0]])
    ally = np.concatenate([sample_xy[:, 1], ref_xy[:, 1]])
    xb = np.concatenate([[allx.min()], edges2[0], [allx.max()]])
    yb = np.concatenate([[ally.min()], edges2[1], [ally.max()]])

    cmap = ListedColormap(["#FFFFFF", "#C6D9F0", "#DD8452"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8.5, 7.5))
    plt.pcolormesh(xb, yb, M.T, cmap=cmap, norm=norm, edgecolors="#BBBBBB", linewidth=0.6)
    plt.scatter(sample_xy[:, 0], sample_xy[:, 1], s=8, c="#7A2E12", alpha=0.45, label="Sample")
    handles = [Patch(facecolor="#C6D9F0", edgecolor="#BBBBBB", label="Reference cell"),
               Patch(facecolor="#DD8452", edgecolor="#BBBBBB", label="Covered (Sample-and-Ref)")]
    plt.legend(handles=handles, loc="upper right")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(f"{title}\nGrid Coverage = {coverage*100:.2f}%  "
              f"(inter {len(inter)} / ref {len(ref_set)} cells, {N_BINS}x{N_BINS})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[BinCount] 2D 그림 저장: {out_path}")


# =============================================================================
# 시각화 2: 축별 독립 커버리지 가로 막대 차트
# =============================================================================
def plot_per_axis_coverage(per_axis, out_path):
    """per_axis: [(name, coverage, inter, denom), ...] → 오름차순 가로 막대(취약 축이 위)."""
    order = sorted(range(len(per_axis)), key=lambda i: per_axis[i][1])
    names = [per_axis[i][0] for i in order]
    covs = np.array([per_axis[i][1] for i in order])
    colors = plt.cm.RdYlGn(covs)              # 낮으면 빨강, 높으면 초록

    plt.figure(figsize=(8, max(5, 0.34 * len(names))))
    y = np.arange(len(names))
    plt.barh(y, covs * 100, color=colors, edgecolor="#888888", linewidth=0.4)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()                  # 취약(낮은) 축을 맨 위로
    plt.xlim(0, 100)
    plt.xlabel("Coverage (%)")
    plt.title("Per-Axis Data Coverage (independent 1D bin-count, ascending)")
    for yi, c in zip(y, covs):                # 막대 끝에 값 표기
        plt.text(min(c * 100 + 1, 96), yi, f"{c*100:.0f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[BinCount] 축별 커버리지 그림 저장: {out_path}")


# =============================================================================
# 메인 파이프라인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # (1) Sample 로드 → 스케일러 → 스케일링
    sample = read_feature_matrix(SAMPLE_PATH, fmt="csv")
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    # (2) Sample 기준 PCA: 누적 설명분산 90% 를 넘는 최소 주성분 수 K
    pca_k, K, cumv = fit_pca_var(sample_scaled, VAR_THRESHOLD)
    print(f"[BinCount] 전체 변이의 {VAR_THRESHOLD*100:.0f}%를 보존하기 위해 선택된 "
          f"최적의 주성분 차원 수: {K}개  (상위 {K}개 실제 보존 {cumv*100:.2f}%)")

    # (3) 전통적 2D PCA (상위 2개 주성분)
    pca2 = PCA(n_components=2).fit(sample_scaled)
    sample_pc_k = pca_k.transform(sample_scaled)
    sample_pc2  = pca2.transform(sample_scaled)
    edges_k = pc_bin_edges(sample_pc_k, N_BINS)
    edges2  = pc_bin_edges(sample_pc2,  N_BINS)

    # (4) 커스텀 2D: X = 원본 피처 1개(raw), Y = 나머지 (N-1)차원 PCA(n=1)
    raw_dim = RAW_FEATURE_IDX
    others  = [j for j in range(N_FEATURES) if j != raw_dim]
    pca_rest = PCA(n_components=1).fit(sample_scaled[:, others])
    evr_rest = float(pca_rest.explained_variance_ratio_[0])
    print(f"[BinCount] 나머지 {len(others)}개 차원을 1차원으로 압축했을 때의 "
          f"설명 분산 비율(Explained Variance Ratio): {evr_rest*100:.2f}%")
    sample_cx = sample_scaled[:, raw_dim]
    sample_cy = pca_rest.transform(sample_scaled[:, others])[:, 0]
    ex_c = np.linspace(sample_cx.min(), sample_cx.max(), N_BINS + 1)[1:-1]
    ey_c = np.linspace(sample_cy.min(), sample_cy.max(), N_BINS + 1)[1:-1]

    # (5) 축별 1D 독립 커버리지용 축별 경계 (원본 축, PER_AXIS_BINS 개)
    raw_edges = pc_bin_edges(sample_scaled, PER_AXIS_BINS)

    # (6) 투영 파라미터 묶음 (dict; pickle 가능 → 프로세스 워커로 전달)
    proj = {"pca_k": pca_k, "edges_k": edges_k, "pca2": pca2, "edges2": edges2,
            "raw_dim": raw_dim, "others": others, "pca_rest": pca_rest,
            "ex_c": ex_c, "ey_c": ey_c, "raw_edges": raw_edges}

    # (7) Sample 쪽 4종 집계
    sam_kd, sam_2d, sam_cu, occ_sam = _project_all(sample_scaled, proj)

    # (8) Reference: 한 번의 스트리밍 패스로 4종 동시 집계 (병렬 유지)
    map_fn = functools.partial(_project_all, proj=proj)
    ref_kd, ref_2d, ref_cu, occ_ref = parallel_reduce_reference(
        REF_PATH, scaler, map_fn, _reduce_all, _init_all,
        n_workers=N_WORKERS_A, chunksize=CHUNKSIZE, fmt=FMT,
    )
    if not ref_kd:
        raise ValueError("Set_ref 가 비어있다. Reference 데이터를 확인하라.")

    # (9) 커버리지 계산
    cov_k  = len(sam_kd & ref_kd) / len(ref_kd)
    cov_2  = len(sam_2d & ref_2d) / len(ref_2d)
    cov_cu = len(sam_cu & ref_cu) / len(ref_cu)

    # 축별 독립 커버리지: |sam_bins ∩ ref_bins| / |ref_bins|  (축마다)
    per_axis = []
    for a in range(N_FEATURES):
        denom = int(occ_ref[a].sum())
        inter = int((occ_ref[a] & occ_sam[a]).sum())
        per_axis.append((f"f{a:02d}", inter / denom if denom else 0.0, inter, denom))

    # (10) 그림 3장 (Reference 는 그림용으로만 downsample)
    ref_pts = reservoir_sample_reference(REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    # 10-a) 전통적 2D PCA
    plot_2d_coverage(sample_pc2, pca2.transform(ref_pts), ref_2d, sam_2d, edges2, cov_2,
                     os.path.join(OUTPUT_DIR, "bincount_2d_coverage.png"),
                     "PC1", "PC2", "Traditional 2D PCA (PC1 vs PC2)")
    # 10-b) 커스텀 2D (원본 1D + 나머지 20D→PC1)
    ref_cx = ref_pts[:, raw_dim]
    ref_cy = pca_rest.transform(ref_pts[:, others])[:, 0]
    plot_2d_coverage(np.column_stack([sample_cx, sample_cy]),
                     np.column_stack([ref_cx, ref_cy]), ref_cu, sam_cu, [ex_c, ey_c], cov_cu,
                     os.path.join(OUTPUT_DIR, "custom_2d_coverage.png"),
                     f"f{raw_dim:02d} (raw)", f"rest-{len(others)}D PC1 (EVR {evr_rest*100:.1f}%)",
                     f"Custom 2D: raw f{raw_dim:02d} + rest-{len(others)}D PCA(1)")
    # 10-c) 축별 독립 커버리지 막대차트
    plot_per_axis_coverage(per_axis, os.path.join(OUTPUT_DIR, "per_axis_coverage.png"))

    # (11) 결과 출력
    print("\n" + "=" * 62)
    print("  [알고리즘 A] BIN COUNT 공간 커버리지 (PCA 축소 + 진단)")
    print("=" * 62)
    print(f"  보존 분산 {VAR_THRESHOLD*100:.0f}%  →  선택 주성분 K = {K}")
    print(f"  >>> Coverage (K={K}차원 PCA)      : {cov_k*100:6.2f}%  "
          f"(∩{len(sam_kd & ref_kd):,} / ref {len(ref_kd):,})")
    print(f"  >>> Coverage (전통 2D PCA)        : {cov_2*100:6.2f}%")
    print(f"  >>> Coverage (커스텀 2D, EVR {evr_rest*100:.1f}%): {cov_cu*100:6.2f}%")
    print("-" * 62)
    print(f"  축별 독립 커버리지 (취약 축 오름차순, Bin={PER_AXIS_BINS}):")
    for name, cov_a, inter, denom in sorted(per_axis, key=lambda t: t[1]):
        bar = "#" * int(cov_a * 20)
        print(f"    {name} : {cov_a*100:6.2f}%  ({inter:>2}/{denom:>2}) {bar}")
    print("=" * 62)
    return {"K": K, "coverage_k": cov_k, "coverage_2d": cov_2,
            "coverage_custom_2d": cov_cu, "per_axis": per_axis}


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
