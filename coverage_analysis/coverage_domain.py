# -*- coding: utf-8 -*-
"""
coverage_domain.py
==================
[Domain Coverage] 물리적 설계 스펙(Spec Bounds) 기준의 커버리지

설계 철학
    기존 BIN COUNT 는 입력된 '데이터 분포'에 종속적(Data-driven)이었다.
    Domain Coverage 는 시스템/장비의 '물리적 설계 스펙(Spec Bounds)'을 기준으로 삼는다.
      * 각 Feature 의 물리적 하한/상한(Lower/Upper Spec)이 이루는 21차원 초입방체(Hyper-rectangle)
        가 '전체 도메인 공간' 이다.
      * 1,800개 Sample 이 이 '절대적 Spec 공간' 을 얼마나 안전하게 검증(대변)하고 있는지 평가한다.

지표 (전체 Spec 공간을 N_BINS 로 균등 분할 → 가상의 도메인 격자 Set_domain, 데이터가 없어도 분모 포함)
    Reference Domain Coverage = |Set_ref ∩ Set_domain| / |Set_domain|   (스펙 대비 Reference 밀집도)
    Sample    Domain Coverage = |Set_sam ∩ Set_domain| / |Set_domain|   (스펙 대비 Sample 검증 영역)
    |Set_domain| = N_BINS ^ N_FEATURES  (실제로 열거하지 않고 정수로 계산)
    스펙 경계를 벗어난 점은 도메인 밖(Out-of-Spec)으로 간주해 격자 집합에서 제외한다.

시각화 (UMAP vs PCA)
    고차원 매니폴드의 비선형 경계를 왜곡 없이 2D 로 보기 위해 UMAP 을 도입하고, 전통적 PCA 2D 와
    비교한다. 각 투영에서 (Reference-only=연파랑 / Sample∩Ref 검증완료=주황 / 데이터 없는 스펙 공백=회색)
    영역을 격자와 공간 테두리로 기하학적으로 시각화한다.
      → domain_coverage_pca.png, domain_coverage_umap.png

엔지니어링
    * 50GB Reference 는 parallel_reduce_reference(멀티프로세스 청크 스트리밍)로 그대로 처리.
    * UMAP/PCA 학습은 메모리 보호를 위해 Sample(1,800) + Reference 하이라이트 표본만으로 고속 fit_transform.

실행:  python coverage_domain.py   (필요 시: pip install umap-learn matplotlib)
"""

import functools
import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 환경용 백엔드
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle
from sklearn.decomposition import PCA

from common import (
    DummyDataGenerator, N_FEATURES,
    fit_scaler, read_feature_matrix,
    reservoir_sample_reference, parallel_reduce_reference,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
#   ※ '읽을 컬럼 범위'(FEATURE_COL_IDX)와 헤더 유무(HAS_HEADER)는 common.py 상단에서 설정한다.
# =============================================================================
SAMPLE_PATH   = "dummy_sample.csv"
REF_PATH      = "dummy_reference.csv"
FMT           = "csv"        # 'csv' 또는 'parquet'
SCALE_METHOD  = "standard"   # 공유 스케일러 (도메인 격자는 affine 이라 선택에 무관, 시각화엔 standard 가 유리)
N_BINS        = 4            # 각 Spec 축을 균등 분할할 Bin 수

# ---- 도메인 스펙 경계 (물리적 Lower/Upper Spec) ----
#   실제 적용: 21개 피처의 물리 스펙을 아래에 직접 넣는다 (원본 단위, 길이 N_FEATURES).
#     예) SPEC_LOWER = [l0, l1, ..., l20];  SPEC_UPPER = [u0, u1, ..., u20]
#   None 이면 아래 SPEC_MODE 로 Sample 분포에서 '외부 Spec 공간'을 자동 정의한다.
SPEC_LOWER    = None
SPEC_UPPER    = None
SPEC_MODE     = "4sigma"     # 'minmax'(Sample 전체 Min~Max) 또는 '4sigma'(mean ± 4σ)

CHUNKSIZE     = 200_000
N_WORKERS_D   = None         # None → common.N_WORKERS 기본값 (Parquet 병렬)

OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000     # UMAP/PCA 학습·시각화용 Reference 하이라이트 표본 수
GRID_2D         = 30         # 2D 시각화 격자 해상도

GEN_DUMMY     = True
REF_ROWS      = 2_000_000


# =============================================================================
# 1. 스펙 경계 → 축별 도메인 격자 경계(scaled 공간)
# =============================================================================
def compute_spec_bounds(sample_raw):
    """Sample 원본값에서 외부 Spec 경계(lower, upper, 원본 단위)를 정한다."""
    if SPEC_LOWER is not None and SPEC_UPPER is not None:
        return np.asarray(SPEC_LOWER, float), np.asarray(SPEC_UPPER, float)
    if SPEC_MODE == "minmax":
        return sample_raw.min(axis=0), sample_raw.max(axis=0)
    mu, sd = sample_raw.mean(axis=0), sample_raw.std(axis=0)     # 4sigma
    return mu - 4 * sd, mu + 4 * sd


def domain_edges(scaler, lower_raw, upper_raw):
    """
    원본 Spec 경계를 스케일러로 변환한 뒤, 축마다 [lower,upper] 를 N_BINS 등분한
    '전체 경계(outer 포함, N_BINS+1개)' 리스트를 만든다. (scaled 공간에서 격자화)
    스케일링은 축별 affine 증가 변환이라 원본 공간의 격자와 1:1 대응한다.
    """
    lo = scaler.transform(lower_raw.reshape(1, -1))[0]
    hi = scaler.transform(upper_raw.reshape(1, -1))[0]
    return [np.linspace(lo[j], hi[j], N_BINS + 1) for j in range(N_FEATURES)]


# =============================================================================
# 2. 도메인 격자 점유 + Out-of-Spec 카운트 (스트리밍 map)
#    누적기 = [set_cells, n_out_of_spec, n_total]
# =============================================================================
def domain_map(block, edges):
    """
    스케일링된 block → Spec 경계 안쪽 점만 격자화한 고유 셀 집합 + 스펙초과/전체 카운트.
    각 축에서 [edges[j][0], edges[j][-1]] 를 벗어나면 Out-of-Spec 으로 간주해 제외한다.
    """
    n = block.shape[0]
    if n == 0:
        return [set(), 0, 0]
    idx = np.empty(block.shape, dtype=np.int16)
    in_dom = np.ones(n, dtype=bool)
    for j in range(block.shape[1]):
        col = block[:, j]
        in_dom &= (col >= edges[j][0]) & (col <= edges[j][-1])      # 스펙 범위 안
        idx[:, j] = np.digitize(col, edges[j][1:-1])               # 0..N_BINS-1
    idx = idx[in_dom]
    if idx.shape[0] == 0:
        return [set(), int(n), int(n)]
    uniq = np.unique(idx, axis=0)
    cells = {row.tobytes() for row in uniq}
    return [cells, int(n - idx.shape[0]), int(n)]


def _domain_init():
    return [set(), 0, 0]


def _domain_reduce(acc, part):
    acc[0].update(part[0])
    acc[1] += part[1]
    acc[2] += part[2]
    return acc


# =============================================================================
# 3. UMAP / PCA 2D 도메인 영역 시각화
# =============================================================================
def plot_domain_regions(emb_sample, emb_ref, ref_cov, sam_cov, out_path, method):
    """
    2D 투영 위에서 GRID_2D×GRID_2D 격자로 영역을 분류해 색칠한다.
        회색   = 스펙 공백(격자 안이지만 데이터 없음)
        연파랑 = Reference 만 존재
        주황   = Sample∩Reference (검증 완료된 안전 영역)
    공간 테두리(전체 extent)를 사각형으로 함께 그린다.
    """
    allx = np.concatenate([emb_sample[:, 0], emb_ref[:, 0]])
    ally = np.concatenate([emb_sample[:, 1], emb_ref[:, 1]])
    xe = np.linspace(allx.min(), allx.max(), GRID_2D + 1)
    ye = np.linspace(ally.min(), ally.max(), GRID_2D + 1)

    Hr = np.histogram2d(emb_ref[:, 0], emb_ref[:, 1], bins=[xe, ye])[0] > 0
    Hs = np.histogram2d(emb_sample[:, 0], emb_sample[:, 1], bins=[xe, ye])[0] > 0
    M = np.zeros_like(Hr, dtype=float)          # 0 공백 / 1 ref-only / 2 검증
    M[Hr & ~Hs] = 1
    M[Hr & Hs] = 2

    cmap = ListedColormap(["#E8E8E8", "#C6D9F0", "#DD8452"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(9, 8))
    plt.pcolormesh(xe, ye, M.T, cmap=cmap, norm=norm, edgecolors="none")
    plt.scatter(emb_sample[:, 0], emb_sample[:, 1], s=5, c="#5A2109", alpha=0.35, linewidths=0)
    plt.gca().add_patch(Rectangle((allx.min(), ally.min()),
                                  allx.max() - allx.min(), ally.max() - ally.min(),
                                  fill=False, edgecolor="#333333", linewidth=1.4))
    handles = [Patch(facecolor="#E8E8E8", edgecolor="#BBB", label="Empty spec gap (no data)"),
               Patch(facecolor="#C6D9F0", edgecolor="#BBB", label="Reference only"),
               Patch(facecolor="#DD8452", edgecolor="#BBB", label="Verified (Sample-and-Ref)")]
    plt.legend(handles=handles, loc="upper right", framealpha=0.9)
    plt.xlabel(f"{method}-1"); plt.ylabel(f"{method}-2")
    plt.title(f"Domain Coverage on {method} 2D\n"
              f"Ref domain cov = {ref_cov:.3e}   Sample domain cov = {sam_cov:.3e}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[Domain] {method} 그림 저장: {out_path}")


def embed_2d(stack, n_sample):
    """(sample+ref) 스택을 PCA 2D 와 UMAP 2D 로 임베딩해 각각 (sample, ref) 로 분리 반환."""
    pca = PCA(n_components=2).fit(stack).transform(stack)
    result = {"PCA": (pca[:n_sample], pca[n_sample:])}
    try:
        import umap
        um = umap.UMAP(n_components=2, random_state=42).fit_transform(stack)
        result["UMAP"] = (um[:n_sample], um[n_sample:])
    except Exception as e:                       # umap 미설치/실패 시 PCA 만
        print(f"[Domain] UMAP 건너뜀 ({e}). 'pip install umap-learn' 후 재실행하면 UMAP 그림도 생성됩니다.")
    return result


# =============================================================================
# 4. 메인 파이프라인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_workers = N_WORKERS_D
    if n_workers is None:
        from common import N_WORKERS
        n_workers = N_WORKERS

    # (1) Sample 로드 → 공유 스케일러 학습 → 스케일링
    sample = read_feature_matrix(SAMPLE_PATH, fmt="csv")
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    # (2) 물리 Spec 경계 정의 → 축별 도메인 격자 경계(scaled)
    lower_raw, upper_raw = compute_spec_bounds(sample)
    edges = domain_edges(scaler, lower_raw, upper_raw)
    n_domain = N_BINS ** N_FEATURES               # |Set_domain| (정수, 열거하지 않음)
    src = "직접 입력" if SPEC_LOWER is not None else f"Sample 자동({SPEC_MODE})"
    print(f"[Domain] Spec 경계: {src},  격자 {N_BINS}^{N_FEATURES} = {n_domain:,} 셀")

    # (3) Sample 도메인 격자 집합
    sam_cells, sam_out, sam_tot = domain_map(sample_scaled, edges)

    # (4) Reference: 스트리밍 1패스로 도메인 격자 집합 + Out-of-Spec 카운트 누적 (병렬 유지)
    map_fn = functools.partial(domain_map, edges=edges)
    ref_cells, ref_out, ref_tot = parallel_reduce_reference(
        REF_PATH, scaler, map_fn, _domain_reduce, _domain_init,
        n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT,
    )

    # (5) 커버리지
    ref_cov = len(ref_cells) / n_domain
    sam_cov = len(sam_cells) / n_domain
    verified = len(sam_cells & ref_cells)         # Sample 이 Reference 도메인 footprint 를 검증한 셀 수

    # (6) 시각화: UMAP vs PCA (Sample + Reference 하이라이트 표본)
    ref_pts = reservoir_sample_reference(REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    stack = np.vstack([sample_scaled, ref_pts])
    print(f"[Domain] 2D 임베딩 학습 (Sample {len(sample_scaled):,} + Ref {len(ref_pts):,}) ...")
    embeds = embed_2d(stack, len(sample_scaled))
    for method, (emb_s, emb_r) in embeds.items():
        plot_domain_regions(emb_s, emb_r, ref_cov, sam_cov,
                            os.path.join(OUTPUT_DIR, f"domain_coverage_{method.lower()}.png"), method)

    # (7) 결과 출력
    print("\n" + "=" * 64)
    print("  [Domain Coverage] 물리 스펙(Spec Bounds) 기준 커버리지")
    print("=" * 64)
    print(f"  |Set_domain| = {N_BINS}^{N_FEATURES}         : {n_domain:,}")
    print(f"  Reference in-spec 셀 |Set_ref|      : {len(ref_cells):,}")
    print(f"  Sample    in-spec 셀 |Set_sam|      : {len(sam_cells):,}")
    print(f"  >>> Reference Domain Coverage       : {ref_cov:.4e}  (스펙 대비 Ref 밀집도)")
    print(f"  >>> Sample    Domain Coverage       : {sam_cov:.4e}  (스펙 대비 Sample 검증영역)")
    print(f"  Sample 이 Ref 도메인 footprint 검증  : {verified:,}/{len(ref_cells):,} 셀 "
          f"({(verified/len(ref_cells)*100 if ref_cells else 0):.2f}%)")
    print(f"  Out-of-Spec 비율  Reference={ref_out/max(ref_tot,1)*100:.2f}%  "
          f"Sample={sam_out/max(sam_tot,1)*100:.2f}%")
    print("=" * 64)
    return {"ref_domain_cov": ref_cov, "sample_domain_cov": sam_cov,
            "n_domain": n_domain, "ref_cells": len(ref_cells), "sam_cells": len(sam_cells)}


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
