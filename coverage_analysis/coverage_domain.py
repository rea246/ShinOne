# -*- coding: utf-8 -*-
"""
coverage_domain.py
==================
[Domain Coverage] 물리적 설계 스펙(Spec Bounds) 기준의 커버리지

설계 철학
    기존 BIN COUNT 는 입력된 '데이터 분포'에 종속적(Data-driven)이었다.
    Domain Coverage 는 시스템/장비의 '물리적 설계 스펙(Spec Bounds)'을 기준으로 삼는다.

두 가지 도메인 정의 (DOMAIN_MODE)
    "box"         : 각 Feature 의 Lower/Upper Spec 이 이루는 축-정렬 초입방체(Hyper-rectangle).
                    스펙의 정의에 가장 충실하지만, 피처 상관 시 '박스 모서리'가 도달 불가능한
                    빈 공간이라 |Set_domain|=N_BINS^N_FEATURES 이 과대 → Coverage 가 극단적으로 작다.
    "mahalanobis" : Reference 의 평균 μ·공분산 Σ 로 정의한 '상관구조 반영 타원체'
                    { x : (x−μ)ᵀΣ⁻¹(x−μ) ≤ χ²(q, d) } 를 '도달 가능한 도메인'으로 삼는다.
                    박스의 빈 모서리를 잘라내 훨씬 타이트한(현실적인) 도메인이 된다.
                    μ·Σ 는 parallel_reduce_reference 로 '한 스트리밍 패스'에 추정한다.
                    커버리지는 whitened(백색화) top-K 타원 '안쪽 격자만' 분모로 삼아
                    차원의 저주 없이 유의미한 값을 준다.

지표
    Reference/Sample Domain Coverage = |Set_ref/sam ∩ Set_domain| / |Set_domain|
    + 스펙/타원 밖(Out-of-Domain) 비율.

시각화 (UMAP vs PCA)
    Sample + Reference 하이라이트 표본을 UMAP·PCA 2D 로 임베딩해, 격자 영역을
    (회색=빈 도메인 / 연파랑=Reference only / 주황=Sample∩Ref 검증완료)로 칠하고 테두리를 그린다.
    Mahalanobis 모드는 추가로 whitened top-2 평면의 타원+격자 커버리지도 저장한다.
      → domain_coverage_pca.png, domain_coverage_umap.png, (mahalanobis) domain_mahalanobis_2d.png

엔지니어링
    50GB Reference 는 parallel_reduce_reference(멀티프로세스 청크 스트리밍)로 처리(모멘트/커버리지 각 1패스).
    UMAP/PCA 학습은 Sample + Reference 하이라이트 표본만으로 고속 fit_transform.

실행:  python coverage_domain.py   (필요 시: pip install umap-learn scipy matplotlib)
"""

import functools
import itertools
import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 환경용 백엔드
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle, Patch, Rectangle
from scipy.stats import chi2
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
SCALE_METHOD  = "standard"   # 공유 스케일러 (Mahalanobis 는 standard 권장)

DOMAIN_MODE   = "mahalanobis"  # 'box' 또는 'mahalanobis'

# ---- box 모드 ----
N_BINS        = 4            # 각 Spec 축을 균등 분할할 Bin 수
SPEC_LOWER    = None         # 물리 스펙 직접 입력(길이 N_FEATURES) 또는 None → SPEC_MODE 자동
SPEC_UPPER    = None
SPEC_MODE     = "4sigma"     # 'minmax' 또는 '4sigma'

# ---- mahalanobis 모드 ----
MAHAL_Q       = 0.99         # 타원체 도메인 경계 분위수 (χ² quantile)
MAHAL_GRID_DIMS = 2          # 커버리지 격자를 칠 whitened 주성분 수 (2 권장, 최대 3)
N_BINS_MAHAL  = 10           # whitened 축당 Bin 수 (1D 아닌 저차원이라 촘촘히)

CHUNKSIZE     = 200_000
N_WORKERS_D   = None         # None → common.N_WORKERS 기본값

OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000     # UMAP/PCA 학습·시각화용 Reference 표본 수
GRID_2D         = 30         # UMAP/PCA 시각화 격자 해상도

GEN_DUMMY     = True
REF_ROWS      = 2_000_000


# =============================================================================
# [BOX] 스펙 경계 → 축별 도메인 격자 경계 + 격자 점유 map
# =============================================================================
def compute_spec_bounds(sample_raw):
    """Sample 원본값에서 외부 Spec 경계(lower, upper, 원본 단위)를 정한다."""
    if SPEC_LOWER is not None and SPEC_UPPER is not None:
        return np.asarray(SPEC_LOWER, float), np.asarray(SPEC_UPPER, float)
    if SPEC_MODE == "minmax":
        return sample_raw.min(axis=0), sample_raw.max(axis=0)
    mu, sd = sample_raw.mean(axis=0), sample_raw.std(axis=0)
    return mu - 4 * sd, mu + 4 * sd


def domain_edges(scaler, lower_raw, upper_raw):
    """원본 Spec 경계를 스케일러로 변환 후, 축마다 [lower,upper] 를 N_BINS 등분한 경계 리스트."""
    lo = scaler.transform(lower_raw.reshape(1, -1))[0]
    hi = scaler.transform(upper_raw.reshape(1, -1))[0]
    return [np.linspace(lo[j], hi[j], N_BINS + 1) for j in range(N_FEATURES)]


def domain_map(block, edges):
    """[box] Spec 범위 안쪽 점만 격자화한 셀 집합 + 스펙초과/전체 카운트."""
    n = block.shape[0]
    if n == 0:
        return [set(), 0, 0]
    idx = np.empty(block.shape, dtype=np.int16)
    in_dom = np.ones(n, dtype=bool)
    for j in range(block.shape[1]):
        col = block[:, j]
        in_dom &= (col >= edges[j][0]) & (col <= edges[j][-1])
        idx[:, j] = np.digitize(col, edges[j][1:-1])
    idx = idx[in_dom]
    if idx.shape[0] == 0:
        return [set(), int(n), int(n)]
    cells = {row.tobytes() for row in np.unique(idx, axis=0)}
    return [cells, int(n - idx.shape[0]), int(n)]


# ---- [box/mahalanobis 공용] 격자셋 + 카운트 누적기 ----
def _domain_init():
    return [set(), 0, 0]


def _domain_reduce(acc, part):
    acc[0].update(part[0]); acc[1] += part[1]; acc[2] += part[2]
    return acc


# =============================================================================
# [MAHALANOBIS] 스트리밍 모멘트(μ, Σ) 추정 → 타원체 도메인 → 격자 커버리지
# =============================================================================
def _moments_map(block):
    """한 chunk 의 [n, Σx, ΣxxT] 를 반환 (μ·Σ 스트리밍 추정용)."""
    return [block.shape[0], block.sum(axis=0), block.T @ block]


def _moments_init():
    return [0, np.zeros(N_FEATURES), np.zeros((N_FEATURES, N_FEATURES))]


def _moments_reduce(acc, part):
    acc[0] += part[0]; acc[1] += part[1]; acc[2] += part[2]
    return acc


def build_mahalanobis(mu, Sigma, q, K):
    """
    μ·Σ 로 Mahalanobis 타원체 도메인을 구성한다.
    반환 dict:
        mu, Sinv, T(full-d χ² 임계), Vk(상위 K 고유벡터), sqrt_lamk,
        Tk(K-d χ² 임계), Rk(=√Tk), internalK(whitened 축 내부경계),
        evr_k(상위 K 설명분산 비율), domain_set(타원 안 격자셀), (w, V) 전체 고유분해
    """
    d = len(mu)
    ridge = 1e-6 * (np.trace(Sigma) / d)
    Sig = Sigma + ridge * np.eye(d)
    Sinv = np.linalg.inv(Sig)
    w, V = np.linalg.eigh(Sig)                     # 오름차순
    order = np.argsort(w)[::-1]                     # 큰 고유값 순
    w, V = w[order], V[:, order]
    Vk, lamk = V[:, :K], w[:K]
    sqrt_lamk = np.sqrt(lamk)
    T, Tk = float(chi2.ppf(q, d)), float(chi2.ppf(q, K))
    Rk = float(np.sqrt(Tk))
    internalK = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)[1:-1]
    evr_k = float(lamk.sum() / w.sum())

    # 타원(반경 √Tk) 안쪽 격자셀만 열거 → Set_domain (corner-free 유한 분모)
    edges = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    domain_set = set()
    for combo in itertools.product(range(N_BINS_MAHAL), repeat=K):
        if (centers[list(combo)] ** 2).sum() <= Tk:
            domain_set.add(np.array(combo, dtype=np.int16).tobytes())

    return {"mu": mu, "Sinv": Sinv, "T": T, "Vk": Vk, "sqrt_lamk": sqrt_lamk,
            "Tk": Tk, "Rk": Rk, "internalK": internalK, "evr_k": evr_k,
            "domain_set": domain_set, "w": w, "V": V}


def _whiten_topk(X, p):
    """X → whitened 상위 K 주성분 좌표 u (isotropic). u_i = vᵢ·(x−μ)/√λᵢ."""
    return ((X - p["mu"]) @ p["Vk"]) / p["sqrt_lamk"]


def mahal_cover_map(block, p):
    """[mahalanobis] whitened top-K 타원 안 격자 셀 집합 + (full-d) 타원밖 카운트."""
    n = block.shape[0]
    if n == 0:
        return [set(), 0, 0]
    Xc = block - p["mu"]
    d2_full = np.einsum("ij,jk,ik->i", Xc, p["Sinv"], Xc)   # 전체 차원 Mahalanobis²
    n_out = int((d2_full > p["T"]).sum())
    u = (Xc @ p["Vk"]) / p["sqrt_lamk"]                     # (n, K) whitened
    m = (u ** 2).sum(axis=1) <= p["Tk"]                     # top-K 타원 안
    uu = u[m]
    if uu.shape[0] == 0:
        return [set(), n_out, n]
    idx = np.empty(uu.shape, dtype=np.int16)
    for a in range(uu.shape[1]):
        idx[:, a] = np.digitize(uu[:, a], p["internalK"])
    cells = {row.tobytes() for row in np.unique(idx, axis=0)}
    return [cells, n_out, n]


def _slim(p):
    """워커로 넘길 최소 파라미터(picklable)만 추린다."""
    return {k: p[k] for k in ("mu", "Sinv", "T", "Vk", "sqrt_lamk", "Tk", "internalK")}


# =============================================================================
# 시각화 A: UMAP / PCA 2D 도메인 영역 (box/mahalanobis 공용)
# =============================================================================
def plot_domain_regions(emb_sample, emb_ref, ref_cov, sam_cov, out_path, method):
    allx = np.concatenate([emb_sample[:, 0], emb_ref[:, 0]])
    ally = np.concatenate([emb_sample[:, 1], emb_ref[:, 1]])
    xe = np.linspace(allx.min(), allx.max(), GRID_2D + 1)
    ye = np.linspace(ally.min(), ally.max(), GRID_2D + 1)
    Hr = np.histogram2d(emb_ref[:, 0], emb_ref[:, 1], bins=[xe, ye])[0] > 0
    Hs = np.histogram2d(emb_sample[:, 0], emb_sample[:, 1], bins=[xe, ye])[0] > 0
    M = np.zeros_like(Hr, dtype=float)
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
    handles = [Patch(facecolor="#E8E8E8", edgecolor="#BBB", label="Empty domain (no data)"),
               Patch(facecolor="#C6D9F0", edgecolor="#BBB", label="Reference only"),
               Patch(facecolor="#DD8452", edgecolor="#BBB", label="Verified (Sample-and-Ref)")]
    plt.legend(handles=handles, loc="upper right", framealpha=0.9)
    plt.xlabel(f"{method}-1"); plt.ylabel(f"{method}-2")
    plt.title(f"Domain Coverage on {method} 2D\n"
              f"Ref cov = {ref_cov:.3e}   Sample cov = {sam_cov:.3e}")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"[Domain] {method} 그림 저장: {out_path}")


def embed_2d(stack, n_sample):
    """(sample+ref) 스택을 PCA 2D 와 UMAP 2D 로 임베딩."""
    pca = PCA(n_components=2).fit(stack).transform(stack)
    result = {"PCA": (pca[:n_sample], pca[n_sample:])}
    try:
        import umap
        um = umap.UMAP(n_components=2, random_state=42).fit_transform(stack)
        result["UMAP"] = (um[:n_sample], um[n_sample:])
    except Exception as e:
        print(f"[Domain] UMAP 건너뜀 ({e}). 'pip install umap-learn' 후 재실행하면 UMAP 그림도 생성됩니다.")
    return result


# =============================================================================
# 시각화 B: Mahalanobis whitened top-2 타원 격자 커버리지
# =============================================================================
def plot_mahalanobis_2d(u_sample, u_ref, ref_set, sam_set, p, coverage, out_path):
    """whitened top-2 평면: 타원(반경 √Tk) 안 격자셀을 도메인/Ref/검증 상태로 칠한다."""
    dom = p["domain_set"]
    edges = np.linspace(-p["Rk"], p["Rk"], N_BINS_MAHAL + 1)
    M = np.zeros((N_BINS_MAHAL, N_BINS_MAHAL))     # 0 밖 / 1 빈도메인 / 2 ref / 3 검증
    for i in range(N_BINS_MAHAL):
        for j in range(N_BINS_MAHAL):
            key = np.array([i, j], dtype=np.int16).tobytes()
            if key not in dom:
                continue
            in_r, in_s = key in ref_set, key in sam_set
            M[i, j] = 3 if (in_r and in_s) else (2 if in_r else 1)
    cmap = ListedColormap(["#FFFFFF", "#E8E8E8", "#C6D9F0", "#DD8452"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    plt.figure(figsize=(8.5, 8))
    plt.pcolormesh(edges, edges, M.T, cmap=cmap, norm=norm, edgecolors="#CCC", linewidth=0.4)
    plt.gca().add_patch(Circle((0, 0), p["Rk"], fill=False, edgecolor="#333", linewidth=1.6))
    plt.scatter(u_sample[:, 0], u_sample[:, 1], s=7, c="#5A2109", alpha=0.4, linewidths=0)
    handles = [Patch(facecolor="#E8E8E8", label="Empty domain (in-ellipse, no ref)"),
               Patch(facecolor="#C6D9F0", label="Reference only"),
               Patch(facecolor="#DD8452", label="Verified (Sample-and-Ref)")]
    plt.legend(handles=handles, loc="upper right", framealpha=0.9)
    plt.xlabel("whitened PC1"); plt.ylabel("whitened PC2")
    plt.title(f"Mahalanobis Domain (q={MAHAL_Q}) on whitened top-2\n"
              f"Sample Domain Coverage = {coverage*100:.2f}%  "
              f"(ellipse radius √Tk={p['Rk']:.2f})")
    plt.axis("equal")
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    print(f"[Domain] Mahalanobis 2D 그림 저장: {out_path}")


# =============================================================================
# 메인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_workers = N_WORKERS_D
    if n_workers is None:
        from common import N_WORKERS
        n_workers = N_WORKERS

    # 공통: Sample 로드 → 공유 스케일러 → 스케일링
    sample = read_feature_matrix(SAMPLE_PATH, fmt="csv")
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    mahal = None
    if DOMAIN_MODE == "box":
        lower_raw, upper_raw = compute_spec_bounds(sample)
        edges = domain_edges(scaler, lower_raw, upper_raw)
        n_domain = N_BINS ** N_FEATURES
        src = "직접 입력" if SPEC_LOWER is not None else f"Sample 자동({SPEC_MODE})"
        print(f"[Domain/box] Spec 경계: {src}, 격자 {N_BINS}^{N_FEATURES} = {n_domain:,} 셀")
        sam_cells, sam_out, sam_tot = domain_map(sample_scaled, edges)
        map_fn = functools.partial(domain_map, edges=edges)
        ref_cells, ref_out, ref_tot = parallel_reduce_reference(
            REF_PATH, scaler, map_fn, _domain_reduce, _domain_init,
            n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT)
        ref_cov, sam_cov = len(ref_cells) / n_domain, len(sam_cells) / n_domain
        verified = len(sam_cells & ref_cells)
        denom_ref = len(ref_cells)

    else:  # ---- mahalanobis ----
        # (1) μ·Σ 스트리밍 추정 (parallel_reduce_reference 1패스)
        n_ref, sum_x, sum_xx = parallel_reduce_reference(
            REF_PATH, scaler, _moments_map, _moments_reduce, _moments_init,
            n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT)
        mu = sum_x / n_ref
        Sigma = sum_xx / n_ref - np.outer(mu, mu)
        mahal = build_mahalanobis(mu, Sigma, MAHAL_Q, MAHAL_GRID_DIMS)
        dom = mahal["domain_set"]
        print(f"[Domain/mahalanobis] μ·Σ 스트리밍 추정({n_ref:,} rows), "
              f"타원 q={MAHAL_Q} (√T_full={np.sqrt(mahal['T']):.2f}); "
              f"whitened top-{MAHAL_GRID_DIMS} EVR={mahal['evr_k']*100:.1f}%, "
              f"|Set_domain|={len(dom):,} 셀")
        # (2) 커버리지 스트리밍 패스
        p_slim = _slim(mahal)
        sam_cells, sam_out, sam_tot = mahal_cover_map(sample_scaled, p_slim)
        map_fn = functools.partial(mahal_cover_map, p=p_slim)
        ref_cells, ref_out, ref_tot = parallel_reduce_reference(
            REF_PATH, scaler, map_fn, _domain_reduce, _domain_init,
            n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT)
        ref_cov = len(ref_cells & dom) / len(dom)
        sam_cov = len(sam_cells & dom) / len(dom)
        verified = len(sam_cells & ref_cells & dom)
        denom_ref = len(ref_cells & dom)

    # 시각화 (공용): UMAP vs PCA
    ref_pts = reservoir_sample_reference(REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    stack = np.vstack([sample_scaled, ref_pts])
    print(f"[Domain] 2D 임베딩 학습 (Sample {len(sample_scaled):,} + Ref {len(ref_pts):,}) ...")
    for method, (emb_s, emb_r) in embed_2d(stack, len(sample_scaled)).items():
        plot_domain_regions(emb_s, emb_r, ref_cov, sam_cov,
                            os.path.join(OUTPUT_DIR, f"domain_coverage_{method.lower()}.png"), method)

    # Mahalanobis 전용: whitened top-2 타원 격자 커버리지
    if mahal is not None:
        u_s, u_r = _whiten_topk(sample_scaled, mahal)[:, :2], _whiten_topk(ref_pts, mahal)[:, :2]
        plot_mahalanobis_2d(u_s, u_r, ref_cells, sam_cells, mahal, sam_cov,
                            os.path.join(OUTPUT_DIR, "domain_mahalanobis_2d.png"))

    # 결과 출력
    ref_in = len(ref_cells & mahal["domain_set"]) if mahal else len(ref_cells)
    sam_in = len(sam_cells & mahal["domain_set"]) if mahal else len(sam_cells)
    print("\n" + "=" * 66)
    print(f"  [Domain Coverage] mode = {DOMAIN_MODE}")
    print("=" * 66)
    if not mahal:
        print(f"  |Set_domain| = {N_BINS}^{N_FEATURES}         : {N_BINS**N_FEATURES:,}")
    print(f"  Reference in-domain 셀 |Set_ref|    : {ref_in:,}")
    print(f"  Sample    in-domain 셀 |Set_sam|    : {sam_in:,}")
    print(f"  >>> Reference Domain Coverage       : {ref_cov*100:8.4f}%  (={ref_cov:.3e})  스펙/도메인 대비 Ref 밀집도")
    print(f"  >>> Sample    Domain Coverage       : {sam_cov*100:8.4f}%  (={sam_cov:.3e})  도메인 대비 Sample 검증영역")
    print(f"  Sample 이 Ref 도메인 footprint 검증  : {verified:,}/{denom_ref:,} 셀 "
          f"({(verified/denom_ref*100 if denom_ref else 0):.2f}%)")
    print(f"  Out-of-Domain 비율  Reference={ref_out/max(ref_tot,1)*100:.2f}%  "
          f"Sample={sam_out/max(sam_tot,1)*100:.2f}%")
    print("=" * 66)
    return {"mode": DOMAIN_MODE, "ref_domain_cov": ref_cov, "sample_domain_cov": sam_cov,
            "verified": verified, "ref_cells": len(ref_cells), "sam_cells": len(sam_cells)}


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
