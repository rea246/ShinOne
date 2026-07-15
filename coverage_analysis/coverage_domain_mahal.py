# -*- coding: utf-8 -*-
"""
coverage_domain_mahal.py
========================
[Mahalanobis Domain Coverage + 근원 항(Attribution)] 전용 분석

목적
    Sample 이 Reference 의
        (1) 어느 영역을 '대변'하는가          (Cat1 Verified)
        (2) 어느 영역을 '대변하지 못'하는가    (Cat2 Gap)
        (3) Reference 와 '무관'한 포인트는 어디인가 (Cat3 Out-of-Domain)
    를 분류하고, 각 분류의 '근원 항'(원인 feature) 을 MYT 분해로 찾는다.

도메인 정의 (Mahalanobis)
    Reference 의 μ·Σ 로 상관구조를 반영한 타원체
        { x : (x−μ)ᵀ Σ⁻¹ (x−μ) ≤ χ²(q, d) }
    를 '도달 가능한 도메인'으로 삼는다. Σ 를 고유분해해 상위 K=2 주성분으로 백색화
    (whitening)하면 타원이 반경 √χ²(q,K) 원이 되어, 저차원 격자로 커버리지를 잰다.
        Coverage = |G_active ∩ G_total| / |G_total|

근원 항 (MYT / Mason–Young–Tracy 분해)
    정밀행렬 P=Σ⁻¹, 잔차 e=x−μ 에 대해
        D²        = eᵀ P e                    (full-d Mahalanobis², Cat3 판정)
        c_j       = (P e)_j² / P_jj           (조건부 기여 = D² − D²_(−j), leave-one-out)
        u_j       = e_j² / Σ_jj               (무조건부 기여, 변수 단독)
    c_j 가 큰 feature = 그 포인트가 도메인 밖으로 벗어난 '주범'.

시각화 (seaborn)
    ① domain_mahalanobis_2d.png   : whitened top-2 격자 커버리지 + Cat 색 산점
    ② domain_mahal_distribution.png: full-d D² 분포 (Ref vs Sample) + χ²(d) 이론곡선
    ③ domain_loadings_biplot.png  : PC1·PC2 로딩 화살표(원본 feature 방향)
    ④ domain_ood_attribution.png  : OOD 근원 항 랭킹 막대 (조건부 vs 무조건부)
    ⑤ domain_gap_attribution.png  : Gap 영역 원인 feature 발산형 막대
    ⑥ domain_ood_heatmap.png      : 극단 OOD top-N × feature 조건부 기여 히트맵

엔지니어링
    Reference(대용량)는 parallel_reduce_reference 로 스트리밍(μ·Σ 1패스 + 커버리지 1패스).
    포인트별 분류·근원 항은 Sample(메모리) 전량에 대해 벡터화 계산.
    Gap 원인(C)은 reservoir 표본 Reference 로 근사한다.

실행:  python coverage_domain_mahal.py   (필요 시: pip install seaborn scipy matplotlib)
"""

import functools
import itertools
import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 백엔드
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2


def _setup_korean_font():
    """설치된 한글 폰트 파일을 직접 등록해 matplotlib 에 연결(없으면 조용히 넘어감)."""
    import glob
    from matplotlib import font_manager
    patterns = [
        "/usr/share/fonts/**/NanumGothic.ttf",
        "/usr/share/fonts/**/NotoSansCJK*.otf",
        "/usr/share/fonts/**/NanumBarunGothic.ttf",
        "/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            font_manager.fontManager.addfont(hits[0])
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=hits[0]).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False        # 한글 폰트에서 '−' 깨짐 방지

from common import (
    DummyDataGenerator, N_FEATURES, FEATURE_COLS,
    fit_scaler, read_feature_matrix,
    reservoir_sample_reference, parallel_reduce_reference,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
#   ※ '읽을 컬럼 범위'(FEATURE_COL_IDX)와 헤더 유무(HAS_HEADER)는 common.py 상단에서 설정.
# =============================================================================
SAMPLE_PATH   = "dummy_sample.csv"
REF_PATH      = "dummy_reference.csv"
FMT           = "csv"          # 'csv' 또는 'parquet'
SCALE_METHOD  = "standard"     # Mahalanobis 는 standard 권장

MAHAL_Q       = 0.99           # 타원체 도메인 경계 분위수 (χ² quantile)
MAHAL_GRID_DIMS = 2            # 커버리지 격자를 칠 whitened 주성분 수 (K, 2 고정 권장)
N_BINS_MAHAL  = 10             # whitened 축당 Bin 수 (격자 해상도)

CHUNKSIZE     = 200_000
N_WORKERS_D   = None           # None → common.N_WORKERS 기본값

OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000       # 시각화/Gap 근사용 Reference reservoir 표본 수
OOD_HEATMAP_TOPN = 25          # ⑥ 히트맵에 표시할 극단 OOD 포인트 수

GEN_DUMMY     = True
REF_ROWS      = 2_000_000

sns.set_theme(style="whitegrid", context="talk")
_setup_korean_font()


# =============================================================================
# [MAHALANOBIS] 스트리밍 모멘트(μ, Σ) 추정
# =============================================================================
def _moments_map(block):
    """한 chunk 의 [n, Σx, ΣxxᵀT] 를 반환 (μ·Σ 스트리밍 추정용)."""
    return [block.shape[0], block.sum(axis=0), block.T @ block]


def _moments_init():
    return [0, np.zeros(N_FEATURES), np.zeros((N_FEATURES, N_FEATURES))]


def _moments_reduce(acc, part):
    acc[0] += part[0]; acc[1] += part[1]; acc[2] += part[2]
    return acc


def build_mahalanobis(mu, Sigma, q, K):
    """
    μ·Σ 로 Mahalanobis 타원체 도메인 + whitened top-K 격자 도메인을 구성한다.
    반환 dict:
        mu, Sinv(=P), T(full-d χ² 임계), Vk(상위 K 고유벡터), sqrt_lamk,
        Tk(K-d χ² 임계), Rk(=√Tk), internalK(격자 내부경계), evr_k(상위 K 설명분산비),
        domain_set(타원 안 격자셀 = G_total), diagP(=P 대각), diagS(=Σ 대각).
    """
    d = len(mu)
    ridge = 1e-6 * (np.trace(Sigma) / d)               # 수치 안정화(특이 방지)
    Sig = Sigma + ridge * np.eye(d)
    Sinv = np.linalg.inv(Sig)
    w, V = np.linalg.eigh(Sig)                          # 오름차순 고유값
    order = np.argsort(w)[::-1]                         # 큰 고유값 순
    w, V = w[order], V[:, order]
    Vk, lamk = V[:, :K], w[:K]
    sqrt_lamk = np.sqrt(lamk)
    T, Tk = float(chi2.ppf(q, d)), float(chi2.ppf(q, K))
    Rk = float(np.sqrt(Tk))
    internalK = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)[1:-1]
    evr_k = float(lamk.sum() / w.sum())

    # 타원(반경 √Tk) 안쪽 격자셀만 열거 → G_total (corner-free 유한 분모)
    edges = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    domain_set = set()
    for combo in itertools.product(range(N_BINS_MAHAL), repeat=K):
        if (centers[list(combo)] ** 2).sum() <= Tk:
            domain_set.add(np.array(combo, dtype=np.int16).tobytes())

    return {"mu": mu, "Sinv": Sinv, "T": T, "Vk": Vk, "sqrt_lamk": sqrt_lamk,
            "Tk": Tk, "Rk": Rk, "internalK": internalK, "evr_k": evr_k,
            "domain_set": domain_set, "diagP": np.diag(Sinv).copy(),
            "diagS": np.diag(Sig).copy()}


# =============================================================================
# 커버리지 격자 (map→reduce) — whitened top-K 셀 집합 + full-d 타원밖 카운트
# =============================================================================
def _cells_of(u, internalK, K):
    """whitened 좌표 u(타원 안 점) → 점유 격자셀 집합(bytes key)."""
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(K):
        idx[:, a] = np.digitize(u[:, a], internalK)
    return {row.tobytes() for row in np.unique(idx, axis=0)}


def mahal_cover_map(block, p):
    """한 block → [top-K 타원 안 점유 셀, full-d 타원밖 카운트, 전체 카운트]."""
    n = block.shape[0]
    if n == 0:
        return [set(), 0, 0]
    Xc = block - p["mu"]
    d2_full = np.einsum("ij,jk,ik->i", Xc, p["Sinv"], Xc)   # full-d Mahalanobis²
    n_out = int((d2_full > p["T"]).sum())
    u = (Xc @ p["Vk"]) / p["sqrt_lamk"]                     # whitened top-K
    uu = u[(u ** 2).sum(axis=1) <= p["Tk"]]                 # top-K 타원 안
    if uu.shape[0] == 0:
        return [set(), n_out, n]
    return [_cells_of(uu, p["internalK"], p["Vk"].shape[1]), n_out, n]


def _domain_init():
    return [set(), 0, 0]


def _domain_reduce(acc, part):
    acc[0].update(part[0]); acc[1] += part[1]; acc[2] += part[2]
    return acc


def _slim(p):
    """워커로 넘길 최소 파라미터(picklable)만 추린다."""
    return {k: p[k] for k in ("mu", "Sinv", "T", "Vk", "sqrt_lamk", "Tk", "internalK")}


# =============================================================================
# 포인트별 분류(A) + 근원 항 MYT 분해(B)
# =============================================================================
def whiten_topk(X, p):
    """X → whitened 상위 K 좌표 u. u_i = vᵢ·(x−μ)/√λᵢ (isotropic)."""
    return ((X - p["mu"]) @ p["Vk"]) / p["sqrt_lamk"]


def classify_points(X, p, ref_cells):
    """
    각 포인트를 Cat 로 분류하고 근원 항(MYT)을 계산한다.
    반환 dict:
        cat   : (n,) 0=Verified(대변) 1=In-domain-only(도메인 내 ref없음) 2=OOD(무관)
        d2    : (n,) full-d Mahalanobis²
        c     : (n,d) 조건부 기여 c_j = (Pe)_j²/P_jj
        uu    : (n,d) 무조건부 기여 u_j = e_j²/Σ_jj
        u2    : (n,K) whitened top-K 좌표
    """
    e = X - p["mu"]
    g = e @ p["Sinv"]                                    # P·e
    d2 = np.einsum("ij,ij->i", e, g)                     # eᵀPe = full-d D²
    c = g ** 2 / p["diagP"]                              # MYT 조건부 기여
    uu = e ** 2 / p["diagS"]                             # 무조건부 기여
    u2 = e @ p["Vk"] / p["sqrt_lamk"]                    # whitened top-K

    cat = np.empty(len(X), dtype=np.int8)
    ood = d2 > p["T"]                                    # Cat3: 타원체 밖
    cat[ood] = 2
    # in-domain 포인트: whitened 셀이 ref 셀에 있으면 Verified, 아니면 in-domain-only
    ind = ~ood
    in_ell = ind & ((u2 ** 2).sum(axis=1) <= p["Tk"])   # top-K 타원 안
    for i in np.where(ind)[0]:
        if not in_ell[i]:
            cat[i] = 1                                   # 타원체엔 있으나 top-K 격자 밖
            continue
        key = _cells_of(u2[i:i + 1], p["internalK"], p["Vk"].shape[1]).pop()
        cat[i] = 0 if key in ref_cells else 1
    return {"cat": cat, "d2": d2, "c": c, "uu": uu, "u2": u2}


def gap_feature_diff(ref_pts, p, sam_cells, names):
    """
    (C) Gap 원인: reservoir Reference 를 top-K 격자로 나눠, Sample 이 못 밟은 셀(gap)에
    속한 ref 포인트 vs 대변된 셀에 속한 ref 포인트의 '원본 feature 평균차'를 구한다.
    반환: (diff (d,), n_gap, n_cov)  — diff>0 이면 gap 쪽이 그 feature 에서 더 큼.
    """
    u = whiten_topk(ref_pts, p)
    in_ell = (u ** 2).sum(axis=1) <= p["Tk"]
    rp, u = ref_pts[in_ell], u[in_ell]
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(u.shape[1]):
        idx[:, a] = np.digitize(u[:, a], p["internalK"])
    keys = [row.tobytes() for row in idx]
    is_gap = np.array([k not in sam_cells for k in keys])
    if is_gap.sum() == 0 or (~is_gap).sum() == 0:
        return np.zeros(N_FEATURES), int(is_gap.sum()), int((~is_gap).sum())
    diff = rp[is_gap].mean(axis=0) - rp[~is_gap].mean(axis=0)
    return diff, int(is_gap.sum()), int((~is_gap).sum())


# =============================================================================
# 시각화 ① whitened top-2 격자 커버리지 + Cat 산점
# =============================================================================
def plot_coverage_2d(u_s, u_r, cat_s, ref_cells, sam_cells, p, sam_cov, out_path):
    dom = p["domain_set"]
    edges = np.linspace(-p["Rk"], p["Rk"], N_BINS_MAHAL + 1)
    # 셀 상태: 0 빈도메인 / 1 gap(ref만) / 2 대변(ref∩sam)
    M = np.full((N_BINS_MAHAL, N_BINS_MAHAL), np.nan)
    for i in range(N_BINS_MAHAL):
        for j in range(N_BINS_MAHAL):
            key = np.array([i, j], dtype=np.int16).tobytes()
            if key not in dom:
                continue
            in_r, in_s = key in ref_cells, key in sam_cells
            M[i, j] = 2 if (in_r and in_s) else (1 if in_r else 0)

    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle, Patch
    cmap = ListedColormap(["#F0F0F0", "#AEC7E8", "#FFD92F"])   # 빈/gap/대변
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.pcolormesh(edges, edges, np.ma.masked_invalid(M).T, cmap=cmap, norm=norm,
                  alpha=0.55, edgecolors="#DDD", linewidth=0.4)
    ax.add_patch(Circle((0, 0), p["Rk"], fill=False, edgecolor="#333", linewidth=1.6))
    # Reference: 투명 회색
    ax.scatter(u_r[:, 0], u_r[:, 1], s=6, c="#999999", alpha=0.20, linewidths=0, label="Reference")
    # Sample: Cat 별 색
    csel = {0: ("#2CA02C", "Sample·대변(Verified)"),
            1: ("#7F7F7F", "Sample·도메인내(ref없음)"),
            2: ("#D62728", "Sample·무관(OOD)")}
    for cval, (color, lab) in csel.items():
        m = cat_s == cval
        if m.any():
            ax.scatter(u_s[m, 0], u_s[m, 1], s=14, c=color, alpha=0.7, linewidths=0, label=lab)

    handles = [Patch(facecolor="#F0F0F0", edgecolor="#BBB", label="빈 도메인(no ref)"),
               Patch(facecolor="#AEC7E8", edgecolor="#BBB", label="Gap(ref만, 미대변)"),
               Patch(facecolor="#FFD92F", edgecolor="#BBB", label="대변 셀(sample cover)")]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9, title="격자 상태")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, title="포인트")
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Mahalanobis Domain Coverage (q={MAHAL_Q})\n"
                 f"Sample Coverage = {sam_cov*100:.2f}%   (√Tk={p['Rk']:.2f})")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ① 커버리지 2D 저장: {out_path}")


# =============================================================================
# 시각화 ② full-d Mahalanobis D² 분포 (Ref vs Sample) + χ²(d) 이론곡선
# =============================================================================
def plot_distribution(d2_ref, d2_sam, p, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    hi = np.percentile(np.concatenate([d2_ref, d2_sam]), 99.5)
    bins = np.linspace(0, hi, 60)
    sns.histplot(d2_ref, bins=bins, stat="density", color="#999999", alpha=0.45,
                 label="Reference", ax=ax, edgecolor=None)
    sns.histplot(d2_sam, bins=bins, stat="density", color="#D62728", alpha=0.40,
                 label="Sample", ax=ax, edgecolor=None)
    xs = np.linspace(1e-6, hi, 400)
    ax.plot(xs, chi2.pdf(xs, N_FEATURES), color="#1F77B4", lw=2.2,
            label=f"χ²(d={N_FEATURES}) 이론")
    ax.axvline(p["T"], color="#333", ls="--", lw=1.6,
               label=f"도메인 경계 T=χ²({MAHAL_Q},d)={p['T']:.1f}")
    ax.set_xlabel("Mahalanobis D² (full-d)"); ax.set_ylabel("density")
    ax.set_title("Reference vs Sample 의 Mahalanobis 분포\n"
                 "(Sample 이 우측으로 치우칠수록 Ref 대비 분포 이탈 큼)")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ② 분포 비교 저장: {out_path}")


# =============================================================================
# 시각화 ③ 로딩 Biplot (PC1·PC2 평면의 원본 feature 방향)
# =============================================================================
def plot_biplot(p, u_s, cat_s, names, out_path):
    # 로딩: 상관 스케일 화살표 = v_j * √λ_j (원본 feature 가 PC 평면에 투영되는 방향·크기)
    load = p["Vk"] * p["sqrt_lamk"]                     # (d, 2)
    scale = 0.9 * p["Rk"] / (np.abs(load).max() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    m = cat_s == 2
    ax.scatter(u_s[~m, 0], u_s[~m, 1], s=10, c="#BBBBBB", alpha=0.35, linewidths=0, label="Sample(in-domain)")
    if m.any():
        ax.scatter(u_s[m, 0], u_s[m, 1], s=16, c="#D62728", alpha=0.6, linewidths=0, label="Sample(OOD)")
    for j in range(N_FEATURES):
        dx, dy = load[j, 0] * scale, load[j, 1] * scale
        ax.arrow(0, 0, dx, dy, color="#1F77B4", alpha=0.8,
                 head_width=0.06 * p["Rk"], length_includes_head=True, linewidth=1.3)
        ax.text(dx * 1.08, dy * 1.08, names[j], color="#0B3D66", fontsize=9,
                ha="center", va="center")
    ax.axhline(0, color="#CCC", lw=0.8); ax.axvline(0, color="#CCC", lw=0.8)
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Loading Biplot — feature 방향 (top-2 EVR={p['evr_k']*100:.1f}%)\n"
                 "OOD(빨강) 이 향한 화살표 = 이탈 방향의 근원 feature")
    ax.set_aspect("equal"); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ③ 로딩 Biplot 저장: {out_path}")


# =============================================================================
# 시각화 ④ OOD 근원 항 랭킹 (조건부 c_j vs 무조건부 u_j)
# =============================================================================
def plot_ood_attribution(cls, names, out_path, topn=15):
    ood = cls["cat"] == 2
    if ood.sum() == 0:
        print("[Plot] ④ OOD 없음 → 근원 항 막대 생략")
        return None
    cond = cls["c"][ood].mean(axis=0)                  # 평균 조건부 기여
    uncond = cls["uu"][ood].mean(axis=0)               # 평균 무조건부 기여
    order = np.argsort(cond)[::-1][:topn]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y + 0.2, cond[order], height=0.4, color="#D62728", label="조건부 c_j (MYT)")
    ax.barh(y - 0.2, uncond[order], height=0.4, color="#F0A0A0", label="무조건부 u_j")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("평균 Mahalanobis² 기여"); ax.set_ylabel("feature")
    ax.set_title(f"OOD 포인트({int(ood.sum())}개) 근원 항 랭킹\n"
                 "c_j 큰 feature = Ref 도메인 이탈의 주범")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ④ OOD 근원 항 저장: {out_path}")
    return [(names[i], float(cond[i])) for i in order[:5]]


# =============================================================================
# 시각화 ⑤ Gap 원인 feature 발산형 막대
# =============================================================================
def plot_gap_attribution(diff, n_gap, n_cov, names, out_path, topn=15):
    if n_gap == 0:
        print("[Plot] ⑤ Gap 없음 → 생략")
        return
    order = np.argsort(np.abs(diff))[::-1][:topn]
    order = order[np.argsort(diff[order])]             # 값 순으로 정렬(발산형 보기 좋게)
    y = np.arange(len(order))
    colors = ["#D62728" if diff[i] > 0 else "#1F77B4" for i in order]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y, diff[order], color=colors)
    ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Gap 셀 평균 - 대변 셀 평균 (scaled)"); ax.set_ylabel("feature")
    ax.set_title(f"Gap 영역 원인 feature  (gap ref={n_gap}, cover ref={n_cov})\n"
                 "빨강=gap 쪽이 큼 / 파랑=대변 쪽이 큼 → sample 이 놓친 방향")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ⑤ Gap 근원 항 저장: {out_path}")


# =============================================================================
# 시각화 ⑥ 극단 OOD top-N × feature 조건부 기여 히트맵
# =============================================================================
def plot_ood_heatmap(cls, names, out_path, topn=OOD_HEATMAP_TOPN):
    ood = np.where(cls["cat"] == 2)[0]
    if len(ood) == 0:
        print("[Plot] ⑥ OOD 없음 → 히트맵 생략")
        return
    ood = ood[np.argsort(cls["d2"][ood])[::-1][:topn]]  # D² 큰 순 top-N
    C = cls["c"][ood]                                   # (topn, d) 조건부 기여
    # 열(feature)은 전체 기여 큰 순으로 정렬해 가독성↑
    fcol = np.argsort(C.sum(axis=0))[::-1]
    C = C[:, fcol]
    fig, ax = plt.subplots(figsize=(min(16, 0.5 * N_FEATURES + 3), max(6, 0.32 * len(ood))))
    sns.heatmap(C, cmap="Reds", ax=ax, cbar_kws={"label": "c_j (조건부 기여)"},
                xticklabels=[names[i] for i in fcol],
                yticklabels=[f"#{i}(D²={cls['d2'][i]:.0f})" for i in ood])
    ax.set_xlabel("feature"); ax.set_ylabel("극단 OOD 포인트")
    ax.set_title(f"극단 OOD top-{len(ood)} 의 feature 별 근원 항(c_j) 히트맵")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ⑥ OOD 히트맵 저장: {out_path}")


# =============================================================================
# 메인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_workers = N_WORKERS_D
    if n_workers is None:
        from common import N_WORKERS
        n_workers = N_WORKERS
    names = FEATURE_COLS

    # Sample 로드 → 공유 스케일러 → 스케일링
    sample = read_feature_matrix(SAMPLE_PATH, fmt="csv")
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    # (1) μ·Σ 스트리밍 추정
    n_ref, sum_x, sum_xx = parallel_reduce_reference(
        REF_PATH, scaler, _moments_map, _moments_reduce, _moments_init,
        n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT)
    mu = sum_x / n_ref
    Sigma = sum_xx / n_ref - np.outer(mu, mu)
    p = build_mahalanobis(mu, Sigma, MAHAL_Q, MAHAL_GRID_DIMS)
    dom = p["domain_set"]
    print(f"[Domain] μ·Σ 추정({n_ref:,} rows), 타원 q={MAHAL_Q} "
          f"(√T_full={np.sqrt(p['T']):.2f}); top-{MAHAL_GRID_DIMS} EVR={p['evr_k']*100:.1f}%, "
          f"|G_total|={len(dom):,} 셀")

    # (2) 커버리지 스트리밍 패스 (Reference)
    p_slim = _slim(p)
    sam_cells, sam_out, sam_tot = mahal_cover_map(sample_scaled, p_slim)
    map_fn = functools.partial(mahal_cover_map, p=p_slim)
    ref_cells, ref_out, ref_tot = parallel_reduce_reference(
        REF_PATH, scaler, map_fn, _domain_reduce, _domain_init,
        n_workers=n_workers, chunksize=CHUNKSIZE, fmt=FMT)
    ref_cov = len(ref_cells & dom) / len(dom)
    sam_cov = len(sam_cells & dom) / len(dom)
    verified = len(sam_cells & ref_cells & dom)         # Cat1 셀
    gap = len(ref_cells & dom) - verified               # Cat2 셀

    # (3) 포인트별 분류(A) + 근원 항(B)  — Sample 전량
    cls = classify_points(sample_scaled, p, ref_cells)
    cat = cls["cat"]
    n_ver, n_ind, n_ood = (cat == 0).sum(), (cat == 1).sum(), (cat == 2).sum()

    # 시각화용 Reference reservoir 표본
    ref_pts = reservoir_sample_reference(REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    d2_ref = np.einsum("ij,jk,ik->i", ref_pts - mu, p["Sinv"], ref_pts - mu)
    u_s, u_r = cls["u2"][:, :2], whiten_topk(ref_pts, p)[:, :2]

    # (4) Gap 원인(C) — reservoir 근사
    gap_diff, n_gap_r, n_cov_r = gap_feature_diff(ref_pts, p, sam_cells, names)

    # 시각화 6종
    j = lambda f: os.path.join(OUTPUT_DIR, f)
    plot_coverage_2d(u_s, u_r, cat, ref_cells, sam_cells, p, sam_cov, j("domain_mahalanobis_2d.png"))
    plot_distribution(d2_ref, cls["d2"], p, j("domain_mahal_distribution.png"))
    plot_biplot(p, u_s, cat, names, j("domain_loadings_biplot.png"))
    top_ood = plot_ood_attribution(cls, names, j("domain_ood_attribution.png"))
    plot_gap_attribution(gap_diff, n_gap_r, n_cov_r, names, j("domain_gap_attribution.png"))
    plot_ood_heatmap(cls, names, j("domain_ood_heatmap.png"))

    # 리포트
    print("\n" + "=" * 68)
    print("  [Mahalanobis Domain Coverage & Attribution]")
    print("=" * 68)
    print(f"  |G_total| (도메인 셀)              : {len(dom):,}")
    print(f"  Reference Domain Coverage          : {ref_cov*100:8.4f}%")
    print(f"  Sample    Domain Coverage          : {sam_cov*100:8.4f}%")
    print(f"  [셀] Cat1 대변={verified:,}  Cat2 gap(미대변)={gap:,}  (of ref {len(ref_cells & dom):,})")
    print(f"  [Sample 포인트] 대변={n_ver:,}  도메인내={n_ind:,}  OOD(무관)={n_ood:,}  "
          f"(총 {len(sample_scaled):,})")
    print(f"  Out-of-Domain 비율  Ref={ref_out/max(ref_tot,1)*100:.2f}%  "
          f"Sample={sam_out/max(sam_tot,1)*100:.2f}%")
    if top_ood:
        print("  OOD 근원 항 top-5 (조건부 c_j)     : "
              + ", ".join(f"{nm}={v:.1f}" for nm, v in top_ood))
    print("=" * 68)
    return {"ref_cov": ref_cov, "sam_cov": sam_cov, "verified": verified, "gap": gap,
            "n_ood": int(n_ood), "top_ood": top_ood}


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
