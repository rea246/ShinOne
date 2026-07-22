# -*- coding: utf-8 -*-
"""
kpc_axis_explain.py
===================
[A1-(a) 증거판] Reference 기준으로 각 Kernel-PCA 축(KPC1..3)을 '정직하게' 설명한다.

목적 (연구 최종 목적에서 역산)
    kPCA 커버리지가 A1(잠재축이 실제 물리 변이를 반영)을 만족하는지의 증거로:
      (1) 분산 질량   : 각 KPC 가 잡는 RKHS 총분산 대비 고유값 비율(축이 신호냐 노이즈냐).
      (2) 축 프로파일 : KPC 를 따라갈 때 원본 feature 가 어떻게 변하나(조건부 평균).
                        → kPCA 는 loading 이 없으므로(RKHS) '사후 관측'으로만 정직하게 해석.
      (3) pseudo-loading : 축↔feature 압축 요약(사후 선형상관; 축 정의가 아님).

    ※ 왜 화살표 biplot 을 안 그리나: kPCA 축은 RKHS 에 있어 원본공간 선형결합이 존재하지 않는다.
      화살표 biplot 은 없는 벡터를 그리는 거짓 해석 → A1(비선형 정당화)과 자기모순.
    ※ 왜 (2) 를 프로파일로: E[feature | KPC bin] 은 선형성을 가정하지 않아 비선형에도 정직하다.
    ※ 확장 훅: (2) 의 'feature' 자리에 나중에 실측오차(EPE)를 끼우면 그대로 A1-(b) 오차 프로파일이
      된다. profile_along_axis() 를 feature/EPE 무관하게 (N,열) 로 받도록 둔 이유.

입력
    coverage_domain_kpca.py 가 만든 Reference 프레임 캐시(.refcache/*.pkl) 재사용.
    → 대용량 CSV 재읽기 없음. 분산 질량만 캐시된 landmark(kpca.X_fit_)로 커널 1회 재구성.

출력 (OUTPUT_DIR)
    kpc_variance_mass.png / kpc_axis_profiles.png / kpc_pseudo_loading.png
    kpc_axis_explain.json  (질량비 + pseudo-loading + 선택 feature)

실행:  python kpc_axis_explain.py
       python kpc_axis_explain.py --selftest   # 합성 캐시로 수치 파이프라인 무결성 자체검증
"""
import argparse
import glob
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import seaborn as sns
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel

# =============================================================================
# CONFIG
# =============================================================================
CACHE_DIR     = ".refcache"
CACHE_PICKLE  = None          # None → CACHE_DIR 최신 *.pkl 자동선택
OUTPUT_DIR    = "coverage_plots"
MASS_MAX_LAND = 3000          # 분산질량 추정 landmark 상한(n×n 커널 메모리/속도)
N_BINS        = 12            # 축 프로파일 KPC 분위 빈 수(분위라 편중분포에 견고)
TOP_FEATURES  = 6            # 프로파일에 그릴 feature 수(|pseudo-loading| 축통합 최대 상위)
SEED          = 0

# 검증된 6색(Okabe-Ito 계열, 저대비 노랑 제외; deutan 근접쌍 비인접 배치).
#   dataviz validate_palette 통과(6–8 floor band) → 라인 우측 '직접 라벨'을 secondary encoding 으로 항상 표기.
FEATURE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#E69F00"]

sns.set_theme(style="white", context="talk")


# =============================================================================
# 캐시 로드 (coverage_domain_kpca.py 의 Reference 프레임 재사용)
# =============================================================================
def load_cache():
    path = CACHE_PICKLE
    if path is None:
        cands = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pkl")), key=os.path.getmtime)
        if not cands:
            raise FileNotFoundError(
                f"{CACHE_DIR}/*.pkl 없음 → 먼저 coverage_domain_kpca.py 를 실행해 Reference 캐시 생성.")
        path = cands[-1]
    with open(path, "rb") as f:
        R = pickle.load(f)
    print(f"[Cache] 로드: {path}  (ref={len(R['ref_raw']):,}, dim={np.shape(R['ref_raw'])[1]})")
    return R


# =============================================================================
# (1) 분산 질량 — RKHS 총분산 대비 KPC1..3 고유값 비율
# =============================================================================
def variance_mass(kpca, gamma, max_land=MASS_MAX_LAND, seed=SEED):
    """
    landmark 로 RBF 커널을 재구성해 '이중 센터링' → 그 행렬의 trace = 모든 고유값 합 = 특징공간 총분산.
    상위 3 고유값 / trace = KPC1..3 의 진짜 질량비.
    (캐시된 kpca 는 n_components=3 이라 고유값 3개뿐 → 분모(총합)가 없어 재구성 필요. 대용량 재읽기는 아님)
    """
    L = np.asarray(kpca.X_fit_, dtype=np.float64)          # KPCA 가 fit 한 표준화 landmark
    if len(L) > max_land:                                   # 왜: n×n 커널 메모리/속도 상한
        idx = np.random.default_rng(seed).choice(len(L), max_land, replace=False)
        L = L[idx]
    K = rbf_kernel(L, gamma=gamma)                          # (n,n) RBF: 대각=1
    rm = K.mean(1, keepdims=True); cm = K.mean(0, keepdims=True)
    Kc = K - rm - cm + K.mean()                            # 이중 센터링
    total = float(np.trace(Kc))                            # = Σ 모든 고유값 = 총분산
    n = len(L)
    w = eigh(Kc, subset_by_index=[n - 3, n - 1], eigvals_only=True)   # 상위 3(오름차순)
    w = np.sort(w)[::-1]
    ratio = np.clip(w / total, 0.0, 1.0) if total > 0 else np.zeros(3)
    return ratio, float(ratio.sum()), n


# =============================================================================
# (3) pseudo-loading — 축↔feature 사후 선형상관 (축 정의가 아님을 명시)
# =============================================================================
def pseudo_loading(Yr, Xs):
    """corr(KPCk, 표준화 feature_i) → (3, d). 비선형 관계는 못 잡으므로 '요약/인덱스' 용도."""
    C = np.corrcoef(np.c_[Yr[:, :3], Xs], rowvar=False)
    return C[:3, 3:]


def select_features(L, k):
    """축을 통틀어 |상관| 최대인 feature 상위 k(프로파일에 그릴 대상). 색은 이 순서로 고정 배정."""
    return np.argsort(np.abs(L).max(0))[::-1][:k]


# =============================================================================
# (2) 축 프로파일 — 축을 따라가며 열(지금=feature, 나중=EPE)의 조건부 평균
# =============================================================================
def profile_along_axis(axis_scores, Y, n_bins=N_BINS):
    """
    axis_scores(N,) 분위 빈 → 각 열의 조건부 평균. 반환 (bin_center_x(nb,), means(nb, ncol)).
    Y 는 (N, ncol): 열이 무엇이든 무관 → 'feature 자리에 EPE' 확장 훅(A1-b).
    """
    z = np.asarray(axis_scores, dtype=np.float64)
    edges = np.quantile(z, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    b = np.clip(np.digitize(z, edges) - 1, 0, n_bins - 1)
    xc = np.full(n_bins, np.nan)
    M = np.full((n_bins, Y.shape[1]), np.nan)
    for i in range(n_bins):
        m = b == i
        if m.any():
            xc[i] = z[m].mean()
            M[i] = Y[m].mean(0)
    return xc, M


# =============================================================================
# 시각화
# =============================================================================
def plot_variance_mass(ratio, cum, n, out):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(3)
    ax.barh(y, ratio * 100, color="#5B8DEF", height=0.62)
    ax.set_yticks(y); ax.set_yticklabels(["KPC1", "KPC2", "KPC3"]); ax.invert_yaxis()
    for yi, r in zip(y, ratio):
        ax.text(r * 100 + 0.4, yi, f"{r * 100:.1f}%", va="center", fontsize=12, color="#333")
    ax.set_xlabel("mass share of RKHS variance (%)")
    ax.set_title(f"Kernel-PCA axis variance mass  "
                 f"(top-3 cumulative {cum * 100:.1f}%, landmarks n={n:,})", fontsize=12)
    sns.despine(ax=ax)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[Plot] variance mass 저장: {out}")


def plot_axis_profiles(Yr, Xs, names, sel, colors, n_bins, out):
    sel_names = [names[i] for i in sel]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for a, ax in enumerate(axes):
        xc, M = profile_along_axis(Yr[:, a], Xs[:, sel], n_bins)
        for jx, nm in enumerate(sel_names):
            c = colors[jx % len(colors)]
            ax.plot(xc, M[:, jx], color=c, lw=2.0, marker="o", ms=4)
            if np.isfinite(M[-1, jx]) and np.isfinite(xc[-1]):        # 직접 라벨=secondary encoding
                ax.annotate(nm, (xc[-1], M[-1, jx]), color=c, fontsize=9,
                            xytext=(4, 0), textcoords="offset points",
                            va="center", fontweight="bold")
        ax.axhline(0, color="#DDD", lw=1.0)                           # 표준화 기준선
        ax.set_xlabel(f"KPC{a + 1} (bin mean)")
        ax.set_title(f"KPC{a + 1}")
        sns.despine(ax=ax)
    axes[0].set_ylabel("conditional mean (standardized feature)")
    fig.suptitle("Per-axis explanation (REF): conditional feature mean along each KPC\n"
                 "kPCA has no loadings -> post-hoc; swap feature for EPE later = A1-b error profile",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[Plot] axis profiles 저장: {out}  (features={sel_names})")


def _diverging_gray():
    # dataviz: diverging = 두 hue + '중립 회색' 중점(무지개 금지, 중점에 hue 금지)
    return LinearSegmentedColormap.from_list("blue_gray_red", ["#2166AC", "#8C8C8C", "#B2182B"])


def plot_pseudo_loading(L, names, out):
    d = L.shape[1]; mx = float(np.abs(L).max()) or 1.0
    fig, ax = plt.subplots(figsize=(max(9, 0.5 * d), 3.6))
    im = ax.imshow(L, cmap=_diverging_gray(), vmin=-mx, vmax=mx, aspect="auto")
    ax.set_yticks(range(3)); ax.set_yticklabels(["KPC1", "KPC2", "KPC3"])
    ax.set_xticks(range(d)); ax.set_xticklabels(names, rotation=90, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="corr (post-hoc, signed)")
    ax.set_title("Pseudo-loading: axis <-> original feature (post-hoc correlation)\n"
                 "NOT the axis definition; nonlinear relations are not captured", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    print(f"[Plot] pseudo-loading 저장: {out}")


# =============================================================================
# 실행
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    R = load_cache()
    ref_raw = np.asarray(R["ref_raw"], dtype=np.float64)
    Xs = (ref_raw - R["center"]) / R["scale"]
    Yr = np.asarray(R["Yr"], dtype=np.float64)[:, :3]
    names = list(R["names"])

    ratio, cum, n_mass = variance_mass(R["kpca"], R["gamma"])
    L = pseudo_loading(Yr, Xs)
    sel = select_features(L, min(TOP_FEATURES, len(names)))

    j = lambda f: os.path.join(OUTPUT_DIR, f)
    plot_variance_mass(ratio, cum, n_mass, j("kpc_variance_mass.png"))
    plot_axis_profiles(Yr, Xs, names, sel, FEATURE_COLORS, N_BINS, j("kpc_axis_profiles.png"))
    plot_pseudo_loading(L, names, j("kpc_pseudo_loading.png"))

    out = {
        "variance_mass_pct": {f"KPC{i + 1}": round(float(ratio[i]) * 100, 3) for i in range(3)},
        "cumulative_top3_pct": round(cum * 100, 3),
        "mass_landmarks": n_mass,
        "feature_names": names,
        "pseudo_loading": np.round(L, 4).tolist(),
        "selected_features": [names[i] for i in sel],
    }
    with open(j("kpc_axis_explain.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"[Done] KPC 질량 {[f'{r * 100:.1f}%' for r in ratio]}  누적 {cum * 100:.1f}%")
    print(f"[Done] 프로파일 feature: {[names[i] for i in sel]}")
    print(f"[Out] {OUTPUT_DIR}/  (variance_mass / axis_profiles / pseudo_loading .png + .json)")


# =============================================================================
# 자체검증 — 합성 캐시로 수치 파이프라인 무결성 확인(실데이터/실캐시 불필요)
# =============================================================================
def demo():
    from sklearn.decomposition import KernelPCA
    from sklearn.metrics.pairwise import euclidean_distances
    rng = np.random.default_rng(0)
    d, n = 8, 1500
    t = rng.normal(size=(n, 2))                            # 잠재 2요인
    X = np.c_[t[:, 0], t[:, 0] ** 2, np.sin(t[:, 1]), t[:, 1],
              rng.normal(scale=0.3, size=(n, d - 4))]      # 비선형 전개 → 축이 feature 를 대변하게 심음
    center = X.mean(0); scale = X.std(0); scale[scale == 0] = 1.0
    Xs = (X - center) / scale
    sub = Xs[:500]
    med = np.median(euclidean_distances(sub, sub)[np.triu_indices(len(sub), 1)] ** 2)
    gamma = 1.0 / (2 * med)
    kpca = KernelPCA(n_components=3, kernel="rbf", gamma=gamma,
                     eigen_solver="arpack", random_state=0).fit(Xs[:800])
    Yr = kpca.transform(Xs)

    ratio, cum, nm = variance_mass(kpca, gamma)
    assert ratio.shape == (3,) and 0.0 <= ratio.min() and ratio.max() <= 1.0, ratio
    assert 0.0 < cum <= 1.0 + 1e-6, cum
    L = pseudo_loading(Yr, Xs)
    assert L.shape == (3, d) and np.abs(L).max() <= 1.0 + 1e-9
    sel = select_features(L, 4)
    assert len(sel) == 4 and len(set(sel.tolist())) == 4
    xc, M = profile_along_axis(Yr[:, 0], Xs[:, sel], 10)
    assert xc.shape == (10,) and M.shape == (10, 4)
    # 심은 구조상 축을 따라 feature 가 변해야 함(평평하면 축-feature 관계 소실 = 버그)
    assert np.nanmax(np.nanstd(M, axis=0)) > 0.05, "프로파일이 평평 — 축 설명력 소실?"
    print(f"[selftest] OK  mass={[round(float(x), 3) for x in ratio]} "
          f"cum={cum:.3f} L{L.shape} sel={sel.tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="합성 캐시로 무결성 자체검증")
    args = ap.parse_args()
    demo() if args.selftest else run()
