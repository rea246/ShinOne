# -*- coding: utf-8 -*-
"""
coverage_domain_kpca.py
=======================
[비선형 잠재공간 커버리지 진단] RBF Kernel PCA + 3D KDE-99%-HDR

목적
    RBF Kernel PCA 잠재공간에서 Reference 대비 Sample 분포를 해석해, 선형 PCA 의
    '가짜 커버리지(착시)'를 제거한 진짜 공백(True Gap)을 정량화·시각화한다.

설계 (coverage_domain_mahal 의 배경 계승)
    * 입력 인터페이스 동일: REF/SAMPLE 경로 + 21개 피처(이름/위치 선택).
    * Reference 고정 프레임: KPCA 를 Reference reservoir 에 '한 번' fit → 캐시 →
      sample1/sample2 를 같은 잠재공간에 투영(비교 일관). Sample 은 투영만.
    * reservoir 근사: O(N²) 커널을 피하려 Reference 표본(landmark)으로 fit·커버리지.
    * γ 자동추정: median heuristic  γ = 1 / (2 · median||xi−xj||²)  (표준화 공간).
    * 지표: 3D(KP1-3) 잠재에서 KDE 99% HDR → cell 기반 True Coverage / True Gap.
    * 선형 vs 비선형 착시: Δ_illusion = Coverage_linearPCA − Coverage_KernelPCA.

산출물
    kpca_coverage_2d.png / kpca_coverage_3d.png
    kpca_vs_linear_pca_comparison.png
    kpca_summary_metrics.json
    uncovered_kpca_gap_patterns.csv   (원본 21-d + KP1-3 + density_gap_score)

실행:  python coverage_domain_kpca.py   (pip install scikit-learn scipy seaborn matplotlib pandas)
"""

import hashlib
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection 등록)
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA, KernelPCA
from sklearn.neighbors import KernelDensity

# =============================================================================
# CONFIG
# =============================================================================
SAMPLE_PATH   = "sample.csv"
REF_PATH      = "reference.csv"
FMT           = "csv"

FEATURE_COLS    = None
FEATURE_COL_IDX = list(range(0, 21))
EXCLUDE_COLS    = []

SCALE_METHOD  = "standard"     # 'standard' | 'minmax'

# ---- Kernel PCA ----
N_LANDMARKS   = 5000           # KPCA fit 표본(O(N²) 관리)
N_REF         = 15000          # 커버리지/플롯용 Reference reservoir (landmark 로 투영)
N_COMPONENTS  = 3              # 비선형 주성분 수 (KP1..KP3)
GAMMA         = None           # None → median heuristic 자동추정, 값 지정 시 그대로

# ---- KDE 99% HDR / 커버리지 ----
HDR_Q         = 0.99           # HDR 질량비 (99%) — Ref/Sample 경계 공통
KDE_BW        = None           # gaussian_kde bw_method (None='scott', 실수=배율)
# 커버리지 = Reference 99% HDR 안에서 'Sample 밀도가 존재(=Sample 99% HDR 안)'하는 비율.
#   Sample KDE 로 판정 → 점 수 차이에 불변(성긴 sample 도 공정 비교).

CHUNKSIZE     = 200_000
OUTPUT_DIR    = "coverage_plots"
RESERVOIR_SEED = 0
USE_CACHE     = True
CACHE_DIR     = ".refcache"

sns.set_theme(style="white", context="talk")
C_REF = "#8C8C8C"; C_SAMPLE = "#2CA02C"; C_GAP = "#F4511E"


# =============================================================================
# 데이터 I/O (이름/위치 기준 feature 선택 + 비유한 제거) — self-contained
# =============================================================================
def _header(path, fmt):
    if fmt == "csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
    else:
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(path).schema_arrow.names)
    return [str(c).strip() for c in cols]


def resolve_feature_names(sample_path, ref_path, fmt):
    s_cols, r_cols = _header(sample_path, fmt), _header(ref_path, fmt)
    if FEATURE_COLS is not None:
        wanted = set(str(c).strip() for c in FEATURE_COLS)
    else:
        bad = [i for i in FEATURE_COL_IDX if i < 0 or i >= len(s_cols)]
        if bad:
            raise ValueError(f"FEATURE_COL_IDX 범위 초과: {bad} (Sample 열 {len(s_cols)})")
        wanted = set(s_cols[i] for i in FEATURE_COL_IDX)
    ex = {str(c).strip() for c in EXCLUDE_COLS}
    s_set, r_set = set(s_cols), set(r_cols)
    names = [c for c in r_cols if c in wanted and c in s_set and c not in ex]
    dropped = [c for c in wanted if c not in set(names)]
    if dropped:
        print(f"[IO] 제외된 컬럼: {sorted(dropped)}")
    if not names:
        raise ValueError("공통 feature 없음.")
    print(f"[IO] feature {len(names)}개(Reference 순서): {names}")
    return names


def _select_raw(df, names):
    df.columns = [str(c).strip() for c in df.columns]
    X = df.loc[:, names].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    return X[finite]


def read_matrix(path, names, fmt):
    want = set(names)
    if fmt == "csv":
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in want)
    else:
        import pyarrow.parquet as pq
        raw = [c for c in pq.ParquetFile(path).schema_arrow.names if str(c).strip() in want]
        df = pq.read_table(path, columns=raw).to_pandas()
    X = _select_raw(df, names)
    if len(X) == 0:
        raise ValueError(f"{path}: 유효행 없음")
    return X


def iter_ref_chunks_raw(path, names, chunksize, fmt):
    want = set(names)
    if fmt == "csv":
        for ch in pd.read_csv(path, usecols=lambda c: str(c).strip() in want, chunksize=chunksize):
            X = _select_raw(ch, names)
            if len(X):
                yield X
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = [c for c in pf.schema_arrow.names if str(c).strip() in want]
        for b in pf.iter_batches(batch_size=chunksize, columns=raw):
            X = _select_raw(b.to_pandas(), names)
            if len(X):
                yield X


def reservoir_raw(path, names, k, chunksize, fmt, seed=0):
    rng = np.random.default_rng(seed)
    d = len(names); res = np.empty((k, d)); filled = seen = 0
    for X in iter_ref_chunks_raw(path, names, chunksize, fmt):
        if filled < k:
            take = min(k - filled, len(X)); res[filled:filled + take] = X[:take]
            filled += take; seen += take; X = X[take:]
            if len(X) == 0:
                continue
        t = seen + np.arange(1, len(X) + 1); acc = rng.random(len(X)) < (k / t)
        if acc.any():
            res[rng.integers(0, k, acc.sum())] = X[acc]
        seen += len(X)
    print(f"[Reservoir] 전체 {seen:,} → {filled:,} 추출")
    return res[:filled]


# =============================================================================
# 표준화 / 격자 / 커버리지 헬퍼
# =============================================================================
def fit_scaler(X, method):
    if method == "minmax":
        center = X.min(0); scale = X.max(0) - X.min(0)
    else:
        center = X.mean(0); scale = X.std(0)
    scale[scale == 0] = 1.0
    return center, scale


def median_gamma(Xs, seed=0, cap=2000):
    """median heuristic: γ = 1/(2·median||xi−xj||²) (표준화 표본에서)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xs), size=min(cap, len(Xs)), replace=False)
    d2 = pdist(Xs[idx], metric="sqeuclidean")
    med = np.median(d2)
    return float(1.0 / (2.0 * med)) if med > 0 else 1.0 / Xs.shape[1]


def _bw(n, d):
    return float(n ** (-1.0 / (d + 4)))                    # Scott 유사 bandwidth (표준화 좌표)


def kde_fit_dens(Yn, bw=None):
    """참조-표준화 좌표 Yn 에 KernelDensity 학습 + 자기 밀도 반환 (특이행렬에 강건)."""
    b = KDE_BW if KDE_BW is not None else (bw or _bw(len(Yn), Yn.shape[1]))
    kd = KernelDensity(kernel="gaussian", bandwidth=b).fit(Yn)
    return kd, np.exp(kd.score_samples(Yn))


def coverage_metrics(Yr, Ys, ystd, dens_r, tau_r):
    """
    밀도 기반 True Coverage: HDR(99%) 안 Reference 포인트 중, 'Sample 밀도가 존재'
    (= Sample 99% HDR 안)하는 비율. Sample KDE 로 판정하므로 점 수 차이에 불변.
    """
    ref_hdr = dens_r >= tau_r
    idx = np.where(ref_hdr)[0]
    gap_mask = np.zeros(len(Yr), dtype=bool)
    if len(Ys) < 5 or len(idx) == 0:
        gap_mask[idx] = True
        return {"true_cov": 0.0, "true_gap": 1.0, "n_ref_hdr": int(len(idx)),
                "n_covered": 0, "gap_mask": gap_mask}
    Yr_n, Ys_n = Yr / ystd, Ys / ystd                      # 공통(참조) 표준화 좌표
    skde, dens_s = kde_fit_dens(Ys_n)
    tau_s = float(np.quantile(dens_s, 1 - HDR_Q))          # Sample 99% HDR 임계
    dens_s_ref = np.exp(skde.score_samples(Yr_n[idx]))     # ref 포인트에서의 sample 밀도
    covered = dens_s_ref >= tau_s
    gap_mask[idx[~covered]] = True
    tc = float(covered.mean())
    return {"true_cov": tc, "true_gap": 1 - tc, "n_ref_hdr": int(len(idx)),
            "n_covered": int(covered.sum()), "gap_mask": gap_mask}


# =============================================================================
# Reference 고정 프레임 구축 (KPCA + 선형PCA + KDE-HDR) — 캐시
# =============================================================================
def _cache_key(ref_path, names, fmt):
    st = os.stat(ref_path)
    key = {"ref": os.path.abspath(ref_path), "size": st.st_size, "mtime": int(st.st_mtime),
           "names": names, "fmt": fmt, "scale": SCALE_METHOD, "nl": N_LANDMARKS, "nref": N_REF,
           "nc": N_COMPONENTS, "gamma": GAMMA, "hdrq": HDR_Q, "bw": KDE_BW,
           "seed": RESERVOIR_SEED, "v": 3}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def build_reference(ref_path, names, fmt):
    key = _cache_key(ref_path, names, fmt)
    cpath = os.path.join(CACHE_DIR, f"kpca_{key}.pkl")
    if USE_CACHE and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            print(f"[Cache] KPCA Reference 프레임 재사용: {cpath}")
            return pickle.load(f)

    print("[Cache] 없음 → Reference reservoir fit")
    ref_raw = reservoir_raw(ref_path, names, N_REF, CHUNKSIZE, FMT, RESERVOIR_SEED)
    center, scale = fit_scaler(ref_raw, SCALE_METHOD)
    Xs = (ref_raw - center) / scale
    landmarks = Xs[:min(N_LANDMARKS, len(Xs))]              # fit 표본
    gamma = GAMMA if GAMMA is not None else median_gamma(landmarks, RESERVOIR_SEED)
    print(f"[KPCA] landmarks={len(landmarks):,}, gamma={gamma:.4g}, n_comp={N_COMPONENTS}")

    kpca = KernelPCA(n_components=N_COMPONENTS, kernel="rbf", gamma=gamma,
                     eigen_solver="arpack", random_state=RESERVOIR_SEED)
    kpca.fit(landmarks)
    Yr = kpca.transform(Xs)                                 # (N_REF, 3) 비선형 잠재
    pca = PCA(n_components=N_COMPONENTS).fit(Xs)
    Yr_lin = pca.transform(Xs)                              # 선형 잠재(비교용)

    # KDE-HDR (각 공간): 참조-표준화 좌표에서 Reference 99% 경계 임계 τ
    def kde_tau(Y):
        ystd = Y.std(0); ystd[ystd == 0] = 1.0
        _, dens = kde_fit_dens(Y / ystd)
        return dens, float(np.quantile(dens, 1 - HDR_Q)), ystd
    dens_k, tau_k, ystd_k = kde_tau(Yr)
    dens_l, tau_l, ystd_l = kde_tau(Yr_lin)

    obj = {"ref_raw": ref_raw, "center": center, "scale": scale, "gamma": gamma,
           "kpca": kpca, "pca": pca, "Yr": Yr, "Yr_lin": Yr_lin,
           "dens_k": dens_k, "tau_k": tau_k, "ystd_k": ystd_k,
           "dens_l": dens_l, "tau_l": tau_l, "ystd_l": ystd_l, "names": names}
    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump(obj, f)
        print(f"[Cache] 저장: {cpath}")
    return obj


# =============================================================================
# 시각화
# =============================================================================
def plot_2d(Yr, Ys, gap_mask, cov, out_path):
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.scatter(Yr[:, 0], Yr[:, 1], s=6, c=C_REF, alpha=0.15, linewidths=0, label="Reference")
    ax.scatter(Ys[:, 0], Ys[:, 1], s=14, c=C_SAMPLE, alpha=0.6, linewidths=0, label="Sample")
    if gap_mask.any():
        ax.scatter(Yr[gap_mask, 0], Yr[gap_mask, 1], s=18, c=C_GAP, alpha=0.8, linewidths=0,
                   label="True Gap (ref, sample-unreached)")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    ax.set_title(f"Nonlinear (Kernel PCA) coverage\nTrue Coverage={cov*100:.1f}%  "
                 f"True Gap={(1-cov)*100:.1f}%")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] 2D 저장: {out_path}")


def plot_3d(Yr, Ys, gap_mask, out_path):
    fig = plt.figure(figsize=(10, 9)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Yr[:, 0], Yr[:, 1], Yr[:, 2], s=4, c=C_REF, alpha=0.12, linewidths=0, label="Reference")
    ax.scatter(Ys[:, 0], Ys[:, 1], Ys[:, 2], s=12, c=C_SAMPLE, alpha=0.6, linewidths=0, label="Sample")
    if gap_mask.any():
        ax.scatter(Yr[gap_mask, 0], Yr[gap_mask, 1], Yr[gap_mask, 2], s=16, c=C_GAP, alpha=0.85,
                   linewidths=0, label="True Gap")
    ax.set_xlabel("KP1"); ax.set_ylabel("KP2"); ax.set_zlabel("KP3")
    ax.set_title("Kernel PCA latent (KP1-3): Reference / Sample / True Gap")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] 3D 저장: {out_path}")


def plot_comparison(Yl_r, Yl_s, cov_l, Yk_r, Yk_s, cov_k, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    for ax, (Yr, Ys, cov, ttl, xl) in zip(axes, [
            (Yl_r, Yl_s, cov_l, "Linear PCA", "PC"), (Yk_r, Yk_s, cov_k, "Kernel PCA", "KP")]):
        ax.scatter(Yr[:, 0], Yr[:, 1], s=6, c=C_REF, alpha=0.15, linewidths=0)
        ax.scatter(Ys[:, 0], Ys[:, 1], s=14, c=C_SAMPLE, alpha=0.6, linewidths=0)
        ax.set_xlabel(f"{xl}1"); ax.set_ylabel(f"{xl}2")
        ax.set_title(f"{ttl}\nTrue Coverage = {cov*100:.1f}%")
    fig.suptitle(f"Linear vs Kernel PCA — fake-coverage illusion  Δ = {(cov_l-cov_k)*100:+.1f}%p",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] comparison 저장: {out_path}")


# =============================================================================
# 메인
# =============================================================================
def run():
    for pth in (SAMPLE_PATH, REF_PATH):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"입력 없음: {pth}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    names = resolve_feature_names(SAMPLE_PATH, REF_PATH, FMT)
    R = build_reference(REF_PATH, names, FMT)

    # Sample 투영 (동일 프레임)
    sample_raw = read_matrix(SAMPLE_PATH, names, FMT)
    Xs_s = (sample_raw - R["center"]) / R["scale"]
    Ys = R["kpca"].transform(Xs_s)                 # 비선형 잠재
    Ys_lin = R["pca"].transform(Xs_s)              # 선형 잠재

    # 커버리지 (KPCA vs 선형)
    mk = coverage_metrics(R["Yr"], Ys, R["ystd_k"], R["dens_k"], R["tau_k"])
    ml = coverage_metrics(R["Yr_lin"], Ys_lin, R["ystd_l"], R["dens_l"], R["tau_l"])
    delta = ml["true_cov"] - mk["true_cov"]        # 착시(가짜 커버리지) 크기

    # True Gap Reference 포인트 + density_gap_score (KPCA 공간)
    gap_mask = mk["gap_mask"]
    gap_pts = R["Yr"][gap_mask]
    if gap_mask.any() and len(Ys):
        nn_d, _ = cKDTree(Ys).query(gap_pts, k=1)   # 가장 가까운 sample 까지 거리
    else:
        nn_d = np.zeros(int(gap_mask.sum()))
    gap_score = R["dens_k"][gap_mask] * nn_d        # 밀집·먼 gap = 고우선순위
    gap_df = pd.DataFrame(R["ref_raw"][gap_mask], columns=names)
    gap_df["KP1"], gap_df["KP2"], gap_df["KP3"] = gap_pts[:, 0], gap_pts[:, 1], gap_pts[:, 2]
    gap_df["density_gap_score"] = gap_score
    gap_df = gap_df.sort_values("density_gap_score", ascending=False)
    gap_df.to_csv(j("uncovered_kpca_gap_patterns.csv"), index=False)

    # 시각화
    plot_2d(R["Yr"], Ys, gap_mask, mk["true_cov"], j("kpca_coverage_2d.png"))
    plot_3d(R["Yr"], Ys, gap_mask, j("kpca_coverage_3d.png"))
    plot_comparison(R["Yr_lin"], Ys_lin, ml["true_cov"], R["Yr"], Ys, mk["true_cov"],
                    j("kpca_vs_linear_pca_comparison.png"))

    # 지표 JSON
    summary = {
        "gamma": R["gamma"], "n_landmarks": min(N_LANDMARKS, len(R["ref_raw"])),
        "n_ref": len(R["ref_raw"]), "n_components": N_COMPONENTS, "hdr_q": HDR_Q,
        "kde_bw": KDE_BW,
        "kpca_true_coverage_pct": round(mk["true_cov"] * 100, 3),
        "kpca_true_gap_pct": round(mk["true_gap"] * 100, 3),
        "linear_true_coverage_pct": round(ml["true_cov"] * 100, 3),
        "delta_illusion_pct": round(delta * 100, 3),
        "n_ref_hdr_kpca": mk["n_ref_hdr"],
        "n_covered_kpca": mk["n_covered"], "n_gap_ref_points": int(gap_mask.sum()),
        "n_sample": len(sample_raw),
    }
    with open(j("kpca_summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print("  [Kernel PCA Nonlinear Coverage]")
    print("=" * 66)
    print(f"  gamma(median heuristic)   : {R['gamma']:.4g}")
    print(f"  Kernel PCA True Coverage  : {mk['true_cov']*100:7.2f}%   (True Gap {mk['true_gap']*100:.2f}%)")
    print(f"  Linear PCA True Coverage  : {ml['true_cov']*100:7.2f}%")
    print(f"  >>> Δ_illusion (가짜커버)  : {delta*100:+7.2f}%p   (선형이 과대평가한 정도)")
    print(f"  True Gap Reference 포인트 : {int(gap_mask.sum()):,}개 → uncovered_kpca_gap_patterns.csv")
    print("=" * 66)
    return summary


if __name__ == "__main__":
    run()
