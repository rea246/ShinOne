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
    * 도메인: 3D(KP1-3) 잠재에서 Reference 99% HDR(희소 outlier 제외).
    * 커버리지(reach 기반, sample 밀도 미사용): Reference 지점 근처 반경 R 안에 sample
      패턴이 하나라도 있으면 '대표됨'. R 은 예산(패턴 수)이 정함(성긴 sample 벌점 없음).
    * 중요도: Reference 밀도 큰 gap 을 우선(중요 gap 그룹).
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

# ★ numpy/scipy import 전에 BLAS 스레드 제한 (OpenBLAS 오버서브스크립션 → hang/segfault 방지)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection 등록)
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
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
# gap/scatter CSV 에 함께 실을 '식별 열'(ref.csv 매칭용). 매칭에 필요한 열만 지정 권장.
#   예) ["lot_id"].   []=식별열 없음.
#   ⚠️ None=모든 비-feature 열을 읽음 → 열 많은/큰 reference 에선 메모리 폭증·크래시 위험(비권장).
KEEP_COLS       = ["lot_id"]

SCALE_METHOD  = "standard"     # 'standard' | 'minmax'

# ---- Kernel PCA ----
N_LANDMARKS   = 5000           # KPCA fit 표본(O(N²) 관리)
N_REF         = 15000          # 커버리지/플롯용 Reference reservoir (landmark 로 투영)
N_COMPONENTS  = 3              # 비선형 주성분 수 (KP1..KP3)
GAMMA         = None           # None → median heuristic 자동추정, 값 지정 시 그대로

# ---- 도메인(Reference HDR) ----
HDR_Q         = 0.99           # Reference 99% HDR (희소 outlier 1% 제외한 유효 도메인)
KDE_BW        = None           # Reference KernelDensity bandwidth (None='scott 유사')
# ---- 커버리지 = reach 기반 (sample 밀도 미사용) ----
#   reference 지점 근처(반경 R)에 sample 패턴이 하나라도 있으면 '대표됨(covered)'.
#   R = 대표 반경. '예산(패턴 수)'이 정함: 패턴 하나가 ceil(N_ref/N_sample)개를 대표해야 하므로
#       R = Reference 의 그 k번째 최근접이웃 거리 중앙값 × R_MULT.  sample 이 성겨도 벌점 없음.
R_MULT        = 1.0            # 대표 반경 허용오차 배율(공학적 tolerance). ↑ 이면 커버리지 완화.
# ---- 중요 Gap 그룹 (Reference 밀도 큰 gap 만 추려 군집) ----
GAP_IMPORTANT_Q = 0.5          # Reference 밀도 상위(1-Q) 만 '중요 gap' (0.5 = 상위 50%)
N_GAP_GROUPS    = 8            # 중요 gap 을 KMeans 로 묶을 그룹 수

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


def _keep_cols(path, names, fmt):
    """gap/scatter CSV 에 실을 비-feature(식별) 열 목록."""
    hdr = _header(path, fmt); nset = set(names)
    if KEEP_COLS is None:
        extra = [c for c in hdr if c not in nset]
        print(f"[IO] ⚠️ KEEP_COLS=None → 비-feature 열 {len(extra)}개 전부 읽음(메모리 위험). "
              f"큰 데이터면 KEEP_COLS=['lot_id'] 처럼 필요한 열만 지정 권장.")
        return extra
    keep = [str(c).strip() for c in KEEP_COLS if str(c).strip() in set(hdr) and str(c).strip() not in nset]
    missing = [str(c).strip() for c in KEEP_COLS if str(c).strip() not in set(hdr)]
    if missing:
        print(f"[IO] 경고: KEEP_COLS 중 Reference 에 없는 열 제외: {missing}")
    return keep


def reservoir_full(path, names, keep, k, chunksize, fmt, seed=0):
    """
    Reference 를 reservoir 로 k행 추출하되, feature 값(resX) 과 식별 열(resK) 을 함께 보존.
    (feature 유한행만 대상, 동일 샘플링 결정으로 두 배열을 같이 갱신)
    """
    rng = np.random.default_rng(seed)
    d, nk = len(names), len(keep)
    want = set(names) | set(keep)
    resX = np.empty((k, d)); resK = np.empty((k, nk), dtype=object)
    filled = seen = 0

    def chunks():
        if fmt == "csv":
            for ch in pd.read_csv(path, usecols=lambda c: str(c).strip() in want, chunksize=chunksize):
                yield ch
        else:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            raw = [c for c in pf.schema_arrow.names if str(c).strip() in want]
            for b in pf.iter_batches(batch_size=chunksize, columns=raw):
                yield b.to_pandas()

    for ch in chunks():
        ch.columns = [str(c).strip() for c in ch.columns]
        X = ch.loc[:, names].to_numpy(dtype=np.float64)
        finite = np.isfinite(X).all(axis=1)
        X = X[finite]
        Kv = ch.loc[:, keep].to_numpy(dtype=object)[finite] if nk else np.empty((len(X), 0), object)
        if len(X) == 0:
            continue
        if filled < k:
            take = min(k - filled, len(X))
            resX[filled:filled + take] = X[:take]; resK[filled:filled + take] = Kv[:take]
            filled += take; seen += take; X = X[take:]; Kv = Kv[take:]
            if len(X) == 0:
                continue
        t = seen + np.arange(1, len(X) + 1); acc = rng.random(len(X)) < (k / t)
        if acc.any():
            pos = rng.integers(0, k, acc.sum())
            resX[pos] = X[acc]; resK[pos] = Kv[acc]
        seen += len(X)
    print(f"[Reservoir] 전체 {seen:,} → {filled:,} 추출 (식별열 {nk}개 보존)")
    return resX[:filled], resK[:filled]


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


def representation_radius(Yr, ystd, n_sample, mult):
    """
    대표 반경 R 을 '예산(패턴 수)'으로 정한다. 패턴 하나가 평균 k=ceil(N_ref/N_sample)개를
    대표해야 하므로, R = Reference 의 k번째 최근접이웃 거리 중앙값 × mult (표준화 좌표).
    """
    Yn = Yr / ystd
    k = int(max(1, round(len(Yn) / max(n_sample, 1))))
    k = min(k, len(Yn) - 1)
    dd, _ = cKDTree(Yn).query(Yn, k=k + 1)                 # [:,0]=self
    return float(np.median(dd[:, k]) * mult), k


def coverage_metrics(Yr, Ys, ystd, dens_r, tau_r, R):
    """
    reach 기반 True Coverage (sample 밀도 미사용):
    Reference 99% HDR 안 지점 중, '최근접 sample 패턴 거리 <= R' 인 비율.
    gap = R 안에 sample 이 없는(=대표 못 된) Reference 지점.
    """
    ref_hdr = dens_r >= tau_r
    idx = np.where(ref_hdr)[0]
    gap_mask = np.zeros(len(Yr), dtype=bool)
    if len(Ys) == 0 or len(idx) == 0:
        gap_mask[idx] = True
        return {"true_cov": 0.0, "true_gap": 1.0, "n_ref_hdr": int(len(idx)),
                "n_covered": 0, "gap_mask": gap_mask}
    Yr_n, Ys_n = Yr / ystd, Ys / ystd                      # 등방(표준화) 좌표에서 거리
    nn_d, _ = cKDTree(Ys_n).query(Yr_n[idx], k=1)          # 최근접 sample 거리
    covered = nn_d <= R
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
           "keep": KEEP_COLS, "seed": RESERVOIR_SEED, "v": 4}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def build_reference(ref_path, names, fmt):
    key = _cache_key(ref_path, names, fmt)
    cpath = os.path.join(CACHE_DIR, f"kpca_{key}.pkl")
    if USE_CACHE and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            print(f"[Cache] KPCA Reference 프레임 재사용: {cpath}")
            return pickle.load(f)

    print("[Cache] 없음 → Reference reservoir fit")
    keep = _keep_cols(ref_path, names, fmt)
    ref_raw, ref_keep = reservoir_full(ref_path, names, keep, N_REF, CHUNKSIZE, FMT, RESERVOIR_SEED)
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

    obj = {"ref_raw": ref_raw, "ref_keep": ref_keep, "keep_cols": keep,
           "center": center, "scale": scale, "gamma": gamma,
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


def plot_gap_groups(Yr, Ys, imp_mask, groups, gsize, out_path):
    """중요 gap 그룹을 KP1-2 에 색·크기(그룹 밀도)로 표시."""
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.scatter(Yr[:, 0], Yr[:, 1], s=5, c=C_REF, alpha=0.12, linewidths=0, label="Reference")
    ax.scatter(Ys[:, 0], Ys[:, 1], s=10, c=C_SAMPLE, alpha=0.35, linewidths=0, label="Sample")
    Yi = Yr[imp_mask]
    pal = sns.color_palette("tab10", int(groups.max()) + 1 if len(groups) else 1)
    for g in np.unique(groups):
        mm = groups == g
        ax.scatter(Yi[mm, 0], Yi[mm, 1], s=22, color=pal[g], alpha=0.9, linewidths=0,
                   label=f"gap grp{g} (n={int(mm.sum())})")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    ax.set_title("Important gap groups (high Reference density, sample-unreached)")
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] gap groups 저장: {out_path}")


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

    # 대표 반경 R (예산=sample 패턴 수 기반) — 공간별
    R_k, k_k = representation_radius(R["Yr"], R["ystd_k"], len(sample_raw), R_MULT)
    R_l, k_l = representation_radius(R["Yr_lin"], R["ystd_l"], len(sample_raw), R_MULT)
    # 커버리지 (KPCA vs 선형) — reach 기반
    mk = coverage_metrics(R["Yr"], Ys, R["ystd_k"], R["dens_k"], R["tau_k"], R_k)
    ml = coverage_metrics(R["Yr_lin"], Ys_lin, R["ystd_l"], R["dens_l"], R["tau_l"], R_l)
    delta = ml["true_cov"] - mk["true_cov"]        # 착시(가짜 커버리지) 크기

    # True Gap → '중요 gap'(Reference 밀도 큰 것)만 추려 KMeans 군집
    gap_mask = mk["gap_mask"]
    dens = R["dens_k"]
    ref_hdr = dens >= R["tau_k"]
    thr = float(np.quantile(dens[ref_hdr], GAP_IMPORTANT_Q)) if ref_hdr.any() else 0.0
    imp_mask = gap_mask & (dens >= thr)             # 중요 gap = gap ∩ 밀도 상위
    Yi = R["Yr"][imp_mask]
    if len(Yi) >= 1:
        ng = int(min(N_GAP_GROUPS, len(Yi)))
        groups = KMeans(n_clusters=ng, n_init=10, random_state=RESERVOIR_SEED).fit_predict(Yi)
    else:
        groups = np.zeros(0, dtype=int)
    # 그룹별 요약(크기·평균 Reference 밀도)
    gsize = {int(g): int((groups == g).sum()) for g in np.unique(groups)}
    grp_summary = []
    for g in np.unique(groups):
        mm = groups == g
        grp_summary.append({"group": int(g), "n": int(mm.sum()),
                            "mean_ref_density": float(dens[imp_mask][mm].mean()),
                            "KP1": float(Yi[mm, 0].mean()), "KP2": float(Yi[mm, 1].mean()),
                            "KP3": float(Yi[mm, 2].mean())})
    grp_summary.sort(key=lambda r: r["mean_ref_density"], reverse=True)

    # 중요 gap 원본행 저장(식별열 + feature + KP + 밀도/그룹, 밀도 내림차순)
    parts = []
    if R.get("keep_cols"):                          # ref.csv 매칭용 식별 열 먼저
        parts.append(pd.DataFrame(R["ref_keep"][imp_mask], columns=R["keep_cols"]).reset_index(drop=True))
    parts.append(pd.DataFrame(R["ref_raw"][imp_mask], columns=names).reset_index(drop=True))
    gap_df = pd.concat(parts, axis=1)
    gap_df["KP1"], gap_df["KP2"], gap_df["KP3"] = Yi[:, 0], Yi[:, 1], Yi[:, 2]
    gap_df["ref_density"] = dens[imp_mask]
    gap_df["gap_group"] = groups
    gap_df = gap_df.sort_values("ref_density", ascending=False)
    gap_df.to_csv(j("uncovered_kpca_gap_patterns.csv"), index=False)

    # scatter 에 쓰인 Reference 전량 저장 (식별열+feature+KP+상태)
    ref_status = np.where(dens < R["tau_k"], "out_of_domain",
                          np.where(gap_mask, "gap", "covered"))
    parts = []
    if R.get("keep_cols"):
        parts.append(pd.DataFrame(R["ref_keep"], columns=R["keep_cols"]).reset_index(drop=True))
    parts.append(pd.DataFrame(R["ref_raw"], columns=names).reset_index(drop=True))
    ref_sc = pd.concat(parts, axis=1)
    ref_sc["KP1"], ref_sc["KP2"], ref_sc["KP3"] = R["Yr"][:, 0], R["Yr"][:, 1], R["Yr"][:, 2]
    ref_sc["ref_density"], ref_sc["cover_status"] = dens, ref_status
    ref_sc.to_csv(j("kpca_reference_scatter.csv"), index=False)

    # scatter 에 쓰인 Sample 전량 저장 (원본 전체 열 + KP)
    if FMT == "csv":
        sdf = pd.read_csv(SAMPLE_PATH)
    else:
        import pyarrow.parquet as pq
        sdf = pq.read_table(SAMPLE_PATH).to_pandas()
    sdf.columns = [str(c).strip() for c in sdf.columns]
    fin = np.isfinite(sdf.loc[:, names].to_numpy(float)).all(axis=1)
    for a, nm in enumerate(["KP1", "KP2", "KP3"]):
        col = np.full(len(sdf), np.nan); col[fin] = Ys[:, a]; sdf[nm] = col
    sdf.to_csv(j("kpca_sample_scatter.csv"), index=False)
    print(f"[Save] kpca_reference_scatter.csv ({len(ref_sc):,}), kpca_sample_scatter.csv ({len(sdf):,})")

    # 시각화
    plot_2d(R["Yr"], Ys, gap_mask, mk["true_cov"], j("kpca_coverage_2d.png"))
    plot_3d(R["Yr"], Ys, gap_mask, j("kpca_coverage_3d.png"))
    if len(Yi):
        plot_gap_groups(R["Yr"], Ys, imp_mask, groups, gsize, j("kpca_gap_groups.png"))
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
        "R_mult": R_MULT, "repr_radius_kpca": round(R_k, 5), "k_neighbors": k_k,
        "n_ref_hdr_kpca": mk["n_ref_hdr"],
        "n_covered_kpca": mk["n_covered"], "n_gap_ref_points": int(gap_mask.sum()),
        "gap_important_q": GAP_IMPORTANT_Q, "n_important_gap": int(imp_mask.sum()),
        "n_gap_groups": len(grp_summary), "gap_groups": grp_summary,
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
    print(f"  대표반경 R(예산기반)      : {R_k:.4g}  (k=ceil(N_ref/N_sample)={k_k}, R_MULT={R_MULT})")
    print(f"  True Gap 포인트           : {int(gap_mask.sum()):,}")
    print(f"  >>> 중요 gap(밀도상위)     : {int(imp_mask.sum()):,}개, {len(grp_summary)}개 그룹 "
          f"→ uncovered_kpca_gap_patterns.csv / kpca_gap_groups.png")
    for r in grp_summary[:5]:
        print(f"       grp{r['group']}: n={r['n']:,}  meanRefDensity={r['mean_ref_density']:.3g}")
    print("=" * 66)
    return summary


if __name__ == "__main__":
    run()
