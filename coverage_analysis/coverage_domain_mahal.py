# -*- coding: utf-8 -*-
"""
coverage_domain_mahal.py
========================
[Mahalanobis Domain Coverage + 근원 항(Attribution)] 전용 분석 (self-contained)

목적
    Sample 이 Reference 의
        (1) 어느 영역을 '대변'하는가          (Cat1 Verified)
        (2) 어느 영역을 '대변하지 못'하는가    (Cat2 Gap)
        (3) Reference 와 '무관'한 포인트는 어디인가 (Cat3 Out-of-Domain)
    를 분류하고, 각 분류의 '근원 항'(원인 feature) 을 MYT 분해로 찾는다.

도메인 정의 (Mahalanobis)
    Reference 의 μ·Σ 로 상관구조를 반영한 타원체
        { x : (x−μ)ᵀ Σ⁻¹ (x−μ) ≤ χ²(q, d) }
    를 도메인으로 삼는다. Σ 고유분해 → 상위 K=2 주성분 백색화(whitening)로 타원을
    반경 √χ²(q,K) 원으로 만들고, 저차원 격자로 커버리지를 잰다.
        Coverage = |G_active ∩ G_total| / |G_total|

근원 항 (MYT / Mason–Young–Tracy 분해)
    정밀행렬 P=Σ⁻¹, 잔차 e=x−μ 에 대해
        D²  = eᵀ P e              (full-d Mahalanobis², Cat3 판정)
        c_j = (P e)_j² / P_jj     (조건부 기여 = D² − D²_(−j), leave-one-out)
        u_j = e_j² / Σ_jj         (무조건부 기여, 변수 단독)
    c_j 가 큰 feature = 그 포인트가 도메인 밖으로 벗어난 '주범'.

데이터 입출력
    * Feature 는 '컬럼 이름' 기준으로 선택·정렬한다(Sample/Reference 열 순서가 달라도 매칭).
    * plot 라벨의 feature 이름 = CSV 컬럼 이름.
    * Reference(대용량)는 chunk 스트리밍(μ·Σ 1패스 + 커버리지 1패스).
    * 비유한(NaN/Inf) 행은 읽는 즉시 제거하고 개수를 보고한다.

실행:  python coverage_domain_mahal.py   (필요 시: pip install seaborn scipy scikit-learn matplotlib pandas)
"""

import functools
import itertools
import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 백엔드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
# =============================================================================
SAMPLE_PATH   = "sample.csv"
REF_PATH      = "reference.csv"
FMT           = "csv"          # 'csv' 또는 'parquet'

# Feature 컬럼 선택 (이름 기준).
#   None      → Sample 의 '수치형' 컬럼만 자동 선택(문자열 ID/라벨 등 뒤쪽 열은 버림).
#   [이름...] → 지정한 컬럼만 그 순서로 feature 로 사용(가장 안전·명시적).
FEATURE_COLS  = None

SCALE_METHOD  = "standard"     # 'standard' | 'minmax' (Mahalanobis 는 standard 권장)

MAHAL_Q       = 0.99           # 타원체 도메인 경계 분위수 (χ² quantile)
MAHAL_GRID_DIMS = 2            # 커버리지 격자를 칠 whitened 주성분 수 (K, 2 권장)
N_BINS_MAHAL  = 10             # whitened 축당 Bin 수 (격자 해상도)

CHUNKSIZE     = 200_000
OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000       # 시각화/Gap 근사용 Reference reservoir 표본 수
OOD_HEATMAP_TOPN = 25          # 히트맵에 표시할 극단 OOD 포인트 수

sns.set_theme(style="whitegrid", context="talk")

N_FEATURES = None              # run() 에서 feature 수로 설정되는 모듈 전역


# =============================================================================
# 데이터 I/O ── 컬럼 이름 기준 선택/정렬 + 비유한 행 제거
# =============================================================================
def _header(path, fmt):
    """파일 헤더의 컬럼 이름 목록(공백 strip)."""
    if fmt == "csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
    else:
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(path).schema_arrow.names)
    return [str(c).strip() for c in cols]


def _numeric_columns(path, fmt):
    """Sample 에서 '수치형' 컬럼 이름만 골라낸다(문자열 ID/라벨 등 비-feature 열은 버림)."""
    if fmt == "csv":
        df = pd.read_csv(path)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table(path).to_pandas()
    df.columns = [str(c).strip() for c in df.columns]
    num = list(df.select_dtypes(include=[np.number]).columns)
    dropped = [c for c in df.columns if c not in num]
    if dropped:
        print(f"[IO] 비수치 컬럼 {len(dropped)}개 자동 제외(feature 아님): {dropped}")
    return num


def resolve_feature_names(sample_path, ref_path, fmt):
    """
    '공통 feature 이름 목록'을 확정한다(양쪽에 모두 있는 컬럼만 사용).
    - FEATURE_COLS 지정 시 그 목록·순서, None 이면 Sample 의 '수치형 컬럼'만 자동 선택.
    - 한쪽에 없거나 중복된 컬럼은 '경고만 하고 제외'하고 계속 진행(exit 하지 않음).
      → 사용할 공통 feature 가 하나도 없을 때만 에러.
    """
    s_cols, r_cols = _header(sample_path, fmt), _header(ref_path, fmt)
    wanted = _numeric_columns(sample_path, fmt) if FEATURE_COLS is None \
        else [str(c).strip() for c in FEATURE_COLS]

    s_set, r_set = set(s_cols), set(r_cols)
    s_dup = {c for c in s_cols if s_cols.count(c) > 1}
    r_dup = {c for c in r_cols if r_cols.count(c) > 1}

    names, dropped = [], []
    for c in wanted:
        if c not in s_set:
            dropped.append((c, "Sample 에 없음"))
        elif c not in r_set:
            dropped.append((c, "Reference 에 없음"))
        elif c in s_dup or c in r_dup:
            dropped.append((c, "중복 컬럼명(매칭 불가)"))
        else:
            names.append(c)

    if dropped:
        print("[IO] 경고: 아래 컬럼은 feature 에서 제외하고 계속 진행합니다:")
        for c, why in dropped:
            print(f"        - {c}: {why}")
    if not names:
        raise ValueError("사용할 공통 feature 컬럼이 하나도 없습니다. (경로/헤더 확인)")
    print(f"[IO] feature {len(names)}개 사용(이름 기준): {names}")
    return names


def _select(df, names):
    """DataFrame 에서 names 컬럼만 그 순서로 뽑아 float ndarray + 비유한 행 제거."""
    df.columns = [str(c).strip() for c in df.columns]
    try:
        X = df.loc[:, names].to_numpy(dtype=np.float64)   # ★ 이름으로 선택·정렬
    except (ValueError, TypeError) as e:
        raise ValueError(f"feature 컬럼을 수치로 변환 실패(비수치 값 포함?): {e}")
    finite = np.isfinite(X).all(axis=1)
    return X[finite], int((~finite).sum())


def read_matrix_by_name(path, names, fmt):
    """작은 파일(Sample) 전체를 (n, d) float64 로 읽는다(feature 열만 선택 + 정제)."""
    want = set(names)
    if fmt == "csv":
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in want)  # feature 열만 파싱
    else:
        import pyarrow.parquet as pq
        raw = [c for c in pq.ParquetFile(path).schema_arrow.names if str(c).strip() in want]
        df = pq.read_table(path, columns=raw).to_pandas()
    X, dropped = _select(df, names)
    if dropped:
        print(f"[IO] Sample 비유한(NaN/Inf) 행 {dropped:,}개 제거")
    if len(X) == 0:
        raise ValueError("Sample 에 유효한 행이 없습니다(전부 NaN/Inf?).")
    return X


def iter_ref_chunks(path, names, scaler, chunksize, fmt):
    """Reference 를 chunk 스트리밍 → 이름 선택·정렬 → 정제 → 스케일링한 배열 yield."""
    dropped_total = [0]
    want = set(names)
    if fmt == "csv":
        # feature 열만 파싱(뒤쪽 미사용/문자열 열은 아예 읽지 않음)
        for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip() in want,
                                 chunksize=chunksize):
            X, dropped = _select(chunk, names)
            dropped_total[0] += dropped
            if len(X):
                yield scaler.transform(X)
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = [c for c in pf.schema_arrow.names if str(c).strip() in set(names)]
        for batch in pf.iter_batches(batch_size=chunksize, columns=raw):
            X, dropped = _select(batch.to_pandas(), names)
            dropped_total[0] += dropped
            if len(X):
                yield scaler.transform(X)
    if dropped_total[0]:
        print(f"[IO] Reference 비유한 행 누적 {dropped_total[0]:,}개 제거")


def stream_reduce(path, names, scaler, map_fn, reduce_fn, init, chunksize, fmt):
    """Reference 전체를 map→reduce 로 1패스 집계(단일 스트리밍)."""
    acc = init()
    for X in iter_ref_chunks(path, names, scaler, chunksize, fmt):
        acc = reduce_fn(acc, map_fn(X))
    return acc


def reservoir_sample(path, names, scaler, k, chunksize, fmt, seed=0):
    """Reference 를 스트리밍하며 균일하게 k개 행 추출(Algorithm R), 스케일링된 배열 반환."""
    rng = np.random.default_rng(seed)
    reservoir = np.empty((k, len(names)), dtype=np.float32)
    filled = seen = 0
    for chunk in iter_ref_chunks(path, names, scaler, chunksize, fmt):
        if filled < k:
            take = min(k - filled, len(chunk))
            reservoir[filled:filled + take] = chunk[:take]
            filled += take; seen += take
            chunk = chunk[take:]
            if len(chunk) == 0:
                continue
        t = seen + np.arange(1, len(chunk) + 1)
        accept = rng.random(len(chunk)) < (k / t)
        if accept.any():
            reservoir[rng.integers(0, k, accept.sum())] = chunk[accept]
        seen += len(chunk)
    print(f"[Reservoir] 전체 {seen:,} rows → {filled:,} rows 추출.")
    return reservoir[:filled]


def fit_scaler(X, method):
    """Sample 배열로 스케일러 학습(standard|minmax). 상수열은 sklearn 이 scale=1 처리."""
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    scaler.fit(X)
    zero_var = np.isclose(getattr(scaler, "var_", np.ones(X.shape[1])), 0).sum() \
        if method == "standard" else 0
    if zero_var:
        print(f"[Scaler] 경고: 분산 0 인 feature {zero_var}개(정보 없음, scale=1 로 처리됨)")
    print(f"[Scaler] Sample 기준 '{method}' 스케일러 학습 (dim={X.shape[1]})")
    return scaler


# =============================================================================
# [MAHALANOBIS] 스트리밍 모멘트(μ, Σ) 추정
# =============================================================================
def _moments_map(block):
    """한 chunk 의 [n, Σx, ΣxxᵀT] (μ·Σ 스트리밍 추정용)."""
    return [block.shape[0], block.sum(axis=0), block.T @ block]


def _moments_init():
    return [0, np.zeros(N_FEATURES), np.zeros((N_FEATURES, N_FEATURES))]


def _moments_reduce(acc, part):
    acc[0] += part[0]; acc[1] += part[1]; acc[2] += part[2]
    return acc


def build_mahalanobis(mu, Sigma, q, K):
    """
    μ·Σ 로 타원체 도메인 + whitened top-K 격자 도메인을 구성한다.
    반환 dict: mu, Sinv(=P), T(full-d χ²), Vk, sqrt_lamk, Tk, Rk, internalK,
               evr_k, domain_set(=G_total), diagP, diagS.
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

    edges = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)     # 타원 안 격자셀만 = G_total
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
# 커버리지 격자 (map→reduce)
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


# =============================================================================
# 포인트별 분류(A) + 근원 항 MYT 분해(B)
# =============================================================================
def whiten_topk(X, p):
    """X → whitened 상위 K 좌표 u. u_i = vᵢ·(x−μ)/√λᵢ."""
    return ((X - p["mu"]) @ p["Vk"]) / p["sqrt_lamk"]


def classify_points(X, p, ref_cells):
    """
    각 포인트를 Cat 로 분류 + 근원 항(MYT) 계산.
    반환: cat(0=대변,1=도메인내,2=OOD), d2, c(조건부), uu(무조건부), u2(whitened topK).
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
    ind = ~ood
    in_ell = ind & ((u2 ** 2).sum(axis=1) <= p["Tk"])   # top-K 타원 안
    for i in np.where(ind)[0]:
        if not in_ell[i]:
            cat[i] = 1                                   # 타원체엔 있으나 top-K 격자 밖
            continue
        key = _cells_of(u2[i:i + 1], p["internalK"], p["Vk"].shape[1]).pop()
        cat[i] = 0 if key in ref_cells else 1
    return {"cat": cat, "d2": d2, "c": c, "uu": uu, "u2": u2}


def gap_feature_diff(ref_pts, p, sam_cells):
    """
    (C) Gap 원인: reservoir Reference 를 top-K 격자로 나눠, Sample 이 못 밟은 셀(gap) ref vs
    대변 셀 ref 의 '원본 feature 평균차'. 반환: (diff(d,), n_gap, n_cov). diff>0=gap 쪽 큼.
    """
    u = whiten_topk(ref_pts, p)
    in_ell = (u ** 2).sum(axis=1) <= p["Tk"]
    rp, u = ref_pts[in_ell], u[in_ell]
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(u.shape[1]):
        idx[:, a] = np.digitize(u[:, a], p["internalK"])
    is_gap = np.array([row.tobytes() not in sam_cells for row in idx])
    if is_gap.sum() == 0 or (~is_gap).sum() == 0:
        return np.zeros(N_FEATURES), int(is_gap.sum()), int((~is_gap).sum())
    diff = rp[is_gap].mean(axis=0) - rp[~is_gap].mean(axis=0)
    return diff, int(is_gap.sum()), int((~is_gap).sum())


# =============================================================================
# 시각화 ① whitened top-2 격자 커버리지 + Cat 산점
# =============================================================================
def plot_coverage_2d(u_s, u_r, cat_s, ref_cells, sam_cells, p, sam_cov, out_path):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle, Patch
    dom = p["domain_set"]
    edges = np.linspace(-p["Rk"], p["Rk"], N_BINS_MAHAL + 1)
    M = np.full((N_BINS_MAHAL, N_BINS_MAHAL), np.nan)    # 0 빈/1 gap/2 대변
    for i in range(N_BINS_MAHAL):
        for j in range(N_BINS_MAHAL):
            key = np.array([i, j], dtype=np.int16).tobytes()
            if key not in dom:
                continue
            in_r, in_s = key in ref_cells, key in sam_cells
            M[i, j] = 2 if (in_r and in_s) else (1 if in_r else 0)

    cmap = ListedColormap(["#F0F0F0", "#AEC7E8", "#FFD92F"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.pcolormesh(edges, edges, np.ma.masked_invalid(M).T, cmap=cmap, norm=norm,
                  alpha=0.55, edgecolors="#DDD", linewidth=0.4)
    ax.add_patch(Circle((0, 0), p["Rk"], fill=False, edgecolor="#333", linewidth=1.6))
    ax.scatter(u_r[:, 0], u_r[:, 1], s=6, c="#999999", alpha=0.20, linewidths=0, label="Reference")
    csel = {0: ("#2CA02C", "Sample: Verified"),
            1: ("#7F7F7F", "Sample: In-domain (no ref)"),
            2: ("#D62728", "Sample: OOD")}
    for cval, (color, lab) in csel.items():
        m = cat_s == cval
        if m.any():
            ax.scatter(u_s[m, 0], u_s[m, 1], s=14, c=color, alpha=0.7, linewidths=0, label=lab)

    handles = [Patch(facecolor="#F0F0F0", edgecolor="#BBB", label="Empty domain (no ref)"),
               Patch(facecolor="#AEC7E8", edgecolor="#BBB", label="Gap (ref only, uncovered)"),
               Patch(facecolor="#FFD92F", edgecolor="#BBB", label="Covered (sample)")]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9, title="Cell state")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, title="Points")
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Mahalanobis Domain Coverage (q={MAHAL_Q})\n"
                 f"Sample Coverage = {sam_cov*100:.2f}%   (sqrt_Tk={p['Rk']:.2f})")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (1) coverage-2d 저장: {out_path}")


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
            label=f"chi2(d={N_FEATURES}) theoretical")
    ax.axvline(p["T"], color="#333", ls="--", lw=1.6,
               label=f"Domain bound T=chi2({MAHAL_Q},d)={p['T']:.1f}")
    ax.set_xlabel("Mahalanobis D^2 (full-d)"); ax.set_ylabel("density")
    ax.set_title("Mahalanobis distance: Reference vs Sample\n"
                 "(Sample shifted right = larger deviation from Reference)")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (2) distribution 저장: {out_path}")


# =============================================================================
# 시각화 ③ 로딩 Biplot (PC1·PC2 평면의 원본 feature 방향)
# =============================================================================
def plot_biplot(p, u_s, cat_s, names, out_path):
    load = p["Vk"] * p["sqrt_lamk"]                     # (d, 2) 상관 스케일 로딩
    scale = 0.9 * p["Rk"] / (np.abs(load).max() + 1e-9)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    m = cat_s == 2
    ax.scatter(u_s[~m, 0], u_s[~m, 1], s=10, c="#BBBBBB", alpha=0.35, linewidths=0,
               label="Sample (in-domain)")
    if m.any():
        ax.scatter(u_s[m, 0], u_s[m, 1], s=16, c="#D62728", alpha=0.6, linewidths=0,
                   label="Sample (OOD)")
    for j in range(N_FEATURES):
        dx, dy = load[j, 0] * scale, load[j, 1] * scale
        ax.arrow(0, 0, dx, dy, color="#1F77B4", alpha=0.8,
                 head_width=0.06 * p["Rk"], length_includes_head=True, linewidth=1.3)
        ax.text(dx * 1.08, dy * 1.08, names[j], color="#0B3D66", fontsize=9,
                ha="center", va="center")
    ax.axhline(0, color="#CCC", lw=0.8); ax.axvline(0, color="#CCC", lw=0.8)
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Loading biplot - feature directions (top-2 EVR={p['evr_k']*100:.1f}%)\n"
                 "Arrows point toward the features driving deviation")
    ax.set_aspect("equal"); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (3) biplot 저장: {out_path}")


# =============================================================================
# 시각화 ④ OOD 근원 항 랭킹 (조건부 c_j vs 무조건부 u_j)
# =============================================================================
def plot_ood_attribution(cls, names, out_path, topn=15):
    ood = cls["cat"] == 2
    if ood.sum() == 0:
        print("[Plot] (4) OOD 없음 → 근원 항 막대 생략")
        return None
    cond = cls["c"][ood].mean(axis=0)
    uncond = cls["uu"][ood].mean(axis=0)
    order = np.argsort(cond)[::-1][:topn]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y + 0.2, cond[order], height=0.4, color="#D62728", label="Conditional c_j (MYT)")
    ax.barh(y - 0.2, uncond[order], height=0.4, color="#F0A0A0", label="Unconditional u_j")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Mean Mahalanobis^2 contribution"); ax.set_ylabel("feature")
    ax.set_title(f"OOD points ({int(ood.sum())}) attribution ranking\n"
                 "High c_j = main driver of leaving the Reference domain")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (4) ood-attribution 저장: {out_path}")
    return [(names[i], float(cond[i])) for i in order[:5]]


# =============================================================================
# 시각화 ⑤ Gap 원인 feature 발산형 막대
# =============================================================================
def plot_gap_attribution(diff, n_gap, n_cov, names, out_path, topn=15):
    if n_gap == 0:
        print("[Plot] (5) Gap 없음 → 생략")
        return
    order = np.argsort(np.abs(diff))[::-1][:topn]
    order = order[np.argsort(diff[order])]
    y = np.arange(len(order))
    colors = ["#D62728" if diff[i] > 0 else "#1F77B4" for i in order]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y, diff[order], color=colors)
    ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Gap-cell mean - Covered-cell mean (scaled)"); ax.set_ylabel("feature")
    ax.set_title(f"Gap region driver features (gap ref={n_gap}, cover ref={n_cov})\n"
                 "Red = higher in gap / Blue = higher in covered")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (5) gap-attribution 저장: {out_path}")


# =============================================================================
# 시각화 ⑥ 극단 OOD top-N × feature 조건부 기여 히트맵
# =============================================================================
def plot_ood_heatmap(cls, names, out_path, topn=OOD_HEATMAP_TOPN):
    ood = np.where(cls["cat"] == 2)[0]
    if len(ood) == 0:
        print("[Plot] (6) OOD 없음 → 히트맵 생략")
        return
    ood = ood[np.argsort(cls["d2"][ood])[::-1][:topn]]
    C = cls["c"][ood]
    fcol = np.argsort(C.sum(axis=0))[::-1]
    C = C[:, fcol]
    fig, ax = plt.subplots(figsize=(min(16, 0.5 * N_FEATURES + 3), max(6, 0.32 * len(ood))))
    sns.heatmap(C, cmap="Reds", ax=ax, cbar_kws={"label": "c_j (conditional contribution)"},
                xticklabels=[names[i] for i in fcol],
                yticklabels=[f"#{i} (D2={cls['d2'][i]:.0f})" for i in ood])
    ax.set_xlabel("feature"); ax.set_ylabel("extreme OOD points")
    ax.set_title(f"Top-{len(ood)} extreme OOD points: per-feature attribution (c_j)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (6) ood-heatmap 저장: {out_path}")


# =============================================================================
# 메인
# =============================================================================
def run():
    global N_FEATURES
    for pth in (SAMPLE_PATH, REF_PATH):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"입력 파일 없음: {pth} (CONFIG 경로 확인)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # feature 이름 확정(양쪽 매칭) → Sample 로드 → 스케일러 → 스케일링
    names = resolve_feature_names(SAMPLE_PATH, REF_PATH, FMT)
    N_FEATURES = len(names)
    sample = read_matrix_by_name(SAMPLE_PATH, names, FMT)
    scaler = fit_scaler(sample, SCALE_METHOD)
    sample_scaled = scaler.transform(sample)

    # (1) μ·Σ 스트리밍 추정
    n_ref, sum_x, sum_xx = stream_reduce(
        REF_PATH, names, scaler, _moments_map, _moments_reduce, _moments_init, CHUNKSIZE, FMT)
    if n_ref < N_FEATURES + 1:
        raise ValueError(f"Reference 유효행 {n_ref} < 차원 {N_FEATURES}+1: 공분산 추정 불가.")
    mu = sum_x / n_ref
    Sigma = sum_xx / n_ref - np.outer(mu, mu)
    p = build_mahalanobis(mu, Sigma, MAHAL_Q, MAHAL_GRID_DIMS)
    dom = p["domain_set"]
    print(f"[Domain] μ·Σ 추정({n_ref:,} rows), 타원 q={MAHAL_Q} "
          f"(√T_full={np.sqrt(p['T']):.2f}); top-{MAHAL_GRID_DIMS} EVR={p['evr_k']*100:.1f}%, "
          f"|G_total|={len(dom):,} 셀")

    # (2) 커버리지 스트리밍 패스(Reference) + Sample(메모리)
    sam_cells, sam_out, sam_tot = mahal_cover_map(sample_scaled, p)
    map_fn = functools.partial(mahal_cover_map, p=p)
    ref_cells, ref_out, ref_tot = stream_reduce(
        REF_PATH, names, scaler, map_fn, _domain_reduce, _domain_init, CHUNKSIZE, FMT)
    ref_cov = len(ref_cells & dom) / len(dom)
    sam_cov = len(sam_cells & dom) / len(dom)
    verified = len(sam_cells & ref_cells & dom)         # Cat1 셀
    gap = len(ref_cells & dom) - verified               # Cat2 셀

    # (3) 포인트별 분류(A) + 근원 항(B)
    cls = classify_points(sample_scaled, p, ref_cells)
    cat = cls["cat"]
    n_ver, n_ind, n_ood = (cat == 0).sum(), (cat == 1).sum(), (cat == 2).sum()

    # 시각화용 Reference reservoir 표본
    ref_pts = reservoir_sample(REF_PATH, names, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT)
    d2_ref = np.einsum("ij,jk,ik->i", ref_pts - mu, p["Sinv"], ref_pts - mu)
    u_s, u_r = cls["u2"][:, :2], whiten_topk(ref_pts, p)[:, :2]

    # (4) Gap 원인(C) — reservoir 근사
    gap_diff, n_gap_r, n_cov_r = gap_feature_diff(ref_pts, p, sam_cells)

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
    run()
