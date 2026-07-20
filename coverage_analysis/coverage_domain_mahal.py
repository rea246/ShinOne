# -*- coding: utf-8 -*-
"""
coverage_domain_mahal.py
========================
[Mahalanobis Domain Coverage + Attribution] — Reference 고정 프레임 / per-feature 직관 / 캐시

핵심 설계
    * Reference 고정 프레임: 스케일러·μ·Σ·PCA(V,λ)를 'Reference' 에서만 학습해 고정하고,
      Sample 은 그 프레임에 투영만 한다. → 같은 Reference 면 sample1/sample2 를 따로 돌려도
      저장 파일(PC1/PC2·커버리지·per-feature)이 동일 좌표계라 서로 비교 가능.
    * 21차원 가혹함 완화: 헤드라인 커버리지는 'per-feature(1D) 커버리지 평균'.
      full-d Mahalanobis OOD·MYT 는 '진단용'으로 항상 함께 표기(데이터 이상 은폐 방지).
    * 캐시: Reference 파생물(프레임·ref 셀·per-feature bin·reservoir)을 파일 캐시.
      두 번째 실행부터 Reference 스트리밍을 건너뜀(같은 ref+config 이면 재사용).

지표
    per-feature coverage_j = |sam_bins_j ∩ ref_bins_j| / |ref_bins_j|   (feature 마다)
    headline = mean_j coverage_j
    (보조) top-2 격자 커버리지, Ref-footprint 커버리지, full-d Sample OOD%

근원 항 (MYT):  D²=eᵀPe,  c_j=(Pe)_j²/P_jj,  u_j=e_j²/Σ_jj   (P=Σ⁻¹, e=z−μ)

실행:  python coverage_domain_mahal.py   (pip install seaborn scipy matplotlib pandas)
"""

import functools
import hashlib
import itertools
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2

# =============================================================================
# CONFIG
# =============================================================================
SAMPLE_PATH   = "sample.csv"
REF_PATH      = "reference.csv"
FMT           = "csv"          # 'csv' 또는 'parquet'

# Feature 선택 (직접 지정). FEATURE_COLS(이름) 우선, 없으면 FEATURE_COL_IDX(Sample 위치).
#   ※ 최종 feature 순서는 'Reference 헤더 순서'로 고정 → sample 순서 무관하게 비교 가능.
FEATURE_COLS    = None
FEATURE_COL_IDX = list(range(0, 21))

# 특정 feature term 을 '끄고' 분석(ablation). 여기에 이름을 넣으면 그 feature 는 제외되고
# 프레임·PCA·K_eff·커버리지·OOD·PC loading 이 그 term 없이 다시 계산된다. 예: ["CM2", "G17"]
EXCLUDE_COLS    = []

SCALE_METHOD  = "standard"     # 'standard' | 'minmax'
MAHAL_Q       = 0.9999         # χ² quantile (도메인 경계) — 넓게
# OOD 경계 방식: 'chi2'(이론 χ²) | 'empirical'(Reference 실측 D²_eff 분위수 → 더 soft·꼬리보정)
OOD_BOUND     = "empirical"
MAHAL_GRID_DIMS = 2            # top-2 격자 시각화 차원(K)
N_BINS_MAHAL  = 10             # top-2 격자 해상도
N_BINS_1D     = 20             # per-feature 1D 격자 해상도

# 유효차원(effective dimension): full-d(21) Mahalanobis 는 노이즈 축까지 세어 가혹·무의미.
#   K_eff 로 줄여 OOD/MYT 를 상위 부분공간에서 계산한다.
EFF_DIM_OVERRIDE = None        # None → participation ratio 자동, 정수면 강제 지정
PAIR_MAX      = 6              # 유효차원 교차 스캐터에 그릴 최대 차원 수

CHUNKSIZE     = 200_000
OUTPUT_DIR      = "coverage_plots"
PLOT_DOWNSAMPLE = 20_000       # 시각화/Gap 근사용 Reference reservoir 표본 수
OOD_HEATMAP_TOPN = 25

USE_CACHE     = True           # Reference 파생물 캐시 사용
CACHE_DIR     = ".refcache"
RESERVOIR_SEED = 0

sns.set_theme(style="white", context="talk")

C_SAMPLE = "#2CA02C"; C_REF = "#8C8C8C"
C_COVER  = "#A1D99B"; C_GAP = "#F4A15A"; C_EMPTY = "#F0F0F0"

N_FEATURES = None


# =============================================================================
# 데이터 I/O ── 컬럼 이름 기준 선택/정렬(Reference 순서 고정) + 비유한 행 제거
# =============================================================================
def _header(path, fmt):
    if fmt == "csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
    else:
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(path).schema_arrow.names)
    return [str(c).strip() for c in cols]


def _wanted_from_config(s_cols):
    if FEATURE_COLS is not None:
        return [str(c).strip() for c in FEATURE_COLS]
    if FEATURE_COL_IDX is not None:
        bad = [i for i in FEATURE_COL_IDX if i < 0 or i >= len(s_cols)]
        if bad:
            raise ValueError(f"FEATURE_COL_IDX 가 Sample 열 범위를 벗어남: {bad} (열 {len(s_cols)}개)")
        return [s_cols[i] for i in FEATURE_COL_IDX]
    raise ValueError("feature 컬럼을 지정하세요: FEATURE_COLS(이름) 또는 FEATURE_COL_IDX(위치).")


def resolve_feature_names(sample_path, ref_path, fmt):
    """
    공통 feature 를 확정하되, 최종 '순서는 Reference 헤더 기준'으로 고정한다.
    (sample 열 순서가 달라도 동일 프레임 → 비교 가능). 없거나 중복은 경고 후 제외.
    """
    s_cols, r_cols = _header(sample_path, fmt), _header(ref_path, fmt)
    wanted = set(_wanted_from_config(s_cols))
    s_set, r_set = set(s_cols), set(r_cols)
    s_dup = {c for c in s_cols if s_cols.count(c) > 1}
    r_dup = {c for c in r_cols if r_cols.count(c) > 1}

    names, dropped = [], []
    for c in r_cols:                       # Reference 순서로 확정
        if c not in wanted:
            continue
        if c not in s_set:
            dropped.append((c, "Sample 에 없음"))
        elif c in s_dup or c in r_dup:
            dropped.append((c, "중복 컬럼명(매칭 불가)"))
        else:
            names.append(c)
    only_ref_missing = [c for c in wanted if c not in r_set]
    for c in only_ref_missing:
        dropped.append((c, "Reference 에 없음"))
    if dropped:
        print("[IO] 경고: 아래 컬럼은 feature 에서 제외하고 계속 진행합니다:")
        for c, why in dropped:
            print(f"        - {c}: {why}")
    if EXCLUDE_COLS:                                  # ablation: 지정 feature 끄기
        ex = {str(c).strip() for c in EXCLUDE_COLS}
        removed = [c for c in names if c in ex]
        names = [c for c in names if c not in ex]
        if removed:
            print(f"[IO] EXCLUDE_COLS 로 끈 feature({len(removed)}개): {removed}")
    if not names:
        raise ValueError("사용할 공통 feature 컬럼이 하나도 없습니다.")
    print(f"[IO] feature {len(names)}개 사용(Reference 순서 고정): {names}")
    return names


def _select_raw(df, names):
    """DataFrame → names 컬럼만 그 순서로 float ndarray + 비유한 행 제거."""
    df.columns = [str(c).strip() for c in df.columns]
    try:
        X = df.loc[:, names].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"feature 컬럼을 수치로 변환 실패(비수치 값 포함?): {e}")
    finite = np.isfinite(X).all(axis=1)
    return X[finite], int((~finite).sum())


def read_matrix_by_name(path, names, fmt):
    """작은 파일(Sample) 전체를 (n,d) float64 로(feature 열만, 정제)."""
    want = set(names)
    if fmt == "csv":
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in want)
    else:
        import pyarrow.parquet as pq
        raw = [c for c in pq.ParquetFile(path).schema_arrow.names if str(c).strip() in want]
        df = pq.read_table(path, columns=raw).to_pandas()
    X, dropped = _select_raw(df, names)
    if dropped:
        print(f"[IO] Sample 비유한 행 {dropped:,}개 제거")
    if len(X) == 0:
        raise ValueError("Sample 에 유효한 행이 없습니다.")
    return X


def iter_ref_chunks_raw(path, names, chunksize, fmt):
    """Reference 를 chunk 스트리밍 → feature 열만, 원시(raw) float ndarray yield."""
    want = set(names)
    dropped = 0
    if fmt == "csv":
        for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip() in want, chunksize=chunksize):
            X, dd = _select_raw(chunk, names); dropped += dd
            if len(X):
                yield X
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = [c for c in pf.schema_arrow.names if str(c).strip() in want]
        for batch in pf.iter_batches(batch_size=chunksize, columns=raw):
            X, dd = _select_raw(batch.to_pandas(), names); dropped += dd
            if len(X):
                yield X
    if dropped:
        print(f"[IO] Reference 비유한 행 누적 {dropped:,}개 제거")


# =============================================================================
# Reference 고정 프레임 (pass 1: 원시 모멘트 → 스케일러·μ·Σ·PCA·격자)
# =============================================================================
def ref_raw_moments(path, names, chunksize, fmt):
    """pass1: Reference 원시 [n, Σx, ΣxxᵀT, min, max]."""
    d = len(names)
    n = 0; s = np.zeros(d); ss = np.zeros((d, d))
    mn = np.full(d, np.inf); mx = np.full(d, -np.inf)
    for X in iter_ref_chunks_raw(path, names, chunksize, fmt):
        n += len(X); s += X.sum(0); ss += X.T @ X
        mn = np.minimum(mn, X.min(0)); mx = np.maximum(mx, X.max(0))
    if n < d + 1:
        raise ValueError(f"Reference 유효행 {n} < 차원 {d}+1: 공분산 추정 불가.")
    mean = s / n
    cov = ss / n - np.outer(mean, mean)
    return n, mean, cov, mn, mx


def build_frame(mean_raw, cov_raw, mn, mx, method, q, K):
    """원시 모멘트에서 스케일러(center/scale)와 표준화공간 μ·Σ·PCA·격자·per-feature 축을 만든다."""
    d = len(mean_raw)
    if method == "minmax":
        center = mn.copy(); scale = (mx - mn).astype(float)
    else:
        center = mean_raw.copy(); scale = np.sqrt(np.diag(cov_raw))
    scale[scale == 0] = 1.0
    mu = (mean_raw - center) / scale
    Sigma = cov_raw / np.outer(scale, scale)

    ridge = 1e-6 * (np.trace(Sigma) / d)
    Sig = Sigma + ridge * np.eye(d)
    Sinv = np.linalg.inv(Sig)
    w, V = np.linalg.eigh(Sig)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Vk, lamk = V[:, :K], w[:K]
    sqrt_lamk = np.sqrt(lamk)
    T, Tk = float(chi2.ppf(q, d)), float(chi2.ppf(q, K))
    Rk, R1 = float(np.sqrt(Tk)), float(np.sqrt(chi2.ppf(q, 1)))
    internalK = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)[1:-1]
    edges1d = np.linspace(-R1, R1, N_BINS_1D + 1)
    evr_k = float(lamk.sum() / w.sum())

    # ── 유효차원 K_eff (participation ratio) → 부분공간 Mahalanobis(=Hotelling T²) ──
    eff_dim = float((w.sum() ** 2) / (w ** 2).sum())        # participation ratio
    K_eff = int(EFF_DIM_OVERRIDE) if EFF_DIM_OVERRIDE else int(np.clip(round(eff_dim), 2, d))
    Veff, lam_eff = V[:, :K_eff], w[:K_eff]
    sqrt_lam_eff = np.sqrt(lam_eff)
    Peff = Veff @ np.diag(1.0 / lam_eff) @ Veff.T           # rank-K_eff 정밀행렬
    T_eff = float(chi2.ppf(q, K_eff))                        # K_eff 차원 χ² 임계
    evr_eff = float(lam_eff.sum() / w.sum())

    edges = np.linspace(-Rk, Rk, N_BINS_MAHAL + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    domain_set = set()
    for combo in itertools.product(range(N_BINS_MAHAL), repeat=K):
        if (centers[list(combo)] ** 2).sum() <= Tk:
            domain_set.add(np.array(combo, dtype=np.int16).tobytes())

    return {"center": center, "scale": scale, "mu": mu, "Sinv": Sinv, "T": T,
            "Vk": Vk, "sqrt_lamk": sqrt_lamk, "Tk": Tk, "Rk": Rk, "R1": R1,
            "internalK": internalK, "edges1d": edges1d, "evr_k": evr_k,
            "domain_set": domain_set, "diagP": np.diag(Sinv).copy(),
            "diagS": np.diag(Sig).copy(), "mean_raw": mean_raw,
            "std_raw": np.sqrt(np.diag(cov_raw)),
            "eff_dim": eff_dim, "K_eff": K_eff, "Veff": Veff,
            "sqrt_lam_eff": sqrt_lam_eff, "Peff": Peff, "T_eff": T_eff,
            "diagPeff": np.diag(Peff).copy(), "evr_eff": evr_eff,
            "V_full": V, "lam_full": w,                    # 전체 PC (loading 전량 출력용)
            "corr": Sig / np.sqrt(np.outer(np.diag(Sig), np.diag(Sig)))}  # 상관행렬


# =============================================================================
# 격자/whiten 헬퍼
# =============================================================================
def standardize(X, p):
    return (X - p["center"]) / p["scale"]


def whiten_topk(z, p):
    return ((z - p["mu"]) @ p["Vk"]) / p["sqrt_lamk"]


def _cells_of(u, internalK, K):
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(K):
        idx[:, a] = np.digitize(u[:, a], internalK)
    return {row.tobytes() for row in np.unique(idx, axis=0)}


def _feature_bins(w, p):
    """per-feature 표준화좌표 w(=(z−μ)/√Σ_jj) → feature 별 점유 bin 집합 리스트."""
    d = w.shape[1]; R1 = p["R1"]; edg = p["edges1d"][1:-1]
    out = [set() for _ in range(d)]
    for jf in range(d):
        wj = w[:, jf]; inb = np.abs(wj) <= R1
        if inb.any():
            out[jf].update(np.unique(np.digitize(wj[inb], edg)).tolist())
    return out


# =============================================================================
# Reference 커버리지 (pass 2: ref 셀 + per-feature bin + OOD + reservoir)
# =============================================================================
def ref_pass2(path, names, p, k, chunksize, fmt, seed=RESERVOIR_SEED):
    K = p["Vk"].shape[1]; d = len(names)
    cells = set(); n_out = 0; n_tot = 0
    feat_bins = [set() for _ in range(d)]
    sd = np.sqrt(p["diagS"])
    rng = np.random.default_rng(seed)
    reservoir = np.empty((k, d), dtype=np.float32); filled = seen = 0

    for X in iter_ref_chunks_raw(path, names, chunksize, fmt):
        z = standardize(X, p); e = z - p["mu"]
        d2 = np.einsum("ij,jk,ik->i", e, p["Peff"], e)   # 유효차원 D²_eff
        n_out += int((d2 > p["T_eff"]).sum()); n_tot += len(z)
        u = (e @ p["Vk"]) / p["sqrt_lamk"]
        uu = u[(u ** 2).sum(1) <= p["Tk"]]
        if len(uu):
            cells |= _cells_of(uu, p["internalK"], K)
        fb = _feature_bins(e / sd, p)
        for jf in range(d):
            feat_bins[jf] |= fb[jf]
        zt = z.astype(np.float32)          # reservoir (scaled)
        if filled < k:
            take = min(k - filled, len(zt))
            reservoir[filled:filled + take] = zt[:take]
            filled += take; seen += take; zt = zt[take:]
            if len(zt) == 0:
                continue
        t = seen + np.arange(1, len(zt) + 1)
        acc = rng.random(len(zt)) < (k / t)
        if acc.any():
            reservoir[rng.integers(0, k, acc.sum())] = zt[acc]
        seen += len(zt)

    ref_pts = reservoir[:filled]
    d2_ref = np.einsum("ij,jk,ik->i", ref_pts - p["mu"], p["Peff"], ref_pts - p["mu"])
    print(f"[Ref] pass2 완료: ref 셀={len(cells)}, reservoir={filled:,}, OOD={n_out/max(n_tot,1)*100:.2f}%")
    return {"ref_cells": cells, "ref_out": n_out, "ref_tot": n_tot,
            "ref_pts": ref_pts, "d2_ref": d2_ref, "feat_bins": feat_bins}


# =============================================================================
# 캐시 (Reference 파생물: 프레임 + pass2). 같은 ref+config 면 스트리밍 생략.
# =============================================================================
def _cache_key(ref_path, names, fmt):
    st = os.stat(ref_path)
    key = {"ref": os.path.abspath(ref_path), "size": st.st_size, "mtime": int(st.st_mtime),
           "names": names, "fmt": fmt, "scale": SCALE_METHOD, "q": MAHAL_Q,
           "K": MAHAL_GRID_DIMS, "nb": N_BINS_MAHAL, "nb1d": N_BINS_1D,
           "ds": PLOT_DOWNSAMPLE, "seed": RESERVOIR_SEED, "effov": EFF_DIM_OVERRIDE, "v": 3}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def load_or_build_reference(ref_path, names, fmt):
    """Reference 프레임 + pass2 결과를 반환(캐시 우선)."""
    key = _cache_key(ref_path, names, fmt)
    cpath = os.path.join(CACHE_DIR, f"ref_{key}.pkl")
    if USE_CACHE and os.path.exists(cpath):
        with open(cpath, "rb") as f:
            obj = pickle.load(f)
        print(f"[Cache] Reference 파생물 재사용: {cpath} (스트리밍 생략)")
        return obj

    print("[Cache] 캐시 없음 → Reference 스트리밍(2 passes)")
    n_ref, mean_raw, cov_raw, mn, mx = ref_raw_moments(ref_path, names, CHUNKSIZE, fmt)
    p = build_frame(mean_raw, cov_raw, mn, mx, SCALE_METHOD, MAHAL_Q, MAHAL_GRID_DIMS)
    print(f"[Frame] Reference 고정 프레임({n_ref:,} rows); 유효차원 K_eff={p['K_eff']} "
          f"(participation ratio={p['eff_dim']:.2f}, EVR={p['evr_eff']*100:.1f}%); "
          f"top-{MAHAL_GRID_DIMS} EVR={p['evr_k']*100:.1f}%, |G_total|={len(p['domain_set']):,}")
    r2 = ref_pass2(ref_path, names, p, PLOT_DOWNSAMPLE, CHUNKSIZE, fmt)
    obj = {"p": p, "n_ref": n_ref, **r2}
    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cpath, "wb") as f:
            pickle.dump(obj, f)
        print(f"[Cache] 저장: {cpath}")
    return obj


# =============================================================================
# Sample 분류(A) + 근원 항(B) + per-feature 커버리지
# =============================================================================
def classify_points(z, p, ref_cells, T_ood):
    e = z - p["mu"]
    g = e @ p["Peff"]                               # 유효차원 부분공간 정밀행렬
    d2 = np.einsum("ij,ij->i", e, g)                # D²_eff = eᵀ P_eff e (=Σ top-K_eff u²)
    c = g ** 2 / p["diagPeff"]                      # MYT 조건부 기여(부분공간)
    uu_ = e ** 2 / p["diagS"]                       # 무조건부 기여(변수 단독)
    u2 = e @ p["Vk"] / p["sqrt_lamk"]               # top-2 (격자/시각화용)
    cat = np.empty(len(z), dtype=np.int8)
    ood = d2 > T_ood; cat[ood] = 2
    ind = ~ood
    in_ell = ind & ((u2 ** 2).sum(1) <= p["Tk"])
    for i in np.where(ind)[0]:
        if not in_ell[i]:
            cat[i] = 1; continue
        key = _cells_of(u2[i:i + 1], p["internalK"], p["Vk"].shape[1]).pop()
        cat[i] = 0 if key in ref_cells else 1
    return {"cat": cat, "d2": d2, "c": c, "uu": uu_, "u2": u2}


def per_feature_coverage(w_s, p, ref_feat_bins):
    """feature 별 coverage_j = |sam∩ref bins| / |ref bins|, 그리고 sample bins."""
    d = w_s.shape[1]; R1 = p["R1"]; edg = p["edges1d"][1:-1]
    cov = np.zeros(d)
    for j in range(d):
        wj = w_s[:, j]; inb = np.abs(wj) <= R1
        sb = set(np.unique(np.digitize(wj[inb], edg)).tolist()) if inb.any() else set()
        rb = ref_feat_bins[j]
        cov[j] = len(sb & rb) / len(rb) if rb else 0.0
    return cov


def gap_feature_diff(ref_pts, p, sam_cells):
    u = whiten_topk(ref_pts, p)
    in_ell = (u ** 2).sum(1) <= p["Tk"]
    rp, u = ref_pts[in_ell], u[in_ell]
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(u.shape[1]):
        idx[:, a] = np.digitize(u[:, a], p["internalK"])
    is_gap = np.array([row.tobytes() not in sam_cells for row in idx])
    if is_gap.sum() == 0 or (~is_gap).sum() == 0:
        return np.zeros(N_FEATURES), int(is_gap.sum()), int((~is_gap).sum())
    diff = rp[is_gap].mean(0) - rp[~is_gap].mean(0)
    return diff, int(is_gap.sum()), int((~is_gap).sum())


# =============================================================================
# 시각화
# =============================================================================
def plot_coverage_2d(u_s, u_r, ref_cells, sam_cells, p, sam_cov, out_path):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Circle, Patch
    dom = p["domain_set"]
    edges = np.linspace(-p["Rk"], p["Rk"], N_BINS_MAHAL + 1)
    M = np.full((N_BINS_MAHAL, N_BINS_MAHAL), np.nan)
    for i in range(N_BINS_MAHAL):
        for j in range(N_BINS_MAHAL):
            key = np.array([i, j], dtype=np.int16).tobytes()
            if key not in dom:
                continue
            in_r, in_s = key in ref_cells, key in sam_cells
            M[i, j] = 2 if (in_r and in_s) else (1 if in_r else 0)
    cmap = ListedColormap([C_EMPTY, C_GAP, C_COVER])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.pcolormesh(edges, edges, np.ma.masked_invalid(M).T, cmap=cmap, norm=norm, edgecolors="none")
    ax.add_patch(Circle((0, 0), p["Rk"], fill=False, edgecolor="#333", linewidth=1.6))
    ax.scatter(u_r[:, 0], u_r[:, 1], s=6, c=C_REF, alpha=0.10, linewidths=0, label="Reference")
    ax.scatter(u_s[:, 0], u_s[:, 1], s=14, c=C_SAMPLE, alpha=0.10, linewidths=0, label="Sample")
    handles = [Patch(facecolor=C_EMPTY, edgecolor="#BBB", label="Empty domain (no ref)"),
               Patch(facecolor=C_GAP, edgecolor="#BBB", label="Gap (ref only, uncovered)"),
               Patch(facecolor=C_COVER, edgecolor="#BBB", label="Covered (sample)")]
    leg1 = ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.9, title="Cell state")
    ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, title="Points")
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Mahalanobis Domain Coverage (q={MAHAL_Q})  |  top-2 grid cov = {sam_cov*100:.1f}%\n"
                 f"circle = Ref Mahalanobis boundary (R={p['Rk']:.2f})", fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] coverage-2d 저장: {out_path}")


def plot_scatter_ref_sample(u_s, u_r, out_path):
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.scatter(u_r[:, 0], u_r[:, 1], s=8, c=C_REF, alpha=0.7, linewidths=0, label="Reference")
    ax.scatter(u_s[:, 0], u_s[:, 1], s=14, c=C_SAMPLE, alpha=0.7, linewidths=0, label="Sample")
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title("Reference vs Sample scatter (whitened top-2)")
    ax.set_aspect("equal"); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ref-sample scatter 저장: {out_path}")


def plot_effective_pairscatter(u_eff_s, u_eff_r, K_eff, out_path):
    """유효차원(whitened top-K_eff) 간 교차 산점 행렬. 대각=분포, 비대각=PC_i vs PC_j."""
    Kp = min(K_eff, PAIR_MAX)
    Rk2 = float(np.sqrt(chi2.ppf(MAHAL_Q, 2)))     # 2D 마진 참고 원(대략적 가이드)
    from matplotlib.patches import Circle
    fig, axes = plt.subplots(Kp, Kp, figsize=(2.3 * Kp, 2.3 * Kp), squeeze=False)
    lim = max(4.0, Rk2 + 1)
    for i in range(Kp):
        for jj in range(Kp):
            ax = axes[i][jj]
            if i == jj:
                b = np.linspace(-lim, lim, 40)
                ax.hist(u_eff_r[:, i], bins=b, density=True, color=C_REF, alpha=0.5)
                ax.hist(u_eff_s[:, i], bins=b, density=True, color=C_SAMPLE, alpha=0.5)
                ax.set_yticks([])
            else:
                ax.scatter(u_eff_r[:, jj], u_eff_r[:, i], s=4, c=C_REF, alpha=0.15, linewidths=0)
                ax.scatter(u_eff_s[:, jj], u_eff_s[:, i], s=6, c=C_SAMPLE, alpha=0.30, linewidths=0)
                ax.add_patch(Circle((0, 0), Rk2, fill=False, edgecolor="#CCC", ls=":", lw=0.8))
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            if i == Kp - 1:
                ax.set_xlabel(f"PC{jj+1}", fontsize=8)
            if jj == 0:
                ax.set_ylabel(f"PC{i+1}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Effective-dim cross scatter (whitened top-{Kp} of K_eff={K_eff})  "
                 "Ref(gray) vs Sample(green)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[Plot] effective-dim pairscatter 저장: {out_path}")


def plot_perfeature_coverage(cov, names, out_path):
    order = np.argsort(cov)                         # 낮은 것부터(부족 축이 위/아래로 눈에)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(order))))
    colors = [C_GAP if cov[i] < 0.5 else C_SAMPLE for i in order]
    ax.barh(y, cov[order] * 100, color=colors)
    ax.axvline(np.mean(cov) * 100, color="#333", ls="--", lw=1.2,
               label=f"mean = {np.mean(cov)*100:.1f}%")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("per-feature coverage (%)"); ax.set_ylabel("feature")
    ax.set_xlim(0, 100)
    ax.set_title("Per-feature 1D coverage (Sample vs Reference)\n"
                 "low bar = feature Sample fails to cover")
    ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] per-feature coverage 저장: {out_path}")


def plot_perfeature_dist(w_s, w_r, names, p, out_path):
    d = len(names); ncol = 5; nrow = int(np.ceil(d / ncol))
    R1 = p["R1"]; bins = np.linspace(-max(R1, 4), max(R1, 4), 40)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.4 * nrow), squeeze=False)
    for j in range(nrow * ncol):
        ax = axes[j // ncol][j % ncol]
        if j >= d:
            ax.axis("off"); continue
        ax.hist(w_r[:, j], bins=bins, density=True, color=C_REF, alpha=0.5, label="Ref")
        ax.hist(w_s[:, j], bins=bins, density=True, color=C_SAMPLE, alpha=0.5, label="Sample")
        ax.axvline(-R1, color="#999", ls=":", lw=0.8); ax.axvline(R1, color="#999", ls=":", lw=0.8)
        ax.set_title(names[j], fontsize=9); ax.set_yticks([]); ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Per-feature distribution: Reference(gray) vs Sample(green)  "
                 "(x = per-feature standardized, dotted = domain ±R1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[Plot] per-feature dist 저장: {out_path}")


def plot_distribution(d2_ref, d2_sam, p, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    hi = np.percentile(np.concatenate([d2_ref, d2_sam]), 99.5)
    bins = np.linspace(0, hi, 60)
    sns.histplot(d2_ref, bins=bins, stat="density", color=C_REF, alpha=0.45, label="Reference", ax=ax, edgecolor=None)
    sns.histplot(d2_sam, bins=bins, stat="density", color=C_SAMPLE, alpha=0.45, label="Sample", ax=ax, edgecolor=None)
    xs = np.linspace(1e-6, hi, 400)
    ax.plot(xs, chi2.pdf(xs, p["K_eff"]), color="#1F77B4", lw=2.2,
            label=f"chi2(K_eff={p['K_eff']}) theoretical")
    ax.axvline(p["T_eff"], color="#333", ls="--", lw=1.6,
               label=f"Domain bound T=chi2({MAHAL_Q},K_eff)={p['T_eff']:.1f}")
    ax.set_xlabel(f"Mahalanobis D^2 (effective {p['K_eff']}-d)"); ax.set_ylabel("density")
    ax.set_title("Mahalanobis distance (effective dim): Reference vs Sample\n"
                 "(Sample shifted right = larger deviation)")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] distribution 저장: {out_path}")


def plot_biplot(p, names, out_path):
    load = p["Vk"] * p["sqrt_lamk"]
    L = load / (np.abs(load).max() + 1e-9)
    colors = sns.color_palette("husl", N_FEATURES)
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#DDD", ls="--", lw=1.0))
    ax.axhline(0, color="#EEE", lw=0.8); ax.axvline(0, color="#EEE", lw=0.8)
    for j in range(N_FEATURES):
        dx, dy = L[j, 0], L[j, 1]
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=colors[j], lw=1.8, alpha=0.9))
        r = np.hypot(dx, dy) + 1e-9
        ax.text(dx + 0.09 * dx / r, dy + 0.09 * dy / r, names[j], color=colors[j],
                fontsize=9, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("whitened PC1"); ax.set_ylabel("whitened PC2")
    ax.set_title(f"Loading biplot - feature directions (top-2 EVR={p['evr_k']*100:.1f}%)\n"
                 "Arrows: original features projected on PC1-PC2 (normalized)")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] biplot 저장: {out_path}")


def plot_corr_clustered(corr, names, out_path):
    """Feature 상관행렬 clustered heatmap (red=유사/양, blue=반대/음)."""
    d = corr.shape[0]
    order = np.arange(d)
    if d >= 3:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        dist = 1.0 - corr; dist = (dist + dist.T) / 2; np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        order = leaves_list(Z)
    C = corr[np.ix_(order, order)]; nm = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(7, 0.5 * d), max(6, 0.45 * d)))
    sns.heatmap(C, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, ax=ax,
                xticklabels=nm, yticklabels=nm, annot=(d <= 20), fmt=".2f",
                annot_kws={"size": 6}, cbar_kws={"label": "correlation"})
    ax.set_title("Feature correlation (clustered)\nred = similar(+corr) / blue = opposite(-corr)")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=7)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] correlation heatmap 저장: {out_path}")


def export_correlation(corr, names, out_csv):
    pd.DataFrame(corr, index=names, columns=names).to_csv(out_csv)
    print(f"[Save] correlation_matrix: {out_csv}")


def pc_dominant_terms(p, names, topn=3):
    """각 유효 PC 를 지배하는 원본 term 산출. 기여율 = loading²(PC별 합=1)."""
    V = p["Veff"]; K = V.shape[1]
    contrib = V ** 2
    evr = (p["sqrt_lam_eff"] ** 2) / p["diagS"].sum()      # PC별 설명분산비
    lines = []
    for i in range(K):
        order = np.argsort(contrib[:, i])[::-1][:topn]
        terms = ", ".join(f"{names[jf]}({contrib[jf, i]*100:.0f}%{'+' if V[jf, i] >= 0 else '-'})"
                          for jf in order)
        lines.append((i + 1, float(evr[i]), terms))
    return contrib, evr, lines


def export_pc_loadings(p, names, out_csv):
    """feature × '전체' PC loading + 기여율(%) CSV (유효차원뿐 아니라 전 PC 출력)."""
    V = p["V_full"]; d = V.shape[1]; contrib = V ** 2
    evr = p["lam_full"] / p["lam_full"].sum()
    df = pd.DataFrame({"feature": names})
    for i in range(d):                                   # 전체 PC 출력
        df[f"PC{i+1}_loading"] = V[:, i]
        df[f"PC{i+1}_contrib%"] = contrib[:, i] * 100
    # 각 PC 의 설명분산비(EVR)와 유효차원 여부를 마지막 요약 행으로 덧붙임
    meta = {"feature": "__EVR%__"}
    for i in range(d):
        meta[f"PC{i+1}_loading"] = np.nan
        meta[f"PC{i+1}_contrib%"] = round(float(evr[i]) * 100, 4)
    df = pd.concat([df, pd.DataFrame([meta])], ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"[Save] pc_loadings (전체 {d} PC): {out_csv}")


def plot_pc_loadings_heatmap(p, names, out_path):
    """유효 PC 별 원본 feature loading 히트맵 (어느 term 이 각 PC 지배하는지)."""
    V = p["Veff"]; K = V.shape[1]
    evr = (p["sqrt_lam_eff"] ** 2) / p["diagS"].sum()
    fig, ax = plt.subplots(figsize=(min(2 + 1.3 * K, 14), max(6, 0.4 * len(names))))
    sns.heatmap(V, cmap="coolwarm", center=0, ax=ax, annot=(len(names) <= 25),
                fmt=".2f", annot_kws={"size": 7},
                xticklabels=[f"PC{i+1}\n(EVR {evr[i]*100:.0f}%)" for i in range(K)],
                yticklabels=names, cbar_kws={"label": "loading (eigenvector component)"})
    ax.set_xlabel("effective PC"); ax.set_ylabel("feature")
    ax.set_title("PC loadings: which feature dominates each effective PC")
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] pc-loadings heatmap 저장: {out_path}")


def plot_ood_attribution(cls, names, out_path, topn=15):
    ood = cls["cat"] == 2
    if ood.sum() == 0:
        print("[Plot] OOD 없음 → attribution 생략"); return None
    cond = cls["c"][ood].mean(0); uncond = cls["uu"][ood].mean(0)
    order = np.argsort(cond)[::-1][:topn]; y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y + 0.2, cond[order], height=0.4, color="#D62728", label="Conditional c_j (MYT)")
    ax.barh(y - 0.2, uncond[order], height=0.4, color="#F0A0A0", label="Unconditional u_j")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Mean Mahalanobis^2 contribution"); ax.set_ylabel("feature")
    ax.set_title(f"OOD points ({int(ood.sum())}) attribution ranking\nHigh c_j = main driver of leaving domain")
    ax.legend(fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ood-attribution 저장: {out_path}")
    return [(names[i], float(cond[i])) for i in order[:5]]


def plot_gap_attribution(diff, n_gap, n_cov, names, out_path, topn=15):
    if n_gap == 0:
        print("[Plot] Gap 없음 → 생략"); return
    order = np.argsort(np.abs(diff))[::-1][:topn]
    order = order[np.argsort(diff[order])]
    y = np.arange(len(order)); colors = ["#D62728" if diff[i] > 0 else "#1F77B4" for i in order]
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * len(order))))
    ax.barh(y, diff[order], color=colors); ax.axvline(0, color="#333", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("Gap-cell mean - Covered-cell mean (scaled)"); ax.set_ylabel("feature")
    ax.set_title(f"Gap region driver features (gap ref={n_gap}, cover ref={n_cov})\n"
                 "Red = higher in gap / Blue = higher in covered")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] gap-attribution 저장: {out_path}")


def plot_ood_heatmap(cls, names, out_path, topn=OOD_HEATMAP_TOPN):
    ood = np.where(cls["cat"] == 2)[0]
    if len(ood) == 0:
        print("[Plot] OOD 없음 → 히트맵 생략"); return
    ood = ood[np.argsort(cls["d2"][ood])[::-1][:topn]]
    C = cls["c"][ood]; fcol = np.argsort(C.sum(0))[::-1]; C = C[:, fcol]
    fig, ax = plt.subplots(figsize=(min(16, 0.5 * N_FEATURES + 3), max(6, 0.32 * len(ood))))
    sns.heatmap(C, cmap="Reds", ax=ax, cbar_kws={"label": "c_j (conditional contribution)"},
                xticklabels=[names[i] for i in fcol],
                yticklabels=[f"#{i} (D2={cls['d2'][i]:.0f})" for i in ood])
    ax.set_xlabel("feature"); ax.set_ylabel("extreme OOD points")
    ax.set_title(f"Top-{len(ood)} extreme OOD points: per-feature attribution (c_j)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] ood-heatmap 저장: {out_path}")


# =============================================================================
# 결과 저장
# =============================================================================
_COVER_LABEL = np.array(["covered", "in_domain_no_ref", "out_of_domain"])


def export_sample(sample_path, names, cls, fmt, d2_ref, out_path):
    if fmt == "csv":
        df = pd.read_csv(sample_path)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table(sample_path).to_pandas()
    df.columns = [str(c).strip() for c in df.columns]
    X = df.loc[:, names].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    pc1 = np.full(len(df), np.nan); pc2 = np.full(len(df), np.nan)
    d2 = np.full(len(df), np.nan); pctl = np.full(len(df), np.nan)
    status = np.array([""] * len(df), dtype=object)
    covered = np.zeros(len(df), dtype=bool)
    pc1[finite] = cls["u2"][:, 0]; pc2[finite] = cls["u2"][:, 1]
    d2[finite] = cls["d2"]; status[finite] = _COVER_LABEL[cls["cat"]]
    covered[finite] = (cls["cat"] == 0)
    # soft 지표: Reference D²_eff 분포 대비 백분위(0~1). 0.5=평범, 1=ref 최외곽, >q=이탈
    ref_sorted = np.sort(d2_ref)
    pctl[finite] = np.searchsorted(ref_sorted, cls["d2"], side="right") / len(ref_sorted)
    df["PC1"], df["PC2"] = pc1, pc2
    df["D2_eff"] = d2                     # 유효차원 Mahalanobis²(=Σ 유효PC whitened²)
    df["D2_eff_ref_pctl"] = pctl          # soft: Reference 분포 대비 백분위
    df["cover_status"], df["covered"] = status, covered
    df.to_csv(out_path, index=False)
    print(f"[Save] sample_pc_cover: {out_path} (rows={len(df):,}, covered={int(covered.sum()):,})")


def export_uncovered_ref(ref_pts, p, sam_cells, names, out_path):
    u = whiten_topk(ref_pts, p)
    in_ell = (u ** 2).sum(1) <= p["Tk"]
    idx = np.empty(u.shape, dtype=np.int16)
    for a in range(u.shape[1]):
        idx[:, a] = np.digitize(u[:, a], p["internalK"])
    keys = [row.tobytes() for row in idx]
    unc = np.array([in_ell[i] and (keys[i] not in sam_cells) for i in range(len(ref_pts))])
    feats = ref_pts[unc] * p["scale"] + p["center"]        # 표준화 역변환 → 원본 스케일
    out = pd.DataFrame(feats, columns=names)
    out["PC1"], out["PC2"] = u[unc, 0], u[unc, 1]
    out.to_csv(out_path, index=False)
    print(f"[Save] reference_uncovered: {out_path} (uncovered={int(unc.sum()):,}/{len(ref_pts):,})")


def export_per_feature(cov, w_s, cls, sample_raw, p, ref_feat_bins, names, out_path):
    d = len(names)
    rows = []
    for j in range(d):
        rows.append({
            "feature": names[j],
            "coverage_1d": round(float(cov[j]), 6),
            "ref_bins": len(ref_feat_bins[j]),
            "ref_mean": float(p["mean_raw"][j]), "ref_std": float(p["std_raw"][j]),
            "sample_mean": float(sample_raw[:, j].mean()), "sample_std": float(sample_raw[:, j].std()),
            "sample_mean_abs_w": float(np.abs(w_s[:, j]).mean()),   # feature 이탈 크기(표준화)
            "sample_mean_cj": float(cls["c"][:, j].mean()),         # MYT 조건부 기여 평균
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[Save] per_feature_coverage: {out_path} (mean cov={cov.mean()*100:.2f}%)")


# =============================================================================
# 메인
# =============================================================================
def run():
    global N_FEATURES
    for pth in (SAMPLE_PATH, REF_PATH):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"입력 파일 없음: {pth}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    names = resolve_feature_names(SAMPLE_PATH, REF_PATH, FMT)
    N_FEATURES = len(names)

    # Reference 고정 프레임 + pass2 (캐시 우선)
    R = load_or_build_reference(REF_PATH, names, FMT)
    p = R["p"]; ref_cells = R["ref_cells"]; dom = p["domain_set"]
    ref_pts = R["ref_pts"]; d2_ref = R["d2_ref"]; ref_feat_bins = R["feat_bins"]

    # OOD 경계(soft): empirical=Reference 실측 D²_eff 분위수 / chi2=이론
    d2_ref = R["d2_ref"]
    if OOD_BOUND == "empirical":
        T_ood = float(np.quantile(d2_ref, MAHAL_Q))
        ref_ood_rate = float(np.mean(d2_ref > T_ood))
    else:
        T_ood = p["T_eff"]; ref_ood_rate = R["ref_out"] / max(R["ref_tot"], 1)

    # Sample → 고정 프레임에 투영
    sample_raw = read_matrix_by_name(SAMPLE_PATH, names, FMT)
    z_s = standardize(sample_raw, p)
    cls = classify_points(z_s, p, ref_cells, T_ood)
    w_s = (z_s - p["mu"]) / np.sqrt(p["diagS"])
    sam_cells = mahal_sample_cells(z_s, p)

    # 커버리지 지표
    ref_cov = len(ref_cells & dom) / len(dom)
    sam_cov = len(sam_cells & dom) / len(dom)
    verified = len(sam_cells & ref_cells & dom)
    denom_ref = len(ref_cells & dom)
    gap = denom_ref - verified
    cov_of_ref = verified / denom_ref if denom_ref else 0.0
    feat_cov = per_feature_coverage(w_s, p, ref_feat_bins)      # ★ 헤드라인
    headline = float(feat_cov.mean())
    cat = cls["cat"]
    n_ver, n_ind, n_ood = (cat == 0).sum(), (cat == 1).sum(), (cat == 2).sum()
    sam_out = int(n_ood)

    # per-feature ref 분포(reservoir) + 유효차원 whitened 좌표
    w_r = (ref_pts - p["mu"]) / np.sqrt(p["diagS"])
    u_s, u_r = cls["u2"][:, :2], whiten_topk(ref_pts, p)[:, :2]
    ueff_s = (z_s - p["mu"]) @ p["Veff"] / p["sqrt_lam_eff"]
    ueff_r = (ref_pts - p["mu"]) @ p["Veff"] / p["sqrt_lam_eff"]
    gap_diff, n_gap_r, n_cov_r = gap_feature_diff(ref_pts, p, sam_cells)

    # 시각화
    plot_perfeature_coverage(feat_cov, names, j("per_feature_coverage_bars.png"))
    plot_perfeature_dist(w_s, w_r, names, p, j("per_feature_distributions.png"))
    plot_effective_pairscatter(ueff_s, ueff_r, p["K_eff"], j("effective_dims_pairscatter.png"))
    plot_coverage_2d(u_s, u_r, ref_cells, sam_cells, p, sam_cov, j("domain_mahalanobis_2d.png"))
    plot_scatter_ref_sample(u_s, u_r, j("domain_scatter_ref_sample.png"))
    plot_distribution(d2_ref, cls["d2"], p, j("domain_mahal_distribution.png"))
    plot_biplot(p, names, j("domain_loadings_biplot.png"))
    plot_pc_loadings_heatmap(p, names, j("pc_loadings_heatmap.png"))
    plot_corr_clustered(p["corr"], names, j("feature_correlation.png"))
    top_ood = plot_ood_attribution(cls, names, j("domain_ood_attribution.png"))
    plot_gap_attribution(gap_diff, n_gap_r, n_cov_r, names, j("domain_gap_attribution.png"))
    plot_ood_heatmap(cls, names, j("domain_ood_heatmap.png"))

    # 저장
    export_sample(SAMPLE_PATH, names, cls, FMT, d2_ref, j("sample_pc_cover.csv"))
    export_uncovered_ref(ref_pts, p, sam_cells, names, j("reference_uncovered_pc.csv"))
    export_per_feature(feat_cov, w_s, cls, sample_raw, p, ref_feat_bins, names, j("per_feature_coverage.csv"))
    _, _, pc_lines = pc_dominant_terms(p, names)
    export_pc_loadings(p, names, j("pc_loadings.csv"))
    export_correlation(p["corr"], names, j("feature_correlation.csv"))

    # 리포트
    print("\n" + "=" * 70)
    print("  [Mahalanobis Domain Coverage & Attribution]  (Reference-anchored)")
    print("=" * 70)
    print(f"  >>> Headline: per-feature coverage 평균 : {headline*100:8.4f}%  (21-d 완화·직관 지표)")
    print(f"  (보조) top-2 격자 Sample coverage        : {sam_cov*100:8.4f}%   (분모=G_total {len(dom)})")
    print(f"  (보조) Ref-footprint coverage            : {cov_of_ref*100:8.4f}%   (분모=ref셀 {denom_ref})")
    print(f"  [셀] Cat1 대변={verified:,}  Cat2 gap={gap:,}  (of ref {denom_ref:,})")
    print(f"  [진단] 유효차원 K_eff={p['K_eff']} (PR={p['eff_dim']:.2f}, EVR={p['evr_eff']*100:.1f}%)")
    print(f"  [진단] OOD 경계 방식={OOD_BOUND} (q={MAHAL_Q}, T_ood={T_ood:.1f})")
    print(f"  [진단] Sample OOD (eff {p['K_eff']}-d)          : {sam_out/max(len(z_s),1)*100:.2f}%  "
          f"(Ref OOD={ref_ood_rate*100:.2f}%)")
    print(f"  [Sample 포인트] 대변={n_ver:,}  도메인내={n_ind:,}  OOD={n_ood:,}  (총 {len(z_s):,})")
    if top_ood:
        print("  OOD 근원 항 top-5 (조건부 c_j)          : "
              + ", ".join(f"{nm}={v:.1f}" for nm, v in top_ood))
    print("  PC별 지배 term (기여율%, 부호):")
    for i, evr_i, terms in pc_lines:
        print(f"     PC{i} (EVR {evr_i*100:4.1f}%): {terms}")
    print("=" * 70)
    return {"headline_cov": headline, "sam_cov": sam_cov, "cov_of_ref": cov_of_ref,
            "verified": verified, "gap": gap, "n_ood": int(n_ood), "top_ood": top_ood}


def mahal_sample_cells(z, p):
    """Sample 의 top-2 타원 안 점유 셀 집합."""
    e = z - p["mu"]
    u = (e @ p["Vk"]) / p["sqrt_lamk"]
    uu = u[(u ** 2).sum(1) <= p["Tk"]]
    if len(uu) == 0:
        return set()
    return _cells_of(uu, p["internalK"], p["Vk"].shape[1])


if __name__ == "__main__":
    run()
