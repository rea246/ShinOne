# -*- coding: utf-8 -*-
"""
coverage_domain_pca.py
======================
[REF 기준 PCA-2D + KDE 99% 등고선 Coverage] 전용 분석 (self-contained)

목적
    Reference(REF) 21차원과 Sample(SAMPLE) 21차원이 들어올 때
        (1) REF 기준으로 PCA 를 학습해 2D 로 투영한다.
        (2) 2D 평면에서 REF·SAMPLE 각각의 확률밀도(KDE) 분포를 그리고,
            REF 99% 등고선과 SAMPLE 99% 등고선을 겹쳐 그린 뒤
            겹치는 영역(면적)을 Coverage 로 정의해 산출한다.
        (3) PCA(PC1·PC2)에 영향이 큰 feature 8개를 추려 교차 scatter(pairplot)를 본다.
        (4) REF 에서 'cover 되지 못한 point' 와 '99% 밖에 분포한 point' 행만 추려
            group 태그를 붙여 CSV 로 저장한다.
        (5) SAMPLE 에서 'REF 를 cover 하는 point' 와 'cover 못하는 point' 를 나눠
            group 태그 열을 붙여 CSV 로 저장한다.

도메인/커버리지 정의 (KDE 99% HDR)
    - PCA 는 REF 를 표준화(REF 평균·표준편차)한 상관행렬의 고유분해로 학습한다.
      투영:  z = (x − μ_ref)/σ_ref ,  score = z · V[:, :2]
    - REF·SAMPLE 각각 2D score 로 gaussian KDE 를 적합한다.
    - "99% 등고선" = 학습 포인트 밀도의 하위 1% 밀도값을 임계치(level)로 잡은
      최고밀도영역(HDR, Highest Density Region). 이 안이 분포의 99% 를 포함한다.
        REF_99  = { density_ref ≥ level_ref }
        SAM_99  = { density_sam ≥ level_sam }
    - Coverage(면적 기반, 격자 적분):
        overlap = REF_99 ∩ SAM_99
        Coverage(SAMPLE→REF) = area(overlap) / area(REF_99)     ← 주지표
        Jaccard              = area(overlap) / area(REF_99 ∪ SAM_99)

포인트 분류(등고선 소속으로 판정, in/out = KDE density ≥ level)
    REF   : outside_99   = REF 99% 등고선 밖
            uncovered    = REF 99% 안 이지만 SAMPLE 99% 등고선 밖(대변 안 됨)
    SAMPLE: cover_ref    = REF 99% 등고선 안(=REF 를 대변/cover)
            not_cover_ref= REF 99% 등고선 밖

데이터 입출력 (coverage_domain_mahal.py 관례 계승)
    * Feature 는 CONFIG(위치 FEATURE_COL_IDX 또는 이름 FEATURE_COLS)로 직접 지정.
      위치로 골라도 Reference 는 '이름'으로 매칭 → 두 파일 열 순서가 달라도 안전.
    * plot 라벨의 feature 이름 = CSV 컬럼 이름.
    * 비유한(NaN/Inf) 행은 읽는 즉시 제거하고 개수를 보고한다.

성능(속도) 설계  ── "전처리 1회 캐시 + 재사용", 격자 lookup, 스레드
    (1) [전처리 캐시] REF feature 행렬을 '한 번만' 파싱해 float32 .npy 로 캐시하고
        (파일 크기·mtime·feature 목록으로 키 생성) 이후 실행/단계에서 재사용한다.
        → CSV 를 3번 읽던 것을 첫 실행 1~2회, 재실행 0~1회로 줄인다.
        캐시 무효화는 자동(파일이 바뀌면 키가 달라짐). USE_CACHE=False 로 끌 수 있다.
    (2) [격자 lookup 분류] 종전엔 REF 전 행마다 gaussian_kde 를 호출(=O(행수×KDE학습점))
        해 대용량에서 폭발했다. 이제 KDE 는 격자에서 '한 번만' 평가해 99% 마스크를 만들고,
        각 포인트는 격자 셀 인덱스로 O(1) 조회한다. → 분류가 O(N) 로 선형화.
    (3) [KDE 표본 상한] KDE 적합에 KDE_MAX_PTS(기본 6000)만 사용(정확도 충분·속도↑).
    (4) [스레드] 격자 KDE 평가를 N_THREADS 로 분할(밀도 격자 평가만; gaussian_kde 의
        C 루프는 GIL 을 풀어 스레드 병렬이 유효). 기본 1(단일).

실행:  python coverage_domain_pca.py
       (필요 시: pip install seaborn scipy scikit-learn matplotlib pandas)
"""

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")                 # 헤드리스 백엔드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
# =============================================================================
SAMPLE_PATH   = "sample.csv"
REF_PATH      = "reference.csv"
FMT           = "csv"           # 'csv' 또는 'parquet'

# Feature 컬럼 선택 ── CSV 에 feature + 좌표/hash/name 이 섞여 있으므로 '직접' 지정.
#   (우선순위: FEATURE_COLS 가 있으면 그것, 없으면 FEATURE_COL_IDX 사용)
#   FEATURE_COLS    : 이름으로 지정. 예) ["f0","f1",...]
#   FEATURE_COL_IDX : 위치로 지정. Sample 헤더의 0-based 열 위치. 예) list(range(0,21))
# ※ 위치로 골라도 Reference 에서는 '이름'으로 매칭하므로 두 파일의 열 순서가 달라도 안전.
FEATURE_COLS    = None
FEATURE_COL_IDX = list(range(0, 21))    # REF/SAMPLE 앞 21차원

HDR_Q         = 0.99            # 등고선(HDR) 분위수 = 99%
GRID_N        = 240            # 밀도 적분/등고선용 격자 해상도(축당)
GRID_MARGIN   = 0.08           # 격자 여백(데이터 범위 대비 비율)

N_TOP_FEATURES = 8             # PCA 영향 큰 feature 상위 N (교차 scatter 대상)
PAIR_MAX_PTS   = 3000          # pairplot 소스별 최대 표본 수(과밀 방지)

CHUNKSIZE      = 200_000
OUTPUT_DIR     = "coverage_pca_plots"
PLOT_DOWNSAMPLE = 20_000       # 시각화용 Reference 표본 수(산점/pairplot)

# ---- 성능(속도) 옵션 ----
USE_CACHE      = True          # REF feature 행렬을 .npy 로 1회 캐시하고 재사용
CACHE_DIR      = "coverage_pca_cache"
KDE_MAX_PTS    = 6000          # KDE 적합에 쓰는 최대 표본 수(속도/정확도 균형)
N_THREADS      = max(1, (os.cpu_count() or 1))   # 격자 KDE 평가 스레드 수(1=단일)
REF_EXPORT_ORIGINAL_COLS = True   # True: 원본 전체 컬럼 유지(원본 파일 1회 재읽기)
                                  # False: feature 컬럼만 저장(캐시만 사용, 파일 재읽기 0)

sns.set_theme(style="white", context="talk")   # 내부 격자(gridline) 없음

# ---- 색상 팔레트 ----
C_SAMPLE = "#2CA02C"           # sample (초록)
C_REF    = "#8C8C8C"           # reference (회색)
C_REF_L  = "#5B8DEF"           # reference 등고선(파랑)
C_OVERLAP = "#F4A15A"          # 겹침 영역(주황)

N_FEATURES = None              # run() 에서 feature 수로 설정되는 모듈 전역


# =============================================================================
# 데이터 I/O ── 컬럼 이름 기준 선택/정렬 + 비유한 행 제거
#   (coverage_domain_mahal.py 의 I/O 관례를 계승. 단, 스트리밍은 '원본(raw)' 반환)
# =============================================================================
def _header(path, fmt):
    """파일 헤더의 컬럼 이름 목록(공백 strip)."""
    if fmt == "csv":
        cols = list(pd.read_csv(path, nrows=0).columns)
    else:
        import pyarrow.parquet as pq
        cols = list(pq.ParquetFile(path).schema_arrow.names)
    return [str(c).strip() for c in cols]


def _wanted_from_config(s_cols):
    """CONFIG(FEATURE_COLS 우선, 없으면 FEATURE_COL_IDX)로 '사용할 feature 이름'을 정한다."""
    if FEATURE_COLS is not None:
        return [str(c).strip() for c in FEATURE_COLS]
    if FEATURE_COL_IDX is not None:
        bad = [i for i in FEATURE_COL_IDX if i < 0 or i >= len(s_cols)]
        if bad:
            raise ValueError(
                f"FEATURE_COL_IDX 위치가 Sample 열 범위를 벗어남: {bad} "
                f"(Sample 열 {len(s_cols)}개: {s_cols})")
        wanted = [s_cols[i] for i in FEATURE_COL_IDX]
        print(f"[IO] 위치 기준 feature 선택(Sample idx {FEATURE_COL_IDX[0]}..{FEATURE_COL_IDX[-1]}, "
              f"{len(wanted)}개): {wanted}")
        return wanted
    raise ValueError("feature 컬럼을 지정하세요: FEATURE_COLS(이름) 또는 FEATURE_COL_IDX(위치).")


def resolve_feature_names(sample_path, ref_path, fmt):
    """
    '공통 feature 이름 목록'을 확정한다(양쪽에 모두 있는 컬럼만 사용).
    - 위치(FEATURE_COL_IDX)로 골라도 Reference 에서는 '이름'으로 매칭한다(열 순서 무관).
    - 한쪽에 없거나 중복된 컬럼은 '경고만 하고 제외'하고 계속 진행(exit 하지 않음).
    """
    s_cols, r_cols = _header(sample_path, fmt), _header(ref_path, fmt)
    wanted = _wanted_from_config(s_cols)

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
    """DataFrame 에서 names 컬럼만 그 순서로 뽑아 float ndarray + 비유한 행 마스크."""
    df.columns = [str(c).strip() for c in df.columns]
    try:
        X = df.loc[:, names].to_numpy(dtype=np.float64)   # ★ 이름으로 선택·정렬
    except (ValueError, TypeError) as e:
        raise ValueError(f"feature 컬럼을 수치로 변환 실패(비수치 값 포함?): {e}")
    finite = np.isfinite(X).all(axis=1)
    return X, finite


def read_matrix_by_name(path, names, fmt):
    """작은 파일(Sample) 전체를 (n, d) float64 로 읽는다(feature 열만 선택 + 정제)."""
    want = set(names)
    if fmt == "csv":
        df = pd.read_csv(path, usecols=lambda c: str(c).strip() in want)
    else:
        import pyarrow.parquet as pq
        raw = [c for c in pq.ParquetFile(path).schema_arrow.names if str(c).strip() in want]
        df = pq.read_table(path, columns=raw).to_pandas()
    X, finite = _select(df, names)
    dropped = int((~finite).sum())
    if dropped:
        print(f"[IO] Sample 비유한(NaN/Inf) 행 {dropped:,}개 제거")
    X = X[finite]
    if len(X) == 0:
        raise ValueError("Sample 에 유효한 행이 없습니다(전부 NaN/Inf?).")
    return X


def iter_ref_chunks_raw(path, names, chunksize, fmt):
    """Reference 를 chunk 스트리밍 → 이름 선택·정렬 → 정제 → '원본(raw)' 배열 yield."""
    dropped_total = [0]
    want = set(names)
    if fmt == "csv":
        for chunk in pd.read_csv(path, usecols=lambda c: str(c).strip() in want,
                                 chunksize=chunksize):
            X, finite = _select(chunk, names)
            dropped_total[0] += int((~finite).sum())
            X = X[finite]
            if len(X):
                yield X
    else:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        raw = [c for c in pf.schema_arrow.names if str(c).strip() in want]
        for batch in pf.iter_batches(batch_size=chunksize, columns=raw):
            X, finite = _select(batch.to_pandas(), names)
            dropped_total[0] += int((~finite).sum())
            X = X[finite]
            if len(X):
                yield X
    if dropped_total[0]:
        print(f"[IO] Reference 비유한 행 누적 {dropped_total[0]:,}개 제거")


# =============================================================================
# [전처리 캐시] REF feature 행렬을 '1회' 파싱해 float32 .npy 로 캐시하고 재사용
# =============================================================================
def _cache_key(path, names):
    """파일 stat(크기·mtime) + feature 목록으로 캐시 키 생성(파일 바뀌면 자동 무효화)."""
    st = os.stat(path)
    sig = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}|{','.join(names)}"
    return hashlib.md5(sig.encode()).hexdigest()[:16]


def load_ref_matrix(path, names, fmt):
    """
    REF 의 feature 행렬 (N, d) float32 를 반환한다(비유한 행 제거 후).
    USE_CACHE 면 <CACHE_DIR>/ref_<key>.npy 로 캐시하고, 이후엔 mmap 으로 즉시 로드한다.
    → CSV 재파싱(가장 큰 비용)을 실행 간·단계 간 1회로 줄이는 핵심.
    """
    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(CACHE_DIR, f"ref_{_cache_key(path, names)}.npy")
        if os.path.exists(cache_path):
            X = np.load(cache_path, mmap_mode="r")
            print(f"[Cache] REF feature 캐시 재사용: {cache_path}  (rows={len(X):,}, dim={X.shape[1]})")
            return X
    # 캐시 미스 → 1회 스트리밍 파싱하여 행렬 구성
    parts = list(iter_ref_chunks_raw(path, names, CHUNKSIZE, fmt))
    if not parts:
        raise ValueError("Reference 에 유효한 행이 없습니다(전부 NaN/Inf?).")
    X = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if USE_CACHE:
        np.save(cache_path, X)
        print(f"[Cache] REF feature 캐시 저장: {cache_path}  (rows={len(X):,}, dim={X.shape[1]})")
    else:
        print(f"[IO] REF feature 파싱 완료(캐시 off): rows={len(X):,}, dim={X.shape[1]}")
    return X


def subsample_rows(X, k, seed=0):
    """행렬에서 최대 k행 균일 표본 추출(인덱스 정렬 → mmap 접근 효율). (표본배열, 인덱스) 반환."""
    n = len(X)
    if n <= k:
        return np.asarray(X, dtype=np.float64), np.arange(n)
    idx = np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    return np.asarray(X[idx], dtype=np.float64), idx


# =============================================================================
# [PCA] REF 모멘트(μ, Σ) → 표준화 상관행렬 PCA 학습
# =============================================================================
def moments_from_matrix(X, chunk=500_000):
    """캐시된 REF 행렬에서 [n, Σx, ΣxxᵀT] 를 float64 누적으로 집계(청크로 메모리·정밀도 관리)."""
    d = X.shape[1]
    n = 0
    sum_x = np.zeros(d)
    sum_xx = np.zeros((d, d))
    for i in range(0, len(X), chunk):
        b = np.asarray(X[i:i + chunk], dtype=np.float64)
        n += b.shape[0]
        sum_x += b.sum(axis=0)
        sum_xx += b.T @ b
    return n, sum_x, sum_xx


def build_pca(n, sum_x, sum_xx):
    """
    REF μ·Σ 로 '표준화 상관행렬 PCA' 를 학습한다.
    반환 dict: mu, sd(표준편차), V(고유벡터 d×d), evr(설명분산비 d,),
               Vk(=V[:, :2]), evr2, load(상관 로딩 d×2), feat_imp(feature 중요도 d,).
    """
    mu = sum_x / n
    cov = sum_xx / n - np.outer(mu, mu)               # raw 공분산
    d = len(mu)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))  # 표준편차(분산 0 방지)
    corr = cov / np.outer(sd, sd)                     # 상관행렬(=표준화 공분산)
    corr = (corr + corr.T) / 2                         # 대칭 보정(수치)
    corr += 1e-9 * np.eye(d)                           # 수치 안정화

    w, V = np.linalg.eigh(corr)                        # 오름차순 고유값
    order = np.argsort(w)[::-1]                         # 큰 고유값 순
    w, V = np.clip(w[order], 0, None), V[:, order]
    evr = w / w.sum()

    Vk = V[:, :2]                                       # PC1·PC2 방향
    load = Vk * np.sqrt(w[:2])                          # 상관 로딩(방향·상대크기)
    # feature 중요도: PC1·PC2 분산에서 각 feature 가 차지하는 몫(설명분산 가중)
    feat_imp = (Vk[:, 0] ** 2) * evr[0] + (Vk[:, 1] ** 2) * evr[1]

    print(f"[PCA] REF 표준화 상관행렬 PCA 학습(dim={d}); "
          f"PC1 EVR={evr[0]*100:.1f}%, PC2 EVR={evr[1]*100:.1f}% "
          f"(top-2 합={evr[:2].sum()*100:.1f}%)")
    return {"mu": mu, "sd": sd, "V": V, "evr": evr, "Vk": Vk, "evr2": float(evr[:2].sum()),
            "load": load, "feat_imp": feat_imp}


def pca_project(X_raw, p):
    """raw feature (n, d) → PC1·PC2 score (n, 2).  z=(x−μ)/σ , score=z·Vk."""
    z = (X_raw - p["mu"]) / p["sd"]
    return z @ p["Vk"]


# =============================================================================
# [KDE / HDR] 2D 밀도 + 99% 최고밀도영역(등고선) 임계치
# =============================================================================
def fit_kde(scores, max_pts=None, seed=0):
    """
    2D score (n, 2) → gaussian_kde.
    max_pts 초과 시 균일 표본으로 축소해 적합(속도↑, 밀도추정 정확도엔 거의 영향 없음).
    """
    s = scores
    if max_pts is not None and len(scores) > max_pts:
        idx = np.random.default_rng(seed).choice(len(scores), max_pts, replace=False)
        s = scores[idx]
    return gaussian_kde(s.T)


def kde_eval(kde, pts, n_threads=1):
    """
    KDE 를 (2, m) 평가점에서 계산. n_threads>1 이면 평가점을 나눠 스레드 병렬 실행.
    (gaussian_kde 의 C 커널 루프는 GIL 을 풀어 스레드 병렬이 유효)
    """
    m = pts.shape[1]
    if n_threads <= 1 or m < 20_000:
        return kde(pts)
    splits = np.array_split(np.arange(m), n_threads)
    out = np.empty(m)
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futs = {ex.submit(kde, pts[:, s]): s for s in splits if len(s)}
        for fut, s in futs.items():
            out[s] = fut.result()
    return out


def hdr_level(kde, scores, q, n_threads=1):
    """
    HDR(최고밀도영역) 임계 밀도값 = 학습 포인트 밀도의 하위 (1−q) 분위수.
    density ≥ level 인 영역이 분포의 약 q(=99%) 를 포함한다.
    """
    dens = kde_eval(kde, scores.T, n_threads)
    return float(np.percentile(dens, (1.0 - q) * 100.0))


def make_grid(scores_a, scores_b, n, margin):
    """두 점군을 모두 감싸는 공통 격자(가장자리 좌표 xs, ys + 셀 중심 XX,YY + 셀면적)."""
    allpts = np.vstack([scores_a, scores_b])
    lo = allpts.min(axis=0); hi = allpts.max(axis=0)
    pad = (hi - lo) * margin + 1e-9
    lo -= pad; hi += pad
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    XX, YY = np.meshgrid(xs, ys)                       # (n, n) — XX[i,j]=xs[j], YY[i,j]=ys[i]
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    return xs, ys, XX, YY, cell_area


def eval_grid(kde, XX, YY, n_threads=1):
    """격자 위 밀도값 (n, n).  스레드 병렬 평가 지원."""
    pts = np.vstack([XX.ravel(), YY.ravel()])
    return kde_eval(kde, pts, n_threads).reshape(XX.shape)


def classify_by_grid(scores, xs, ys, mask):
    """
    각 포인트를 '격자 셀 마스크 조회'로 in/out 판정한다(KDE 재호출 없이 O(N)).
    격자(xs, ys)는 등간격 linspace → 셀 인덱스는 반올림으로 O(1). 격자 밖은 False.
    mask 형상은 (len(ys), len(xs)) = (row=y, col=x).
    """
    nx, ny = len(xs), len(ys)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    sx, sy = scores[:, 0], scores[:, 1]
    inb = (sx >= xs[0]) & (sx <= xs[-1]) & (sy >= ys[0]) & (sy <= ys[-1])
    ix = np.clip(np.round((sx - xs[0]) / dx).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.round((sy - ys[0]) / dy).astype(np.int64), 0, ny - 1)
    res = np.zeros(len(scores), dtype=bool)
    res[inb] = mask[iy[inb], ix[inb]]
    return res


# =============================================================================
# 시각화 ① REF·SAMPLE KDE 밀도 + 99% 등고선 + 겹침(Coverage) 영역
# =============================================================================
def plot_density_coverage(XX, YY, dens_r, dens_s, lvl_r, lvl_s,
                          scores_r, scores_s, cov_txt, out_path):
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(10, 9))

    # 밀도 배경(옅게): REF 회색조, SAMPLE 초록조
    ax.contourf(XX, YY, dens_r, levels=12, cmap="Greys", alpha=0.32)
    ax.contourf(XX, YY, dens_s, levels=12, cmap="Greens", alpha=0.28)

    # 겹침(Coverage) 영역 = REF_99 ∩ SAM_99 를 주황으로 채움
    overlap = ((dens_r >= lvl_r) & (dens_s >= lvl_s)).astype(float)
    ax.contourf(XX, YY, overlap, levels=[0.5, 1.5], colors=[C_OVERLAP], alpha=0.45)

    # 99% 등고선(굵게)
    ax.contour(XX, YY, dens_r, levels=[lvl_r], colors=[C_REF_L], linewidths=2.4)
    ax.contour(XX, YY, dens_s, levels=[lvl_s], colors=[C_SAMPLE], linewidths=2.4)

    # 산점(옅게)
    ax.scatter(scores_r[:, 0], scores_r[:, 1], s=5, c=C_REF, alpha=0.10, linewidths=0)
    ax.scatter(scores_s[:, 0], scores_s[:, 1], s=7, c=C_SAMPLE, alpha=0.12, linewidths=0)

    handles = [
        Line2D([0], [0], color=C_REF_L, lw=2.4, label=f"REF {int(HDR_Q*100)}% contour"),
        Line2D([0], [0], color=C_SAMPLE, lw=2.4, label=f"SAMPLE {int(HDR_Q*100)}% contour"),
        Patch(facecolor=C_OVERLAP, alpha=0.5, label="Coverage (overlap)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=10, framealpha=0.9)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"REF-PCA 2D density & {int(HDR_Q*100)}% coverage\n" + cov_txt, fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (1) density-coverage 저장: {out_path}")


# =============================================================================
# 시각화 ①' REF vs SAMPLE 산점만 (PCA 2D)
# =============================================================================
def plot_scatter_ref_sample(scores_r, scores_s, p, out_path):
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.scatter(scores_r[:, 0], scores_r[:, 1], s=8, c=C_REF, alpha=0.6, linewidths=0, label="Reference")
    ax.scatter(scores_s[:, 0], scores_s[:, 1], s=12, c=C_SAMPLE, alpha=0.6, linewidths=0, label="Sample")
    ax.set_xlabel(f"PC1 (EVR={p['evr'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 (EVR={p['evr'][1]*100:.1f}%)")
    ax.set_title("Reference vs Sample scatter (REF-PCA 2D)")
    ax.set_aspect("equal"); ax.legend(fontsize=10, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (1') ref-sample scatter 저장: {out_path}")


# =============================================================================
# 시각화 ② PCA feature 중요도(상위) + 로딩 biplot
# =============================================================================
def plot_feature_importance(p, names, top_idx, out_path):
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * N_FEATURES)))
    order = np.argsort(p["feat_imp"])[::-1]
    y = np.arange(len(order))[::-1]
    colors = ["#D62728" if i in set(top_idx) else "#BBBBBB" for i in order]
    ax.barh(y, p["feat_imp"][order], color=colors)
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("PC1·PC2 variance share (importance)")
    ax.set_title(f"PCA feature importance (top-{len(top_idx)} = red)\n"
                 f"top-2 EVR={p['evr2']*100:.1f}%")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (2) feature-importance 저장: {out_path}")


def plot_biplot(p, names, top_idx, out_path):
    """PC1·PC2 평면의 원본 feature 방향(상관 로딩). top feature 는 강조."""
    from matplotlib.patches import Circle
    L = p["load"] / (np.abs(p["load"]).max() + 1e-9)
    top_set = set(top_idx)
    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#DDD", ls="--", lw=1.0))
    ax.axhline(0, color="#EEE", lw=0.8); ax.axvline(0, color="#EEE", lw=0.8)
    for j in range(N_FEATURES):
        dx, dy = L[j, 0], L[j, 1]
        is_top = j in top_set
        col = "#D62728" if is_top else "#C8C8C8"
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col,
                                    lw=2.0 if is_top else 1.0,
                                    alpha=0.95 if is_top else 0.5))
        if is_top:
            r = np.hypot(dx, dy) + 1e-9
            ax.text(dx + 0.09 * dx / r, dy + 0.09 * dy / r, names[j], color=col,
                    fontsize=9, fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8))
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Loading biplot - feature directions (top-2 EVR={p['evr2']*100:.1f}%)\n"
                 f"Red = top-{len(top_idx)} influential features")
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (2') biplot 저장: {out_path}")


# =============================================================================
# 시각화 ③ PCA 영향 큰 top-N feature 교차 scatter (pairplot)
# =============================================================================
def plot_top_feature_pairplot(ref_raw, sample_raw, names, top_idx, out_path, seed=0):
    """top-N feature 원본값의 교차 산점(pairplot). REF/SAMPLE 을 hue 로 겹쳐 본다."""
    rng = np.random.default_rng(seed)
    top_names = [names[i] for i in top_idx]

    def _sub(arr, tag):
        m = min(PAIR_MAX_PTS, len(arr))
        sel = rng.choice(len(arr), m, replace=False) if len(arr) > m else np.arange(len(arr))
        df = pd.DataFrame(arr[sel][:, top_idx], columns=top_names)
        df["source"] = tag
        return df

    df = pd.concat([_sub(ref_raw, "Reference"), _sub(sample_raw, "Sample")],
                   ignore_index=True)
    g = sns.pairplot(df, hue="source", vars=top_names,
                     palette={"Reference": C_REF, "Sample": C_SAMPLE},
                     plot_kws=dict(s=10, alpha=0.35, linewidth=0),
                     diag_kind="kde", diag_kws=dict(common_norm=False, fill=True, alpha=0.4),
                     corner=True)
    g.figure.suptitle(f"Top-{len(top_idx)} PCA-influential features: cross scatter",
                      y=1.02, fontsize=14)
    g.figure.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    print(f"[Plot] (3) top-feature pairplot 저장: {out_path}  (features={top_names})")


# =============================================================================
# 결과 저장 ④ REF: uncovered + outside_99 행만 group 태그로 저장 (스트리밍)
# =============================================================================
def _ref_group_masks(sc, xs, ys, ref_mask, sam_mask):
    """PC score → (keep, group) : outside_99 / uncovered 만 keep. 격자 lookup(KDE 재호출 없음)."""
    in_ref = classify_by_grid(sc, xs, ys, ref_mask)
    in_sam = classify_by_grid(sc, xs, ys, sam_mask)
    outside = ~in_ref                              # 99% 밖
    uncovered = in_ref & (~in_sam)                 # 99% 안 이나 sample 이 대변 못함
    keep = outside | uncovered
    group = np.where(outside, "outside_99", "uncovered")
    return keep, group


def export_ref_groups(ref_path, names, p, xs, ys, ref_mask, sam_mask, fmt, out_path, ref_X=None):
    """
    REF 에서 아래 두 group 에 해당하는 행 + PC1·PC2 + group 을 CSV 로 저장한다.
      - outside_99 : REF 99% 등고선 밖
      - uncovered  : REF 99% 안 이지만 SAMPLE 99% 밖(대변 안 됨)
    (covered = REF 99% 안 & SAMPLE 99% 안 → 저장 대상 아님)

    분류는 KDE 재호출 없이 '격자 마스크 lookup'(O(N))으로 수행한다.
    REF_EXPORT_ORIGINAL_COLS:
      True  → 원본 파일을 1회 스트리밍(모든 컬럼 유지). 원본 메타 컬럼(hash 등) 보존.
      False → 캐시된 feature 행렬(ref_X)만 사용(파일 재읽기 0). feature 컬럼만 저장.
    """
    header_written = [False]
    n_out99 = n_unc = n_total = 0
    if os.path.exists(out_path):
        os.remove(out_path)

    if REF_EXPORT_ORIGINAL_COLS:
        def _flush(df_full):
            nonlocal n_out99, n_unc, n_total
            df_full.columns = [str(c).strip() for c in df_full.columns]
            Xf, finite = _select(df_full, names)
            n_total += int(finite.sum())
            if finite.sum() == 0:
                return
            rows = df_full.loc[finite].reset_index(drop=True)
            sc = pca_project(Xf[finite], p)
            keep, group = _ref_group_masks(sc, xs, ys, ref_mask, sam_mask)
            if not keep.any():
                return
            out = rows.loc[keep].copy()
            out["PC1"] = sc[keep, 0]; out["PC2"] = sc[keep, 1]
            out["group"] = group[keep]
            out.to_csv(out_path, index=False, mode="a", header=not header_written[0])
            header_written[0] = True
            n_out99 += int((group[keep] == "outside_99").sum())
            n_unc += int((group[keep] == "uncovered").sum())

        if fmt == "csv":
            for chunk in pd.read_csv(ref_path, chunksize=CHUNKSIZE):
                _flush(chunk)
        else:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(ref_path)
            for batch in pf.iter_batches(batch_size=CHUNKSIZE):
                _flush(batch.to_pandas())
        if not header_written[0]:
            pd.DataFrame(columns=list(_header(ref_path, fmt)) + ["PC1", "PC2", "group"]) \
                .to_csv(out_path, index=False)
    else:
        # 캐시 행렬만 사용(파일 재읽기 없음). feature 컬럼 + PC + group.
        n_total = len(ref_X)
        for i in range(0, len(ref_X), CHUNKSIZE):
            Xf = np.asarray(ref_X[i:i + CHUNKSIZE], dtype=np.float64)
            sc = pca_project(Xf, p)
            keep, group = _ref_group_masks(sc, xs, ys, ref_mask, sam_mask)
            if not keep.any():
                continue
            out = pd.DataFrame(Xf[keep], columns=names)
            out["PC1"] = sc[keep, 0]; out["PC2"] = sc[keep, 1]
            out["group"] = group[keep]
            out.to_csv(out_path, index=False, mode="a", header=not header_written[0])
            header_written[0] = True
            n_out99 += int((group[keep] == "outside_99").sum())
            n_unc += int((group[keep] == "uncovered").sum())
        if not header_written[0]:
            pd.DataFrame(columns=names + ["PC1", "PC2", "group"]).to_csv(out_path, index=False)

    print(f"[Save] REF groups 저장: {out_path}  "
          f"(outside_99={n_out99:,}, uncovered={n_unc:,}, 전체 유효 REF={n_total:,})")
    return {"outside_99": n_out99, "uncovered": n_unc, "ref_total": n_total}


# =============================================================================
# 결과 저장 ⑤ SAMPLE: cover_ref / not_cover_ref group 태그 열 추가 저장
# =============================================================================
def export_sample_groups(sample_path, names, p, xs, ys, ref_mask, fmt, out_path):
    """
    Sample 원본(모든 컬럼)에 PC1·PC2 와 group 열을 붙여 저장한다.
      - cover_ref     : REF 99% 등고선 안(=REF 를 대변/cover)
      - not_cover_ref : REF 99% 등고선 밖
    분류는 격자 마스크 lookup(KDE 재호출 없음). NaN/Inf 행은 PC/group 이 비어있음.
    """
    if fmt == "csv":
        df = pd.read_csv(sample_path)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table(sample_path).to_pandas()
    df.columns = [str(c).strip() for c in df.columns]
    Xf, finite = _select(df, names)

    pc1 = np.full(len(df), np.nan); pc2 = np.full(len(df), np.nan)
    group = np.array([""] * len(df), dtype=object)

    sc = pca_project(Xf[finite], p)
    in_ref = classify_by_grid(sc, xs, ys, ref_mask)
    g = np.where(in_ref, "cover_ref", "not_cover_ref")
    pc1[finite] = sc[:, 0]; pc2[finite] = sc[:, 1]
    group[finite] = g

    df["PC1"], df["PC2"], df["group"] = pc1, pc2, group
    df.to_csv(out_path, index=False)
    n_cov = int((g == "cover_ref").sum()); n_not = int((g == "not_cover_ref").sum())
    print(f"[Save] SAMPLE groups 저장: {out_path}  "
          f"(cover_ref={n_cov:,}, not_cover_ref={n_not:,}, rows={len(df):,})")
    return {"cover_ref": n_cov, "not_cover_ref": n_not, "sample_total": int(finite.sum())}


# =============================================================================
# 메인
# =============================================================================
def run():
    global N_FEATURES
    for pth in (SAMPLE_PATH, REF_PATH):
        if not os.path.exists(pth):
            raise FileNotFoundError(f"입력 파일 없음: {pth} (CONFIG 경로 확인)")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    # feature 이름 확정(양쪽 매칭)
    names = resolve_feature_names(SAMPLE_PATH, REF_PATH, FMT)
    N_FEATURES = len(names)

    # (0) REF feature 행렬을 '1회'만 파싱해 캐시(이후 모든 단계·재실행에서 재사용)
    ref_X = load_ref_matrix(REF_PATH, names, FMT)

    # (1) 캐시 행렬에서 μ·Σ → 표준화 상관행렬 PCA 학습
    n_ref, sum_x, sum_xx = moments_from_matrix(ref_X)
    if n_ref < N_FEATURES + 1:
        raise ValueError(f"Reference 유효행 {n_ref} < 차원 {N_FEATURES}+1: 공분산 추정 불가.")
    p = build_pca(n_ref, sum_x, sum_xx)

    # PCA 영향 큰 top-N feature
    top_idx = list(np.argsort(p["feat_imp"])[::-1][:N_TOP_FEATURES])
    print(f"[PCA] 영향 큰 top-{N_TOP_FEATURES} feature: {[names[i] for i in top_idx]}")

    # (2) 시각화용 표본: Sample 전체(메모리) + REF 캐시에서 균일 표본
    sample_raw = read_matrix_by_name(SAMPLE_PATH, names, FMT)
    ref_raw, _ = subsample_rows(ref_X, PLOT_DOWNSAMPLE)
    print(f"[Sample] REF 시각화 표본 {len(ref_raw):,} rows (of {n_ref:,})")

    scores_s = pca_project(sample_raw, p)
    scores_r = pca_project(ref_raw, p)

    # (3) KDE 적합(표본 상한 KDE_MAX_PTS) + 99% HDR 임계치
    kde_r = fit_kde(scores_r, KDE_MAX_PTS)
    kde_s = fit_kde(scores_s, KDE_MAX_PTS)
    lvl_r = hdr_level(kde_r, scores_r, HDR_Q, N_THREADS)
    lvl_s = hdr_level(kde_s, scores_s, HDR_Q, N_THREADS)

    # (4) 격자 KDE 평가(스레드 병렬) → Coverage(겹침 면적) 산출
    xs, ys, XX, YY, cell_area = make_grid(scores_r, scores_s, GRID_N, GRID_MARGIN)
    dens_r = eval_grid(kde_r, XX, YY, N_THREADS)
    dens_s = eval_grid(kde_s, XX, YY, N_THREADS)
    ref_mask = dens_r >= lvl_r
    sam_mask = dens_s >= lvl_s
    overlap_mask = ref_mask & sam_mask
    area_ref = ref_mask.sum() * cell_area
    area_sam = sam_mask.sum() * cell_area
    area_ovl = overlap_mask.sum() * cell_area
    area_uni = (ref_mask | sam_mask).sum() * cell_area
    cov_sample_of_ref = area_ovl / area_ref if area_ref else 0.0   # 주지표
    cov_ref_of_sample = area_ovl / area_sam if area_sam else 0.0
    jaccard = area_ovl / area_uni if area_uni else 0.0
    cov_txt = (f"Coverage(SAMPLE→REF)={cov_sample_of_ref*100:.2f}%  |  "
               f"Jaccard={jaccard*100:.2f}%")

    # 시각화
    plot_density_coverage(XX, YY, dens_r, dens_s, lvl_r, lvl_s,
                          scores_r, scores_s, cov_txt, j("pca_density_coverage.png"))
    plot_scatter_ref_sample(scores_r, scores_s, p, j("pca_scatter_ref_sample.png"))
    plot_feature_importance(p, names, top_idx, j("pca_feature_importance.png"))
    plot_biplot(p, names, top_idx, j("pca_loadings_biplot.png"))
    plot_top_feature_pairplot(ref_raw, sample_raw, names, top_idx, j("pca_top8_pairplot.png"))

    # (5) CSV 저장 — 분류는 격자 lookup(KDE 재호출 없음)
    ref_stats = export_ref_groups(REF_PATH, names, p, xs, ys, ref_mask, sam_mask,
                                  FMT, j("reference_uncovered_outside.csv"), ref_X=ref_X)
    sam_stats = export_sample_groups(SAMPLE_PATH, names, p, xs, ys, ref_mask,
                                     FMT, j("sample_cover_group.csv"))

    # 리포트
    print("\n" + "=" * 70)
    print("  [REF-PCA 2D | KDE 99% Coverage]")
    print("=" * 70)
    print(f"  REF 유효행 수                       : {n_ref:,}")
    print(f"  PCA top-2 설명분산(EVR)             : {p['evr2']*100:6.2f}%  "
          f"(PC1={p['evr'][0]*100:.1f}%, PC2={p['evr'][1]*100:.1f}%)")
    print(f"  영향 큰 top-{N_TOP_FEATURES} feature : {[names[i] for i in top_idx]}")
    print("  ---- 99% 등고선(HDR) 면적 ----")
    print(f"  area(REF_99)                        : {area_ref:.4f}")
    print(f"  area(SAMPLE_99)                     : {area_sam:.4f}")
    print(f"  area(overlap)                       : {area_ovl:.4f}")
    print(f"  >>> Coverage(SAMPLE→REF)            : {cov_sample_of_ref*100:7.3f}%   "
          f"(overlap / REF_99)")
    print(f"      Coverage(REF→SAMPLE)            : {cov_ref_of_sample*100:7.3f}%   "
          f"(overlap / SAMPLE_99)")
    print(f"      Jaccard(overlap/union)          : {jaccard*100:7.3f}%")
    print("  ---- 포인트 group 집계 ----")
    print(f"  [REF]    outside_99={ref_stats['outside_99']:,}  "
          f"uncovered={ref_stats['uncovered']:,}  (전체 {ref_stats['ref_total']:,})")
    print(f"  [SAMPLE] cover_ref={sam_stats['cover_ref']:,}  "
          f"not_cover_ref={sam_stats['not_cover_ref']:,}  (전체 {sam_stats['sample_total']:,})")
    print("=" * 70)
    return {"coverage_sample_of_ref": cov_sample_of_ref, "jaccard": jaccard,
            "evr2": p["evr2"], "top_features": [names[i] for i in top_idx],
            "ref_stats": ref_stats, "sam_stats": sam_stats}


if __name__ == "__main__":
    run()
