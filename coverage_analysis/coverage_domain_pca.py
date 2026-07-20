# -*- coding: utf-8 -*-
"""
coverage_domain_pca.py
======================
[REF 기준 PCA-2D + KDE 99% 등고선 Coverage] 전용 분석 (self-contained)

목적
    Reference(REF) 21차원과 Sample(SAMPLE) 21차원이 들어올 때
        (1) REF 기준으로 PCA 를 학습해 상위 PC(기본 3D)로 투영한다.
        (2) REF·SAMPLE 각각의 확률밀도(KDE) 99% 등고선을 그리고 겹치는 영역을 Coverage 로 정의한다.
            - 3D 격자 부피비로 주 Coverage 산출(PC2 이후 저설명 방향까지 반영).
            - 3D 산점 + 각 면(PC1-2, PC1-3, PC2-3) 2D projection 등고선/겹침을 함께 그린다.
            - 추가로 누적 EVR 80/90% 차원에서 '질량 기반' Coverage 를 산출(고차원 확장).
        (3) PCA(상위 PC) 분산 기여가 임계값(기본 0.01) 이상인 feature 를 모두 추려 교차 scatter.
        (4) REF 에서 'cover 되지 못한 point' 와 '99% 밖에 분포한 point' 행만 추려
            group 태그를 붙여 CSV 로 저장한다(PC1..PCk 포함).
        (5) SAMPLE 에서 'REF 를 cover 하는 point' 와 'cover 못하는 point' 를 나눠
            group 태그 열을 붙여 CSV 로 저장한다.

    ※ Coverage 정의 2종:
        - 부피(volume) 기반: 99% HDR 영역의 겹침 면적/부피 비율(2D·3D, 등고선 그림과 일치).
          고차원에선 HDR 부피가 급감·비겹침 → 0 으로 붕괴(차원의 저주)하므로 3D 까지만 사용.
        - 질량(mass) 기반: REF 도메인(자기 99%) 포인트 중 SAMPLE 99% 에도 드는 비율.
          실제 포인트로 추정 → 고차원에서도 안정적. 80/90%EVR 확장은 이 방식.

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

여러 sample 비교 (REF 고정, sample1·sample2·… 교체)
    * PCA 축(고유벡터)·표준화 통계(μ_ref·σ_ref)는 '오직 REF' 로만 계산한다.
      → sample 을 바꿔도 REF 의 PC 값·벡터·EVR 은 절대 바뀌지 않는다(참고 mahal 스크립트는
        스케일러를 SAMPLE 로 fit 해서 sample 마다 축이 흔들렸다 — 이를 제거).
    * REF PCA basis 는 <CACHE_DIR>/pca_<key>.npz 로 '1회 fit → 재사용'(get_ref_pca) 하여
      sample 실행 간 μ·σ·고유벡터·PC 가 100% 동일함을 보장한다.
    * 고유벡터 부호를 결정적으로 고정(build_pca) → 부호 뒤집힘 없이 재현.
    * GRID_ANCHOR='ref' 면 플롯 축을 REF 범위로 고정 → sample1·sample2 그림을 그대로 겹쳐 비교.
    * outside_99(=REF 자기 99% 밖)는 sample 과 무관하게 동일; uncovered·Coverage 만 sample 별로 달라진다.

성능(속도) 설계  ── "전처리 1회 캐시 + 재사용", 격자 lookup, 스레드
    (1) [전처리 캐시] REF feature 행렬 + PCA basis 를 '한 번만' 만들어 캐시하고
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
GRID_N        = 240            # 2D 평면 등고선/적분 격자 해상도(축당)
GRID_MARGIN   = 0.08           # 격자 여백(데이터 범위 대비 비율)
GRID_ANCHOR   = "ref"          # 'ref'|'both' : 격자(플롯 축) 기준.
                               #   'ref'  = REF 범위로 축 고정 → sample 을 바꿔도 축이 동일해
                               #            sample1·sample2 플롯을 그대로 겹쳐 비교(권장).
                               #   'both' = REF+SAMPLE 합집합 범위(sample 마다 축이 달라짐).

# ---- Coverage 차원 (PC2 이후 저설명 방향까지 반영) ----
COVERAGE_DIMS = 3              # 주 Coverage/분류/시각화 차원 (PC1..PC3). 3D 격자(부피비)로 산출.
GRID_N_3D     = 64            # 3D 격자 해상도(축당) → GRID_N_3D^3 셀. 분류·부피적분용.
COVERAGE_EVR_TARGETS = [0.80, 0.90]   # 이 '누적 EVR' 차원에서 '질량(mass) 기반' Coverage 추가 산출.
                               #   ※ 고차원은 부피 기반이 0 으로 붕괴(차원의 저주)하므로 질량 기반 사용.

# ---- PCA 영향 feature 선정 ----
FEAT_IMP_THRESH = 0.01         # PC(상위 COVERAGE_DIMS) 분산 기여가 이 값 이상인 feature 를 모두 분석
PAIR_MAX_VARS   = 12           # pairplot 변수 상한(초과 시 기여 상위만; 과대 격자 방지)
PAIR_MAX_PTS    = 3000         # pairplot 소스별 최대 표본 수(과밀 방지)

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


def _derive_pca(mu, sd, V, evr):
    """
    고정된 (mu, sd, V, evr) 에서 config(COVERAGE_DIMS) 에 맞는 파생값을 계산한다.
    → 캐시된 basis 를 재사용해도 COVERAGE_DIMS 만 바꾸면 파생값이 즉시 반영된다.
    반환 추가키: K, Vk(=V[:, :K]), evr2, load(상관 로딩 d×2), feat_imp(상위 K PC 기여 d,).
    """
    d = len(mu)
    K = int(min(max(COVERAGE_DIMS, 2), d))             # 최소 2, 최대 d
    Vk = V[:, :K]                                       # PC1..PCK 방향
    load = V[:, :2] * np.sqrt(np.clip(evr[:2] * d, 0, None))   # 상관 로딩(상관행렬 trace=d)
    # feature 중요도 = 상위 K PC 분산에서 각 feature 가 차지하는 몫(설명분산 가중)
    feat_imp = np.zeros(d)
    for m in range(K):
        feat_imp += (V[:, m] ** 2) * evr[m]
    return {"mu": mu, "sd": sd, "V": V, "evr": evr, "K": K, "Vk": Vk,
            "evr2": float(evr[:2].sum()), "load": load, "feat_imp": feat_imp}


def build_pca(n, sum_x, sum_xx):
    """REF μ·Σ 로 '표준화 상관행렬 PCA' 를 학습한다(고유벡터 부호 고정 → 재현성)."""
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
    # 고유벡터 부호를 결정적으로 고정(재현성): 각 PC 에서 |성분| 최대 좌표를 +로.
    for k in range(V.shape[1]):
        jmax = int(np.argmax(np.abs(V[:, k])))
        if V[jmax, k] < 0:
            V[:, k] = -V[:, k]
    evr = w / w.sum()

    p = _derive_pca(mu, sd, V, evr)
    cum = np.cumsum(evr)
    print(f"[PCA] REF 표준화 상관행렬 PCA 학습(dim={d}); "
          f"PC1={evr[0]*100:.1f}% PC2={evr[1]*100:.1f}% "
          f"PC3={evr[2]*100:.1f}% (top-{p['K']} 누적={cum[p['K']-1]*100:.1f}%)")
    return p


def pca_project(X_raw, p):
    """raw feature (n, d) → PC1..PCK score (n, K).  z=(x−μ)/σ , score=z·Vk."""
    z = (X_raw - p["mu"]) / p["sd"]
    return z @ p["Vk"]


def dims_for_evr(evr, target):
    """누적 EVR 이 target 에 처음 도달하는 차원 수(최소 2, 최대 d)."""
    cum = np.cumsum(evr)
    k = int(np.searchsorted(cum, target) + 1)
    return int(min(max(k, 2), len(evr)))


def save_pca_model(path, p, n_ref):
    """REF PCA basis(mu, sd, V, evr)를 npz 로 저장 → 여러 sample 실행이 동일 프레임 재사용."""
    np.savez(path, mu=p["mu"], sd=p["sd"], V=p["V"], evr=p["evr"], n_ref=np.int64(n_ref))


def load_pca_model(path):
    """저장된 REF PCA basis 를 로드 → 현재 config 로 파생값 재계산 후 dict + n_ref 반환."""
    z = np.load(path)
    p = _derive_pca(z["mu"], z["sd"], z["V"], z["evr"])
    return p, int(z["n_ref"])


def get_ref_pca(ref_X, names):
    """
    REF PCA basis 를 얻는다.  USE_CACHE 면 <CACHE_DIR>/pca_<key>.npz 로 '1회 fit → 재사용'.
    → sample 을 바꿔가며 여러 번 돌려도 REF 의 μ·σ·고유벡터·PC 가 100% 동일하게 고정된다.
    """
    pca_path = None
    if USE_CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)
        pca_path = os.path.join(CACHE_DIR, f"pca_{_cache_key(REF_PATH, names)}.npz")
        if os.path.exists(pca_path):
            p, n_ref = load_pca_model(pca_path)
            cum = float(np.cumsum(p["evr"])[p["K"] - 1])
            print(f"[Cache] REF PCA basis 재사용: {pca_path}  "
                  f"(PC1={p['evr'][0]*100:.1f}% PC2={p['evr'][1]*100:.1f}% "
                  f"PC3={p['evr'][2]*100:.1f}%, top-{p['K']} 누적={cum*100:.1f}%)")
            return p, n_ref
    n_ref, sum_x, sum_xx = moments_from_matrix(ref_X)
    if n_ref < ref_X.shape[1] + 1:
        raise ValueError(f"Reference 유효행 {n_ref} < 차원 {ref_X.shape[1]}+1: 공분산 추정 불가.")
    p = build_pca(n_ref, sum_x, sum_xx)
    if pca_path:
        save_pca_model(pca_path, p, n_ref)
        print(f"[Cache] REF PCA basis 저장: {pca_path}")
    return p, n_ref


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


def make_grid(scores_ref, scores_sample, n, margin, anchor="ref"):
    """
    격자(가장자리 xs, ys + 셀중심 XX,YY + 셀면적) 생성.
    anchor='ref'  : REF 범위로 축 고정(여러 sample 비교 시 축이 동일 → 그림 겹쳐 보기 좋음).
    anchor='both' : REF+SAMPLE 합집합 범위.
    """
    base = scores_ref if anchor == "ref" else np.vstack([scores_ref, scores_sample])
    lo = base.min(axis=0); hi = base.max(axis=0)
    pad = (hi - lo) * margin + 1e-9
    lo -= pad; hi += pad
    xs = np.linspace(lo[0], hi[0], n)
    ys = np.linspace(lo[1], hi[1], n)
    XX, YY = np.meshgrid(xs, ys)                       # (n, n) — XX[i,j]=xs[j], YY[i,j]=ys[i]
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    # anchor='ref' 일 때 sample 이 REF 프레임 밖으로 얼마나 벗어나는지 보고(등고선 clip 경고)
    if anchor == "ref" and len(scores_sample):
        out = ((scores_sample[:, 0] < xs[0]) | (scores_sample[:, 0] > xs[-1]) |
               (scores_sample[:, 1] < ys[0]) | (scores_sample[:, 1] > ys[-1]))
        frac = float(out.mean())
        if frac > 0.005:
            print(f"[Grid] 주의: SAMPLE 의 {frac*100:.1f}% 가 REF 축 프레임 밖(등고선 일부 clip). "
                  f"필요시 GRID_ANCHOR='both' 로 전환.")
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
# [k-D Coverage] 3D 는 격자(부피 적분·분류), 80/90%EVR 등 고차원은 Monte-Carlo
# =============================================================================
def grid_axes_kd(base, dims, n, margin, anchor_pad=None):
    """dims 개 축의 등간격 셀 중심(centers) + 셀경계(edges) + 셀부피. base 범위 + 여백."""
    lo = base[:, :dims].min(axis=0); hi = base[:, :dims].max(axis=0)
    pad = (hi - lo) * margin + 1e-9
    lo -= pad; hi += pad
    edges = [np.linspace(lo[a], hi[a], n + 1) for a in range(dims)]
    centers = [(e[:-1] + e[1:]) / 2 for e in edges]
    cell_vol = float(np.prod([e[1] - e[0] for e in edges]))
    return edges, centers, cell_vol


def kde_mask_kd(kde, centers, lvl, n_threads=1):
    """dims-차원 격자 중심에서 KDE 평가 → (density ≥ lvl) 마스크. 형상 (n,)*dims."""
    mesh = np.meshgrid(*centers, indexing="ij")
    pts = np.vstack([m.ravel() for m in mesh])
    dens = kde_eval(kde, pts, n_threads).reshape(mesh[0].shape)
    return dens >= lvl


def classify_by_grid_kd(scores, edges, mask):
    """dims-차원 셀 마스크 조회로 in/out 판정(O(N)). 격자 밖은 False."""
    dims = len(edges)
    inb = np.ones(len(scores), dtype=bool)
    idx = []
    for a in range(dims):
        e = edges[a]
        idx.append(np.clip(np.digitize(scores[:, a], e) - 1, 0, len(e) - 2))
        inb &= (scores[:, a] >= e[0]) & (scores[:, a] <= e[-1])
    res = np.zeros(len(scores), dtype=bool)
    res[inb] = mask[tuple(i[inb] for i in idx)]
    return res


def coverage_grid_kd(scores_r, scores_s, dims, q, n, margin, anchor, threads):
    """
    dims-차원 격자로 REF_99 / SAM_99 마스크·부피비 Coverage 산출.
    반환: cov_s_of_r, cov_r_of_s, jaccard, edges, mask_r, mask_s, lvl_r, lvl_s, kde_r, kde_s.
    (분류·시각화에 재사용하려고 마스크·KDE 도 함께 반환)
    """
    r, s = scores_r[:, :dims], scores_s[:, :dims]
    kde_r = fit_kde(r, KDE_MAX_PTS); kde_s = fit_kde(s, KDE_MAX_PTS)
    lvl_r = hdr_level(kde_r, r, q, threads); lvl_s = hdr_level(kde_s, s, q, threads)
    base = r if anchor == "ref" else np.vstack([r, s])
    edges, centers, _ = grid_axes_kd(base, dims, n, margin)
    mask_r = kde_mask_kd(kde_r, centers, lvl_r, threads)
    mask_s = kde_mask_kd(kde_s, centers, lvl_s, threads)
    ovl = int((mask_r & mask_s).sum()); ar = int(mask_r.sum()); as_ = int(mask_s.sum())
    uni = int((mask_r | mask_s).sum())
    return {"cov_s_of_r": ovl / ar if ar else 0.0,
            "cov_r_of_s": ovl / as_ if as_ else 0.0,
            "jaccard": ovl / uni if uni else 0.0,
            "edges": edges, "mask_r": mask_r, "mask_s": mask_s,
            "lvl_r": lvl_r, "lvl_s": lvl_s, "kde_r": kde_r, "kde_s": kde_s, "dims": dims}


def coverage_mass(scores_r, scores_s, dims, q, threads):
    """
    고차원 Coverage 를 '질량(mass) 기반'으로 산출(부피 기반은 고차원에서 0 으로 붕괴 → 무의미).
    정의: REF 의 도메인(자기 99% HDR) 안 포인트 중, SAMPLE 99% HDR 에도 드는 비율.
        Coverage_mass(SAMPLE→REF) = |{REF∈REF_99 ∧ REF∈SAM_99}| / |{REF∈REF_99}|
    REF·SAMPLE 실제 포인트에서 직접 추정하므로 차원이 커도 안정적이고 빠르다.
    반환: cov_s_of_r, cov_r_of_s, n_dom(신뢰도=REF 도메인 포인트 수), dims.
    """
    r, s = scores_r[:, :dims], scores_s[:, :dims]
    kde_r = fit_kde(r, KDE_MAX_PTS); kde_s = fit_kde(s, KDE_MAX_PTS)
    lvl_r = hdr_level(kde_r, r, q, threads); lvl_s = hdr_level(kde_s, s, q, threads)
    # REF 포인트의 소속 → Coverage(SAMPLE→REF)
    in_ref_r = kde_eval(kde_r, r.T, threads) >= lvl_r
    in_sam_r = kde_eval(kde_s, r.T, threads) >= lvl_s
    n_dom = int(in_ref_r.sum())
    cov_s_of_r = int((in_ref_r & in_sam_r).sum()) / max(n_dom, 1)
    # SAMPLE 포인트의 소속 → Coverage(REF→SAMPLE)
    in_sam_s = kde_eval(kde_s, s.T, threads) >= lvl_s
    in_ref_s = kde_eval(kde_r, s.T, threads) >= lvl_r
    n_sam = int(in_sam_s.sum())
    cov_r_of_s = int((in_sam_s & in_ref_s).sum()) / max(n_sam, 1)
    return {"cov_s_of_r": cov_s_of_r, "cov_r_of_s": cov_r_of_s, "n_dom": n_dom, "dims": dims}


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
def plot_feature_importance(p, names, sel_idx, out_path):
    fig, ax = plt.subplots(figsize=(9.5, max(5, 0.42 * N_FEATURES)))
    order = np.argsort(p["feat_imp"])[::-1]
    y = np.arange(len(order))[::-1]
    sel = set(sel_idx)
    colors = ["#D62728" if i in sel else "#BBBBBB" for i in order]
    ax.barh(y, p["feat_imp"][order], color=colors)
    ax.axvline(FEAT_IMP_THRESH, color="#333", ls="--", lw=1.4,
               label=f"threshold = {FEAT_IMP_THRESH}")
    ax.set_yticks(y); ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel(f"top-{p['K']} PC variance share (importance)")
    ax.set_title(f"PCA feature importance (>= {FEAT_IMP_THRESH}: {len(sel_idx)} features = red)\n"
                 f"top-{p['K']} 누적 EVR={np.cumsum(p['evr'])[p['K']-1]*100:.1f}%")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"[Plot] (2) feature-importance 저장: {out_path}")


# =============================================================================
# 시각화 ③' 3D 산점(PC1-PC2-PC3) + 각 면 2D projection(등고선·겹침)
# =============================================================================
def _draw_plane(ax, sr, ss, xlab, ylab, threads):
    """한 PC 평면(2D)에 REF/SAMPLE 밀도·99% 등고선·겹침을 그리고 평면 Coverage 반환."""
    kr = fit_kde(sr, KDE_MAX_PTS); ks = fit_kde(ss, KDE_MAX_PTS)
    lr = hdr_level(kr, sr, HDR_Q, threads); ls = hdr_level(ks, ss, HDR_Q, threads)
    xs, ys, XX, YY, _ = make_grid(sr, ss, 150, GRID_MARGIN, GRID_ANCHOR)
    dr = eval_grid(kr, XX, YY, threads); ds = eval_grid(ks, XX, YY, threads)
    ax.contourf(XX, YY, dr, levels=10, cmap="Greys", alpha=0.28)
    ov = ((dr >= lr) & (ds >= ls)).astype(float)
    ax.contourf(XX, YY, ov, levels=[0.5, 1.5], colors=[C_OVERLAP], alpha=0.45)
    ax.contour(XX, YY, dr, levels=[lr], colors=[C_REF_L], linewidths=2.0)
    ax.contour(XX, YY, ds, levels=[ls], colors=[C_SAMPLE], linewidths=2.0)
    ax.scatter(sr[:, 0], sr[:, 1], s=3, c=C_REF, alpha=0.08, linewidths=0)
    ax.scatter(ss[:, 0], ss[:, 1], s=5, c=C_SAMPLE, alpha=0.10, linewidths=0)
    mr = dr >= lr; ms = ds >= ls
    cov = int((mr & ms).sum()) / max(int(mr.sum()), 1)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_aspect("equal")
    ax.set_title(f"{xlab}-{ylab}  cov(S→R)={cov*100:.1f}%", fontsize=11)
    return cov


def plot_pca_3d_projections(scores_r, scores_s, p, cov3d_txt, out_path, threads=1, seed=0):
    """좌상=3D 산점(PC1-PC2-PC3), 나머지 3칸=각 면(PC1-2, PC1-3, PC2-3) 2D projection."""
    rng = np.random.default_rng(seed)
    def _ds(a, k):
        return a if len(a) <= k else a[rng.choice(len(a), k, replace=False)]
    r3, s3 = _ds(scores_r, 4000), _ds(scores_s, 4000)

    fig = plt.figure(figsize=(15, 13))
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    ax0.scatter(r3[:, 0], r3[:, 1], r3[:, 2], s=4, c=C_REF, alpha=0.15, linewidths=0, label="Reference")
    ax0.scatter(s3[:, 0], s3[:, 1], s3[:, 2], s=8, c=C_SAMPLE, alpha=0.25, linewidths=0, label="Sample")
    ax0.set_xlabel("PC1"); ax0.set_ylabel("PC2"); ax0.set_zlabel("PC3")
    ax0.set_title("3D scatter (PC1-PC2-PC3)", fontsize=12)
    ax0.legend(fontsize=9, loc="upper left")

    pairs = [(0, 1, "PC1", "PC2"), (0, 2, "PC1", "PC3"), (1, 2, "PC2", "PC3")]
    for k, (a, b, xl, yl) in enumerate(pairs, start=2):
        ax = fig.add_subplot(2, 2, k)
        _draw_plane(ax, scores_r[:, [a, b]], scores_s[:, [a, b]], xl, yl, threads)
    fig.suptitle("REF-PCA 3D coverage & plane projections\n" + cov3d_txt, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[Plot] (3D) 3d-projections 저장: {out_path}")


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
def _pc_cols(dims):
    return [f"PC{a+1}" for a in range(dims)]


def _ref_group_masks(sc, edges, mask_r, mask_s):
    """PC score(k-D) → (keep, group) : outside_99 / uncovered 만 keep. k-D 격자 lookup."""
    in_ref = classify_by_grid_kd(sc, edges, mask_r)
    in_sam = classify_by_grid_kd(sc, edges, mask_s)
    outside = ~in_ref                              # 99% 밖
    uncovered = in_ref & (~in_sam)                 # 99% 안 이나 sample 이 대변 못함
    keep = outside | uncovered
    group = np.where(outside, "outside_99", "uncovered")
    return keep, group


def export_ref_groups(ref_path, names, p, edges, mask_r, mask_s, fmt, out_path, ref_X=None):
    """
    REF 에서 아래 두 group 에 해당하는 행 + PC1..PCk + group 을 CSV 로 저장한다.
      - outside_99 : REF 99% 등고선(k-D HDR) 밖
      - uncovered  : REF 99% 안 이지만 SAMPLE 99% 밖(대변 안 됨)
    분류는 KDE 재호출 없이 'k-D 격자 마스크 lookup'(O(N))으로 수행(COVERAGE_DIMS 차원).
    REF_EXPORT_ORIGINAL_COLS: True=원본 전체 컬럼 유지(파일 1회 재읽기) / False=캐시만(재읽기 0).
    """
    dims = len(edges)
    pcs = _pc_cols(dims)
    header_written = [False]
    n_out99 = n_unc = n_total = 0
    if os.path.exists(out_path):
        os.remove(out_path)

    def _emit(base_df, sc, keep, group):
        nonlocal n_out99, n_unc
        out = base_df.loc[keep].copy()
        for a in range(dims):
            out[pcs[a]] = sc[keep, a]
        out["group"] = group[keep]
        out.to_csv(out_path, index=False, mode="a", header=not header_written[0])
        header_written[0] = True
        n_out99 += int((group[keep] == "outside_99").sum())
        n_unc += int((group[keep] == "uncovered").sum())

    if REF_EXPORT_ORIGINAL_COLS:
        def _flush(df_full):
            nonlocal n_total
            df_full.columns = [str(c).strip() for c in df_full.columns]
            Xf, finite = _select(df_full, names)
            n_total += int(finite.sum())
            if finite.sum() == 0:
                return
            rows = df_full.loc[finite].reset_index(drop=True)
            sc = pca_project(Xf[finite], p)
            keep, group = _ref_group_masks(sc, edges, mask_r, mask_s)
            if keep.any():
                _emit(rows, sc, keep, group)

        if fmt == "csv":
            for chunk in pd.read_csv(ref_path, chunksize=CHUNKSIZE):
                _flush(chunk)
        else:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(ref_path)
            for batch in pf.iter_batches(batch_size=CHUNKSIZE):
                _flush(batch.to_pandas())
        if not header_written[0]:
            pd.DataFrame(columns=list(_header(ref_path, fmt)) + pcs + ["group"]) \
                .to_csv(out_path, index=False)
    else:
        n_total = len(ref_X)
        for i in range(0, len(ref_X), CHUNKSIZE):
            Xf = np.asarray(ref_X[i:i + CHUNKSIZE], dtype=np.float64)
            sc = pca_project(Xf, p)
            keep, group = _ref_group_masks(sc, edges, mask_r, mask_s)
            if not keep.any():
                continue
            out = pd.DataFrame(Xf[keep], columns=names)
            for a in range(dims):
                out[pcs[a]] = sc[keep, a]
            out["group"] = group[keep]
            out.to_csv(out_path, index=False, mode="a", header=not header_written[0])
            header_written[0] = True
            n_out99 += int((group[keep] == "outside_99").sum())
            n_unc += int((group[keep] == "uncovered").sum())
        if not header_written[0]:
            pd.DataFrame(columns=names + pcs + ["group"]).to_csv(out_path, index=False)

    print(f"[Save] REF groups 저장: {out_path}  "
          f"(outside_99={n_out99:,}, uncovered={n_unc:,}, 전체 유효 REF={n_total:,})")
    return {"outside_99": n_out99, "uncovered": n_unc, "ref_total": n_total}


# =============================================================================
# 결과 저장 ⑤ SAMPLE: cover_ref / not_cover_ref group 태그 열 추가 저장
# =============================================================================
def export_sample_groups(sample_path, names, p, edges, mask_r, fmt, out_path):
    """
    Sample 원본(모든 컬럼)에 PC1..PCk 와 group 열을 붙여 저장한다.
      - cover_ref     : REF 99% 등고선(k-D) 안(=REF 를 대변/cover)
      - not_cover_ref : REF 99% 등고선 밖
    분류는 k-D 격자 마스크 lookup(KDE 재호출 없음). NaN/Inf 행은 PC/group 이 비어있음.
    """
    dims = len(edges)
    pcs = _pc_cols(dims)
    if fmt == "csv":
        df = pd.read_csv(sample_path)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table(sample_path).to_pandas()
    df.columns = [str(c).strip() for c in df.columns]
    Xf, finite = _select(df, names)

    pc = {c: np.full(len(df), np.nan) for c in pcs}
    group = np.array([""] * len(df), dtype=object)

    sc = pca_project(Xf[finite], p)
    in_ref = classify_by_grid_kd(sc, edges, mask_r)
    g = np.where(in_ref, "cover_ref", "not_cover_ref")
    for a in range(dims):
        pc[pcs[a]][finite] = sc[:, a]
    group[finite] = g

    for c in pcs:
        df[c] = pc[c]
    df["group"] = group
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

    # (1) REF PCA basis '1회 fit → 재사용'(sample 을 바꿔도 REF μ·σ·고유벡터·PC 100% 고정)
    p, n_ref = get_ref_pca(ref_X, names)
    K = p["K"]

    # PCA 영향 feature: 상위 K PC 분산 기여 >= FEAT_IMP_THRESH 인 feature 를 모두 선정
    order = np.argsort(p["feat_imp"])[::-1]
    sel_idx = [int(i) for i in order if p["feat_imp"][i] >= FEAT_IMP_THRESH]
    if not sel_idx:
        sel_idx = [int(i) for i in order[:3]]
    pair_idx = sel_idx[:PAIR_MAX_VARS]
    print(f"[PCA] 영향 feature(>= {FEAT_IMP_THRESH}) {len(sel_idx)}개: {[names[i] for i in sel_idx]}")
    if len(sel_idx) > PAIR_MAX_VARS:
        print(f"[PCA] pairplot 은 상위 {PAIR_MAX_VARS}개만 사용(과대 격자 방지).")

    # (2) 시각화용 표본: Sample 전체(메모리) + REF 캐시에서 균일 표본 → K-D PCA score
    sample_raw = read_matrix_by_name(SAMPLE_PATH, names, FMT)
    ref_raw, _ = subsample_rows(ref_X, PLOT_DOWNSAMPLE)
    print(f"[Sample] REF 시각화 표본 {len(ref_raw):,} rows (of {n_ref:,})")
    scores_s = pca_project(sample_raw, p)      # (n, K)
    scores_r = pca_project(ref_raw, p)

    # (3) [주] 3D Coverage(부피비) + 분류용 3D 격자 마스크
    c3 = coverage_grid_kd(scores_r, scores_s, COVERAGE_DIMS, HDR_Q,
                          GRID_N_3D, GRID_MARGIN, GRID_ANCHOR, N_THREADS)
    edges, mask_r3, mask_s3 = c3["edges"], c3["mask_r"], c3["mask_s"]
    cov3_txt = (f"{COVERAGE_DIMS}D Coverage(SAMPLE→REF)={c3['cov_s_of_r']*100:.2f}%  |  "
                f"Jaccard={c3['jaccard']*100:.2f}%")

    # (4) 추가: 3D(bridge) + 누적 EVR 80/90% 차원에서 '질량 기반' Coverage 산출(수치)
    #     (부피 기반은 고차원에서 0 으로 붕괴하므로 질량 기반으로 확장)
    def _scores_dd(dd):
        if dd <= scores_r.shape[1]:
            return scores_r[:, :dd], scores_s[:, :dd]
        Vd = p["V"][:, :dd]                                  # score 차원 부족 시 확장 투영
        return ((ref_raw - p["mu"]) / p["sd"]) @ Vd, ((sample_raw - p["mu"]) / p["sd"]) @ Vd

    mass_targets = [(f"{COVERAGE_DIMS}D", COVERAGE_DIMS)]
    for tgt in COVERAGE_EVR_TARGETS:
        mass_targets.append((f"EVR{int(tgt*100)}%", dims_for_evr(p["evr"], tgt)))
    mass_results = []
    for label, dd in mass_targets:
        sr, ss = _scores_dd(dd)
        mm = coverage_mass(sr, ss, dd, HDR_Q, N_THREADS)
        mm["label"] = label
        mass_results.append(mm)

    # (5) 2D(PC1-PC2) 메인 등고선 플롯 (해상도 높게)
    kde_r2 = fit_kde(scores_r[:, :2], KDE_MAX_PTS); kde_s2 = fit_kde(scores_s[:, :2], KDE_MAX_PTS)
    lvl_r2 = hdr_level(kde_r2, scores_r[:, :2], HDR_Q, N_THREADS)
    lvl_s2 = hdr_level(kde_s2, scores_s[:, :2], HDR_Q, N_THREADS)
    xs, ys, XX, YY, _ = make_grid(scores_r[:, :2], scores_s[:, :2], GRID_N, GRID_MARGIN, GRID_ANCHOR)
    dens_r2 = eval_grid(kde_r2, XX, YY, N_THREADS); dens_s2 = eval_grid(kde_s2, XX, YY, N_THREADS)
    cov2_s_of_r = int(((dens_r2 >= lvl_r2) & (dens_s2 >= lvl_s2)).sum()) / max(int((dens_r2 >= lvl_r2).sum()), 1)
    cov_txt2 = f"PC1-PC2 Coverage(SAMPLE→REF)={cov2_s_of_r*100:.2f}%"

    # 시각화
    plot_density_coverage(XX, YY, dens_r2, dens_s2, lvl_r2, lvl_s2,
                          scores_r[:, :2], scores_s[:, :2], cov_txt2, j("pca_density_coverage.png"))
    plot_pca_3d_projections(scores_r, scores_s, p, cov3_txt, j("pca_3d_projections.png"), N_THREADS)
    plot_scatter_ref_sample(scores_r, scores_s, p, j("pca_scatter_ref_sample.png"))
    plot_feature_importance(p, names, sel_idx, j("pca_feature_importance.png"))
    plot_biplot(p, names, sel_idx, j("pca_loadings_biplot.png"))
    plot_top_feature_pairplot(ref_raw, sample_raw, names, pair_idx, j("pca_feature_pairplot.png"))

    # (6) CSV 저장 — 분류는 3D 격자 lookup(KDE 재호출 없음)
    ref_stats = export_ref_groups(REF_PATH, names, p, edges, mask_r3, mask_s3,
                                  FMT, j("reference_uncovered_outside.csv"), ref_X=ref_X)
    sam_stats = export_sample_groups(SAMPLE_PATH, names, p, edges, mask_r3,
                                     FMT, j("sample_cover_group.csv"))

    # 리포트
    cum = np.cumsum(p["evr"])
    print("\n" + "=" * 70)
    print(f"  [REF-PCA {COVERAGE_DIMS}D | KDE 99% Coverage]")
    print("=" * 70)
    print(f"  REF 유효행 수                       : {n_ref:,}")
    print(f"  EVR: PC1={p['evr'][0]*100:.1f}% PC2={p['evr'][1]*100:.1f}% "
          f"PC3={p['evr'][2]*100:.1f}%  (top-{K} 누적={cum[K-1]*100:.1f}%)")
    print(f"  영향 feature(>= {FEAT_IMP_THRESH}) {len(sel_idx)}개 : {[names[i] for i in sel_idx]}")
    print("  ---- Coverage(SAMPLE→REF) [부피 기반, 등고선 그림과 일치] ----")
    print(f"  PC1-PC2 (2D)                        : {cov2_s_of_r*100:7.3f}%")
    print(f"  {COVERAGE_DIMS}D (격자 부피비)                  : {c3['cov_s_of_r']*100:7.3f}%   "
          f"(REF→SAMPLE={c3['cov_r_of_s']*100:.1f}%, Jaccard={c3['jaccard']*100:.1f}%)")
    print("  ---- Coverage(SAMPLE→REF) [질량 기반, 고차원 확장] ----")
    for mm in mass_results:
        flag = "  ⚠신뢰도낮음(n_dom<300)" if mm["n_dom"] < 300 else ""
        print(f"  {mm['label']:<8} ({mm['dims']:>2}D)                  "
              f": {mm['cov_s_of_r']*100:7.3f}%   (REF→SAMPLE={mm['cov_r_of_s']*100:.1f}%, "
              f"n_dom={mm['n_dom']:,}){flag}")
    print("  ---- 포인트 group 집계 (분류=" + f"{COVERAGE_DIMS}D) ----")
    print(f"  [REF]    outside_99={ref_stats['outside_99']:,}  "
          f"uncovered={ref_stats['uncovered']:,}  (전체 {ref_stats['ref_total']:,})")
    print(f"  [SAMPLE] cover_ref={sam_stats['cover_ref']:,}  "
          f"not_cover_ref={sam_stats['not_cover_ref']:,}  (전체 {sam_stats['sample_total']:,})")
    print("=" * 70)
    return {"cov_2d": cov2_s_of_r, "cov_3d": c3["cov_s_of_r"], "jaccard_3d": c3["jaccard"],
            "mass": mass_results, "evr": p["evr"][:K].tolist(),
            "sel_features": [names[i] for i in sel_idx],
            "ref_stats": ref_stats, "sam_stats": sam_stats}


if __name__ == "__main__":
    run()
