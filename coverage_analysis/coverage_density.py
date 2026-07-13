# -*- coding: utf-8 -*-
"""
coverage_density.py
===================
[알고리즘 B] 확률 밀도 기반 커버리지 (Density-based Coverage)

Reference 분포의 PDF 를 학습하고, 1,800 Sample 이 '고밀도 영역' 을 얼마나 커버하는지 본다.

파이프라인
    1) Reference 를 스트리밍하며 Reservoir Sampling 으로 균일하게 k개(10만~50만)만 추출.
    2) 추출 표본으로 KDE 또는 GMM 학습 → Reference PDF.
    3) Reference 표본의 log-likelihood 하위 분위수를 Threshold 로 결정(데이터 기반 컷오프).
    4) Sample 을 통과시켜 평균 log-likelihood 와 'Threshold 이상 샘플 비율' 을 산출.

가속
    * KDE 의 score_samples 는 무거우므로 '프로세스'로 분할 병렬 실행(parallel_score, 실측 ~3.7x).
      (스레드는 GIL 때문에 효과 없음 → ProcessPoolExecutor 사용)
    * Threshold 는 Reference 표본 전체가 아니라 서브샘플로 추정 → 불필요한 O(k^2) 비용 제거.

실행:  python coverage_density.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from common import (
    DummyDataGenerator, FEATURE_COLS, N_WORKERS,
    fit_scaler, iter_reference_chunks, parallel_score,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
# =============================================================================
SAMPLE_PATH   = "dummy_sample.csv"
REF_PATH      = "dummy_reference.csv"
FMT           = "csv"        # 'csv' 또는 'parquet'
MODEL_TYPE    = "kde"        # 'kde' 또는 'gmm'
SCALE_METHOD  = "standard"   # 밀도 모델은 standard 스케일이 일반적으로 유리
DOWNSAMPLE_K  = 200_000      # Reference 에서 추출할 표본 수 (10만~50만)
CHUNKSIZE     = 200_000
THRESHOLD_PCT = 10.0         # Reference log-likelihood 하위 몇 % 를 Threshold 로 쓸지
THR_SUBSAMPLE = 20_000       # Threshold 추정용 서브샘플 크기 (KDE O(k^2) 방지)

BANDWIDTH     = 0.3          # KDE 하이퍼파라미터
N_COMPONENTS  = 32           # GMM 성분 수

GEN_DUMMY     = True
REF_ROWS      = 2_000_000


# =============================================================================
# 1. Reservoir Sampling — 단일 스트리밍 패스로 균일 k개 추출 (메모리 O(k))
# =============================================================================
def reservoir_sample(ref_path, scaler, k, seed=0):
    """Reference 를 chunk 스트리밍하며 전체에서 균일하게 k개 행을 추출한다."""
    rng = np.random.default_rng(seed)
    reservoir = np.empty((k, len(FEATURE_COLS)), dtype=np.float32)
    filled = seen = 0
    for chunk in iter_reference_chunks(ref_path, scaler, CHUNKSIZE, FMT):
        # (A) reservoir 가 아직 안 찼으면 우선 채운다
        if filled < k:
            take = min(k - filled, len(chunk))
            reservoir[filled:filled + take] = chunk[:take]
            filled += take
            seen += take
            chunk = chunk[take:]
            if len(chunk) == 0:
                continue
        # (B) 가득 찬 상태: t번째 행을 확률 k/t 로 임의 슬롯과 교체 (Algorithm R)
        t = seen + np.arange(1, len(chunk) + 1)
        accept = rng.random(len(chunk)) < (k / t)
        if accept.any():
            reservoir[rng.integers(0, k, accept.sum())] = chunk[accept]
        seen += len(chunk)
    reservoir = reservoir[:filled]
    print(f"[Density] Reservoir Sampling: 전체 {seen:,} rows → {len(reservoir):,} rows 추출.")
    return reservoir


# =============================================================================
# 2. 밀도 모델 학습 (KDE / GMM) — 둘 다 score_samples = log-likelihood 로 통일
# =============================================================================
def fit_model(samples):
    if MODEL_TYPE == "kde":
        model = KernelDensity(kernel="gaussian", bandwidth=BANDWIDTH).fit(samples)
        print(f"[Density] KDE 학습 완료 (bandwidth={BANDWIDTH}, n={len(samples):,})")
    else:
        model = GaussianMixture(
            n_components=N_COMPONENTS, covariance_type="full",
            reg_covar=1e-4, random_state=0, max_iter=200,
        ).fit(samples)
        print(f"[Density] GMM 학습 완료 (n_components={N_COMPONENTS}, n={len(samples):,})")
    return model


# =============================================================================
# 3. 메인 파이프라인
# =============================================================================
def run():
    # (1) Sample 로드 → (2) Sample 기준 스케일러 학습
    sample_df = pd.read_csv(SAMPLE_PATH)
    scaler = fit_scaler(sample_df, SCALE_METHOD)
    sample_scaled = scaler.transform(sample_df[FEATURE_COLS].values)

    # (3) Reference Downsampling → (4) 밀도 모델 학습
    ref_samples = reservoir_sample(REF_PATH, scaler, DOWNSAMPLE_K)
    model = fit_model(ref_samples)

    # (5) Threshold: Reference 서브샘플의 log-likelihood 하위 THRESHOLD_PCT% 지점
    sub = ref_samples[:THR_SUBSAMPLE]
    threshold = np.percentile(parallel_score(model, sub, N_WORKERS), THRESHOLD_PCT)
    print(f"[Density] Threshold(하위 {THRESHOLD_PCT:.0f}%) = {threshold:.4f}")

    # (6) Sample 평가 (score_samples 병렬)
    sample_ll = parallel_score(model, sample_scaled, N_WORKERS)
    coverage_ratio = float(np.mean(sample_ll >= threshold))

    print("\n" + "=" * 56)
    print(f"  [알고리즘 B] 확률 밀도 커버리지 (model={MODEL_TYPE.upper()})")
    print("=" * 56)
    print(f"  Sample 평균 log-likelihood : {sample_ll.mean():.4f}")
    print(f"  Sample 중앙값              : {np.median(sample_ll):.4f}")
    print(f"  고밀도 영역 커버 샘플 수    : "
          f"{int((sample_ll >= threshold).sum()):,} / {len(sample_ll):,}")
    print(f"  >>> Coverage Ratio (≥Thr)  : {coverage_ratio:.4f} ({coverage_ratio*100:.2f}%)")
    print("=" * 56)
    return coverage_ratio


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
