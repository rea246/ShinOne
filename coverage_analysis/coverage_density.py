# -*- coding: utf-8 -*-
"""
coverage_density.py
===================
[알고리즘 B] 확률 밀도 기반 커버리지 (Density-based Coverage)

목적
----
Reference 데이터의 분포를 대변하는 확률 밀도 함수(PDF)를 학습하고,
1,800개 Sample 이 '밀도가 높은 유의미한 영역'을 얼마나 잘 커버하는지 평가한다.

파이프라인
----------
  1) 50GB Reference 전체를 밀도 모델에 넣을 수 없으므로,
     스트리밍하며 '무작위 Downsampling'(Reservoir Sampling)으로 10만~50만 행만 추출.
  2) 추출 표본으로 KernelDensity(KDE) 또는 GaussianMixture(GMM) 를 학습 -> Reference PDF.
  3) 1,800 Sample 을 학습된 모델에 통과시켜 각 샘플의 log-likelihood(=score_samples)를 계산.
  4) 정량화 지표:
       - 평균 log-likelihood
       - Reference 표본의 log-likelihood 분포에서 얻은 Threshold 이상을 만족하는
         Sample 비율 (= '유의미한 고밀도 영역'을 커버한 비율)

왜 Reservoir Sampling 인가?
---------------------------
전체 행 수 N 을 미리 모르고(또는 매우 크고) 한 번의 스트리밍만으로
정확히 균일한(uniform) k개 표본을 뽑아야 하기 때문. 메모리는 O(k) 로 고정된다.

실행
----
    python coverage_density.py --model kde
    python coverage_density.py --model gmm
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from common import (
    DummyDataGenerator,
    FEATURE_COLS,
    fit_scaler,
    iter_reference_chunks,
)


# =============================================================================
# 1. Reservoir Sampling 으로 Reference Downsampling (스트리밍)
# =============================================================================
def reservoir_sample_reference(
    ref_path: str,
    scaler,
    k: int = 200_000,
    chunksize: int = 200_000,
    fmt: str = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Reference 를 chunk 단위로 스트리밍하며, 전체에서 균일하게 k개 행을 추출한다.

    표준 Reservoir Sampling(Algorithm R)의 벡터화(chunk) 버전.
    - 처음 k개는 그대로 채운다.
    - 이후 t번째 행(1-index)은 확률 k/t 로 reservoir 의 임의 슬롯을 교체한다.
    메모리 사용량은 reservoir 크기 (k, 15) 로 고정.

    Returns
    -------
    reservoir : shape (k, 15) 의 스케일링된 표본 (전체 행이 k 미만이면 그만큼만).
    """
    rng = np.random.default_rng(seed)
    reservoir = np.empty((k, len(FEATURE_COLS)), dtype=np.float32)
    filled = 0          # 현재 reservoir 에 채워진 개수
    seen = 0            # 지금까지 스트리밍으로 본 총 행 수 (t)

    for chunk in iter_reference_chunks(ref_path, scaler, chunksize, fmt):
        n = chunk.shape[0]

        # --- (A) reservoir 가 아직 다 안 찼으면 우선 채운다 ---
        if filled < k:
            take = min(k - filled, n)
            reservoir[filled:filled + take] = chunk[:take]
            filled += take
            seen += take
            # 이 chunk 에 남은 행이 있으면 아래 (B) 교체 로직으로 이어서 처리
            if take == n:
                continue
            chunk = chunk[take:]
            n = chunk.shape[0]

        # --- (B) reservoir 가 가득 찬 상태: 확률적 교체 ---
        # chunk 안 각 행에 대해 t = seen+1, seen+2, ... 로 증가.
        # 각 행이 reservoir 에 들어갈 확률 = k / t.
        t = seen + np.arange(1, n + 1)                 # 각 행의 전역 순번
        probs = k / t                                  # 채택 확률
        accept = rng.random(n) < probs                 # 채택 여부 마스크
        if accept.any():
            # 채택된 행들을 reservoir 의 무작위 슬롯에 덮어쓴다.
            slots = rng.integers(0, k, size=accept.sum())
            reservoir[slots] = chunk[accept]
        seen += n

    if filled < k:
        # 전체 행 수가 k 미만인 경우: 채운 만큼만 반환
        reservoir = reservoir[:filled]

    print(f"[Density] Reservoir Sampling 완료: 전체 {seen:,} rows 중 "
          f"{reservoir.shape[0]:,} rows 추출.")
    return reservoir


# =============================================================================
# 2. 밀도 모델 학습 (KDE 또는 GMM)
# =============================================================================
def fit_density_model(samples: np.ndarray, model_type: str = "kde", **kwargs):
    """
    Reference 표본으로 밀도 모델을 학습한다.

    model_type
    ----------
    'kde' : KernelDensity. bandwidth 가 핵심 하이퍼파라미터.
            15차원에서는 표본이 충분히 많아야(수십만) 안정적.
    'gmm' : GaussianMixture. n_components(혼합 성분 수)가 핵심.
            KDE 보다 학습/추론이 빠르고 고차원에서 더 견고한 경우가 많다.

    두 모델 모두 score_samples(X) 가 'log-likelihood' 를 반환하도록 통일되어 있다.
    """
    if model_type == "kde":
        bandwidth = kwargs.get("bandwidth", 0.3)
        model = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        model.fit(samples)
        print(f"[Density] KDE 학습 완료 (bandwidth={bandwidth}, "
              f"n_train={samples.shape[0]:,})")

    elif model_type == "gmm":
        n_components = kwargs.get("n_components", 32)
        model = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-4,          # 공분산 특이성 방지
            random_state=0,
            max_iter=200,
        )
        model.fit(samples)
        print(f"[Density] GMM 학습 완료 (n_components={n_components}, "
              f"n_train={samples.shape[0]:,})")
    else:
        raise ValueError("model_type 은 'kde' 또는 'gmm' 만 지원한다.")

    return model


# =============================================================================
# 3. Threshold 결정 (Reference 표본 log-likelihood 분위수 기반)
# =============================================================================
def choose_threshold(model, ref_samples: np.ndarray, percentile: float = 10.0) -> float:
    """
    '유의미한 고밀도 영역'의 경계값을 Reference 표본 스스로의 분포에서 정한다.

    Reference 표본의 log-likelihood 를 구한 뒤 하위 `percentile`% 지점을 Threshold 로 삼는다.
      - 예) percentile=10 이면, Reference 자신의 90% 가 이 Threshold 이상.
      - 즉 "Reference 가 실제로 밀집한 영역"을 정의하는 데이터 기반 컷오프.
    이 Threshold 이상을 만족하는 Sample 비율이 곧 '고밀도 영역 커버율'.
    """
    ref_ll = model.score_samples(ref_samples)
    thr = np.percentile(ref_ll, percentile)
    print(f"[Density] Threshold 결정: Reference log-likelihood 하위 "
          f"{percentile:.0f}% 지점 = {thr:.4f}")
    return thr


# =============================================================================
# 4. Sample 평가 및 정량화
# =============================================================================
def evaluate_samples(model, sample_scaled: np.ndarray, threshold: float) -> dict:
    """
    Sample(1800행)을 학습된 밀도 모델로 평가한다.

    반환 지표
    ---------
    * mean_log_likelihood   : Sample log-likelihood 평균 (클수록 Reference 분포에 잘 부합)
    * median_log_likelihood : 중앙값 (이상치에 robust)
    * coverage_ratio        : Threshold 이상을 만족하는 Sample 비율
                              (= Reference 고밀도 영역을 커버한 Sample 비율)
    """
    sample_ll = model.score_samples(sample_scaled)

    coverage_ratio = float(np.mean(sample_ll >= threshold))
    result = {
        "mean_log_likelihood": float(np.mean(sample_ll)),
        "median_log_likelihood": float(np.median(sample_ll)),
        "std_log_likelihood": float(np.std(sample_ll)),
        "coverage_ratio": coverage_ratio,
        "n_samples": int(sample_ll.shape[0]),
        "n_in_region": int(np.sum(sample_ll >= threshold)),
    }
    return result


# =============================================================================
# 5. 메인 파이프라인
# =============================================================================
def run(
    sample_path: str,
    ref_path: str,
    model_type: str = "kde",
    scale_method: str = "standard",
    downsample_k: int = 200_000,
    chunksize: int = 200_000,
    threshold_pct: float = 10.0,
    fmt: str = None,
    **model_kwargs,
):
    # (1) Sample 로드
    sample_df = pd.read_csv(sample_path)

    # (2) Sample 기준 스케일러 학습 (밀도 모델은 StandardScaling 이 일반적으로 유리)
    scaler = fit_scaler(sample_df, method=scale_method)
    sample_scaled = scaler.transform(sample_df[FEATURE_COLS].values)

    # (3) Reference Downsampling (스트리밍 + Reservoir Sampling)
    ref_samples = reservoir_sample_reference(
        ref_path, scaler, k=downsample_k, chunksize=chunksize, fmt=fmt
    )

    # (4) 밀도 모델 학습 -> Reference PDF
    model = fit_density_model(ref_samples, model_type=model_type, **model_kwargs)

    # (5) Threshold 결정 (Reference 표본 기반)
    threshold = choose_threshold(model, ref_samples, percentile=threshold_pct)

    # (6) Sample 평가
    result = evaluate_samples(model, sample_scaled, threshold)

    print("\n" + "=" * 60)
    print(f"  [알고리즘 B] 확률 밀도 기반 커버리지 결과 (model={model_type.upper()})")
    print("=" * 60)
    print(f"  Sample 평균 log-likelihood   : {result['mean_log_likelihood']:.4f}")
    print(f"  Sample 중앙값 log-likelihood : {result['median_log_likelihood']:.4f}")
    print(f"  Sample 표준편차              : {result['std_log_likelihood']:.4f}")
    print(f"  고밀도 영역 커버 샘플 수      : "
          f"{result['n_in_region']:,} / {result['n_samples']:,}")
    print(f"  >>> Coverage Ratio (≥Thr)    : {result['coverage_ratio']:.4f} "
          f"({result['coverage_ratio'] * 100:.2f}%)")
    print("=" * 60)
    return result


# =============================================================================
# 6. 엔트리포인트 (더미 데이터 자동 생성 후 실행)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="확률 밀도 기반 커버리지")
    parser.add_argument("--sample", default="dummy_sample.csv")
    parser.add_argument("--ref", default="dummy_reference.csv")
    parser.add_argument("--fmt", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--model", default="kde", choices=["kde", "gmm"])
    parser.add_argument("--scale", default="standard", choices=["minmax", "standard"])
    parser.add_argument("--downsample_k", type=int, default=200_000,
                        help="Reference 에서 추출할 표본 수 (10만~50만 권장)")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--threshold_pct", type=float, default=10.0,
                        help="Reference log-likelihood 하위 몇 %% 를 Threshold 로 쓸지")
    parser.add_argument("--bandwidth", type=float, default=0.3, help="KDE bandwidth")
    parser.add_argument("--n_components", type=int, default=32, help="GMM 성분 수")
    parser.add_argument("--ref_rows", type=int, default=2_000_000)
    parser.add_argument("--gen", action="store_true")
    args = parser.parse_args()

    # 더미 데이터 생성
    if args.gen or not (os.path.exists(args.sample) and os.path.exists(args.ref)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample_csv(args.sample, n_rows=1800)
        gen.make_reference(args.ref, total_rows=args.ref_rows, fmt=args.fmt)

    # 모델별 하이퍼파라미터 전달
    model_kwargs = {}
    if args.model == "kde":
        model_kwargs["bandwidth"] = args.bandwidth
    else:
        model_kwargs["n_components"] = args.n_components

    run(
        sample_path=args.sample,
        ref_path=args.ref,
        model_type=args.model,
        scale_method=args.scale,
        downsample_k=args.downsample_k,
        chunksize=args.chunksize,
        threshold_pct=args.threshold_pct,
        fmt=args.fmt,
        **model_kwargs,
    )
