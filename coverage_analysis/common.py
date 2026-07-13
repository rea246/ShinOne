# -*- coding: utf-8 -*-
"""
common.py
=========
15차원 커버리지 분석을 위한 공통 유틸리티 모듈.

이 모듈은 알고리즘 A(BIN COUNT)와 알고리즘 B(Density) 스크립트가 함께 사용하는
아래 기능을 제공한다.

    1) DummyDataGenerator : 50GB 급 대용량 Reference / Sample 데이터를 시뮬레이션
    2) fit_scaler         : Sample Set 기준으로 스케일러(Scaler)를 학습
    3) iter_reference_chunks : 대용량 Reference 파일을 chunksize 단위로 스트리밍

핵심 설계 원칙
--------------
* Reference(약 50GB)는 절대 메모리에 한 번에 올리지 않는다.
  -> pandas 의 chunksize / pyarrow 의 batch 단위로만 접근한다.
* 정규화 기준은 "Sample Set" 이다.
  -> Sample 로 스케일러를 fit 한 뒤, 그 스케일러를 Reference 에도 동일하게 transform.
  -> (Sample 이 우리가 커버 여부를 평가하려는 '관심 분포'이므로 그 스케일을 기준으로 삼음)
"""

import os
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# =============================================================================
# 전역 상수
# =============================================================================
N_FEATURES = 15                                  # Feature Vector 차원 수
FEATURE_COLS = [f"f{i:02d}" for i in range(N_FEATURES)]  # ['f00', 'f01', ... 'f14']


# =============================================================================
# 1. Dummy Data Generator
#    - 실제 50GB 파일이 없어도 파이프라인을 end-to-end 로 검증할 수 있도록
#      가상의 Reference / Sample 데이터를 생성한다.
#    - Reference 는 파일 크기를 키우기 위해 '행을 나눠' 반복 append 한다.
# =============================================================================
class DummyDataGenerator:
    """
    15차원 가상 데이터 생성기.

    Parameters
    ----------
    n_features : int
        Feature 차원 수 (기본 15).
    seed : int
        재현성을 위한 난수 시드.
    """

    def __init__(self, n_features: int = N_FEATURES, seed: int = 42):
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

        # Feature 별로 스케일이 '서로 다르다'는 전제를 반영하기 위해
        # 각 축마다 서로 다른 평균(loc)과 표준편차(scale)를 부여한다.
        #   예) f00 은 0~1 스케일, f14 는 수천 단위 스케일이 되도록.
        self.locs = self.rng.uniform(-5.0, 5.0, size=n_features) * \
            self.rng.integers(1, 1000, size=n_features)
        self.scales = self.rng.uniform(0.5, 3.0, size=n_features) * \
            self.rng.integers(1, 500, size=n_features)

    # -------------------------------------------------------------------------
    def _sample_block(self, n_rows: int, shift: float = 0.0) -> np.ndarray:
        """
        n_rows 개의 15차원 벡터를 생성한다.

        shift : Reference 대비 Sample 의 분포를 살짝 이동시켜
                '커버리지가 100% 가 아닌' 현실적인 상황을 만들기 위한 파라미터.
        """
        # 다변량 정규분포(축마다 독립) 기반. 축별 스케일 차이를 그대로 반영.
        block = self.rng.normal(
            loc=self.locs + shift * self.scales,
            scale=self.scales,
            size=(n_rows, self.n_features),
        )
        return block.astype(np.float32)

    # -------------------------------------------------------------------------
    def make_sample_csv(self, path: str, n_rows: int = 1800) -> str:
        """
        가벼운 Sample Set 을 생성해 CSV 로 저장한다.
        Sample 은 Reference 대비 분포를 조금 이동(shift)시켜,
        커버리지가 자연스럽게 100% 미만이 되도록 만든다.
        """
        block = self._sample_block(n_rows, shift=0.3)
        df = pd.DataFrame(block, columns=FEATURE_COLS)
        df.to_csv(path, index=False)
        print(f"[DummyData] Sample Set 저장 완료: {path}  (shape={df.shape})")
        return path

    # -------------------------------------------------------------------------
    def make_reference(
        self,
        path: str,
        total_rows: int = 2_000_000,
        rows_per_block: int = 200_000,
        fmt: str = "csv",
    ) -> str:
        """
        대용량 Reference Set 을 생성한다.

        실제로 50GB 를 만들면 테스트가 오래 걸리므로, total_rows 를 조절해
        '충분히 큰' 파일을 만든다. (기본 200만 행)
        메모리 폭발을 피하기 위해 rows_per_block 단위로 나누어 append 저장한다.

        fmt : 'csv' 또는 'parquet'
        """
        if fmt not in ("csv", "parquet"):
            raise ValueError("fmt 는 'csv' 또는 'parquet' 만 지원한다.")

        # 기존 파일이 있으면 삭제 (append 모드로 누적하므로)
        if os.path.exists(path):
            os.remove(path)

        n_blocks = (total_rows + rows_per_block - 1) // rows_per_block

        # parquet 는 append 를 위해 ParquetWriter 를 사용한다.
        writer = None
        try:
            for b in range(n_blocks):
                n = min(rows_per_block, total_rows - b * rows_per_block)
                block = self._sample_block(n, shift=0.0)  # Reference 는 shift 없음
                df = pd.DataFrame(block, columns=FEATURE_COLS)

                if fmt == "csv":
                    # 첫 블록만 header 기록, 이후는 append
                    header = (b == 0)
                    df.to_csv(path, mode="a", header=header, index=False)
                else:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(path, table.schema)
                    writer.write_table(table)

                if (b + 1) % 5 == 0 or b == n_blocks - 1:
                    print(f"[DummyData] Reference 진행률: "
                          f"{(b + 1) / n_blocks * 100:5.1f}%  "
                          f"({(b + 1) * rows_per_block:,} rows)")
        finally:
            if writer is not None:
                writer.close()

        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"[DummyData] Reference Set 저장 완료: {path}  "
              f"({total_rows:,} rows, {size_mb:.1f} MB, fmt={fmt})")
        return path


# =============================================================================
# 2. 스케일러 학습 (Sample Set 기준)
# =============================================================================
def fit_scaler(sample_df: pd.DataFrame, method: str = "minmax"):
    """
    Sample Set 을 기준으로 스케일러를 학습한다.

    Parameters
    ----------
    sample_df : Sample Set DataFrame (FEATURE_COLS 컬럼 보유)
    method    : 'minmax' (Min-Max Scaling) 또는 'standard' (Standard Scaling)

    Returns
    -------
    scaler : 학습된 sklearn Scaler 객체 (Reference 에도 동일하게 transform 에 사용)
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("method 는 'minmax' 또는 'standard' 만 지원한다.")

    scaler.fit(sample_df[FEATURE_COLS].values)
    print(f"[Scaler] Sample 기준 '{method}' 스케일러 학습 완료. "
          f"(n_features={scaler.n_features_in_})")
    return scaler


# =============================================================================
# 3. 대용량 Reference 스트리밍 리더
#    - CSV : pandas.read_csv(chunksize=...)
#    - Parquet : pyarrow.ParquetFile.iter_batches(...)
#    - 두 경우 모두 '스케일링된 numpy 배열' 을 chunk 단위로 yield 한다.
# =============================================================================
def iter_reference_chunks(
    ref_path: str,
    scaler,
    chunksize: int = 200_000,
    fmt: Optional[str] = None,
) -> Iterator[np.ndarray]:
    """
    Reference 파일을 chunk 단위로 스트리밍하며,
    각 chunk 를 (Sample 기준으로 학습된) scaler 로 변환해 yield 한다.

    Parameters
    ----------
    ref_path  : Reference 파일 경로
    scaler    : fit_scaler 로 학습된 스케일러
    chunksize : 한 번에 읽을 행 수 (메모리 사용량을 결정)
    fmt       : 'csv' / 'parquet'. None 이면 확장자로 자동 판별.

    Yields
    ------
    np.ndarray : shape (chunk_rows, 15), 스케일링 완료된 배열
    """
    if fmt is None:
        fmt = "parquet" if ref_path.endswith((".parquet", ".pq")) else "csv"

    if fmt == "csv":
        # pandas 의 chunksize 로 반복자(iterator)를 얻는다.
        reader = pd.read_csv(ref_path, chunksize=chunksize, usecols=FEATURE_COLS)
        for chunk_df in reader:
            # 스케일 변환 후 numpy 배열로 반환
            yield scaler.transform(chunk_df[FEATURE_COLS].values)

    elif fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(ref_path)
        # batch_size 단위로 스트리밍 (RowGroup 을 넘나들며 균일 배치 제공)
        for batch in pf.iter_batches(batch_size=chunksize, columns=FEATURE_COLS):
            chunk_df = batch.to_pandas()
            yield scaler.transform(chunk_df[FEATURE_COLS].values)

    else:
        raise ValueError("fmt 는 'csv' 또는 'parquet' 만 지원한다.")
