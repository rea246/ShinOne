# -*- coding: utf-8 -*-
"""
common.py
=========
15차원 커버리지 분석 공통 유틸리티.

제공 기능
    1) DummyDataGenerator       : 대용량 Reference / 가벼운 Sample 가상 데이터 생성
    2) fit_scaler               : Sample Set 기준 스케일러 학습
    3) iter_reference_chunks    : 대용량 Reference 파일을 chunk 단위 스트리밍
    4) parallel_reduce_reference: Reference 를 map→reduce 로 집계 (Parquet 는 프로세스 병렬)
    5) parallel_score           : 밀도 모델 score_samples 를 프로세스로 분할 병렬

설계 원칙
    * Reference(약 50GB)는 절대 한 번에 메모리에 올리지 않는다 (chunk 스트리밍).
    * 정규화 기준은 Sample Set 이다 (Sample 로 fit → Reference 에 동일 transform).
    * 병렬화는 '스레드'가 아니라 '프로세스'로 한다.
      CPython 의 GIL 때문에 pandas CSV 파싱 / sklearn KDE 스코어링 같은 CPU 작업은
      스레드로는 거의 빨라지지 않는다(실측 ~1.0x). ProcessPoolExecutor 로 각 프로세스가
      독립 GIL 을 가질 때에만 멀티코어로 실제 단축된다(실측 2~4x).
        - Parquet Reference : row group 을 프로세스에 분배해 병렬 집계.
        - CSV Reference     : 분할이 어려워 단일 패스 스트리밍(→ 50GB 급은 Parquet 권장).
        - 밀도 모델 스코어링 : score_samples 를 프로세스로 분할 병렬.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---- 입력 데이터의 Feature 컬럼 선택 (위치 index 기준) ----------------------
# 실제 입력 CSV/Parquet 는 '앞쪽 열들' 이 Feature Vector 이고 '뒤쪽 열' 은 사용하지 않는다.
# 아래 인덱스(0-based, 위치 기준)에 해당하는 열만 읽고, 그 외 열은 아예 파싱하지 않는다.
#   예) 0~20번째 열이 벡터, 그 뒤는 미사용  →  FEATURE_COL_IDX = list(range(0, 21))
FEATURE_COL_IDX = list(range(0, 21))     # 0..20 열 = 총 21개 (끝 포함). 20개면 range(0, 20).
HAS_HEADER      = True                   # 입력 CSV 에 헤더 행이 있으면 True, 없으면 False

# ---- 위 선택에서 자동 산출되는 상수 (직접 수정 불필요) ----------------------
N_FEATURES   = len(FEATURE_COL_IDX)                        # 실제 사용 차원 수
FEATURE_COLS = [f"f{i:02d}" for i in range(N_FEATURES)]    # 내부 표준 컬럼명
N_WORKERS    = min(8, os.cpu_count() or 1)                 # 기본 병렬 프로세스 수


def _parquet_feature_names(path):
    """Parquet 의 '위치 index' 를 실제 컬럼명으로 변환해 선택 대상 이름 목록을 돌려준다."""
    import pyarrow.parquet as pq
    all_names = pq.ParquetFile(path).schema_arrow.names
    return [all_names[i] for i in FEATURE_COL_IDX]


def read_feature_matrix(path, fmt=None):
    """
    작은 파일(예: Sample) 전체를 읽어 (n, N_FEATURES) float32 배열로 반환한다.
    FEATURE_COL_IDX 위치의 열만 선택하고 나머지(뒤쪽 미사용 열)는 무시한다.
    """
    fmt = fmt or ("parquet" if path.endswith((".parquet", ".pq")) else "csv")
    if fmt == "csv":
        # usecols 에 정수 리스트를 주면 '열 위치' 기준으로 선택 (뒤쪽 열은 파싱조차 안 함)
        df = pd.read_csv(path, usecols=FEATURE_COL_IDX,
                         header=(0 if HAS_HEADER else None))
    else:
        import pyarrow.parquet as pq
        names = _parquet_feature_names(path)
        df = pq.read_table(path, columns=names).to_pandas()
    return df.values.astype(np.float32)


# =============================================================================
# 1. Dummy Data Generator
# =============================================================================
class DummyDataGenerator:
    """
    N차원(기본 N_FEATURES=21) 가상 데이터 생성기. 축마다 스케일이 다른 상황을 재현한다.
    실제 입력을 모사하기 위해 Feature 열 뒤에 '사용하지 않는 열'(n_extra개)도 함께 써서,
    위치 기준 컬럼 선택(FEATURE_COL_IDX)이 뒤쪽 열을 제대로 무시하는지 확인할 수 있게 한다.
    """

    def __init__(self, seed=42, n_extra=3):
        self.rng = np.random.default_rng(seed)
        self.n_extra = n_extra              # Feature 뒤에 붙일 미사용(junk) 열 개수
        # 축별로 서로 다른 평균/표준편차 → 'feature 마다 스케일이 다르다'는 전제 반영
        self.locs = self.rng.uniform(-5, 5, N_FEATURES) * self.rng.integers(1, 1000, N_FEATURES)
        self.scales = self.rng.uniform(0.5, 3, N_FEATURES) * self.rng.integers(1, 500, N_FEATURES)

    def _block(self, n_rows, shift=0.0):
        """축별 스케일을 반영한 (n_rows, N_FEATURES) 정규분포 블록. shift 는 분포 이동량."""
        return self.rng.normal(
            self.locs + shift * self.scales, self.scales, size=(n_rows, N_FEATURES)
        ).astype(np.float32)

    def _frame(self, n_rows, shift=0.0):
        """Feature 열 + 미사용 열(문자열/숫자)을 붙인 DataFrame. 위치 기준 선택 검증용."""
        df = pd.DataFrame(self._block(n_rows, shift), columns=FEATURE_COLS)
        for e in range(self.n_extra):        # 뒤쪽에 '안 쓰는 열' 추가 (일부러 문자열도 섞음)
            df[f"unused_{e}"] = "junk" if e == 0 else self.rng.random(n_rows)
        return df

    def make_sample(self, path, n_rows=1800):
        """가벼운 Sample Set 을 CSV 로 저장. Reference 대비 살짝 shift → 커버리지 < 100%."""
        self._frame(n_rows, shift=0.3).to_csv(path, index=False)
        print(f"[DummyData] Sample 저장: {path} (rows={n_rows}, "
              f"feature={N_FEATURES} + unused={self.n_extra})")
        return path

    def make_reference(self, path, total_rows=2_000_000, rows_per_block=200_000, fmt="csv"):
        """대용량 Reference 를 block 단위로 나눠 append 저장 (메모리 폭발 방지)."""
        if os.path.exists(path):
            os.remove(path)
        n_blocks = (total_rows + rows_per_block - 1) // rows_per_block
        writer = None                       # parquet append 용 ParquetWriter
        try:
            for b in range(n_blocks):
                n = min(rows_per_block, total_rows - b * rows_per_block)
                df = self._frame(n)
                if fmt == "csv":
                    df.to_csv(path, mode="a", header=(b == 0), index=False)
                else:
                    import pyarrow as pa, pyarrow.parquet as pq
                    table = pa.Table.from_pandas(df, preserve_index=False)
                    writer = writer or pq.ParquetWriter(path, table.schema)
                    writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        mb = os.path.getsize(path) / 1024 ** 2
        print(f"[DummyData] Reference 저장: {path} ({total_rows:,} rows, {mb:.1f} MB, {fmt})")
        return path


# =============================================================================
# 2. 스케일러 학습 (Sample Set 기준)
# =============================================================================
def fit_scaler(sample, method="minmax"):
    """Sample Set(배열 (n, N_FEATURES))으로 스케일러를 학습한다. method: 'minmax'|'standard'."""
    X = sample.values if hasattr(sample, "values") else np.asarray(sample)
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    scaler.fit(X)
    print(f"[Scaler] Sample 기준 '{method}' 스케일러 학습 완료. (dim={X.shape[1]})")
    return scaler


# =============================================================================
# 3. 대용량 Reference chunk 스트리밍 (스케일링 포함)
# =============================================================================
def iter_reference_chunks(ref_path, scaler, chunksize=200_000, fmt=None):
    """
    Reference 를 chunk 단위로 읽어 scaler 로 변환한 numpy 배열을 yield 한다.
    FEATURE_COL_IDX 위치의 열만 읽고, 뒤쪽 미사용 열은 무시한다.
    """
    fmt = fmt or ("parquet" if ref_path.endswith((".parquet", ".pq")) else "csv")
    if fmt == "csv":
        reader = pd.read_csv(ref_path, chunksize=chunksize,
                             usecols=FEATURE_COL_IDX, header=(0 if HAS_HEADER else None))
        for chunk in reader:                       # chunk 는 선택 열만, 오름차순 위치 순서
            yield scaler.transform(chunk.values)
    else:
        import pyarrow.parquet as pq
        names = _parquet_feature_names(ref_path)
        for batch in pq.ParquetFile(ref_path).iter_batches(batch_size=chunksize, columns=names):
            yield scaler.transform(batch.to_pandas().values)


# =============================================================================
# 3-b. Reservoir Sampling — 단일 스트리밍 패스로 균일 k개 추출 (메모리 O(k))
#      밀도 모델 학습(알고리즘 B)과 시각화(PCA)에서 공용으로 쓴다.
# =============================================================================
def reservoir_sample_reference(ref_path, scaler, k, chunksize=200_000, fmt=None, seed=0):
    """Reference 를 chunk 스트리밍하며 전체에서 균일하게 k개 행을 추출한다 (Algorithm R)."""
    rng = np.random.default_rng(seed)
    reservoir = np.empty((k, N_FEATURES), dtype=np.float32)
    filled = seen = 0
    for chunk in iter_reference_chunks(ref_path, scaler, chunksize, fmt):
        # (A) reservoir 가 아직 안 찼으면 우선 채운다
        if filled < k:
            take = min(k - filled, len(chunk))
            reservoir[filled:filled + take] = chunk[:take]
            filled += take
            seen += take
            chunk = chunk[take:]
            if len(chunk) == 0:
                continue
        # (B) 가득 찬 상태: t번째 행을 확률 k/t 로 임의 슬롯과 교체
        t = seen + np.arange(1, len(chunk) + 1)
        accept = rng.random(len(chunk)) < (k / t)
        if accept.any():
            reservoir[rng.integers(0, k, accept.sum())] = chunk[accept]
        seen += len(chunk)
    reservoir = reservoir[:filled]
    print(f"[Reservoir] 전체 {seen:,} rows → {len(reservoir):,} rows 추출.")
    return reservoir


# =============================================================================
# 4. Reference map→reduce 집계 (Parquet 는 프로세스 병렬)
#    map_fn / reduce_fn / init_factory 는 반드시 '모듈 최상위 함수' 여야 pickle 가능.
#    - map_fn(X)           : 스케일링된 chunk(np.ndarray) → 부분 결과
#    - reduce_fn(acc, part): 부분 결과 누적 (in-place 후 acc 반환 권장)
#    - init_factory()      : 빈 누적기 생성 (예: set)
# =============================================================================
def set_union(acc, other):
    """set 누적용 reduce 함수 (합집합)."""
    acc |= other
    return acc


def _rowgroup_reduce(task):
    """[프로세스 워커] 배정된 Parquet row group 들을 읽어 map→reduce 한 부분 결과 반환."""
    ref_path, rg_indices, scaler, map_fn, reduce_fn, init_factory = task
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(ref_path)
    names = _parquet_feature_names(ref_path)          # 위치 index → 실제 컬럼명
    acc = init_factory()
    for rg in rg_indices:
        tbl = pf.read_row_group(rg, columns=names)
        X = scaler.transform(tbl.to_pandas().values)
        acc = reduce_fn(acc, map_fn(X))
    return acc


def parallel_reduce_reference(ref_path, scaler, map_fn, reduce_fn, init_factory,
                              n_workers=N_WORKERS, chunksize=200_000, fmt=None):
    """
    Reference 전체를 map→reduce 로 집계한다.
    Parquet 이면 row group 을 프로세스에 분배해 멀티코어 병렬, 그 외엔 단일 패스 스트리밍.
    """
    fmt = fmt or ("parquet" if ref_path.endswith((".parquet", ".pq")) else "csv")

    if fmt == "parquet" and n_workers > 1:
        import pyarrow.parquet as pq
        n_rg = pq.ParquetFile(ref_path).num_row_groups
        if n_rg > 1:
            # round-robin 으로 row group 을 워커에 분배 (부하 균형)
            buckets = [list(range(i, n_rg, n_workers)) for i in range(n_workers)]
            buckets = [b for b in buckets if b]
            tasks = [(ref_path, b, scaler, map_fn, reduce_fn, init_factory) for b in buckets]
            result = init_factory()
            with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
                for part in ex.map(_rowgroup_reduce, tasks):
                    result = reduce_fn(result, part)
            return result

    # ---- CSV 또는 단일 row group: 단일 패스 스트리밍 ----
    result = init_factory()
    for X in iter_reference_chunks(ref_path, scaler, chunksize, fmt):
        result = reduce_fn(result, map_fn(X))
    return result


# =============================================================================
# 5. 밀도 모델 score_samples 병렬화 (KDE 스코어링이 무거울 때, 프로세스 분할)
# =============================================================================
def _score_part(task):
    """[프로세스 워커] 모델로 부분 X 를 스코어링."""
    model, X = task
    return model.score_samples(X)


def parallel_score(model, X, n_workers=N_WORKERS):
    """X 를 n_workers 조각으로 나눠 score_samples 를 프로세스 병렬 실행 후 이어붙인다."""
    if len(X) < 4000 or n_workers <= 1:            # 작은 입력은 프로세스 오버헤드가 더 큼
        return model.score_samples(X)
    parts = np.array_split(X, n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        return np.concatenate(list(ex.map(_score_part, [(model, p) for p in parts])))
