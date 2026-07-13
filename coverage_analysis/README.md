# 15차원 공간 커버리지 분석 (Coverage Analysis)

15차원 반도체/EDA Feature Vector 데이터셋에 대해, 대용량(약 50GB) **Reference Set** 대비
1,800행 **Sample Set** 의 공간 커버리지를 계산하는 베이스라인 코드.

## 구성

| 파일 | 설명 |
|------|------|
| `common.py` | 공통 유틸: 더미 데이터 생성기, Sample 기준 스케일러 학습, 대용량 파일 chunk 스트리밍 |
| `coverage_bincount.py` | **알고리즘 A** — BIN COUNT 기반 격자 점유 커버리지 (Grid-based Occupancy) |
| `coverage_density.py` | **알고리즘 B** — 확률 밀도(KDE/GMM) 기반 커버리지 |

## 핵심 설계

- **메모리 안전**: Reference(50GB)는 절대 한 번에 로드하지 않는다.
  - CSV → `pandas.read_csv(chunksize=...)`, Parquet → `pyarrow.iter_batches(batch_size=...)`
- **정규화 기준은 Sample Set**: Sample 으로 스케일러를 `fit` 한 뒤, 동일 스케일러를 Reference 에 `transform`.
- **차원의 저주 대응 (알고리즘 A)**: 축당 Bin 을 3~5개로 거칠게 이산화하고, 전체 격자 배열(N^15) 대신
  "실제로 점유된 격자 좌표 tuple 의 `set`"만 관리(Sparse).
- **Downsampling (알고리즘 B)**: 단일 스트리밍 패스에서 **Reservoir Sampling**으로 균일하게 10만~50만 행만 추출해
  KDE/GMM 학습에 사용. 메모리는 O(k) 고정.

## 실행

```bash
pip install numpy pandas scikit-learn pyarrow

# 알고리즘 A (더미 데이터 자동 생성)
python coverage_bincount.py --gen --n_bins 4

# 알고리즘 B — KDE
python coverage_density.py --gen --model kde --downsample_k 200000

# 알고리즘 B — GMM
python coverage_density.py --gen --model gmm --n_components 32

# 실제 데이터 사용 (Parquet 예시)
python coverage_bincount.py --sample sample.csv --ref reference.parquet --fmt parquet
```

주요 옵션: `--fmt {csv,parquet}`, `--chunksize`, `--scale {minmax,standard}`,
`--ref_rows`(더미 크기), `--threshold_pct`(알고리즘 B 고밀도 컷오프 분위수).

## 지표

- **알고리즘 A**: `Coverage = |Set_sam ∩ Set_ref| / |Set_ref|`
- **알고리즘 B**: Sample 평균 log-likelihood + Reference 분포에서 데이터 기반으로 정한
  Threshold(하위 `threshold_pct`% 분위수) 이상을 만족하는 Sample 비율.
