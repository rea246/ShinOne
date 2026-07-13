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
- **병렬화 = 멀티프로세스 (스레드 아님)**: CPython 의 GIL 때문에 pandas CSV 파싱 / KDE 스코어링 같은
  CPU 작업은 스레드로는 거의 안 빨라진다(실측 ~1.0x). `ProcessPoolExecutor` 로 각 프로세스가 독립 GIL 을
  가질 때에만 멀티코어로 실제 단축된다(실측 2~4x).
  - 알고리즘 A: **Parquet** Reference 는 row group 을 프로세스에 분배해 병렬 집계. (CSV 는 분할이 어려워 단일 패스)
  - 알고리즘 B: KDE `score_samples` 를 프로세스로 분할 병렬.
  - → **50GB 급 데이터는 Parquet 포맷을 권장** (병렬 + 빠른 파싱).

## 실행

파라미터는 CLI 인자가 아니라 각 스크립트 상단의 `# CONFIG` 블록 변수로 지정한다 (argparse 미사용).
값을 바꾸려면 해당 변수를 직접 편집한다.

```bash
pip install numpy pandas scikit-learn pyarrow

# 알고리즘 A — 상단 CONFIG(N_BINS, FMT, N_WORKERS_A ...) 편집 후:
python coverage_bincount.py

# 알고리즘 B — 상단 CONFIG(MODEL_TYPE='kde'|'gmm', DOWNSAMPLE_K ...) 편집 후:
python coverage_density.py
```

실제 데이터를 쓰려면 `SAMPLE_PATH`, `REF_PATH`, `FMT`('csv'|'parquet') 를 실제 경로로 바꾸고
`GEN_DUMMY = False` 로 둔다. 주요 CONFIG: `SCALE_METHOD`('minmax'|'standard'), `CHUNKSIZE`,
`N_BINS`(A), `MODEL_TYPE`/`DOWNSAMPLE_K`/`THRESHOLD_PCT`(B), `N_WORKERS`(공통, 기본 CPU 수).

## 지표

- **알고리즘 A**: `Coverage = |Set_sam ∩ Set_ref| / |Set_ref|`
- **알고리즘 B**: Sample 평균 log-likelihood + Reference 분포에서 데이터 기반으로 정한
  Threshold(하위 `threshold_pct`% 분위수) 이상을 만족하는 Sample 비율.
