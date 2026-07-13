# 다차원 공간 커버리지 분석 (Coverage Analysis)

N차원 반도체/EDA Feature Vector 데이터셋에 대해, 대용량(약 50GB) **Reference Set** 대비
1,800행 **Sample Set** 의 공간 커버리지를 계산하는 베이스라인 코드.
(차원 수는 입력 컬럼 선택으로 결정 — 아래 *입력 컬럼 설정* 참고. 기본값은 21차원.)

## 구성

| 파일 | 설명 |
|------|------|
| `common.py` | 공통 유틸: 더미 데이터 생성기, Sample 기준 스케일러 학습, 대용량 파일 chunk 스트리밍 |
| `coverage_bincount.py` | **알고리즘 A** — BIN COUNT 기반 격자 점유 커버리지 (Grid-based Occupancy) |
| `coverage_density.py` | **알고리즘 B** — 확률 밀도(KDE/GMM) 기반 커버리지 |
| `coverage_domain.py` | **Domain Coverage** — 물리 도메인(스펙 박스 / Mahalanobis 타원체) 기준 커버리지 + UMAP/PCA 시각화 |
| `coverage_visualize.py` | **시각화** — PCA 2D 겹침 그림 + 차원별 Factor 영향력 그림(N_FEATURES 장) |

## 입력 컬럼 설정 (중요)

입력 CSV/Parquet 에서 **앞쪽 열들만 Feature Vector 이고 뒤쪽 열은 사용하지 않는** 경우가 많다.
읽을 컬럼은 `common.py` 상단에서 **위치(index) 기준**으로 한 곳에서 지정한다
(Sample·Reference·모든 스크립트가 동일 선택을 공유해야 하므로):

```python
# common.py
FEATURE_COL_IDX = list(range(0, 21))   # 0~20번째 열(총 21개)만 사용, 그 뒤 열은 무시
HAS_HEADER      = True                  # CSV 에 헤더 행이 있으면 True, 없으면 False
# N_FEATURES 는 위 선택에서 자동 산출 (여기선 21)
```

- `usecols` 로 **선택한 열만 파싱**하므로, 뒤쪽 미사용 열은 문자열이어도 안전하게 무시된다.
- 20개만 쓰려면 `range(0, 20)`, 특정 열만 쓰려면 임의 인덱스 리스트(예: `[0,1,2,5,8]`)도 가능.

## 핵심 설계

- **메모리 안전**: Reference(50GB)는 절대 한 번에 로드하지 않는다.
  - CSV → `pandas.read_csv(chunksize=...)`, Parquet → `pyarrow.iter_batches(batch_size=...)`
- **정규화 기준은 Sample Set**: Sample 으로 스케일러를 `fit` 한 뒤, 동일 스케일러를 Reference 에 `transform`.
- **차원의 저주 대응 (알고리즘 A) — PCA 축소**: N차원 전체 축을 그대로 격자화하면 고차원 Sparsity 로
  교집합이 0 이 되어 Coverage 가 0% 로 수렴한다. 이를 막기 위해 **Sample 로 PCA 를 학습해 누적 설명분산이
  90%(`VAR_THRESHOLD`)를 넘는 최소 주성분 수 K 만 남기고**(그 PCA 를 Reference 에도 공유), 축소된 PC 공간에서
  격자화한다. PC 축마다 스케일이 다르므로 **Sample PC 좌표의 축별 [min,max] 로 축마다 개별 Bin 경계**를 만든다.
  격자 배열(Bin^K) 대신 "실제 점유 격자의 `set`"만 관리(Sparse). 선택된 K 는 실행 로그에 출력된다.
  - **한 번의 스트리밍 패스**로 아래 4종을 동시 집계한다 (map 이 4종 부분결과 반환 → reduce 가 각각 합침;
    `parallel_reduce_reference` 구조 유지):
    1. `bincount_2d_coverage.png` — 전통적 2D PCA(PC1 vs PC2) 격자 커버리지.
    2. `custom_2d_coverage.png` — **원본 핵심 피처 1개(`RAW_FEATURE_IDX`)를 X축에 그대로 살리고**,
       나머지 (N-1)차원을 `PCA(n=1)`로 압축해 Y축에 둔 커버리지. 축의 물리적 의미를 보존하며,
       "나머지 차원을 1D로 압축했을 때의 설명 분산 비율(EVR)"을 로그·타이틀에 명시.
    3. `per_axis_coverage.png` — **축(피처)별 완전 독립 1D bin-count 커버리지** (Bin=`PER_AXIS_BINS`).
       21개 피처 각각 `|sam_bins ∩ ref_bins| / |ref_bins|` 를 구해, 취약 축 오름차순으로 터미널 출력 +
       가로 막대 차트로 저장. → 어느 Feature 축에 Sample 공백(구멍)이 있는지 진단.
    (1D 축별 커버리지는 차원의 저주가 없어 `N_BINS` 보다 촘촘한 `PER_AXIS_BINS` 를 쓴다.)
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

# 시각화 — 상단 CONFIG(PLOT_DOWNSAMPLE, OUTPUT_DIR ...) 편집 후:
python coverage_visualize.py    # 필요 시 pip install matplotlib seaborn
```

### 시각화 (`coverage_visualize.py`)

- **Plot 1 — `coverage_plots/pca_2d_overlap.png`**: N차원 → PCA 2D 투영 후 Reference(모집단 표본)와
  Sample 을 seaborn 산점도 + 각 집단 2D KDE 등고선으로 겹쳐 그린다. → 두 분포가 **겹치는 영역**과
  Sample 이 Reference 를 **벗어난 영역**을 눈으로 구분.
- **Plot 2 — `coverage_plots/per_dimension/factor_fNN.png` (N_FEATURES 장)**: 차원 `d` 를 그대로 x축에 두고
  **나머지 (N-1)차원을 PCA 로 1D 압축**해 y축에 둔 산점도를 차원마다 한 장씩 저장. → 각 Factor 축을 따라
  Sample/Reference 의 분포·이탈을 개별 확인.
- Reference(50GB)는 `reservoir_sample_reference` 로 `PLOT_DOWNSAMPLE` 개만 균일 추출해 그린다.
- 그림 텍스트는 폰트 호환을 위해 ASCII(영문) 로 표기(한글 폰트 미설치 환경에서도 안 깨짐).

실제 데이터를 쓰려면 `SAMPLE_PATH`, `REF_PATH`, `FMT`('csv'|'parquet') 를 실제 경로로 바꾸고
`GEN_DUMMY = False` 로 둔다. 주요 CONFIG: `SCALE_METHOD`('minmax'|'standard'), `CHUNKSIZE`,
`N_BINS`(A), `MODEL_TYPE`/`DOWNSAMPLE_K`/`THRESHOLD_PCT`(B), `N_WORKERS`(공통, 기본 CPU 수).

## Domain Coverage (`coverage_domain.py`)

데이터 분포가 아니라 **물리적 도메인**을 기준으로 삼는다. `DOMAIN_MODE` 로 두 가지 도메인 정의를 고른다.

### `DOMAIN_MODE = "box"` — 축-정렬 스펙 초입방체
각 피처의 Lower/Upper Spec 이 이루는 N차원 초입방체를 `N_BINS` 로 균등 분할해 가상의 도메인 격자
`Set_domain`(= `N_BINS^N_FEATURES`, 데이터 없어도 분모 포함)을 정의. `Reference/Sample Domain Coverage
= |Set_ref/sam ∩ Set_domain| / |Set_domain|`. 스펙 밖 점은 Out-of-Spec 으로 제외.
스펙 경계는 `SPEC_LOWER`/`SPEC_UPPER` 직접 입력 또는 `SPEC_MODE`('minmax'|'4sigma') 자동.
- **한계**: 피처가 상관되면 박스 '모서리'가 도달 불가능한 빈 공간인데도 분모(`N_BINS^N_FEATURES`)에 들어가
  절대 커버리지가 극단적으로 작다(예 ~1e-8). → 상관구조를 반영하는 `mahalanobis` 모드 권장.

### `DOMAIN_MODE = "mahalanobis"` — 상관구조 반영 타원체 (권장)
Reference 의 평균 μ·공분산 Σ 로 정의한 타원체 `{ x : (x−μ)ᵀΣ⁻¹(x−μ) ≤ χ²(q, d) }` 를 '도달 가능한
도메인'으로 삼는다(박스의 빈 모서리를 잘라냄).
- **μ·Σ 는 `parallel_reduce_reference` 로 한 스트리밍 패스에 추정**(`[n, Σx, ΣxxT]` 누적) — 모델 학습 불필요.
- 커버리지는 **whitened(백색화) top-`MAHAL_GRID_DIMS` 타원 안쪽 격자만** 분모로 삼아(corner-free, 유한),
  차원의 저주 없이 유의미한 값을 준다. `MAHAL_Q`(기본 0.99)로 타원 크기 조절.
- **Out-of-Domain 비율**(full-d Mahalanobis > χ²)도 출력: Reference 는 ≈`1−q`, Sample 은 분포 이탈 정도를 나타냄.
- 주요 CONFIG: `MAHAL_Q`, `MAHAL_GRID_DIMS`(2 권장, 최대 3), `N_BINS_MAHAL`.

### 시각화 (공통 + Mahalanobis 전용)
- Sample + Reference 하이라이트 표본만으로 **UMAP·PCA 2D** 임베딩 → `domain_coverage_umap.png` /
  `domain_coverage_pca.png`. 격자 영역을 (회색=빈 도메인 / 연파랑=Reference only / 주황=Sample∩Ref 검증완료)로
  칠하고 테두리를 그린다. (UMAP 미설치 시 PCA 만)
- Mahalanobis 모드는 추가로 `domain_mahalanobis_2d.png`: whitened top-2 평면에 **타원 경계 + 격자 커버리지**
  (도메인=타원 안, Reference/검증 칸)를 그린다.

## 지표

- **알고리즘 A**: `Coverage = |Set_sam ∩ Set_ref| / |Set_ref|` (PCA 90% 축소된 K차원 PC 격자 위에서 계산;
  검증용 2차원·커스텀 2D·축별 1D 커버리지도 함께 출력).
- **알고리즘 B**: Sample 평균 log-likelihood + Reference 분포에서 데이터 기반으로 정한
  Threshold(하위 `threshold_pct`% 분위수) 이상을 만족하는 Sample 비율.
- **Domain Coverage**: 물리 도메인(`box`=스펙 초입방체 / `mahalanobis`=상관구조 타원체) 대비
  Reference·Sample 커버리지 + Out-of-Domain 비율.
