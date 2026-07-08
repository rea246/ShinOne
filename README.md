# ShinOne

CSV 데이터에서 `cluster_id` 별로 대표 행을 하나씩 추출하는 도구.

두 추출 스크립트의 기본 선택 방식은 `random`(시드 42) 이다. 입력 CSV가
좌표순으로 정렬돼 있어도 클러스터마다 모든 행이 같은 확률로 뽑히므로
(reservoir sampling), 파일을 실제로 섞지 않고도 "셔플 후 첫 행 추출"과
통계적으로 동일하다. 기존처럼 첫 행이 필요하면 `--mode first` 를 쓴다.

| 스크립트 | 용도 |
|----------|------|
| `extract_cluster_samples.py` | 순차 처리. 작은 파일용, 코드가 단순함 |
| `extract_cluster_samples_fast.py` | 멀티프로세스 병렬 처리. 수십 GB 대용량 파일용 |
| `match_nearest_gauge.py` | 추출 결과(B.csv)에 input.pkl 의 최근접 게이지를 붙여 C.csv 생성 |

## extract_cluster_samples_fast.py (대용량 파일용)

파일을 워커 수만큼 바이트 구간으로 나눠 여러 CPU가 동시에 스캔한다.
행 전체를 파싱하지 않고 `cluster_id` 필드만 잘라내므로 순차 버전보다
코어당 처리량도 훨씬 높다. 메모리 사용량은 파일 크기와 무관하게
거의 일정하다(클러스터별 후보 행만 유지).

`main()` 상단의 변수를 수정한 뒤 실행한다:

```python
def main():
    input_path = "input.csv"   # 입력 CSV
    output_path = "B.csv"      # 출력 CSV
    key_column = "cluster_id"  # 그룹 기준 컬럼
    mode = "random"            # first / last / random
    seed = 42                  # random 모드 재현용 시드
    workers = None             # None 이면 CPU 코어 수
```

```bash
python3 extract_cluster_samples_fast.py
```

실측: 4코어 컨테이너에서 709 MiB / 1,600만 행 처리에 **3.2초 (약 220 MiB/s)**.
같은 파일을 순차 버전으로 처리하면 102초가 걸린다 (약 32배).
first/last 모드 결과는 순차 버전과 동일하고 워커 수와 무관하다.
random 모드는 시드+워커 수가 같으면 재현된다.

제약: 따옴표로 감싼 필드 **안에 줄바꿈**이 있는 CSV는 지원하지 않는다
(따옴표 안의 쉼표는 지원). 숫자/해시 데이터에는 해당 사항이 없다.

## match_nearest_gauge.py (B.csv → C.csv 최근접 게이지 매칭)

`extract_cluster_samples_fast.py` 의 결과물(B.csv)의 각 행에 대해,
`preprocessor.py` 가 읽는 것과 같은 포맷의 게이지 리스트
(`input.pkl`, `[[gauge_name, cx, cy], ...]` 를 pickle.dump 한 파일)에서
가장 가까운 게이지를 찾아 `gauge_name, gauge_x, gauge_y, dist` 컬럼을
붙인 C.csv 를 만든다.

B.csv 의 X, Y 와 input.pkl 의 좌표는 단위가 달라서, B.csv 좌표를
`--scale`(기본 20000)로 나눠 input.pkl 좌표계로 맞춘 뒤 비교한다.
`dist` 는 input.pkl 좌표계 기준 유클리드 거리다.

다음 두 경우의 행은 C.csv 에서 제외된다:

- 최근접 게이지가 `max_dist`(기본 0.1, pkl 좌표계 단위) 이상 떨어져 있는 행
- 매칭된 `gauge_name` 이 앞선 행에서 이미 사용된 행
  (B.csv 순서 기준, 먼저 나온 행이 게이지를 차지)

`main()` 상단의 변수를 수정한 뒤 실행한다:

```python
def main():
    b_csv = "B.csv"
    input_pkl = "input.pkl"
    output = "C.csv"
    scale = 20000.0   # B.csv 좌표를 pkl 좌표계로 맞추는 나눗셈 값
    max_dist = 0.1    # 이 거리 이상 떨어진 행은 제외
    inspect = None    # cluster_id 를 넣으면 그 행의 top-5 최근접만 출력 (검증용)
```

```bash
python3 match_nearest_gauge.py
```

실행 시 양쪽 좌표 범위를 출력하고, 범위가 겹치지 않거나 크기가 100배
이상 다르면 scale 오류 가능성을 경고한다.

의존성: numpy 필수, scipy 권장(cKDTree — 게이지가 수백만 개여도 빠름).
scipy 가 없으면 numpy 브루트포스로 자동 대체된다.

## extract_cluster_samples.py (순차 버전)

`G1, G14, ..., cluster_id, X, Y, hashkey` 형태의 CSV를 읽어,
`cluster_id` 값별로 한 행씩 골라 새 CSV로 저장한다.
`cluster_id` 컬럼만 있으면 나머지 컬럼 구성은 자유이며,
파이썬 표준 라이브러리만 사용하므로 별도 설치가 필요 없다 (Python 3.8+).

### 사용법

```bash
# 기본: 각 클러스터에서 무작위로 한 행 추출(시드 42) → cluster_samples.csv
python3 extract_cluster_samples.py input.csv

# 출력 경로 지정
python3 extract_cluster_samples.py input.csv -o output.csv

# 다른 시드로 다른 표본 추출
python3 extract_cluster_samples.py input.csv -o output.csv --seed 7

# 각 클러스터의 첫 번째 행 추출 (예전 기본 동작)
python3 extract_cluster_samples.py input.csv --mode first

# 그룹 기준 컬럼 이름이 다른 경우
python3 extract_cluster_samples.py input.csv -k my_cluster_col
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o`, `--output` | `cluster_samples.csv` | 출력 CSV 경로 |
| `-k`, `--key-column` | `cluster_id` | 그룹 기준 컬럼 이름 |
| `-m`, `--mode` | `random` | 클러스터 내 행 선택 방식: `random` / `first` / `last` |
| `--seed` | `42` | `random` 모드에서 결과 재현용 시드 |

### 예시

`sample_data.csv` 로 바로 테스트할 수 있다:

```bash
python3 extract_cluster_samples.py sample_data.csv -o result.csv
```

출력(클러스터 0~3에서 무작위로 한 행씩, 시드 42 기준):

```csv
G1,G14,cluster_id,X,Y,hashkey
0.60,2.2,0,10.4,20.0,s1t2u3
0.98,1.2,1,11.0,19.8,d4e5f6
0.31,4.1,2,15.2,25.6,j1k2l3
0.15,1.8,3,30.1,40.2,v4w5x6
```
