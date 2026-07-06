# ShinOne

CSV 데이터에서 `cluster_id` 별로 대표 행을 하나씩 추출하는 도구.

| 스크립트 | 용도 |
|----------|------|
| `extract_cluster_samples.py` | 순차 처리. 작은 파일용, 코드가 단순함 |
| `extract_cluster_samples_fast.py` | 멀티프로세스 병렬 처리. 수십 GB 대용량 파일용 |

## extract_cluster_samples_fast.py (대용량 파일용)

파일을 워커 수만큼 바이트 구간으로 나눠 여러 CPU가 동시에 스캔한다.
행 전체를 파싱하지 않고 `cluster_id` 필드만 잘라내고, 이미 본 클러스터의
행은 즉시 건너뛰기 때문에 순차 버전보다 코어당 처리량 자체도 훨씬 높다.
메모리 사용량은 파일 크기와 무관하게 거의 일정하다(클러스터별 후보 행만 유지).

```bash
# 기본: CPU 코어 수만큼 워커 사용
python3 extract_cluster_samples_fast.py input.csv -o output.csv

# 워커 수 지정
python3 extract_cluster_samples_fast.py input.csv -o output.csv -j 16

# random / last 모드, 기준 컬럼 변경은 순차 버전과 동일한 옵션 사용
python3 extract_cluster_samples_fast.py input.csv --mode random --seed 42
```

실측: 4코어 컨테이너에서 709 MiB / 1,600만 행 처리에 **3.2초 (약 220 MiB/s)**.
같은 파일을 순차 버전으로 처리하면 102초가 걸린다 (약 32배).
결과는 순차 버전과 동일함을 검증했다 (first/last 모드 diff 일치,
워커 수를 바꿔도 결과 불변, random 모드는 시드 고정 시 재현 가능).

제약: 따옴표로 감싼 필드 **안에 줄바꿈**이 있는 CSV는 지원하지 않는다
(따옴표 안의 쉼표는 지원). 숫자/해시 데이터에는 해당 사항이 없다.

## extract_cluster_samples.py (순차 버전)

`G1, G14, ..., cluster_id, X, Y, hashkey` 형태의 CSV를 읽어,
`cluster_id` 값별로 한 행씩 골라 새 CSV로 저장한다.
`cluster_id` 컬럼만 있으면 나머지 컬럼 구성은 자유이며,
파이썬 표준 라이브러리만 사용하므로 별도 설치가 필요 없다 (Python 3.8+).

### 사용법

```bash
# 기본: 각 클러스터의 첫 번째 행을 추출 → cluster_samples.csv
python3 extract_cluster_samples.py input.csv

# 출력 경로 지정
python3 extract_cluster_samples.py input.csv -o output.csv

# 각 클러스터에서 무작위로 한 행 추출 (시드로 재현 가능)
python3 extract_cluster_samples.py input.csv -o output.csv --mode random --seed 42

# 그룹 기준 컬럼 이름이 다른 경우
python3 extract_cluster_samples.py input.csv -k my_cluster_col
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-o`, `--output` | `cluster_samples.csv` | 출력 CSV 경로 |
| `-k`, `--key-column` | `cluster_id` | 그룹 기준 컬럼 이름 |
| `-m`, `--mode` | `first` | 클러스터 내 행 선택 방식: `first` / `last` / `random` |
| `--seed` | 없음 | `random` 모드에서 결과 재현용 시드 |

### 예시

`sample_data.csv` 로 바로 테스트할 수 있다:

```bash
python3 extract_cluster_samples.py sample_data.csv -o result.csv
```

출력(클러스터 0~3의 첫 행 하나씩):

```csv
G1,G14,cluster_id,X,Y,hashkey
0.12,3.4,0,10.5,20.1,a1b2c3
0.98,1.2,1,11.0,19.8,d4e5f6
0.31,4.1,2,15.2,25.6,j1k2l3
0.15,1.8,3,30.1,40.2,v4w5x6
```
