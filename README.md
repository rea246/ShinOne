# ShinOne

CSV 데이터에서 `cluster_id` 별로 대표 행을 하나씩 추출하는 도구.

## extract_cluster_samples.py

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
