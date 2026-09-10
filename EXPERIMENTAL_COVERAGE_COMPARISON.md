# Stage 8: A / B / REF coverage comparison

## 1. Names and execution

| Name in Stage 8 | Source |
|---|---|
| A | 6번 topology 대표군을 7번에서 저장한 `topology_sample.npz` |
| B | 다른 방법으로 추린 `B.txt`의 pattern key 목록 |
| REF | 같은 7번 run의 `reference_sample.npz` |

7번에서는 기존 topology 대표군을 역사적으로 B라고 불렀다. **8번은 사용자의 비교 명칭에
맞춰 기존 topology=A, 외부 목록=B로 명시한다.** 7번의 기존 저장 파일이나 필드를 바꾸지 않는다.

프로젝트 폴더에 `B.txt`를 두고 실행한다. 한 줄에 실제 gauge name(pattern key) 하나를 쓴다.
첫 줄의 `pattern_key` 헤더는 선택 사항이다. 아래 이름은 형식 설명용 예시다.

```text
pattern_key
gauge_000001
gauge_000017
gauge_000052
```

```bash
python 8.compare_coverage.py
```

기본 입력은 같은 폴더의 `B.txt`, `hkeys_features.pt`, 그리고
`topology_analysis_out/latest_run.json`이 가리키는 성공한 7번 run이다.
다른 파일이나 특정 run을 쓰려면 다음과 같이 지정한다.

```bash
python 8.compare_coverage.py --b-keys B.txt --stage7-run topology_analysis_out/run_<timestamp>
python 8.compare_coverage.py --b-keys /path/to/B.txt --cache /path/to/hkeys_features.pt
```

결과는 `coverage_comparison_out/run_<UTC>/comparison_report.html`에서 본다.
PNG를 내장하므로 인터넷 연결 없이 HTML 하나로 표와 그림을 열 수 있다.
입력·검증·벡터 조회·거리 계산·저장 단계와 예외는 `comparison.log`에 즉시 기록한다.
실패하면 partial 출력과 로그를 유지하고 마지막 성공 run의 pointer는 바꾸지 않는다.

```bash
python 8.compare_coverage.py --replot latest
python 8.compare_coverage.py --replot coverage_comparison_out/run_<timestamp>
```

`--replot`은 8번 snapshot의 fingerprint를 검증하고 저장된 조건으로 진단 대표·그림·HTML을
재생성한다. 캐시/GPU/재투영/최근접 거리 계산이 필요 없다. 기본 비교 NPZ·CSV·summary와
7번 원본은 유지한다. 다른 출력 폴더는 `--out-dir`로 지정한다.

## 2. Pattern keys to the original learned embedding

1. UTF-8(BOM 허용) text를 읽고 빈 줄 및 각 줄 앞뒤 공백을 제거한다. 대소문자, 이름 내부
   문자와 선행 0은 보존한다. CSV 여러 열이나 좌표 파일을 자동 추정하지 않는다.
2. 중복 key는 첫 등장 순서를 유지해 한 번만 사용하고 입력 횟수를 기록한다.
3. `hkeys_features.pt['keys']`에서 문자열 완전 일치로 찾는다. 누락되거나 같은 key가
   여러 cache row에 존재하면 `b_key_lookup.csv`에 전부 기록하고 중단한다. 가까운 이름이나
   좌표로 대체하지 않고, 실패한 패턴을 빼서 coverage를 산출하지 않는다.
4. 해당 row의 `h0(8), h1(8), h2(8), h3(8), edge(8)`을 순서대로 연결한다.
   현재 40D를 그대로 읽으며 7번 frame의 feature 이름·차원·순서와 일치하는지 확인한다.
5. 캐시에서 읽은 REF와 A의 key/row 및 실제 embedding 값이 7번 snapshot과 같은지 확인한다.
   인코더를 바꾼 캐시나 row 순서가 다른 캐시로 비교하지 않는다. 전체 캐시 모든 tensor의
   동일성까지 검증한다는 의미는 아니며, 저장된 REF/A 값과의 호환성 검사다.

B는 캐시에 존재하는 패턴이면 읽을 수 있다. 6번에서 B를 다시 clustering하거나 선택하지
않는다. 1,172개 등 실제 고유 key 개수를 사용하며 1,000개로 자르지 않는다. 이 비교는
실제 두 집합의 coverage이며, 대표 수가 동일한 조건의 알고리즘 우위 실험은 아니다.

## 3. Reuse the Stage-7 frame

REF sampling, normalization fit, landmark selection, gamma, kPCA eigenvectors,
axis scaling, R은 모두 7번에서 고정한 것을 사용한다. 7번의 성공 snapshot이 이후 변경된
6번 CSV보다 우선한다. 원본 40D의 정규화는 `normalize_embeddings`, 새 B의 projection은
`transform_raw_embeddings`를 재사용한다. 재학습이나 반경 재산정은 하지 않는다.

B 중 이미 A 또는 REF에 있는 동일 row는 저장된 projection을 재사용한다. A와 REF가
겹치면 A snapshot 좌표를 우선한다. 별도 batch/backend의 수치 오차가 동일 패턴을 서로
다른 위치로 만들지 않도록 하며, 특히 A=B control에서 가짜 exclusive 영역을 피한다.
나머지 B만 기존 frame에 transform한다.

RBF kernel은 PyTorch CUDA 사용 가능 시 GPU, 나머지 exact block distance와 그래프 집계는
NumPy/SciPy를 사용한다. 추가 RAPIDS 설치는 필요 없다. 20K × 20K kernel이나 REF 전량의
재학습을 수행하지 않는다.

## 4. Coverage events and bin probability

표준화된 모든 KPC 축에서 `d_A(r)`와 `d_B(r)`는 REF r에서 각 대표군까지의 최근접 거리다.
A 거리는 7번 저장값을 읽고 B 거리를 같은 metric으로 계산한다. `d <= R`이 covered다.

| REF state | Condition |
|---|---|
| both_covered | d_A ≤ R, d_B ≤ R |
| A_only | d_A ≤ R, d_B > R |
| B_only | d_A > R, d_B ≤ R |
| both_uncovered | d_A > R, d_B > R |

네 상태는 겹치지 않고 전체 REF를 정확히 나눈다. A/B의 최근접 거리 CDF를 같은 축에 그려
R에 따른 coverage trend를 비교한다. 원래 정규화 40D 거리 요약도 보조 지표로 저장하지만,
coverage와 bin 확률은 표준화 kPCA 거리 기준이다.

bin j에서 사용자가 요청한 A-only 비율은 다음과 같다.

\[
p_{A\text{-only},j} =
\frac{\#\{r\in REF_j : d_A(r)\le R,\ d_B(r)>R\}}{\#REF_j}.
\]

**분모는 그 bin 안의 전체 REF**다. A가 커버한 REF만 분모에 넣는 조건부 비율이 아니다.
B 미커버 비율은 `(A_only + both_uncovered) / REF`이며 A-only 비율과 다르다.
같은 bin에서 네 상태의 비율 합은 1이다.

경계는 전체 REF의 KP 좌표 최소/최대 범위에서 정하고 모든 A/B 그림이 공유한다.
`HEATMAP_BINS=None`은 저장된 7번 진단 grid를 이어받고, 해당 정보가 없으면 50을 사용한다.
정수로 설정하면 해당 격자를 사용한다. 내부 경계는 오른쪽/위 bin에 배정하며 마지막 bin은
최댓값을 포함한다. 중복 좌표의 서로 다른 REF도 각각 센다. 빈 bin은 회색/NaN이며,
CSV 비율을 공란으로 저장한다. KDE, smoothing, 비어 있는 공간의 보간은 없다.

REF 중 A 또는 B에 포함된 row를 union으로 찾아 **양쪽에서 동일하게 제외한 보조 비교**도
기록한다. 기본 coverage와 heatmap은 기존 7번처럼 전체 REF를 사용한다. A-only 패턴이
A 대표 자체인지도 대표 표와 CSV/NPZ에서 확인할 수 있다.

## 5. Real representatives from 100% A-only bins

- 비어 있지 않고 `A_only_count == REF_count`인 bin에서만 후보를 모은다. 반올림된 100%를
  사용하지 않는다. REF 1개짜리 bin도 포함하되 표본 수를 함께 보여준다.
- KP1/KP2, KP1/KP3, KP2/KP3 중 하나라도 해당하면 후보로 넣고 row 기준 한 번만 센다.
- 후보 집합에서 기존 farthest-first radius cover를 사용한다. 첫 패턴은 B에서 가장 먼 후보,
  이후에는 이미 선택한 진단 대표에서 가장 먼 후보를 고른다. 모든 후보가 진단 대표 중
  하나의 `R * GAP_RADIUS_MULTIPLIER` 안에 들어오면 종료한다.
- 실제 패턴만 추출하며 고정 개수나 bin당 한 개 규칙을 강제하지 않는다. 그룹 ID는
  `AO0001` 등으로 붙이며, 구성원 수 내림차순으로 표시한다.
- 그룹 대표는 자기 bin이 100% A-only인 축 쌍에만 표시한다. 번호는 이동할 수 있지만
  연결선 끝의 마름모가 실제 위치다. 상위 `GAP_PREVIEW_GROUPS=20`개는 그림과 HTML의
  첫 표에 표시하고 나머지도 HTML 펼침 표와 CSV/NPZ/JSON에 모두 저장한다.
- 100% bin이 없으면 진단 대표는 0개다. 혼합 bin의 A-only REF는 전체
  `a_only_ref_patterns.csv`에 남으며 coverage에서 제거하지 않는다.

100%는 이번 REF 표본에서 관측한 비율이며 전체 연속 공간의 물리적 커버 보장은 아니다.
격자 수를 바꾸면 진단 후보/대표도 바뀌지만 개별 패턴의 A/B coverage와 R은 그대로다.

## 6. Outputs

| Output | Contents |
|---|---|
| `comparison_report.html` | 오프라인 A/B/REF 요약·거리 통계·곡선·히트맵·실제 진단 대표 표 |
| `comparison.log`, `comparison_summary.json` | 단계별 로그, input identity, A/B 실제 개수, 네 상태 통계, 공통 보조 비교, 조건·fingerprint |
| `b_input.txt`, `b_key_lookup.csv` | 실제 입력 사본과 이름별 일치/누락/다중 매칭·중복 횟수 |
| `b_features.csv`, `b_sample.npz` | B의 실제 key/row, 원본 learned 40D와 kPCA 좌표; NPZ에 normalized/distance 좌표 포함 |
| `a_sample.npz`, `reference_frame.npz` | A snapshot과 고정 frame |
| `reference_comparison.npz`, `ref_comparison.csv` | 전체 REF의 key/row/40D/좌표, A/B 거리·최근접 key, 네 상태·self membership |
| `b_uncovered_ref_patterns.csv` | B가 놓친 REF 전량 |
| `a_only_ref_patterns.csv` | A가 커버하고 B가 놓친 REF 전량; 혼합 bin도 포함 |
| `coverage_bins.csv` | 세 축 쌍의 모든 bin 경계·REF 수·네 상태의 개수와 비율 |
| `a_only_bins.csv`, `a_only_summary.json` | 100% bin 경계·REF 수, 후보/제외 수, 대표 묶음 조건과 전체 그룹 |
| `a_only_representatives.csv`, `.npz` | 진단 대표의 실제 ID/40D/좌표, 구성원 수, A/B 거리·최근접 key, source bin |
| `a_only_members.csv` | 100% bin 후보 REF와 진단 대표의 연결 |
| `ab_ref_coverage_heatmap.png` | 위 REF 수, 가운데 A 미커버 비율, 아래 B 미커버 비율 |
| `a_only_coverage_heatmap.png` | A 커버 ∩ B 미커버 / 전체 REF in bin |
| `a_only_representatives_heatmap.png` | 같은 A-only heatmap과 100% bin 실제 대표 |
| `coverage_radius_trend.png` | 같은 REF에서의 A/B coverage–R 곡선 |

검증: `python -m unittest test_compare_coverage -v`
