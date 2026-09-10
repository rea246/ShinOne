# B Case: Reference 기준 coverage와 집단 영향력 보고서

A 브랜치의 `7.analysis.py`와 `8.compare_coverage.py`에서 사용한 방식처럼, 이미 계산된
전체 KP 공간의 거리 판정을 reference 기준 2D 격자에 집계한다. B의 기존
`group_reinforce_nnvote.py`의 역거리 top-K 투표 함수를 공유한다. Sample 또는 집단으로
scaler/kPCA를 다시 fit하지 않으며 집단마다 반경을 바꾸지 않는다.

## 실행

기존 `coverage_domain_kpca.py`의 입력 설정 후 실행하면 기존 출력과 함께
`coverage_plots/influence/coverage_influence_report.html`이 자동 생성된다.
`WRITE_INFLUENCE_REPORT`, `INFLUENCE_TOPK`, `HEATMAP_BINS`,
`INFLUENCE_GROUP_WORDS`, `INFLUENCE_GROUP_DELIM`으로 보고서 옵션을 설정한다.
`gauge_name`이 없으면 전체 sample을 한 집단으로 표시한다.

이미 만든 scatter를 재사용할 때는 저장된 반경을 읽는다. 다시 fit할 필요가 없다.

```bash
python coverage_analysis/coverage_influence_report.py \
  --ref out_s1/kpca_reference_scatter.csv \
  --sample out_s1/kpca_sample_scatter.csv \
  --summary out_s1/kpca_summary_metrics.json \
  --out-dir coverage_analysis/coverage_plots/influence
```

특정 REF 집단 또는 gap 부분집합을 평가하려면, 동일 reference scatter를 필터링한
CSV/TSV를 `--target`으로 지정한다. 정규화에는 항상 전체 `--ref`를 사용한다.

```bash
python coverage_analysis/coverage_influence_report.py \
  --ref out_s1/kpca_reference_scatter.csv \
  --sample out_s1/kpca_sample_scatter.csv --target filtered.csv \
  --radius 0.2666 --group-words A B C --topk 3 --bins 24 \
  --out-dir coverage_analysis/coverage_plots/filtered_influence
```

위 반경은 사용법 예시다. 실제 실험은 해당 run의 R 또는 사전에 정한 공통 R을 사용한다.
기본 집단은 `gauge_name`의 `_` 앞 접두어다. `--group-col`로 다른 열을 지정할 수 있다.
단어 그룹은 첫 일치 단어에만 배정하며 미일치 sample은 `other`다. 중복 소속은 없다.

`--target`이 없으면 기존 `cover_status`에서 out_of_domain을 제외한 HDR REF를 분석한다.
상태 열이 없으면 전체 유효 REF를 쓴다. 명시한 `--target`은 입력된 유효행 전부를
분석하므로 out_of_domain 행도 사용자가 넣었다면 포함한다. 결과 JSON에 범위를 기록한다.

## 순위의 의미

| 지표 | 정의 / 해석 |
|---|---|
| total_score, score_pct | 각 target가 top-K sample에 배분한 1점의 집단별 합 / 전체 투표 비율 |
| score_per_pattern | total_score / 집단의 유효 sample 수; 그룹 크기를 나눈 점수 |
| coverage_pct | 해당 집단만으로 고정 R 안에서 덮는 target 비율 |
| unique_loss_pp | 해당 집단을 통째로 제거했을 때 전체 coverage 감소량 (퍼센트포인트) |
| shared_coverage_pct | 해당 집단과 다른 집단이 함께 덮는 target 비율 |
| gain_in_vote_order_pp | 총점 내림차순으로 집단을 추가할 때 새로 덮는 target 비율; 순서에 의존 |
| cumulative_coverage_pct | 위 순서로 집단들을 합친 coverage |

`vote_rank`, `efficiency_rank`, `coverage_rank`, `unique_rank`를 따로 기록한다.
동점은 같은 순위이며 누적 추가 순서의 동점은 집단명 순으로 처리한다.
Sample CSV에도 개별 패턴의 투표점수, 단독 coverage, 제거 손실과 원본 열을 기록한다.
0점 집단과 패턴도 저장한다. 모든 비율의 분모는 선택된 target 수이며,
단독 coverage들을 더하면 중복 때문에 100%를 넘을 수 있다.

**투표는 R-free이고 미커버 target도 항상 1점을 배분한다.** 먼 sample이 높은 투표를 받더라도
실제 reach가 충분하다는 뜻이 아니다. 반경 coverage, 거리 지도, coverage–R 곡선을 같이 본다.
K 경계의 동일 거리 이웃은 SciPy cKDTree의 선택을 따른다. 동점 sample의 순서를 바꾸거나
K를 바꾸면 배점이 변할 수 있다. 한 집단뿐이면 제거 손실은 전체 coverage와 같다.
중복 sample은 서로를 대체할 수 있어 개별 제거 손실이 0일 수 있다.

## Heatmap과 HTML

- 전체 REF 수, 선택 target 수, 전체 sample coverage, 평균 최근접 거리: 세 KP 축 쌍.
- 집단 선택: 집단 coverage, 집단 제거 손실, 칸별 평균 투표점유율.
- 총점 / 패턴당 점수 / 제거 손실 순위 그림, 정렬 가능한 전체 집단 표.
- 상위 10개 집단과 전체 sample의 coverage–R 곡선, 모든 집단의 누적 coverage.
- Sample 상위 100개 표; CSV는 전체 sample을 포함.

격자 경계는 전체 REF의 좌표 범위에서 계산한다. 모든 집단은 같은 경계를 사용한다.
빈 target bin은 NaN/회색이며 0%와 구별한다. 보간·KDE smoothing은 하지 않는다.
내부 경계는 오른쪽/위 bin, 마지막 경계는 마지막 bin에 포함한다.
중복 좌표 REF도 개별 행으로 센다. 격자 수를 바꿔도 coverage와 순위는 변하지 않는다.
한 REF만 있는 bin도 포함하므로 비율은 target 수 heatmap과 함께 읽는다.

보고서 이미지와 선택/정렬 기능은 HTML 하나에 내장되어 인터넷 없이 열린다.
CSV 다운로드 링크를 쓰려면 같은 폴더의 CSV도 함께 전달한다.

| 출력 | 내용 |
|---|---|
| coverage_influence_report.html | 오프라인 보고서 |
| group_influence_ranking.csv | 모든 집단의 투표·coverage·제거 손실·누적 순위 |
| sample_influence_ranking.csv | 모든 sample의 점수·coverage·제거 손실 + 원본 행 |
| reference_influence.csv | 선택 target의 원본 행, coverage, 최근접 top-K sample/집단/거리/투표 |
| coverage_influence_bins.csv | 공유 bin 경계와 전체 REF/target 수, 전체·집단 coverage와 투표 |
| coverage_influence_summary.json | 전체 지표·반경·축·정규화·제외행 수·입력 출처·프레임 정보 |
| coverage_heatmap.png, group_*_heatmap.png | 전체 및 집단별 지도 |
| group_ranking.png, coverage_curves.png | 순위와 변화 곡선 |

## 일관성 및 한계

새 kPCA run은 실제 fitted frame으로 만든 `kpca_frame_id`를 reference/sample/gap CSV에
같이 기록한다. 서로 다른 ID 또는 일부에만 ID가 있는 입력은 거부한다. 새로운 summary는
R을 반올림하지 않고 저장해 후처리 시 경계 판정이 바뀌지 않도록 한다.
기존 CSV는 동작하지만 shared frame을 확인할 수 없다는 메시지를 남긴다.
Frame ID는 출처 일관성 검사이며 CSV 좌표 변조를 탐지하는 전자서명은 아니다.
명시 target의 KP 좌표와 중복 횟수는 reference 부분집합인지 검사한다. CSV 재저장의
부동소수점 오차를 허용하기 위해 이 소속 검사만 소수점 10자리로 비교한다.
거리 계산과 coverage 판정에는 반올림하지 않은 좌표를 사용한다.
추가 kPCA 축도 scatter에 모두 저장하며 자동 보고서는 모든 축으로 거리를 계산한다.
그림은 첫 세 축의 쌍을 사용한다. 별도 CLI에서 축을 바꾸려면 `--kp-cols`를 지정한다.

비유한 좌표는 제외하고 개수를 보고한다. Reference scale은 전체 유효 REF의 population
standard deviation(ddof=0), 상수축은 1이다. 원본 scatter의 0-based 데이터 행번호를
`influence_source_row`로 보존한다. 외부 sample과 REF가 같은 실제 패턴인지 ID로 추론해
자동 제거하지 않으며 self-hit도 기본 coverage에 포함한다.

분석은 reference reservoir의 기하학적 관계다. 물리적 중요도/모델 성능 기여도는
`CLAIM_STRATEGY.md`의 독립 Mask/OPC 검증으로 확인해야 한다.
집단 수가 수천 개면 target × group 행렬과 집단별 PNG/CSV/HTML 크기가 커진다.
그 경우 도메인에 맞는 상위 그룹 열을 사용하거나 집단 단위 batch 처리로 확장한다.

## 검증 / 합성 미리보기

기존 의존성 NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn만 사용한다.

```bash
python -m unittest discover -s coverage_analysis -p test_coverage_influence.py -v
python coverage_analysis/demo_coverage_influence.py
```

Demo는 겹치는 집단·독립 영역 집단·0점 집단·공동 공백을 가진 고정 seed 합성 데이터다.
`coverage_analysis/coverage_plots/influence_demo/coverage_influence_report.html`에 저장하며,
실제 ShinOne 측정 결과가 아님을 보고서 제목과 provenance에 표시한다.
