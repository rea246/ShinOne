# Stage 7: REF-fixed kPCA coverage of topology representatives

## 1. Scope and execution

```bash
python 7.analysis.py
```

사람이 결과를 읽을 때는 결과 폴더의 `analysis_report.html`을 브라우저로 연다.
요약 수치·표·그래프를 한 페이지에서 보며, 이미지를 HTML 안에 포함해 인터넷 연결이나
추가 뷰어 설치가 필요 없다. JSON/NPZ는 자동 처리와 후속 8번 재사용용으로 유지한다.

이미 성공한 7번 결과가 있으면 kPCA를 다시 실행하지 않고 새 heatmap과 보고서만 포함해
그림을 재생성할 수 있다.

```bash
python 7.analysis.py --replot latest
# 특정 run을 지정할 수도 있다.
python 7.analysis.py --replot topology_analysis_out/run_<timestamp>
```

`--replot`은 기존 run의 PNG/HTML을 갱신하고 `replot.log`를 추가 기록한다. 저장된
REF 표본·좌표계·반경·거리·coverage를 그대로 읽으며 NPZ/JSON/CSV와 latest pointer는
수정하지 않는다. 원본 cache나 GPU 없이도 NumPy와 Matplotlib로 재시각화할 수 있다.

6번에서 이미 선택한 약 1,000개의 실제 대표 패턴(B)을 REF와 비교한다.
6번의 clustering/representative selection은 다시 수행하지 않는다.
외부 방식 대표군(A)을 읽고 A/B/REF를 비교하는 작업은 **후속 8번의 범위**다.
이번 7번에는 A 입력이나 A/B 우위 판정을 넣지 않는다.

참고한 흐름은 `claude/kpca-gap-patterns-validation-e9uk0s` 브랜치의
`coverage_analysis/coverage_domain_kpca.py`다. REF로만 fit하고 sample은 transform만
하는 원칙과 최근접 거리 기반 reach coverage를 이어받는다. 다음은 다르다.

- 입력: handcrafted 21D가 아니라 cached learned topology embedding 40D.
- 기본 REF 평가 표본 20,000개, kPCA 학습 landmark 2,000개.
- HDR 99% 필터나 density 상위 gap 필터를 적용하지 않고, 추출된 REF 전부를 평가한다.
- 반경은 B의 실제 개수(예: 995)가 아니라 고정 비교 예산 1,000으로 계산한다.
- 원본 row/key를 보존한다. 좌표 반올림 조인이나 feature 중복 제거를 하지 않는다.
- 8번용 좌표계·REF·대표군 snapshot을 pickle 없는 NPZ로 보존한다.

## 2. Input identity and the original 40 dimensions

| Input | Use |
|---|---|
| `hkeys_features.pt` | Stage 4의 `keys`와 `features[h0,h1,h2,h3,edge]` 복원 |
| `topology_clustering_out/topology_labels.npz` | 6번 유효 population row와 H0/topology label |
| `topology_clustering_out/topology_representatives.csv` | B의 실제 `global_row`, `pattern_key`, label |
| `topology_clustering_out/topology_run_metadata.json` | block 순서·차원·가중치·population fingerprint 검증 |
| 선택적 `REF_PATTERN_KEYS_CSV` | 별도 REF 그룹의 고유 `pattern_key` 목록 |

기본 REF는 **6번의 전체 유효 population**이다. 별도 REF 그룹이 있으면 파일 상단에
`REF_PATTERN_KEYS_CSV = _here / "ref_pattern_keys.csv"`처럼 설정한다. 해당 key는
같은 cache와 6번 population에 존재해야 한다. 다른 DB의 별도 embedding 파일은 이번
7번에서 자동 결합하지 않는다.

각 block이 8D인 현재 모델에서 입력은 `[h0,h1,h2,h3,edge]`의 40D다. 미래 차원 변경에는
저장된 6번 metadata 차원을 따르며, hard-coded 첫 40개 열을 읽지 않는다. `h0_raw`나
원래 geometry feature를 learned block 대신 넣지 않는다.

row 범위·중복, representative key/label, population fingerprint가 다르면 중단한다.
이 fingerprint는 **row/key 정렬** 검증이지 전체 cache tensor의 내용 해시는 아니다.
같은 key로 encoder를 다시 추출했다면 6번도 같은 cache로 다시 실행해야 한다.
샘플에 NaN/Inf가 있으면 조용히 삭제하지 않고 오류로 알린다.

## 3. REF sampling and normalization

REF 후보 population에서 seed 0의 uniform sampling without replacement로 20,000개를
뽑는다. 후보가 더 작으면 전량을 쓴다. 대표군 B 포함 여부는 sampling에 영향을 주지
않는다. H0 rare `-1`도 후보에 포함되며 별도 rare oversampling은 하지 않는다.

20,000개는 빠른 전체 분포 비교의 기본값이지 희귀 유형 포착 보장은 아니다.
빈도 0.01%인 유형을 하나도 못 뽑을 확률은 단순 근사에서
`(1-0.0001)^20000 ≈ 13.5%`다. 희소 영역 추정이 중요해지면 `N_REF=50_000` 같은
확대 실험을 할 수 있지만, A/B 결과를 본 뒤 유리한 seed나 표본을 고르면 안 된다.

40D 정규화의 평균과 표준편차는 **추출된 REF 20,000개만**으로 계산한다.
6번의 full-population scaler를 그대로 재사용하는 것이 아니다. block별 가중치 규칙은
6번 metadata에서 이어받는다.

\[
z^{(b)}_{ij} = \frac{x^{(b)}_{ij}-\mu^{(b)}_{R,j}}{\sigma^{(b)}_{R,j}}
              \sqrt{w_b/d_b}.
\]

상수 feature의 표준편차는 1로 둔다. B와 향후 A는 이 고정 정규화에 transform만 한다.

## 4. Landmark kPCA and saved coordinates

REF sample 중 seed를 고정해 최대 2,000개 landmark를 별도 추출한다.
20,000 × 20,000 커널은 생성하지 않는다.

\[
K_{ij}=\exp(-\gamma\lVert z_i-z_j\rVert^2),\qquad
\gamma=\frac{1}{2\operatorname{median}\lVert z_i-z_j\rVert^2}.
\]

gamma는 REF landmark pair의 거리로만 정한다. 2,000개의 float64 커널 행렬 하나는
약 30.5 MiB다. 전체 메모리는 eigen workspace, 변환 block, 원본 cache 등을 더 필요로 한다.
`TRANSFORM_BLOCK=2_048`로 20,000개 REF 및 B를 나누어 변환한다.

PyTorch CUDA가 사용 가능하면 RBF kernel 계산을 GPU에서 수행한다. 없으면 SciPy CPU를
사용한다. 작은 landmark 커널의 상위 eigenpair는 CPU ARPACK, projection은 NumPy BLAS를
사용하며 `CPU_THREADS=8`로 제한한다. cuML 같은 추가 RAPIDS 설치는 필요 없다.

기본 KPC 수는 3이다. landmark kernel trace 대비 선택한 eigenvalue 합의 비율을 출력해
압축 정도를 확인한다. 이는 40D 원본 분산 비율이 아니라 **landmark RKHS variance mass**다.
실제 40D/3D 입력의 Euclidean 거리나 모든 topology 정보가 보존된다는 보장은 없다.

`KP1..KP3`는 kPCA 원좌표다. 거리 계산에는 각 축을 REF projection 표준편차로 나눈
`distance_coordinates = kpca_coordinates / kpc_scale`을 쓴다. 이 axis scale도 저장한다.

## 5. Coverage and uncoverage

REF의 표준화된 KPC 좌표 집합을 `Y_R`, 대표군 B의 좌표를 `Y_B`라고 한다.

\[
d_B(r)=\min_{b\in B}\lVert Y_R(r)-Y_B(b)\rVert_2.
\]

반경은 REF와 사전에 고정한 `RADIUS_BUDGET=1000`으로만 정한다.

\[
k=\min(n_R-1,\max(1,\lceil n_R/1000\rceil)),\quad
R=\operatorname{median}_r d_{\mathrm{REF},k}(r)\times\mathrm{R\_MULT}.
\]

REF k번째 이웃 계산에서는 self 한 개를 제외한다. 같은 위치에 있는 서로 다른 pattern은
삭제하지 않는다. B의 실제 개수가 995개여도 k나 R은 바꾸지 않는다.

- `covered`: `d_B(r) <= R`
- `gap`: `d_B(r) > R`
- 분모: 추출된 REF sample 전부. 임의의 HDR/outlier 제외 없음.
- H0 coarse group을 넘어서 전체 B 중 최근접 pattern을 검색한다. 6번의 같은-H0
  label assignment를 검증하는 지표가 아니라 대표군의 공간 coverage다.

추출된 REF가 B에도 포함되어 생기는 자기 자신 매칭은 표시하고, 전체 coverage와
그 self-hit을 제외한 보조 coverage를 함께 저장한다. 기본 REF 표본을 B에 맞춰
제거/교체하지는 않는다. 향후 8번의 공정한 self-hit 제외 비교는 A/B의 합집합을 양쪽에
공통으로 적용해야 한다.

정규화된 원래 40D 공간의 최근접 거리 mean/median/P95/P99/max도 보조 지표로 저장한다.
40D용 coverage 임계값을 추가로 만들지는 않는다. 최인접 representative는 40D와 KPC에서
다를 수 있으므로 두 이웃 key도 따로 남긴다.

## 6. Outputs

결과는 `topology_analysis_out/run_<UTC timestamp>/`에 실행별로 저장한다. 이전 실행의
좌표계나 성공 결과를 덮어쓰지 않는다. `topology_analysis_out/latest_run.json`은 성공한
실행의 상대 경로와 frame ID를 가리킨다. 실패하면 이전 성공 pointer를 유지한다.

| Output | Contents |
|---|---|
| `analysis_report.html` | 사람이 읽는 요약 수치·거리 통계·H0별 표·그래프; 이미지를 내장한 오프라인 보고서 |
| `analysis.log` | 입력·sampling·fit·block transform·coverage·저장 단계와 Python 예외 |
| `reference_frame.npz` | scaler, block multiplier, landmarks, gamma, kernel centering, eigenpairs, KPC scale, REF rows, 고정 radius, frame hash |
| `reference_sample.npz` | REF 20K key/row/label, 원본 40D와 normalized 40D, KPC 좌표, B 최근접 거리·coverage |
| `topology_sample.npz` | B 실제 대표 key/row/label, 원본·normalized 40D와 같은 frame의 KPC 좌표 |
| `kpca_reference_scatter.csv` | REF sample의 원본 feature, KP 좌표, covered/gap, 최근접 B key/거리, self-hit |
| `kpca_topology_scatter.csv` | B의 원본 feature, KP 좌표, key/row/label |
| `uncovered_kpca_gap_patterns.csv` | REF gap 전량의 원본 feature, KP 좌표, 최근접 B key/거리 |
| `coverage_summary.json` | 표본 수, 실제 B 수, coverage/gap 수와 비율, 거리 통계, H0별 결과, frame·input identity |
| `kpca_coverage_2d.png` | KPC pair별 REF covered/gap과 B의 겹침 |
| `kpca_coverage_3d.png` | 같은 점들의 KP1–3 view |
| `kpca_coverage_distance_cdf.png` | 최근접 B 거리의 누적분포와 고정 반경 |
| `kpca_coverage_2d_heatmap.png` | 위쪽: 칸별 REF 수(로그 색상), 아래쪽: 칸별 미커버 비율; KP1/2, KP1/3, KP2/3 |
| `kpca_nearest_distance_2d_heatmap.png` | 칸별 최근접 B 거리의 평균; coverage 반경을 사용하지 않는 거리 지도 |

도표는 2D/3D 투영이며 coverage label은 설정한 모든 KPC 차원을 사용해 계산한다.
일부 고밀도 gap만 저장하는 것이 아니라 추출된 REF에서 발견한 gap 전부를 저장한다.
원래 수백만 REF의 gap 전량을 검사했다는 뜻은 아니다.

Heatmap은 `HEATMAP_BINS=50`의 50×50 격자로 각 KP pair를 표시한다. 경계는 각 pair의
REF 최솟값/최댓값만으로 정하며, 모든 REF 점과 중복 좌표를 유지한다. 밀도 그림의 색은
확률밀도 추정치가 아니라 해당 칸의 REF 개수다. 같은 종류의 세 패널은 같은 색상 범위를 쓴다.

- 미커버 비율은 `해당 칸의 gap REF 수 / 해당 칸의 전체 REF 수`다.
- 최근접 거리 지도는 해당 칸 REF의 저장된 전체-KPC 최근접 거리 평균이다.
- **회색은 REF 표본이 없는 칸**이다. 이를 coverage 100% 또는 gap 0%로 처리하지 않는다.
- 원래 전체-KPC coverage 판정을 2D 위치별로 집계한다. 2D에서 거리를 다시 계산하거나
  KDE/smoothing으로 비어 있는 영역을 채우지 않는다. 숨겨진 축의 서로 다른 점들이 같은
  칸으로 모일 수 있다.
- REF가 적은 칸에서도 gap 비율 100%가 나올 수 있으므로 위쪽 REF 수와 함께 해석한다.
- 격자 수는 시각화 해상도만 바꾸며, 기존 개별 패턴 coverage·반경·통계는 바꾸지 않는다.

## 7. Stage-8 contract (not implemented yet)

8번은 같은 성공 run에서 다음 순서로 이어간다.

1. `load_reference_frame(path)`로 frame을 읽고 fingerprint를 검증한다.
2. `load_projection_bundle(path, frame)`로 REF와 B snapshot을 읽는다.
3. 외부 대표군 A의 key를 같은 cache row에 연결하고 같은 순서의 learned 40D를 읽는다.
4. `transform_raw_embeddings(raw_A, frame)`으로 **재학습 없이** projection한다.
5. projection을 저장된 `frame['kpc_scale']`로 나눈 뒤 저장된 `frame['radius']`로 평가한다.
6. 고유 key/global row로 REF를 매칭하고 `both_covered`, `A_only`, `B_only`, `both_gap`을
   구분한다. 좌표로 조인하거나 대표 개수에 맞춰 반경을 다시 계산하지 않는다.

현재 7번의 B snapshot이 기존 6번 CSV보다 우선적인 비교 입력이다. 6번을 나중에 재실행해
representatives.csv가 바뀌어도 이전 7번의 REF/B/frame은 같은 run 안에 보존된다.

## 8. Interpretation boundaries

- 이 coverage는 고정 embedding/kPCA 공간의 **기하학적 REF 표본 coverage**다.
- 40D가 이번 방법의 선택 공간이라는 점을 숨기지 않는다. 이것만으로 다른 방식 대비
  독립적인 성능 우위나 Mask/OPC/wafer 개선을 주장하지 않는다.
- 표본 누락과 차원 축소 손실이 있고, 아주 멀리 있는 점의 RBF projection은 수축할 수 있다.
  저장한 원래 40D 거리도 함께 확인한다.
- 20K·landmark 수·gamma·KPC 수·반경은 이번 run의 고정 조건이다. 8번 비교를 위해
  대표군마다 별도 fit 또는 유리한 설정 선택을 하지 않는다.
- Python stdout/stderr는 즉시 flush한다. SIGKILL/전원 차단/네이티브 코드 직접 출력의
  종료 메시지까지 보장하지 않으며, 이미 기록한 로그와 partial 파일만 남을 수 있다.

기술 정의: [scikit-learn KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html),
[SciPy eigsh](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html).
