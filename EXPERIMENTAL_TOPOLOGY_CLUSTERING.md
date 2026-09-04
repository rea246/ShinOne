# Experimental Protocol: Sampled Topology Representatives

## 1. Objective

본 실험의 목적은 기존 handcrafted feature-vector 기반 대표 패턴과 비교하여,
학습된 graph/topology representation이 전체 layout population의 구조적 다양성을
얼마나 잘 대표하는지 평가하는 것이다.

현재 구현은 약 1,000개의 topology community를 만들고 각 community에서 실제
pattern 하나를 대표로 산출한다. community 및 대표 패턴 수는 정확히 1,000개로
강제하지 않고 5% tolerance인 950~1,050개를 목표로 한다.

전체 population의 exact kNN graph는 계산 시간이 지나치게 길기 때문에 사용하지
않는다. Stage-1 H0 group별 stratified sample에서 topology community를 발견하고,
전체 population은 sample community에 kNN으로 일괄 배정한다.

## 2. Candidate population

`4.h0_clustering.py`는 학습된 `3.train_GAE.py` encoder를 clean input에 적용하여
`hkeys_features.pt`를 생성한다. 캐시는 다음 정보를 포함한다.

- `features`: `h0`, `h1`, `h2`, `h3`, `edge` 및 진단용 raw block
- `keys`: cache global row와 대응하는 pattern identifier
- `chunks`: 원본 preprocessed chunk와 global row의 대응 정보

Stage 1은 `h0` embedding이 0이 아닌 pattern만 유효 population으로 사용한다.

\[
\mathcal{P}=\{i \mid \lVert h_{0,i}\rVert_1 > 0\}.
\]

ShinOne과 비교 대상은 동일한 candidate population에서 대표 패턴을 선택해야 한다.

## 3. Learned topology representation

Pattern별 full-topology embedding은 다음과 같다.

\[
z_i=[h_{0,i},h_{1,i},h_{2,i},h_{3,i},e_i].
\]

현재 encoder에서는 각 block이 8차원이므로 총 40차원이다. 구현은 cache tensor에서
실제 dimension을 읽기 때문에 encoder dimension이 변경되어도 동일한 규칙을
적용한다.

- `h0`: POI 중심 node의 learned representation
- `h1`, `h2`, `h3`: 각 hop node embedding의 mean pooling
- `edge`: edge embedding의 mean pooling

## 4. Normalization and distance

각 block은 전체 Stage-1 population을 기준으로 독립 표준화한다.

\[
\hat{z}^{(b)}_{i,j}=\frac{z^{(b)}_{i,j}-\mu^{(b)}_j}{\sigma^{(b)}_j}.
\]

Block dimension에 관계없이 block별 총 기여도가 같도록 다음 scaling을 적용한다.

\[
\tilde{z}^{(b)}_i=\sqrt{\frac{w_b}{d_b}}\hat{z}^{(b)}_i,
\qquad w_b=1.
\]

Topology distance는 concatenated normalized embedding의 Euclidean distance다.

## 5. Stage 1: H0 coarse stratification

Stage 1은 `h0` embedding으로 dense H0 group과 HDBSCAN noise `-1`을 구분한다.
Noise pattern은 삭제하지 않고 `rare` coarse group으로 유지한다. 전량 label은
`h0_clustering_out/h0_labels_full.npz`에 저장된다.

Integrated topology pipeline은 서로 다른 H0 coarse group을 합치지 않는다.
Sampling, topology clustering 및 full-population assignment는 각 H0 group 내부에서
독립적으로 수행한다.

## 6. Integrated sampled topology pipeline

`6.topology_clustering.py` 한 번의 실행이 sample clustering, full-population
assignment, community summary 및 representative extraction을 모두 수행한다.

### 6.1 Deterministic stratified sampling

전체 sample budget은 100,000개다. 각 H0 coarse group에서 최소 한 개를 포함한 뒤,
남은 sample quota를 group population에 비례하여 deterministic largest-remainder
방식으로 배분한다. Group에 배정된 quota가 group size와 같으면 해당 group은 전량
사용하고, 그렇지 않으면 seed 42의 uniform sampling을 사용한다.

별도의 rare oversampling, novelty 재탐색 또는 iterative sample 보강은 적용하지
않는다.

### 6.2 Sample exact kNN graph

각 H0 group의 sampled normalized embedding에서 self-edge를 제외한 exact kNN graph를
만든다.

\[
G_g^{S}=(V_g^{S},E_g^{S}),\qquad
(i,j)\in E_g^{S}\ \text{if}\ j\in\operatorname{kNN}_{20}(i).
\]

기본 graph는 undirected union-kNN이다. Edge weight는 sample graph의 local scale을
사용하는 Gaussian similarity다.

\[
w_{ij}=\exp\left(-\frac{d(i,j)^2}{s_i s_j}\right).
\]

Exact search backend 우선순위는 FAISS GPU, PyTorch CUDA, FAISS CPU, scikit-learn
multi-CPU 순서다.

### 6.3 Resolution selection with 5% tolerance

전체 graph를 100,000개 sample graph로 축소하면 같은 `k`와 resolution에서도 작은
community 경계가 합쳐질 수 있다. 따라서 먼저 `전체 pattern 수 / 실제 sample 수`를
resolution scale로 사용한다. 약 2,063,000개를 100,000개로 줄인 경우 초기 scale은
약 20.63이다.

아래 base Leiden resolution 후보에 이 scale을 곱하고, 비교 기준을 위해 resolution
1.0도 포함하여 동일한 sample graph에서 평가한다.

```text
0.25, 0.35, 0.50, 0.70, 0.85, 1.00, 1.20, 1.50, 2.00
```

모든 H0 group의 community 수를 resolution별로 합산한다. 전체 community 수가
950~1,050 범위인 후보 중 target 1,000에 가장 가까운 resolution을 선택한다.
동률이면 더 높은 resolution을 선택한다. 초기 후보가 범위를 만들지 못하면 resolution을
2배씩 자동 확장한다. target의 위·아래 community count가 확보되면 log-resolution
공간에서 보간하여 최대 12회 세부 탐색한다. 그래도 범위에 들어오지 못할 때만 평가한
전체 후보와 가장 가까운 결과를 출력한 뒤, 잘못된 개수의 representative 파일을 쓰지
않고 명시적으로 중단한다.

Resolution 선택에는 비교 대상인 REF 결과를 사용하지 않는다.

기본 Leiden backend는 A100의 RAPIDS cuGraph다. Symmetric weighted CSR을 그대로
GPU graph로 전달하고, 각 resolution에서 최대 100 iterations와 seed 42를 사용한다.

### 6.4 Full-population assignment

선택된 sample partition을 class label로 사용한다. 각 full-population pattern은 동일한
H0 coarse group의 sampled pattern만 reference로 하는 exact 5-NN uniform vote로
배정한다. Vote tie는 가장 작은 sample community ID로 결정한다.

Sample vertex는 자신이 발견한 Leiden community label을 유지한다. 따라서 sample에서
발견된 community가 full assignment 과정에서 사라지지 않는다. 배정 후 community ID는
full-population size 내림차순으로 다시 번호를 붙인다.

이 단계는 Stage-1의 sample clustering 후 KNN full assignment와 동일한 engineering
구조다.

### 6.5 One representative per community

Full assignment가 끝난 뒤 전체 구성원으로 각 community centroid를 다시 계산한다.
Community `c`의 representative는 centroid와 가장 가까운 실제 member다.

\[
r_c=\arg\min_{i\in c}\lVert\tilde z_i-\mu_c\rVert_2.
\]

이는 합성 centroid나 embedding vector가 아니라 `hkeys_features.pt`에 존재하는 실제
`pattern_key`다. Final community 수와 representative 수는 항상 동일하다. Community
수가 tolerance 안에서 987개면 대표 패턴도 987개이고, 1,032개면 대표 패턴도
1,032개다.

Final cluster ID는 다음 규칙을 유지한다.

```text
dense: H0_<h0_label>_T<topology_label>
rare : RARE_T<topology_label>
```

## 7. Baseline configuration

| Parameter | Value | Purpose |
|---|---:|---|
| `SAMPLE_BUDGET` | 100,000 | sampled topology discovery population |
| `K_NEIGHBORS` | 20 | sample graph local degree |
| `ASSIGN_NEIGHBORS` | 5 | full-population label vote |
| `COMMUNITY_TARGET` | 1,000 | approximate representative count |
| `COMMUNITY_TOLERANCE` | 0.05 | accepted range 950~1,050 |
| Base resolution candidates | 0.25~2.0 | multiplied by full/sample ratio |
| Adaptive resolution rounds | 12 | expansion and interpolated refinement |
| `LEIDEN_BACKEND` | `cugraph` | A100 weighted Leiden |
| `CUGRAPH_MAX_ITERATIONS` | 100 | bounded Leiden iterations |
| `EDGE_WEIGHT_MODE` | `local_gaussian` | distance-to-similarity mapping |
| `MUTUAL_KNN` | `False` | union-kNN baseline |
| `RANDOM_SEED` | 42 | deterministic sampling and voting |
| Block weights | all 1.0 | equal block contribution |
| Normalization block | 250,000 | bounded CPU working memory |
| Query block | 10,000 | bounded query-result memory |

## 8. Computational scope

기존 full graph의 rough distance candidate 수는 \(N^2\)이다. Sample size를 `S`라
하면 integrated approximation의 주요 distance candidate 수는 다음과 같다.

\[
S^2+NS.
\]

`N=2,062,997`, `S=100,000`이면 full graph의 약 4.26조 후보 대신 약 0.216조
후보를 평가하므로 rough candidate count가 약 20배 감소한다. 실제 runtime은 GPU
backend, H0 group 분포, I/O 및 Leiden 실행 시간에 따라 달라진다.

Full DB는 normalization, 5-NN assignment, centroid 및 representative 계산을 위해
선형 scan한다. Full-population 간 \(N^2\) graph는 생성하지 않는다.

## 9. Approximation boundary

이 구현은 현실적인 runtime을 위해 다음 차이를 명시적으로 허용한다.

- Sample에 포함되지 않은 topology는 독립 Leiden community를 직접 만들 수 없다.
- Full-population label은 full graph partition이 아니라 sampled partition의 5-NN
  extension이다.
- Sample seed, sample budget 및 resolution 후보가 결과에 영향을 준다.
- GPU와 CPU Leiden은 같은 objective를 사용해도 bit-identical하지 않을 수 있다.

이 손실은 accepted engineering approximation이다. 현재 baseline에는 별도의 rare
oversampling, novelty detection, uncovered-pattern promotion 또는 iterative refinement를
추가하지 않는다.

## 10. Outputs

모든 결과는 `topology_clustering_out/`에 저장된다.

### 10.1 Full assignments

`topology_assignments.csv`

```text
pattern_key
original_h0_label
coarse_group
topology_cluster
final_cluster
```

### 10.2 Community summary

`topology_cluster_summary.csv`

```text
h0_label
coarse_group
topology_cluster
final_cluster
pattern_count
fraction_in_coarse_group
fraction_in_total
centroid_distance_mean
centroid_distance_p95
centroid_distance_max
```

Distance 값은 full-population normalized topology space에서 계산한다.

### 10.3 Representatives

`topology_representatives.csv`

```text
pattern_key
global_row
h0_label
topology_cluster
final_cluster
representative_rank
community_pattern_count
selection_reason
distance_to_centroid
```

`topology_representatives.npz`에는 representative의 global row와 label을 저장한다.

### 10.4 Machine-readable labels and diagnostics

`topology_labels.npz`는 다음을 저장한다.

- full-population `rows`, `h0_labels`, `topology_labels`
- sampled cache rows인 `sample_rows`
- 선택된 `selected_resolution`
- ordered global-row/pattern-key population fingerprint

`nearest_neighbor_diagnostics.csv`는 full-population anchor와 sampled reference
neighbor의 distance 및 동일 final community 여부를 기록한다.

`topology_run_metadata.json`은 sample size, resolution별 community 수, 선택된
resolution, 실제 representative 수, backend 및 전체 parameter를 기록한다.

## 11. Comparison with REF representatives

ShinOne 결과 수는 950~1,050개를 허용한다. Coverage distance는 실제 representative
수의 영향을 받으므로 비교 보고서에는 각 방법의 representative 수를 반드시 함께
표시한다.

Representative set `R`에 대해 population pattern `i`의 coverage distance는 다음과
같다.

\[
d_i(R)=\min_{r\in R}\lVert\tilde z_i-\tilde z_r\rVert_2.
\]

주요 지표는 다음과 같다.

- 전체 population의 mean/P95/P99 coverage distance
- rare population의 mean/P95/P99 coverage distance
- H0/final community별 coverage
- 두 representative set의 overlap과 Jaccard index

## 12. Execution

Python 3.12 / CUDA 12 환경은 repository installer를 사용한다.

```bash
# 사내 pip index
./install-py312-cu12.sh

# offline wheel directory
./install-py312-cu12.sh /path/to/wheels
```

실행은 한 번이다.

```bash
python 6.topology_clustering.py
```

`7.select_representatives.py`는 과거 exact-budget Stage-2 결과를 재처리하기 위한
legacy utility이며 integrated baseline 실행에는 사용하지 않는다.

Default Leiden backend에는 CUDA와 호환되는 RAPIDS `cugraph`, `cudf`, `cupy`가
필요하다. FAISS GPU가 있으면 exact sample graph 및 assignment에 우선 사용하고,
없으면 PyTorch CUDA exact backend가 A100을 사용한다.

## 13. Interpretation limit

현재 구현은 다음 결과를 생성한다.

- sampled topology community discovery
- full-population community assignment
- 약 1,000개의 actual centroid-nearest representative

아직 full-population exact graph와의 assignment stability, sample seed stability 및
REF representative와의 최종 coverage 결과를 주장하지 않는다.
