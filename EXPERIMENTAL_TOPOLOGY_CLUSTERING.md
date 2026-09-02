# Experimental Protocol: Hierarchical Topology Clustering

## 1. Objective and hypothesis

본 실험의 목적은 기존 handcrafted feature-vector 기반 대표 패턴 집합과 비교하여, 학습된 graph/topology representation이 전체 layout pattern population의 구조적 다양성, 특히 저밀도 희소 구조를 더 균일하게 보존할 수 있는지 검증하는 것이다.

핵심 가설은 다음과 같다.

> H0 HDBSCAN is used as a rare-aware coarse structural stratification, while full hierarchical GNN representations including h0, h1, h2, h3, and edge information are subsequently used to construct local similarity graphs and identify fine-grained topology communities without discarding low-density structural patterns.

이번 구현 범위는 최종 topology community 산출까지이다. 고정 예산 1,000개의 대표 패턴 선택과 상용툴 결과 비교는 후속 실험 단계로 분리한다.

## 2. Dataset and candidate population

### 2.1 Input patterns

각 layout pattern은 전처리 단계에서 local graph로 변환된다. 노드는 polygon geometry를, edge는 노드 간 공간 관계를 표현하며, 각 노드는 POI 중심으로부터의 hop membership(`h0`–`h3`)을 갖는다.

`4.h0_clustering.py`는 학습된 `3.train_GAE.py` encoder를 clean input에 한 번 적용하여 `hkeys_features.pt`를 생성한다. 캐시는 다음 항목을 포함한다.

- `features`: `h0`, `h1`, `h2`, `h3`, `edge` 및 진단용 raw block
- `keys`: cache global row와 대응하는 pattern identifier
- `chunks`: 원본 preprocessed chunk와 global row의 대응 정보

### 2.2 Inclusion rule

Stage 1은 `h0` embedding이 0이 아닌 pattern만 유효 population으로 사용한다. 따라서 비교 실험의 candidate population은 다음과 같이 고정한다.

\[
\mathcal{P}=\{i \mid \lVert h_{0,i}\rVert_1 > 0\}.
\]

ShinOne과 상용툴의 1,000개 결과는 반드시 동일한 \(\mathcal{P}\)에서 선택되어야 한다. 제외된 pattern 수와 제외 사유는 실험 로그에 기록한다.

## 3. Learned topology representation

GNN encoder는 pattern별로 다섯 개의 hierarchy block을 산출한다.

\[
z_i=[h_{0,i},h_{1,i},h_{2,i},h_{3,i},e_i].
\]

현재 encoder의 각 block은 8차원이므로 전체 embedding은 40차원이다. 구현에서는 향후 encoder 변경에 대비하여 cache tensor에서 실제 block dimension을 읽고 검증한다.

- `h0`: POI 중심 노드의 learned representation
- `h1`, `h2`, `h3`: 각 hop node embedding의 mean pooling
- `edge`: edge embedding의 mean pooling

`h0`는 순수 raw center geometry가 아니다. message passing의 결과이므로 주변 node와 edge relation의 정보도 포함한다.

## 4. Block normalization and topology distance

특정 hierarchy block의 variance 또는 dimension이 전체 거리를 지배하지 않도록 각 block을 전체 Stage-1 population 기준으로 독립 표준화한다.

\[
\hat{z}^{(b)}_{i,j}=\frac{z^{(b)}_{i,j}-\mu^{(b)}_j}{\sigma^{(b)}_j}.
\]

Block dimension이 달라질 경우에도 block별 총 기여도를 일정하게 유지하기 위해 다음 scaling을 적용한다.

\[
\tilde{z}^{(b)}_i=\sqrt{\frac{w_b}{d_b}}\hat{z}^{(b)}_i,
\]

여기서 \(d_b\)는 block dimension이고 초기 baseline weight는 모든 block에 대해 \(w_b=1\)이다. 최종 topology distance는 concatenated normalized embedding의 Euclidean distance이다.

\[
d(i,j)=\left\lVert
[\tilde{h}_{0,i},\tilde{h}_{1,i},\tilde{h}_{2,i},\tilde{h}_{3,i},\tilde{e}_i]
-
[\tilde{h}_{0,j},\tilde{h}_{1,j},\tilde{h}_{2,j},\tilde{h}_{3,j},\tilde{e}_j]
\right\rVert_2.
\]

임의의 hop-weight 최적화는 baseline 실험에 포함하지 않는다.

## 5. Stage 1: rare-aware H0 stratification

Stage 1은 `h0` embedding에 HDBSCAN을 적용하여 dominant structural mode와 low-density pattern을 coarse하게 분리한다.

1. 유효 population에서 최대 300,000개를 seed-fixed sampling한다.
2. `h0`를 표준화하고 PCA-reduced discovery space에서 HDBSCAN을 수행한다.
3. persistence 또는 count 기준을 만족하지 못한 약한 cluster는 noise로 편입한다.
4. HDBSCAN noise label `-1`은 삭제하지 않고 rare/noise group으로 보존한다.
5. sample의 dense/rare label을 class로 사용하는 KNN으로 전체 유효 population을 배정한다.

Stage 1의 전량 출력은 `h0_clustering_out/h0_labels_full.npz`이다.

- `rows`: `hkeys_features.pt`의 global row
- `labels`: dense H0 cluster ID 또는 rare/noise `-1`

이 단계의 rare group은 clustering failure가 아니라 low-density 또는 atypical structural pattern pool로 정의한다.

## 6. Stage 2: within-group topology community detection

Stage 2는 Stage-1 coarse group을 합쳐서 다시 global clustering하지 않는다. 각 dense H0 group과 rare group 내부에서 독립적으로 다음 절차를 수행한다.

### 6.1 k-nearest-neighbor graph

각 coarse group \(g\)의 normalized full embedding으로 self-edge를 제외한 kNN graph를 구성한다.

\[
G_g=(V_g,E_g),\qquad
(i,j)\in E_g \;\text{if}\; j\in\operatorname{kNN}_k(i).
\]

기본값은 일반 undirected union-kNN이고, sensitivity analysis를 위해 mutual-kNN을 선택할 수 있다.

### 6.2 Edge similarity

기본 edge weight는 local-scale Gaussian similarity이다. 각 vertex \(i\)의 local scale \(s_i\)는 k번째 이웃까지의 거리로 정의한다.

\[
w_{ij}=\exp\left(-\frac{d(i,j)^2}{s_i s_j}\right).
\]

따라서 embedding distance가 작을수록 edge weight가 커진다. 비교 실험을 위해 inverse-distance와 binary graph도 선택 가능하다.

### 6.3 Leiden partition

각 sparse kNN graph에 weighted Leiden community detection을 적용한다. 구현은 resolution parameter를 갖는 `RBConfigurationVertexPartition`을 사용하며, 수렴할 때까지 iteration하고 random seed를 고정한다.

동일한 Stage-1 group 내부의 Leiden community는 크기 내림차순으로 재번호하여 label permutation을 제거한다. 최종 cluster ID는 다음 규칙을 따른다.

```text
dense: H0_<h0_label>_T<topology_label>
rare : RARE_T<topology_label>
```

## 7. Baseline parameters

| Parameter | Baseline | Purpose |
|---|---:|---|
| `K_NEIGHBORS` | 20 | local topology graph degree |
| `LEIDEN_RESOLUTION` | 1.0 | community granularity |
| `EDGE_WEIGHT_MODE` | `local_gaussian` | distance-to-similarity mapping |
| `MUTUAL_KNN` | `False` | union-kNN baseline |
| `RANDOM_SEED` | 42 | reproducibility |
| Block weights | all 1.0 | no learned/manual hierarchy weighting |
| Normalization chunk | 250,000 | bounded scaler working memory |
| kNN query chunk | 50,000 | bounded query-result memory |

Baseline parameter는 최종 성능 최적값이 아니라 sensitivity analysis의 기준점이다. 상용툴 선택 결과를 관찰한 뒤 clustering parameter를 조정하지 않는다.

## 8. Sensitivity and stability analysis

최종 representative selection 전에 최소한 다음 조건을 비교한다.

- \(k\in\{10,20,30\}\)
- 여러 Leiden resolution 값
- union-kNN과 mutual-kNN
- local Gaussian과 binary edge
- 여러 random seed

각 조건에서 다음 항목을 기록한다.

- final community 수
- community size의 min/median/mean/max
- singleton 및 small-community 비율
- rare-group community 수
- graph connected component 및 isolated vertex 수
- Leiden quality 또는 modularity
- seed 간 assignment stability

Parameter는 topology coverage와 partition stability를 기준으로 고정한다. 상용툴 결과와의 유사도는 parameter selection criterion으로 사용하지 않는다.

## 9. Outputs and diagnostics

`6.topology_clustering.py`는 다음 파일을 `topology_clustering_out/`에 저장한다.

### 9.1 Pattern assignments

`topology_assignments.csv`

```text
pattern_key
original_h0_label
coarse_group
topology_cluster
final_cluster
```

### 9.2 Community summary

`topology_cluster_summary.csv`

```text
h0_label
coarse_group
topology_cluster
final_cluster
pattern_count
fraction_in_coarse_group
fraction_in_total
```

### 9.3 Nearest-neighbor inspection

`nearest_neighbor_diagnostics.csv`는 coarse group별 random anchor와 nearest neighbor를 기록한다.

```text
coarse_group
anchor_key
neighbor_rank
neighbor_key
distance
same_final_cluster
```

이 진단은 embedding-space neighbor가 실제 layout topology에서도 유사한지 시각적으로 확인하는 데 사용한다.

### 9.4 Machine-readable labels

`topology_labels.npz`는 후속 representative selection이 사용할 `rows`, `h0_labels`, `topology_labels`를 보존한다.

## 10. Representative selection under a fixed budget

Clustering은 population을 partition하지만 1,000개 pattern을 직접 선택하지 않는다. 고정 예산 representative selection은 별도 단계로 수행한다.

1. 모든 final community에 minimum quota를 배정한다.
2. rare community도 dense community와 동일하게 minimum quota 대상에 포함한다.
3. 남은 budget을 community size 및 uncovered topology volume을 고려하여 배분한다.
4. 각 community 내부에서는 medoid만 반복 선택하지 않고 coverage-based 또는 farthest-point selection을 적용한다.
5. 선택 결과가 정확히 1,000개의 unique pattern key인지 검증한다.

Final community 수가 1,000보다 많으면 community당 최소 1개 조건과 budget이 양립하지 않는다. 이 경우 selection 전에 resolution을 재검토하거나, 사전에 정의한 small-community consolidation rule을 적용해야 한다.

예상 selection manifest는 다음과 같다.

```text
pattern_key
global_row
h0_label
topology_cluster
final_cluster
selection_rank
selection_reason
```

## 11. Comparison with a commercial-tool selection

ShinOne과 상용툴은 동일한 candidate population \(\mathcal{P}\)에서 각각 정확히 1,000개를 선택한다.

\[
S_{\mathrm{ours}}\subset\mathcal{P},\quad
S_{\mathrm{commercial}}\subset\mathcal{P},\quad
|S_{\mathrm{ours}}|=|S_{\mathrm{commercial}}|=1000.
\]

주 평가 지표는 normalized full-topology space에서의 nearest-representative distance이다. Representative set \(S\)에 대해 pattern \(i\)의 coverage distance를 다음과 같이 정의한다.

\[
d_i(S)=\min_{s\in S}\lVert\tilde{z}_i-\tilde{z}_s\rVert_2.
\]

각 방법에 대해 다음 값을 보고한다.

- mean \(d_i(S)\)
- P95 \(d_i(S)\)
- P99 \(d_i(S)\)
- rare population에서의 mean/P95/P99
- coarse/final community별 selected count와 uncovered count

P95와 P99는 low-density structural direction의 coverage를 평가하는 핵심 지표이다. 두 1,000개 집합의 overlap 또는 Jaccard index는 보조 지표로만 사용하며, overlap 자체를 coverage quality로 해석하지 않는다. 필요하면 pattern bootstrap으로 각 coverage statistic의 confidence interval을 산출한다.

## 12. Reproducibility and provenance

최종 실험에서는 다음 metadata를 결과와 함께 저장한다.

- Git commit hash
- feature-cache key hash 또는 cache identifier
- candidate population size와 excluded pattern 수
- 실제 block dimensions
- 모든 Stage-1/Stage-2 parameter
- random seed
- dependency versions
- CPU/GPU 정보
- 단계별 실행 시간

`hkeys_features.pt`가 Stage-1 label 생성 후 변경되지 않았음을 cache fingerprint로 검증해야 한다. 단순히 row 수가 동일한지만 확인하는 것은 key-order mismatch를 탐지하기에 충분하지 않다.

## 13. Computational implementation

### 13.1 Current acceleration

- Feature extraction in `4.h0_clustering.py`: CUDA가 있으면 GPU 사용
- HDBSCAN and Stage-1 KNN assignment: `n_jobs=-1` 또는 `core_dist_n_jobs=-1`로 multi-CPU 사용
- Stage-2 exact kNN: `auto`에서 FAISS GPU → PyTorch CUDA → FAISS CPU(OpenMP) → scikit-learn multi-CPU 순으로 사용
- Scaling과 kNN query는 block processing으로 peak working memory를 제한
- Sparse adjacency는 \(O(Nk)\) storage 사용

### 13.2 Current limitations

Stage-2는 GPU에서도 full-population exact search를 유지하므로 계산 복잡도 자체는 \(O(N^2)\)이다. FAISS GPU가 없더라도 기존 PyTorch CUDA를 이용한 blocked matrix search가 A100 등 CUDA device를 사용한다. Leiden은 native CPU implementation을 사용하며 coarse group은 memory peak를 제한하기 위해 순차 처리한다.

### 13.3 Planned acceleration path

과학적 정의를 유지하는 첫 번째 가속인 FAISS `GpuIndexFlatL2` exact Euclidean kNN과 PyTorch CUDA exact fallback을 구현했다. 추가 가속을 위해 approximate IVF/HNSW를 사용할 경우에는 exact subset 대비 recall@k를 보고하고 graph 및 community 결과에 미치는 영향을 별도로 검증한다.

## 14. Execution

```bash
python 4.h0_clustering.py
python 6.topology_clustering.py
```

Stage 2의 필수 추가 dependency는 `python-igraph`와 `leidenalg`이다. 기존 pipeline dependency인 NumPy, SciPy, scikit-learn, PyTorch도 필요하다. CUDA 지원 FAISS가 설치되어 있으면 가장 먼저 사용하며, 없으면 PyTorch CUDA exact backend가 GPU를 사용한다.

```bash
pip install python-igraph leidenalg
```

## 15. Scope and limitations

현재 구현은 topology community 생성까지 검증한다. 다음 항목은 아직 결과로 주장하지 않는다.

- fixed-budget 1,000-pattern representative selection
- 상용툴 1,000개와의 coverage comparison
- 자동 parameter sweep 및 최적값
- approximate kNN의 recall/community stability 검증
- full production dataset에서의 runtime과 peak memory

따라서 본 문서는 완결된 성능 결과가 아니라, 결과 생성 전에 고정해야 할 experimental protocol과 구현 경계를 정의한다.
