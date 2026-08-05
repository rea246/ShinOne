# Experimental strategy — Reference-centered kPCA group weighting

## 핵심 주장

이 연구는 kPCA coverage가 물리적 성능을 직접 나타낸다고 가정하지 않는다. kPCA는
Reference와 Sample1 그룹 사이의 **기하학적 연관도**를 정의하는 표현공간으로만 사용한다.
이 연관도로 만든 학습 가중치의 유효성은 독립적인 Mask contour 및 OPC simulation으로
검증한다.

---

## 1. Reference 중심 비선형 표현공간

Reference와 Sample1의 각 pattern을 동일한 21차원 feature vector로 변환한다. 결측값이나
비유한 값을 포함한 pattern은 제외한다.

Feature scaling과 RBF Kernel PCA는 Reference만으로 학습한다. Reference feature
$\mathbf{x}$는 Reference 평균 $\boldsymbol{\mu}_R$와 표준편차
$\boldsymbol{\sigma}_R$로 표준화한다.

$$
\widetilde{\mathbf{x}}
=
\frac{\mathbf{x}-\boldsymbol{\mu}_R}{\boldsymbol{\sigma}_R}
$$

RBF kernel은 다음과 같이 정의한다.

$$
K(\mathbf{x}_i,\mathbf{x}_j)
=
\exp\left(-\gamma\lVert\widetilde{\mathbf{x}}_i-
\widetilde{\mathbf{x}}_j\rVert_2^2\right)
$$

$\gamma$는 Reference pairwise squared distance의 median heuristic으로 정한다.

$$
\gamma=
\frac{1}{2\,\operatorname{median}_{i<j}
\lVert\widetilde{\mathbf{x}}_i-\widetilde{\mathbf{x}}_j\rVert_2^2}
$$

학습된 Reference frame을 사용하여 Reference와 Sample1을 동일한 KPC1–3 공간에 투영한다.
Sample1은 scaling이나 kPCA를 다시 fit하지 않고 transform만 수행한다. 이 공통 frame 덕분에
그룹 간 거리 비교가 Sample1 구성에 따라 바뀌지 않는다.

- 구현: `coverage_domain_kpca.py`
- 현재 기본값: Reference reservoir 15,000개, kPCA landmark 최대 5,000개, random seed 0
- 산출물: 고정 kPCA cache, Reference/Sample1의 KPC1–3 좌표
- 해석 제한: 이 단계의 거리와 coverage는 물리적 중요도나 모델 성능을 의미하지 않는다.

---

## 2. 표현공간 구성 확인

KPC1–3 공간을 downstream 점수에 사용하기 전에 다음을 확인한다.

1. KPC1–3이 RKHS 전체 분산에서 차지하는 질량
2. 각 KPC를 따라 변화하는 원본 feature의 조건부 평균
3. KPC와 원본 feature 사이의 사후 상관관계(pseudo-loading)
4. 특정 feature 하나 또는 noise가 축을 지배하는지 여부

kPCA 축은 원본 feature의 선형결합이 아니므로 PCA loading처럼 해석하지 않는다.
Pseudo-loading은 축의 정의가 아니라 사후 요약값으로만 사용한다.

- 구현: `kpc_axis_explain.py`
- 판정: KPC1–3의 분산질량이 충분하지 않으면 3차원 결과를 확정하지 않는다.
- 추가 확인: KPC 차원 수, $\gamma$, landmark seed를 바꿨을 때 그룹 순위의 안정성을 비교한다.

이 단계는 physics 검증이 아니라 **representation QA**다.

---

## 3. Sample1 그룹의 Reference 연관도

Sample1은 A–F 그룹으로 구성된다. 각 Reference target pattern에 대해 KPC 공간에서 가장
가까운 Sample1 pattern top-K를 찾고, 역거리 비율로 총 1점의 vote를 분배한다.

$$
v_{ij}
=
\frac{(d_{ij}+\epsilon)^{-1}}
{\sum_{l\in\operatorname{NN}_K(i)}(d_{il}+\epsilon)^{-1}}
$$

Reference 전체에 대한 pattern vote를 그룹별로 합산하여 그룹 점수 $S_g$를 계산한다.

$$
S_g=\sum_i\sum_{j\in g}v_{ij}
$$

- 구현: `group_reinforce_nnvote.py`
- 기본 해석: $S_g$는 그룹 $g$의 pattern이 Reference 가까이에 얼마나 자주 위치하는지를
  나타내는 **기하학적 연관도**다.
- 금지된 해석: downstream 검증 전에는 이를 물리적 중요도 또는 모델 성능 기여도로 부르지 않는다.

그룹 크기가 크면 최근접 이웃으로 선택될 기회도 증가한다. 따라서 학습 가중치에는 총점
$S_g$를 그대로 사용하지 않고 그룹 pattern 수 $n_g$로 나눈 값을 사용한다.

$$
q_g=\frac{S_g}{n_g}
$$

Reference scatter가 없을 때 Sample1과 target으로 거리 scale을 다시 추정하면 공통 frame
전제가 약해지므로, 최종 실험에서는 Reference 기반 scale을 필수로 사용한다.

---

## 4. Mask Process Model의 그룹 가중 학습

그룹 연관도 $q_g$를 Mask Process Model의 pattern loss weight로 변환한다. 극단적인
가중치를 줄이기 위해 완화 지수 $\alpha$를 적용할 수 있다.

$$
w_g\propto q_g^{\alpha},\qquad 0\leq\alpha\leq1
$$

전체 학습 loss scale이 POR와 같도록 가중치를 정규화한다.

$$
\sum_g n_g w_g=N
$$

필요하면 사전에 정한 범위로 $w_g$를 clipping한다. 가중치 효과와 데이터 제거 효과를
분리하기 위해 첫 실험에서는 pattern을 제거하지 않는다.

최소 대조실험은 다음과 같다.

| Model | Training patterns | Weight | 목적 |
|---|---|---|---|
| POR | 전체 Sample1 | Uniform | 기준 |
| Weighted | POR와 동일 | kPCA group weight | 제안 방법 |
| Shuffled-weight | POR와 동일 | 그룹 weight 무작위 치환 | 기하학 정보의 효과 통제 |

세 모델은 학습 pattern 수, optimizer step, seed와 training budget을 동일하게 유지한다.
Weighted model이 POR와 shuffled-weight보다 개선된 뒤에만 pruning을 후속 실험한다.
Pruning 시에는 low-score 제거와 동일 개수 random 제거를 비교하며, 희귀 그룹을 전부
삭제하지 않도록 그룹별 최소 pattern 수를 유지한다.

---

## 5. Mask contour 검증

가중치 계산 및 모델 학습에 사용하지 않은 Reference holdout pattern을 동일 조건으로 mask
simulation한다. POR와 Weighted model을 pattern 단위 paired comparison으로 평가한다.

- mean/median 및 p95·p99 contour error 또는 EPE
- worst-pattern error와 hotspot 수
- 개선 pattern 수와 악화 pattern 수
- feature/KPC 영역별 개선·악화 분포

평균 오차만 감소하고 tail error가 증가하면 개선으로 판정하지 않는다. 가능하면 Reference를
가중치 산출용과 평가용으로 분리한다.

---

## 6. OPC simulation 검증

Mask contour에서 유의미한 개선이 확인된 모델만 동일한 OPC recipe와 process condition으로
평가한다. POR 대비 wafer EPE, PV band, process-window margin, hotspot과 worst-case pattern을
비교한다. Mask에서 개선된 pattern이 wafer에서도 개선되는지를 추적하여 기하학적 그룹
가중치가 실제 wafer 성능으로 이어지는지 최종 검증한다.

---

## 해석 원칙

1. kPCA는 공통 비선형 표현공간을 제공한다.
2. 최근접 투표 점수는 Reference와의 기하학적 연관도다.
3. Mask/OPC 결과로 검증되기 전에는 해당 점수를 physics 중요도로 해석하지 않는다.
4. 가중치와 pruning은 별도 실험으로 분리한다.
5. 모든 성능 주장은 독립 holdout과 POR·placebo 대조를 기준으로 한다.
