---
title: 토폴로지 임베딩 기반 2단계 계층적 클러스터링 파이프라인 설계
date: 2026-07-14
tags:
  - Pattern_Classification
  - HDBSCAN
  - Louvain
  - Layout_GNN
  - Obsidian_Workflow
---
# 2단계 계층적 클러스터링 파이프라인 (HDBSCAN + Louvain k-NN)

[[2.Pattern_Classification/4.DEVELOP/기획안_토폴로지그래프_패턴다양성|기획안_토폴로지그래프_패턴다양성]]

본 파이프라인은 **"전역적/물리적 대표 구조 기반의 1차 분류(Coarse-grained)"**를 수행한 후, **"로컬 위상 문맥 기반의 2차 정밀 분류(Fine-grained)"**를 연쇄적으로 적용하여 반도체 레이아웃의 구조적 다양성을 극대화하고 대표 패턴의 신뢰성을 확보합니다.

---

## 1. 전체 Clustering 진행 Flow


graph TD
    A[입력 데이터: 40차원 임베딩 벡터] -->|h0: POI 중심점 8차원| B(Step 1: h0 기준 1차 HDBSCAN 분할)
    A -->|h1~h3, edge: 주변부 32차원| C
    B -->|Sub-cluster 분할 및 노이즈 필터링| C[Step 2: 세부 위상 기반 k-NN 그래프 구성]
    C --> D[Step 3: 각 Sub-cluster 내 Louvain 커뮤니티 탐지]
    D --> E[Step 4: 그룹 크기 비례 + 다양성 보존 대표 패턴 추출]

    style B fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style C fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style D fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    style E fill:#fff3e0,stroke:#e65100,stroke-width:2px


### 1.1 임베딩 벡터의 실제 산출 구조 (`3.train_GAE.py` 인코더 기준)

위 flow 의 "40차원 임베딩 벡터"는 학습 시 손실 계산에 쓰이는 `graph_emb` 와는 **다른 벡터**다. (이름만 둘 다 40d 라 혼동하기 쉬우므로 산출 경로를 명확히 한다.) 클러스터링 입력은 `claasify_tree.py` 가 학습된 인코더로 clean 입력을 1-pass 추론해 만든 **hop 분해 블록**이다.

- **인코더 출력**: `3.train_GAE.py` 의 `LayoutGATEncoder`(`EMBEDDING_DIM = 8`, `train_GAE.py:81, 350-360`)는 그래프마다 노드 임베딩 `z_node`(노드당 8d)와 엣지 임베딩 `z_edge`(엣지당 8d)를 낸다.

- **클러스터링용 40차원 = hop 분해 5블록 × 8d** (`claasify_tree.py:extract_block_features`): `z_node`/`z_edge` 를 hop mask 별 mean-pool 한다.
    - `h0`(8d) = `z_node` 를 `hop0_mask`(POI 중심 노드)로 mean-pool → **Step 1** 입력
    - `h1`, `h2`, `h3`(각 8d) = `z_node` 를 각 hop mask 로 mean-pool
    - `edge`(8d) = `z_edge` mean-pool
    - → `h1 + h2 + h3 + edge = 32d` = **Step 2** 입력
    - 즉 flow 의 "h0 8차원 / h1~h3·edge 32차원" 분해는 이 hop-pool 블록 구조와 정확히 대응한다.

- **학습 시 `graph_emb` 와의 구분 (중요)**: `train_GAE.py:413` 의
  `graph_emb = concat[max_emb, mean_emb, center_emb, emean, emax]` 도 8d × 5 = 40d 지만,
  이는 hop 분해가 아니라 **pooling 종류별**(노드 max/mean, 중심 center, 엣지 mean/max) 블록이며
  **VICReg·재구성 손실 전용**이다. 클러스터링은 이 벡터를 사용하지 않는다.
    - 특히 클러스터링의 `h0` 는 순수 `hop0` mean-pool 이라, `graph_emb` 의 `center_emb`
      (= `center_proj` MLP 통과 + 중심 노드가 없을 때 `mean_emb` 로 fallback, `train_GAE.py:405-407`)
      와는 값이 다르다. 둘 다 "POI 중심"을 뜻하지만 동일 벡터가 아니라는 점에 유의한다.

## 2. 세부 단계별 핵심 디테일 및 보완 가이드

### 2.1. [Step 1] HDBSCAN (h0 기준 1차 분할)

- **목적**
    
    - 레이아웃 상에서 가장 지배적인 물리적 형태(예: 긴 Line, Contact, Corner 등)를 기준으로 거대한 데이터셋을 대분류(Coarse-grained Partitioning)합니다.
        
- **핵심 보완 디테일**
    
    - **Feature Scaling**: $h_0$가 가지는 물리적 차원/스케일이 왜곡되지 않도록 알고리즘 적용 전 표준화(Standardization)를 반드시 선행합니다.
        
    - **노이즈(-1) 제어**: `min_samples` 값을 너무 크게 설정하면 대량의 유효 데이터가 노이즈로 분류되어 누락됩니다. 초기 실험 단계에서는 `5 ~ 10` 수준의 보수적인 값으로 시작하여 노이즈 비율을 모니터링해야 합니다.
        
    - **Soft Clustering 기법 도입**: 노이즈(-1)로 분류된 임베딩을 단순히 제외하기보다는, HDBSCAN의 소속 확률(`probabilities_`)을 계산하여 가장 가까운 군집에 임시 배정하거나 별도의 '잡음 군집(Noise Pool)'으로 관리하여 Step 2로 인계합니다.
        

### 2.2. [Step 2 & 3] 로컬 k-NN 그래프 구성 및 Louvain (2차 정밀 분류)

- **목적**
    
    - 동일한 중심 구조($h_0$)를 가졌더라도, 주변 위상($h_1 \sim h_3$ 및 edge)의 차이에 의해 발생하는 광학적 간섭(OPE) 및 공정 마진 차이를 정밀 분할합니다.
        
- **핵심 보완 디테일**
    
    - **가중치 스케일링 기법 연계**: [[기획안_토폴로지그래프_패턴다양성#4.3 Stage 3 — Hop 분리 임베딩 기반 커뮤니티 탐지|기획안 4.3절]]의 핵심 공식인 $\sqrt{\text{weight} / \text{block\_width}}$ 기법을 적용하여, POI 중심에서 멀어질수록(Hop 번호가 커질수록) 표현 가중치가 감쇠하도록 벡터를 전처리합니다.
        
    - **차원 변별력 극대화**: 2차 분류 단계에서는 이미 $h_0$의 유사성이 통제된 그룹 내이므로, 변별력을 극대화하기 위해 **h1, h2, h3, edge(총 32차원)** 정보만 사용하거나 이 정보들의 가중치를 $h_0$보다 대폭 높여 거리를 계산합니다.
        
    - **적응형 해상도(Adaptive Resolution)**: Louvain의 `resolution` 파라미터를 일괄 고정하는 대신, 각 Sub-cluster의 크기나 밀도 분산에 비례하여 유동적으로 조정하는 로직을 추가함으로써 전체 데이터셋에서 일관된 커뮤니티 세분성(Granularity)을 유지합니다.
        

### 2.3. [Step 4] 대표 패턴 추출 (Representative Sampling)

- **목적**
    
    - 한정된 검증 및 시뮬레이션(OPC, SEM) 리소스 내에서 전체 레이아웃의 구조적 다양성을 대변할 수 있는 알짜배기 대표 패턴을 스크리닝합니다.
        
- **핵심 보완 디테일**
    
    - **최소 추출 하한선 (Minimum Floor) 설정**: 군집 크기에 완벽히 비례해서만 샘플 수를 배분하면, 치명적인 불량을 유발할 수 있으나 발생 빈도가 낮은 '희소 패턴 군집(Long-tail Cluster)'에서 대표 패턴이 단 한 개도 추출되지 않는 심각한 문제가 발생합니다. "군집당 최소 $N$개(예: 3개)"의 하한선을 확보한 뒤 비례 배분합니다.
        
    - **다양성 샘플링 (Diversity-aware Sampling)**: 각 최종 군집의 중심점(Medoid)만 추출하게 되면 가장 평범하고 안전한 패턴 위주로 선별됩니다. 공정 결함 및 마진 분석 목적을 위해, 군집의 중심부 패턴(80%)과 군집 경계면에 위치하여 변동성이 큰 외곽 패턴(20%)을 혼합 추출하는 샘플링 전략을 설계합니다.
        

## 3. 학술적 프레이밍 및 기대 효과

1. **물리적 개연성 확보**
    
    - 리소그래피 공정에서 빛의 회절 한계에 따른 물리적 영향 반경(Interaction Radius)을 GNN의 Hop 범위($h_0 \sim h_3$) 및 계층적 클러스터링 단계와 매핑하여 논리적 타당성을 제시할 수 있습니다.
        
2. **단일 플랫 클러스터링 대비 우위**
    
    - K-Means나 DBSCAN 단독 모델을 적용했을 때 대비, 본 2단계 계층적 구조가 **Silhouette Score** 및 **Modularity** 측면에서 구조적 변별력이 뛰어남을 실험 지표로 정량화하기 용이합니다.