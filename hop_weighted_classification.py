"""
hop_weighted_classification.py — h0/h1/h2/h3/edge_space/edge_vector 를 각각 가중치로
결합한 거리로 KNN-graph + Louvain community detection 분류를 하고, 그 6개 가중치를
random search 로 자동 탐색한다.

기존 스크립트와 관계:
  - visualize_encoder.py / extract_representatives.py 는 LayoutGAE.forward()가 만드는
    단일 graph_emb(= 5개 pool 을 이미 섞어 이어붙인 것)로 클러스터링한다. hop1/hop2/hop3
    가 max_emb/mean_emb 안에 뒤섞여 있어 "h3 가중치를 낮게"같은 걸 할 수 없었다.
  - 이 스크립트는 model.encoder()/model.edge_decoder() 를 직접 호출해 hop별/엣지속성별로
    분리된 pooled 벡터를 뽑는다 — 학습 코드(train_ver2.py 등)는 전혀 안 건드린다.

방법 선택 배경(사용자 확인):
  - 클러스터링 백본: KMeans 대신 KNN-graph + Louvain. 이전에 KMeans 기반 silhouette 이
    0.02 근방으로 나온 게 "둥근 덩어리" 가정이 이 임베딩엔 안 맞을 가능성을 시사해서,
    비볼록/가변밀도 구조에 강한 그래프 기반 커뮤니티 탐지로 바꿨다.
    (신규 의존성: networkx>=2.8 — louvain_communities 가 내장돼 있어 python-louvain 은
    별도로 안 깔아도 된다.)
  - 가중치 탐색: random search. 6개 가중치라 grid search 는 조합이 바로 폭발한다.

가중 결합 방법:
  h0/h1/h2/h3 (각 embedding_dim 폭) 와 espace/evec(각 2폭)를 블록별로 StandardScaler 로
  표준화한 뒤, "폭이 달라도 weight=1일 때 기여도가 비슷하게" sqrt(weight/block_width) 를
  곱해 이어붙인다. 이 벡터에 대한 일반 유클리드 거리가 정확히 가중합 거리가 되므로
  KNN 그래프 구축에 sklearn 을 그대로 쓸 수 있다(거리행렬을 직접 O(N^2)로 안 만들어도 됨
  — N 이 수만~수십만이어도 감당 가능).

평가 지표:
  - modularity: Louvain 이 직접 최적화하는 지표라 이 방법엔 가장 자연스러운 품질 척도.
  - silhouette: 기존 KMeans 결과와 비교하기 위한 참고 지표(표본 기반).
  - largest_frac: 가장 큰 커뮤니티가 전체에서 차지하는 비율. 1에 가까우면 사실상
    "다 한 덩어리"라는 뜻이라 의미 있는 분류가 아니다 — 낮을수록(여러 그룹으로 갈라짐) 좋음.

runtime: 탐색(random search)은 서브샘플(--search-sample-size, 기본 2만개)로 돌리고,
최종 확정된 가중치만 --max-graphs 표본 전체에 적용한다. encoder forward(가장 비싼 부분)는
탐색 전체를 통틀어 딱 1번만 실행하고, 그 결과(hop/edge 임베딩)를 재사용해 가중치만
바꿔가며 반복하므로 트라이얼당 비용은 KNN 구축+Louvain 정도로 저렴하다.

사용 예:
  python hop_weighted_classification.py --ckpt checkpoint_last_v4_cov.pt --dir T2T \
         --max-graphs 200000 --search-sample-size 20000 --n-trials 200
"""

import os
import csv
import json
import argparse

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score

import networkx as nx
from networkx.algorithms.community import louvain_communities, modularity

import train_ver2 as T                        # LayoutGAE, EMBEDDING_DIM 재사용
from visualize_encoder import load_model, resolve_files
from extract_representatives import ParallelFastDataset


# ══════════════════════════════════════════════════════════════════
# [1) hop별 / edge 속성별 분리 임베딩 추출]
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_hop_and_edge_embeddings(model, dataset, indices, device, batch_size, num_workers):
    """
    indices 순서대로 그래프별 h0/h1/h2/h3/espace/evec 를 계산해 (N,dim) numpy dict 로 반환.
    h0~h3 : model.encoder() 가 주는 raw z_node 를 hop 마스크로 masked-mean-pool.
    espace/evec : model.edge_decoder(z_edge) 로 복원한 edge_attr 를
                  [:, :2](box_dist,center_dist)/[:, 2:4](unit_vector) 로 나눠 pool.
    한 그래프에 특정 hop 노드가 하나도 없으면(작은 패턴에서 흔함) 0벡터로 채워진다
    (scatter(reduce='mean') 가 기여 원소 0개인 인덱스는 0으로 남겨두는 동작을 그대로 이용).
    """
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    out = {k: [] for k in ("h0", "h1", "h2", "h3", "espace", "evec")}
    seen = 0
    total = len(indices)

    for batch in loader:
        batch = batch.to(device)
        batch.edge_index = batch.edge_index.long()
        batch.x          = batch.x.float()
        batch.edge_attr  = batch.edge_attr.float()

        z_node, z_edge = model.encoder(batch)
        recon_ea = model.edge_decoder(z_edge)
        num_graphs = batch.num_graphs

        for hop_name, mask_attr in (("h0", "hop0_mask"), ("h1", "hop1_mask"),
                                     ("h2", "hop2_mask"), ("h3", "hop3_mask")):
            mask = getattr(batch, mask_attr)
            idx = batch.batch[mask]
            if idx.numel() == 0:
                pooled = torch.zeros(num_graphs, z_node.size(1), device=device)
            else:
                pooled = scatter(z_node[mask], idx, dim=0, dim_size=num_graphs, reduce="mean")
            out[hop_name].append(pooled.float().cpu().numpy())

        eb = batch.batch[batch.edge_index[0]]
        espace = scatter(recon_ea[:, :2], eb, dim=0, dim_size=num_graphs, reduce="mean")
        evec   = scatter(recon_ea[:, 2:4], eb, dim=0, dim_size=num_graphs, reduce="mean")
        out["espace"].append(espace.float().cpu().numpy())
        out["evec"].append(evec.float().cpu().numpy())

        seen += num_graphs
        if seen % (batch_size * 20) < batch_size:
            print(f"  hop-embed {seen}/{total} ({seen/total:.0%})", end="\r")
    print(f"  hop-embed {total}/{total} (100%)      ")
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ══════════════════════════════════════════════════════════════════
# [2) 블록 표준화 + 가중 결합]
# ══════════════════════════════════════════════════════════════════
def standardize_blocks(raw_blocks: dict) -> dict:
    return {name: StandardScaler().fit_transform(arr).astype(np.float32)
            for name, arr in raw_blocks.items()}


def weighted_concat(std_blocks: dict, weights: dict) -> np.ndarray:
    """
    블록별 표준화된 벡터에 sqrt(weight / block_width) 를 곱해 이어붙인다.
    block_width 로 나누는 이유: h0~h3(폭=embedding_dim, 보통 8)와 espace/evec(폭=2)를
    같은 weight 값으로 섞으면 폭이 넓은 블록이 거리에 훨씬 더 많이 기여해버린다
    (표준화된 D차원 블록의 제곱거리 기여량은 대략 D에 비례). 폭으로 정규화해두면
    weight=1일 때 각 블록의 기여도가 폭과 무관하게 비슷해져서, weight 값 자체가
    "이 블록을 얼마나 중요하게 볼지"를 곧이곧대로 반영하게 된다.
    """
    parts = []
    for name, block in std_blocks.items():
        w = max(float(weights.get(name, 0.0)), 0.0)
        scale = np.sqrt(w / block.shape[1])
        parts.append(block * scale)
    return np.concatenate(parts, axis=1).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# [3) KNN-graph + Louvain]
# ══════════════════════════════════════════════════════════════════
def build_knn_graph(X: np.ndarray, k: int) -> nx.Graph:
    """X(N,D) -> 대칭화된 KNN 그래프. edge weight = 가우시안 커널 유사도(거리 -> 유사도 변환).
    (Louvain 은 weight 가 클수록 '강하게 연결'로 보므로 거리를 그대로 못 씀)"""
    knn = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False, n_jobs=-1)
    knn = knn.maximum(knn.T)   # 비대칭 KNN -> 대칭화(둘 중 하나라도 이웃이면 엣지 유지)
    knn = knn.tocoo()

    sigma = float(np.median(knn.data)) + 1e-8
    sim = np.exp(-(knn.data ** 2) / (2 * sigma ** 2))

    G = nx.Graph()
    G.add_nodes_from(range(X.shape[0]))
    G.add_weighted_edges_from(zip(knn.row.tolist(), knn.col.tolist(), sim.tolist()))
    return G


def run_louvain(G: nx.Graph, resolution: float, seed: int):
    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    labels = np.empty(G.number_of_nodes(), dtype=np.int64)
    for cid, members in enumerate(communities):
        for node in members:
            labels[node] = cid
    return labels, communities


def evaluate_partition(X: np.ndarray, G: nx.Graph, labels: np.ndarray, communities: list) -> dict:
    mod = float(modularity(G, communities, weight="weight"))
    n_comm = len(communities)
    sizes = np.array([len(c) for c in communities]) if n_comm else np.array([0])
    sil = float("nan")
    if n_comm >= 2:
        try:
            s_idx = np.random.default_rng(0).choice(len(X), size=min(5000, len(X)), replace=False)
            sil = float(silhouette_score(X[s_idx], labels[s_idx]))
        except Exception:
            pass
    return {
        "modularity": mod,
        "silhouette": sil,
        "n_communities": n_comm,
        "largest_frac": float(sizes.max() / sizes.sum()) if sizes.sum() > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════
# [4) Random search — 가중치 자동 탐색]
# ══════════════════════════════════════════════════════════════════
def sample_weight_vector(rng: np.random.Generator, block_names: list) -> dict:
    w = rng.dirichlet(np.ones(len(block_names)))   # 심플렉스 위 균등 샘플(합=1)
    return dict(zip(block_names, w.tolist()))


def random_search_weights(std_blocks: dict, block_names: list, n_trials: int, k: int,
                          resolution: float, seed: int, sample_size: int):
    rng = np.random.default_rng(seed)
    N = next(iter(std_blocks.values())).shape[0]
    sample_idx = rng.choice(N, size=min(sample_size, N), replace=False)
    sub_blocks = {name: arr[sample_idx] for name, arr in std_blocks.items()}

    trials, best = [], None
    for t in range(n_trials):
        weights = sample_weight_vector(rng, block_names)
        X = weighted_concat(sub_blocks, weights)
        G = build_knn_graph(X, k=k)
        labels, communities = run_louvain(G, resolution=resolution, seed=seed + t)
        metrics = evaluate_partition(X, G, labels, communities)

        row = {"trial": t, **weights, **metrics}
        trials.append(row)
        if best is None or metrics["modularity"] > best["modularity"]:
            best = row

        print(f"  [{t + 1}/{n_trials}] modularity={metrics['modularity']:.4f}  "
              f"silhouette={metrics['silhouette']:.4f}  "
              f"n_communities={metrics['n_communities']}  "
              f"largest={metrics['largest_frac']:.1%}")

    return best, trials


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="hop/edge-type 가중 결합 + KNN-graph/Louvain 분류")
    ap.add_argument("--ckpt", type=str, default="checkpoint_last_v4_cov.pt",
                    help="전체 모델 체크포인트(권장) — center_proj 는 안 쓰지만 encoder/edge_decoder"
                         " 는 그대로 필요")
    ap.add_argument("--file", type=str, default=None, help="PREPROCESSED_*.pt 단일 파일")
    ap.add_argument("--dir", type=str, default=None, help="청크 디렉터리")
    ap.add_argument("--max-graphs", type=int, default=None,
                    help="hop 임베딩 추출 + 최종 클러스터링에 쓸 최대 그래프 수(랜덤 표본)")
    ap.add_argument("--search-sample-size", type=int, default=20000,
                    help="random search 트라이얼마다 쓰는 서브샘플 크기")
    ap.add_argument("--n-trials", type=int, default=200, help="random search 시도 횟수")
    ap.add_argument("--k", type=int, default=15, help="KNN 그래프의 이웃 수")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Louvain resolution(1보다 크면 더 잘게, 작으면 더 크게 묶임)")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--load-workers", type=int, default=None)
    ap.add_argument("--out-dir", type=str, default="hop_weighted_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  | out_dir: {args.out_dir}")

    files = resolve_files(args)
    print(f"[files] {len(files)}개 청크")

    dataset = ParallelFastDataset(files, max_load_workers=args.load_workers)
    model = load_model(args.ckpt, dataset.num_node_features, dataset.num_edge_features,
                        T.EMBEDDING_DIM, device)

    N = len(dataset)
    rng = np.random.default_rng(args.seed)
    if args.max_graphs and args.max_graphs < N:
        sel = np.sort(rng.choice(N, size=args.max_graphs, replace=False))
    else:
        sel = np.arange(N)
    keys_sel = [dataset.keys[i] for i in sel]
    print(f"[sample] {len(sel)}/{N} graphs 사용")

    print("\n[1] hop / edge 속성별 임베딩 추출 (encoder forward — 탐색 전체에서 1번만)")
    raw_blocks = compute_hop_and_edge_embeddings(
        model, dataset, list(sel), device, args.batch_size, args.num_workers
    )

    print("\n[2] 블록별 표준화 (h0,h1,h2,h3,espace,evec)")
    std_blocks = standardize_blocks(raw_blocks)
    block_names = list(std_blocks.keys())

    print(f"\n[3] Random search ({args.n_trials}회, 서브샘플 {args.search_sample_size}개, k={args.k})")
    best, trials = random_search_weights(
        std_blocks, block_names, args.n_trials, args.k, args.resolution,
        args.seed, args.search_sample_size
    )

    trials_csv = os.path.join(args.out_dir, "weight_search_trials.csv")
    with open(trials_csv, "w", newline="") as f:
        fieldnames = ["trial"] + block_names + ["modularity", "silhouette", "n_communities", "largest_frac"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trials:
            w.writerow(row)
    print(f"    saved {trials_csv}")

    best_weights = {name: best[name] for name in block_names}
    print(f"\n[best] weights={ {k: round(v, 4) for k, v in best_weights.items()} }")
    print(f"       modularity={best['modularity']:.4f}  silhouette={best['silhouette']:.4f}  "
          f"n_communities={best['n_communities']}  largest_frac={best['largest_frac']:.1%}")

    print("\n[4] 최적 가중치로 표본 전체 재클러스터링")
    X_full = weighted_concat(std_blocks, best_weights)
    G_full = build_knn_graph(X_full, k=args.k)
    labels_full, communities_full = run_louvain(G_full, resolution=args.resolution, seed=args.seed)
    final_metrics = evaluate_partition(X_full, G_full, labels_full, communities_full)
    print(f"    표본 {len(sel)}개 기준: modularity={final_metrics['modularity']:.4f}  "
          f"silhouette={final_metrics['silhouette']:.4f}  "
          f"n_communities={final_metrics['n_communities']}  "
          f"largest_frac={final_metrics['largest_frac']:.1%}")

    summary_path = os.path.join(args.out_dir, "best_weights.json")
    with open(summary_path, "w") as f:
        json.dump({
            "weights": best_weights,
            "search_metrics": {k: best[k] for k in
                               ("modularity", "silhouette", "n_communities", "largest_frac")},
            "full_sample_metrics": final_metrics,
            "k": args.k, "resolution": args.resolution,
            "n_trials": args.n_trials, "search_sample_size": args.search_sample_size,
            "n_graphs_full": int(len(sel)),
        }, f, indent=2, ensure_ascii=False)
    print(f"    saved {summary_path}")

    assign_path = os.path.join(args.out_dir, "cluster_assignments.csv")
    with open(assign_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key(gauge_name)", "community_id"])
        for key, label in zip(keys_sel, labels_full):
            w.writerow([key, int(label)])
    print(f"    saved {assign_path}")

    print(f"\n완료. 결과물: {args.out_dir}/")
    print("  weight_search_trials.csv, best_weights.json, cluster_assignments.csv")


if __name__ == "__main__":
    main()
