"""
visualize_encoder.py — 학습된 encoder(GAE) 검증/시각화 파이프라인

요청 3가지를 모두 수행한다:
  1) preprocessor 산출물 → encoder → graph embedding 이 어떻게 나오는지
     (임베딩 통계 = collapse 여부 / 유효 차원 수 확인, 차원별 히스토그램)
  2) encoder output(graph embedding) 로 unsupervised classification(클러스터링) 후
     각 클러스터의 대표 패턴(medoid) = 100개 point 와 그 key(gauge_name) 산출 (CSV)
  3) classification 과 대표패턴이 잘 뽑혔는지 보여주는 plot:
     - 2D 투영에 클러스터 색칠 + 대표(medoid) 별표
     - 클러스터 크기 분포 + silhouette 점수
     - 대표 gauge 들의 실제 패턴(노드 bbox, hop 색) 격자 — "대표패턴이 말이 되는지" 눈으로 확인

임베딩 대상:
  LayoutGAE.forward 가 만드는 graph_emb = concat[max, mean, center, emean, emax] (= embedding_dim*5).
  center 브랜치가 GAE 의 center_proj 를 쓰므로, 정확한 재현을 위해선
  전체 모델이 저장된 checkpoint_last_v2.pt(RESUME) 를 쓰는 것이 정확하다.
  (encoder 단독 best_gat_v2_*.pt 는 center_proj 가 복원되지 않아 근사값이 됨 — 경고 출력)

사용 예:
  python visualize_encoder.py --ckpt checkpoint_last_v2.pt --dir T2T --n-clusters 100
  python visualize_encoder.py --ckpt checkpoint_last_v2.pt --file T2T/PREPROCESSED_T2T.oas_part0.pt \
         --max-graphs 200000 --method umap --out-dir enc_viz
"""

import os
import csv
import glob
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from torch_geometric.loader import DataLoader

import train_ver2 as T                       # LayoutGAE, FastPreloadedDataset, EMBEDDING_DIM 재사용
from plot_preprocessor_check import load_chunk, unpack_graph, HOP_COLORS, HOP_LABELS


# ══════════════════════════════════════════════════════════════════
# [파일/키 인덱스]
# ══════════════════════════════════════════════════════════════════
def resolve_files(args) -> list:
    if args.file:
        return [args.file]
    search_dir = args.dir or "."
    cands = sorted(glob.glob(os.path.join(search_dir, "PREPROCESSED_*part*.pt")))
    if not cands:
        raise FileNotFoundError(f"PREPROCESSED_*part*.pt 없음: {search_dir}")
    return cands


def build_key_index(files: list):
    """
    FastPreloadedDataset 와 동일한 순서로 (key, (file, local_idx)) 를 만든다.
    → dataset[i] 의 gauge_name = keys[i], 원본 위치 = idx_map[i].
    """
    keys, idx_map = [], []
    n_node_feat = n_edge_feat = None
    for f in files:
        d = load_chunk(f)
        if n_node_feat is None:
            n_node_feat = d["x"].size(1)
            n_edge_feat = d["edge_attr"].size(1)
        ks = d.get("keys", [f"{os.path.basename(f)}#{i}" for i in range(d["meta"].size(0))])
        for i, k in enumerate(ks):
            keys.append(str(k))
            idx_map.append((f, i))
        del d
    return keys, idx_map, n_node_feat, n_edge_feat


# ══════════════════════════════════════════════════════════════════
# [모델 로드]
# ══════════════════════════════════════════════════════════════════
def load_model(ckpt_path, n_node_feat, n_edge_feat, emb_dim, device):
    model = T.LayoutGAE(num_node_features=n_node_feat,
                        num_edge_features=n_edge_feat,
                        embedding_dim=emb_dim)
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    try:
        model.load_state_dict(state)
        print(f"[model] 전체 GAE 로드 완료 (graph_emb 정확 재현): {ckpt_path}")
    except Exception:
        # encoder 단독 state_dict 로 판단 → encoder 에만 로드
        res = model.encoder.load_state_dict(state, strict=False)
        print(f"[model] ⚠ encoder 단독 로드: center_proj 미복원 → center 브랜치는 근사값!")
        print(f"        (정확한 임베딩을 원하면 checkpoint_last_v2.pt(전체 모델)를 쓰세요)")
        if getattr(res, "missing_keys", None):
            print(f"        missing={list(res.missing_keys)[:4]}...")
    return model.to(device).eval()


# ══════════════════════════════════════════════════════════════════
# [1) 임베딩 추출]
# ══════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_embeddings(model, dataset, indices, device, batch_size, num_workers):
    """indices 순서대로 graph_emb 를 계산해 (N, D) numpy 로 반환."""
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    out = []
    seen = 0
    total = len(indices)
    for batch in loader:
        batch = batch.to(device)
        batch.edge_index = batch.edge_index.long()
        batch.x          = batch.x.float()
        batch.edge_attr  = batch.edge_attr.float()
        _, _, graph_emb, _ = model(batch)
        out.append(graph_emb.float().cpu().numpy())
        seen += graph_emb.size(0)
        if seen % (batch_size * 20) < batch_size:
            print(f"  embedding {seen}/{total} ({seen/total:.0%})", end="\r")
    print(f"  embedding {total}/{total} (100%)      ")
    return np.concatenate(out, axis=0)


def report_embedding_stats(emb: np.ndarray, out_dir: str):
    """collapse 여부 / 유효 차원 확인 + 차원별 히스토그램."""
    D = emb.shape[1]
    mean = emb.mean(0); std = emb.std(0)
    active = int((std > 1e-3).sum())
    print("\n[1] Embedding stats  (graph_emb = embedding_dim × 5 pools)")
    print(f"    shape={emb.shape}  mean|std over dims: std_avg={std.mean():.4f}  "
          f"active_dims(std>1e-3)={active}/{D}")
    # 유효 랭크(설명분산 95%에 필요한 PCA 성분 수) — 실질 정보 차원
    p = PCA().fit(StandardScaler().fit_transform(emb))
    cum = np.cumsum(p.explained_variance_ratio_)
    eff = int(np.searchsorted(cum, 0.95) + 1)
    print(f"    effective rank (95% var) = {eff}/{D}  "
          f"→ {'다양성 충분' if eff >= 3 else '⚠ 저차원으로 붕괴 의심'}")

    ncol = min(5, D); nrow = int(np.ceil(D / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow), squeeze=False)
    for i in range(nrow * ncol):
        ax = axes[i // ncol][i % ncol]
        if i < D:
            ax.hist(emb[:, i], bins=60, color="#3b7dd8")
            ax.set_title(f"dim {i}  (std={std[i]:.2f})", fontsize=8)
            ax.tick_params(labelsize=6)
        else:
            ax.axis("off")
    fig.suptitle("[1] Per-dimension embedding distribution (collapse check)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "1_embedding_dims.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"    saved {path}")
    return eff


# ══════════════════════════════════════════════════════════════════
# [2) 분류 + 대표(medoid) 산출]
# ══════════════════════════════════════════════════════════════════
def classify_and_representatives(emb, keys, idx_map, n_clusters, seed=0):
    """StandardScaler → MiniBatchKMeans → 클러스터별 medoid(실제 gauge) 선정."""
    N = emb.shape[0]
    K = min(n_clusters, max(2, N // 2))
    if K < n_clusters:
        print(f"    ⚠ 표본이 적어 n_clusters {n_clusters}→{K} 로 축소")

    scaler = StandardScaler().fit(emb)
    X = scaler.transform(emb)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed,
                         batch_size=max(1024, N // 100 + 1), n_init=3)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # 각 점 → 자기 centroid 거리 (표준화 공간)
    dist = np.linalg.norm(X - centroids[labels], axis=1)

    reps = []   # (cluster, size, medoid_global_row, key, dist)
    for c in range(K):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        m = members[np.argmin(dist[members])]
        reps.append({"cluster": c, "size": int(len(members)),
                     "row": int(m), "key": keys[m], "dist": float(dist[m])})
    reps.sort(key=lambda r: r["size"], reverse=True)   # 큰 클러스터 우선
    return labels, X, reps


def save_representatives_csv(reps, proj2d, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank_by_size", "cluster_id", "key(gauge_name)",
                    "cluster_size", "dist_to_centroid", "proj_x", "proj_y"])
        for rank, r in enumerate(reps):
            px, py = proj2d[r["row"]]
            w.writerow([rank, r["cluster"], r["key"], r["size"],
                        f"{r['dist']:.4f}", f"{px:.4f}", f"{py:.4f}"])
    print(f"    saved {path}  (대표 패턴 {len(reps)}개)")


# ══════════════════════════════════════════════════════════════════
# [3) Plots]
# ══════════════════════════════════════════════════════════════════
def project_2d(X, method, seed=0, sample_for_fit=200000):
    """대용량이면 표본으로 fit 후 전체 transform (PCA), umap/tsne 는 표본만."""
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "umap":
        try:
            import umap
            return umap.UMAP(n_components=2, random_state=seed).fit_transform(X)
        except Exception as e:
            print(f"    ⚠ umap 사용 불가({e}) → PCA 로 대체")
            return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(X)
    raise ValueError(method)


def plot_clusters(proj2d, labels, reps, out_dir, sample=40000, seed=0):
    N = proj2d.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(sample, N), replace=False)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.scatter(proj2d[idx, 0], proj2d[idx, 1], c=labels[idx], cmap="tab20",
               s=4, alpha=0.45, linewidths=0)
    # 대표(medoid) 별표 + 상위 클러스터 라벨
    for rank, r in enumerate(reps):
        px, py = proj2d[r["row"]]
        ax.plot(px, py, "*", color="black", ms=13, markerfacecolor="yellow",
                markeredgewidth=0.8, zorder=5)
        if rank < 15:   # 큰 클러스터 15개만 key 라벨
            ax.text(px, py, f" {r['key']}", fontsize=6.5, zorder=6)
    ax.set_title(f"[3a] Cluster projection ({N} pts, {len(reps)} clusters)\n"
                 f"★ = representative medoid (real gauge)")
    ax.set_xlabel("proj-1"); ax.set_ylabel("proj-2")
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(out_dir, "3a_cluster_projection.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"    saved {path}")


def plot_cluster_quality(X, labels, reps, out_dir, seed=0):
    sizes = np.array([r["size"] for r in reps])
    # silhouette 는 표본으로 (전체는 O(N^2) 라 비쌈)
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    s_idx = rng.choice(N, size=min(10000, N), replace=False)
    try:
        sil = silhouette_score(X[s_idx], labels[s_idx])
    except Exception:
        sil = float("nan")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.bar(range(len(sizes)), sizes, color="#3b7dd8")
    a1.set_title(f"[3b] Cluster sizes (sorted)  |  silhouette≈{sil:.3f}")
    a1.set_xlabel("cluster rank"); a1.set_ylabel("# gauges")
    a2.hist(sizes, bins=40, color="#e08a3b")
    a2.set_title("Cluster size histogram")
    a2.set_xlabel("cluster size"); a2.set_ylabel("# clusters")
    fig.tight_layout()
    path = os.path.join(out_dir, "3b_cluster_quality.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"    saved {path}  (silhouette≈{sil:.3f}: 1에 가까울수록 분리 양호)")
    return sil


def _draw_pattern(ax, g, title):
    x, hop = g["x"], g["hop"]
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    for i in range(len(x)):
        col = HOP_COLORS[hop[i]]
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               facecolor=col, edgecolor="black", lw=0.4, alpha=0.6))
    h0 = np.where(hop == 0)[0]
    if len(h0):
        ax.plot(cx[h0], cy[h0], "*", color="black", ms=12,
                markerfacecolor=HOP_COLORS[0], zorder=4)
    ax.set_title(title, fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=6); ax.grid(True, ls=":", alpha=0.3)


def plot_representative_patterns(reps, idx_map, out_dir, top=12):
    """상위 클러스터 medoid gauge 들의 실제 패턴(노드 bbox) 격자."""
    top = min(top, len(reps))
    ncol = 4; nrow = int(np.ceil(top / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    chunk_cache = {}
    for j in range(nrow * ncol):
        ax = axes[j // ncol][j % ncol]
        if j >= top:
            ax.axis("off"); continue
        r = reps[j]
        f, local_idx = idx_map[r["row"]]
        if f not in chunk_cache:
            chunk_cache[f] = load_chunk(f)
        g = unpack_graph(chunk_cache[f], local_idx)
        _draw_pattern(ax, g, f"#{j} c{r['cluster']} n={r['size']}\n{r['key']}")
    fig.suptitle("[3c] Representative (medoid) patterns of largest clusters\n"
                 "colored by hop0(center,star)/hop1/hop2/hop3+ "
                 "- check clusters look visually distinct", fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "3c_representative_patterns.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"    saved {path}")


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="학습된 encoder 검증 시각화")
    ap.add_argument("--ckpt", type=str, default="checkpoint_last_v2.pt",
                    help="전체 모델 체크포인트(권장) 또는 encoder 단독 .pt")
    ap.add_argument("--file", type=str, default=None, help="PREPROCESSED_*.pt 단일 파일")
    ap.add_argument("--dir", type=str, default=None, help="청크 디렉터리")
    ap.add_argument("--max-graphs", type=int, default=200000,
                    help="임베딩/클러스터링에 쓸 최대 그래프 수(랜덤 표본). 전체면 크게.")
    ap.add_argument("--n-clusters", type=int, default=100, help="클러스터 수 = 대표 패턴 수")
    ap.add_argument("--method", type=str, default="pca", choices=["pca", "umap", "tsne"],
                    help="2D 투영 방법")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", type=str, default="enc_viz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  | out_dir: {args.out_dir}")

    files = resolve_files(args)
    print(f"[files] {len(files)}개 청크")
    keys, idx_map, n_node_feat, n_edge_feat = build_key_index(files)
    print(f"[index] 총 {len(keys)} graphs  | node_feat={n_node_feat} edge_feat={n_edge_feat}")

    model = load_model(args.ckpt, n_node_feat, n_edge_feat, T.EMBEDDING_DIM, device)

    dataset = T.FastPreloadedDataset(files)
    assert len(dataset) == len(keys), f"dataset({len(dataset)}) != keys({len(keys)}) 정렬 불일치"

    # 표본 선택 (정렬해 캐시 지역성 유지)
    N = len(dataset)
    rng = np.random.default_rng(args.seed)
    if args.max_graphs < N:
        sel = np.sort(rng.choice(N, size=args.max_graphs, replace=False))
    else:
        sel = np.arange(N)
    keys_sel = [keys[i] for i in sel]
    idx_map_sel = [idx_map[i] for i in sel]
    print(f"[sample] {len(sel)}/{N} graphs 사용")

    # ── 1) 임베딩 ────────────────────────────────────────────────
    emb = compute_embeddings(model, dataset, list(sel), device,
                             args.batch_size, args.num_workers)
    np.savez(os.path.join(args.out_dir, "embeddings.npz"),
             emb=emb, keys=np.array(keys_sel))
    report_embedding_stats(emb, args.out_dir)

    # ── 2) 분류 + 대표 ──────────────────────────────────────────
    print("\n[2] Clustering + representatives")
    labels, X, reps = classify_and_representatives(
        emb, keys_sel, idx_map_sel, args.n_clusters, seed=args.seed)

    # ── 3) Plots ────────────────────────────────────────────────
    print("\n[3] Projection + plots")
    proj2d = project_2d(X, args.method, seed=args.seed)
    save_representatives_csv(reps, proj2d,
                             os.path.join(args.out_dir, "representatives.csv"))
    plot_clusters(proj2d, labels, reps, args.out_dir, seed=args.seed)
    plot_cluster_quality(X, labels, reps, args.out_dir, seed=args.seed)
    plot_representative_patterns(reps, idx_map_sel, args.out_dir, top=12)

    print(f"\n완료. 결과물: {args.out_dir}/")
    print("  embeddings.npz, representatives.csv,")
    print("  1_embedding_dims.png, 3a_cluster_projection.png,")
    print("  3b_cluster_quality.png, 3c_representative_patterns.png")


if __name__ == "__main__":
    main()
