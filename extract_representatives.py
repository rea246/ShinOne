"""
extract_representatives.py — 학습된 encoder 로 preprocessor 산출물 전체를 임베딩한 뒤,
데이터셋을 대표하는 N개 샘플(medoid)을 뽑고 "왜 대표인지"에 대한 근거 데이터를 산출한다.

visualize_encoder.py 와 기능 분리:
  - visualize_encoder.py       : 사람이 눈으로 보는 시각화(플롯) 중심 (기본 클러스터 100개)
  - extract_representatives.py : 대표 샘플 자체 + 대표성 근거 수치를 뽑는 파이프라인 전용.
    (예: 1000개 coreset 뽑아서 후속 라벨링/재학습/QA 샘플링에 바로 쓰는 용도)

"왜 대표인지" 근거 (representatives.csv 컬럼):
  - cluster_size / cluster_size_pct   : 이 샘플이 대표하는 그룹 크기(=데이터셋에서 차지하는 비중)
  - dist_to_centroid / typicality_ratio : 그룹 중심에 얼마나 가까운지(= 얼마나 "전형적"인지).
    typicality_ratio = dist_to_centroid / intra_cluster_mean_dist. 1보다 작을수록 그룹 평균 멤버
    보다 더 중심에 있다는 뜻 = medoid 로 뽑힌 근거.
  - intra_cluster_mean/std/max_dist   : 이 그룹이 얼마나 뭉쳐 있는지(응집도). 크면 그룹 내부
    편차가 크다는 뜻이라 대표 1개로 뭉뚱그리는 데 대한 리스크 신호.
  - nearest_cluster_id / nearest_cluster_centroid_dist / separation_ratio : 가장 가까운 다른
    그룹과 얼마나 떨어져 있는지(구별성). separation_ratio = nearest_dist / intra_mean_dist,
    1보다 훨씬 크면 다른 그룹과 확실히 구분되는 패턴이라는 뜻.
  - proj_x / proj_y                    : PCA 2D 투영 좌표 (representatives.csv 만 봐도 분포 확인용)

성능(멀티코어):
  - 청크(.pt) 로드: ThreadPoolExecutor 로 파일들을 동시에 읽음 (I/O + torch 역직렬화 구간은
    GIL 이 풀리므로 코어 수만큼 실제로 빨라짐). 기존 FastPreloadedDataset(train_ver2.py) 는
    파일을 순차(for)로 읽어 이 부분이 직렬 병목이었음.
  - 인코더 forward: DataLoader(num_workers>0) 로 배치 collation 을 CPU 코어에 분산.
  - 클러스터별 대표성 통계: 클러스터마다 독립적인 numpy 축약 연산이라 ThreadPoolExecutor 로
    클러스터 단위 병렬화.

사용 예:
  python extract_representatives.py --ckpt checkpoint_last_v2.pt --dir T2T --n-samples 1000
  python extract_representatives.py --ckpt checkpoint_last_v2.pt --dir T2T --n-samples 1000 \
         --max-graphs 200000 --num-workers 6 --plot
"""

import os
import csv
import json
import time
import argparse
import concurrent.futures

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import train_ver2 as T                        # LayoutGAE, EMBEDDING_DIM 재사용
from plot_preprocessor_check import load_chunk
from visualize_encoder import (
    load_model, compute_embeddings, resolve_files,
    plot_clusters, plot_cluster_quality, plot_representative_patterns,
)


# ══════════════════════════════════════════════════════════════════
# [병렬 청크 로드]
#   FastPreloadedDataset(train_ver2.py) 은 for 문으로 파일을 순차 로드한다.
#   여기서는 ThreadPoolExecutor 로 동시에 읽어 로딩 시간을 단축하고,
#   같은 패스에서 (file, local_idx) idx_map 까지 함께 만들어 재읽기를 없앤다.
# ══════════════════════════════════════════════════════════════════
def _load_one_chunk(path: str):
    d = load_chunk(path)
    keys = d.get("keys", [f"{os.path.basename(path)}#{i}" for i in range(d["meta"].size(0))])
    return path, d["x"], d["edge_index"], d["edge_attr"], d["hop_packed"], d["meta"], keys


class ParallelFastDataset(Dataset):
    """PREPROCESSED_*.pt 청크들을 스레드풀로 동시 로드해 그래프 단위로 슬라이싱하는 Dataset.
    train_ver2.FastPreloadedDataset 과 텐서 포맷은 동일하되 로딩만 병렬화했다."""

    def __init__(self, file_paths, max_load_workers=None):
        super().__init__()
        max_load_workers = max_load_workers or min(32, (os.cpu_count() or 4) * 2, len(file_paths))
        print(f"[ParallelFastDataset] {len(file_paths)}개 청크를 {max_load_workers}개 스레드로 로드...")
        t0 = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_load_workers) as ex:
            loaded = list(ex.map(_load_one_chunk, file_paths))

        xs, eis, eas, hms = [], [], [], []
        self.slices = []          # (x0,x1,e0,e1) 전역 오프셋
        self.keys = []            # gauge 이름
        self.idx_map = []         # (file_path, local_idx) — 원본 위치 추적용
        x_off = e_off = 0

        for path, x, ei, ea, hp, meta, keys in loaded:
            xs.append(x); eis.append(ei); eas.append(ea); hms.append(hp)
            for i, (x0, x1, e0, e1) in enumerate(meta.tolist()):
                self.slices.append((x0 + x_off, x1 + x_off, e0 + e_off, e1 + e_off))
                self.keys.append(str(keys[i]))
                self.idx_map.append((path, i))
            x_off += x.size(0)
            e_off += ei.size(1)

        self.all_x          = torch.cat(xs, dim=0)
        self.all_edge_index = torch.cat(eis, dim=1)
        self.all_edge_attr  = torch.cat(eas, dim=0)
        self.all_hop_packed = torch.cat(hms, dim=0)
        del loaded, xs, eis, eas, hms

        self.num_node_features = self.all_x.size(1)
        self.num_edge_features = self.all_edge_attr.size(1)
        print(f"[ParallelFastDataset] 총 {len(self.slices)}개 그래프 로드 완료 ({time.time()-t0:.1f}s)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        xs, xe, es, ee = self.slices[idx]
        x          = self.all_x[xs:xe].float()
        edge_attr  = self.all_edge_attr[es:ee].float()
        edge_index = self.all_edge_index[:, es:ee].long()
        packed     = self.all_hop_packed[xs:xe]
        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            hop0_mask=(packed & 1).bool(),
            hop1_mask=((packed >> 1) & 1).bool(),
            hop2_mask=((packed >> 2) & 1).bool(),
            hop3_mask=((packed >> 3) & 1).bool()
        )


# ══════════════════════════════════════════════════════════════════
# [클러스터링 + 대표(medoid) + "왜 대표인지" 통계]
# ══════════════════════════════════════════════════════════════════
def cluster_and_explain(emb: np.ndarray, keys: list, n_samples: int, seed: int, stat_workers: int):
    N = emb.shape[0]
    K = min(n_samples, max(2, N // 2))
    if K < n_samples:
        print(f"    ⚠ 표본이 적어 n_samples {n_samples}→{K} 로 축소")

    scaler = StandardScaler().fit(emb)
    X = scaler.transform(emb)
    km = MiniBatchKMeans(n_clusters=K, random_state=seed,
                         batch_size=max(1024, N // 100 + 1), n_init=3)
    labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # 각 점 → 자기 centroid 까지 거리(표준화 공간)
    dist = np.linalg.norm(X - centroids[labels], axis=1)

    # 클러스터별로 연속된 블록이 되도록 정렬 → 블록 단위 슬라이싱(스레드에 독립적으로 분배)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    boundaries = np.searchsorted(sorted_labels, np.arange(K + 1))
    member_groups = [order[boundaries[c]:boundaries[c + 1]] for c in range(K)]

    # 클러스터 간 중심 거리 (구별성 판단용) — K x K, K<=수천이라 벡터화로 충분히 빠름
    centroid_dists = cdist(centroids, centroids)
    np.fill_diagonal(centroid_dists, np.inf)
    nearest_other = np.argmin(centroid_dists, axis=1)
    nearest_other_dist = centroid_dists[np.arange(K), nearest_other]

    def _stat_for_cluster(c):
        members = member_groups[c]
        if len(members) == 0:
            return None
        d = dist[members]
        local_argmin = int(np.argmin(d))
        medoid_row = int(members[local_argmin])
        intra_mean = float(d.mean())
        return {
            "cluster": c,
            "size": int(len(members)),
            "row": medoid_row,
            "key": keys[medoid_row],
            "dist": float(d[local_argmin]),          # dist_to_centroid
            "intra_mean_dist": intra_mean,
            "intra_std_dist": float(d.std()),
            "intra_max_dist": float(d.max()),
            "typicality_ratio": float(d[local_argmin] / (intra_mean + 1e-8)),
            "nearest_cluster": int(nearest_other[c]),
            "nearest_cluster_dist": float(nearest_other_dist[c]),
            "separation_ratio": float(nearest_other_dist[c] / (intra_mean + 1e-8)),
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=stat_workers) as ex:
        reps = list(ex.map(_stat_for_cluster, range(K)))
    reps = [r for r in reps if r is not None]
    for r in reps:
        r["size_pct"] = r["size"] / N * 100.0
    reps.sort(key=lambda r: r["size"], reverse=True)

    return labels, X, reps


def save_representatives_csv(reps, proj2d, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank_by_size", "cluster_id", "key(gauge_name)",
            "cluster_size", "cluster_size_pct",
            "dist_to_centroid", "typicality_ratio",
            "intra_cluster_mean_dist", "intra_cluster_std_dist", "intra_cluster_max_dist",
            "nearest_cluster_id", "nearest_cluster_centroid_dist", "separation_ratio",
            "proj_x", "proj_y",
        ])
        for rank, r in enumerate(reps):
            px, py = proj2d[r["row"]]
            w.writerow([
                rank, r["cluster"], r["key"],
                r["size"], f"{r['size_pct']:.4f}",
                f"{r['dist']:.4f}", f"{r['typicality_ratio']:.4f}",
                f"{r['intra_mean_dist']:.4f}", f"{r['intra_std_dist']:.4f}", f"{r['intra_max_dist']:.4f}",
                r["nearest_cluster"], f"{r['nearest_cluster_dist']:.4f}", f"{r['separation_ratio']:.4f}",
                f"{px:.4f}", f"{py:.4f}",
            ])
    print(f"    saved {path}  (대표 패턴 {len(reps)}개)")


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="학습된 encoder 로 데이터셋 대표 샘플 추출")
    ap.add_argument("--ckpt", type=str, default="checkpoint_last_v2.pt",
                    help="전체 모델 체크포인트(권장) 또는 encoder 단독 .pt")
    ap.add_argument("--file", type=str, default=None, help="PREPROCESSED_*.pt 단일 파일")
    ap.add_argument("--dir", type=str, default=None, help="청크 디렉터리")
    ap.add_argument("--n-samples", type=int, default=1000, help="추출할 대표 샘플 수")
    ap.add_argument("--max-graphs", type=int, default=None,
                    help="임베딩/클러스터링에 쓸 최대 그래프 수(랜덤 표본). 미지정 시 전체 사용.")
    ap.add_argument("--batch-size", type=int, default=2048, help="인코더 forward 배치 크기")
    ap.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="DataLoader collation 워커 수 (인코더 forward 병렬화)")
    ap.add_argument("--load-workers", type=int, default=None,
                    help="청크 파일 로드 스레드 수 (기본: min(32, cpu*2, 파일수))")
    ap.add_argument("--stat-workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="클러스터별 대표성 통계 계산 스레드 수")
    ap.add_argument("--out-dir", type=str, default="representatives_out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true",
                    help="분포/품질/대표패턴 PNG 도 함께 생성(visualize_encoder.py 플롯 재사용)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  | out_dir: {args.out_dir}")

    timing = {}
    t0 = time.time()

    files = resolve_files(args)
    print(f"[files] {len(files)}개 청크")

    dataset = ParallelFastDataset(files, max_load_workers=args.load_workers)
    timing["load_sec"] = time.time() - t0
    N = len(dataset)

    model = load_model(args.ckpt, dataset.num_node_features, dataset.num_edge_features,
                        T.EMBEDDING_DIM, device)

    # 표본 선택 (정렬해 슬라이싱 캐시 지역성 유지)
    rng = np.random.default_rng(args.seed)
    if args.max_graphs and args.max_graphs < N:
        sel = np.sort(rng.choice(N, size=args.max_graphs, replace=False))
    else:
        sel = np.arange(N)
    keys_sel = [dataset.keys[i] for i in sel]
    idx_map_sel = [dataset.idx_map[i] for i in sel]
    print(f"[sample] {len(sel)}/{N} graphs 사용")

    # ── 1) 임베딩 (인코더 forward, DataLoader num_workers 로 CPU 코어 분산) ──
    t1 = time.time()
    emb = compute_embeddings(model, dataset, list(sel), device, args.batch_size, args.num_workers)
    timing["embed_sec"] = time.time() - t1
    np.savez(os.path.join(args.out_dir, "embeddings.npz"), emb=emb, keys=np.array(keys_sel))

    # ── 2) 클러스터링 + 대표 + 대표성 통계 (클러스터 단위 스레드 병렬) ──
    print("\n[2] Clustering + representatives")
    t2 = time.time()
    labels, X, reps = cluster_and_explain(emb, keys_sel, args.n_samples, args.seed, args.stat_workers)
    timing["cluster_sec"] = time.time() - t2
    K = len(reps)

    # ── 3) 근거 데이터 산출 (CSV) ────────────────────────────────
    print("\n[3] Explain + export")
    t3 = time.time()
    proj2d = PCA(n_components=2, random_state=args.seed).fit_transform(X)
    save_representatives_csv(reps, proj2d, os.path.join(args.out_dir, "representatives.csv"))

    # 전체 표본 대비 대표 세트의 품질(=대표성 신뢰도) 지표 — 표본 기반이라 저렴함
    s_idx = rng.choice(len(X), size=min(10000, len(X)), replace=False)
    try:
        sil = float(silhouette_score(X[s_idx], labels[s_idx]))
    except Exception:
        sil = float("nan")
    timing["stats_sec"] = time.time() - t3
    timing["total_sec"] = time.time() - t0

    summary = {
        "total_graphs_in_dataset": int(N),
        "graphs_used_for_clustering": int(len(sel)),
        "n_representatives_requested": args.n_samples,
        "n_representatives_actual": K,
        "coverage_pct_sum": float(sum(r["size_pct"] for r in reps)),   # ≈100 이어야 정상
        "embedding_dim": int(emb.shape[1]),
        "silhouette_sample_10000": sil,
        "timing_sec": timing,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    saved {os.path.join(args.out_dir, 'summary.json')}")
    print(f"    silhouette(sample)≈{sil:.3f}  (1에 가까울수록 대표 세트의 클러스터 분리가 뚜렷함)")

    # ── (옵션) 시각화 — visualize_encoder.py 의 플롯 재사용 ─────────
    if args.plot:
        print("\n[plot] visualize_encoder 플롯 재사용")
        plot_clusters(proj2d, labels, reps, args.out_dir, seed=args.seed)
        plot_cluster_quality(X, labels, reps, args.out_dir, seed=args.seed)
        plot_representative_patterns(reps, idx_map_sel, args.out_dir, top=12)

    print(f"\n완료 ({timing['total_sec']:.1f}s: load {timing['load_sec']:.1f}s / "
          f"embed {timing['embed_sec']:.1f}s / cluster {timing['cluster_sec']:.1f}s / "
          f"stats {timing['stats_sec']:.1f}s). 결과물: {args.out_dir}/")
    print("  embeddings.npz, representatives.csv, summary.json" + (" , *.png" if args.plot else ""))


if __name__ == "__main__":
    main()
