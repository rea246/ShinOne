"""
hdbscan_search.py — h0 임베딩으로 HDBSCAN 조건 탐색만 (K·noise 눈으로 보고 결정)

하는 일 (딱 이것만):
  1) hkeys_features.pt 캐시에서 h0(8d) 임베딩을 읽는다
  2) 표준화 → PCA(REDUCE_DIM)로 축소한다 (h0 는 effective rank ≈ 4 라 4d 권장)
  3) 아래 CONDITIONS 에 적은 (mcs, min_samples, eps) 마다 HDBSCAN 을 돌린다
  4) 조건마다 군집수 K / noise 비율을 콘솔에 찍고, PCA-3D scatter 로 나란히 그린다
  5) "최적 조건"은 K 와 noise 를 보고 사람이 직접 고른다 (자동 추천 없음)

argv 안 씀 — 아래 CONFIG 변수만 고쳐서 그냥 실행하면 된다.
core 는 HDBSCAN 이 자동으로 전부 쓴다(core_dist_n_jobs=-1).

    python hdbscan_search.py
"""

import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════
# CONFIG — 여기만 고치면 된다
# ══════════════════════════════════════════════════════════════════
_here       = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(_here, "hkeys_features.pt")
OUT_PNG     = os.path.join(_here, "hdbscan_search.png")

SAMPLE      = 300_000    # 군집화에 쓸 표본 수 (전량 20M 은 너무 커서 불가)
REDUCE_DIM  = 4          # 표준화 후 PCA 축소 차원 (8 로 두면 축소 안 함)
SCATTER_MAX = 50_000     # 3D scatter 에 실제로 찍을 점 수 (많으면 렌더 느림)
SEED        = 42

# 탐색할 HDBSCAN 조건들 — 줄을 추가/수정/삭제해서 원하는 만큼 넣으면 된다.
#   min_cluster_size : 군집 최소 크기 (작을수록 군집 많고 잘게 쪼개짐)
#   min_samples      : 밀도 보수성 (None=mcs 와 동일, 작을수록 noise 줄고 군집 넓어짐)
#   epsilon          : 군집 경계 확장 거리 (0 이면 확장 없음)
CONDITIONS = [
    dict(min_cluster_size=300,  min_samples=None, epsilon=0.0),
    dict(min_cluster_size=500,  min_samples=None, epsilon=0.0),
    dict(min_cluster_size=500,  min_samples=100,  epsilon=0.0),
    dict(min_cluster_size=750,  min_samples=None, epsilon=0.0),
    dict(min_cluster_size=1000, min_samples=None, epsilon=0.0),
    dict(min_cluster_size=1000, min_samples=100,  epsilon=0.0),
]


# ══════════════════════════════════════════════════════════════════
# 1. 임베딩 로드 + 표본 → 표준화 → 축소
# ══════════════════════════════════════════════════════════════════
def load_and_prepare():
    d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    h0 = d["features"]["h0"]                              # (N, 8) fp16
    valid = np.where((h0.abs().sum(1) != 0).numpy())[0]   # 빈 hop(0벡터) 제외
    print(f"h0 임베딩: 전체 {h0.shape[0]:,}  유효 {len(valid):,}")

    rng = np.random.default_rng(SEED)
    rows = (valid if len(valid) <= SAMPLE
            else np.sort(rng.choice(valid, SAMPLE, replace=False)))
    X = h0[torch.from_numpy(rows)].numpy().astype(np.float32)
    X = StandardScaler().fit_transform(X)                 # 8d 표준화
    print(f"표본 {len(rows):,}개, {X.shape[1]}차원")

    # 군집화 공간: PCA 축소 (REDUCE_DIM==8 이면 그대로 8d)
    if REDUCE_DIM < X.shape[1]:
        pca = PCA(n_components=REDUCE_DIM, random_state=SEED).fit(X)
        X_cluster = pca.transform(X).astype(np.float32)
        evr = pca.explained_variance_ratio_.sum()
        print(f"군집화 공간: PCA {X.shape[1]}d → {REDUCE_DIM}d (설명분산 {evr:.0%})")
    else:
        X_cluster = X
        print(f"군집화 공간: 축소 없음 ({X.shape[1]}d)")

    # 시각화 공간: 항상 PCA-3D (군집 결과를 3D 로 눈으로 보기 위함)
    X_view3d = PCA(n_components=3, random_state=SEED).fit_transform(X)
    return X_cluster, X_view3d, rng


# ══════════════════════════════════════════════════════════════════
# 2. HDBSCAN 한 조건 실행 → 라벨, K, noise
# ══════════════════════════════════════════════════════════════════
def run_hdbscan(X_cluster, cond):
    t = time.time()
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=cond["min_cluster_size"],
                             min_samples=cond["min_samples"],
                             cluster_selection_epsilon=float(cond["epsilon"]),
                             core_dist_n_jobs=-1)          # core 전부 사용
    except ImportError:
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        cl = SKHDBSCAN(min_cluster_size=cond["min_cluster_size"],
                       min_samples=cond["min_samples"],
                       cluster_selection_epsilon=float(cond["epsilon"]),
                       n_jobs=-1)
    lab = cl.fit_predict(X_cluster)
    k = len(set(lab) - {-1})                # noise(-1) 제외한 군집 수
    noise = float((lab == -1).mean())
    print(f"  mcs={cond['min_cluster_size']:>5}  ms={str(cond['min_samples']):>5}  "
          f"eps={cond['epsilon']:<4} →  K={k:>3}   noise={noise:6.1%}   "
          f"({time.time()-t:.0f}s)")
    return lab, k, noise


# ══════════════════════════════════════════════════════════════════
# 3. PCA-3D scatter 격자 (조건별로 나란히 비교)
# ══════════════════════════════════════════════════════════════════
def plot_grid(X_view3d, results, rng):
    n = len(X_view3d)
    draw = (np.arange(n) if n <= SCATTER_MAX
            else rng.choice(n, SCATTER_MAX, replace=False))
    P = X_view3d[draw]

    ncol = min(len(results), 3)
    nrow = (len(results) + ncol - 1) // ncol
    fig = plt.figure(figsize=(6 * ncol, 5.2 * nrow))

    for i, (cond, lab, k, noise) in enumerate(results):
        L = lab[draw]
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="3d")
        # noise 는 회색으로 옅게
        m = L == -1
        ax.scatter(P[m, 0], P[m, 1], P[m, 2], s=2, c="#cccccc", alpha=0.15)
        # 군집은 색깔별로
        ids = sorted(set(L) - {-1})
        colors = cm.tab20(np.linspace(0, 1, max(len(ids), 1)))
        for j, cid in enumerate(ids):
            mm = L == cid
            ax.scatter(P[mm, 0], P[mm, 1], P[mm, 2], s=3, color=colors[j], alpha=0.5)
        ax.set_title(f"mcs={cond['min_cluster_size']}  ms={cond['min_samples']}  "
                     f"eps={cond['epsilon']}\nK={k}   noise={noise:.0%}", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=20, azim=45)

    fig.suptitle("HDBSCAN 조건 탐색 — K 와 noise 를 보고 조건을 고르세요", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"\n그림 저장: {OUT_PNG}")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    X_cluster, X_view3d, rng = load_and_prepare()

    print(f"\nHDBSCAN {len(CONDITIONS)}개 조건 실행:")
    results = []
    for cond in CONDITIONS:
        lab, k, noise = run_hdbscan(X_cluster, cond)
        results.append((cond, lab, k, noise))

    plot_grid(X_view3d, results, rng)
    print(f"\n완료 ({time.time()-t0:.0f}s). "
          f"K·noise 보고 CONDITIONS 에서 원하는 조건을 고르면 됩니다.")


if __name__ == "__main__":
    main()
