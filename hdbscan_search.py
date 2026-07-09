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
#   min_samples      : 밀도 보수성 (작을수록 noise 줄고 빠름). ⚠ None 으로 두면
#                      자동으로 min_cluster_size 와 같아져서, mcs 가 크면(500~1000)
#                      각 점의 500~1000번째 최근접을 계산하느라 매우 느려지고
#                      noise 도 폭증한다. 항상 작은 고정값(10~25) 권장.
#   epsilon          : 군집 "병합" 임계 거리 (0=병합 안 함). ⚠ K 를 줄이는 손잡이가
#                      아니다 — 키우면 가까운 군집을 다 합쳐 9→2 처럼 붕괴한다.
#                      K 를 줄이려면 eps 말고 min_cluster_size 를 키워라. eps 는 0 권장.
MIN_SAMPLES = 15         # 모든 조건 공통 밀도 보수성 (작을수록 빠르고 noise ↓)
CONDITIONS = [
    dict(min_cluster_size=300,  min_samples=MIN_SAMPLES, epsilon=0.0),
    dict(min_cluster_size=400,  min_samples=MIN_SAMPLES, epsilon=0.0),
    dict(min_cluster_size=500,  min_samples=MIN_SAMPLES, epsilon=0.0),
    dict(min_cluster_size=600,  min_samples=MIN_SAMPLES, epsilon=0.0),
    dict(min_cluster_size=750,  min_samples=MIN_SAMPLES, epsilon=0.0),
    dict(min_cluster_size=1000, min_samples=MIN_SAMPLES, epsilon=0.0),
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
# 2. HDBSCAN 한 조건 실행 → 라벨 + 지표(K / noise / DBCV / 최대군집비율)
# ══════════════════════════════════════════════════════════════════
#   DBCV      : 밀도 군집 전용 검증 점수(높을수록 좋음). "명쾌한 지표".
#               hdbscan 의 relative_validity_ (gen_min_span_tree=True 필요, 거의 공짜).
#   max_share : 최대 군집이 전체 표본에서 차지하는 비율. 한 군집이 대부분을 삼키면
#               (eps 를 키워 9→2 로 뭉갠 경우처럼) 이 값이 치솟음 → 붕괴 즉시 포착.
def run_hdbscan(X_cluster, cond):
    # min_samples=None 이면 hdbscan 이 내부적으로 mcs 로 바꿈 → 크면 core distance 폭증
    eff_ms = (cond["min_cluster_size"] if cond["min_samples"] is None
              else cond["min_samples"])
    if eff_ms > 100:
        print(f"  ⚠ 실효 min_samples={eff_ms} 로 큼 → 느려집니다 (작은 고정값 10~25 권장)")

    t = time.time()
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=cond["min_cluster_size"],
                             min_samples=cond["min_samples"],
                             cluster_selection_epsilon=float(cond["epsilon"]),
                             gen_min_span_tree=True,      # DBCV(relative_validity_) 위해 필요
                             core_dist_n_jobs=-1)          # core distance/MST 는 전 코어 사용
        lab = cl.fit_predict(X_cluster)
        dbcv = float(getattr(cl, "relative_validity_", float("nan")))
    except ImportError:                                   # 폴백: DBCV 미제공 → nan
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        cl = SKHDBSCAN(min_cluster_size=cond["min_cluster_size"],
                       min_samples=cond["min_samples"],
                       cluster_selection_epsilon=float(cond["epsilon"]),
                       n_jobs=-1)
        lab = cl.fit_predict(X_cluster)
        dbcv = float("nan")

    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)   # noise(-1) 제외
    k = len(ids)                                               # 군집 수
    noise = float((lab == -1).mean())
    max_share = float(cnts.max() / len(lab)) if k else 0.0     # 전체 대비 최대 군집
    print(f"  mcs={cond['min_cluster_size']:>5}  ms={str(cond['min_samples']):>5}  "
          f"eps={cond['epsilon']:<4} →  K={k:>3}   noise={noise:6.1%}   "
          f"DBCV={dbcv:6.3f}   최대군집={max_share:5.1%}   ({time.time()-t:.0f}s)")
    return lab, k, noise, dbcv, max_share


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

    for i, (cond, lab, k, noise, dbcv, max_share) in enumerate(results):
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
                     f"eps={cond['epsilon']}\nK={k}  noise={noise:.0%}  "
                     f"DBCV={dbcv:.3f}  최대군집={max_share:.0%}", fontsize=9)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.view_init(elev=20, azim=45)

    fig.suptitle("HDBSCAN 조건 탐색 — DBCV(높을수록↑)·noise(낮을수록↑)·최대군집(독식 아님) 으로 판단",
                 fontsize=12)
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
        lab, k, noise, dbcv, max_share = run_hdbscan(X_cluster, cond)
        results.append((cond, lab, k, noise, dbcv, max_share))

    plot_grid(X_view3d, results, rng)

    # ── DBCV 내림차순 정렬표 (자동 선택은 안 함 — 최종 판단은 사람이) ──
    print("\n" + "=" * 64)
    print("  DBCV 순위표 (DBCV↑ 좋음 / noise↓ 좋음 / 최대군집 과대=한 군집 독식)")
    print("=" * 64)
    print(f"  {'mcs':>5} {'ms':>5} {'eps':>4} | {'K':>3} {'noise':>6} "
          f"{'DBCV':>7} {'최대군집':>7}")
    for cond, lab, k, noise, dbcv, max_share in sorted(
            results, key=lambda r: (float('-inf') if np.isnan(r[4]) else r[4]),
            reverse=True):
        print(f"  {cond['min_cluster_size']:>5} {str(cond['min_samples']):>5} "
              f"{cond['epsilon']:>4} | {k:>3} {noise:>6.1%} {dbcv:>7.3f} {max_share:>7.1%}")
    print("=" * 64)
    print(f"\n완료 ({time.time()-t0:.0f}s). 위 표에서 DBCV 높고 noise 낮고 "
          f"최대군집이 과하지 않은 조건을 고르세요.")


if __name__ == "__main__":
    main()
