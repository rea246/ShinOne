"""
h0_clustering.py — h0 임베딩 HDBSCAN "첫 군집화" 확정 + 눈으로 검증

이 파일의 목적 (여기서 h0 군집을 확정하고, 이걸 기점으로 다음 분류를 설계):
  1) 내가 CONFIG 에 정한 값(mcs/ms/eps/method)으로 h0 를 HDBSCAN 1회 군집화
  2) 결과를 PCA-2D scatter 로 시각화 (군집 색 / noise 회색 / 군집 id 라벨)
  3) 각 군집의 대표 h0 패턴 5개(medoid 1 + 랜덤 4)를 실제 게이지로 렌더
  4) 분류 건강도 수치 출력 (DBCV / noise / 최대군집 / eff_K / 군집별 persistence)

argv 없음 — 아래 CONFIG 만 고쳐서 실행. core 는 HDBSCAN 이 전부 사용.
대표패턴 렌더는 원본 청크(PREPROCESSED_*part*.pt)가 FOLDER_PATH 에 있어야 함.

    python h0_clustering.py
"""

import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════
# CONFIG — 여기만 고치면 된다
# ══════════════════════════════════════════════════════════════════
_here       = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(_here, "hkeys_features.pt")
FOLDER_PATH = "C:/Users/rea24/Documents/shinwon_note/pattern/2.Pattern_Classification/3.CODE/pythonProject2/dummy_dataset"
OUT_DIR     = os.path.join(_here, "h0_clustering_out")

SAMPLE      = 300_000    # 군집화에 쓸 표본 수
REDUCE_DIM  = 4          # 표준화 후 PCA 축소 차원 (h0 effective rank ≈ 4)
SCATTER_MAX = 100_000    # 2D scatter 에 찍을 점 수 상한
SEED        = 42

# 내가 정한 군집화 조건 (지금까지 탐색으로 고른 값)
MIN_CLUSTER_SIZE = 1000
MIN_SAMPLES      = 20
EPSILON          = 0.0
CLUSTER_METHOD   = "leaf"     # 'leaf'(잘게) | 'eom'(안정 큰 군집)

N_REPS  = 5             # 군집당 대표 패턴 수 (medoid 1 + 랜덤 4)

HOP_COLORS = {0: "#e6194B", 1: "#f58231", 2: "#3cb44b", 3: "#9aa0a6"}


# ══════════════════════════════════════════════════════════════════
# 1. 임베딩 로드 + 표본 → 표준화 → PCA (군집화 4d / 시각화 2d)
# ══════════════════════════════════════════════════════════════════
def load_and_prepare():
    d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    h0, all_keys, chunks = d["features"]["h0"], d["keys"], d["chunks"]
    valid = np.where((h0.abs().sum(1) != 0).numpy())[0]      # 빈 hop 제외
    print(f"h0 임베딩: 전체 {h0.shape[0]:,}  유효 {len(valid):,}")

    rng = np.random.default_rng(SEED)
    samp_rows = (valid if len(valid) <= SAMPLE
                 else np.sort(rng.choice(valid, SAMPLE, replace=False)))
    X = h0[torch.from_numpy(samp_rows)].numpy().astype(np.float32)
    X = StandardScaler().fit_transform(X)                    # 8d 표준화
    print(f"표본 {len(samp_rows):,}개, {X.shape[1]}차원")

    if REDUCE_DIM < X.shape[1]:
        pca = PCA(n_components=REDUCE_DIM, random_state=SEED).fit(X)
        X_cluster = pca.transform(X).astype(np.float32)
        print(f"군집화 공간: PCA {X.shape[1]}d → {REDUCE_DIM}d "
              f"(설명분산 {pca.explained_variance_ratio_.sum():.0%})")
    else:
        X_cluster = X

    X_view2d = PCA(n_components=2, random_state=SEED).fit_transform(X)  # 시각화용 2D
    return X_cluster, X_view2d, samp_rows, all_keys, chunks, rng


# ══════════════════════════════════════════════════════════════════
# 2. HDBSCAN 1회 (내가 정한 조건)
# ══════════════════════════════════════════════════════════════════
def run_hdbscan(X_cluster):
    t = time.time()
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                             min_samples=MIN_SAMPLES,
                             cluster_selection_epsilon=float(EPSILON),
                             cluster_selection_method=CLUSTER_METHOD,
                             gen_min_span_tree=True,      # DBCV 위해
                             core_dist_n_jobs=-1)          # core 전부 사용
        lab = cl.fit_predict(X_cluster)
        dbcv = float(getattr(cl, "relative_validity_", float("nan")))
        persist = getattr(cl, "cluster_persistence_", None)
    except ImportError:
        print("  ⚠ hdbscan 패키지 없음 → sklearn 내장 사용 (DBCV·persistence 없음). "
              "pip install hdbscan 권장")
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        cl = SKHDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                       cluster_selection_epsilon=float(EPSILON),
                       cluster_selection_method=CLUSTER_METHOD, n_jobs=-1)
        lab = cl.fit_predict(X_cluster)
        dbcv, persist = float("nan"), None
    print(f"HDBSCAN fit 완료 ({time.time()-t:.0f}s)  "
          f"mcs={MIN_CLUSTER_SIZE} ms={MIN_SAMPLES} eps={EPSILON} method={CLUSTER_METHOD}")
    return lab, dbcv, persist


# ══════════════════════════════════════════════════════════════════
# 3. 분류 건강도 수치 출력
# ══════════════════════════════════════════════════════════════════
def print_health(lab, dbcv, persist):
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    n = len(lab)
    k = len(ids)
    noise = float((lab == -1).mean())
    max_share = float(cnts.max() / n) if k else 0.0
    # eff_K = 군집 크기 분포의 유효 개수 (한 군집 독식이면 1 에 가까움)
    p = cnts / cnts.sum() if k else np.array([1.0])
    eff_k = float(np.exp(-(p * np.log(p)).sum())) if k else 0.0

    print("\n" + "=" * 60)
    print("  분류 건강도 (h0 첫 군집화)")
    print("=" * 60)
    print(f"  군집 수 K        : {k}")
    print(f"  noise 비율       : {noise:.1%}    (낮을수록↑)")
    print(f"  DBCV             : {dbcv:.3f}    (높을수록↑, 밀도군집 품질)")
    print(f"  최대군집 비율    : {max_share:.1%}    (한 군집 독식이면 큼)")
    print(f"  유효 군집수 eff_K: {eff_k:.2f} / {k}  (군집 크기 균형; K 에 가까울수록 균형)")
    print(f"  군집별:")
    for j in np.argsort(-cnts):
        cid = int(ids[j])
        line = f"    c{cid:<2} {cnts[j]:>9,} ({cnts[j]/n:5.1%})"
        if persist is not None and cid < len(persist):
            line += f"   persistence={persist[cid]:.3f}"   # 클수록 안정/뚜렷한 군집
        print(line)
    print("=" * 60)
    return ids, cnts


# ══════════════════════════════════════════════════════════════════
# 4. PCA-2D scatter
# ══════════════════════════════════════════════════════════════════
def plot_scatter_2d(X_view2d, lab, rng):
    n = len(lab)
    draw = (np.arange(n) if n <= SCATTER_MAX
            else np.sort(rng.choice(n, SCATTER_MAX, replace=False)))
    P = X_view2d
    ids = np.unique(lab[lab >= 0])
    colors = cm.tab20(np.linspace(0, 1, max(len(ids), 1)))

    fig, ax = plt.subplots(figsize=(11, 9))
    nm = draw[lab[draw] < 0]
    ax.scatter(P[nm, 0], P[nm, 1], s=2, c="#cccccc", alpha=0.25,
               linewidths=0, label=f"noise ({(lab<0).sum():,})")
    for i, cid in enumerate(ids):
        m = draw[lab[draw] == cid]
        ax.scatter(P[m, 0], P[m, 1], s=3, color=colors[i], alpha=0.5, linewidths=0)
        allm = np.where(lab == cid)[0]
        ax.text(np.median(P[allm, 0]), np.median(P[allm, 1]),
                f"c{cid}\n{len(allm):,}", fontsize=9, ha="center", va="center",
                weight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                         ec=colors[i], alpha=0.85))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"h0 HDBSCAN | K={len(ids)}  noise={(lab<0).mean():.1%}  "
                 f"(PCA-2D, 표본 {n:,})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "h0_scatter_2d.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  [scatter] {path}")


# ══════════════════════════════════════════════════════════════════
# 5. 각 군집 대표 h0 패턴 5개 렌더  (h0_hdbscan_umap.py 로직 재사용)
# ══════════════════════════════════════════════════════════════════
def _row_to_file(chunks, folder, row):
    off = 0
    for base, cnt in chunks:
        if row < off + cnt:
            return os.path.join(folder, base), row - off
        off += cnt
    return None, None


def _extract_graphs(chunks, folder, rows):
    """global row 집합 → {row: {'x','hop'}}. 청크 동시 1개만 로드."""
    byfile = {}
    for r in set(int(v) for v in rows):
        f, li = _row_to_file(chunks, folder, r)
        if f is not None:
            byfile.setdefault(f, []).append((r, li))
    out = {}
    for f, pairs in byfile.items():
        if not os.path.exists(f):
            continue
        ch = torch.load(f, map_location="cpu", weights_only=False)
        for r, li in pairs:
            x0, x1, _, _ = [int(v) for v in ch["meta"][li].tolist()]
            x = ch["x"][x0:x1].float().numpy()
            packed = ch["hop_packed"][x0:x1].numpy().astype(np.int64)
            hop = np.full(len(packed), 3, dtype=int)
            hop[((packed >> 2) & 1).astype(bool)] = 2
            hop[((packed >> 1) & 1).astype(bool)] = 1
            hop[(packed & 1).astype(bool)] = 0
            out[r] = {"x": x, "hop": hop}
        del ch
    return out


def _draw_pattern(ax, g, key):
    """h0 노드 색 강조 + 나머지 hop 회색 윤곽."""
    x, hop = g["x"], g["hop"]
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    for i in np.where(hop != 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               fill=False, edgecolor="#cccccc", lw=0.4))
    for i in np.where(hop == 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               facecolor=HOP_COLORS[0], edgecolor="black",
                               lw=0.5, alpha=0.8))
    h0n = np.where(hop == 0)[0]
    if len(h0n):
        ax.plot(cx[h0n], cy[h0n], "*", color="black", ms=10,
                markerfacecolor=HOP_COLORS[0], zorder=4)
    if len(x):
        x0b = float((cx - w / 2).min()); x1b = float((cx + w / 2).max())
        y0b = float((cy - h / 2).min()); y1b = float((cy + h / 2).max())
        pad = 0.05 * max(x1b - x0b, y1b - y0b, 1.0)
        ax.set_xlim(x0b - pad, x1b + pad); ax.set_ylim(y0b - pad, y1b + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(key, fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.grid(True, ls=":", alpha=0.3)


def plot_representatives(lab, X_cluster, samp_rows, all_keys, chunks, rng):
    """군집마다 medoid 1 + 랜덤 4 = 5개 대표를 실제 h0 패턴으로 렌더."""
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    if len(ids) == 0:
        print("  ⚠ 군집이 없어 대표패턴 생략 (전부 noise)")
        return
    order = np.argsort(-cnts)
    groups = [(int(ids[j]), int(cnts[j])) for j in order]

    reps = []   # {cluster, size, rank, row}
    for cid, size in groups:
        members = np.where(lab == cid)[0]                 # 표본-로컬 idx
        d = np.linalg.norm(X_cluster[members] - X_cluster[members].mean(0), axis=1)
        medoid = members[int(np.argmin(d))]               # 중심에 가장 가까운 = 대표
        pool = members[members != medoid]
        extra = (rng.choice(pool, min(N_REPS - 1, len(pool)), replace=False)
                 if len(pool) else np.array([], dtype=int))
        for rank, m in enumerate([medoid] + list(extra)):
            reps.append({"cluster": cid, "size": size, "rank": rank,
                         "row": int(samp_rows[m])})

    graphs = _extract_graphs(chunks, FOLDER_PATH, [r["row"] for r in reps])
    by_cluster = {}
    for r in reps:
        by_cluster.setdefault(r["cluster"], []).append(r)

    nrow = len(groups)
    fig, axes = plt.subplots(nrow, N_REPS, figsize=(3.2 * N_REPS, 2.9 * nrow),
                             squeeze=False)
    for r_i, (cid, size) in enumerate(groups):
        rs = by_cluster.get(cid, [])
        for c in range(N_REPS):
            ax = axes[r_i][c]
            if c >= len(rs) or graphs.get(rs[c]["row"]) is None:
                ax.axis("off"); continue
            tag = "medoid" if rs[c]["rank"] == 0 else "member"
            key = str(all_keys[rs[c]["row"]])
            _draw_pattern(ax, graphs[rs[c]["row"]],
                          f"c{cid} n={size:,} [{tag}]\n{key}")
    fig.suptitle("h0 군집별 대표 패턴 (h0 노드=빨강+별, 다른 hop=회색 윤곽)\n"
                 "행 내부는 비슷 / 행 간은 달라야 = 군집이 진짜", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "h0_representatives.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [대표패턴] {path}")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    X_cluster, X_view2d, samp_rows, all_keys, chunks, rng = load_and_prepare()

    lab, dbcv, persist = run_hdbscan(X_cluster)
    print_health(lab, dbcv, persist)

    plot_scatter_2d(X_view2d, lab, rng)
    plot_representatives(lab, X_cluster, samp_rows, all_keys, chunks, rng)

    # 라벨 저장 (다음 분류 단계의 입력으로 쓸 수 있게)
    np.savez(os.path.join(OUT_DIR, "h0_labels.npz"),
             rows=samp_rows, labels=lab)
    print(f"\n완료 ({time.time()-t0:.0f}s). 산출물: {OUT_DIR}/")
    print("  h0_scatter_2d.png / h0_representatives.png / h0_labels.npz")


if __name__ == "__main__":
    main()
