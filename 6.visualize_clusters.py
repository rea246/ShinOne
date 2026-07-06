"""
6.visualize_clusters.py — 클러스터별 실제 레이아웃 패턴 시각화 (논문 figure 용)

5.classify_tree.py 결과(레벨별 key)를 받아, 각 key 로 분류된 그래프의 실제 기하
(노드 bbox 사각형 + 엣지 연결선, hop 별 색)를 그린다.

사용법 (Config 만 바꿔 재실행):
  LEVEL='h0', PREFIX={}              → h0 의 모든 key 별 시각화
  LEVEL='h1', PREFIX={'h0': 3}       → h0=3 안에서 h1 key 별 (드릴다운)
  LEVEL='h2', PREFIX={'h0':3,'h1':1} → 계속 이어가기

출력 (OUT_DIR 아래 PNG):
  overview_<...>.png       : key 당 대표 1개 그리드 — 논문 메인 figure 용
  <LEVEL>_key<k>_<...>.png : key 별 예시 N_EXAMPLES 개
                             (centroid 최근접 절반 = 대표성, 무작위 절반 = 다양성)

메모리: 선택된 대표 그래프가 속한 원본 청크만 1개씩 로드 (동시 ≈2GB).
전제: 5.classify_tree.py 실행 완료 (feature 캐시 + labels + kmeans 산출물 필요).
"""

import os
import time
import importlib.util

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ══════════════════════════════════════════════════════════════════
# [Config]
# ══════════════════════════════════════════════════════════════════
LEVEL      = 'h0'     # 시각화할 레벨
PREFIX     = {}       # 드릴다운 조건. 예: {'h0': 3} / {'h0': 3, 'h1': 1}
N_EXAMPLES = 6        # key 당 예시 수 (그리드 2×3)
OUT_DIR    = "viz"
DPI        = 150
SEED       = 0
HALF_WIN   = 300.0    # 표시 범위 ±nm (전처리 window 0.512µm → half 256nm + 여유)

# hop 별 색 (h0=center 강조)
HOP_COLORS = {0: '#d62728', 1: '#ff7f0e', 2: '#2ca02c', 3: '#bdbdbd'}

# ══════════════════════════════════════════════════════════════════
# [입력 로드 — 5.classify_tree.py 의 경로/캐시 설정 재사용]
# ══════════════════════════════════════════════════════════════════
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "classify_tree", os.path.join(_here, "5.classify_tree.py"))
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)


def load_inputs():
    cache = torch.load(os.path.join(_here, ct.FEATURE_CACHE),
                       map_location='cpu', weights_only=False)
    lab = torch.load(os.path.join(_here, f"{ct.OUT_PREFIX}_labels.pt"),
                     map_location='cpu', weights_only=False)
    km = torch.load(os.path.join(_here, f"{ct.OUT_PREFIX}_kmeans.pt"),
                    map_location='cpu', weights_only=False)
    labels = {lv: v.numpy().astype(np.int64) for lv, v in lab['labels'].items()}
    return cache['features'], lab['keys'], cache['chunks'], labels, km


# ══════════════════════════════════════════════════════════════════
# [global index → (청크, 로컬 index) 역참조 + 원본 기하 로드]
# ══════════════════════════════════════════════════════════════════
def fetch_graphs(global_indices, chunks):
    """선택 그래프들의 원본 기하 로드. 청크별로 묶어 1개씩만 메모리에 올린다."""
    starts = np.cumsum([0] + [c for _, c in chunks])
    by_chunk = {}
    for g in global_indices:
        ci = int(np.searchsorted(starts, g, side='right') - 1)
        by_chunk.setdefault(ci, []).append(int(g))

    out = {}
    for ci, gids in by_chunk.items():
        fname = os.path.join(ct.FOLDER_PATH, chunks[ci][0])
        d = torch.load(fname, map_location='cpu', weights_only=False)
        meta = d['meta']
        for g in gids:
            li = g - int(starts[ci])
            x0, x1, e0, e1 = meta[li].tolist()
            packed = d['hop_packed'][x0:x1].numpy()
            hop = np.select(
                [packed & 1, (packed >> 1) & 1, (packed >> 2) & 1],
                [0, 1, 2], default=3)
            out[g] = {
                'x':    d['x'][x0:x1].float().numpy(),        # (n,4) [cx,cy,w,h] nm
                'ei':   d['edge_index'][:, e0:e1].numpy(),     # (2,e) 로컬, 양방향
                'hop':  hop,
                'name': d['keys'][li],
            }
        del d
    return out


# ══════════════════════════════════════════════════════════════════
# [그리기 / 대표 선정]
# ══════════════════════════════════════════════════════════════════
def draw_graph(ax, g):
    x = g['x']
    if g['ei'].size:                              # 양방향 저장 → u<v 만 (중복 제거)
        u, v = g['ei']
        for a, b in zip(u[u < v], v[u < v]):
            ax.plot([x[a, 0], x[b, 0]], [x[a, 1], x[b, 1]],
                    color='#999999', lw=0.7, ls='--', zorder=1)
    for i in range(len(x)):
        cx, cy, w, h = x[i]
        c = HOP_COLORS[int(g['hop'][i])]
        if w <= 0 or h <= 0:                      # 합성 노드(T2T/Bridge) — × 마커
            ax.plot(cx, cy, marker='x', color=c, ms=8, mew=2, zorder=3)
        else:
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                                   facecolor=c, edgecolor='black',
                                   lw=0.5, alpha=0.65, zorder=2))
    ax.set_xlim(-HALF_WIN, HALF_WIN)
    ax.set_ylim(-HALF_WIN, HALF_WIN)
    ax.set_aspect('equal')
    ax.set_xticks([]);  ax.set_yticks([])
    ax.set_title(str(g['name']), fontsize=6)


def select_examples(cand, feat, centroid, n, rng):
    """centroid 최근접 절반(대표성) + 나머지 무작위(다양성).
    feat 는 fp16 텐서 그대로 받아 후보 행만 float32 변환 (전량 변환 회피)."""
    if centroid is None or len(cand) <= n:
        return rng.permutation(cand)[:n]
    Xc = feat[torch.from_numpy(cand)].numpy().astype(np.float32)
    order = np.argsort(np.linalg.norm(Xc - centroid[None, :], axis=1))
    n_near = max(1, n // 2)
    near, pool = cand[order[:n_near]], cand[order[n_near:]]
    rand = rng.choice(pool, min(n - n_near, len(pool)), replace=False)
    return np.concatenate([near, rand])


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(_here, OUT_DIR), exist_ok=True)

    features, keys, chunks, labels, km = load_inputs()
    n_graphs = len(keys)
    feat_lv = features[LEVEL]                     # fp16 유지 — 후보 행만 변환
    centroids = km['centroids'].get(LEVEL)

    mask = np.ones(n_graphs, dtype=bool)          # PREFIX 필터 (드릴다운)
    for lv, k in PREFIX.items():
        mask &= (labels[lv] == k)
    prefix_tag = "_".join(f"{lv}{k}" for lv, k in PREFIX.items()) or "all"
    print(f"[Filter] PREFIX={PREFIX} → 대상 {int(mask.sum()):,}/{n_graphs:,} 그래프")

    lv_labels = labels[LEVEL]
    present = sorted(set(lv_labels[mask].tolist()))
    print(f"[Level {LEVEL}] 관측 key: {present}")

    picks = {}
    for k in present:
        cand = np.where(mask & (lv_labels == k))[0]
        cent = centroids[k] if (centroids is not None and 0 <= k < len(centroids)) else None
        picks[k] = select_examples(cand, feat_lv, cent, N_EXAMPLES, rng)

    all_gids = np.concatenate(list(picks.values()))
    print(f"[Load] 대표 그래프 {len(all_gids)}개 원본 로드 중...")
    graphs = fetch_graphs(all_gids, chunks)

    # ── key 별 상세 figure ────────────────────────────────────────
    tot = int(mask.sum())
    for k in present:
        gids = picks[k]
        ncols = min(3, len(gids));  nrows = int(np.ceil(len(gids) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(gids):]:
            ax.axis('off')
        for ax, g in zip(axes, gids):
            draw_graph(ax, graphs[g])
        cnt = int((mask & (lv_labels == k)).sum())
        klabel = f"key {k}" if k >= 0 else f"key {k} (빈 hop)"
        fig.suptitle(f"[{prefix_tag}]  {LEVEL} {klabel}  —  n={cnt:,} ({cnt/tot:.1%})",
                     fontsize=11)
        out = os.path.join(_here, OUT_DIR, f"{LEVEL}_key{k}_{prefix_tag}.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out, dpi=DPI);  plt.close(fig)
        print(f"  [Saved] {out}")

    # ── overview: key 당 대표 1개 (논문 메인 figure) ──────────────
    ncols = min(5, len(present));  nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.9 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(present):]:
        ax.axis('off')
    for ax, k in zip(axes, present):
        draw_graph(ax, graphs[picks[k][0]])       # centroid 최근접 1개
        cnt = int((mask & (lv_labels == k)).sum())
        ax.set_title(f"{LEVEL}={k}  (n={cnt:,}, {cnt/tot:.1%})", fontsize=8)
    fig.suptitle(f"{LEVEL} clusters  [{prefix_tag}]   "
                 f"(색: h0=빨강 h1=주황 h2=초록 h3=회색)", fontsize=11)
    out = os.path.join(_here, OUT_DIR, f"overview_{LEVEL}_{prefix_tag}.png")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=DPI);  plt.close(fig)
    print(f"  [Saved] {out}")

    print(f"\nTotal Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
