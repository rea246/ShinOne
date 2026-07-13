"""
h0_analyze.py — h0_labels_full.npz 열어서 군집별 w/h 분포·PCA 분석 플롯

h0_clustering.py 가 만든 전량 라벨(full.npz) + 캐시(hkeys_features.pt)의
중심 raw [w,h]·h0 임베딩을 합쳐서:
  1) 군집별 w, h 히스토그램          → h0_wh_hist.png
  2) 군집별 w, h 확률분포(KDE)       → h0_wh_kde.png
  3) PCA 축(PC1,PC2) vs w,h 산점도    → h0_pca_wh_scatter.png
  4) (보너스) w vs h 기하 산점도       → h0_wh_scatter.png

argv 없음 — 아래 CONFIG 만 고쳐 실행. 의존: seaborn, pandas.

    python h0_analyze.py
"""

import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
_here      = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_here, "hkeys_features.pt")
LABELS_NPZ = os.path.join(_here, "h0_clustering_out", "h0_labels_full.npz")
OUT_DIR    = os.path.join(_here, "h0_clustering_out")

CLUSTERS    = None        # None = 전체 군집, 또는 [0, 1, 2, 3, 4] 처럼 지정
PER_CLUSTER = 40_000      # 군집당 플롯 표본 상한 (분포 비교용 — 큰 군집이 압도 안 하게)
SEED        = 42

# h0_clustering.py 와 동일한 검증된 팔레트(고정 순서)
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]


# ══════════════════════════════════════════════════════════════════
# 로드 + df 구성 (군집당 균형 표본 → w,h,PC1,PC2)
# ══════════════════════════════════════════════════════════════════
def load_df():
    z = np.load(LABELS_NPZ)
    rows, labels = z["rows"], z["labels"]              # 전체 유효 global row, 군집
    d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    h0, h0_raw = d["features"]["h0"], d["features"].get("h0_raw")
    if h0_raw is None:
        raise SystemExit("캐시에 h0_raw([w,h]) 없음 — w,h 분석 불가")

    rng = np.random.default_rng(SEED)
    clusters = (np.unique(labels[labels >= 0]) if CLUSTERS is None
                else np.asarray(CLUSTERS))
    sel = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        sel.append(idx if len(idx) <= PER_CLUSTER
                   else rng.choice(idx, PER_CLUSTER, replace=False))
    sel = np.sort(np.concatenate(sel))

    r = rows[sel]
    wh = h0_raw[torch.from_numpy(r)].numpy().astype(np.float64)          # [w,h]
    Xs = StandardScaler().fit_transform(
        h0[torch.from_numpy(r)].numpy().astype(np.float32))              # 표준화 8d
    P2 = PCA(n_components=2, random_state=SEED).fit_transform(Xs)        # 시각화 PC1,PC2
    df = pd.DataFrame(dict(cluster=[f"c{int(v)}" for v in labels[sel]],
                           w=wh[:, 0], h=wh[:, 1], PC1=P2[:, 0], PC2=P2[:, 1]))
    print(f"플롯 표본 {len(df):,}  (군집 {len(clusters)}개, 군집당 ≤{PER_CLUSTER:,})")
    return df


def _order_pal(df):
    cl = sorted(df["cluster"].unique(), key=lambda s: int(s[1:]))
    return cl, {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cl)}


# ══════════════════════════════════════════════════════════════════
# 플롯들
# ══════════════════════════════════════════════════════════════════
def plot_hist(df):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col in zip(axes, ["w", "h"]):
        sns.histplot(df, x=col, hue="cluster", hue_order=cl, palette=pal,
                     element="step", stat="count", bins=60, alpha=0.25,
                     ax=ax, legend=(col == "h"))
        ax.set_title(f"{col} histogram by cluster")
        sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_hist.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [hist] {p}")


def plot_kde(df):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col in zip(axes, ["w", "h"]):
        sns.kdeplot(df, x=col, hue="cluster", hue_order=cl, palette=pal,
                    fill=True, alpha=0.25, common_norm=False, ax=ax,
                    legend=(col == "h"))
        ax.set_title(f"{col} density by cluster")
        sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_kde.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [kde] {p}")


def plot_pca_wh(df):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    pairs = [("PC1", "w"), ("PC1", "h"), ("PC2", "w"), ("PC2", "h")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (x, y) in zip(axes.flat, pairs):
        for c in cl:
            dd = df[df["cluster"] == c]
            ax.scatter(dd[x], dd[y], s=4, color=pal[c], alpha=0.4,
                       linewidths=0, label=c)
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
        sns.despine(ax=ax)
    axes.flat[0].legend(markerscale=3, fontsize=9, frameon=False, title="cluster")
    fig.suptitle("PCA axes vs center geometry (w, h) by cluster",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_pca_wh_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [pca-wh] {p}")


def plot_wh(df):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in cl:
        dd = df[df["cluster"] == c]
        ax.scatter(dd["w"], dd["h"], s=4, color=pal[c], alpha=0.4,
                   linewidths=0, label=c)
    ax.set_xlabel("w"); ax.set_ylabel("h")
    ax.set_title("center geometry: w vs h by cluster")
    ax.legend(markerscale=3, fontsize=10, frameon=False, title="cluster")
    sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [w-h] {p}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_df()
    plot_hist(df)
    plot_kde(df)
    plot_pca_wh(df)
    plot_wh(df)
    print(f"완료. 산출물: {OUT_DIR}/  "
          "(h0_wh_hist / h0_wh_kde / h0_pca_wh_scatter / h0_wh_scatter)")


if __name__ == "__main__":
    main()
