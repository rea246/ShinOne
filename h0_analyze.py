"""
h0_analyze.py — h0_labels_full.npz 열어서 군집별 w/h 분포·PCA 분석 플롯

h0_clustering.py 가 만든 전량 라벨(full.npz) + 캐시(hkeys_features.pt)의
중심 raw [w,h]·h0 임베딩을 합쳐서:
  1) 군집별 w, h 히스토그램          → h0_wh_hist.png
  2) 군집별 w, h 확률분포(KDE)       → h0_wh_kde.png
  3) PCA 축(PC1,PC2) vs w,h 산점도    → h0_pca_wh_scatter.png
  4) (보너스) w vs h 기하 산점도       → h0_wh_scatter.png

핵심: h0 임베딩엔 w,h 가 이미 녹아 있어 PCA 축이 w,h 와 상관된다. 그래서
  RESIDUALIZE_WH=True 면 임베딩에서 [w,h] 선형성분을 회귀로 제거한 뒤 PCA →
  PCA 축이 w,h 와 독립이 되어 "w,h 로 설명 안 되는 구조"만 본다.

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
PER_CLUSTER = 40_000      # 군집당 플롯 표본 상한 (큰 군집이 분포를 압도 안 하게)
SEED        = 42

# PCA 축에서 w,h 성분 제거 여부. True = w,h 와 독립인 임베딩 구조만 PCA.
RESIDUALIZE_WH = True

# 축 범위 — None = 자동(데이터 0.5~99.5 분위, 이상치에 안 늘어남).
#   한 번 그려보고 아래를 (min, max) 로 바꿔 직접 조정하면 됨. 예: W_RANGE = (0, 20)
W_RANGE   = None
H_RANGE   = None
PC1_RANGE = None
PC2_RANGE = None

# 팔레트 (전부 validator 통과, CVD-safe). reference | okabe | jewel
PALETTES = {
    # 현재 h0_clustering.py 와 동일 (파랑·아쿠아·노랑·초록·보라·빨강·마젠타·주황)
    "reference": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                  "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    # Okabe-Ito 계열 (색약 안전 클래식, 부드러움)
    "okabe":     ["#0072B2", "#009E73", "#E69F00", "#D55E00",
                  "#CC79A7", "#56B4E9", "#e34948", "#7a5da8"],
    # jewel/bold (깊고 선명, 대비 최고 — scatter 에 또렷함) ← 추천
    "jewel":     ["#1b6ca8", "#158a6b", "#d98b1e", "#c1440e",
                  "#6a3fb0", "#c42348", "#2f8f4e", "#b0731f"],
}
PALETTE_NAME = "jewel"
PALETTE = PALETTES[PALETTE_NAME]

# 밝은 배경 잉크
INK, INK2 = "#0b0b0b", "#52514e"


# ══════════════════════════════════════════════════════════════════
# 로드 + df 구성
# ══════════════════════════════════════════════════════════════════
def residualize_wh(X8, wh):
    """8d 임베딩에서 [w,h] 선형성분 제거(회귀 잔차) → PCA 축이 w,h 와 독립."""
    A = np.column_stack([wh, np.ones(len(wh))])           # [w, h, 1]
    coef, *_ = np.linalg.lstsq(A, X8, rcond=None)
    return X8 - A @ coef


def load_df():
    z = np.load(LABELS_NPZ)
    rows, labels = z["rows"], z["labels"]                 # 전체 유효 global row, 군집
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
        h0[torch.from_numpy(r)].numpy().astype(np.float32))             # 표준화 8d
    if RESIDUALIZE_WH:
        Xs = residualize_wh(Xs, wh)                                      # w,h 제거
    P2 = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    df = pd.DataFrame(dict(cluster=[f"c{int(v)}" for v in labels[sel]],
                           w=wh[:, 0], h=wh[:, 1], PC1=P2[:, 0], PC2=P2[:, 1]))
    tag = "w,h 제거 후" if RESIDUALIZE_WH else "원본"
    print(f"플롯 표본 {len(df):,}  (군집 {len(clusters)}개, 군집당 ≤{PER_CLUSTER:,}) "
          f"| PCA={tag} | palette={PALETTE_NAME}")
    return df


def _order_pal(df):
    cl = sorted(df["cluster"].unique(), key=lambda s: int(s[1:]))
    return cl, {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cl)}


def axis_ranges(df):
    """축 범위 dict. CONFIG 가 None 이면 0.5~99.5 분위(+여백)로 robust 자동."""
    def rng(col, cfg):
        if cfg is not None:
            return tuple(cfg)
        lo, hi = np.percentile(df[col], [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-9)
        return (lo - pad, hi + pad)
    R = {"w": rng("w", W_RANGE), "h": rng("h", H_RANGE),
         "PC1": rng("PC1", PC1_RANGE), "PC2": rng("PC2", PC2_RANGE)}
    print("  [축 범위] " + "  ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})"
                                     for k, v in R.items())
          + "   ← CONFIG 의 W_RANGE/H_RANGE/PC1_RANGE/PC2_RANGE 로 조정")
    return R


# ══════════════════════════════════════════════════════════════════
# 플롯들
# ══════════════════════════════════════════════════════════════════
def plot_hist(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col in zip(axes, ["w", "h"]):
        sns.histplot(df, x=col, hue="cluster", hue_order=cl, palette=pal,
                     element="step", stat="count", bins=60, alpha=0.25,
                     ax=ax, legend=(col == "h"))
        ax.set_xlim(R[col]); ax.set_title(f"{col} histogram by cluster")
        sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_hist.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [hist] {p}")


def plot_kde(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col in zip(axes, ["w", "h"]):
        sns.kdeplot(df, x=col, hue="cluster", hue_order=cl, palette=pal,
                    fill=True, alpha=0.25, common_norm=False, ax=ax,
                    legend=(col == "h"))
        ax.set_xlim(R[col]); ax.set_title(f"{col} density by cluster")
        sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_kde.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [kde] {p}")


def plot_pca_wh(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    pairs = [("PC1", "w"), ("PC1", "h"), ("PC2", "w"), ("PC2", "h")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (x, y) in zip(axes.flat, pairs):
        for c in cl:
            dd = df[df["cluster"] == c]
            ax.scatter(dd[x], dd[y], s=4, color=pal[c], alpha=0.4,
                       linewidths=0, label=c)
        ax.set_xlim(R[x]); ax.set_ylim(R[y])
        ax.set_xlabel(x); ax.set_ylabel(y); ax.set_title(f"{x} vs {y}")
        sns.despine(ax=ax)
    axes.flat[0].legend(markerscale=3, fontsize=9, frameon=False, title="cluster")
    sub = "w,h-residualized" if RESIDUALIZE_WH else "raw"
    fig.suptitle(f"PCA axes ({sub}) vs center geometry (w, h) by cluster",
                 fontsize=15, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_pca_wh_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [pca-wh] {p}")


def plot_wh(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 8))
    for c in cl:
        dd = df[df["cluster"] == c]
        ax.scatter(dd["w"], dd["h"], s=4, color=pal[c], alpha=0.4,
                   linewidths=0, label=c)
    ax.set_xlim(R["w"]); ax.set_ylim(R["h"])
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
    R = axis_ranges(df)
    plot_hist(df, R)
    plot_kde(df, R)
    plot_pca_wh(df, R)
    plot_wh(df, R)
    print(f"완료. 산출물: {OUT_DIR}/  "
          "(h0_wh_hist / h0_wh_kde / h0_pca_wh_scatter / h0_wh_scatter)")


if __name__ == "__main__":
    main()
