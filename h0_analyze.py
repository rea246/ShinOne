"""
h0_analyze.py — h0_labels_full.npz 열어서 군집별 기하/엣지 분포·PCA 분석

목표: 각 군집이 "무엇으로" 갈렸는지(=ref DB 특징)를 읽어내기.
h0_clustering.py 의 전량 라벨(full.npz) + 캐시(hkeys_features.pt)의
중심 raw [w,h] / edge_raw / h0 임베딩을 합쳐서:
  1) 군집별 w, h 히스토그램            → h0_wh_hist.png
  2) 군집별 w, h 확률분포(KDE)         → h0_wh_kde.png
  3) PCA 축 vs w,h 산점도               → h0_pca_wh_scatter.png
  4) w vs h 기하 산점도                 → h0_wh_scatter.png
  5) 군집별 edge 분포(KDE)             → h0_edge_kde.png

PCA residualize (RESIDUALIZE):
  임베딩엔 w,h 가 녹아 있어 PCA 축이 그것과 상관됨. 그래서 회귀로 제거:
    'per_var' → 그리는 변수만 제거. "PC vs w" 는 w 만, "PC vs h" 는 h 만 제거
                (그 변수 자기상관만 없애고 나머지 관계는 보존).  ← 기본
    'both'    → w,h 둘 다 한 번에 제거(공통 PCA). 순수 비-기하 구조.
    'none'    → 원본 PCA.

edge_raw(8d) = edge_attr[box_dist, center_dist, ux, uy] 의 [mean4 + std4].
  방향(ux,uy)은 양방향 엣지라 mean≈0 → 이방성은 std 로 본다.

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

CLUSTERS    = None        # None = 전체 군집, 또는 [0, 1, 2, 3, 4]
PER_CLUSTER = 40_000      # 군집당 플롯 표본 상한 (큰 군집이 분포 압도 안 하게)
SEED        = 42

RESIDUALIZE = "per_var"   # 'per_var'(B) | 'both'(A) | 'none'

# scatter 가시성 개선. 'contour' = 군집별 밀도 등고선(+옅은 배경 점구름) → 겹침/분리
#   또렷. 'points' = 기존 원점 산점. 'facet' = 군집별 소분할(별도 파일, 겹침 0).
SCATTER_STYLE = "contour"

# 축 범위 — None = 자동(0.5~99.5 분위, 이상치 무시). 보고 (min,max) 로 조정.
W_RANGE   = None
H_RANGE   = None
PC1_RANGE = None
PC2_RANGE = None

# 팔레트 (전부 validator 통과, CVD-safe). reference | okabe | jewel
PALETTES = {
    "reference": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                  "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "okabe":     ["#0072B2", "#009E73", "#E69F00", "#D55E00",
                  "#CC79A7", "#56B4E9", "#e34948", "#7a5da8"],
    "jewel":     ["#1b6ca8", "#158a6b", "#d98b1e", "#c1440e",
                  "#6a3fb0", "#c42348", "#2f8f4e", "#b0731f"],
}
PALETTE_NAME = "jewel"
PALETTE = PALETTES[PALETTE_NAME]

# edge 분포로 그릴 4개 (해석 가능한 것 위주). (df컬럼, 표시라벨)
EDGE_PLOTS = [("e_box",   "box_dist mean (nm)"),
              ("e_cdist", "center_dist mean (nm)"),
              ("e_cstd",  "center_dist std (nm)"),
              ("e_dir",   "direction anisotropy (|u| std)")]


# ══════════════════════════════════════════════════════════════════
# 로드 + df 구성
# ══════════════════════════════════════════════════════════════════
def _resid(X, cols):
    """X(n,d) 에서 cols(각 n,) 선형성분을 회귀로 제거한 잔차."""
    A = np.column_stack([*cols, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def _pcas(Xs, w, h):
    """RESIDUALIZE 모드에 따라 PC1_w/PC2_w(‘vs w’용), PC1_h/PC2_h(‘vs h’용) 생성."""
    def pca2(M):
        return PCA(n_components=2, random_state=SEED).fit_transform(M)
    if RESIDUALIZE == "per_var":
        Pw, Ph = pca2(_resid(Xs, [w])), pca2(_resid(Xs, [h]))
    elif RESIDUALIZE == "both":
        Pw = Ph = pca2(_resid(Xs, [w, h]))
    else:
        Pw = Ph = pca2(Xs)
    return dict(PC1_w=Pw[:, 0], PC2_w=Pw[:, 1], PC1_h=Ph[:, 0], PC2_h=Ph[:, 1])


def load_df():
    z = np.load(LABELS_NPZ)
    rows, labels = z["rows"], z["labels"]
    d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    feats = d["features"]
    h0, h0_raw, edge_raw = feats["h0"], feats.get("h0_raw"), feats.get("edge_raw")
    if h0_raw is None:
        raise SystemExit("캐시에 h0_raw([w,h]) 없음 — 분석 불가")

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

    wh = h0_raw[torch.from_numpy(r)].numpy().astype(np.float64)
    Xs = StandardScaler().fit_transform(
        h0[torch.from_numpy(r)].numpy().astype(np.float32))
    data = dict(cluster=[f"c{int(v)}" for v in labels[sel]],
                w=wh[:, 0], h=wh[:, 1])
    data.update(_pcas(Xs, wh[:, 0], wh[:, 1]))

    if edge_raw is not None:                        # edge_raw = [box,cen,ux,uy]×[mean,std]
        er = edge_raw[torch.from_numpy(r)].numpy().astype(np.float64)
        data["e_box"]   = er[:, 0]                  # box_dist mean
        data["e_cdist"] = er[:, 1]                  # center_dist mean
        data["e_cstd"]  = er[:, 5]                  # center_dist std
        data["e_dir"]   = np.sqrt(er[:, 6] ** 2 + er[:, 7] ** 2)   # 방향 이방성
    else:
        print("  ⚠ 캐시에 edge_raw 없음 → edge 분석 생략")

    df = pd.DataFrame(data)
    print(f"플롯 표본 {len(df):,}  (군집 {len(clusters)}개, 군집당 ≤{PER_CLUSTER:,}) "
          f"| PCA residualize={RESIDUALIZE} | palette={PALETTE_NAME}")
    return df


def _order_pal(df):
    cl = sorted(df["cluster"].unique(), key=lambda s: int(s[1:]))
    return cl, {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cl)}


def _lim(s, cfg=None):
    if cfg is not None:
        return tuple(cfg)
    lo, hi = np.percentile(s, [0.5, 99.5])
    pad = 0.05 * (hi - lo + 1e-9)
    return (lo - pad, hi + pad)


def axis_ranges(df):
    R = {"w": _lim(df["w"], W_RANGE), "h": _lim(df["h"], H_RANGE),
         "PC1": _lim(np.r_[df["PC1_w"], df["PC1_h"]], PC1_RANGE),
         "PC2": _lim(np.r_[df["PC2_w"], df["PC2_h"]], PC2_RANGE)}
    print("  [축 범위] " + "  ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})"
                                     for k, v in R.items())
          + "   ← CONFIG W/H/PC1/PC2_RANGE 로 조정")
    return R


def _pc_label(xcol, yvar):
    """PC 축 라벨: residualize 모드 반영."""
    pc = xcol.split("_")[0]
    if RESIDUALIZE == "per_var":
        return f"{pc} (⊥{yvar})"
    if RESIDUALIZE == "both":
        return f"{pc} (⊥w,h)"
    return pc


def _legend_handles(cl, pal):
    return [plt.Line2D([0], [0], marker="o", ls="", mfc=pal[c], mec="none",
                       ms=9, label=c) for c in cl]


def _draw_2d(ax, df, xc, yv, cl, pal):
    """SCATTER_STYLE 에 따라 한 축쌍을 그린다.
    contour = 옅은 배경 점구름 + 군집별 밀도 등고선(겹침 덜함)."""
    if SCATTER_STYLE != "points":                            # contour/facet 공용 오버레이
        ax.scatter(df[xc], df[yv], s=2, c="#dcdcdc", alpha=0.10,
                   linewidths=0, zorder=1)                    # 전체 점구름(옅게)
        for c in cl:
            dd = df[df["cluster"] == c]
            if len(dd) < 10:
                continue
            try:
                sns.kdeplot(x=dd[xc], y=dd[yv], ax=ax, color=pal[c], levels=4,
                            thresh=0.15, linewidths=1.4, alpha=0.95, zorder=2)
            except Exception:
                pass
    else:
        for c in cl:
            dd = df[df["cluster"] == c]
            ax.scatter(dd[xc], dd[yv], s=4, color=pal[c], alpha=0.4,
                       linewidths=0, zorder=2)


# ══════════════════════════════════════════════════════════════════
# 플롯들
# ══════════════════════════════════════════════════════════════════
def _kde_grid(df, cols_labels, ncol, fname, suptitle):
    """공통: (컬럼,라벨) 리스트를 군집별 KDE 격자로."""
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    n = len(cols_labels); nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.5 * ncol, 5 * nrow),
                             squeeze=False)
    flat = axes.flat
    for i, (col, label) in enumerate(cols_labels):
        ax = flat[i]
        sns.kdeplot(df, x=col, hue="cluster", hue_order=cl, palette=pal,
                    fill=True, alpha=0.25, common_norm=False, ax=ax,
                    legend=(i == 0))
        ax.set_xlim(_lim(df[col])); ax.set_xlabel(label)
        ax.set_title(label); sns.despine(ax=ax)
    for j in range(n, nrow * ncol):
        flat[j].axis("off")
    fig.suptitle(suptitle, fontsize=14, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, fname)
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [{fname}] {p}")


def plot_wh_hist(df, R):
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


def plot_pca_wh(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    pairs = [("PC1_w", "w"), ("PC1_h", "h"), ("PC2_w", "w"), ("PC2_h", "h")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (xc, yv) in zip(axes.flat, pairs):
        _draw_2d(ax, df, xc, yv, cl, pal)
        ax.set_xlim(R["PC1" if xc.startswith("PC1") else "PC2"])
        ax.set_ylim(R[yv])
        ax.set_xlabel(_pc_label(xc, yv)); ax.set_ylabel(yv)
        ax.set_title(f"{_pc_label(xc, yv)} vs {yv}"); sns.despine(ax=ax)
    axes.flat[0].legend(handles=_legend_handles(cl, pal), fontsize=9,
                        frameon=False, title="cluster")
    fig.suptitle(f"PCA (residualize={RESIDUALIZE}) vs center geometry "
                 f"[{SCATTER_STYLE}]", fontsize=15, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_pca_wh_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [pca-wh] {p}")


def plot_wh_scatter(df, R):
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_2d(ax, df, "w", "h", cl, pal)
    ax.set_xlim(R["w"]); ax.set_ylim(R["h"])
    ax.set_xlabel("w"); ax.set_ylabel("h")
    ax.set_title(f"center geometry: w vs h by cluster [{SCATTER_STYLE}]")
    ax.legend(handles=_legend_handles(cl, pal), fontsize=10, frameon=False,
              title="cluster")
    sns.despine(ax=ax)
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [w-h] {p}")


def plot_facet_wh(df, R):
    """군집별 소분할(facet) — 겹침 0. 각 패널: 해당 군집(색) + 전체(회색 배경)."""
    cl, pal = _order_pal(df)
    sns.set_theme(style="white", context="talk")
    ncol = min(len(cl), 3); nrow = (len(cl) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.5 * nrow),
                             squeeze=False)
    flat = axes.flat
    for i, c in enumerate(cl):
        ax = flat[i]
        ax.scatter(df["w"], df["h"], s=2, c="#e2e2e2", alpha=0.25, linewidths=0)
        dd = df[df["cluster"] == c]
        ax.scatter(dd["w"], dd["h"], s=4, color=pal[c], alpha=0.5, linewidths=0)
        ax.set_xlim(R["w"]); ax.set_ylim(R["h"])
        ax.set_title(f"{c}  (n={len(dd):,})"); ax.set_xlabel("w"); ax.set_ylabel("h")
        sns.despine(ax=ax)
    for j in range(len(cl), nrow * ncol):
        flat[j].axis("off")
    fig.suptitle("w vs h — per-cluster facet (grey = all)", fontsize=14, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "h0_wh_facet.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [w-h facet] {p}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_df()
    R = axis_ranges(df)

    plot_wh_hist(df, R)
    _kde_grid(df, [("w", "w"), ("h", "h")], 2, "h0_wh_kde.png",
              "center geometry (w, h) density by cluster")
    plot_pca_wh(df, R)
    plot_wh_scatter(df, R)
    if SCATTER_STYLE == "facet":
        plot_facet_wh(df, R)
    if "e_box" in df.columns:
        _kde_grid(df, EDGE_PLOTS, 2, "h0_edge_kde.png",
                  "edge geometry density by cluster (edge_raw)")

    print(f"완료. 산출물: {OUT_DIR}/  (h0_wh_hist / h0_wh_kde / "
          "h0_pca_wh_scatter / h0_wh_scatter"
          + (" / h0_wh_facet" if SCATTER_STYLE == "facet" else "")
          + (" / h0_edge_kde" if "e_box" in df.columns else "") + ")")


if __name__ == "__main__":
    main()
