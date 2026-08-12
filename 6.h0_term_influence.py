"""
6.h0_term_influence.py — 각 term(w, h, edge)이 h0 군집을 얼마나 가르는지 scatter 로 진단

아이디어(per-var residualize):
  8차원 h0 임베딩에는 w·h·edge 가 (GAE 메시지패싱으로) 녹아 있다. 한 term 의
  선형성분만 회귀로 제거하고, 남은 부분공간을 PC 하나로 압축하면 "그 term 을 뺀
  뒤에도 남는 최강 구조" 를 얻는다. 이 PC 와 term 을 함께 scatter 하면:
    · 군집이 term 축(세로)으로 갈리면  → 그 term 이 군집을 가른 것
    · 군집이 PC 축(가로)으로 갈리면    → term 이 아닌 다른 구조가 가른 것

  edge 는 box_distance(edge_raw[:,0], box_dist mean) 로 대표한다.

5개 scatter:
    [PC_w, w]   [PC_h, h]   [PC_e, e]      ← term 영향 (term 뺀 PC vs term)
    [w, h]      [w, e]                     ← 원본 기하 관계

두 형태로 출력(둘 다 군집색, noise 는 회색 점):
    1) 점 산점도               → h0_term_scatter.png
    2) 밀도 등고선(채움, α=0.2) → h0_term_contour.png

각 term 이 임베딩 분산을 선형으로 얼마나 설명하는지(R²)를 함께 출력/제목에 병기 —
w 는 1차원, e 는 1차원이라 제거량이 대등해 케이스 비교가 공정하다.

입력: h0_clustering_out/h0_labels_full.npz (rows, labels)
      hkeys_features.pt (features: h0, h0_raw[w,h], edge_raw[box,cen,ux,uy]×[mean,std])
argv 없음 — 아래 CONFIG 만 고쳐 실행. 의존: seaborn, pandas, sklearn.

    python 6.h0_term_influence.py
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

CLUSTERS      = None      # None = 전체 군집, 또는 [0, 1, 2, 3, 4]
INCLUDE_NOISE = True      # noise(-1) 그룹도 회색으로 함께 플랏
PER_CLUSTER   = 40_000    # 그룹당 플롯 표본 상한 (큰 군집이 분포 압도 안 하게)
SEED          = 42

# 점 산점도
POINT_SIZE  = 4
POINT_ALPHA = 0.4

# 등고선(seaborn KDE): 최고밀도의 thresh% 부터 levels 분할, 내부 fill_alpha 로 채움.
CONTOUR_THRESH     = 0.15    # 최고밀도의 15% 미만 꼬리는 안 그림
CONTOUR_LEVELS     = 4       # 4 분할
CONTOUR_FILL_ALPHA = 0.2     # 등고선 내부 채움 투명도 (요청값)

# 축 범위 자동 분위 (이상치 무시). 보고 좁히고 싶으면 값만 조정.
AXIS_PCTL = (0.5, 99.5)

# 팔레트 (validator 통과, CVD-safe). reference | okabe | jewel
PALETTES = {
    "reference": ["#2a78d6", "#1baf7a", "#eda100", "#008300",
                  "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"],
    "okabe":     ["#0072B2", "#009E73", "#E69F00", "#D55E00",
                  "#CC79A7", "#56B4E9", "#e34948", "#7a5da8"],
    "jewel":     ["#1b6ca8", "#158a6b", "#d98b1e", "#c1440e",
                  "#6a3fb0", "#c42348", "#2f8f4e", "#b0731f"],
}
PALETTE_NAME = "reference"
PALETTE = list(PALETTES[PALETTE_NAME])
PALETTE[3] = "#e34948"    # C3: 초록(#008300) → 빨강 (사용자 지정; C5 와 같은 빨강)
NOISE_COLOR = "#9a9a9a"   # noise 그룹 회색

# 5개 scatter 패널 (x컬럼, y컬럼). 앞 3개는 term 뺀 PC vs term, 뒤 2개는 원본 기하.
PANELS = [("PC_w", "w"), ("PC_h", "h"), ("PC_e", "e"),
          ("w", "h"), ("w", "e")]

# 축 라벨 (제목이 패널 폭을 넘지 않게 간결하게)
LABELS = {"w": "w (nm)", "h": "h (nm)", "e": "box_dist (nm)",
          "PC_w": "PC ⊥w", "PC_h": "PC ⊥h", "PC_e": "PC ⊥e"}


# ══════════════════════════════════════════════════════════════════
# 로드 + df 구성
# ══════════════════════════════════════════════════════════════════
def _resid(X, cols):
    """X(n,d) 에서 cols(각 n,) 선형성분을 회귀로 제거한 잔차."""
    A = np.column_stack([*cols, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def _term_pc(Xs, term):
    """term 을 회귀 제거한 뒤 PC1 하나 + 그 term 의 설명분산 비율(R²) 반환.

    R² = 1 - Var(잔차)/Var(원본)  (전 차원 합) — 임베딩 분산 중 term 이
    선형으로 설명하는 비율. term 영향의 정량 지표.
    """
    R = _resid(Xs, [term])
    r2 = 1.0 - float(R.var(axis=0).sum()) / float(Xs.var(axis=0).sum())
    pc = PCA(n_components=1, random_state=SEED).fit_transform(R)[:, 0]
    return pc, r2


def load_df():
    z = np.load(LABELS_NPZ)
    rows, labels = z["rows"], z["labels"]
    d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    feats = d["features"]
    h0, h0_raw, edge_raw = feats["h0"], feats.get("h0_raw"), feats.get("edge_raw")
    if h0_raw is None:
        raise SystemExit("캐시에 h0_raw([w,h]) 없음 — 분석 불가")
    if edge_raw is None:
        raise SystemExit("캐시에 edge_raw 없음 — edge term 분석 불가")

    rng = np.random.default_rng(SEED)
    groups = list(np.unique(labels[labels >= 0]) if CLUSTERS is None
                  else np.asarray(CLUSTERS))
    if INCLUDE_NOISE and (labels < 0).any():
        groups.append(-1)                               # noise 그룹 포함
    sel = []
    for c in groups:
        idx = np.where(labels == c)[0]
        sel.append(idx if len(idx) <= PER_CLUSTER
                   else rng.choice(idx, PER_CLUSTER, replace=False))
    sel = np.sort(np.concatenate(sel))
    r = rows[sel]

    wh = h0_raw[torch.from_numpy(r)].numpy().astype(np.float64)
    er = edge_raw[torch.from_numpy(r)].numpy().astype(np.float64)
    Xs = StandardScaler().fit_transform(
        h0[torch.from_numpy(r)].numpy().astype(np.float32))

    w, h, e = wh[:, 0], wh[:, 1], er[:, 0]              # e = box_dist mean

    PC_w, r2w = _term_pc(Xs, w)
    PC_h, r2h = _term_pc(Xs, h)
    PC_e, r2e = _term_pc(Xs, e)
    r2 = {"w": r2w, "h": r2h, "e": r2e}

    df = pd.DataFrame(dict(
        cluster=["noise" if v < 0 else f"c{int(v)}" for v in labels[sel]],
        w=w, h=h, e=e, PC_w=PC_w, PC_h=PC_h, PC_e=PC_e))

    print(f"플롯 표본 {len(df):,}  (그룹 {len(groups)}개, 그룹당 ≤{PER_CLUSTER:,}) "
          f"| palette={PALETTE_NAME}")
    print(f"  [term 설명분산 R²] w={r2w:.3f}  h={r2h:.3f}  e(box_dist)={r2e:.3f}"
          "   ← 임베딩 분산 중 각 term 이 선형 설명하는 비율")
    return df, r2


# ══════════════════════════════════════════════════════════════════
# 색/축/렌더 헬퍼
# ══════════════════════════════════════════════════════════════════
def _order_pal(df):
    """군집(팔레트 색, 번호순) + noise(회색, 맨 뒤)."""
    names = df["cluster"].unique().tolist()
    cl = sorted([x for x in names if x != "noise"], key=lambda s: int(s[1:]))
    pal = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(cl)}
    if "noise" in names:
        cl = cl + ["noise"]
        pal["noise"] = NOISE_COLOR
    return cl, pal


def _lim(s):
    lo, hi = np.percentile(s, AXIS_PCTL)
    pad = 0.05 * (hi - lo + 1e-9)
    return (lo - pad, hi + pad)


def _legend_handles(cl, pal):
    return [plt.Line2D([0], [0], marker="o", ls="", mfc=pal[c], mec="none",
                       ms=10, label=c) for c in cl]


def _title(xc, yc, r2):
    """PC 패널엔 그 term 의 R²(설명분산) 병기."""
    if xc.startswith("PC_"):
        term = xc.split("_")[1]
        return f"{LABELS[xc]} vs {LABELS[yc]}   (R²({term})={r2[term]:.2f})"
    return f"{LABELS[xc]} vs {LABELS[yc]}"


def _draw_panel(ax, df, xc, yc, cl, pal, mode):
    """한 패널 렌더. noise = 회색 점(α0.5). 군집: points 또는 contour(채움)."""
    for c in cl:
        dd = df[df["cluster"] == c]
        if c == "noise":                                 # noise 는 항상 회색 점
            ax.scatter(dd[xc], dd[yc], s=POINT_SIZE, c=NOISE_COLOR, alpha=0.5,
                       linewidths=0, zorder=1)
        elif mode == "points":
            ax.scatter(dd[xc], dd[yc], s=POINT_SIZE, color=pal[c],
                       alpha=POINT_ALPHA, linewidths=0, zorder=2)
        else:                                            # 밀도 등고선(채움)
            try:
                sns.kdeplot(x=dd[xc], y=dd[yc], ax=ax, color=pal[c],
                            fill=True, thresh=CONTOUR_THRESH, levels=CONTOUR_LEVELS,
                            alpha=CONTOUR_FILL_ALPHA, zorder=2)
            except Exception as ex:                      # 표본 부족/특이행렬 등
                print(f"    ⚠ KDE 실패 ({c}, {xc}-{yc}): {ex}")


# ══════════════════════════════════════════════════════════════════
# 플롯
# ══════════════════════════════════════════════════════════════════
def make_figure(df, r2, mode, fname, suptitle):
    """5개 패널을 2×3 격자로 (6번째 칸은 범례). mode='points'|'contour'."""
    cl, pal = _order_pal(df)
    lims = {col: _lim(df[col]) for col in ["w", "h", "e", "PC_w", "PC_h", "PC_e"]}
    sns.set_theme(style="white", context="talk")
    fig, axes = plt.subplots(2, 3, figsize=(6.2 * 3, 5.4 * 2), squeeze=False)
    flat = axes.flat
    for i, (xc, yc) in enumerate(PANELS):
        ax = flat[i]
        _draw_panel(ax, df, xc, yc, cl, pal, mode)
        ax.set_xlim(lims[xc]); ax.set_ylim(lims[yc])
        ax.set_xlabel(LABELS[xc]); ax.set_ylabel(LABELS[yc])
        ax.set_title(_title(xc, yc, r2)); sns.despine(ax=ax)
    leg_ax = flat[len(PANELS)]                            # 남는 칸 = 범례
    leg_ax.axis("off")
    leg_ax.legend(handles=_legend_handles(cl, pal), loc="center",
                  fontsize=12, frameon=False, title="cluster")
    fig.suptitle(suptitle, fontsize=15, weight="bold")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, fname)
    fig.savefig(p, dpi=130); plt.close(fig); print(f"  [{mode}] {p}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, r2 = load_df()
    make_figure(df, r2, "points", "h0_term_scatter.png",
                "term influence — per-var residualized PC vs term (points)")
    make_figure(df, r2, "contour", "h0_term_contour.png",
                f"term influence — density contour (fill α={CONTOUR_FILL_ALPHA})")
    print(f"완료. 산출물: {OUT_DIR}/  (h0_term_scatter, h0_term_contour)")


if __name__ == "__main__":
    main()
