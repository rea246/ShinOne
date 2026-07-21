# -*- coding: utf-8 -*-
"""
scatter_compare.py
==================
coverage_domain_mahal.py 가 뽑은 결과 CSV(reference_scatter_pc.csv / sample_pc_cover.csv)를
겹쳐서 PC scatter 로 비교하는 도구.

기능
    1) 두 데이터셋(예: Reference vs Sample1, 또는 Sample1 vs Sample2)을 겹쳐 그린다.
    2) sample 을 그릴 때 gauge_name 열에서 '특정 단어가 포함된' gauge 만 필터해 그린다.
    3) 단어는 리스트로 여러 개 지정 가능(부분일치, 대소문자 무시).
    4) 필터된 sample gauge '주변'의 Reference gauge 를 랜덤 N개(기본 1000) 선별해 CSV 저장.
    5) 어떤 PC/컬럼으로 그릴지는 PLOT_COLS 에 '컬럼 이름'으로 직접 선언(기본 PC1~PC4).

※ 같은 Reference 프레임에서 나온 CSV 라야 PC 좌표가 같은 축(비교 유효).
실행:  python scatter_compare.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================
# 겹쳐 그릴 두 데이터셋. gauge_words 가 비어있으면 전체, 있으면 그 단어(부분일치)만.
DATASET_A = {"path": "out_ref/reference_scatter_pc.csv", "label": "Reference",
             "color": "#8C8C8C", "gauge_words": [], "alpha": 0.25, "size": 6}
DATASET_B = {"path": "out_s1/sample_pc_cover.csv", "label": "Sample1",
             "color": "#2CA02C", "gauge_words": ["CM", "G17"], "alpha": 0.7, "size": 16}

GAUGE_COL = "gauge_name"                 # 필터 기준 열 이름
PLOT_COLS = ["PC1", "PC2", "PC3", "PC4"]  # ★ 그릴 컬럼을 '이름'으로 직접 선언

# ---- 필터된 sample 주변 Reference gauge 선별 저장 ----
REF_NEIGHBOR_SRC   = "out_ref/reference_scatter_pc.csv"  # 주변을 뽑을 Reference CSV (None=안함)
NEIGHBOR_COLS      = ["PC1", "PC2", "PC3", "PC4"]         # 거리 계산에 쓸 컬럼
NEIGHBOR_N         = 1000                                 # 선별 개수
NEIGHBOR_RADIUS    = None                                 # None→자동(가까운 풀에서 랜덤), 값→그 반경 내 랜덤
NEIGHBOR_POOL_MULT = 3                                    # 자동 풀 크기 = N * 이 배수
NEIGHBOR_SEED      = 0
OUT_NEIGHBOR       = "ref_near_filtered_gauges.csv"

OUT_PLOT = "scatter_compare.png"
sns.set_theme(style="white", context="talk")


# =============================================================================
def load_filtered(ds):
    """CSV 로드 + gauge_name 부분일치 필터(단어 리스트)."""
    df = pd.read_csv(ds["path"])
    words = ds.get("gauge_words") or []
    if words:
        if GAUGE_COL not in df.columns:
            raise ValueError(f"'{GAUGE_COL}' 열이 {ds['path']} 에 없습니다. (열: {list(df.columns)})")
        s = df[GAUGE_COL].astype(str)
        mask = np.zeros(len(df), dtype=bool)
        for w in words:
            mask |= s.str.contains(str(w), case=False, na=False, regex=False)
        df = df[mask].copy()
        print(f"[Filter] {ds['label']}: gauge_name 에 {words} 포함 → {len(df):,}행")
    else:
        print(f"[Load] {ds['label']}: {len(df):,}행 (필터 없음)")
    return df


def _check_cols(df, cols, who):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{who} 에 없는 컬럼: {miss} (있는 컬럼: {list(df.columns)})")


def _clean(df, cols):
    X = df[cols].to_numpy(dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    return df[finite], X[finite]


def plot_pairscatter(dfA, dfB, cols, out_path):
    _check_cols(dfA, cols, DATASET_A["label"]); _check_cols(dfB, cols, DATASET_B["label"])
    A = dfA[cols].to_numpy(float); B = dfB[cols].to_numpy(float)
    A = A[np.isfinite(A).all(1)]; B = B[np.isfinite(B).all(1)]
    K = len(cols)
    lo = np.nanpercentile(np.vstack([A, B]), 1, axis=0)
    hi = np.nanpercentile(np.vstack([A, B]), 99, axis=0)

    if K == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(A[:, 0], bins=40, density=True, color=DATASET_A["color"], alpha=0.5, label=DATASET_A["label"])
        ax.hist(B[:, 0], bins=40, density=True, color=DATASET_B["color"], alpha=0.5, label=DATASET_B["label"])
        ax.set_xlabel(cols[0]); ax.legend()
        fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
        print(f"[Plot] 저장: {out_path}"); return

    fig, axes = plt.subplots(K, K, figsize=(3.0 * K, 3.0 * K), squeeze=False)
    for i in range(K):
        for jj in range(K):
            ax = axes[i][jj]
            if i == jj:
                ax.hist(A[:, i], bins=40, density=True, color=DATASET_A["color"], alpha=0.5)
                ax.hist(B[:, i], bins=40, density=True, color=DATASET_B["color"], alpha=0.5)
                ax.set_yticks([])
            else:
                ax.scatter(A[:, jj], A[:, i], s=DATASET_A["size"], c=DATASET_A["color"],
                           alpha=DATASET_A["alpha"], linewidths=0)
                ax.scatter(B[:, jj], B[:, i], s=DATASET_B["size"], c=DATASET_B["color"],
                           alpha=DATASET_B["alpha"], linewidths=0)
                ax.set_xlim(lo[jj], hi[jj]); ax.set_ylim(lo[i], hi[i])
            if i == K - 1:
                ax.set_xlabel(cols[jj], fontsize=9)
            if jj == 0:
                ax.set_ylabel(cols[i], fontsize=9)
            ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=DATASET_A["color"], label=f"{DATASET_A['label']} ({len(A):,})"),
               Line2D([0], [0], marker="o", ls="", color=DATASET_B["color"], label=f"{DATASET_B['label']} ({len(B):,})")]
    fig.legend(handles=handles, loc="upper right", fontsize=11)
    fig.suptitle(f"Scatter compare: {DATASET_A['label']} vs {DATASET_B['label']}  ({'/'.join(cols)})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[Plot] 저장: {out_path}")


def select_ref_neighbors(filt_df, out_path):
    """필터된 sample(filt_df) 주변의 Reference gauge 를 랜덤 N개 선별해 저장."""
    if not REF_NEIGHBOR_SRC:
        return
    if len(filt_df) == 0:
        print("[Neighbor] 필터된 sample 이 0행 → 주변 ref 선별 생략"); return
    ref = pd.read_csv(REF_NEIGHBOR_SRC)
    _check_cols(ref, NEIGHBOR_COLS, "REF_NEIGHBOR_SRC"); _check_cols(filt_df, NEIGHBOR_COLS, "filtered sample")
    ref_c, R = _clean(ref, NEIGHBOR_COLS)
    _, P = _clean(filt_df, NEIGHBOR_COLS)
    if len(P) == 0 or len(R) == 0:
        print("[Neighbor] 유효 좌표 없음 → 생략"); return

    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(R, k=1)          # 각 ref 의 '가장 가까운 필터sample' 거리
    if NEIGHBOR_RADIUS is not None:
        pool = np.where(d <= NEIGHBOR_RADIUS)[0]
    else:
        pool_size = min(len(R), NEIGHBOR_N * NEIGHBOR_POOL_MULT)
        pool = np.argsort(d)[:pool_size]     # 가장 가까운 풀
    rng = np.random.default_rng(NEIGHBOR_SEED)
    take = min(NEIGHBOR_N, len(pool))
    sel = rng.choice(pool, size=take, replace=False) if take > 0 else np.array([], int)
    out = ref_c.iloc[sel].copy()
    out["nn_dist_to_filtered"] = d[sel]
    out.to_csv(out_path, index=False)
    print(f"[Neighbor] 필터 sample 주변 ref {len(out):,}개 저장(풀 {len(pool):,}): {out_path}")


def main():
    dfA = load_filtered(DATASET_A)
    dfB = load_filtered(DATASET_B)
    plot_pairscatter(dfA, dfB, PLOT_COLS, OUT_PLOT)
    select_ref_neighbors(dfB, OUT_NEIGHBOR)   # B(sample) 필터 주변 ref 선별


if __name__ == "__main__":
    main()
