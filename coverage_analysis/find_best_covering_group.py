# -*- coding: utf-8 -*-
"""
find_best_covering_group.py
===========================
Sample2 의 특정 타깃(gauge 단어로 선별)을 '제일 잘 커버하는' Sample1 의 gauge 그룹을 찾는다.
(눈으로 그룹별 겹침 보던 걸 reach 커버리지로 자동 랭킹)

입력 (같은 Reference/프레임으로 뽑은 KPCA scatter CSV)
    kpca_reference_scatter.csv  : ystd(정규화 스케일)용 reference KP
    Sample1 kpca_sample_scatter.csv : 후보 그룹들(gauge_name)
    Sample2 kpca_sample_scatter.csv : 타깃(gauge_name 로 선별)

방법
    타깃 sample2 포인트를, 각 Sample1 gauge 그룹의 패턴이 반경 R 안에서 덮는 비율(=target coverage)
    로 그룹을 랭킹. Sample1 그룹은 후보 단어 리스트 또는 gauge_name 접두어로 자동 도출.

산출물
    group_target_coverage.csv : Sample1 그룹별 target_coverage / mean_nn_dist / n_patterns (내림차순)
    group_target_scatter.png  : 타깃(주황) + 최상위 그룹(보라) 겹침
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG
# =============================================================================
REF_SCATTER     = "out_s1/kpca_reference_scatter.csv"   # ystd 용(없으면 sample 합으로 대체)
SAMPLE1_SCATTER = "out_s1/kpca_sample_scatter.csv"      # 후보 그룹들
SAMPLE2_SCATTER = "out_s2/kpca_sample_scatter.csv"      # 타깃 소속

GAUGE_COL    = "gauge_name"
TARGET_WORDS = ["BK"]        # Sample2 타깃 선별 단어(부분일치, 대소문자무시)
# Sample1 후보 그룹 정의: 단어 리스트를 주면 각 단어=한 그룹; None 이면 gauge_name 접두어로 자동
SAMPLE1_GROUP_WORDS = None
GROUP_DELIM  = "_"           # 자동 그룹화 시 gauge_name 을 이 구분자로 나눠 앞부분을 그룹키로

KP_COLS   = ["KP1", "KP2", "KP3"]
RADIUS    = 0.2666           # 정규화 KP 반경 — run 리포트의 R 값
#   ⚠️ 모든 그룹이 ~100%로 포화(순위 구분 사라짐)하면 KP 퍼짐 대비 R 이 큼 → 값을 낮춰 재실행.
TOPN_PLOT = 1                # 플롯에 강조할 상위 그룹 수
OUTPUT_DIR = "coverage_plots"
sns.set_theme(style="white", context="talk")


def _match_any(series, words):
    s = series.astype(str).str.lower()
    m = np.zeros(len(series), dtype=bool)
    for w in words:
        m |= s.str.contains(str(w).lower(), regex=False, na=False).to_numpy()
    return m


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    S1 = pd.read_csv(SAMPLE1_SCATTER)
    S2 = pd.read_csv(SAMPLE2_SCATTER)
    for nm, df in (("Sample1", S1), ("Sample2", S2)):
        for c in KP_COLS + [GAUGE_COL]:
            if c not in df.columns:
                raise ValueError(f"{nm} 에 '{c}' 열 없음 (열: {list(df.columns)})")

    # ystd (정규화 스케일)
    if REF_SCATTER and os.path.exists(REF_SCATTER):
        A = pd.read_csv(REF_SCATTER)
        ystd = A[KP_COLS].to_numpy(float).std(axis=0)
    else:
        both = pd.concat([S1[KP_COLS], S2[KP_COLS]])
        ystd = both.to_numpy(float).std(axis=0)
        print("[warn] REF_SCATTER 없음 → sample 합으로 ystd 추정")
    ystd[ystd == 0] = 1.0

    # 타깃 (Sample2)
    tmask = _match_any(S2[GAUGE_COL], TARGET_WORDS)
    T = S2[tmask]
    Tk = T[KP_COLS].to_numpy(float); Tf = np.isfinite(Tk).all(axis=1)
    Tn = Tk[Tf] / ystd
    if len(Tn) == 0:
        raise ValueError(f"Sample2 타깃({TARGET_WORDS})에 유효 포인트 없음")
    print(f"[Target] Sample2 ⊃ {TARGET_WORDS}: {len(Tn):,} 포인트")

    # Sample1 후보 그룹 정의
    if SAMPLE1_GROUP_WORDS is not None:
        groups = {str(w): _match_any(S1[GAUGE_COL], [w]) for w in SAMPLE1_GROUP_WORDS}
    else:
        key = S1[GAUGE_COL].astype(str).apply(lambda x: x.split(GROUP_DELIM)[0])
        groups = {g: (key == g).to_numpy() for g in sorted(key.unique())}
    print(f"[Groups] Sample1 후보 그룹 {len(groups)}개")

    # 그룹별 타깃 커버리지 (반경 R 내)
    rows = []
    for g, gm in groups.items():
        Gk = S1.loc[gm, KP_COLS].to_numpy(float); Gf = np.isfinite(Gk).all(axis=1)
        Gn = Gk[Gf] / ystd
        if len(Gn) == 0:
            continue
        d, _ = cKDTree(Gn).query(Tn, k=1)          # 타깃 → 이 그룹 최근접 거리
        rows.append({"group": g, "n_patterns": int(len(Gn)),
                     "target_coverage_pct": round(float((d <= RADIUS).mean()) * 100, 3),
                     "mean_nn_dist": round(float(d.mean()), 5)})
    res = pd.DataFrame(rows).sort_values("target_coverage_pct", ascending=False).reset_index(drop=True)
    res.to_csv(j("group_target_coverage.csv"), index=False)

    # 시각화
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.scatter(S1["KP1"], S1["KP2"], s=5, c="#DDDDDD", alpha=0.3, linewidths=0, label="Sample1 (all)")
    ax.scatter(T["KP1"], T["KP2"], s=16, c="#F4A15A", alpha=0.7, linewidths=0,
               label=f"Sample2 target {TARGET_WORDS}")
    pal = sns.color_palette("Dark2", max(TOPN_PLOT, 1))
    for i, g in enumerate(res["group"].head(TOPN_PLOT)):
        gm = groups[g]
        ax.scatter(S1.loc[gm, "KP1"], S1.loc[gm, "KP2"], s=22, color=pal[i], alpha=0.9,
                   linewidths=0, label=f"S1 '{g}' (cov {res.iloc[i]['target_coverage_pct']:.0f}%)")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    ax.set_title(f"Which Sample1 gauge group best covers Sample2 target?\n"
                 f"target {TARGET_WORDS} ({len(Tn)} pts), R={RADIUS}")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(j("group_target_scatter.png"), dpi=130); plt.close(fig)

    print("\n" + "=" * 60)
    print(f"  [Best covering group] Sample2 타깃 {TARGET_WORDS} 를 잘 덮는 Sample1 그룹")
    print("=" * 60)
    for _, r in res.head(8).iterrows():
        print(f"   {r['group']:>12}: target_cov {r['target_coverage_pct']:6.2f}%  "
              f"(n={int(r['n_patterns'])}, meanNN={r['mean_nn_dist']:.3f})")
    print("  → group_target_coverage.csv / group_target_scatter.png")
    print("=" * 60)
    return res


if __name__ == "__main__":
    run()
