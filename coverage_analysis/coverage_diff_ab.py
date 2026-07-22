# -*- coding: utf-8 -*-
"""
coverage_diff_ab.py
===================
같은 Reference(고정 프레임)로 SampleA·SampleB 를 각각 돌려 얻은
reference_scatter_pc.csv 두 개를 비교해, Reference 각 지점을 A/B 커버 여부로
4사분면 태깅하고 'A는 못 덮는데 B가 덮는' 영역을 찾아 시각화·추출한다.

전제
    두 CSV 는 '같은 Reference + 같은 프레임(캐시)'에서 나와야 한다.
    → 동일 Reference 포인트가 동일 PC 좌표를 가지므로 PC 좌표로 조인한다.

4사분면 (in-domain 지점 대상)
    both_covered      A✔ B✔   (중복)
    A_only            A✔ B✘   (A만 덮음)
    B_fills_A         A✘ B✔   ★ A 공백을 B 가 메움 (핵심)
    both_gap          A✘ B✘   (공동 공백)

산출물
    diff_ab_scatter.png            : PC 평면 4사분면 색 (B_fills_A 강조)
    diff_B_fills_A.csv             : A 공백 ∩ B 커버 Reference 원본행 + PC
    diff_summary.json              : 사분면 카운트/증분 지표

실행:  python coverage_diff_ab.py
"""

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =============================================================================
# CONFIG
# =============================================================================
A_CSV   = "out_A/reference_scatter_pc.csv"   # SampleA 실행 결과
B_CSV   = "out_B/reference_scatter_pc.csv"   # SampleB 실행 결과
A_LABEL = "SampleA"
B_LABEL = "SampleB"

STATUS_COL = "cover_status"                  # covered / gap / out_of_domain
PLOT_COLS  = ["PC1", "PC2", "PC3", "PC4"]    # 그릴 PC (존재하는 것만)
ROUND_DEC  = 6                               # PC 조인 시 반올림 자리수

OUTPUT_DIR = "coverage_plots"
sns.set_theme(style="white", context="talk")

C_BOTH = "#DDDDDD"; C_AONLY = "#1F77B4"; C_BFILL = "#FF7F0E"; C_GAP = "#333333"


# =============================================================================
def _pc_cols(df):
    return [c for c in df.columns if re.fullmatch(r"PC\d+", str(c))]


def load_and_join(a_csv, b_csv):
    A = pd.read_csv(a_csv); B = pd.read_csv(b_csv)
    for nm, df in ((A_LABEL, A), (B_LABEL, B)):
        if STATUS_COL not in df.columns:
            raise ValueError(f"{nm}: '{STATUS_COL}' 열 없음 (열: {list(df.columns)})")
    pca, pcb = _pc_cols(A), _pc_cols(B)
    if pca != pcb or not pca:
        raise ValueError(f"두 파일의 PC 열이 다릅니다: {pca} vs {pcb}")
    # 조인 키 = 반올림한 PC 좌표(동일 Reference 포인트는 동일 프레임 좌표를 가짐)
    rk = [f"_r{c}" for c in pca]
    for c, r in zip(pca, rk):
        A[r] = A[c].round(ROUND_DEC); B[r] = B[c].round(ROUND_DEC)
    m = A.merge(B[rk + [STATUS_COL]].rename(columns={STATUS_COL: STATUS_COL + "_B"}),
                on=rk, how="inner")
    m = m.drop_duplicates(subset=rk)
    lo = min(len(A), len(B))
    if len(m) < 0.5 * lo:
        print(f"[경고] 매칭 {len(m):,}/{lo:,} — 두 CSV 가 같은 Reference/프레임에서 나온 게 맞는지 확인.")
    else:
        print(f"[Join] 공통 Reference 포인트 {len(m):,}개 매칭")
    return m, pca


def tag_quadrants(m):
    sa = m[STATUS_COL].astype(str).to_numpy()
    sb = m[STATUS_COL + "_B"].astype(str).to_numpy()
    ca, cb = (sa == "covered"), (sb == "covered")
    in_dom = (sa != "out_of_domain")            # 도메인은 프레임 속성(A/B 동일)
    q = np.full(len(m), "ood", dtype=object)
    q[in_dom & ca & cb] = "both_covered"
    q[in_dom & ca & ~cb] = "A_only"
    q[in_dom & ~ca & cb] = "B_fills_A"          # ★ A 공백 ∩ B 커버
    q[in_dom & ~ca & ~cb] = "both_gap"
    return q, in_dom


def plot_diff(m, pcs, q, out_path):
    cols = [c for c in PLOT_COLS if c in pcs]
    K = len(cols)
    order = [("both_covered", C_BOTH, 5, 0.25),
             ("A_only", C_AONLY, 10, 0.5),
             ("both_gap", C_GAP, 10, 0.5),
             ("B_fills_A", C_BFILL, 20, 0.9)]   # 마지막=위, 강조
    Y = m[cols].to_numpy(float)

    def draw(ax, i, jj):
        for lab, col, s, al in order:
            mm = q == lab
            if mm.any():
                ax.scatter(Y[mm, jj], Y[mm, i], s=s, c=col, alpha=al, linewidths=0)

    if K == 1:
        fig, ax = plt.subplots(figsize=(8, 6)); draw(ax, 0, 0)
    else:
        fig, axes = plt.subplots(K, K, figsize=(3.0 * K, 3.0 * K), squeeze=False)
        for i in range(K):
            for jj in range(K):
                ax = axes[i][jj]
                if i == jj:
                    ax.hist(Y[:, i], bins=40, color="#CCC"); ax.set_yticks([])
                else:
                    draw(ax, i, jj)
                if i == K - 1:
                    ax.set_xlabel(cols[jj], fontsize=9)
                if jj == 0:
                    ax.set_ylabel(cols[i], fontsize=9)
                ax.tick_params(labelsize=7)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", color=c, label=f"{l} ({int((q==l).sum()):,})")
           for l, c in [("both_covered", C_BOTH), ("A_only", C_AONLY),
                        ("both_gap", C_GAP), ("B_fills_A", C_BFILL)]]
    fig.legend(handles=leg, loc="upper right", fontsize=10)
    fig.suptitle(f"Coverage diff: {A_LABEL} vs {B_LABEL}  "
                 f"(orange = A-gap filled by B)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out_path, dpi=120); plt.close(fig)
    print(f"[Plot] diff scatter 저장: {out_path}")


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)
    m, pcs = load_and_join(A_CSV, B_CSV)
    q, in_dom = tag_quadrants(m)

    n_dom = int(in_dom.sum())
    cnt = {k: int((q == k).sum()) for k in ["both_covered", "A_only", "B_fills_A", "both_gap"]}
    a_gap = cnt["B_fills_A"] + cnt["both_gap"]        # A 공백(=in-domain 중 A 미커버)
    b_gap = cnt["A_only"] + cnt["both_gap"]           # B 공백
    b_fills_a_ratio = cnt["B_fills_A"] / a_gap if a_gap else 0.0
    a_fills_b_ratio = cnt["A_only"] / b_gap if b_gap else 0.0

    # ★ 핵심 산출: A 공백 ∩ B 커버 Reference 원본행 + PC
    feat_cols = [c for c in m.columns
                 if c not in pcs and not str(c).startswith("_r")
                 and c not in (STATUS_COL, STATUS_COL + "_B")]
    bf = m[q == "B_fills_A"]
    bf[feat_cols + pcs].to_csv(j("diff_B_fills_A.csv"), index=False)

    plot_diff(m, pcs, q, j("diff_ab_scatter.png"))

    summary = {"A": A_LABEL, "B": B_LABEL, "n_common": len(m), "n_domain": n_dom,
               **{f"n_{k}": v for k, v in cnt.items()},
               "B_fills_A_ratio_pct": round(b_fills_a_ratio * 100, 3),
               "A_fills_B_ratio_pct": round(a_fills_b_ratio * 100, 3)}
    with open(j("diff_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 64)
    print(f"  [Coverage Diff]  {A_LABEL} vs {B_LABEL}")
    print("=" * 64)
    print(f"  도메인 Reference 포인트           : {n_dom:,}")
    print(f"  both_covered={cnt['both_covered']:,}  A_only={cnt['A_only']:,}  "
          f"both_gap={cnt['both_gap']:,}")
    print(f"  >>> B_fills_A (A공백을 B가 메움)   : {cnt['B_fills_A']:,}  "
          f"(A 공백 {a_gap:,} 중 {b_fills_a_ratio*100:.2f}%)")
    print(f"  (참고) A_fills_B                   : {cnt['A_only']:,}  "
          f"(B 공백 {b_gap:,} 중 {a_fills_b_ratio*100:.2f}%)")
    print(f"  → diff_B_fills_A.csv ({len(bf):,}행), diff_ab_scatter.png, diff_summary.json")
    print("=" * 64)
    return summary


if __name__ == "__main__":
    run()
