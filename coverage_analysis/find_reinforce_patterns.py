# -*- coding: utf-8 -*-
"""
find_reinforce_patterns.py
==========================
Sample1 이 못 덮는(gap) Reference 영역을, Sample2 의 어떤 패턴으로 보강하면 좋은지 찾는다.

전제
    coverage_domain_kpca.py 를 '같은 Reference(같은 프레임/캐시)'로 Sample1·Sample2 각각 실행해
    나온 CSV 를 입력으로 쓴다(동일 KPCA 잠재 좌표 → 비교 가능).
      - Sample1 run:  kpca_reference_scatter.csv   (reference + KP + cover_status + ref_density)
      - Sample2 run:  kpca_sample_scatter.csv       (sample2 패턴 + KP + 원본열)

아이디어 (reach + set-cover)
    1) Sample1 의 gap(=대변 못한 reference) 지점 중 '중요(밀도 큰)'것만 대상으로.
    2) 각 Sample2 패턴이 '반경 R 안에서 새로 덮을 수 있는 gap 지점'을 계산(중요도=ref_density 가중).
    3) 단순 랭킹(패턴별 보강효과) + 그리디 선택(중복 없이 gap 을 최대 감소시키는 소수 패턴).

산출물
    reinforce_ranked.csv   : Sample2 패턴 전체 + reinforce_count/importance (보강효과 내림차순)
    reinforce_greedy.csv   : 그리디로 고른 '추가하면 좋은' 패턴 순서 + 누적 커버
    reinforce_scatter.png  : gap(주황) / 선택된 보강 패턴(보라) / 이미 covered(회색)

실행:  python find_reinforce_patterns.py
"""

import os
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
REF_SCATTER_A    = "out_s1/kpca_reference_scatter.csv"   # Sample1 실행의 reference scatter
SAMPLE_SCATTER_B = "out_s2/kpca_sample_scatter.csv"      # Sample2 실행의 패턴들(후보)

KP_COLS   = ["KP1", "KP2", "KP3"]
RADIUS    = 0.054       # 대표 반경(정규화 KP 공간) — Sample1 run 리포트의 R 값 사용
IMPORTANT_ONLY = True   # Sample1 gap 중 '밀도 상위'만 대상
GAP_DENSITY_Q  = 0.5    # 중요 gap = ref_density 상위(1-Q). 0.5=상위 50%
TOPN_GREEDY    = 100    # 그리디로 뽑을 보강 패턴 수

OUTPUT_DIR = "coverage_plots"
sns.set_theme(style="white", context="talk")


# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    A = pd.read_csv(REF_SCATTER_A)     # reference (sample1 기준 태깅)
    B = pd.read_csv(SAMPLE_SCATTER_B)  # sample2 패턴
    for c in KP_COLS + ["cover_status", "ref_density"]:
        if c not in A.columns:
            raise ValueError(f"{REF_SCATTER_A} 에 '{c}' 열 없음")
    for c in KP_COLS:
        if c not in B.columns:
            raise ValueError(f"{SAMPLE_SCATTER_B} 에 '{c}' 열 없음")

    # 정규화 스케일(ystd) = reference KP 의 축별 표준편차 (본 모듈과 동일 정규화 재현)
    ystd = A[KP_COLS].to_numpy(float).std(axis=0); ystd[ystd == 0] = 1.0

    # Sample1 gap 지점 (중요도 필터)
    gapA = A[A["cover_status"] == "gap"].copy()
    if IMPORTANT_ONLY and len(gapA):
        thr = gapA["ref_density"].quantile(GAP_DENSITY_Q)
        gapA = gapA[gapA["ref_density"] >= thr]
    if len(gapA) == 0:
        print("[Reinforce] Sample1 의 (중요) gap 이 없습니다. 보강 대상 없음."); return
    G = gapA[KP_COLS].to_numpy(float) / ystd
    gdens = gapA["ref_density"].to_numpy(float)

    # Sample2 패턴 좌표(유효행)
    S = B[KP_COLS].to_numpy(float); fin = np.isfinite(S).all(axis=1)
    Bf = B[fin].reset_index(drop=True); Sn = S[fin] / ystd

    # 각 Sample2 패턴이 반경 R 안에서 덮는 gap 지점 목록
    tree = cKDTree(G)
    sets = tree.query_ball_point(Sn, RADIUS)
    gain_cnt = np.array([len(s) for s in sets])
    gain_imp = np.array([gdens[s].sum() if s else 0.0 for s in sets])
    print(f"[Reinforce] Sample1 중요 gap {len(G):,}개, Sample2 후보 {len(Sn):,}개, R={RADIUS}")

    # (1) 단순 랭킹: 패턴별 보강효과
    Bf["reinforce_count"] = gain_cnt
    Bf["reinforce_importance"] = gain_imp
    ranked = Bf.sort_values("reinforce_importance", ascending=False)
    ranked.to_csv(j("reinforce_ranked.csv"), index=False)

    # (2) 그리디 set-cover: 중복 없이 gap 을 최대 감소시키는 소수 패턴
    covered = np.zeros(len(G), dtype=bool)
    chosen, rows = [], []
    total_imp = gdens.sum()
    for _ in range(min(TOPN_GREEDY, len(Sn))):
        best, best_gain, best_new = -1, 0.0, None
        for p in range(len(Sn)):
            if p in chosen or gain_cnt[p] == 0:
                continue
            new = [i for i in sets[p] if not covered[i]]
            g = gdens[new].sum()
            if g > best_gain:
                best, best_gain, best_new = p, g, new
        if best < 0 or best_gain <= 0:
            break
        covered[best_new] = True
        chosen.append(best)
        rows.append({"order": len(chosen), "sample2_row": int(best),
                     "new_gap_covered": len(best_new),
                     "importance_gain": float(best_gain),
                     "cum_gap_covered": int(covered.sum()),
                     "cum_importance_pct": round(gdens[covered].sum() / total_imp * 100, 3)})
    greedy = pd.DataFrame(rows)
    # 원본 식별열 붙이기
    if len(greedy):
        greedy = pd.concat([greedy.reset_index(drop=True),
                            Bf.iloc[greedy["sample2_row"].to_numpy()].reset_index(drop=True)], axis=1)
    greedy.to_csv(j("reinforce_greedy.csv"), index=False)

    # 시각화 (KP1-2)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    cov_ref = A[A["cover_status"] == "covered"]
    ax.scatter(cov_ref["KP1"], cov_ref["KP2"], s=5, c="#CCCCCC", alpha=0.3, linewidths=0,
               label="Ref covered by S1")
    ax.scatter(gapA["KP1"], gapA["KP2"], s=14, c="#F4A15A", alpha=0.6, linewidths=0,
               label="S1 gap (important)")
    ax.scatter(Bf["KP1"], Bf["KP2"], s=8, c="#2CA02C", alpha=0.25, linewidths=0, label="Sample2 (all)")
    if len(chosen):
        sel = Bf.iloc[chosen]
        ax.scatter(sel["KP1"], sel["KP2"], s=40, c="#8E24AA", alpha=0.95, linewidths=0,
                   marker="*", label=f"Reinforce picks (n={len(chosen)})")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    cov_pct = greedy["cum_importance_pct"].iloc[-1] if len(greedy) else 0.0
    ax.set_title(f"Reinforce Sample1 gaps with Sample2 patterns\n"
                 f"greedy {len(chosen)} picks cover {cov_pct:.1f}% of important-gap importance")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(j("reinforce_scatter.png"), dpi=130); plt.close(fig)

    print("\n" + "=" * 64)
    print("  [Reinforce] Sample1 gap 을 Sample2 로 보강")
    print("=" * 64)
    print(f"  Sample1 중요 gap 지점         : {len(G):,}")
    print(f"  보강효과 있는 Sample2 패턴    : {int((gain_cnt>0).sum()):,} / {len(Sn):,}")
    print(f"  그리디 선택 패턴 수           : {len(chosen):,}")
    if len(greedy):
        print(f"  → 이 {len(chosen)}개로 중요 gap 의 {cov_pct:.1f}% (중요도 기준) 보강 가능")
    print(f"  reinforce_ranked.csv / reinforce_greedy.csv / reinforce_scatter.png")
    print("=" * 64)
    return {"n_gap": len(G), "n_useful_s2": int((gain_cnt > 0).sum()), "n_picks": len(chosen)}


if __name__ == "__main__":
    run()
