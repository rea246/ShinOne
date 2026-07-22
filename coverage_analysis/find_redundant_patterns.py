# -*- coding: utf-8 -*-
"""
find_redundant_patterns.py
==========================
Reference 를 덮는 데 '필요 없는(중복)' Sample 패턴을 찾는다(= 빼도 커버리지 손실 없음).

전제
    coverage_domain_kpca.py 를 실행해 나온 (같은 프레임) CSV 를 입력으로 쓴다.
      - kpca_reference_scatter.csv : reference + KP (+ cover_status, ref_density)
      - kpca_sample_scatter.csv    : 평가할 sample 패턴 + KP + 원본열

아이디어 (reach + set-cover 의 역방향)
    1) Sample 이 실제로 덮는 reference 지점(반경 R 내) 을 대상으로.
    2) 각 reference 지점을 몇 개의 패턴이 덮는지(multiplicity) 계산.
    3) 패턴별 '유일 기여'(자기만 덮는 reference, 중요도=ref_density 가중) 산출.
       유일기여=0 → 그 패턴 하나는 빼도 손실 없음(중복).
    4) 그리디 최소덮개 → 유지할 최소 패턴집합. 나머지 = '한꺼번에 빼도 되는' 불필요 패턴.

산출물
    pattern_redundancy.csv : 패턴별 n_covered / unique_contribution / is_redundant / keep(최소집합)
    redundant_scatter.png  : 유지(별)/제거가능(연회색) 패턴 + covered reference
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
REF_SCATTER    = "out_s1/kpca_reference_scatter.csv"   # 대상 sample 실행의 reference scatter
SAMPLE_SCATTER = "out_s1/kpca_sample_scatter.csv"      # 평가/가지치기할 sample 패턴
KP_COLS   = ["KP1", "KP2", "KP3"]
RADIUS    = 0.2666      # 대표 반경(정규화 KP) — 해당 run 리포트의 R 값
USE_IMPORTANCE = True   # ref_density 로 중요도 가중(있으면)
OUTPUT_DIR = "coverage_plots"
sns.set_theme(style="white", context="talk")


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    A = pd.read_csv(REF_SCATTER)      # reference
    B = pd.read_csv(SAMPLE_SCATTER)   # sample 패턴
    for c in KP_COLS:
        if c not in A.columns or c not in B.columns:
            raise ValueError(f"KP 열 '{c}' 가 입력 CSV 에 없습니다.")

    ystd = A[KP_COLS].to_numpy(float).std(axis=0); ystd[ystd == 0] = 1.0

    # 도메인 reference (out_of_domain 제외; 열 없으면 전체)
    if "cover_status" in A.columns:
        A = A[A["cover_status"] != "out_of_domain"].reset_index(drop=True)
    Rn = A[KP_COLS].to_numpy(float) / ystd
    dens = A["ref_density"].to_numpy(float) if ("ref_density" in A.columns and USE_IMPORTANCE) \
        else np.ones(len(A))

    S = B[KP_COLS].to_numpy(float); fin = np.isfinite(S).all(axis=1)
    Bf = B[fin].reset_index(drop=True); Sn = S[fin] / ystd

    # 각 패턴이 덮는 reference 지점 목록 + reference 별 multiplicity
    tree_ref = cKDTree(Rn)
    pat_covers = tree_ref.query_ball_point(Sn, RADIUS)      # 패턴 → ref 인덱스들
    mult = np.zeros(len(Rn), dtype=np.int32)
    for cov in pat_covers:
        if cov:
            mult[np.asarray(cov)] += 1
    covered_ref = mult > 0                                  # sample 이 실제 덮는 reference

    # 패턴별 지표
    n_cov = np.array([len(c) for c in pat_covers])
    uniq = np.array([dens[[i for i in c if mult[i] == 1]].sum() if c else 0.0 for c in pat_covers])
    is_redundant = uniq <= 0                                 # 자기만 덮는 ref 없음 → 단독 제거 가능

    # 그리디 최소덮개: covered_ref 를 전부 덮는 최소 패턴집합
    need = covered_ref.copy()
    keep = np.zeros(len(Sn), dtype=bool)
    cover_sets = [set(c) for c in pat_covers]
    while need.any():
        # 남은 need 를 가장 많이(중요도) 덮는 패턴 선택
        best, best_gain = -1, 0.0
        for p in range(len(Sn)):
            if keep[p] or not cover_sets[p]:
                continue
            g = dens[[i for i in cover_sets[p] if need[i]]].sum()
            if g > best_gain:
                best, best_gain = p, g
        if best < 0 or best_gain <= 0:
            break
        keep[best] = True
        need[np.asarray(list(cover_sets[best]), dtype=int)] = False

    removable = ~keep & (n_cov >= 0)                         # 최소집합 밖 = 한꺼번에 빼도 됨
    Bf["n_covered_ref"] = n_cov
    Bf["unique_contribution"] = uniq
    Bf["is_redundant"] = is_redundant                        # 단독 제거 가능 여부
    Bf["keep_min_cover"] = keep                              # 최소집합 유지 대상
    Bf["removable"] = removable                              # 빼도 되는(불필요) 패턴
    Bf.sort_values(["removable", "unique_contribution"], ascending=[False, True]) \
      .to_csv(j("pattern_redundancy.csv"), index=False)

    # 시각화 (KP1-2)
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.scatter(A["KP1"][covered_ref], A["KP2"][covered_ref], s=5, c="#CCCCCC", alpha=0.3,
               linewidths=0, label="Reference covered")
    ax.scatter(Bf["KP1"][removable], Bf["KP2"][removable], s=10, c="#BDBDBD", alpha=0.5,
               linewidths=0, label=f"Removable (n={int(removable.sum())})")
    ax.scatter(Bf["KP1"][keep], Bf["KP2"][keep], s=40, c="#8E24AA", alpha=0.95, linewidths=0,
               marker="*", label=f"Keep (min-cover, n={int(keep.sum())})")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    ax.set_title(f"Redundant sample patterns\n"
                 f"{int(keep.sum())} keep covers same reference as all {len(Sn)} "
                 f"(removable {int(removable.sum())})")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(j("redundant_scatter.png"), dpi=130); plt.close(fig)

    print("\n" + "=" * 64)
    print("  [Redundant] Reference 커버에 불필요한 Sample 패턴")
    print("=" * 64)
    print(f"  Sample 패턴 수                : {len(Sn):,}")
    print(f"  Sample 이 덮는 reference 지점 : {int(covered_ref.sum()):,}")
    print(f"  단독 제거 가능(중복) 패턴     : {int(is_redundant.sum()):,}")
    print(f"  >>> 최소집합 유지 패턴        : {int(keep.sum()):,}")
    print(f"  >>> 한꺼번에 빼도 되는 패턴   : {int(removable.sum()):,}  "
          f"({removable.mean()*100:.1f}%)  → pattern_redundancy.csv")
    print("=" * 64)
    return {"n_sample": len(Sn), "n_keep": int(keep.sum()), "n_removable": int(removable.sum())}


if __name__ == "__main__":
    run()
