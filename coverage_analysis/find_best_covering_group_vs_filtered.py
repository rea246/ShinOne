# -*- coding: utf-8 -*-
"""
find_best_covering_group_vs_filtered.py
=======================================
find_best_covering_group.py 의 변형: 타깃(Sample2)을 'filtered.csv' 로 대체한다.

배경
    원본(find_best_covering_group.py)은 Sample2 의 kpca_sample_scatter.csv 를
    gauge 단어로 선별해 타깃으로 삼고, 그 타깃을 제일 잘 덮는 Sample1 gauge 그룹을
    reach 커버리지로 랭킹했다.
    이 스크립트는 그 타깃을, kpca_reference_scatter.csv 를 '사용자가 column 값 기준으로
    필터링'해 만든 filtered.csv(= Reference 부분집합)로 대체한다.
      → 질문이 "Sample2 의 특정 gauge 를 잘 덮는 Sample1 그룹?" 에서
        "이 (필터된) Reference 영역을 잘 덮는 Sample1 그룹?" 으로 바뀐다.

일관성
    filtered.csv 는 coverage_domain_kpca.py 가 만든 kpca_reference_scatter.csv 의 부분집합
    이므로 KP1..KP3 좌표가 '동일 KPCA 프레임' 에서 왔다. Sample1 scatter 역시 같은
    프레임으로 투영된 것이어야 거리 비교가 유효하다(원본과 동일 전제).

입력 (모두 같은 Reference/KPCA 프레임에서 나온 KP 좌표여야 함)
    kpca_reference_scatter.csv     : ystd(정규화 스케일)용 — 전체 Reference KP
    Sample1 kpca_sample_scatter.csv : 후보 그룹들(gauge_name)
    filtered.csv                   : 타깃. kpca_reference_scatter.csv 를 열 값으로 필터링한 부분집합
                                     (KP1..KP3 열 필수. gauge_name 불필요)

방법
    filtered.csv 의 각 포인트를, 각 Sample1 gauge 그룹 패턴이 반경 R 안에서 덮는 비율
    (= target_coverage) 로 그룹 랭킹. (reach 커버리지 — 원본과 동일 정의)

산출물
    group_target_coverage_filtered.csv : 그룹별 target_coverage / mean_nn_dist / n_patterns (내림차순)
    group_target_scatter_filtered.png  : 타깃(주황) + 최상위 그룹(색상) 겹침
    group_top{N}_covered_targets.csv   : 그룹마다 '가장 잘 덮는'(최근접) target 패턴 N종을
                                         INPUT TARGET 행 그대로 + group/rank/nn_dist/covered
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
REF_SCATTER     = "out_s1/kpca_reference_scatter.csv"   # ystd 용(없으면 sample1+타깃 합으로 대체)
SAMPLE1_SCATTER = "out_s1/kpca_sample_scatter.csv"      # 후보 그룹들
TARGET_FILTERED = "filtered.csv"                        # 타깃 = kpca_reference_scatter.csv 필터 결과

GAUGE_COL    = "gauge_name"   # Sample1 그룹 구분용(타깃엔 불필요)
# Sample1 후보 그룹 정의: 단어 리스트를 주면 각 단어=한 그룹; None 이면 gauge_name 접두어로 자동
SAMPLE1_GROUP_WORDS = None
GROUP_DELIM  = "_"           # 자동 그룹화 시 gauge_name 을 이 구분자로 나눠 앞부분을 그룹키로

# ---- (선택) filtered.csv 를 스크립트 안에서 '추가로' 열-값 필터 ----
#   이미 밖에서 걸러 filtered.csv 를 만들었다면 None 로 두어 그대로 사용.
#   예) FILTER_COL="cover_status", FILTER_VALUES=["gap"]  → 그 값 행만 타깃으로.
FILTER_COL    = None
FILTER_VALUES = []

KP_COLS   = ["KP1", "KP2", "KP3"]
RADIUS    = 0.2666           # 정규화 KP 반경 — coverage_domain_kpca run 리포트의 R 값과 맞출 것
#   ⚠️ 모든 그룹이 ~100%로 포화(순위 구분 사라짐)하면 KP 퍼짐 대비 R 이 큼 → 값을 낮춰 재실행.
TOPN_PLOT = 1                # 플롯에 강조할 상위 그룹 수
TOPN_PER_GROUP = 3           # 그룹마다 '가장 잘 덮는' target 패턴 몇 개를 원본행 그대로 뽑을지
OUTPUT_DIR = "coverage_plots"
sns.set_theme(style="white", context="talk")


def _read_table(path):
    """CSV/TSV 자동 판별(구분자 sniff) + 열 이름 공백 제거.
    filtered.csv 가 탭 구분(TSV)이어도, 콤마여도 동일하게 읽는다."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _match_any(series, words):
    s = series.astype(str).str.lower()
    m = np.zeros(len(series), dtype=bool)
    for w in words:
        m |= s.str.contains(str(w).lower(), regex=False, na=False).to_numpy()
    return m


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    S1 = _read_table(SAMPLE1_SCATTER)
    T_all = _read_table(TARGET_FILTERED)
    # Sample1 은 KP + gauge 필요, 타깃(filtered)은 KP 만 필요
    for c in KP_COLS + [GAUGE_COL]:
        if c not in S1.columns:
            raise ValueError(f"Sample1 에 '{c}' 열 없음 (열: {list(S1.columns)})")
    for c in KP_COLS:
        if c not in T_all.columns:
            raise ValueError(f"타깃(filtered.csv)에 '{c}' 열 없음 (열: {list(T_all.columns)})")

    # (선택) 스크립트 내 추가 열-값 필터
    if FILTER_COL is not None:
        if FILTER_COL not in T_all.columns:
            raise ValueError(f"FILTER_COL '{FILTER_COL}' 이 filtered.csv 에 없음 (열: {list(T_all.columns)})")
        keep = T_all[FILTER_COL].astype(str).isin([str(v) for v in FILTER_VALUES]).to_numpy()
        print(f"[Filter] {FILTER_COL} ∈ {FILTER_VALUES}: {int(keep.sum()):,}/{len(T_all):,} 행 선택")
        T_all = T_all[keep]

    # ystd (정규화 스케일) — 전체 Reference 기준. 없으면 sample1+타깃 합으로 추정
    if REF_SCATTER and os.path.exists(REF_SCATTER):
        A = _read_table(REF_SCATTER)
        ystd = A[KP_COLS].to_numpy(float).std(axis=0)
    else:
        both = pd.concat([S1[KP_COLS], T_all[KP_COLS]])
        ystd = both.to_numpy(float).std(axis=0)
        print("[warn] REF_SCATTER 없음 → sample1+타깃 합으로 ystd 추정")
    ystd[ystd == 0] = 1.0

    # 타깃 = filtered.csv 의 유효 KP 포인트 전부 (gauge 선별 없음)
    Tk = T_all[KP_COLS].to_numpy(float); Tf = np.isfinite(Tk).all(axis=1)
    Tn = Tk[Tf] / ystd
    if len(Tn) == 0:
        raise ValueError("타깃(filtered.csv)에 유효 KP 포인트 없음")
    # Tn(순서) 과 1:1 정렬된 '원본 target 행' — top-N 출력 시 이 행을 그대로 뽑는다
    T_use = T_all.iloc[np.where(Tf)[0]].reset_index(drop=True)
    print(f"[Target] filtered.csv: {len(Tn):,} 포인트 (유효행 기준)")

    # Sample1 후보 그룹 정의
    if SAMPLE1_GROUP_WORDS is not None:
        groups = {str(w): _match_any(S1[GAUGE_COL], [w]) for w in SAMPLE1_GROUP_WORDS}
    else:
        key = S1[GAUGE_COL].astype(str).apply(lambda x: x.split(GROUP_DELIM)[0])
        groups = {g: (key == g).to_numpy() for g in sorted(key.unique())}
    print(f"[Groups] Sample1 후보 그룹 {len(groups)}개")

    # 그룹별 타깃 커버리지 (반경 R 내에 그룹 패턴이 하나라도 있으면 그 타깃점은 '덮임')
    rows = []
    top_parts = []                                  # 그룹별 top-N target 원본행 모음
    for g, gm in groups.items():
        Gk = S1.loc[gm, KP_COLS].to_numpy(float); Gf = np.isfinite(Gk).all(axis=1)
        Gn = Gk[Gf] / ystd
        if len(Gn) == 0:
            continue
        d, _ = cKDTree(Gn).query(Tn, k=1)          # 타깃 → 이 그룹 최근접 거리
        rows.append({"group": g, "n_patterns": int(len(Gn)),
                     "target_coverage_pct": round(float((d <= RADIUS).mean()) * 100, 3),
                     "mean_nn_dist": round(float(d.mean()), 5)})
        # '가장 잘 덮는' = 최근접 거리 최소. top-N target 을 원본행 그대로 추출
        n_top = int(min(TOPN_PER_GROUP, len(Tn)))
        order = np.argsort(d, kind="stable")[:n_top]
        part = T_use.iloc[order].copy()
        part.insert(0, "group", g)
        part.insert(1, "rank_in_group", np.arange(1, n_top + 1))
        part.insert(2, "nn_dist", np.round(d[order], 5))
        part.insert(3, "covered", d[order] <= RADIUS)
        top_parts.append(part)
    res = pd.DataFrame(rows).sort_values("target_coverage_pct", ascending=False).reset_index(drop=True)
    res.to_csv(j("group_target_coverage_filtered.csv"), index=False)

    # 그룹별 top-N target 원본행 (INPUT TARGET 열 그대로 + group/rank/nn_dist/covered 앞에 부착)
    if top_parts:
        top_df = pd.concat(top_parts, axis=0, ignore_index=True)
        top_df.to_csv(j(f"group_top{TOPN_PER_GROUP}_covered_targets.csv"), index=False)
        print(f"[Save] group_top{TOPN_PER_GROUP}_covered_targets.csv "
              f"({len(groups)}그룹 × 최대 {TOPN_PER_GROUP}행 = {len(top_df):,}행)")

    # 시각화
    Tplot = Tk[Tf]
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    ax.scatter(S1["KP1"], S1["KP2"], s=5, c="#DDDDDD", alpha=0.3, linewidths=0, label="Sample1 (all)")
    ax.scatter(Tplot[:, 0], Tplot[:, 1], s=16, c="#F4A15A", alpha=0.7, linewidths=0,
               label=f"Target (filtered.csv, n={len(Tn)})")
    pal = sns.color_palette("Dark2", max(TOPN_PLOT, 1))
    for i, g in enumerate(res["group"].head(TOPN_PLOT)):
        gm = groups[g]
        ax.scatter(S1.loc[gm, "KP1"], S1.loc[gm, "KP2"], s=22, color=pal[i], alpha=0.9,
                   linewidths=0, label=f"S1 '{g}' (cov {res.iloc[i]['target_coverage_pct']:.0f}%)")
    ax.set_xlabel("Kernel PC1"); ax.set_ylabel("Kernel PC2")
    ax.set_title(f"Which Sample1 gauge group best covers the filtered Reference target?\n"
                 f"target = filtered.csv ({len(Tn)} pts), R={RADIUS}")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout(); fig.savefig(j("group_target_scatter_filtered.png"), dpi=130); plt.close(fig)

    print("\n" + "=" * 60)
    print("  [Best covering group] filtered.csv 타깃을 잘 덮는 Sample1 그룹")
    print("=" * 60)
    for _, r in res.head(8).iterrows():
        print(f"   {r['group']:>12}: target_cov {r['target_coverage_pct']:6.2f}%  "
              f"(n={int(r['n_patterns'])}, meanNN={r['mean_nn_dist']:.3f})")
    print("  → group_target_coverage_filtered.csv / group_target_scatter_filtered.png")
    print(f"  → group_top{TOPN_PER_GROUP}_covered_targets.csv (그룹별 최근접 target {TOPN_PER_GROUP}종, 원본행)")
    print("=" * 60)
    return res


if __name__ == "__main__":
    run()
