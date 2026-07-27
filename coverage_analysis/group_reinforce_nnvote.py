# -*- coding: utf-8 -*-
"""
group_reinforce_nnvote.py
=========================
[R-프리 그룹 중요도] target 패턴별 '역거리 가중 top-K 투표' 로 Sample1 그룹 랭킹.

목적
    filtered(target) 패턴 하나하나가 '이 영역을 강화하려면 어느 게이지 그룹을 늘려야
    하나' 를 투표한다. 반경 R(임계값)을 전혀 쓰지 않고, 최근접 거리 그 자체로 가중.

방법 (매우 직관적)
    * filtered 패턴은 한정적 → 각 패턴에 점수 1점을 부여.
    * 그 패턴의 최근접 Sample1 패턴 top-K 를 찾는다(기본 K=3).
    * 1점을 top-K 에 '거리의 역수' 비율로 나눠 준다(정규화).
        예) 1/2/3위 거리가 1, 2, 4 이면 역수 1, 0.5, 0.25 → 합 1.75 로 정규화
            → 0.571 / 0.286 / 0.143.  (가까울수록 큰 점수)
    * 각 Sample1 패턴은 게이지 그룹에 속하므로, 전 target 에 걸쳐
        '그룹이 받은 점수 합' = 그룹 중요도.
      (전 그룹 점수 합 = target 개수. score_pct = 그룹이 가져간 '관심'의 비율)

산출물
    target_top{K}_sample1_match.csv : target 원본행 + 최근접 top-K (group/gauge/dist/score)
    group_reinforce_nnvote_score.csv: 그룹별 total_score / score_pct / n_rank1 /
                                      n_targets / n_points / score_per_pattern (내림차순)
"""

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =============================================================================
# CONFIG
# =============================================================================
REF_SCATTER     = "out_s1/kpca_reference_scatter.csv"   # ystd(정규화 스케일)용(없으면 sample1+타깃 합)
SAMPLE1_SCATTER = "out_s1/kpca_sample_scatter.csv"      # 후보 그룹들(gauge_name)
TARGET_FILTERED = "filtered.csv"                        # 타깃 = kpca_reference_scatter.csv 필터 결과

GAUGE_COL    = "gauge_name"   # Sample1 그룹 구분용(타깃엔 불필요)
# Sample1 그룹 정의: 단어 리스트를 주면 각 점을 '처음 일치하는 단어' 그룹으로(그 외 'other');
#   None 이면 gauge_name 을 GROUP_DELIM 로 나눈 앞부분을 그룹키로.
SAMPLE1_GROUP_WORDS = None
GROUP_DELIM  = "_"

# ---- (선택) filtered.csv 를 스크립트 안에서 '추가로' 열-값 필터 ----
FILTER_COL    = None
FILTER_VALUES = []

KP_COLS   = ["KP1", "KP2", "KP3"]
TOPK      = 3                 # target 마다 점수 1점을 나눠 줄 최근접 Sample1 패턴 수
USE_YSTD  = True              # True=참조 KP 스케일(ystd)로 정규화한 거리 사용(파이프라인 일관)
EPS       = 1e-12             # 0거리(정확 일치) 안전 처리용
OUTPUT_DIR = "coverage_plots"


def _read_table(path):
    """CSV/TSV 자동 판별(구분자 sniff) + 열 이름 공백 제거."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def group_labels(gauge_series):
    """각 Sample1 점에 단일 그룹 라벨을 매긴다(파티션)."""
    if SAMPLE1_GROUP_WORDS is not None:
        low = gauge_series.astype(str).str.lower()
        lab = np.full(len(gauge_series), "other", dtype=object)
        # 리스트 앞 단어가 우선하도록 역순으로 덮어씀
        for w in reversed([str(x) for x in SAMPLE1_GROUP_WORDS]):
            m = low.str.contains(w.lower(), regex=False, na=False).to_numpy()
            lab[m] = str(w)
        return lab
    return gauge_series.astype(str).apply(lambda x: x.split(GROUP_DELIM)[0]).to_numpy()


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    j = lambda f: os.path.join(OUTPUT_DIR, f)

    S1 = _read_table(SAMPLE1_SCATTER)
    T_all = _read_table(TARGET_FILTERED)
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

    # ystd (정규화 스케일)
    if USE_YSTD and REF_SCATTER and os.path.exists(REF_SCATTER):
        A = _read_table(REF_SCATTER)
        ystd = A[KP_COLS].to_numpy(float).std(axis=0)
    elif USE_YSTD:
        both = pd.concat([S1[KP_COLS], T_all[KP_COLS]])
        ystd = both.to_numpy(float).std(axis=0)
        print("[warn] REF_SCATTER 없음 → sample1+타깃 합으로 ystd 추정")
    else:
        ystd = np.ones(len(KP_COLS))
    ystd = np.asarray(ystd, float); ystd[ystd == 0] = 1.0

    # Sample1 유효 점 + 그룹 라벨 + gauge/원본인덱스
    S1k = S1[KP_COLS].to_numpy(float); S1f = np.isfinite(S1k).all(axis=1)
    Sn = S1k[S1f] / ystd
    s1_lab   = group_labels(S1[GAUGE_COL])[S1f]
    s1_gauge = S1[GAUGE_COL].astype(str).to_numpy()[S1f]
    s1_oidx  = np.where(S1f)[0]                        # Sample1 원본 행번호(추적용)
    if len(Sn) == 0:
        raise ValueError("Sample1 에 유효 KP 점 없음")

    # target 유효 점 + 원본행 정렬
    Tk = T_all[KP_COLS].to_numpy(float); Tf = np.isfinite(Tk).all(axis=1)
    Tn = Tk[Tf] / ystd
    T_use = T_all.iloc[np.where(Tf)[0]].reset_index(drop=True)
    if len(Tn) == 0:
        raise ValueError("타깃(filtered.csv)에 유효 KP 점 없음")

    k = int(min(TOPK, len(Sn)))
    print(f"[Match] target {len(Tn):,}개 → Sample1 {len(Sn):,}점 중 최근접 top-{k} (거리 역수 가중)")

    # 최근접 top-k 거리/인덱스
    dist, idx = cKDTree(Sn).query(Tn, k=k)
    if k == 1:                                          # k=1 이면 1D → 2D로
        dist = dist[:, None]; idx = idx[:, None]
    dist = np.atleast_2d(dist); idx = np.atleast_2d(idx)

    # 거리 역수 정규화 (행 합 = 1). 0거리는 EPS로 그 항이 지배.
    w = 1.0 / (dist + EPS)
    w = w / w.sum(axis=1, keepdims=True)

    # ---- target 별 top-k 매칭 원본행 출력 ----
    out = T_use.copy()
    for r in range(k):
        gi = idx[:, r]
        out[f"nn{r+1}_group"] = s1_lab[gi]
        out[f"nn{r+1}_gauge"] = s1_gauge[gi]
        out[f"nn{r+1}_s1idx"] = s1_oidx[gi]
        out[f"nn{r+1}_dist"]  = np.round(dist[:, r], 6)
        out[f"nn{r+1}_score"] = np.round(w[:, r], 6)
    out.to_csv(j(f"target_top{k}_sample1_match.csv"), index=False)

    # ---- 그룹 점수 집계 ----
    lab_flat = s1_lab[idx.ravel()]
    w_flat   = w.ravel()
    tgt_flat = np.repeat(np.arange(len(Tn)), k)
    agg = pd.DataFrame({"group": lab_flat, "score": w_flat, "t": tgt_flat})
    total_score = agg.groupby("group")["score"].sum()
    n_targets   = agg.drop_duplicates(["group", "t"]).groupby("group")["t"].count()
    rank1_lab   = s1_lab[idx[:, 0]]
    n_rank1     = pd.Series(rank1_lab).value_counts()
    uniq, cnt   = np.unique(s1_lab, return_counts=True)
    n_points    = pd.Series(cnt, index=uniq)

    nT = len(Tn)
    rows = []
    for g in n_points.index:                            # 모든 Sample1 그룹(0점 포함)
        ts = float(total_score.get(g, 0.0))
        npt = int(n_points.get(g, 0))
        rows.append({
            "group": g,
            "total_score": round(ts, 5),
            "score_pct": round(ts / nT * 100, 3),        # 전체 target 관심 중 비율
            "n_rank1": int(n_rank1.get(g, 0)),           # 1순위로 뽑힌 횟수
            "n_targets": int(n_targets.get(g, 0)),       # 점수를 받은 서로 다른 target 수
            "n_points": npt,                             # 그룹 패턴 수
            "score_per_pattern": round(ts / npt, 6) if npt else 0.0,  # 카운트 통제(패턴당 효율)
        })
    res = pd.DataFrame(rows).sort_values("total_score", ascending=False).reset_index(drop=True)
    res.to_csv(j("group_reinforce_nnvote_score.csv"), index=False)

    print("\n" + "=" * 66)
    print(f"  [Reinforce importance] target {nT:,}개의 역거리 top-{k} 투표 → 그룹 랭킹")
    print("=" * 66)
    print(f"  {'group':>14} {'total':>8} {'pct%':>7} {'rank1':>7} {'nTgt':>7} {'nPts':>7} {'perPat':>8}")
    for _, r in res.head(10).iterrows():
        print(f"  {str(r['group'])[:14]:>14} {r['total_score']:8.2f} {r['score_pct']:7.2f} "
              f"{int(r['n_rank1']):7d} {int(r['n_targets']):7d} {int(r['n_points']):7d} "
              f"{r['score_per_pattern']:8.4f}")
    print("  → target_top{}_sample1_match.csv / group_reinforce_nnvote_score.csv".format(k))
    print("  (total_score=능력(카운트 포함), score_per_pattern=패턴당 효율(카운트 통제))")
    print("=" * 66)
    return res


if __name__ == "__main__":
    run()
