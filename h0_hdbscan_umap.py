"""
h0_hdbscan_umap.py — h0(중심 hop) 임베딩으로 HDBSCAN "첫 군집화"

목적 (조건부 트리의 "뿌리(h0)"를 밀도 기반 HDBSCAN 으로 만드는 첫 단계):
  1) h0 8d 임베딩 → effective 차원(기본 4d PCA)으로 축소 → HDBSCAN 밀도 군집
     → 자연 군집 수 / noise 비율 실측 (여기까지가 "첫 군집화"의 핵심)
  2) 최적 HDBSCAN 파라미터 탐색(--sweep): (mcs, min_samples, epsilon) 그리드를
     같은 표본 위에서 돌려 DBCV/silhouette/noise 로 best 조합 추천
  3) 결과 2D scatter + 군집 대표패턴(선택적 눈 검증)

데이터 출처 (중요): 이 스크립트는 GDS DB 를 직접 긁지 않는다. 파이프라인은
  GDS(klayout) → preprocessor.py → PREPROCESSED_*part*.pt 청크
  → classify_tree.py(GAE 인코더 추론) → hkeys_features.pt 캐시 → (여기)
  즉 h0 "8차원" 은 원기하가 아니라 GAE 가 학습한 임베딩이다. 군집 품질은
  GAE 인코더 품질에 종속된다.

차원 축소 (핵심): h0 8d 는 effective rank ≈ 4/8 (특이값 엔트로피 기준) 로
  8차원 중 절반만 실제 정보를 담는다. 남는 4개 축은 밀도 추정에 노이즈로만
  작용(거리 집중 현상) → HDBSCAN 을 흐린다. 그래서 표준화 후 PCA 로
  effective 차원(기본 EFF_DIM=4)만 남겨 군집화한다. 실측 effective rank 를
  실행 시 출력하니 EFF_DIM 을 데이터로 확인/조정할 것.
  ※ 축소는 "없던 구조를 만들지 않는다" — 실재하는 밀도 골짜기만 또렷하게 한다.

군집화 공간 CLUSTER_SPACE:
  'pca'(기본) → 표준화 8d → PCA(EFF_DIM). effective 차원만 남긴 정직한 축소.
  'raw'       → 표준화 8d 원공간 직접 (축소 없이 비교용).
  'umap'      → UMAP(min_dist=0) 압축. 약한 골짜기 증폭 — 탐지력↑ 이지만
                연속 데이터에서 가짜 군집 위험 → 대표패턴 눈 검증 필수.

표본  : HDBSCAN 은 초선형 비용+메모리라 전량(20M) 불가 → HDB_SAMPLE 만큼
        전체에서 균일추출해 구조를 "발견"한다 (밀도 구조는 분포 성질이라
        균일 표본으로 불편추정; 전량 라벨 배정은 --assign-all 로 다음 단계).
        탐지 하한: 전체 환산 (MIN_CLUSTER_SIZE/HDB_SAMPLE)×N 미만 군집은 안 보임.

판정 기준:
  - 군집 수가 2~수십 개 + noise < 30% + DBCV↑ → h0 에 진짜 밀도 구조 있음
    (조건부 트리의 h0 층을 HDBSCAN 으로 교체할 근거)
  - 군집 1~2개 + noise 많음 → h0 도 연속체 (KMeans 양자화가 정직한 방법)

사용 예:
  python h0_hdbscan_umap.py                          # 표준화→PCA4→HDBSCAN
  python h0_hdbscan_umap.py --sweep                  # best 파라미터 탐색
  python h0_hdbscan_umap.py --eff-dim 4 --space pca  # 명시적 4d 축소
  python h0_hdbscan_umap.py --space raw              # 축소 없이 8d 비교

의존: hdbscan(권장, DBCV·approximate_predict), umap-learn(--space umap/--viz umap).
      없으면 각각 sklearn.cluster.HDBSCAN / PCA 로 대체.
"""

import os
import csv
import time
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ══════════════════════════════════════════════════════════════════
# [Config]  — classify_tree 와 동일 데이터 경로
# ══════════════════════════════════════════════════════════════════
FOLDER_PATH   = "C:/Users/rea24/Documents/shinwon_note/pattern/2.Pattern_Classification/3.CODE/pythonProject2/dummy_dataset"
FEATURE_CACHE = "hkeys_features.pt"
OUT_DIR       = "h0_hdbscan_viz"

# 표본: UMAP/HDBSCAN 은 초선형+대용량 메모리라 20M 전량이 물리적으로 불가 →
#   전체 유효행에서 "균일 무작위" 추출(불편추정). 탐지 하한 주의:
#   전체 환산 (MIN_CLUSTER_SIZE/HDB_SAMPLE)×N 미만의 군집은 원리적으로 안 보임.
HDB_SAMPLE       = 300_000
# 군집화 공간: h0 8d 는 effective rank ≈ 4/8 → 남는 축이 밀도 추정에 노이즈로만
#   작용. 'pca'(기본) = 표준화 후 PCA 로 effective 차원만 남긴 정직한 축소.
#   'raw' = 축소 없는 8d(비교용). 'umap' = min_dist=0 압축(가짜 군집 위험).
CLUSTER_SPACE    = "pca"     # 'pca' | 'raw' | 'umap'
EFF_DIM          = 4         # PCA 축소 차원 (h0 의 effective rank ≈ 4). 실행 시
                             #   실측 effective rank 를 출력하니 데이터로 확인할 것.
UMAP_DIM         = 5         # 군집화용 UMAP 차원 (2 는 과압축 — 5~10 권장)
UMAP_NEIGHBORS   = 30
UMAP_MIN_DIST    = 0.0       # 군집화용: 0.0 = 밀도 응집 최대
MIN_CLUSTER_SIZE = 1_000     # HDBSCAN 최소 군집 크기 (표본의 ~0.3%)
# 군집 커버리지(noise 비율) 조절 손잡이 2개:
#   MIN_SAMPLES ↓ (예: 25~100) → 밀도 추정이 관대해져 noise ↓, 군집이 넓어짐
#   HDB_EPSILON ↑ (예: 0.1~0.5, 군집화 공간 거리 단위) → 경계 확장/저밀도 흡수
MIN_SAMPLES      = None      # None = min_cluster_size 와 동일 (보수적 기본)
HDB_EPSILON      = 0.0       # cluster_selection_epsilon
N_GROUPS_PLOT    = 12        # 대표패턴을 그릴 상위 군집 수
N_REPS           = 5         # 군집당 대표 수 (medoid 1 + 랜덤 4)
SCATTER_MAX      = 200_000   # scatter 렌더 점 수 상한
SEED             = 42

HOP_COLORS = {0: "#e6194B", 1: "#f58231", 2: "#3cb44b", 3: "#9aa0a6"}
_here = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════════
# [캐시 로드 / 청크 역참조]  (claasify_tree 와 동일 포맷)
# ══════════════════════════════════════════════════════════════════
def load_cache():
    cache = os.path.join(_here, FEATURE_CACHE)
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"feature 캐시 없음: {cache} — 5.classify_tree.py 를 먼저 실행하세요")
    d = torch.load(cache, map_location="cpu", weights_only=False)
    return d["features"], d["keys"], d["chunks"]


def _row_to_file(chunks, folder, row):
    off = 0
    for base, cnt in chunks:
        if row < off + cnt:
            return os.path.join(folder, base), row - off
        off += cnt
    return None, None


def _extract_graphs(chunks, folder, rows) -> dict:
    """global row 집합 → {row: {'x','hop'}}. 청크 동시 1개만 로드."""
    byfile = {}
    for r in set(int(v) for v in rows):
        f, li = _row_to_file(chunks, folder, r)
        if f is not None:
            byfile.setdefault(f, []).append((r, li))
    out = {}
    for f, pairs in byfile.items():
        if not os.path.exists(f):
            continue
        ch = torch.load(f, map_location="cpu", weights_only=False)
        for r, li in pairs:
            x0, x1, _, _ = [int(v) for v in ch["meta"][li].tolist()]
            x = ch["x"][x0:x1].float().numpy()
            packed = ch["hop_packed"][x0:x1].numpy().astype(np.int64)
            hop = np.full(len(packed), 3, dtype=int)
            hop[((packed >> 2) & 1).astype(bool)] = 2
            hop[((packed >> 1) & 1).astype(bool)] = 1
            hop[(packed & 1).astype(bool)] = 0
            out[r] = {"x": x, "hop": hop}
        del ch
    return out


# ══════════════════════════════════════════════════════════════════
# [차원 진단 / 축소]
# ══════════════════════════════════════════════════════════════════
def effective_rank(X):
    """특이값 엔트로피 기반 effective rank (Roy & Vetterli 2007).
    erank = exp(H),  H = -Σ p_i ln p_i,  p_i = σ_i / Σσ_j.
    X 는 이미 표준화된 (n, d). '8차원 중 몇 축이 실제로 정보를 담는가'의 척도."""
    # 큰 표본은 SVD 가 비싸므로 특이값만 필요 → 최대 50k 행으로 충분(안정적)
    Xs = X if len(X) <= 50_000 else X[np.random.default_rng(0).choice(
        len(X), 50_000, replace=False)]
    sv = np.linalg.svd(Xs - Xs.mean(0), compute_uv=False)
    p = sv / sv.sum()
    p = p[p > 0]
    H = -(p * np.log(p)).sum()
    return float(np.exp(H)), sv


def pca_reduce(X, n_components, seed):
    """표준화된 X → PCA(n_components). 화이트닝 없음(신호 축의 분산 순서를
    보존해야 HDBSCAN 거리에서 신호 축이 지배함). 반환 (Z, model)."""
    model = PCA(n_components=n_components, random_state=seed).fit(X)
    evr = model.explained_variance_ratio_
    print(f"  [PCA] {X.shape[1]}d → {n_components}d  누적 설명분산 "
          f"{evr[:n_components].sum():.1%}  (축별 {np.round(evr, 3).tolist()})")
    return model.transform(X).astype(np.float32), model


# ══════════════════════════════════════════════════════════════════
# [UMAP / HDBSCAN — 미설치 fallback 포함]
# ══════════════════════════════════════════════════════════════════
def umap_embed(X, n_components, min_dist, seed):
    """UMAP 임베딩. umap-learn 없으면 PCA 로 대체(경고).
    반환 (embedding, impl, model) — model 은 전량 배정 시 transform 용."""
    try:
        import umap
        t = time.time()
        model = umap.UMAP(n_components=n_components, n_neighbors=UMAP_NEIGHBORS,
                          min_dist=min_dist, random_state=seed, verbose=True)
        emb = model.fit_transform(X)
        print(f"  [UMAP] {X.shape} → {n_components}d  ({time.time()-t:.0f}s)")
        return emb, "umap", model
    except ImportError:
        print("  ⚠ umap-learn 미설치 → PCA 대체 (pip install umap-learn 권장)")
        model = PCA(n_components=n_components, random_state=seed).fit(X)
        return model.transform(X), "pca", model


def run_hdbscan(X, min_cluster_size, min_samples, epsilon=0.0):
    """HDBSCAN fit (발견 단계). hdbscan 패키지 우선(DBCV·approximate_predict 제공),
    없으면 sklearn 내장. 반환 (labels, dbcv, clusterer, impl)."""
    t = time.time()
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                             min_samples=min_samples,
                             cluster_selection_epsilon=float(epsilon),
                             gen_min_span_tree=True, prediction_data=True,
                             core_dist_n_jobs=-1)
        lab = cl.fit_predict(X)
        dbcv = float(getattr(cl, "relative_validity_", float("nan")))
        impl = "hdbscan"
    except ImportError:
        from sklearn.cluster import HDBSCAN as SKHDBSCAN   # sklearn>=1.3
        cl = SKHDBSCAN(min_cluster_size=min_cluster_size,
                       min_samples=min_samples,
                       cluster_selection_epsilon=float(epsilon))
        lab = cl.fit_predict(X)
        dbcv, impl = float("nan"), "sklearn"
    print(f"  [HDBSCAN/{impl}] fit {X.shape}  min_samples={min_samples} "
          f"eps={epsilon}  ({time.time()-t:.0f}s)")
    return lab.astype(np.int64), dbcv, cl, impl


# ══════════════════════════════════════════════════════════════════
# [파라미터 스윕 — 최적 HDBSCAN 탐색]
#   min_cluster_size × min_samples × epsilon 그리드를 같은 표본 위에서 순차
#   실행, 조합마다 (군집 수 / noise / silhouette / DBCV / 크기 균형)을 기록.
#   LSF 병렬 분할: 그리드 인자를 job 별로 나눠 던지면 됨 (CSV 파일명에 그리드
#   태그가 붙어 서로 안 덮어씀; 끝나고 CSV 들을 이어붙여 비교).
#   예) bsub ... python h0_hdbscan_umap.py --sweep --sweep-mcs 500
#       bsub ... python h0_hdbscan_umap.py --sweep --sweep-mcs 2000,5000
# ══════════════════════════════════════════════════════════════════
def _parse_list(s, cast):
    out = []
    for tok in s.split(','):
        tok = tok.strip().lower()
        out.append(None if tok in ('none', 'auto') else cast(tok))
    return out


def _cluster_metrics(lab, X_cl, rng):
    """조합 하나의 평가 지표 묶음."""
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    k, noise = len(ids), float((lab < 0).mean())
    sil = float('nan')
    if k >= 2:
        idx = np.where(lab >= 0)[0]
        s_idx = idx if len(idx) <= 10_000 else rng.choice(idx, 10_000, replace=False)
        try:
            sil = float(silhouette_score(X_cl[s_idx], lab[s_idx]))
        except ValueError:
            pass
    if k:
        p = cnts / cnts.sum()
        eff_k = float(np.exp(-(p * np.log(p)).sum()))
        max_share = float(cnts.max() / cnts.sum())
        top5 = np.sort(cnts)[::-1][:5].tolist()
    else:
        eff_k, max_share, top5 = 0.0, 0.0, []
    return {'k': k, 'noise': noise, 'sil': sil, 'eff_k': eff_k,
            'max_share': max_share, 'sizes_top5': top5}


def run_sweep(X_cl, rng, args, out_dir):
    import itertools
    mcs_list = _parse_list(args.sweep_mcs, int)
    ms_list = _parse_list(args.sweep_ms, int)
    eps_list = _parse_list(args.sweep_eps, float)
    combos = list(itertools.product(mcs_list, ms_list, eps_list))
    print(f"\n[스윕] {len(combos)}개 조합 × 표본 {len(X_cl):,} "
          f"(공간={CLUSTER_SPACE}) — 조합당 수 분")

    results = []
    for i, (mcs, ms, eps) in enumerate(combos):
        t = time.time()
        lab, dbcv, _, _ = run_hdbscan(X_cl, mcs, ms, eps or 0.0)
        m = _cluster_metrics(lab, X_cl, rng)
        m.update(mcs=mcs, ms=ms, eps=(eps or 0.0), dbcv=dbcv,
                 sec=round(time.time() - t, 1))
        results.append(m)
        print(f"  [{i+1}/{len(combos)}] mcs={mcs} ms={ms} eps={eps or 0.0} → "
              f"k={m['k']}  noise={m['noise']:.1%}  sil={m['sil']:.3f}  "
              f"DBCV={m['dbcv']:.3f}  최대군집 {m['max_share']:.0%}  ({m['sec']}s)")

    # ── CSV (그리드 태그 포함 — LSF 분할 실행 시 서로 안 덮어씀) ──
    tag = (f"mcs{args.sweep_mcs}_ms{args.sweep_ms}_eps{args.sweep_eps}"
           .replace(',', '-').replace('.', 'p'))
    csv_path = os.path.join(out_dir, f"h0_hdbscan_sweep_{tag}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as fp:
        w = csv.writer(fp)
        w.writerow(['min_cluster_size', 'min_samples', 'epsilon', 'n_clusters',
                    'noise_frac', 'silhouette', 'dbcv', 'eff_k', 'max_share',
                    'sizes_top5', 'sec'])
        for r in results:
            w.writerow([r['mcs'], ('auto' if r['ms'] is None else r['ms']),
                        r['eps'], r['k'], f"{r['noise']:.4f}",
                        f"{r['sil']:.4f}", f"{r['dbcv']:.4f}",
                        f"{r['eff_k']:.2f}", f"{r['max_share']:.4f}",
                        '/'.join(map(str, r['sizes_top5'])), r['sec']])
    print(f"\n  [스윕] CSV: {csv_path}")

    # ── 비교 plot: noise vs sil / noise vs DBCV ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, key, name in ((axes[0], 'sil', 'silhouette (noise excl.)'),
                          (axes[1], 'dbcv', 'DBCV (relative validity)')):
        for r in results:
            if np.isnan(r[key]):
                continue
            ax.scatter(r['noise'], r[key], s=30 + 12 * r['k'], alpha=0.7)
            ax.annotate(f"{r['mcs']}/{('a' if r['ms'] is None else r['ms'])}"
                        f"/{r['eps']:g}\nk={r['k']}",
                        (r['noise'], r[key]), fontsize=6.5,
                        textcoords="offset points", xytext=(5, 3))
        ax.set_xlabel("noise fraction (lower-left is better)")
        ax.set_ylabel(name)
        ax.set_title(f"{name} vs noise | label = mcs/min_samples/eps, "
                     f"marker size ~ n_clusters")
        ax.grid(True, ls=':', alpha=0.4)
    fig.suptitle("HDBSCAN parameter sweep - pick upper-left with sensible k",
                 fontsize=12)
    fig.tight_layout()
    png = os.path.join(out_dir, f"h0_hdbscan_sweep_{tag}.png")
    fig.savefig(png, dpi=120)
    plt.close(fig)
    print(f"  [스윕] plot: {png}")

    # ── 추천: noise≤30% & 3≤k≤50 중 DBCV(없으면 sil) 최고 ──
    def score(r):
        return r['dbcv'] if not np.isnan(r['dbcv']) else \
            (r['sil'] if not np.isnan(r['sil']) else -9)
    ok = [r for r in results if r['noise'] <= 0.30 and 3 <= r['k'] <= 50]
    pool = ok if ok else results
    ranked = sorted(pool, key=score, reverse=True)
    print("\n  [순위] (제약: noise≤30%, 3≤k≤50" +
          ("" if ok else " — 만족 조합 없음, 전체에서 순위") + ")")
    for r in ranked[:8]:
        print(f"    mcs={r['mcs']:>5} ms={('auto' if r['ms'] is None else r['ms']):>5} "
              f"eps={r['eps']:g}  k={r['k']:>3}  noise={r['noise']:.1%}  "
              f"sil={r['sil']:.3f}  DBCV={r['dbcv']:.3f}  최대 {r['max_share']:.0%}")
    b = ranked[0]
    ms_arg = "" if b['ms'] is None else f" --min-samples {b['ms']}"
    eps_arg = "" if b['eps'] == 0 else f" --epsilon {b['eps']:g}"
    print(f"\n  [추천] 재현+전량배정:\n    python h0_hdbscan_umap.py "
          f"--space {CLUSTER_SPACE} --eff-dim {args.eff_dim} "
          f"--min-cluster-size {b['mcs']}{ms_arg}{eps_arg} "
          f"--assign-all --viz pca")
    return results


# ══════════════════════════════════════════════════════════════════
# [전량 배정 — 발견(표본 fit) 후 전체 DB 라벨링 (선형, predict 만)]
#   KMeans 파이프라인의 FIT_SAMPLE fit → 전량 predict 와 같은 2단계 구조.
#   hdbscan 패키지: approximate_predict (condensed tree 기반, noise 포함 판정)
#   sklearn 내장 : 최근접 centroid + 군집별 반경(멤버 거리 95% 분위) 밖 = noise
# ══════════════════════════════════════════════════════════════════
ASSIGN_BLOCK = 1_000_000


def _fallback_assigner(X_cl, lab):
    """sklearn HDBSCAN 용 근사 배정기 (최근접 centroid + 반경 규칙)."""
    ids = np.unique(lab[lab >= 0])
    cents = np.stack([X_cl[lab == c].mean(0) for c in ids]).astype(np.float32)
    radius = np.array([np.quantile(
        np.linalg.norm(X_cl[lab == c] - cents[i], axis=1), 0.95)
        for i, c in enumerate(ids)])

    def assign(Xb):
        d2 = ((Xb ** 2).sum(1)[:, None] + (cents ** 2).sum(1)[None]
              - 2.0 * Xb @ cents.T)
        j = d2.argmin(1)
        dm = np.sqrt(np.maximum(d2[np.arange(len(Xb)), j], 0))
        out = ids[j].astype(np.int64)
        out[dm > radius[j]] = -1
        return out
    return assign


def assign_all_rows(cl, impl, scaler, h0, valid, X_cl_sample, lab_sample,
                    space_model, out_dir):
    """유효 h0 전 행에 라벨 배정 (블록 스트리밍, 선형)."""
    t = time.time()
    if impl == "hdbscan":
        import hdbscan as _h
        def assign(Xb):
            out, _ = _h.approximate_predict(cl, Xb)
            return out.astype(np.int64)
    else:
        print("  ⚠ sklearn HDBSCAN 은 approximate_predict 미지원 → "
              "최근접 centroid+반경 근사 배정")
        assign = _fallback_assigner(X_cl_sample, lab_sample)

    full = np.full(len(valid), -1, dtype=np.int64)
    for s in range(0, len(valid), ASSIGN_BLOCK):
        blk = valid[s:s + ASSIGN_BLOCK]
        Xb = scaler.transform(
            h0[torch.from_numpy(blk)].numpy().astype(np.float32))
        if space_model is not None:                 # umap 공간 군집화였다면
            Xb = space_model.transform(Xb).astype(np.float32)
        full[s:s + len(blk)] = assign(np.ascontiguousarray(Xb))
        print(f"  [배정] {min(s + len(blk), len(valid)):,}/{len(valid):,}",
              end="\r")
        del Xb
    print()

    ids, cnts = np.unique(full[full >= 0], return_counts=True)
    n_noise = int((full < 0).sum())
    print(f"  [전량 배정 완료] 유효 {len(valid):,} rows → 군집 {len(ids)}개, "
          f"noise {n_noise:,} ({n_noise/len(valid):.1%})  ({time.time()-t:.0f}s)")
    order = np.argsort(-cnts)
    print("  전량 군집 크기(상위): "
          + "  ".join(f"c{ids[j]}:{cnts[j]:,}" for j in order[:12]))
    path = os.path.join(out_dir, "h0_hdbscan_full_labels.npz")
    np.savez(path, rows=valid, labels=full)
    print(f"  [저장] {path}  (rows=global row, labels=-1 은 noise)")
    return full


# ══════════════════════════════════════════════════════════════════
# [1) 군집 요약 콘솔]
# ══════════════════════════════════════════════════════════════════
def summarize(lab, X_cl, dbcv, rng):
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    n_noise = int((lab < 0).sum())
    n = len(lab)
    print("\n" + "█" * 64)
    print("█  h0 HDBSCAN KEY METRICS")
    print("█" * 64)
    print(f"  표본 {n:,}개  →  군집 {len(ids)}개  +  noise {n_noise:,} ({n_noise/n:.1%})")
    if len(ids):
        order = np.argsort(-cnts)
        top = "  ".join(f"c{ids[j]}:{cnts[j]:,}" for j in order[:12])
        print(f"  군집 크기(상위): {top}")
        # noise 제외 silhouette (군집화 공간에서, 표본)
        mask = lab >= 0
        if mask.sum() > 1000 and len(ids) >= 2:
            idx = np.where(mask)[0]
            s_idx = idx if len(idx) <= 10_000 else rng.choice(idx, 10_000, replace=False)
            try:
                sil = float(silhouette_score(X_cl[s_idx], lab[s_idx]))
            except ValueError:
                sil = float("nan")
            print(f"  silhouette(noise 제외, 군집화 공간) = {sil:.3f}")
    if not np.isnan(dbcv):
        print(f"  DBCV(relative validity) = {dbcv:.3f}  (↑좋음, 밀도군집 전용 품질)")
    print("  판정: 군집 2개 이상 + noise<30% 면 h0 에 밀도 구조 실재 →")
    print("        조건부 트리의 h0 층을 HDBSCAN 으로 교체할 근거")
    print("█" * 64 + "\n")
    return ids, cnts


# ══════════════════════════════════════════════════════════════════
# [2) UMAP 2D scatter]
# ══════════════════════════════════════════════════════════════════
def plot_scatter(P2, lab, viz_impl, out_dir, rng):
    n = len(lab)
    idx = np.arange(n) if n <= SCATTER_MAX else np.sort(
        rng.choice(n, SCATTER_MAX, replace=False))
    ids = np.unique(lab[lab >= 0])
    colors = cm.tab20(np.linspace(0, 1, max(len(ids), 1)))

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    noise = idx[lab[idx] < 0]
    ax.scatter(P2[noise, 0], P2[noise, 1], s=2, c="#cccccc", alpha=0.3,
               linewidths=0, label=f"noise ({(lab<0).sum():,})")
    for i, cid in enumerate(ids):
        m = idx[lab[idx] == cid]
        ax.scatter(P2[m, 0], P2[m, 1], s=3, color=colors[i], alpha=0.5,
                   linewidths=0)
        # 군집 라벨: 멤버 median 위치
        allm = np.where(lab == cid)[0]
        mx, my = np.median(P2[allm, 0]), np.median(P2[allm, 1])
        ax.text(mx, my, f"c{cid}\n{len(allm):,}", fontsize=8, ha="center",
                va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i],
                          alpha=0.85))
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"h0 HDBSCAN on {CLUSTER_SPACE}-space | {len(ids)} clusters, "
                 f"noise {(lab<0).mean():.1%} | 2D={viz_impl} "
                 f"(sample {n:,}, drawn {len(idx):,})")
    ax.set_xlabel(f"{viz_impl}-1")
    ax.set_ylabel(f"{viz_impl}-2")
    ax.grid(True, ls=":", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "h0_hdbscan_umap.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [scatter] {path}")


# ══════════════════════════════════════════════════════════════════
# [3) 군집별 대표패턴 5종 렌더]
# ══════════════════════════════════════════════════════════════════
def _draw_pattern(ax, g, key):
    """h0 노드 색 강조 + 나머지 hop 회색 윤곽 (h0 기준 군집이므로)."""
    x, hop = g["x"], g["hop"]
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    for i in np.where(hop != 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               fill=False, edgecolor="#cccccc", lw=0.4))
    for i in np.where(hop == 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               facecolor=HOP_COLORS[0], edgecolor="black",
                               lw=0.5, alpha=0.8))
    h0 = np.where(hop == 0)[0]
    if len(h0):
        ax.plot(cx[h0], cy[h0], "*", color="black", ms=10,
                markerfacecolor=HOP_COLORS[0], zorder=4)
    if len(x):
        x0b = float((cx - w / 2).min()); x1b = float((cx + w / 2).max())
        y0b = float((cy - h / 2).min()); y1b = float((cy + h / 2).max())
        pad = 0.05 * max(x1b - x0b, y1b - y0b, 1.0)
        ax.set_xlim(x0b - pad, x1b + pad)
        ax.set_ylim(y0b - pad, y1b + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(key, fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.grid(True, ls=":", alpha=0.3)


def plot_representatives(lab, X_cl, samp_rows, all_keys, chunks, out_dir, rng):
    """상위 군집마다 medoid 1 + 랜덤 4 = 5종 대표를 실제 패턴으로 렌더 + CSV."""
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    if len(ids) == 0:
        print("  ⚠ 군집이 없어 대표패턴 생략 (전부 noise)")
        return
    order = np.argsort(-cnts)
    top = [(int(ids[j]), int(cnts[j])) for j in order[:N_GROUPS_PLOT]]

    reps = []   # (cluster, size, rank, global_row, key)
    for cid, size in top:
        members = np.where(lab == cid)[0]                 # 표본-로컬 idx
        d = np.linalg.norm(X_cl[members] - X_cl[members].mean(0), axis=1)
        medoid = members[int(np.argmin(d))]
        pool = members[members != medoid]
        extra = (rng.choice(pool, min(N_REPS - 1, len(pool)), replace=False)
                 if len(pool) else np.array([], dtype=int))
        for rank, m in enumerate([medoid] + list(extra)):
            grow = int(samp_rows[m])                      # 표본-로컬 → global row
            reps.append({"cluster": cid, "size": size, "rank": rank,
                         "row": grow, "key": str(all_keys[grow])})

    csv_path = os.path.join(out_dir, "h0_hdbscan_reps.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["cluster", "cluster_size", "rep_rank(0=medoid)",
                    "global_row", "gauge_name"])
        for r in reps:
            w.writerow([r["cluster"], r["size"], r["rank"], r["row"], r["key"]])
    print(f"  [대표] CSV: {csv_path}")

    graphs = _extract_graphs(chunks, FOLDER_PATH, [r["row"] for r in reps])
    nrow = len(top)
    fig, axes = plt.subplots(nrow, N_REPS,
                             figsize=(3.2 * N_REPS, 2.9 * nrow), squeeze=False)
    by_cluster = {}
    for r in reps:
        by_cluster.setdefault(r["cluster"], []).append(r)
    for r_i, (cid, size) in enumerate(top):
        rs = by_cluster.get(cid, [])
        for c in range(N_REPS):
            ax = axes[r_i][c]
            if c >= len(rs) or graphs.get(rs[c]["row"]) is None:
                ax.axis("off")
                continue
            tag = "medoid" if rs[c]["rank"] == 0 else "member"
            _draw_pattern(ax, graphs[rs[c]["row"]],
                          f"c{cid} n={size:,} [{tag}]\n{rs[c]['key']}")
    fig.suptitle("h0 HDBSCAN clusters - 5 representatives each "
                 "(h0 nodes colored red + star, other hops gray outline)\n"
                 "rows similar within & different across = density clusters "
                 "are real", fontsize=11)
    if hasattr(fig, "supxlabel"):
        fig.supxlabel("x (nm, gauge-centered)", fontsize=10)
        fig.supylabel("y (nm, gauge-centered)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "h0_hdbscan_reps.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  [대표] 패턴 격자: {path}")


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    global HDB_SAMPLE, MIN_CLUSTER_SIZE, CLUSTER_SPACE
    ap = argparse.ArgumentParser(description="h0 UMAP+HDBSCAN 밀도 군집 실험")
    ap.add_argument("--sample", type=int, default=HDB_SAMPLE)
    ap.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE)
    ap.add_argument("--space", type=str, default=CLUSTER_SPACE,
                    choices=["pca", "raw", "umap"],
                    help="군집화 공간. pca(기본)=표준화→PCA(eff-dim), "
                         "raw=축소 없는 8d, umap=밀도 압축(가짜 군집 위험)")
    ap.add_argument("--eff-dim", type=int, default=EFF_DIM,
                    help="--space pca 축소 차원 (h0 effective rank ≈ 4). "
                         "실행 로그의 실측 effective rank 로 조정")
    ap.add_argument("--assign-all", action="store_true",
                    help="표본 fit 후 유효 h0 전 행에 라벨 배정 (선형 predict)")
    ap.add_argument("--viz", type=str, default="umap", choices=["umap", "pca"],
                    help="2D scatter 투영. umap=예쁘지만 수십 분(보통 job 의 "
                         "최대 병목), pca=수 초 (군집 판정은 KEY METRICS 가 "
                         "하므로 pca 로도 충분)")
    ap.add_argument("--min-samples", type=int, default=MIN_SAMPLES,
                    help="밀도 추정 보수성. 낮출수록(25~100) noise ↓·군집 넓어짐. "
                         "기본 None=min_cluster_size (보수적)")
    ap.add_argument("--epsilon", type=float, default=HDB_EPSILON,
                    help="cluster_selection_epsilon. 키우면 군집 경계가 저밀도 "
                         "영역까지 확장돼 커버리지 ↑ (군집화 공간 거리 단위)")
    ap.add_argument("--sweep", action="store_true",
                    help="파라미터 그리드 스윕 (단일 fit/plot 대신 조합 비교 "
                         "CSV+PNG+추천 출력)")
    ap.add_argument("--sweep-mcs", type=str, default="500,1000,2000,5000",
                    help="스윕할 min_cluster_size 목록 (콤마 구분)")
    ap.add_argument("--sweep-ms", type=str, default="none,100,25",
                    help="스윕할 min_samples 목록 (none=auto)")
    ap.add_argument("--sweep-eps", type=str, default="0,0.25,0.5",
                    help="스윕할 cluster_selection_epsilon 목록")
    args = ap.parse_args()
    HDB_SAMPLE, MIN_CLUSTER_SIZE, CLUSTER_SPACE = \
        args.sample, args.min_cluster_size, args.space

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out_dir = os.path.join(_here, OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    features, all_keys, chunks = load_cache()
    h0 = features["h0"]
    valid = np.where((h0.abs().sum(dim=1) != 0).numpy())[0]
    print(f"[h0] 전체 {h0.shape[0]:,}  유효 {len(valid):,}")

    samp_rows = (valid if len(valid) <= HDB_SAMPLE
                 else np.sort(rng.choice(valid, HDB_SAMPLE, replace=False)))
    X_raw = h0[torch.from_numpy(samp_rows)].numpy().astype(np.float32)
    scaler = StandardScaler().fit(X_raw)     # 전량 배정 시 블록 transform 에 재사용
    X = scaler.transform(X_raw)
    del X_raw
    floor = int(np.ceil(MIN_CLUSTER_SIZE / len(samp_rows) * len(valid)))
    print(f"[표본] {len(samp_rows):,} rows (전체 유효행에서 균일추출)")
    print(f"[탐지 하한] min_cluster_size={MIN_CLUSTER_SIZE:,} → 전체 환산 "
          f"약 {floor:,}개({MIN_CLUSTER_SIZE/len(samp_rows):.2%}) 미만 군집은 안 보임")

    # ── effective rank 진단 (축소 차원 EFF_DIM 을 데이터로 확인) ──
    erank, sv = effective_rank(X)
    print(f"[effective rank] 표준화 8d 의 erank = {erank:.2f} / {X.shape[1]} "
          f"(특이값 {np.round(sv / sv.max(), 2).tolist()})")
    if CLUSTER_SPACE == "pca" and abs(erank - args.eff_dim) > 1.5:
        print(f"  ⚠ 실측 erank({erank:.1f}) 와 EFF_DIM({args.eff_dim}) 차이 큼 — "
              f"--eff-dim {int(round(erank))} 로 맞추는 것 고려")

    # ── 군집화 공간 ──
    space_model = None            # pca/umap 공간이면 전량 배정 때 transform 필요
    if CLUSTER_SPACE == "pca":
        X_cl, space_model = pca_reduce(X, args.eff_dim, SEED)
    elif CLUSTER_SPACE == "umap":
        X_cl, cl_impl, model = umap_embed(X, UMAP_DIM, UMAP_MIN_DIST, SEED)
        if cl_impl == "pca":
            print("  ⚠ UMAP 불가 → PCA(EFF_DIM) 로 군집화")
            X_cl, space_model = pca_reduce(X, args.eff_dim, SEED)
        else:
            space_model = model
    else:
        X_cl = X

    # ── 스윕 모드: 조합 비교만 하고 종료 ──
    if args.sweep:
        run_sweep(X_cl, rng, args, out_dir)
        print(f"\n완료 ({time.time()-t0:.0f}s). 추천 조합으로 --assign-all 재실행하세요.")
        return

    # ── HDBSCAN (발견 = 표본 fit) ──
    lab, dbcv, cl, hdb_impl = run_hdbscan(X_cl, MIN_CLUSTER_SIZE,
                                          args.min_samples, args.epsilon)
    ids, cnts = summarize(lab, X_cl, dbcv, rng)

    # ── 2D 시각화 좌표 ───────────────────────────────────────────
    #   ※ raw 군집화일 때 이 UMAP fit 이 보통 job 전체의 최대 병목(수십 분).
    #     빨리 돌리려면 --viz pca (수 초; 판정은 KEY METRICS 로 이미 끝났음).
    t = time.time()
    if args.viz == "pca":
        P2 = PCA(n_components=2, random_state=SEED).fit_transform(X)
        viz_impl = "pca"
        print(f"  [viz] PCA 2D  ({time.time()-t:.0f}s)")
    else:
        P2, viz_impl, _ = umap_embed(X, 2, 0.1, SEED)

    np.savez(os.path.join(out_dir, "h0_hdbscan_result.npz"),
             rows=samp_rows, labels=lab, proj2d=P2)
    t = time.time()
    plot_scatter(P2, lab, viz_impl, out_dir, rng)
    print(f"  [시간] scatter 렌더 {time.time()-t:.0f}s")
    t = time.time()
    plot_representatives(lab, X_cl, samp_rows, all_keys, chunks, out_dir, rng)
    print(f"  [시간] 대표패턴 렌더(청크 I/O 포함) {time.time()-t:.0f}s")

    # ── 전량 배정 (선형 predict — KMeans 의 fit/predict 분리와 동일 구조) ──
    if args.assign_all:
        print(f"\n[전량 배정] 표본 fit 구조 → 유효 h0 전 행 {len(valid):,}개 라벨링")
        if CLUSTER_SPACE == "umap" and space_model is not None:
            print("  ⚠ umap 공간 배정은 블록마다 umap.transform 이 필요해 느림 "
                  "(pca/raw 공간이면 즉시)")
        assign_all_rows(cl, hdb_impl, scaler, h0, valid, X_cl, lab,
                        space_model, out_dir)

    print(f"\n완료 ({time.time()-t0:.0f}s). 산출물: {OUT_DIR}/")
    print("  h0_hdbscan_umap.png / h0_hdbscan_reps.png / "
          "h0_hdbscan_reps.csv / h0_hdbscan_result.npz")


if __name__ == "__main__":
    main()
