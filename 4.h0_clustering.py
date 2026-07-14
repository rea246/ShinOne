"""
h0_clustering_ver2.py — h0 HDBSCAN 군집화 (noise 를 "별도 그룹"으로 보존하는 방향)

ver1(h0_clustering.py) 과의 차이 = 철학이 다름:
  · ver1 : noise 를 soft 로 인근 군집에 강제 편입(noise 0). 전량을 몇 개 군집으로.
  · ver2 : soft 없음. noise(-1) 를 그대로 "하나의 그룹"으로 표기·보존.
           → 군집 = 순수 밀도 코어(확실한 멤버만), noise = 다양·희귀 패턴 그룹.
           다양한 패턴을 분류·서술하는 데는 이게 더 확실(군집이 오염 안 됨).

파이프라인:
  1) HDBSCAN 1회 → 군집 + noise
  2) 약한 군집(persistence 낮음) → noise 로 편입(FILTER_WEAK) — 여기까지 spurious 제거
  3) soft 없음: noise 는 -1 그대로 유지 (별도 그룹)
  4) 전량 배정: noise 를 하나의 클래스로 포함한 KNN → 전체 유효 h0 를 군집 or noise 로
     (approximate_predict 대신 KNN: 필터 유지 + noise 를 클래스로 보존)
  5) PCA-2D scatter / 대표패턴 / 건강도 / w,h 기하 — 모두 noise 그룹 포함

argv 없음 — 아래 CONFIG 만 고쳐서 실행. 대표패턴 렌더는 원본 청크 필요(FOLDER_PATH).

    python h0_clustering_ver2.py
"""

import os
import csv
import glob
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ══════════════════════════════════════════════════════════════════
# CONFIG — 여기만 고치면 된다
# ══════════════════════════════════════════════════════════════════
_here       = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH  = os.path.join(_here, "hkeys_features.pt")
FOLDER_PATH = "C:/Users/rea24/Documents/shinwon_note/pattern/2.Pattern_Classification/3.CODE/pythonProject2/dummy_dataset"
OUT_DIR     = os.path.join(_here, "h0_clustering_out")   # 5.h0_cluster_analyze.py 가 읽는 경로와 일치

# feature 캐시(CACHE_PATH)가 없을 때만 3.train_GAE best 인코더로 추출 (구 claasify_tree.py 의 [A] 이식).
CKPT_GLOB     = "best_gae_v4_cov_*.pt"   # 3.train_GAE.py 가 저장한 best 전체 GAE 파일 glob
EXTRACT_BATCH = 8192                     # 캐시 추출 시 추론 배치 (학습과 동일 규모)

SAMPLE      = 300_000    # 군집화에 쓸 표본 수
REDUCE_DIM  = 4          # 표준화 후 PCA 축소 차원 (h0 effective rank ≈ 4)
SCATTER_MAX = 100_000    # 2D scatter 에 찍을 점 수 상한
SEED        = 42

# 내가 정한 군집화 조건 (탐색으로 확정: K=7, DBCV≈0.6, 최대군집≈40%)
MIN_CLUSTER_SIZE = 1000
MIN_SAMPLES      = 40
EPSILON          = 0.11
CLUSTER_METHOD   = "leaf"     # 'leaf'(잘게) | 'eom'(안정 큰 군집)

# 약한 군집 필터: persistence 낮거나 너무 작은 군집을 noise 로 편입(spurious 제거).
#   ver2 에선 편입된 점들도 그대로 noise 그룹에 남는다(soft 재배정 없음).
FILTER_WEAK          = True
MIN_KEEP_PERSISTENCE = 0.05    # 이 미만 persistence 군집 → noise (persistence 없으면 무시)
MIN_KEEP_COUNT       = 0       # 이 미만 크기 군집 → noise (0 = 미사용)

# ver2 = soft 없음. noise 는 보존.

# 전량 배정: noise(-1)를 "하나의 클래스"로 포함한 KNN → 전체 유효 h0 를 군집 or noise 로.
#   → noise 영역(저밀도) 근처 점은 noise 로 배정되어 특수패턴 그룹이 전량에서도 유지됨.
ASSIGN_ALL       = True
ASSIGN_K         = 15          # KNN 이웃 수
ASSIGN_BLOCK     = 1_000_000   # 블록당 행 수

N_REPS  = 5             # 군집당 대표 패턴 수 (medoid 1 + 랜덤 4)

HOP_COLORS = {0: "#e6194B", 1: "#f58231", 2: "#3cb44b", 3: "#9aa0a6"}

# 검증된 categorical 팔레트(고정 순서, CVD-safe ΔE 24.2). 색만으로 식별 안 하도록
#   각 군집에 직접 라벨을 찍는다(relief rule). surface/ink 는 밝은 배경용.
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#008300",
           "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
SURFACE, INK, INK2, GRID, NOISEC = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c9c8c2"


# ══════════════════════════════════════════════════════════════════
# 0. feature 캐시 생성 — 없을 때만 3.train_GAE 인코더로 추출 (3→4 직접 연결)
#    구 claasify_tree.py 의 [A] 단계 이식. 캐시가 있으면 이 경로는 전부 건너뛴다.
#    torch_geometric / 모델 import 는 추출할 때만 lazy 로 불러온다(캐시만 있으면 불필요).
# ══════════════════════════════════════════════════════════════════
def _load_best_gae(device):
    """3.train_GAE.py 의 LayoutGAE 클래스를 파일에서 로드 → best 체크포인트 주입."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_gae", os.path.join(_here, "3.train_GAE.py"))
    tg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tg)                               # main() 은 __main__ 가드라 안 돎

    cands = sorted(glob.glob(os.path.join(_here, CKPT_GLOB)))
    if not cands:
        raise FileNotFoundError(f"best 체크포인트 없음: {os.path.join(_here, CKPT_GLOB)}")
    ckpt_path = cands[-1]                                     # 파일명 timestamp → sorted 마지막 = 최신
    print(f"  [Model] 로드: {ckpt_path}")
    model = tg.LayoutGAE(num_node_features=4, num_edge_features=4,
                         embedding_dim=tg.EMBEDDING_DIM).to(device)   # 학습 아키텍처 그대로 재현
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


def _load_chunk_as_data_list(path):
    """PREPROCESSED_*part*.pt 청크 하나 → (Data 리스트, keys). x/edge_attr 는 fp16 slice 유지."""
    from torch_geometric.data import Data
    d = torch.load(path, map_location="cpu", weights_only=False)
    x_all, ei_all, ea_all = d["x"], d["edge_index"], d["edge_attr"]
    hp_all, meta, keys = d["hop_packed"], d["meta"], d["keys"]

    data_list = []
    for x0, x1, e0, e1 in meta.tolist():
        packed = hp_all[x0:x1]
        data_list.append(Data(
            x=x_all[x0:x1],
            edge_index=ei_all[:, e0:e1].long(),              # 그래프-로컬 인덱스
            edge_attr=ea_all[e0:e1],
            hop0_mask=(packed & 1).bool(),
            hop1_mask=((packed >> 1) & 1).bool(),
            hop2_mask=((packed >> 2) & 1).bool(),
            hop3_mask=((packed >> 3) & 1).bool(),
        ))
    return data_list, list(keys)


def _extract_block_features(model, data_list, device):
    """청크 하나 → 블록별 (G, D) CPU float32 텐서 dict.
    h0/h1/h2/h3 = z_node 를 hop mask 별 mean-pool(각 8d), edge = z_edge mean-pool(8d),
    h0_raw = 중심 raw [w,h](2d), edge_raw = edge_attr mean+std(8d)."""
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import scatter

    def mpool(z, mask, bi, G):
        return scatter(z[mask], bi[mask], dim=0, dim_size=G, reduce="mean")

    loader = DataLoader(data_list, batch_size=EXTRACT_BATCH, shuffle=False)
    acc = {k: [] for k in ("h0", "h1", "h2", "h3", "edge", "h0_raw", "edge_raw")}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch.x         = batch.x.float()                # fp16 → fp32 (GPU 에서 1회)
            batch.edge_attr = batch.edge_attr.float()
            G = batch.num_graphs
            z_node, z_edge = model.encoder(batch)            # clean 입력, 결정적

            bi = batch.batch
            acc["h0"].append(mpool(z_node, batch.hop0_mask, bi, G).cpu())
            acc["h1"].append(mpool(z_node, batch.hop1_mask, bi, G).cpu())
            acc["h2"].append(mpool(z_node, batch.hop2_mask, bi, G).cpu())
            acc["h3"].append(mpool(z_node, batch.hop3_mask, bi, G).cpu())

            eb = bi[batch.edge_index[0]]
            acc["edge"].append(scatter(z_edge, eb, dim=0, dim_size=G, reduce="mean").cpu())

            # raw 블록 — 인코더를 거치지 않은 순수 기하
            acc["h0_raw"].append(mpool(batch.x[:, 2:4], batch.hop0_mask, bi, G).cpu())
            e_mean = scatter(batch.edge_attr, eb, dim=0, dim_size=G, reduce="mean")
            e_sq   = scatter(batch.edge_attr ** 2, eb, dim=0, dim_size=G, reduce="mean")
            acc["edge_raw"].append(
                torch.cat([e_mean, (e_sq - e_mean ** 2).clamp_min(0).sqrt()], dim=1).cpu())

    return {k: torch.cat(v, dim=0).float() for k, v in acc.items()}


def ensure_features():
    """캐시가 있으면 로드(추출 생략), 없으면 3.train_GAE best 인코더로 추출 후 저장.
    반환 (features_dict, keys, chunks) — 포맷은 구 claasify_tree.py 캐시와 동일."""
    if os.path.exists(CACHE_PATH):                           # ← feature 있으면 실행 안 함
        print(f"[Cache] feature 캐시 로드: {CACHE_PATH}")
        d = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
        return d["features"], d["keys"], d["chunks"]

    print(f"[Cache] '{CACHE_PATH}' 없음 → 3.train_GAE best 인코더로 feature 추출")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model = _load_best_gae(device)

    files = sorted(glob.glob(os.path.join(FOLDER_PATH, "PREPROCESSED_*part*.pt")))
    if not files:
        raise FileNotFoundError(f"청크 파일 없음: {FOLDER_PATH}")

    block_parts, all_keys, chunks = None, [], []
    for i, f in enumerate(files):
        t = time.time()
        data_list, keys = _load_chunk_as_data_list(f)
        feats = _extract_block_features(model, data_list, device)
        if block_parts is None:
            block_parts = {k: [] for k in feats}
        for k, v in feats.items():
            block_parts[k].append(v.half())                  # fp16 누적 — 캐시/RAM 절반
        all_keys.extend(keys)
        chunks.append((os.path.basename(f), len(keys)))
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}  graphs={len(keys)}  "
              f"({time.time()-t:.1f}s)")
        del data_list, feats

    features = {k: torch.cat(v, dim=0) for k, v in block_parts.items()}
    tmp = CACHE_PATH + ".tmp"
    torch.save({"features": features, "keys": all_keys, "chunks": chunks}, tmp)
    os.replace(tmp, CACHE_PATH)                              # 원자적 교체
    print(f"[Saved] feature 캐시: {CACHE_PATH}")
    return features, all_keys, chunks


# ══════════════════════════════════════════════════════════════════
# 1. 임베딩 로드 + 표본 → 표준화 → PCA (군집화 4d / 시각화 2d)
# ══════════════════════════════════════════════════════════════════
def load_and_prepare():
    features, all_keys, chunks = ensure_features()           # 캐시 있으면 로드, 없으면 3→4 추출
    h0 = features["h0"]
    h0_raw = features.get("h0_raw")                          # 중심 raw [w,h] (없으면 None)
    valid = np.where((h0.abs().sum(1) != 0).numpy())[0]      # 빈 hop 제외
    print(f"h0 임베딩: 전체 {h0.shape[0]:,}  유효 {len(valid):,}")

    rng = np.random.default_rng(SEED)
    samp_rows = (valid if len(valid) <= SAMPLE
                 else np.sort(rng.choice(valid, SAMPLE, replace=False)))
    X = h0[torch.from_numpy(samp_rows)].numpy().astype(np.float32)
    scaler = StandardScaler().fit(X)                         # 8d 표준화(전량배정 때 재사용)
    X = scaler.transform(X)
    print(f"표본 {len(samp_rows):,}개, {X.shape[1]}차원")

    if REDUCE_DIM < X.shape[1]:
        pca = PCA(n_components=REDUCE_DIM, random_state=SEED).fit(X)
        X_cluster = pca.transform(X).astype(np.float32)
        print(f"군집화 공간: PCA {X.shape[1]}d → {REDUCE_DIM}d "
              f"(설명분산 {pca.explained_variance_ratio_.sum():.0%})")
    else:
        pca = None
        X_cluster = X

    X_view2d = PCA(n_components=2, random_state=SEED).fit_transform(X)  # 시각화용 2D
    # scaler/pca/h0/valid 는 전량 배정(assign_all) 때 블록 변환에 재사용
    return (X_cluster, X_view2d, samp_rows, all_keys, chunks, rng,
            h0, h0_raw, valid, scaler, pca)


# ══════════════════════════════════════════════════════════════════
# 2. HDBSCAN 1회 (내가 정한 조건)
# ══════════════════════════════════════════════════════════════════
def run_hdbscan(X_cluster):
    t = time.time()
    try:
        import hdbscan
        cl = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                             min_samples=MIN_SAMPLES,
                             cluster_selection_epsilon=float(EPSILON),
                             cluster_selection_method=CLUSTER_METHOD,
                             gen_min_span_tree=True,      # DBCV 위해
                             prediction_data=True,        # soft 배정 위해
                             core_dist_n_jobs=-1)          # core 전부 사용
        lab = cl.fit_predict(X_cluster)
        dbcv = float(getattr(cl, "relative_validity_", float("nan")))
        persist = getattr(cl, "cluster_persistence_", None)
        impl = "hdbscan"
    except ImportError:
        print("  ⚠ hdbscan 패키지 없음 → sklearn 내장 사용 (DBCV·persistence·soft 없음). "
              "pip install hdbscan 권장")
        from sklearn.cluster import HDBSCAN as SKHDBSCAN
        cl = SKHDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES,
                       cluster_selection_epsilon=float(EPSILON),
                       cluster_selection_method=CLUSTER_METHOD, n_jobs=-1)
        lab = cl.fit_predict(X_cluster)
        dbcv, persist, impl = float("nan"), None, "sklearn"
    print(f"HDBSCAN fit 완료 ({time.time()-t:.0f}s)  "
          f"mcs={MIN_CLUSTER_SIZE} ms={MIN_SAMPLES} eps={EPSILON} method={CLUSTER_METHOD}")
    return cl, lab, dbcv, persist, impl


# ══════════════════════════════════════════════════════════════════
# 2-b. 약한 군집 → noise 전환 (필터)
# ══════════════════════════════════════════════════════════════════
def filter_weak_clusters(lab, persist):
    """persistence/count 문턱 미만 군집을 noise(-1) 로 전환.
    반환 (lab_filtered, keep_ids). persist 없으면 count 만 적용."""
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    cnt_by = dict(zip(ids.tolist(), cnts.tolist()))
    keep, dropped = [], []
    for c in ids.tolist():
        p = persist[c] if (persist is not None and c < len(persist)) else None
        ok_p = (p is None) or (p >= MIN_KEEP_PERSISTENCE)
        ok_n = cnt_by[c] >= MIN_KEEP_COUNT
        (keep if (ok_p and ok_n) else dropped).append(c)

    out = lab.copy()
    if not dropped:
        print("  [필터] 녹일 약한 군집 없음")
        return out, keep
    if len(keep) < 2:
        print(f"  ⚠ [필터] 문턱이 너무 세서 남는 군집 {len(keep)}개 → 필터 건너뜀")
        return out, ids.tolist()
    out[np.isin(out, dropped)] = -1            # 약한 군집 → noise
    for c in dropped:
        p = persist[c] if (persist is not None and c < len(persist)) else float("nan")
        print(f"  [필터] c{c} 녹임 → noise (count={cnt_by[c]:,}, persistence={p:.3f})")
    return out, keep


# ══════════════════════════════════════════════════════════════════
# 2-c. 전량 배정 — noise(-1)를 클래스로 포함한 KNN (noise 보존)
# ══════════════════════════════════════════════════════════════════
def assign_all_rows(h0, valid, samp_rows, X_cluster, lab_final, scaler, pca):
    """전량 배정 — 표본 최종라벨(군집 + noise -1)로 KNN 학습 → 전체 유효 h0 예측.
    noise 를 하나의 클래스로 학습하므로 저밀도 영역 점은 noise(-1)로 배정되어
    특수패턴 그룹이 전량에서도 보존·확장된다. 표본 행은 확정 라벨 그대로."""
    from sklearn.neighbors import KNeighborsClassifier
    t = time.time()
    knn = KNeighborsClassifier(n_neighbors=ASSIGN_K, n_jobs=-1).fit(X_cluster, lab_final)

    full = np.full(len(valid), -1, dtype=np.int32)
    pos = np.searchsorted(valid, samp_rows)
    full[pos] = lab_final                            # 표본은 확정 라벨(군집 or noise)
    rest = np.setdiff1d(np.arange(len(valid)), pos)
    print(f"[전량 배정/KNN k={ASSIGN_K}, noise 클래스 포함] 유효 {len(valid):,} "
          f"= 표본 {len(pos):,}(확정) + 나머지 {len(rest):,}")

    for s in range(0, len(rest), ASSIGN_BLOCK):
        bi = rest[s:s + ASSIGN_BLOCK]
        Xb = h0[torch.from_numpy(valid[bi])].numpy().astype(np.float32)
        Xb = scaler.transform(Xb)
        if pca is not None:
            Xb = pca.transform(Xb).astype(np.float32)
        full[bi] = knn.predict(Xb)                   # 군집 or -1(noise)
        print(f"  [배정] {min(s + len(bi), len(rest)):,}/{len(rest):,}", end="\r")
        del Xb
    print()
    ids, cnts = np.unique(full, return_counts=True)
    n_noise = int((full < 0).sum())
    print(f"[전량 완료] noise(별도그룹) {n_noise:,} ({n_noise/len(full):.1%})  "
          f"({time.time()-t:.0f}s)")
    print("  전량 군집 크기: " +
          "  ".join(f"c{int(i)}:{c:,}({c/len(full):.1%})"
                    for i, c in zip(ids, cnts) if i >= 0))
    return full


# ══════════════════════════════════════════════════════════════════
# 3. 분류 건강도 수치 출력
# ══════════════════════════════════════════════════════════════════
def print_health(lab, dbcv, persist, title="h0 첫 군집화"):
    ids, cnts = np.unique(lab[lab >= 0], return_counts=True)
    n = len(lab)
    k = len(ids)
    noise = float((lab == -1).mean())
    max_share = float(cnts.max() / n) if k else 0.0
    # eff_K = 군집 크기 분포의 유효 개수 (한 군집 독식이면 1 에 가까움)
    p = cnts / cnts.sum() if k else np.array([1.0])
    eff_k = float(np.exp(-(p * np.log(p)).sum())) if k else 0.0

    print("\n" + "=" * 60)
    print(f"  분류 건강도 ({title})")
    print("=" * 60)
    print(f"  군집 수 K        : {k}")
    print(f"  noise 비율       : {noise:.1%}    (낮을수록↑)")
    print(f"  DBCV             : {dbcv:.3f}    (높을수록↑, 밀도군집 품질)")
    print(f"  최대군집 비율    : {max_share:.1%}    (한 군집 독식이면 큼)")
    print(f"  유효 군집수 eff_K: {eff_k:.2f} / {k}  (군집 크기 균형; K 에 가까울수록 균형)")
    print(f"  군집별:")
    for j in np.argsort(-cnts):
        cid = int(ids[j])
        line = f"    c{cid:<2} {cnts[j]:>9,} ({cnts[j]/n:5.1%})"
        if persist is not None and cid < len(persist):
            line += f"   persistence={persist[cid]:.3f}"   # 클수록 안정/뚜렷한 군집
        print(line)
    print("=" * 60)
    return ids, cnts


# ══════════════════════════════════════════════════════════════════
# 4. PCA-2D scatter
# ══════════════════════════════════════════════════════════════════
NOISE_DARK = "#5a5a5a"   # noise 그룹 강조색(별도 그룹으로 표기)


def plot_scatter_2d(X_view2d, lab, rng):
    """군집 색(고정 팔레트) + noise 를 '별도 그룹'으로 색·중심라벨·legend 표기."""
    sns.set_theme(style="white", context="talk")
    n = len(lab)
    draw = (np.arange(n) if n <= SCATTER_MAX
            else np.sort(rng.choice(n, SCATTER_MAX, replace=False)))
    P, ids = X_view2d, np.unique(lab[lab >= 0])

    fig, ax = plt.subplots(figsize=(11, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # noise = 하나의 그룹으로 그린다 (회색 점 + 중심 'noise' 라벨)
    nm = draw[lab[draw] < 0]
    ax.scatter(P[nm, 0], P[nm, 1], s=5, c=NOISEC, alpha=0.35, linewidths=0, zorder=1)
    n_noise = int((lab < 0).sum())
    handles = [plt.Line2D([0], [0], marker="o", ls="", mfc=NOISEC, mec="none",
                          ms=9, label=f"noise  {n_noise:,} ({n_noise/n:.0%})")]
    if n_noise:
        nall = np.where(lab < 0)[0]
        ax.annotate("noise", (np.median(P[nall, 0]), np.median(P[nall, 1])),
                    ha="center", va="center", fontsize=12, fontweight="bold",
                    color="white", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.3", fc=NOISE_DARK,
                              ec="white", lw=1.5, alpha=0.9))

    for i, cid in enumerate(ids):
        color = PALETTE[i] if i < len(PALETTE) else "#9a9a94"   # 8개 초과는 회색(fold)
        m = draw[lab[draw] == cid]
        ax.scatter(P[m, 0], P[m, 1], s=7, color=color, alpha=0.65, linewidths=0, zorder=2)
        allm = np.where(lab == cid)[0]
        ax.annotate(f"c{cid}", (np.median(P[allm, 0]), np.median(P[allm, 1])),
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color=INK, zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec=color,
                              lw=2, alpha=0.92))
        handles.append(plt.Line2D([0], [0], marker="o", ls="", mfc=color, mec="none",
                                  ms=9, label=f"c{cid}  {len(allm):,} ({len(allm)/n:.0%})"))

    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False, fontsize=11, labelcolor=INK2,
              title="group", title_fontsize=12)
    ax.set_title(f"h0 HDBSCAN (ver2: noise 별도)  ·  {len(ids)} clusters + noise "
                 f"{(lab<0).mean():.0%}",
                 fontsize=15, fontweight="bold", color=INK, pad=14)
    ax.set_xlabel("PC1", color=INK2); ax.set_ylabel("PC2", color=INK2)
    ax.tick_params(colors=INK2, labelsize=10)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "h0_scatter_2d.png")
    fig.savefig(path, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  [scatter] {path}")


# ══════════════════════════════════════════════════════════════════
# 5. 각 군집 대표 h0 패턴 5개 렌더  (h0_hdbscan_umap.py 로직 재사용)
# ══════════════════════════════════════════════════════════════════
def _row_to_file(chunks, folder, row):
    off = 0
    for base, cnt in chunks:
        if row < off + cnt:
            return os.path.join(folder, base), row - off
        off += cnt
    return None, None


def _extract_graphs(chunks, folder, rows):
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


def _draw_pattern(ax, g, key):
    """h0 노드 색 강조 + 나머지 hop 회색 윤곽."""
    x, hop = g["x"], g["hop"]
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    for i in np.where(hop != 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               fill=False, edgecolor="#cccccc", lw=0.4))
    for i in np.where(hop == 0)[0]:
        ax.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                               facecolor=HOP_COLORS[0], edgecolor="black",
                               lw=0.5, alpha=0.8))
    h0n = np.where(hop == 0)[0]
    if len(h0n):
        ax.plot(cx[h0n], cy[h0n], "*", color="black", ms=10,
                markerfacecolor=HOP_COLORS[0], zorder=4)
    if len(x):
        x0b = float((cx - w / 2).min()); x1b = float((cx + w / 2).max())
        y0b = float((cy - h / 2).min()); y1b = float((cy + h / 2).max())
        pad = 0.05 * max(x1b - x0b, y1b - y0b, 1.0)
        ax.set_xlim(x0b - pad, x1b + pad); ax.set_ylim(y0b - pad, y1b + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(key, fontsize=6.5)
    ax.tick_params(labelsize=5)
    ax.grid(True, ls=":", alpha=0.3)


def plot_representatives(lab, X_cluster, samp_rows, all_keys, chunks, rng):
    """군집 + noise 그룹마다 medoid 1 + 랜덤 4 = 5개 대표를 실제 h0 패턴으로 렌더.
    ver2: noise 도 하나의 그룹으로 대표패턴을 봐서 '특수패턴'을 눈으로 특성화."""
    cids, ccnt = np.unique(lab[lab >= 0], return_counts=True)
    groups = [(int(cids[j]), int(ccnt[j])) for j in np.argsort(-ccnt)]  # 군집(크기순)
    if (lab < 0).any():
        groups.append((-1, int((lab < 0).sum())))                       # noise 마지막
    if not groups:
        print("  ⚠ 그룹 없음 → 대표패턴 생략")
        return

    def gname(cid):
        return "noise" if cid < 0 else f"c{cid}"

    reps = []
    for cid, size in groups:
        members = np.where(lab == cid)[0]
        d = np.linalg.norm(X_cluster[members] - X_cluster[members].mean(0), axis=1)
        medoid = members[int(np.argmin(d))]
        pool = members[members != medoid]
        extra = (rng.choice(pool, min(N_REPS - 1, len(pool)), replace=False)
                 if len(pool) else np.array([], dtype=int))
        for rank, m in enumerate([medoid] + list(extra)):
            reps.append({"cluster": cid, "size": size, "rank": rank,
                         "row": int(samp_rows[m])})

    graphs = _extract_graphs(chunks, FOLDER_PATH, [r["row"] for r in reps])
    by_cluster = {}
    for r in reps:
        by_cluster.setdefault(r["cluster"], []).append(r)

    nrow = len(groups)
    fig, axes = plt.subplots(nrow, N_REPS, figsize=(3.2 * N_REPS, 2.9 * nrow),
                             squeeze=False)
    for r_i, (cid, size) in enumerate(groups):
        rs = by_cluster.get(cid, [])
        for c in range(N_REPS):
            ax = axes[r_i][c]
            if c >= len(rs) or graphs.get(rs[c]["row"]) is None:
                ax.axis("off"); continue
            tag = "medoid" if rs[c]["rank"] == 0 else "member"
            key = str(all_keys[rs[c]["row"]])
            _draw_pattern(ax, graphs[rs[c]["row"]],
                          f"{gname(cid)} n={size:,} [{tag}]\n{key}")
    fig.suptitle("h0 그룹별 대표 패턴 (noise 포함; h0 노드=빨강+별, 다른 hop=회색)\n"
                 "noise 행 = 군집에 안 든 다양·희귀 패턴", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "h0_representatives.png")
    fig.savefig(path, dpi=120); plt.close(fig)
    print(f"  [대표패턴] {path}")


# ══════════════════════════════════════════════════════════════════
# 6. 군집별 h0 중심 raw 기하 [w,h] 요약 (대략적 그룹 특징)
# ══════════════════════════════════════════════════════════════════
def cluster_geometry(lab, samp_rows, h0_raw, out_dir):
    """각 최종 군집의 h0 중심 raw 기하 [w,h] 평균/표준편차 + 종횡비(h/w).
    캐시 h0_raw = 중심 hop 노드의 [w,h] (x,y 는 게이지 중심이라 ~0 → 저장 안 됨).
    → 군집이 '대체로 어떤 크기/납작함의 중심 피처'인지 대략 특징이 보인다."""
    if h0_raw is None:
        print("  ⚠ 캐시에 h0_raw 없음 → 군집 기하 요약 생략")
        return
    wh = h0_raw[torch.from_numpy(samp_rows)].numpy().astype(np.float64)   # (n,2)=[w,h]
    ids = list(np.unique(lab[lab >= 0]))
    if (lab < 0).any():
        ids.append(-1)                                   # noise 그룹도 포함
    print("\n" + "=" * 62)
    print("  그룹별 h0 중심 기하 (w,h)  [noise 포함; x,y ≈ 0 = 게이지 중심]")
    print("=" * 62)
    print(f"  {'group':>7} {'n':>8} | {'w_mean':>8} {'w_std':>7} | "
          f"{'h_mean':>8} {'h_std':>7} | {'h/w':>5}")
    rows = []
    for cid in ids:
        m = lab == cid
        w, h = wh[m, 0], wh[m, 1]
        ratio = float(h.mean() / w.mean()) if w.mean() else float("nan")
        name = "noise" if cid < 0 else f"c{int(cid)}"
        print(f"  {name:>7} {int(m.sum()):>8,} | {w.mean():>8.3f} {w.std():>7.3f} | "
              f"{h.mean():>8.3f} {h.std():>7.3f} | {ratio:>5.2f}")
        rows.append((name, int(m.sum()), w.mean(), w.std(),
                     h.mean(), h.std(), ratio))
    print("=" * 62)
    path = os.path.join(out_dir, "h0_cluster_geometry.csv")
    with open(path, "w", newline="", encoding="utf-8") as fp:
        wr = csv.writer(fp)
        wr.writerow(["group", "n", "w_mean", "w_std", "h_mean", "h_std", "h_over_w"])
        for r in rows:
            wr.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])
    print(f"  [군집 기하] {path}")


# ══════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    (X_cluster, X_view2d, samp_rows, all_keys, chunks, rng,
     h0, h0_raw, valid, scaler, pca) = load_and_prepare()

    cl, lab, dbcv, persist, impl = run_hdbscan(X_cluster)
    print_health(lab, dbcv, persist, title="발견 단계 (noise 포함)")

    # 약한 군집(persistence/count 문턱 미만) → noise 편입 (spurious 제거)
    if FILTER_WEAK:
        lab, _ = filter_weak_clusters(lab, persist)

    # ver2: soft 없음. 군집만 0..K-1 로 재번호, noise(-1)는 그대로 별도 그룹 유지.
    final_ids = np.unique(lab[lab >= 0])
    lab_final = lab.copy()
    if len(final_ids) and not np.array_equal(final_ids, np.arange(len(final_ids))):
        print(f"  [재번호] 군집 {final_ids.tolist()} → 0..{len(final_ids)-1} (noise=-1 유지)")
        pos = lab_final >= 0
        lab_final[pos] = np.searchsorted(final_ids, lab_final[pos])
    print_health(lab_final, dbcv, None, title="필터 후 (noise 별도 그룹 유지)")

    # 그룹별 h0 중심 raw 기하 요약 (noise 포함)
    cluster_geometry(lab_final, samp_rows, h0_raw, OUT_DIR)

    # 시각화·대표패턴·표본 저장 (noise 별도 그룹)
    plot_scatter_2d(X_view2d, lab_final, rng)
    plot_representatives(lab_final, X_cluster, samp_rows, all_keys, chunks, rng)
    np.savez(os.path.join(OUT_DIR, "h0_labels.npz"),
             rows=samp_rows, labels=lab_final, labels_discovery=lab)

    # ── 전량 배정 (noise 를 클래스로 포함한 KNN — noise 보존) ──
    if ASSIGN_ALL:
        full = assign_all_rows(h0, valid, samp_rows, X_cluster, lab_final,
                               scaler, pca)
        np.savez(os.path.join(OUT_DIR, "h0_labels_full.npz"),
                 rows=valid, labels=full)
        print(f"  [저장] h0_labels_full.npz  (rows=전체 유효 global row, "
              f"labels=군집; -1=noise 그룹; 총 {len(valid):,}개)")

    print(f"\n완료 ({time.time()-t0:.0f}s). 산출물: {OUT_DIR}/")
    print("  h0_scatter_2d.png / h0_representatives.png / h0_cluster_geometry.csv"
          " / h0_labels.npz(표본)"
          + ("  / h0_labels_full.npz(전량)" if ASSIGN_ALL else ""))


if __name__ == "__main__":
    main()
