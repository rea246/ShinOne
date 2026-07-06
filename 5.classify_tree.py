"""
5.classify_tree.py — auto-K + hop 전개 트리 기반 unique pattern 추출 (자체 완결판)

파이프라인:
  [A] feature 추출 : best GAE 인코더로 청크 스트리밍 추론 → 그래프당 블록 벡터
        h0/h1/h2/h3 : hop 별 z_node mean-pool (각 8d, 빈 hop 은 0벡터)
        edge        : z_edge mean-pool (8d)     — LEVELS 확장용
        h0_raw      : center [w,h] (2d)         — 순수 기하 축이 필요할 때
        edge_raw    : edge_attr mean+std (8d)   — edge 진단/확장용
      결과는 FEATURE_CACHE 에 fp16 저장 → 재실행 시 추론 생략.
  [B] auto-K    : 레벨마다 silhouette 스윕으로 "이상적 클러스터 수" 자동 선택
  [C] N_TARGET  : 목표 unique pattern 수 지정 시 K 를 비율 유지 스케일 보정 (±TOL)
  [D] 전개 트리 : pattern_id = 레벨 key 연결 path("3-1-2-0"). 관측된 path 만 기록
                  → 관측 path 수 = unique pattern 수, transitions = 조합 제약 실측
  [E] edge 진단 : path 내 raw edge 잔여 분산 비율 → edge 축 추가 필요성 판단

[대규모 실행 노트 — 40GB(2GB×20청크), 그래프 ~2천만 개 기준]
  RAM  피크 ~10GB. 청크 동시 1개(fp16 slice ≈2GB) + feature 누적(fp16 ≈2GB).
       float32 변환은 "표본 또는 처리 중인 레벨"만 수행하고 즉시 해제.
  GPU  encoder forward 전용(no grad), batch 8192 — 학습이 돌던 GPU 면 충분.
  TIME feature 추출 1 pass ≈ 수십 분(1회, 캐시 재사용). silhouette 스윕 레벨당 수 분.
       KMeans 학습은 FIT_SAMPLE 표본으로만, 전량은 predict(라벨링)만 수행.
  path 는 int64 mixed-radix 코드(2천만 행 문자열 생성 회피), 전이 집계 벡터화.
  SAVE_CSV 기본 False — 라벨은 hkeys_tree_labels.pt 사용.
"""

import os
import csv
import json
import glob
import time
import importlib.util

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

# ══════════════════════════════════════════════════════════════════
# [Config]
# ══════════════════════════════════════════════════════════════════
FOLDER_PATH = "C:/Users/rea24/Documents/shinwon_note/pattern/2.Pattern_Classification/3.CODE/pythonProject2/dummy_dataset"
CKPT_GLOB   = "best_gae_v4_cov_*.pt"   # 3.train_GAE.py 가 저장한 best 전체 GAE
BATCH_SIZE  = 8192                      # 추론 배치 (학습과 동일 규모)
FEATURE_CACHE = "hkeys_features.pt"

# 분류 축 (전개 순서). edge 진단 비율이 크면 끝에 'edge_raw' 또는 'edge' 추가.
LEVELS = ['h0', 'h1', 'h2', 'h3']

# 원하는 unique pattern 수. None 이면 silhouette auto-K 결과 그대로.
N_TARGET     = None          # 예: 2000 → 관측 path 수가 2000(±TARGET_TOL) 되도록 K 보정
TARGET_TOL   = 0.05
CAL_SAMPLE   = 2_000_000     # 보정용 표본 (전체가 더 작으면 전량)
MAX_CAL_ITER = 8
K_HARD_MAX   = 64            # 레벨당 K 상한 (폭주 방지)

K_MIN, K_MAX = 2, 20         # silhouette 스윕 범위
FIT_SAMPLE   = 200_000       # KMeans 학습 표본
SIL_SAMPLE   = 10_000        # silhouette 표본 (O(n²) — 1만이면 호출당 수 초)
SEED         = 42
EMPTY_KEY    = -1            # 빈 hop 전용 key ("그 거리에 이웃 없음"도 의미 있는 범주)
OUT_PREFIX   = "hkeys_tree"
SAVE_CSV     = False

# ══════════════════════════════════════════════════════════════════
# [A-1. 모델 — 3.train_GAE.py 의 클래스 재사용]
# ══════════════════════════════════════════════════════════════════
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("train_gae", os.path.join(_here, "3.train_GAE.py"))
tg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tg)


def load_best_model(device):
    cands = sorted(glob.glob(os.path.join(_here, CKPT_GLOB)))
    if not cands:
        raise FileNotFoundError(f"best 체크포인트 없음: {os.path.join(_here, CKPT_GLOB)}")
    ckpt_path = cands[-1]   # 파일명 timestamp → sorted 마지막 = 최신
    print(f"[Model] 로드: {ckpt_path}")
    model = tg.LayoutGAE(num_node_features=4, num_edge_features=4,
                         embedding_dim=tg.EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════
# [A-2. 청크 로드 + 블록 feature 추출]
# ══════════════════════════════════════════════════════════════════
def load_chunk_as_data_list(path):
    """keys 를 유지한 채 청크 → Data 리스트. x/edge_attr 는 fp16 slice(storage 공유,
    복사 없음) 유지 — 캐스팅은 GPU 배치 후 1회만 (청크 RAM 2배 방지)."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    x_all, ei_all, ea_all = d['x'], d['edge_index'], d['edge_attr']
    hp_all, meta, keys = d['hop_packed'], d['meta'], d['keys']

    data_list = []
    for x0, x1, e0, e1 in meta.tolist():
        packed = hp_all[x0:x1]
        data_list.append(Data(
            x=x_all[x0:x1],
            edge_index=ei_all[:, e0:e1].long(),   # 그래프-로컬 인덱스
            edge_attr=ea_all[e0:e1],
            hop0_mask=(packed & 1).bool(),
            hop1_mask=((packed >> 1) & 1).bool(),
            hop2_mask=((packed >> 2) & 1).bool(),
            hop3_mask=((packed >> 3) & 1).bool(),
        ))
    return data_list, list(keys)


def _masked_mean_pool(z, batch_idx, mask, num_graphs):
    """mask 노드들의 z 를 그래프별 mean-pool. 빈 그래프 행은 0벡터."""
    return scatter(z[mask], batch_idx[mask], dim=0, dim_size=num_graphs, reduce='mean')


@torch.no_grad()
def extract_block_features(model, data_list, device):
    """청크 하나 → 블록별 (G, D) CPU float32 텐서 dict."""
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    acc = {k: [] for k in ('h0', 'h1', 'h2', 'h3', 'edge', 'h0_raw', 'edge_raw')}

    for batch in loader:
        batch = batch.to(device)
        batch.x         = batch.x.float()         # fp16 → fp32 (GPU 에서 1회)
        batch.edge_attr = batch.edge_attr.float()
        G = batch.num_graphs
        z_node, z_edge = model.encoder(batch)     # clean 입력, 결정적

        bi = batch.batch
        acc['h0'].append(_masked_mean_pool(z_node, bi, batch.hop0_mask, G).cpu())
        acc['h1'].append(_masked_mean_pool(z_node, bi, batch.hop1_mask, G).cpu())
        acc['h2'].append(_masked_mean_pool(z_node, bi, batch.hop2_mask, G).cpu())
        acc['h3'].append(_masked_mean_pool(z_node, bi, batch.hop3_mask, G).cpu())

        eb = bi[batch.edge_index[0]]
        acc['edge'].append(scatter(z_edge, eb, dim=0, dim_size=G, reduce='mean').cpu())

        # raw 블록 — 인코더를 거치지 않은 순수 기하
        acc['h0_raw'].append(_masked_mean_pool(batch.x[:, 2:4], bi, batch.hop0_mask, G).cpu())
        e_mean = scatter(batch.edge_attr, eb, dim=0, dim_size=G, reduce='mean')
        e_sq   = scatter(batch.edge_attr ** 2, eb, dim=0, dim_size=G, reduce='mean')
        acc['edge_raw'].append(
            torch.cat([e_mean, (e_sq - e_mean ** 2).clamp_min(0).sqrt()], dim=1).cpu())

    return {k: torch.cat(v, dim=0).float() for k, v in acc.items()}


def ensure_features():
    """캐시 로드 또는 청크 스트리밍 생성.
    포맷: {'features': {block: (N,D) fp16}, 'keys': [...],
           'chunks': [(basename, count), ...]}  ← 시각화의 global→로컬 역참조용."""
    cache = os.path.join(_here, FEATURE_CACHE)
    if os.path.exists(cache):
        print(f"[Cache] feature 캐시 로드: {cache}")
        d = torch.load(cache, map_location='cpu', weights_only=False)
        return d['features'], d['keys'], d['chunks']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = load_best_model(device)

    files = sorted(glob.glob(os.path.join(FOLDER_PATH, "PREPROCESSED_*part*.pt")))
    if not files:
        raise FileNotFoundError(f"청크 파일 없음: {FOLDER_PATH}")

    block_parts, all_keys, chunks = None, [], []
    for i, f in enumerate(files):
        t = time.time()
        data_list, keys = load_chunk_as_data_list(f)
        feats = extract_block_features(model, data_list, device)
        if block_parts is None:
            block_parts = {k: [] for k in feats}
        for k, v in feats.items():
            block_parts[k].append(v.half())       # fp16 누적 — 캐시/RAM 절반
        all_keys.extend(keys)
        chunks.append((os.path.basename(f), len(keys)))
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)}  graphs={len(keys)}  "
              f"({time.time()-t:.1f}s)")
        del data_list, feats

    features = {k: torch.cat(v, dim=0) for k, v in block_parts.items()}
    tmp = cache + ".tmp"
    torch.save({'features': features, 'keys': all_keys, 'chunks': chunks}, tmp)
    os.replace(tmp, cache)
    print(f"[Saved] feature 캐시: {cache}")
    return features, all_keys, chunks


# ══════════════════════════════════════════════════════════════════
# [path 인코딩 — int64 mixed-radix]
# ══════════════════════════════════════════════════════════════════
def encode_paths(labels: dict, levels: list, ks: dict) -> np.ndarray:
    code = np.zeros(len(labels[levels[0]]), dtype=np.int64)
    for lv in levels:
        radix = ks[lv] + 2                       # 라벨 -1..k-1 → shift 후 0..k
        code = code * radix + (labels[lv].astype(np.int64) + 1)
    return code


def decode_code(code: int, levels: list, ks: dict) -> str:
    parts = []
    for lv in reversed(levels):
        radix = ks[lv] + 2
        parts.append(int(code % radix) - 1)
        code //= radix
    return '-'.join(str(p) for p in reversed(parts))


# ══════════════════════════════════════════════════════════════════
# [B. Auto-K — silhouette 스윕]  (표본만 float32 변환 — 전량 변환 회피)
# ══════════════════════════════════════════════════════════════════
def choose_k(feat: torch.Tensor, valid_idx: np.ndarray, level: str):
    if len(valid_idx) < 5000:
        print(f"  [{level}] 유효 표본 {len(valid_idx)}개 — 부족 → K=1 (레벨 비활성)")
        return 1, []

    rng = np.random.default_rng(SEED)
    fit_idx = rng.choice(valid_idx, min(FIT_SAMPLE, len(valid_idx)), replace=False)
    Xf = feat[torch.from_numpy(fit_idx)].numpy().astype(np.float32)
    k_hi = min(K_MAX, len(Xf) - 1)

    sweep = []
    for k in range(K_MIN, k_hi + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=SEED,
                             batch_size=100_000, n_init='auto').fit(Xf)
        sil_idx = rng.choice(len(Xf), min(SIL_SAMPLE, len(Xf)), replace=False)
        try:
            s = silhouette_score(Xf[sil_idx], km.labels_[sil_idx])
        except ValueError:       # 표본에 라벨 1종뿐인 극단 케이스
            continue
        sweep.append((k, float(s)))

    best_k = max(sweep, key=lambda t: t[1])[0]
    print(f"  [{level}] sweep → " + "  ".join(f"k={k}:{s:.3f}" for k, s in sweep))
    print(f"  [{level}] 선택된 K = {best_k} (silhouette 최대)")
    return best_k, sweep


# ══════════════════════════════════════════════════════════════════
# [C. K 보정 — 목표 unique pattern 수 맞추기]
# ══════════════════════════════════════════════════════════════════
def _fit_predict_level(X_val: np.ndarray, k: int, rng) -> tuple:
    """FIT_SAMPLE 표본으로 학습 → 전체 predict. k=1 이면 전부 key 0.
    반환 (labels, centers)."""
    if k <= 1 or len(X_val) == 0:
        centers = X_val.mean(axis=0, keepdims=True) if len(X_val) else None
        return np.zeros(len(X_val), dtype=np.int64), centers
    fit_idx = rng.choice(len(X_val), min(FIT_SAMPLE, len(X_val)), replace=False)
    km = MiniBatchKMeans(n_clusters=min(k, len(fit_idx)), random_state=SEED,
                         batch_size=100_000, n_init='auto').fit(X_val[fit_idx])
    return km.predict(X_val).astype(np.int64), km.cluster_centers_


def calibrate_ks(features: dict, levels: list, base_ks: dict, n_target: int):
    """
    K(α) = round(base_k·α) 로 표본 위 unique path 수를 세고
    α ← α·(n_target/N_obs)^(1/L) 로 갱신 (path 수는 K 에 대해 대체로 단조증가).
    (레벨, K) 결과는 메모이제이션 — α 반복 중 K 가 안 변한 레벨은 재학습하지 않음.
    표본 기반이라 희귀 path 일부 누락 가능 → 전체 데이터 최종 집계와 ±수 % 차이(허용 전제).
    """
    rng = np.random.default_rng(SEED)
    n = features[levels[0]].shape[0]
    samp = np.sort(rng.choice(n, min(CAL_SAMPLE, n), replace=False))
    Xs, empt = {}, {}
    for lv in levels:
        Xs[lv] = features[lv][torch.from_numpy(samp)].numpy().astype(np.float32)
        empt[lv] = (np.abs(Xs[lv]).sum(axis=1) == 0)

    memo = {}
    def level_labels(lv, k):
        if (lv, k) not in memo:
            lab = np.full(len(samp), EMPTY_KEY, dtype=np.int64)
            lab[~empt[lv]], _ = _fit_predict_level(Xs[lv][~empt[lv]], k, rng)
            memo[(lv, k)] = lab
        return memo[(lv, k)]

    alpha, history, best = 1.0, [], None
    for it in range(MAX_CAL_ITER):
        ks = {lv: int(np.clip(round(base_ks[lv] * alpha), 1, K_HARD_MAX)) for lv in levels}
        labs = {lv: level_labels(lv, ks[lv]) for lv in levels}
        n_obs = len(np.unique(encode_paths(labs, levels, ks)))
        err = abs(n_obs - n_target) / n_target
        history.append({'iter': it, 'alpha': round(alpha, 4), 'ks': dict(ks), 'n_obs': int(n_obs)})
        print(f"  [보정 {it+1}/{MAX_CAL_ITER}] α={alpha:.3f}  K={list(ks.values())}  "
              f"unique={n_obs} (목표 {n_target}, 오차 {err:.1%})")
        if best is None or err < best[0]:
            best = (err, ks)
        if err <= TARGET_TOL:
            break
        alpha *= (n_target / max(n_obs, 1)) ** (1.0 / len(levels))

    print(f"  [보정 완료] 최종 K = {best[1]} (오차 {best[0]:.1%})")
    return best[1], history


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    features, all_keys, _chunks = ensure_features()
    n_graphs = len(all_keys)
    print(f"총 그래프 수: {n_graphs:,}\n")

    # ── [B] silhouette base K (레벨 간 비율의 근거) ──────────────
    print("[Step 1] 레벨별 auto-K (silhouette)")
    base_ks, sweeps, empty_full = {}, {}, {}
    for lv in LEVELS:
        empty = (features[lv].abs().sum(dim=1) == 0).numpy()   # fp16 그대로 (변환 없음)
        empty_full[lv] = empty
        valid_idx = np.where(~empty)[0]
        print(f"[Level {lv}] dim={features[lv].shape[1]}  유효 {len(valid_idx):,}/{n_graphs:,} "
              f"(빈 hop {int(empty.sum()):,}개 → key={EMPTY_KEY})")
        base_ks[lv], sweeps[lv] = choose_k(features[lv], valid_idx, lv)

    # ── [C] N_TARGET 보정 ────────────────────────────────────────
    cal_history = None
    if N_TARGET is not None:
        print(f"\n[Step 2] K 보정 — 목표 unique pattern {N_TARGET}개 (±{TARGET_TOL:.0%})")
        ks, cal_history = calibrate_ks(features, LEVELS, base_ks, N_TARGET)
    else:
        ks = base_ks
        print(f"\n[Step 2] N_TARGET 미지정 → silhouette K 그대로: {ks}")

    # ── 최종 클러스터링 (레벨별 float32 변환 → 즉시 해제) ────────
    print("\n[Step 3] 최종 클러스터링 (전체 데이터 라벨링)")
    labels, centroids = {}, {}
    for lv in LEVELS:
        t = time.time()
        valid = ~empty_full[lv]
        Xv = features[lv].numpy()[valid].astype(np.float32)
        lab = np.full(n_graphs, EMPTY_KEY, dtype=np.int64)
        lab[valid], centroids[lv] = _fit_predict_level(Xv, ks[lv], rng)
        labels[lv] = lab
        del Xv
        print(f"  [{lv}] K={ks[lv]}  ({time.time()-t:.1f}s)")

    # ── [D] 전개 트리 ────────────────────────────────────────────
    code = encode_paths(labels, LEVELS, ks)
    uniq_code, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
    uniq_str = [decode_code(int(c), LEVELS, ks) for c in uniq_code]

    max_cells = int(np.prod([ks[lv] + 1 for lv in LEVELS]))    # +1 = 빈 hop key
    print(f"\n{'='*62}")
    print(f"[Unique Patterns] 관측된 path = {len(uniq_code):,}개  "
          f"(이론상 최대 {max_cells:,}칸의 {len(uniq_code)/max_cells:.1%})")
    if N_TARGET is not None:
        print(f"  목표 {N_TARGET} 대비 전체 데이터 실측 오차: "
              f"{abs(len(uniq_code) - N_TARGET) / N_TARGET:.1%}")
    print(f"{'='*62}")

    order = np.argsort(-counts)
    print(f"  상위 20개 path ({' → '.join(LEVELS)}):")
    for j in order[:20]:
        print(f"    {uniq_str[j]:>18s} : {counts[j]:>9,d} ({counts[j]/n_graphs:.2%})")

    # 레벨 간 전이 — 벡터화 (2천만 행 Python 루프 회피)
    transitions = {}
    for a, b in zip(LEVELS[:-1], LEVELS[1:]):
        radix = ks[b] + 2
        up = np.unique((labels[a] + 1) * radix + (labels[b] + 1))
        tr = {}
        for x, y in zip((up // radix - 1).tolist(), (up % radix - 1).tolist()):
            tr.setdefault(x, []).append(y)
        transitions[f"{a}->{b}"] = {str(k): sorted(v) for k, v in sorted(tr.items())}
        avg_branch = np.mean([len(v) for v in tr.values()])
        print(f"  [전이 {a}→{b}] {a} key 당 평균 {avg_branch:.1f}종의 {b} key 관측 "
              f"(무제약이면 {ks[b]+1}종) → 제약 강도 {1 - avg_branch/(ks[b]+1):.0%}")

    # ── [E] edge 포함 여부 진단 ──────────────────────────────────
    #   ※ 열 단위 float64 캐스팅 필수 — edge_raw 는 distance(수백 nm)를 담고 있어
    #     fp16 상태로 제곱하면 65504 를 넘어 inf(오버플로)가 된다.
    er = features['edge_raw'].numpy()
    n_paths, ratios = len(uniq_code), []
    for j in range(er.shape[1]):
        col = er[:, j].astype(np.float64)
        g_var = col.var()
        s1 = np.bincount(inv, weights=col, minlength=n_paths)
        s2 = np.bincount(inv, weights=col * col, minlength=n_paths)
        wv = np.clip(s2 / counts - (s1 / counts) ** 2, 0, None)
        ratios.append(((wv * counts).sum() / counts.sum()) / (g_var + 1e-12))
    ratio = float(np.mean(ratios))
    print(f"\n[Edge 진단] path 내 잔여 edge 분산 비율 = {ratio:.3f}")
    if ratio < 0.3:
        print("  → h-path 가 edge 기하를 대부분 설명. edge 축 불필요 (현 구성 유지).")
    elif ratio < 0.7:
        print("  → 부분적으로만 설명. 정밀 분류가 필요하면 LEVELS 끝에 'edge_raw' 추가 고려.")
    else:
        print("  → h-path 가 edge 를 거의 못 가름. LEVELS 끝에 'edge_raw' 추가 권장.")

    # ── 저장 ─────────────────────────────────────────────────────
    torch.save({'levels': LEVELS, 'ks': ks, 'empty_key': EMPTY_KEY,
                'centroids': centroids, 'seed': SEED},
               os.path.join(_here, f"{OUT_PREFIX}_kmeans.pt"))
    torch.save({'keys': all_keys,
                'labels': {lv: torch.from_numpy(labels[lv].astype(np.int16)) for lv in LEVELS},
                'path_code': torch.from_numpy(code)},
               os.path.join(_here, f"{OUT_PREFIX}_labels.pt"))

    report = {
        'levels': LEVELS, 'base_ks': base_ks, 'final_ks': ks,
        'silhouette_sweep': sweeps,
        'n_target': N_TARGET, 'calibration_history': cal_history,
        'n_graphs': n_graphs,
        'n_unique_patterns': int(len(uniq_code)), 'max_cells': max_cells,
        'transitions': transitions,
        'path_counts': {uniq_str[j]: int(counts[j]) for j in order},
        'edge_residual_variance_ratio': ratio,
    }
    with open(os.path.join(_here, f"{OUT_PREFIX}_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {OUT_PREFIX}_kmeans.pt / {OUT_PREFIX}_labels.pt / {OUT_PREFIX}_report.json")

    if SAVE_CSV:
        csv_path = os.path.join(_here, f"{OUT_PREFIX}_labels.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['gauge_name'] + [f'key_{lv}' for lv in LEVELS] + ['pattern_id'])
            for i in range(n_graphs):
                w.writerow([all_keys[i]] + [int(labels[lv][i]) for lv in LEVELS]
                           + [uniq_str[inv[i]]])
        print(f"[Saved] {csv_path}")

    print(f"\nTotal Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
