"""
second_classify.py — 2차 미시 분류 (거시 군집 내부 → 루베인 미시 분할)

파이프라인 위치:
  h0_clustering.py (1차 HDBSCAN + 노이즈 soft 편입 완료)
      → h0_labels_full.npz  (rows=전역 인덱스, labels=macro_label, -1 없음)
  hkeys_features.pt          (features: h0/h1/h2/h3/edge 각 8차원 → 40차원)
      → (여기) 거시 군집 내부를 루베인으로 미시 분할, 최종 목표 1,000 군집

핵심 설계 (요구사항):
  Step1  거시 군집 Ci 크기 비례로 목표 하위 클러스터 수 Ki 할당 (최소 1)
  Step2  5개 피처 블록 가중치 W=[w0,w1,w2,w3,w_edge] 최적 탐색 (합=1)
  Step3  W_opt 적용 → 거시 군집별 적응형 k-NN 그래프 → 루베인 미시 분할
         (K 직접 지정 불가 → resolution 이진 탐색으로 Ki 에 맞춤. KMeans 금지)
  Step4  모듈러티 / 실루엣 / Local DBCV 로 분할 품질 검증

argv 없음 — 아래 CONFIG 만 고쳐서 실행:  python second_classify.py
"""

import logging

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ── 선택적 의존성 (없으면 명확히 실패 / 폴백) ─────────────────────────
try:
    import networkx as nx
    import community as community_louvain          # python-louvain
    _HAS_LOUVAIN = True
except ImportError:                                # pragma: no cover
    _HAS_LOUVAIN = False

try:
    from hdbscan.validity import validity_index    # DBCV (Local DBCV 용)
    _HAS_DBCV = True
except ImportError:                                # pragma: no cover
    _HAS_DBCV = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("second_classify")


class SecondaryLouvainClassifier(BaseEstimator, TransformerMixin):
    """거시 군집 내부를 루베인으로 미시 분할하는 Scikit-learn 호환 트랜스포머.

    입력  X : (N, n_blocks*block_dim)  블록 순서 [h0,h1,h2,h3,edge]
          y : (N,)  1차 macro_label (노이즈 -1 없음, 전원 유효)
    출력  transform(X) → (N, 1)  최종 미시 클러스터 라벨 (전역 고유 정수)

    Parameters
    ----------
    total_clusters : 최종 목표 총 클러스터 수 (기본 1000)
    n_blocks, block_dim : 피처 블록 수 / 블록 차원 (5, 8 → 40차원)
    weight_trials : 가중치 Random Search 시도 횟수
    weight_sample : 가중치 평가 시 거시 군집당 샘플 상한
    weight_metric : 'silhouette'(최대화) | 'dbi'(최소화)
    knn_frac, knn_min, knn_max : 적응형 k = clip(frac*크기, min, max)
    res_bounds : 루베인 resolution 이진 탐색 (low, high)
    res_iter : resolution 이진 탐색 최대 반복
    """

    def __init__(self, total_clusters=1000, n_blocks=5, block_dim=8,
                 weight_trials=30, weight_sample=1500, weight_metric="silhouette",
                 knn_frac=0.02, knn_min=10, knn_max=50,
                 res_bounds=(0.05, 30.0), res_iter=18,
                 random_state=42, verbose=True):
        self.total_clusters = total_clusters
        self.n_blocks = n_blocks
        self.block_dim = block_dim
        self.weight_trials = weight_trials
        self.weight_sample = weight_sample
        self.weight_metric = weight_metric
        self.knn_frac = knn_frac
        self.knn_min = knn_min
        self.knn_max = knn_max
        self.res_bounds = res_bounds
        self.res_iter = res_iter
        self.random_state = random_state
        self.verbose = verbose

    # ── 유틸 ──────────────────────────────────────────────────────────
    def _log(self, msg):
        if self.verbose:
            log.info(msg)

    def _check(self, X, y=None):
        """형상(Shape) 검증: X 는 40차원(=n_blocks*block_dim), y 는 노이즈 없어야 함."""
        X = np.asarray(X, dtype=np.float32)
        exp = self.n_blocks * self.block_dim
        if X.ndim != 2 or X.shape[1] != exp:
            raise ValueError(f"X 형상 오류: (N,{exp}) 기대, 받은 값 {X.shape}")
        if y is not None:
            y = np.asarray(y)
            if y.shape[0] != X.shape[0]:
                raise ValueError(f"X/y 길이 불일치: {X.shape[0]} vs {y.shape[0]}")
            if (y < 0).any():
                raise ValueError("y 에 노이즈 라벨(-1) 존재 — 1차 처리가 미완료입니다")
        return X, y

    def _apply_weights(self, Xs, W):
        """표준화된 피처에 블록별 가중치를 곱한다. (N,40) → 블록별 스케일링."""
        B = Xs.reshape(-1, self.n_blocks, self.block_dim)
        return (B * W[None, :, None]).reshape(len(Xs), -1)

    # ── Step 3 핵심: 그래프 빌드 + 루베인 (resolution 이진 탐색) ───────
    def _louvain_at(self, G, resolution):
        """주어진 resolution 에서 루베인 1회 → (라벨배열, 커뮤니티수, 모듈러티)."""
        part = community_louvain.best_partition(
            G, resolution=resolution, random_state=self.random_state)
        lab = np.array([part[i] for i in range(G.number_of_nodes())])
        mod = community_louvain.modularity(part, G)
        return lab, len(set(part.values())), mod

    def _build_graph(self, Xw):
        """적응형 k-NN 그래프. 거리(mode='distance') → 역거리 유사도 가중치 간선.

        k 는 군집 크기에 비례(adaptive). 그래프가 자연스레 끊긴 성분(희소 영역)은
        루베인에서 독립 커뮤니티로 승격되어 소외되지 않는다."""
        m = len(Xw)
        k = int(np.clip(int(self.knn_frac * m), self.knn_min, self.knn_max))
        k = min(k, m - 1)
        A = kneighbors_graph(Xw, n_neighbors=k, mode="distance", include_self=False)
        A = A.maximum(A.T)                       # 대칭화 (상호 최근접 보존)
        A.data = 1.0 / (A.data + 1e-9)           # 거리에 반비례하는 간선 가중치
        return nx.from_scipy_sparse_array(A), k

    def _split_cluster(self, Xw, Ki):
        """━━ [핵심 제어 루프] resolution 이진 탐색으로 커뮤니티 수를 Ki 에 맞춤 ━━

        루베인은 분할 개수 K 를 직접 지정할 수 없다. resolution 이 커질수록
        커뮤니티 수는 (단조에 가깝게) 증가하므로, 이진 탐색으로 Ki 에 가장 근접한
        resolution 을 찾는다. Ki==1 이거나 분할 불가 크기면 통짜 1군집 반환."""
        m = len(Xw)
        if Ki <= 1 or m <= 2:
            return np.zeros(m, dtype=int), float("nan")

        G, k = self._build_graph(Xw)
        lo, hi = self.res_bounds
        best_lab, best_mod, best_gap = None, float("nan"), np.inf

        # ── resolution 이진 탐색: 목표 Ki 커뮤니티에 수렴 ──
        for it in range(self.res_iter):
            mid = 0.5 * (lo + hi)
            lab, n_comm, mod = self._louvain_at(G, mid)
            gap = abs(n_comm - Ki)
            if gap < best_gap:                   # 지금까지 Ki 에 가장 근접한 분할 채택
                best_lab, best_mod, best_gap = lab, mod, gap
            if n_comm == Ki:                     # 정확히 일치 → 조기 종료
                break
            if n_comm < Ki:                      # 커뮤니티 부족 → resolution ↑
                lo = mid
            else:                                # 커뮤니티 과다 → resolution ↓
                hi = mid

        got = len(np.unique(best_lab))
        self._log(f"    [split] m={m:,} k={k} 목표Ki={Ki} → 실제={got} "
                  f"(gap={best_gap:.0f}) modularity={best_mod:.3f}")
        return best_lab, best_mod

    # ── Step 2: 피처별 최적 가중치 탐색 (Random Search) ───────────────
    def _sample_weight(self, rng):
        """W=[w0..w_edge] 를 Dirichlet 로 샘플 → 각 0~1, 합=1 자동 정규화."""
        return rng.dirichlet(np.ones(self.n_blocks)).astype(np.float32)

    def _score_weight(self, Xs, y, W, eval_cids, rng):
        """━━ 가중치 W 의 미시 분리도 평가 ━━

        평가 대상 거시 군집들에서 W 를 적용해 피처를 결합하고, resolution=1.0 으로
        임시(temporary) 루베인 분할을 수행한 뒤 실루엣/DBI 로 분리도를 산출한다.
        여러 군집 점수의 평균을 W 의 적합도로 반환한다."""
        scores = []
        for cid in eval_cids:
            idx = np.where(y == cid)[0]
            if len(idx) > self.weight_sample:            # 평가 비용 제한: 군집당 샘플
                idx = rng.choice(idx, self.weight_sample, replace=False)
            if len(idx) < 20:
                continue
            Xw = self._apply_weights(Xs[idx], W)
            G, _ = self._build_graph(Xw)
            lab, n_comm, _ = self._louvain_at(G, 1.0)    # 임시 분할 (고정 resolution)
            if n_comm < 2 or n_comm >= len(idx):         # 분리 불가 → 평가 제외
                continue
            if self.weight_metric == "dbi":
                scores.append(-davies_bouldin_score(Xw, lab))   # 최소화 → 부호 반전
            else:
                scores.append(silhouette_score(Xw, lab))        # 최대화
        return float(np.mean(scores)) if scores else -np.inf

    def _search_weights(self, Xs, y, cluster_ids):
        """━━ 가중치 Random Search 루프 ━━

        균등 가중치를 기준선으로 두고, Dirichlet 샘플 W 후보들을 반복 평가하여
        미시 분리도(실루엣↑/DBI↓)가 최대가 되는 W_opt 를 탐색한다. h0 가 edge 를
        이미 중복 보유하므로 특정 블록에 고정 가중치를 두지 않고 전부 탐색 대상."""
        rng = np.random.default_rng(self.random_state)
        # 평가용 거시 군집: 큰 순서로 상위 몇 개만 (탐색 비용 절감, 대표성 확보)
        sizes = {c: int((y == c).sum()) for c in cluster_ids}
        eval_cids = sorted(sizes, key=sizes.get, reverse=True)[:min(5, len(cluster_ids))]

        uniform = np.full(self.n_blocks, 1.0 / self.n_blocks, dtype=np.float32)
        best_W = uniform
        best_s = self._score_weight(Xs, y, uniform, eval_cids, rng)
        self._log(f"  [W-search] 기준(균등) W={np.round(uniform,3)} "
                  f"{self.weight_metric}={best_s:.4f}")

        for t in range(self.weight_trials):
            W = self._sample_weight(rng)                 # 후보 가중치 (합=1)
            s = self._score_weight(Xs, y, W, eval_cids, rng)
            if s > best_s:                               # 더 나은 분리도 → 갱신
                best_W, best_s = W, s
                self._log(f"  [W-search] {t+1:>3}/{self.weight_trials} "
                          f"★갱신 W={np.round(W,3)} {self.weight_metric}={s:.4f}")
        self._log(f"  [W-search] W_opt={np.round(best_W,3)} "
                  f"(score={best_s:.4f})")
        return best_W

    # ── Step 4: Local DBCV (거시 군집별 DBCV 평균) ────────────────────
    def _local_dbcv(self, Xw_full, y, final_labels):
        """최종 퀄리티 체크: 거시 군집 내부의 세부 라벨에 대해 DBCV 를 산출·평균.
        hdbscan.validity 없으면 실루엣으로 폴백."""
        vals = []
        for cid in np.unique(y):
            idx = np.where(y == cid)[0]
            sub = final_labels[idx]
            if len(np.unique(sub)) < 2:
                continue
            Xc = np.ascontiguousarray(Xw_full[idx], dtype=np.float64)
            try:
                if _HAS_DBCV:
                    vals.append(validity_index(Xc, sub.astype(np.int64)))
                else:
                    vals.append(silhouette_score(Xc, sub))
            except Exception:                            # 수치 불안정 군집은 건너뜀
                continue
        return float(np.mean(vals)) if vals else float("nan")

    # ── 메인 fit ──────────────────────────────────────────────────────
    def fit(self, X, y):
        X, y = self._check(X, y)
        N = len(X)
        cluster_ids = np.unique(y)
        self._log(f"입력: N={N:,}  거시 군집 {len(cluster_ids)}개  "
                  f"차원={X.shape[1]} ({self.n_blocks}×{self.block_dim})")

        # 블록 비교 가능하도록 40차원 표준화 (가중치·거리의 전제)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)

        # ── Step 1: 거시 군집별 목표 하위 클러스터 수 Ki ──
        # Ki = max(1, round(total * Ci / N))
        self.target_ki_ = {int(c): max(1, int(round(self.total_clusters *
                            int((y == c).sum()) / N))) for c in cluster_ids}
        self._log(f"[Step1] Ki 합계={sum(self.target_ki_.values())} "
                  f"(목표 {self.total_clusters})")

        # ── Step 2: 최적 가중치 W_opt ──
        self._log("[Step2] 피처 블록 가중치 탐색 시작")
        self.weights_ = self._search_weights(Xs, y, cluster_ids)

        # ── Step 3: W_opt 적용 → 군집별 k-NN 그래프 + 루베인 미시 분할 ──
        self._log("[Step3] 거시 군집별 루베인 미시 분할")
        Xw = self._apply_weights(Xs, self.weights_)
        final = np.full(N, -1, dtype=np.int64)
        self.modularity_ = {}
        offset = 0                                       # 전역 고유 라벨 오프셋
        for cid in cluster_ids:
            idx = np.where(y == cid)[0]
            Ki = self.target_ki_[int(cid)]
            sub, mod = self._split_cluster(Xw[idx], Ki)
            final[idx] = sub + offset                    # (macro,sub) → 전역 고유
            offset += len(np.unique(sub))
            self.modularity_[int(cid)] = mod

        self.labels_ = final
        self.n_clusters_ = len(np.unique(final))

        # ── Step 4: 검증 지표 ──
        valid_mods = [m for m in self.modularity_.values() if not np.isnan(m)]
        self.mean_modularity_ = float(np.mean(valid_mods)) if valid_mods else float("nan")
        self.local_dbcv_ = self._local_dbcv(Xw, y, final)
        self._log(f"[Step4] 최종 군집수={self.n_clusters_}  "
                  f"평균 모듈러티={self.mean_modularity_:.3f}  "
                  f"Local DBCV={self.local_dbcv_:.3f}")
        return self

    def transform(self, X):
        """학습된 최종 라벨을 (N,1) 로 반환 (Scikit-learn 파이프라인 호환)."""
        if not hasattr(self, "labels_"):
            raise RuntimeError("fit 을 먼저 호출하세요")
        return self.labels_.reshape(-1, 1)

    def fit_transform(self, X, y=None, **kw):
        return self.fit(X, y).transform(X)


# ══════════════════════════════════════════════════════════════════
# CONFIG — argv 없이 여기서 변수만 고쳐 실행:  python second_classify.py
# ══════════════════════════════════════════════════════════════════
import os

_here        = os.path.dirname(os.path.abspath(__file__))
FEATURE_CACHE = os.path.join(_here, "hkeys_features.pt")                 # 40차원 원천
MACRO_LABELS  = os.path.join(_here, "h0_clustering_out", "h0_labels_full.npz")
OUT_PATH      = os.path.join(_here, "second_classify_labels.npz")

BLOCKS        = ["h0", "h1", "h2", "h3", "edge"]     # 블록 순서 (각 8차원 → 40)
TOTAL_CLUSTERS = 1000


def _load_inputs():
    """hkeys_features.pt(40차원) + h0_labels_full.npz(macro_label) 결합.
    rows(전역 인덱스)로 정렬해 X(40차원)·y(macro_label) 를 만든다."""
    import torch
    d = torch.load(FEATURE_CACHE, map_location="cpu", weights_only=False)
    feats = d["features"]
    blocks = [feats[b].numpy().astype(np.float32) for b in BLOCKS]
    X_full = np.concatenate(blocks, axis=1)          # (전체, 40)

    m = np.load(MACRO_LABELS)
    rows, y = m["rows"], m["labels"]                 # rows=전역 인덱스, y=macro_label
    return X_full[rows], y


def main():
    if not _HAS_LOUVAIN:
        raise ImportError("networkx + python-louvain 필요: "
                          "pip install networkx python-louvain")
    log.info("입력 데이터 로드")
    X, y = _load_inputs()

    model = SecondaryLouvainClassifier(total_clusters=TOTAL_CLUSTERS,
                                       n_blocks=len(BLOCKS), block_dim=8)
    labels = model.fit_transform(X, y).ravel()

    np.savez(OUT_PATH, labels=labels, macro_labels=y,
             weights=model.weights_)
    log.info(f"저장: {OUT_PATH}  (최종 {model.n_clusters_}군집, "
             f"평균 모듈러티 {model.mean_modularity_:.3f}, "
             f"Local DBCV {model.local_dbcv_:.3f})")


if __name__ == "__main__":
    main()
