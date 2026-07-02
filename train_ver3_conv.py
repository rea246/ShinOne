"""
train_ver3_conv.py  —  train_ver2.py 로부터 분기한 실험용 버전 (base: epoch188 종료 런)
train_ver2.py 대비 변경점:
  1. VICREG_WEIGHT_COV 0.1 → 0.5 — ep180 기준 vic_cov 가 0.75 로 높게 남아 있어
     (weight 반영 후 total loss 기여는 0.075 로 작았음) 차원 간 상관 억제 압력을 올림.
     다른 후보(코드 미적용, 아래 "Cov 하향 대안" 참조)보다 위험이 가장 낮고
     그래프 클러스터링에 실제로 쓰는 z(graph_emb) 자체에 직접 압력을 준다는 점에서 채택.
  2. best 체크포인트(ckpt_name) 저장 대상을 encoder 단독 → 전체 GAE(model.state_dict())로 변경.
     (기존엔 best_gat_v2_*.pt 가 encoder 만 담아 center_proj 가 없어서, 다운스트림
     visualize_encoder.py/extract_representatives.py 가 "최종 epoch" 상태인
     checkpoint_last_v2.pt(RESUME) 를 쓸 수밖에 없었음 — early stopping patience 만큼
     성능이 이미 떨어진 상태일 수 있음. 이제 best 시점의 전체 모델을 그대로 보관한다.)
  3. RESUME_PATH / ckpt_name 파일명을 v3_conv 전용으로 분리 — train_ver2.py 와
     같은 폴더에서 실행해도 서로의 체크포인트를 덮어쓰지 않는다.
     (global_scales 캐시는 전처리 데이터에서만 결정되는 값이라 v2 와 공유해도 무방 → 그대로 재사용)
  4. EMBEDDING_DIM 8 → 16 — 애초에 "Cov 하향 대안" 목록엔 별도 실험으로 분리 권장했지만,
     사용자 요청으로 Cov=0.5 변경과 함께 이 파일에 합쳐서 적용. 두 변경이 섞여 있어
     "무엇 때문에 좋아졌는지" 원인 분리는 안 됨(필요하면 나중에 EMBEDDING_DIM=8 로 되돌려
     Cov=0.5 단독 효과와 비교할 것). hidden_dim=max(16, embedding_dim*2) 등 관련 레이어
     크기는 EMBEDDING_DIM 에서 자동 계산되므로 이 상수 외에 추가로 손댈 곳은 없다.
     주의: 차원이 늘면 dim=8 대비 상관시킬 축이 더 많아져 Cov 항이 더 커질 여지도 있음 —
     "차원 늘리기"와 "Cov weight 올리기"가 서로 다른 방향으로 당길 수 있으니 vic_cov 추이를
     v2/v3_conv(dim=8) 로그와 비교해서 지켜볼 것.

train_ver2.py 원본 변경 이력(참고):
  1. vicreg_loss() 추가 — 3항 완전 구현
  2. compute_losses()  — clean forward 추가로 Invariance term 계산
  3. train_one_epoch() — clean forward pass 포함
  4. validate_epoch()  — VICReg 3항 모니터링 추가
  5. 로그 — VICReg 세부 분해 출력

Cov 하향 대안 (이 파일엔 미적용 — 방향성이 다르거나 트레이드오프가 있어 보류):
  - Covariance → Correlation 정규화(Barlow Twins 식, 차원별 표준화 후 off-diag 제곽)
    장점: 각 차원의 raw variance 스케일에 덜 휘둘려 더 확실하게 상관 제거.
    단점: loss 형태가 바뀌어 weight 재튜닝 필요, Variance 항과 일부 역할 중복.
  - 프로젝터/expander head 추가(VICReg 원 논문 구조: 고차원 projector 출력에만 Cov 적용,
    downstream 은 backbone 출력 사용)
    장점: 표준 SSL 레시피, backbone(z)에 과도한 제약을 주지 않아 재구성 품질 보존에 유리.
    단점: **이 프로젝트엔 부적합할 가능성 높음** — 우리가 최종적으로 쓰는 건 projector 가
    아니라 graph_emb(z) 자체(클러스터링/대표추출 입력)라서, Cov 압력을 projector 에만 걸면
    정작 다운스트림에 쓰는 z 는 상관 제거 혜택을 못 받음.
  - BATCH_SIZE 추가 증가
    장점: Var/Cov 는 배치 통계라 배치가 클수록 추정이 안정적.
    단점: 이미 2048 로 큼, GPU 메모리/LR 재조정 부담 대비 효과 불확실.
"""

import os
import gc
import time
import glob
import random
import datetime
import threading
import concurrent.futures
from multiprocessing import cpu_count

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

# fast 포맷 로더/스케일 계산은 이 파일 하단에 인라인 구현되어 있다.
# (preprocessor.py 의 _pack_fast_chunk 가 저장하는 포맷 전용 — 외부 의존 없음)

# ══════════════════════════════════════════════════════════════════
# [Global Config]
# ══════════════════════════════════════════════════════════════════
# NOISE_STD — denoising 강도. dim 을 키우면(아래 8) 큰 latent 가 입력을 거의
# 복사할 여지가 생기므로, 노이즈를 함께 올려 "복사로는 못 푸는" 압력을 유지한다.
NOISE_STD  = 0.1
# BATCH_SIZE — VICReg의 Var/Cov는 배치 통계라 배치가 클수록 추정이 안정적이다.
# 다만 32→2048 은 64배라, 에포크당 업데이트 횟수가 64배 줄어든다.
# → 수렴 속도 유지를 위해 LR을 sqrt 규칙으로 함께 올린다(아래 main 참조).
BATCH_SIZE = 2048
REFERENCE_BATCH = 32   # LR sqrt-scaling 기준 배치 (원래 lr=1e-4 가 검증된 배치)

# NUM_WORKERS — DataLoader 병렬 collation 워커 수.
#   0 이면 메인 스레드가 단독으로 수천 개 작은 그래프를 매 배치 PyG Batch 로 합쳐
#   GPU 가 데이터를 기다리며 논다(= GPU 메모리 적게 쓰는데 느린 전형적 증상).
#   >0 이면 collation 을 CPU 코어로 분산해 GPU 연산과 겹친다.
#   - Linux(fork): 프리로드 텐서를 copy-on-write 로 공유 → 메모리 이득 거의 없이 빠름.
#     (read-only 슬라이싱이라 5.3GB 데이터 페이지는 워커 간 공유 유지 → RAM 폭증 없음)
#   - Windows(spawn): 워커가 데이터셋을 복제 → RAM 급증 시 0 으로 되돌릴 것.
# bsub -n 8 기준: 워커 6 + 메인 1 = 7 (코어 1 여유). span[hosts=1] 로 한 호스트에 몰 것.
NUM_WORKERS = 6

# USE_TORCH_COMPILE — 실험적/기본 OFF. GAT 레이어들이 hidden_dim 이 작은(16~32) 커널을
# 배치마다 아주 많이 launch 하는 구조라(attention, edge_mlp, LayerNorm, GELU 등)
# launch-overhead-bound 일 가능성이 높고, 이 경우 torch.compile 의 커널 퓨전이
# t_step 을 크게 줄여줄 수 있다. 다만:
#   - 그래프마다 노드/엣지 수가 달라 배치 shape 이 계속 바뀐다 → dynamic=True 없이 켜면
#     매 배치 재컴파일(guard 재수집)이 발생해 오히려 느려질 수 있어 dynamic=True 로 켤 것.
#   - torch_geometric 의 scatter/GATv2Conv 연산이 inductor 백엔드에서 100% 검증된 조합은
#     아니라 이 환경(torch_geometric 미설치 샌드박스)에선 실행 테스트를 못 했다.
# → 켜서 몇 epoch 만 돌려보고 t_step 이 줄었는지 직접 확인 후 유지 여부 결정할 것.
USE_TORCH_COMPILE = False

# embedding_dim — 노드 임베딩 차원.
# 노드 디코더는 z_node(embedding_dim) -> num_node_features(=4) 복원이라,
# dim=3 이면 3->4 로 정보 병목이 심해 재구성 정확도의 상한을 누른다(epoch 140에서
# h1~39%/E_Space~26% 로 조기 수렴 관찰됨).
# 원래 dim 을 작게 둔 이유(copy-collapse 방지)는 이제 denoising + VICReg 가 대신
# 막아주므로, 병목을 완화해 표현력을 키운다: 3 -> 8 -> 16(1차 실험, Cov=0.5 와 병행) -> 9.
# 16 은 Cov(0.75→0.32)는 확실히 낮췄지만 Inv/Var/재구성 오차가 전반적으로 소폭 악화되고
# patience(30)에 62 epoch 만에 조기종료돼, "8과 16 중간"으로 좁혀 8과 16 사이 최소 증분
# 효과를 다시 본다. dim=9 는 아키텍처상 아무 제약이 없어(hidden_dim=max(16, dim*2)=18,
# GATv2Conv 는 heads 로 나눠떨어질 필요 없이 head 당 out_channels=dim 을 그대로 쓰고
# concat 후 node_head_proj 로 투영 — 홀수/짝수 무관하게 동작) 8로 원복할 필요는 없다.
# 다만 GPU tensor-core 는 8의 배수(FP16/BF16)에서 살짝 더 효율적이라 8/16 대비 미세한
# 속도 손해가 있을 수 있음(수치 정확도엔 영향 없음, 무시 가능한 수준).
# (copy 여부/collapse 여부는 visualize_encoder 의 effective rank / 클러스터 품질로 검증.)
EMBEDDING_DIM = 9

# VICReg 가중치 — 재구성 loss (≈1~5 수렴 구간)와 스케일 맞춤
# weight_inv : clean/noisy 임베딩을 가깝게 → 같은 패턴 = 같은 임베딩
# weight_var : 각 차원 std >= 1 강제 → collapse 방지
# weight_cov : 차원 간 상관 제거 → 정보 중복 방지
VICREG_WEIGHT_INV = 1.0   # Invariance (MSE between z_clean, z_noisy)
VICREG_WEIGHT_VAR = 5.0   # Variance   (relu(1 - std))
# Covariance (off-diagonal suppression) — v2 에서 0.1 로 ep180 raw vic_cov≈0.75 (loss 기여 0.075)
# 로 잔류 상관이 남음. 0.5 로 올려 raw cov 값 자체가 낮아지도록 직접 압력을 강화(v3_conv 실험).
# recon_total(대략 weight_hop 합 기준 1~수 단위) 대비 여전히 작아 재구성을 크게 해치진 않을
# 것으로 예상 — 그래도 h0/h1/E_space 가 학습 중 눈에 띄게 나빠지면 0.3 선으로 낮출 것.
VICREG_WEIGHT_COV = 0.5   # Covariance (off-diagonal suppression) — v2 대비 0.1→0.5 (실험)

FOLDER_PATH = "C:/Users/rea24/Documents/shinwon_note/pattern/2.Pattern_Classification/3.CODE/pythonProject2/dummy_dataset"


# ══════════════════════════════════════════════════════════════════
# [Data Loading — layout_graphencoder.py 와 동일]
# ══════════════════════════════════════════════════════════════════
def _parallel_file_worker(f):
    raw_data = torch.load(f, map_location='cpu', weights_only=False)
    local_x, local_edge_index, local_edge_attr, local_hop_masks, local_metadata = [], [], [], [], []

    for key, val in raw_data.items():
        target = val['local'] if 'local' in val else val
        if target['edge_attr'].size(0) == 0:
            continue

        x          = target['x'].to(torch.float16)
        edge_attr  = target['edge_attr'].to(torch.float16)
        edge_index = target['edge_index']
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        assert edge_index.shape[0] == 2
        edge_index = edge_index.to(torch.int32)

        h0 = target['hop0_mask'].bool()
        h1 = target['hop1_mask'].bool()
        h2 = target['hop2_mask'].bool()
        h3 = target['hop3_mask'].bool()
        packed = (h0.to(torch.int8) | (h1.to(torch.int8) << 1) |
                  (h2.to(torch.int8) << 2) | (h3.to(torch.int8) << 3))

        local_x.append(x);  local_edge_index.append(edge_index)
        local_edge_attr.append(edge_attr);  local_hop_masks.append(packed)
        local_metadata.append((x.size(0), edge_attr.size(0)))

    del raw_data
    if not local_x:
        return None

    x_cat         = torch.cat(local_x, dim=0);          del local_x
    edge_index_cat = torch.cat(local_edge_index, dim=1); del local_edge_index
    edge_attr_cat  = torch.cat(local_edge_attr, dim=0);  del local_edge_attr
    hop_masks_cat  = torch.cat(local_hop_masks, dim=0);  del local_hop_masks

    return {'x': x_cat, 'edge_index': edge_index_cat,
            'edge_attr': edge_attr_cat, 'hop_masks': hop_masks_cat,
            'metadata': local_metadata}


class CompactPreloadedDataset(Dataset):
    def __init__(self, file_paths):
        super().__init__()
        max_workers = min(cpu_count(), len(file_paths), 8)
        print(f"스레드 풀 가속 엔진 기동 ({max_workers}개 스레드)...")

        temp_x, temp_ei, temp_ea, temp_hm = [], [], [], []
        self.slices = []
        x_off = edge_off = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_parallel_file_worker, file_paths))

        for res in results:
            if res is None:
                continue
            temp_x.append(res['x']);    temp_ei.append(res['edge_index'])
            temp_ea.append(res['edge_attr']); temp_hm.append(res['hop_masks'])
            for nn_, ne in res['metadata']:
                self.slices.append((x_off, x_off + nn_, edge_off, edge_off + ne))
                x_off += nn_;  edge_off += ne

        del results;  gc.collect()
        self.all_x          = torch.cat(temp_x, dim=0);  del temp_x
        self.all_edge_index = torch.cat(temp_ei, dim=1); del temp_ei
        self.all_edge_attr  = torch.cat(temp_ea, dim=0); del temp_ea
        self.all_hop_masks  = torch.cat(temp_hm, dim=0); del temp_hm
        gc.collect()
        print(f"총 {len(self.slices)}개 그래프 시스템 RAM 안착 완료!")

    def __len__(self):  return len(self.slices)

    def __getitem__(self, idx):
        xs, xe, es, ee = self.slices[idx]
        x          = self.all_x[xs:xe]
        edge_attr  = self.all_edge_attr[es:ee]
        edge_index = self.all_edge_index[:, es:ee]
        packed     = self.all_hop_masks[xs:xe]
        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            hop0_mask=(packed & 1).bool(),
            hop1_mask=((packed >> 1) & 1).bool(),
            hop2_mask=((packed >> 2) & 1).bool(),
            hop3_mask=((packed >> 3) & 1).bool()
        )


# ══════════════════════════════════════════════════════════════════
# [Fast Format — preprocessor.py 산출물 전용 로더 / 스케일 계산]
#   preprocessor._pack_fast_chunk 가 저장하는 포맷:
#     x          : (N,4) float16   노드 feature [cx, cy, w, h]
#     edge_index : (2,E) int32      그래프-로컬 인덱스 (양방향)
#     edge_attr  : (E,4) float16    [box_dist, center_dist, unit_x, unit_y]
#     hop_packed : (N,)  int8       bit0=hop0 .. bit3=hop3+
#     meta       : (G,4) int64      그래프별 슬라이스 [x0, x1, e0, e1]
#     keys       : [gauge_name, ...]
# ══════════════════════════════════════════════════════════════════
class FastPreloadedDataset(Dataset):
    """여러 PREPROCESSED_*.pt 청크를 RAM 에 병합 후 그래프 단위로 슬라이싱해 반환."""
    def __init__(self, file_paths):
        super().__init__()
        xs, eis, eas, hms = [], [], [], []
        self.slices = []                 # (x0,x1,e0,e1) — 전역 오프셋
        x_off = e_off = 0

        for f in file_paths:
            d = torch.load(f, map_location='cpu', weights_only=False)
            x, ei, ea = d['x'], d['edge_index'], d['edge_attr']
            hp, meta = d['hop_packed'], d['meta']

            xs.append(x); eis.append(ei); eas.append(ea); hms.append(hp)
            # edge_index 값은 그래프-로컬(0-base)이라 전역 오프셋을 더하지 않는다.
            # meta 슬라이스만 전역 오프셋으로 옮겨서 __getitem__ 이 올바른 블록을 자르게 한다.
            for x0, x1, e0, e1 in meta.tolist():
                self.slices.append((x0 + x_off, x1 + x_off, e0 + e_off, e1 + e_off))
            x_off += x.size(0)
            e_off += ei.size(1)

        self.all_x          = torch.cat(xs, dim=0)
        self.all_edge_index = torch.cat(eis, dim=1)   # (2, totalE) — 값은 그래프-로컬
        self.all_edge_attr  = torch.cat(eas, dim=0)
        self.all_hop_packed = torch.cat(hms, dim=0)
        del xs, eis, eas, hms
        gc.collect()
        print(f"총 {len(self.slices)}개 그래프 시스템 RAM 안착 완료! (fast format)")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        xs, xe, es, ee = self.slices[idx]
        x          = self.all_x[xs:xe].float()
        edge_attr  = self.all_edge_attr[es:ee].float()
        edge_index = self.all_edge_index[:, es:ee].long()   # 로컬 인덱스 → 슬라이스한 x 와 정합
        packed     = self.all_hop_packed[xs:xe]
        return Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            hop0_mask=(packed & 1).bool(),
            hop1_mask=((packed >> 1) & 1).bool(),
            hop2_mask=((packed >> 2) & 1).bool(),
            hop3_mask=((packed >> 3) & 1).bool()
        )


def compute_global_scales_fast(file_paths):
    """
    사전 병합된 x / edge_attr 전체 행 기준으로 feature 별 std(스케일) 계산.
    (샘플링 없이 스트리밍 누적 → 메모리 안전)
    반환: (node_scale (Fn,), edge_scale (Fe,))  — 0 방지용 하한 1e-6 클램프.
    """
    n_cnt = e_cnt = 0
    n_sum = n_sqsum = e_sum = e_sqsum = None

    for f in file_paths:
        d = torch.load(f, map_location='cpu', weights_only=False)
        x  = d['x'].float()
        ea = d['edge_attr'].float()

        xs, xsq = x.sum(0), (x * x).sum(0)
        es, esq = ea.sum(0), (ea * ea).sum(0)
        if n_sum is None:
            n_sum, n_sqsum, e_sum, e_sqsum = xs, xsq, es, esq
        else:
            n_sum += xs;  n_sqsum += xsq
            e_sum += es;  e_sqsum += esq
        n_cnt += x.size(0)
        e_cnt += ea.size(0)

    if n_cnt == 0 or e_cnt == 0:
        raise ValueError("스케일 계산 실패: 유효한 노드/엣지가 없습니다.")

    n_mean = n_sum / n_cnt
    e_mean = e_sum / e_cnt
    node_scale = torch.sqrt((n_sqsum / n_cnt - n_mean ** 2).clamp_min(0)).clamp_min(1e-6)
    edge_scale = torch.sqrt((e_sqsum / e_cnt - e_mean ** 2).clamp_min(0)).clamp_min(1e-6)
    return node_scale, edge_scale


def load_graph(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def convert_pyg_data_direct(raw_data):
    data_list = []
    for key, val in raw_data.items():
        target = val['local'] if 'local' in val else val
        x          = target['x'].float()
        edge_index = target['edge_index']
        if edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()
        edge_attr = target['edge_attr'].float()
        if edge_attr.size(0) == 0:
            continue
        data_list.append(Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr,
            hop0_mask=target['hop0_mask'].bool(),
            hop1_mask=target['hop1_mask'].bool(),
            hop2_mask=target['hop2_mask'].bool(),
            hop3_mask=target['hop3_mask'].bool()
        ))
    return data_list


def get_global_scales(file_paths, cache_path="global_scales_train_only_v2.pt"):
    if os.path.exists(cache_path):
        print(f"글로벌 스케일 캐시 로드: {cache_path}")
        s = torch.load(cache_path, map_location='cpu', weights_only=False)
        return s['node_scale'], s['edge_scale']

    # fast 포맷: 사전 병합된 x/edge_attr 전체로 정확한 std 계산 (샘플링 불필요)
    print("글로벌 스케일 계산 (fast 포맷, 전체 행 기준)...")
    node_scale, edge_scale = compute_global_scales_fast(file_paths)

    tmp = cache_path + ".tmp"
    torch.save({'node_scale': node_scale, 'edge_scale': edge_scale}, tmp)
    os.replace(tmp, cache_path)
    print(f"글로벌 스케일 저장: {cache_path}")
    return node_scale, edge_scale


# ══════════════════════════════════════════════════════════════════
# [Model — layout_graphencoder.py 와 완전 동일]
# ══════════════════════════════════════════════════════════════════
class EdgeUpdatingGATLayer(nn.Module):
    def __init__(self, in_node_dim, in_edge_dim, out_node_dim, out_edge_dim, is_final=False, heads=4):
        super().__init__()
        self.is_final = is_final
        self.gat           = GATv2Conv(in_node_dim, out_node_dim, heads=heads, concat=True, edge_dim=in_edge_dim)
        self.node_head_proj = nn.Linear(out_node_dim * heads, out_node_dim) if heads > 1 else nn.Identity()
        self.edge_mlp      = nn.Sequential(
            nn.Linear(out_node_dim * 2 + in_edge_dim, out_edge_dim), nn.GELU(),
            nn.Linear(out_edge_dim, out_edge_dim))
        self.node_norm = nn.LayerNorm(out_node_dim)
        self.edge_norm = nn.LayerNorm(out_edge_dim)
        self.node_proj = nn.Linear(in_node_dim, out_node_dim) if in_node_dim != out_node_dim else nn.Identity()
        self.edge_proj = nn.Linear(in_edge_dim, out_edge_dim)  if in_edge_dim != out_edge_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        x_new = self.node_head_proj(self.gat(x, edge_index, edge_attr))
        x_new = self.node_norm(x_new + self.node_proj(x))
        row, col = edge_index[0], edge_index[1]
        ef = F.gelu(x_new)
        edge_attr_new = self.edge_norm(
            self.edge_mlp(torch.cat([ef[row], ef[col], edge_attr], dim=-1)) + self.edge_proj(edge_attr))
        if not self.is_final:
            x_new = F.gelu(x_new);  edge_attr_new = F.gelu(edge_attr_new)
        return x_new, edge_attr_new


class LayoutGATEncoder(nn.Module):
    def __init__(self, num_node_features=4, num_edge_features=6, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # hidden_dim 은 GPU 8배수 정렬(텐서 코어)로 올림 — embedding_dim 자체가 홀수(예: 9)라도
        # 가장 비싼 attention 레이어(layer1, heads=4)는 정렬 깨진 GEMM/커널로 떨어지지 않게 한다.
        # dim=8→16, dim=16→32 는 원래도 8배수라 이 올림으로 값이 안 바뀐다(하위 호환).
        # dim=9→18 처럼 어긋나는 값만 24 로 올라간다 — v3_conv 에서 dim=9 로 바꾼 뒤 compute
        # time 이 dim=16(60s) 보다 오히려 dim=9(140s) 에서 더 걸린 원인으로 의심되는 지점.
        hidden_dim  = max(16, -(-embedding_dim * 2 // 8) * 8)   # ceil to multiple of 8
        self.layer1 = EdgeUpdatingGATLayer(num_node_features, num_edge_features, hidden_dim, hidden_dim, heads=4)
        self.layer2 = EdgeUpdatingGATLayer(hidden_dim, hidden_dim, embedding_dim, embedding_dim, is_final=True, heads=1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, edge_attr = self.layer1(x, edge_index, edge_attr)
        return self.layer2(x, edge_index, edge_attr)


class NodeAttributeDecoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_node_features=4):
        super().__init__()
        h = max(16, (embedding_dim + num_node_features) // 2)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, h), nn.GELU(), nn.Linear(h, num_node_features))
    def forward(self, z): return self.mlp(z)


class EdgeAttributeDecoder(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM, num_edge_features=6):
        super().__init__()
        h = max(16, (embedding_dim + num_edge_features) // 2)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, h), nn.GELU(), nn.Linear(h, num_edge_features))
    def forward(self, z): return self.mlp(z)


class LayoutGAE(nn.Module):
    def __init__(self, num_node_features=4, num_edge_features=4, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.encoder      = LayoutGATEncoder(num_node_features, num_edge_features, embedding_dim)
        self.node_decoder = NodeAttributeDecoder(embedding_dim, num_node_features)
        self.edge_decoder = EdgeAttributeDecoder(embedding_dim, num_edge_features)
        self.center_proj  = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.GELU())

    def forward(self, data, decode: bool = True):
        z_node, z_edge = self.encoder(data)
        batch, num_graphs = data.batch, data.num_graphs

        # decode=False (VICReg clean-pass 전용): recon_* 는 compute_losses 에서
        # 쓰이지 않는데도 항상 계산되고 있었다 — node/edge decoder MLP 만큼 헛일이라 스킵.
        recon_node_attr = self.node_decoder(z_node) if decode else None
        recon_edge_attr = self.edge_decoder(z_edge) if decode else None

        max_emb   = global_max_pool(z_node, batch)
        mean_emb  = global_mean_pool(z_node, batch)

        cg_idx    = batch[data.hop0_mask]
        cc        = torch.zeros(num_graphs, dtype=torch.long, device=cg_idx.device)
        cc.scatter_add_(0, cg_idx, torch.ones_like(cg_idx))
        valid     = (cc == 1)
        final_h0  = data.hop0_mask & valid[batch]
        center_z  = z_node[final_h0]
        center_emb = scatter(center_z, batch[final_h0], dim=0, dim_size=num_graphs, reduce='mean')
        center_emb = self.center_proj(center_emb)
        center_emb = torch.where(valid.unsqueeze(-1), center_emb, mean_emb)

        eb         = batch[data.edge_index[0]]
        emean      = scatter(z_edge, eb, dim=0, dim_size=num_graphs, reduce='mean')
        emax       = scatter(z_edge, eb, dim=0, dim_size=num_graphs, reduce='max')

        graph_emb  = torch.cat([max_emb, mean_emb, center_emb, emean, emax], dim=1)
        return recon_edge_attr, recon_node_attr, graph_emb, valid[batch]


# ══════════════════════════════════════════════════════════════════
# [Loss Functions]
# ══════════════════════════════════════════════════════════════════
def safe_mask_loss(pred, target, mask, feature_scale):
    mp = pred[mask]
    if mp.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    loss = F.mse_loss(mp / (feature_scale.view(1, -1) + 1e-8),
                      target[mask] / (feature_scale.view(1, -1) + 1e-8))
    return torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)


def edge_loss(recon, target, esc_spatial, esc_extra):
    if recon.size(0) == 0:
        z = torch.tensor(0.0, device=recon.device)
        return z, z, z
    ls = F.mse_loss(recon[:, :2] / (esc_spatial + 1e-8), target[:, :2] / (esc_spatial + 1e-8))
    lv = (1 - F.cosine_similarity(recon[:, 2:4], target[:, 2:4], dim=1)).mean()
    le = F.mse_loss(recon[:, 4:] / (esc_extra.view(1, -1) + 1e-8),
                    target[:, 4:] / (esc_extra.view(1, -1) + 1e-8)) if recon.size(1) > 4 \
         else torch.tensor(0.0, device=recon.device)
    return (torch.nan_to_num(ls, 0.), torch.nan_to_num(lv, 0.), torch.nan_to_num(le, 0.))


def vicreg_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> tuple:
    """
    VICReg 3항 분해 반환:  (inv_loss, var_loss, cov_loss)

    z_a : (B, D) — noisy 경로 임베딩
    z_b : (B, D) — clean  경로 임베딩 (stop-gradient 없음 — 양방향 학습)

    Invariance : MSE(z_a, z_b)   → 같은 패턴의 clean/noisy를 가깝게
    Variance   : relu(1 - std(z)) → 각 차원이 0으로 collapse 되지 않도록
    Covariance : off-diag(Cov)^2  → 차원 간 정보 중복 제거
    """
    B, D = z_a.shape

    # Invariance
    inv = F.mse_loss(z_a, z_b)

    # Variance — 두 뷰 모두에 적용
    def _var(z):
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return F.relu(1.0 - std).mean()

    var = (_var(z_a) + _var(z_b)) / 2

    # Covariance — 두 뷰 모두에 적용
    def _cov(z):
        zc  = z - z.mean(dim=0)
        cov = (zc.T @ zc) / (B - 1)
        off = cov.pow(2)
        off.fill_diagonal_(0.0)
        return off.sum() / D

    cov = (_cov(z_a) + _cov(z_b)) / 2

    return (torch.nan_to_num(inv, 0.), torch.nan_to_num(var, 0.), torch.nan_to_num(cov, 0.))


def compute_losses(
    noisy_batch, clean_batch, model,
    node_scale, edge_scale_spatial, edge_scale_extra,
    weight_hop,
    clean_x=None, clean_edge_attr=None
):
    """
    noisy_batch : 노이즈 주입된 배치 (인코더 입력)
    clean_batch : 원본 배치 (VICReg Invariance용 — 구조 동일, feature만 clean)
    """
    # ── noisy forward ───────────────────────────────────────────
    recon_ea, recon_na, z_noisy, valid_mask = model(noisy_batch)

    # ── clean forward (VICReg Invariance) ───────────────────────
    # z_clean(graph_emb)만 쓰고 recon 출력은 버려지므로 decode=False 로 계산 생략.
    _, _, z_clean, _ = model(clean_batch, decode=False)

    # ── Reconstruction losses ───────────────────────────────────
    tx = clean_x if clean_x is not None else noisy_batch.x
    te = clean_edge_attr if clean_edge_attr is not None else noisy_batch.edge_attr

    lh0 = safe_mask_loss(recon_na, tx, noisy_batch.hop0_mask & valid_mask, node_scale)
    lh1 = safe_mask_loss(recon_na, tx, noisy_batch.hop1_mask & valid_mask, node_scale)
    lh2 = safe_mask_loss(recon_na, tx, noisy_batch.hop2_mask & valid_mask, node_scale)
    lh3 = safe_mask_loss(recon_na, tx, noisy_batch.hop3_mask & valid_mask, node_scale)
    ls, lv, le = edge_loss(recon_ea, te, edge_scale_spatial, edge_scale_extra)

    recon_total = (
        weight_hop[0]*lh0 + weight_hop[1]*lh1 + weight_hop[2]*lh2 + weight_hop[3]*lh3 +
        weight_hop[4]*ls  + weight_hop[5]*lv  + weight_hop[6]*le
    )

    # ── VICReg ──────────────────────────────────────────────────
    vic_inv, vic_var, vic_cov = vicreg_loss(z_noisy, z_clean)
    vic_total = (VICREG_WEIGHT_INV * vic_inv +
                 VICREG_WEIGHT_VAR * vic_var +
                 VICREG_WEIGHT_COV * vic_cov)

    total = recon_total + vic_total

    return {
        'total_loss':        total,
        'recon_loss':        recon_total,
        'loss_h0': lh0, 'loss_h1': lh1, 'loss_h2': lh2, 'loss_h3': lh3,
        'loss_e_spatial_base': ls, 'loss_e_vec': lv, 'loss_e_extra': le,
        'vic_inv': vic_inv, 'vic_var': vic_var, 'vic_cov': vic_cov,
        'recon_edge_attr': recon_ea,
        'graph_embedding': z_noisy,     # 임베딩 추출 시 noisy 경로 사용
    }


# ══════════════════════════════════════════════════════════════════
# [Train / Val Loop]
# ══════════════════════════════════════════════════════════════════
def _make_clean_batch(batch):
    """
    기존 batch 와 동일한 구조이되 x/edge_attr 가 이미 clean 인 복사본 반환.
    (train loop 에서 clean_x, clean_edge 를 noisy 주입 이전에 clone 해 둠)
    """
    clean = batch.clone()
    return clean


def train_one_epoch(
    model, train_loader,
    node_scale, edge_scale, edge_scale_spatial, edge_scale_extra,
    weight_hop, optimizer, device, scaler
):
    model.train()
    # ── [v3_conv 성능 수정] 로깅용 누적을 GPU 텐서로 유지, epoch 끝에 .item() 1회만 호출.
    #   기존엔 배치마다 loss/recon/vi/vv/vc(5) + grad-norm(2) 총 ~7번 .item() 을 불렀는데,
    #   .item() 은 그 시점까지 큐잉된 GPU 커널이 전부 끝날 때까지 CPU 를 블로킹한다.
    #   이 모델은 그래프가 작아 커널이 매우 많고 짧아서(launch-overhead-bound), 배치당
    #   sync 7회가 t_step 을 실제 연산량보다 크게 부풀리는 주범이었다(data 3s/compute 59s
    #   케이스). NaN/Inf loss 를 걸러 backward 를 스킵하는 체크(아래 isfinite(loss))만
    #   배치당 필수 동기화로 남기고 나머지는 epoch 끝 1회로 미룬다.
    #   부작용: 이제 배치 간 GPU 작업이 파이프라이닝되므로, 아래 t_data/t_step 로그는
    #   "정확한 GPU 전용 시간" 이 아니라 근사치가 된다 — 실측 속도 향상은 epoch 전체
    #   walltime(메인 루프의 Epoch 로그 줄)로 확인할 것.
    total_loss_t  = torch.zeros((), device=device)
    total_recon_t = torch.zeros((), device=device)
    total_vi_t    = torch.zeros((), device=device)
    total_vv_t    = torch.zeros((), device=device)
    total_vc_t    = torch.zeros((), device=device)
    total_grad_t  = torch.zeros((), device=device)
    n = 0

    # ── 병목 계측: nvidia-smi 를 못 볼 때 데이터 대기 vs 연산 시간을 로그로 판별 ──
    #   t_data 가 지배적  → DataLoader(collation) 병목 → NUM_WORKERS↑ 로 개선
    #   t_step 이 지배적  → GPU 연산/커널 병목       → torch.compile 등 고려
    t_data = t_step = 0.0
    _end = time.time()

    for batch in train_loader:
        t_data += time.time() - _end
        _t0 = time.time()

        batch = batch.to(device, non_blocking=True)
        batch.edge_index = batch.edge_index.long()
        batch.x          = batch.x.float()
        batch.edge_attr  = batch.edge_attr.float()

        clean_x    = batch.x.clone()
        clean_edge = batch.edge_attr.clone()

        # ── 노이즈 주입 ─────────────────────────────────────────
        batch.x         += torch.randn_like(batch.x)         * NOISE_STD * node_scale.unsqueeze(0)
        batch.edge_attr += torch.randn_like(batch.edge_attr) * NOISE_STD * edge_scale.unsqueeze(0)

        # ── clean 배치 구성 (같은 topology, clean feature) ──────
        clean_batch = _make_clean_batch(batch)
        clean_batch.x         = clean_x
        clean_batch.edge_attr = clean_edge

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda'):
            losses = compute_losses(
                batch, clean_batch, model,
                node_scale, edge_scale_spatial, edge_scale_extra, weight_hop,
                clean_x=clean_x, clean_edge_attr=clean_edge
            )
            loss = losses['total_loss']

        if not torch.isfinite(loss):   # 배치당 필수 동기화 1회 (NaN/Inf 스텝 스킵)
            t_step += time.time() - _t0
            _end = time.time()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)   # GPU 텐서, 동기화 없음
        total_grad_t += torch.where(torch.isfinite(gn), gn, torch.zeros_like(gn))
        scaler.step(optimizer)
        scaler.update()

        total_loss_t  += loss.detach()
        total_recon_t += losses['recon_loss'].detach()
        total_vi_t    += losses['vic_inv'].detach()
        total_vv_t    += losses['vic_var'].detach()
        total_vc_t    += losses['vic_cov'].detach()
        n += 1

        t_step += time.time() - _t0
        _end = time.time()

    if n == 0:
        return {k: 0. for k in ['total','recon','vi','vv','vc','grad','t_data','t_step']}
    return {
        'total': (total_loss_t / n).item(), 'recon': (total_recon_t / n).item(),
        'vi': (total_vi_t / n).item(), 'vv': (total_vv_t / n).item(), 'vc': (total_vc_t / n).item(),
        'grad': (total_grad_t / n).item(),
        't_data': t_data, 't_step': t_step,   # epoch 누적 (초)
    }


def validate_epoch(
    model, val_loader,
    node_scale, edge_scale, edge_scale_spatial, edge_scale_extra,
    weight_hop, device, val_noise_gen
):
    model.eval()
    val_noise_gen.manual_seed(42)

    # [v3_conv 성능 수정] train_one_epoch 와 동일한 이유로 GPU 텐서 누적 + epoch 끝 1회 sync.
    # validate_epoch 은 배치당 원래 .item() 이 train 보다도 더 많았다(loss 5종 + emb_std +
    # rel 6종 + angle_sum = 배치당 최대 13회 sync).
    def _zero():
        return torch.zeros((), device=device)
    # 주의: `a = b = _zero()` 처럼 체이닝하면 같은 텐서 객체를 공유해 += (in-place) 가
    # 서로를 오염시킨다 — 반드시 각자 별도로 만든다.
    total_loss_t       = _zero()
    total_recon_t      = _zero()
    total_clean_loss_t = _zero()
    total_vi_t = _zero(); total_vv_t = _zero(); total_vc_t = _zero()
    total_latent_std_t = _zero()
    rel_t = {k: _zero() for k in ('h0', 'h1', 'h2', 'h3', 'e_space', 'e_extra')}
    angle_sum_t = _zero()
    edge_cnt = n = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device, non_blocking=True)
            batch.edge_index = batch.edge_index.long()
            batch.x          = batch.x.float()
            batch.edge_attr  = batch.edge_attr.float()

            clean_x    = batch.x.clone()
            clean_edge = batch.edge_attr.clone()

            # ── Clean loss (노이즈 없이) ─────────────────────────
            clean_batch = _make_clean_batch(batch)
            with torch.amp.autocast(device_type='cuda'):
                cl = compute_losses(
                    batch, clean_batch, model,
                    node_scale, edge_scale_spatial, edge_scale_extra, weight_hop,
                    clean_x=clean_x, clean_edge_attr=clean_edge
                )
                total_clean_loss_t += cl['total_loss'].detach()

            # ── 노이즈 주입 ─────────────────────────────────────
            nn_  = torch.randn(batch.x.shape, generator=val_noise_gen,
                               dtype=batch.x.dtype, device=device)
            batch.x         = clean_x + nn_ * NOISE_STD * node_scale.unsqueeze(0)
            en_  = torch.randn(batch.edge_attr.shape, generator=val_noise_gen,
                               dtype=batch.edge_attr.dtype, device=device)
            batch.edge_attr = clean_edge + en_ * NOISE_STD * edge_scale.unsqueeze(0)

            clean_batch_n = _make_clean_batch(batch)
            clean_batch_n.x         = clean_x
            clean_batch_n.edge_attr = clean_edge

            with torch.amp.autocast(device_type='cuda'):
                losses = compute_losses(
                    batch, clean_batch_n, model,
                    node_scale, edge_scale_spatial, edge_scale_extra, weight_hop,
                    clean_x=clean_x, clean_edge_attr=clean_edge
                )
                recon_ea = losses['recon_edge_attr'].float()
                emb_std_t = losses['graph_embedding'].std(dim=0, correction=0).mean()

            total_loss_t  += losses['total_loss'].detach()
            total_recon_t += losses['recon_loss'].detach()
            total_vi_t    += losses['vic_inv'].detach()
            total_vv_t    += losses['vic_var'].detach()
            total_vc_t    += losses['vic_cov'].detach()
            total_latent_std_t += emb_std_t.detach()

            rel_t['h0']      += torch.sqrt(losses['loss_h0'].detach()) * 100
            rel_t['h1']      += torch.sqrt(losses['loss_h1'].detach()) * 100
            rel_t['h2']      += torch.sqrt(losses['loss_h2'].detach()) * 100
            rel_t['h3']      += torch.sqrt(losses['loss_h3'].detach()) * 100
            rel_t['e_space'] += torch.sqrt(losses['loss_e_spatial_base'].detach()) * 100
            rel_t['e_extra'] += torch.sqrt(losses['loss_e_extra'].detach()) * 100

            if recon_ea.size(0) > 0:   # .size(0) 은 shape 메타데이터라 동기화 없음
                cs   = F.cosine_similarity(recon_ea[:, 2:4], clean_edge[:, 2:4], dim=1).clamp(-1., 1.)
                angle_sum_t += torch.rad2deg(torch.acos(cs)).sum()
                edge_cnt    += recon_ea.size(0)
            n += 1

    # 여기서 처음으로 .item() 을 부른다 — epoch 전체에서 딱 1번 동기화.
    def _avg(v): return (v / n).item() if n > 0 else 0.
    metrics = {k: _avg(v) for k, v in rel_t.items()}
    metrics.update({
        'e_vec':          (angle_sum_t / edge_cnt).item() if edge_cnt > 0 else 0.,
        'clean_val_loss': _avg(total_clean_loss_t),
        'latent_std':     _avg(total_latent_std_t),
        'vic_inv':        _avg(total_vi_t),
        'vic_var':        _avg(total_vv_t),
        'vic_cov':        _avg(total_vc_t),
        'recon_loss':     _avg(total_recon_t),
    })
    return _avg(total_loss_t), metrics


# ══════════════════════════════════════════════════════════════════
# [Checkpoint — 전체 학습 상태 저장/복원 (Resume 지원)]
# ══════════════════════════════════════════════════════════════════
RESUME_PATH = "checkpoint_last_v3_conv.pt"   # 매 에포크 갱신되는 '이어하기'용 풀 체크포인트


def save_full_checkpoint(path, model, optimizer, scheduler, scaler,
                         epoch, best_val_loss, patience_counter, ckpt_name):
    """모델 전체(encoder+decoder)+옵티마이저+스케줄러+스케일러+진행상태+RNG 원자적 저장."""
    payload = {
        'epoch':            epoch,
        'model':            model.state_dict(),         # 전체 GAE (encoder 단독 아님)
        'optimizer':        optimizer.state_dict(),     # Adam 모멘텀
        'scheduler':        scheduler.state_dict(),     # ReduceLROnPlateau 내부 상태
        'scaler':           scaler.state_dict(),        # AMP GradScaler
        'best_val_loss':    best_val_loss,
        'patience_counter': patience_counter,
        'ckpt_name':        ckpt_name,                  # best 전체모델 파일명 (연속성 유지)
        'torch_rng':        torch.get_rng_state(),
        'cuda_rng':         torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)   # 원자적 교체 — 저장 중 크래시해도 이전 체크포인트 보존


def load_full_checkpoint(path, model, optimizer, scheduler, scaler, device):
    """저장된 풀 체크포인트로 복원. (epoch, best_val_loss, patience_counter, ckpt_name) 반환."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    scaler.load_state_dict(ckpt['scaler'])
    # RNG 복원 — set_rng_state는 CPU ByteTensor 요구
    try:
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        if ckpt.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
    except Exception as e:
        print(f"  [Resume] RNG 복원 건너뜀 (무해): {e}")
    return ckpt['epoch'], ckpt['best_val_loss'], ckpt['patience_counter'], ckpt['ckpt_name']


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    all_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "PREPROCESSED_*part*.pt")))
    if not all_files:
        raise FileNotFoundError(f"청크 파일 없음: {FOLDER_PATH}")
    print(f"총 {len(all_files)}개 청크 파일 발견")

    # VAL_SPLIT_SEED — v2 도 Train/Val 이 0.15/0.4 로 벌어져 있어(v3_conv 의 0.24/0.62 와
    # 같은 비율대) capacity(dim) 문제가 아니라 val 로 뽑힌 청크 1개 자체가 유난히 어렵거나
    # 편향된 분포일 가능성이 있다. seed 를 바꿔 다른 청크가 val 로 뽑히게 해서 갭이
    # seed 에 따라 흔들리는지(=val 청크 편향) 아니면 그대로인지(=진짜 학습/일반화 문제)
    # 갈라본다. (근본적으로는 preprocessor.py 의 merge_and_shuffle_gauges 로 청크 자체를
    # 이미 섞도록 고쳐뒀으니, 그 버전으로 재전처리한 데이터를 쓰면 어느 청크가 val 이 되든
    # 편향이 줄어든다 — 지금 청크가 그 이전에 만들어졌다면 이 seed 변경은 임시 진단용.)
    VAL_SPLIT_SEED = 7   # v2/기존 v3_conv 런과 다른 값(기존 42)
    random.seed(VAL_SPLIT_SEED);  random.shuffle(all_files)
    val_file    = all_files[0]
    train_files = all_files[1:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── [Speedup — 정확도 trade-off 없는 순수 처리량 향상] ──────────
    # TF32: Ampere(RTX 30xx/40xx, A100 등)+ 에서 fp32 matmul 을 TF32 로 가속.
    #        학습 정확도 영향은 사실상 없음(이미 AMP 사용 중). 구형 GPU 면 자동 무시(no-op).
    # cudnn.benchmark: 입력 shape 자동 튜닝(그래프라 효과 작지만 무해).
    pin = (device.type == 'cuda')
    if pin:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    node_scale, edge_scale = get_global_scales(train_files, cache_path="global_scales_train_only_v2.pt")
    node_scale  = node_scale.to(device)
    edge_scale  = edge_scale.to(device)
    edge_scale_spatial = edge_scale[:2]
    edge_scale_extra   = edge_scale[4:] if edge_scale.size(0) > 4 else torch.empty(0, device=device)

    val_noise_gen = torch.Generator(device=device).manual_seed(42)

    # DataLoader 성능 옵션:
    #   - NUM_WORKERS>0: collation 을 CPU 코어로 분산 → GPU starvation 완화(가장 큰 이득).
    #   - persistent_workers: 에포크마다 워커 재생성 비용 제거.
    #   - prefetch_factor: 워커가 미리 배치를 쌓아 GPU 를 굶기지 않음.
    #   - pin_memory + non_blocking: H2D 전송을 연산과 겹침.
    dl_kw = dict(pin_memory=pin, num_workers=NUM_WORKERS)
    if NUM_WORKERS > 0:
        dl_kw.update(persistent_workers=True, prefetch_factor=2)
    print(f"[DataLoader] num_workers={NUM_WORKERS}, pin_memory={pin}")

    print("\n[1/2] 검증 데이터셋 로드 (fast)...")
    val_dataset = FastPreloadedDataset([val_file])
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, **dl_kw)

    print("\n[2/2] 훈련 데이터셋 프리로드 (fast)...")
    train_dataset = FastPreloadedDataset(train_files)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **dl_kw)

    model = LayoutGAE(
        num_node_features=node_scale.size(0),
        num_edge_features=edge_scale.size(0),
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    if USE_TORCH_COMPILE:
        # 주의: compile 된 모델의 state_dict() 는 torch 버전에 따라 키에 "_orig_mod." 접두어가
        # 붙을 수 있다 — visualize_encoder.py/extract_representatives.py 처럼 compile 없이
        # 순수 LayoutGAE 에 load_state_dict 할 때 키 불일치가 날 수 있으니, best 체크포인트
        # 로딩이 실패하면 이 접두어부터 확인할 것.
        print("[train_ver3_conv] torch.compile 활성화 (dynamic=True — 배치별 그래프 크기 변동 대응)")
        model = torch.compile(model, dynamic=True)

    # LR sqrt-scaling: 큰 배치의 낮은 그래디언트 분산을 보상해 에포크당 진척을 유지.
    # batch=2048 → 1e-4 * sqrt(2048/32) = 8e-4. (linear scaling 은 Adam 에 과함)
    # warmup 없이도 grad clip(max_norm=5) 이 초반 발산을 막는 안전망 역할.
    base_lr = 1e-4
    lr = base_lr * (BATCH_SIZE / REFERENCE_BATCH) ** 0.5
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"[train_ver3_conv] batch={BATCH_SIZE}  lr={lr:.2e} (sqrt-scaled from {base_lr:.0e}@{REFERENCE_BATCH})")
    # weight_hop = [h0, h1, h2, h3, edge_spatial, edge_vec, edge_extra]
    # 재구성 loss 의 상대 비중. VICReg 대비 recon 지배도를 낮추려 하향 조정.
    weight_hop  = torch.tensor([3.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0], device=device)
    scaler      = torch.amp.GradScaler('cuda')
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss     = float('inf')
    patience_counter  = 0
    # 30 -> 40. 이전 v3_conv 런이 62 epoch(=best~32)에서 patience 30 으로 조기종료됐는데,
    # LR 이 이미 2번(8e-4->2e-4) 줄어든 뒤였다 — 더 볼 여지가 있는지 10 epoch 더 준다.
    # (무한정 늘리진 않음: "40까지만" — LR 스케줄러 patience=10 대비 4배 여유면 충분히
    # plateau 판단 가능하다고 보고 상한을 둠.)
    patience          = 40
    epochs            = 1000
    ckpt_name         = f"best_gae_v3_conv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    _save_thread      = None
    start_epoch       = 0

    # ── [Resume] 이전 체크포인트가 있으면 자동 복원 ──────────────
    #   아키텍처(embedding_dim 등)를 바꾸면 이전 체크포인트와 shape 가 안 맞아
    #   load_state_dict 가 RuntimeError 를 낸다. 이때 크래시 대신 '신규 학습'으로
    #   안전하게 전환한다. (load_state_dict 는 strict 실패 시 아무것도 로드하지 않으므로
    #   model 은 방금 생성한 초기 상태 그대로 → optimizer/scheduler 도 초기 상태로 일관)
    if os.path.exists(RESUME_PATH):
        print(f"\n[Resume] '{RESUME_PATH}' 발견 → 학습 상태 복원 시도...")
        try:
            last_epoch, best_val_loss, patience_counter, ckpt_name = load_full_checkpoint(
                RESUME_PATH, model, optimizer, scheduler, scaler, device
            )
            start_epoch = last_epoch + 1
            print(f"[Resume] Epoch {start_epoch}부터 재개 | Best Val: {best_val_loss:.6f} | Patience: {patience_counter}")
        except Exception as e:
            print(f"[Resume] ⚠ 복원 실패(아키텍처/포맷 불일치 가능): {e}")
            print(f"[Resume] → 신규 학습으로 시작합니다. '{RESUME_PATH}' 는 다음 에포크부터 새 상태로 덮어써집니다.")
            start_epoch = 0
    else:
        print(f"\n[train_ver3_conv] 신규 학습 시작 (RESUME_PATH 없음)")

    print(f"[train_ver3_conv] embedding_dim={EMBEDDING_DIM} | "
          f"VICReg (Inv={VICREG_WEIGHT_INV}, Var={VICREG_WEIGHT_VAR}, Cov={VICREG_WEIGHT_COV})")

    for epoch in range(start_epoch, epochs):
        _ep_t0 = time.time()
        tr = train_one_epoch(
            model, train_loader,
            node_scale, edge_scale, edge_scale_spatial, edge_scale_extra,
            weight_hop, optimizer, device, scaler
        )
        avg_val_loss, vm = validate_epoch(
            model, val_loader,
            node_scale, edge_scale, edge_scale_spatial, edge_scale_extra,
            weight_hop, device, val_noise_gen
        )
        scheduler.step(avg_val_loss)

        # ── epoch 마다 성능/병목 한 줄 (epoch 이 길어 즉각 피드백 필요) ──
        _ep_dt = time.time() - _ep_t0
        _td, _ts = tr.get('t_data', 0.), tr.get('t_step', 0.)
        _ratio = _td / (_td + _ts + 1e-9)
        print(
            f"Epoch {epoch+1:04d} | Val {avg_val_loss:.4f} | "
            f"epoch {_ep_dt:.0f}s (train data {_td:.0f}s / compute {_ts:.0f}s, "
            f"data {_ratio:.0%}) "
            f"{'← DataLoader 병목: NUM_WORKERS↑' if _ratio > 0.5 else '← 연산 병목: compile 고려'}"
        )

        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:04d} | "
                f"Train Total: {tr['total']:.4f} (Recon: {tr['recon']:.4f}) | "
                f"Val: {avg_val_loss:.4f} (Clean: {vm['clean_val_loss']:.4f}) | LR: {lr:.6f}"
            )
            print(
                f"  [Health]  GradNorm: {tr['grad']:.3f} | "
                f"LatentStd: {vm['latent_std']:.4f}"
            )
            print(
                f"  [VICReg]  Inv: {vm['vic_inv']:.4f} | "
                f"Var: {vm['vic_var']:.4f} | "
                f"Cov: {vm['vic_cov']:.4f}  "
                f"(목표: Inv↓  Var→0  Cov→0)"
            )
            print(
                f"  [Recon]   h0={vm['h0']:.1f}% h1={vm['h1']:.1f}% h2={vm['h2']:.1f}% h3={vm['h3']:.1f}% | "
                f"E_Space={vm['e_space']:.1f}% | E_Vec={vm['e_vec']:.2f}°"
                # h2/h3 를 새로 노출: encoder 가 GAT 2층(=2-hop receptive field)이라
                # hop3+ 노드엔 구조적으로 center 정보가 도달하지 못한다. 레이어를 늘릴지
                # 판단하려면 h0/h1 대비 h2/h3 오차가 뚜렷이 큰지부터 데이터로 봐야 한다.
            )

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            if _save_thread is not None:
                _save_thread.join()
            # v2 는 encoder 단독 저장 → center_proj 없어 다운스트림이 근사값만 얻었음.
            # v3_conv 는 best 시점의 전체 GAE(encoder+decoder+center_proj) 를 그대로 저장해
            # visualize_encoder.py/extract_representatives.py 가 정확한 graph_emb 를
            # "최종 epoch"가 아니라 "최적 epoch" 기준으로 재현할 수 있게 한다.
            cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            _save_thread = threading.Thread(target=torch.save, args=(cpu_sd, ckpt_name), daemon=False)
            _save_thread.start()
        else:
            patience_counter += 1

        # ── [Resume] 매 에포크 풀 체크포인트 갱신 (크래시 대비) ──
        save_full_checkpoint(
            RESUME_PATH, model, optimizer, scheduler, scaler,
            epoch, best_val_loss, patience_counter, ckpt_name
        )

        if patience_counter >= patience:
            print(f"\n[Early Stopping] Epoch {epoch+1}. Best Val: {best_val_loss:.6f}")
            break

    if _save_thread is not None:
        _save_thread.join()
    del train_loader;  gc.collect()


if __name__ == "__main__":
    main()
