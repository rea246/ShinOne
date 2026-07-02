"""
plot_preprocessor_check.py — preprocessor.py 산출물(PREPROCESSED_*.pt) 검증 시각화

preprocessor 의 fast 포맷(_pack_fast_chunk)을 그대로 읽어 게이지 1개(그래프 1개)를
골라 다음 3가지를 한 장의 그림으로 확인한다:

  1) Hop 위치     : 각 노드(폴리곤 bbox)를 hop0/hop1/hop2/hop3+ 색으로 칠해
                    중심(hop0)에서 hop 단계가 바깥으로 퍼지는지 눈으로 확인
  2) 엣지 의도성   : 연결된 노드 중심끼리 선으로 이어, 4방향(E/N/W/S) 최근접
                    이웃만 연결됐는지(의도대로인지) 확인
  3) 엣지 속성     : 저장된 edge_attr = [box_dist, center_dist, unit_x, unit_y] 를
                    노드 좌표/크기로부터 재계산해 일치하는지 수치 검증 +
                    엣지 위에 unit vector 화살표/거리 라벨 표시

fast 포맷 텐서 (preprocessor._pack_fast_chunk 참조):
  x          : (N,4) [cx, cy, w, h]  (nm, 게이지 중심 기준 상대좌표)
  edge_index : (2,E) int  (그래프-로컬 인덱스, 양방향)
  edge_attr  : (E,4) [box_dist(nm), center_dist(nm), unit_x, unit_y]
  hop_packed : (N,)  int8  (bit0=hop0, bit1=hop1, bit2=hop2, bit3=hop3+)
  meta       : (G,4) [x0, x1, e0, e1]   그래프별 슬라이스
  keys       : [gauge_name, ...]

사용 예:
  python plot_preprocessor_check.py --file T2T/PREPROCESSED_T2T.oas_part0.pt --index 0
  python plot_preprocessor_check.py --dir T2T --key MY_GAUGE_NAME --num 5
"""

import os
import glob
import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")          # 화면 없는 환경(서버)에서도 PNG 저장 가능
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# hop 단계별 색/라벨
HOP_COLORS = {0: "#e6194B", 1: "#f58231", 2: "#3cb44b", 3: "#9aa0a6"}
HOP_LABELS = {0: "hop0 (center)", 1: "hop1", 2: "hop2", 3: "hop3+"}


# ══════════════════════════════════════════════════════════════════
# [Load / Slice]
# ══════════════════════════════════════════════════════════════════
def load_chunk(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def unpack_graph(chunk: dict, idx: int) -> dict:
    """fast 포맷 청크에서 idx번째 그래프만 잘라 numpy dict 로 반환."""
    G = chunk["meta"].size(0)
    if not (0 <= idx < G):
        raise IndexError(f"graph index {idx} 범위 밖 (0..{G-1})")

    x0, x1, e0, e1 = [int(v) for v in chunk["meta"][idx].tolist()]
    x  = chunk["x"][x0:x1].float().numpy()              # (N,4)
    ei = chunk["edge_index"][:, e0:e1].long().numpy()   # (2,E) local
    ea = chunk["edge_attr"][e0:e1].float().numpy()      # (E,4)
    packed = chunk["hop_packed"][x0:x1].numpy().astype(np.int64)  # (N,)

    # 패킹 비트 → 단일 hop 레벨 (마스크는 상호배타적; 안전하게 낮은 hop 우선)
    hop = np.full(len(packed), 3, dtype=int)
    hop[((packed >> 2) & 1).astype(bool)] = 2
    hop[((packed >> 1) & 1).astype(bool)] = 1
    hop[(packed & 1).astype(bool)] = 0

    key = chunk["keys"][idx] if "keys" in chunk and idx < len(chunk["keys"]) else f"graph_{idx}"
    return {"x": x, "edge_index": ei, "edge_attr": ea, "hop": hop, "key": key}


def undirected_edges(ei: np.ndarray) -> np.ndarray:
    """양방향 (2,E) → 중복 제거한 무방향 엣지 (M,2). 각 쌍은 정방향(u,v) 하나만 남김."""
    if ei.size == 0:
        return np.empty((0, 2), dtype=int)
    seen = {}
    for u, v in ei.T:
        key = (int(min(u, v)), int(max(u, v)))
        if key not in seen:
            seen[key] = (int(u), int(v))   # 처음 등장한 방향 보존
    return np.array(list(seen.values()), dtype=int)


# ══════════════════════════════════════════════════════════════════
# [Verification — edge_attr 가 노드 좌표/크기와 일치하는지]
# ══════════════════════════════════════════════════════════════════
def verify_edge_attr(g: dict) -> dict:
    """
    저장된 edge_attr 를 노드 feature(x = [cx,cy,w,h]) 로부터 재계산해 비교.
      center_dist : ||c_v - c_u||                       (정확히 재계산 가능 — 그대로 검증)
      unit_vec    : (c_v - c_u) / center_dist            (정확히 재계산 가능 — 그대로 검증)
      box_dist    : preprocessor.py 가 이제 실제 폴리곤 hull 꼭짓점 간 최소거리를 저장한다
                    (_calculate_polygon_min_distance). 그런데 x 에는 bbox(w,h)만 남아있고
                    원본 폴리곤 꼭짓점은 저장 안 하므로, 여기선 옛 방식대로 "축정렬 사각형 간
                    gap"(hypot(max(0,|dx|-(wu+wv)/2), max(0,|dy|-(hu+hv)/2)))으로만 근사
                    재계산한다. 두 값은 완전 직각(rectilinear) 사각형에서는 일치하지만,
                    폴리곤이 사각형이 아니면 갈라진다 — 즉 아래 반환값의 "box_dist" 오차는
                    이제 버그 신호가 아니라 "이 엣지의 폴리곤이 사각형에서 얼마나 벗어났는지"
                    를 보여주는 참고 지표다. OK/MISMATCH 판정은 center_dist/unit_vec 만으로 낸다.
    반환: 각 항목 최대 절대 오차.
    """
    x, ei, ea = g["x"], g["edge_index"], g["edge_attr"]
    if ei.size == 0:
        return {"n_edges": 0}

    u, v = ei
    c = x[:, :2]
    w = x[:, 2]
    h = x[:, 3]

    diff = c[v] - c[u]
    cdist = np.hypot(diff[:, 0], diff[:, 1])
    norm = np.where(cdist == 0, 1e-6, cdist)
    uvec = diff / norm[:, None]

    dxc = np.abs(c[v, 0] - c[u, 0])
    dyc = np.abs(c[v, 1] - c[u, 1])
    gx = np.maximum(0.0, dxc - (w[u] + w[v]) / 2.0)
    gy = np.maximum(0.0, dyc - (h[u] + h[v]) / 2.0)
    box = np.hypot(gx, gy)

    return {
        "n_edges":     int(ei.shape[1]),
        "box_dist":    float(np.max(np.abs(box - ea[:, 0]))),
        "center_dist": float(np.max(np.abs(cdist - ea[:, 1]))),
        "unit_x":      float(np.max(np.abs(uvec[:, 0] - ea[:, 2]))),
        "unit_y":      float(np.max(np.abs(uvec[:, 1] - ea[:, 3]))),
    }


# ══════════════════════════════════════════════════════════════════
# [Plot]
# ══════════════════════════════════════════════════════════════════
def plot_graph(g: dict, out_path: str, max_attr_labels: int = 30) -> dict:
    x, ei, ea, hop, key = g["x"], g["edge_index"], g["edge_attr"], g["hop"], g["key"]
    cx, cy, w, h = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    uedges = undirected_edges(ei)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 9.2))

    # ── 좌측: hop 위치 + 엣지 의도성 ────────────────────────────
    ax1.set_title(f"[1+2] Hop levels & edges  |  {key}\nnodes={len(x)}  undirected_edges={len(uedges)}")
    # 엣지 (노드 중심끼리)
    for u, v in uedges:
        ax1.plot([cx[u], cx[v]], [cy[u], cy[v]], "-", color="#4040ff", lw=1.0, alpha=0.6, zorder=1)
    # 노드 (bbox 사각형 + 중심점), hop 색
    for i in range(len(x)):
        col = HOP_COLORS[hop[i]]
        ax1.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                                fill=True, facecolor=col, edgecolor="black",
                                lw=0.6, alpha=0.55, zorder=2))
        ax1.plot(cx[i], cy[i], "o", color=col, ms=3, zorder=3)
    # hop0(중심) 별표 강조
    h0 = np.where(hop == 0)[0]
    if len(h0):
        ax1.plot(cx[h0], cy[h0], "*", color="black", ms=18,
                 markerfacecolor=HOP_COLORS[0], markeredgewidth=1.2, zorder=4)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=HOP_COLORS[k],
               markeredgecolor="black", markersize=12, label=HOP_LABELS[k])
        for k in (0, 1, 2, 3)
    ]
    legend_handles.append(Line2D([0], [0], color="#4040ff", lw=1.5, label="edge"))
    ax1.legend(handles=legend_handles, loc="upper right", fontsize=9)
    ax1.set_xlabel("x (nm, gauge-centered)")
    ax1.set_ylabel("y (nm, gauge-centered)")
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.grid(True, ls=":", alpha=0.4)

    # ── 우측: 엣지 속성 (방향 화살표 + 거리 라벨 + 검증) ─────────
    ax2.set_title(f"[3] Edge attributes  |  {key}\nedge_attr = [box_dist, center_dist, unit_x, unit_y]")
    # 노드 윤곽(연하게)
    for i in range(len(x)):
        ax2.add_patch(Rectangle((cx[i] - w[i] / 2, cy[i] - h[i] / 2), w[i], h[i],
                                fill=False, edgecolor="#bbbbbb", lw=0.5, zorder=1))
        ax2.plot(cx[i], cy[i], ".", color="#888888", ms=3, zorder=2)

    # 저장된 unit vector 를 화살표로(중심 u→v 방향), 거리 라벨
    if ei.size > 0:
        # 화살표 길이 스케일: 좌표 범위의 8%
        span = max(cx.max() - cx.min(), cy.max() - cy.min(), 1.0)
        arr_len = span * 0.08
        label_step = max(1, len(uedges) // max_attr_labels)   # 너무 빽빽하면 일부만 라벨

        for k, (u, v) in enumerate(uedges):
            # 무방향 쌍 (u,v)에 해당하는 정방향 edge_attr 찾기
            mask = (ei[0] == u) & (ei[1] == v)
            ai = np.where(mask)[0]
            if len(ai) == 0:
                # 방향이 (v,u)로 저장된 경우
                mask = (ei[0] == v) & (ei[1] == u)
                ai = np.where(mask)[0]
                su, sv = v, u
            else:
                su, sv = u, v
            if len(ai) == 0:
                continue
            box_d, cen_d, ux, uy = ea[ai[0]]

            mx, my = (cx[su] + cx[sv]) / 2, (cy[su] + cy[sv]) / 2
            ax2.arrow(mx, my, ux * arr_len, uy * arr_len,
                      head_width=arr_len * 0.35, head_length=arr_len * 0.4,
                      fc="#d000d0", ec="#d000d0", lw=1.0, length_includes_head=True, zorder=4)
            ax2.plot([cx[su], cx[sv]], [cy[su], cy[sv]], "-", color="#ffb000", lw=0.8, alpha=0.7, zorder=3)
            if k % label_step == 0:
                ax2.text(mx, my, f"b={box_d:.0f}\nc={cen_d:.0f}", fontsize=6.5,
                         ha="center", va="center", color="#202020",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7), zorder=5)

    # 검증 결과 텍스트 박스
    vr = verify_edge_attr(g)
    if vr.get("n_edges", 0) > 0:
        # box_dist 는 이제 폴리곤 최소거리라 x=[cx,cy,w,h] 만으로는 정확 재계산이 안 됨(비사각형
        # 폴리곤에서 어긋나는 게 정상) — OK/MISMATCH 판정은 center_dist/unit_vec 로만 낸다.
        ok = max(vr["center_dist"], vr["unit_x"], vr["unit_y"]) < 1.0
        vtxt = (f"edge_attr re-check (max abs err vs node geom)\n"
                f"  box_dist   : {vr['box_dist']:.4f} nm  (참고용 — 폴리곤이 사각형에서 벗어난 정도)\n"
                f"  center_dist: {vr['center_dist']:.4f} nm\n"
                f"  unit_x     : {vr['unit_x']:.5f}\n"
                f"  unit_y     : {vr['unit_y']:.5f}\n"
                f"  -> {'CONSISTENT (OK)' if ok else 'MISMATCH (check format!)'}\n"
                f"  (small err = float16 storage precision)")
    else:
        vtxt = "edges: 0 (nothing to verify)"
    ax2.text(0.01, 0.99, vtxt, transform=ax2.transAxes, fontsize=8.5, va="top", ha="left",
             family="monospace",
             bbox=dict(boxstyle="round", fc="#fffbe6", ec="#cccccc"))
    arrow_legend = [
        Line2D([0], [0], color="#d000d0", lw=1.5, marker=">", label="unit vector (u→v)"),
        Line2D([0], [0], color="#ffb000", lw=1.5, label="edge (b=box_dist, c=center_dist [nm])"),
    ]
    ax2.legend(handles=arrow_legend, loc="lower right", fontsize=8)
    ax2.set_xlabel("x (nm, gauge-centered)")
    ax2.set_ylabel("y (nm, gauge-centered)")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return vr


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def resolve_file(args) -> str:
    if args.file:
        return args.file
    search_dir = args.dir or "."
    cands = sorted(glob.glob(os.path.join(search_dir, "PREPROCESSED_*part*.pt")))
    if not cands:
        raise FileNotFoundError(f"PREPROCESSED_*part*.pt 없음: {search_dir}")
    print(f"[auto] 첫 청크 사용: {cands[0]}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser(description="preprocessor 산출물 검증 시각화")
    ap.add_argument("--file", type=str, default=None, help="PREPROCESSED_*.pt 경로")
    ap.add_argument("--dir", type=str, default=None, help="청크 디렉터리 (file 미지정 시 자동탐색)")
    ap.add_argument("--index", type=int, default=0, help="그래프 인덱스 (기본 0)")
    ap.add_argument("--key", type=str, default=None, help="게이지 이름으로 선택 (index 무시)")
    ap.add_argument("--num", type=int, default=1, help="index 부터 N개 연속 출력")
    ap.add_argument("--out-dir", type=str, default="preproc_check", help="PNG 저장 폴더")
    args = ap.parse_args()

    path = resolve_file(args)
    chunk = load_chunk(path)
    G = chunk["meta"].size(0)
    print(f"[load] {path}  | graphs={G}  nodes={chunk['x'].size(0)}  edges={chunk['edge_index'].size(1)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 대상 인덱스 결정
    if args.key is not None:
        keys = chunk.get("keys", [])
        if args.key not in keys:
            raise KeyError(f"key '{args.key}' 없음. 예시 keys: {keys[:5]}")
        indices = [keys.index(args.key)]
    else:
        indices = [i for i in range(args.index, min(args.index + args.num, G))]

    for idx in indices:
        g = unpack_graph(chunk, idx)
        safe_key = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(g["key"]))
        out_path = os.path.join(args.out_dir, f"check_{idx:04d}_{safe_key}.png")
        vr = plot_graph(g, out_path)
        if vr.get("n_edges", 0) > 0:
            # box_dist 는 참고용(폴리곤이 사각형에서 벗어난 정도) — 판정엔 안 씀. 주석은
            # verify_edge_attr() docstring 참조.
            status = "OK" if max(vr["center_dist"], vr["unit_x"], vr["unit_y"]) < 1.0 else "MISMATCH"
            print(f"  [{idx}] {g['key']}: nodes={len(g['x'])} edges={vr['n_edges']} "
                  f"| attr err(box={vr['box_dist']:.3f} cen={vr['center_dist']:.3f} "
                  f"ux={vr['unit_x']:.4f} uy={vr['unit_y']:.4f}) → {status} | saved {out_path}")
        else:
            print(f"  [{idx}] {g['key']}: nodes={len(g['x'])} edges=0 | saved {out_path}")


if __name__ == "__main__":
    main()
