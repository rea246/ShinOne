# -*- coding: utf-8 -*-
"""
coverage_visualize.py
=====================
커버리지를 '수치'가 아니라 '그림'으로 확인하는 시각화 스크립트.

두 종류의 그림을 그린다.

  [Plot 1] PCA 2D 산점도 (겹침/비겹침 확인)
      15차원 → PCA 로 2D 압축 후, Reference 와 Sample 을 seaborn 으로 겹쳐 그린다.
      산점도 + 각 집단의 2D KDE 등고선을 함께 그려, 두 분포가
      '겹치는 영역' 과 'Sample 이 Reference 를 벗어난 영역' 을 눈으로 구분한다.

  [Plot 2] 차원별 Factor 영향력 (15장)
      각 차원 d 를 그대로(raw) x축에 두고, 나머지 14차원을 PCA 로 1D 압축해 y축에 둔다.
      → "차원 d 를 따라 Sample 과 Reference 가 어떻게 분포/이탈하는지"를
        차원마다 한 장씩(총 15장) 저장한다.

대용량 대응
    Reference(50GB)는 전부 그릴 수 없으므로 common.reservoir_sample_reference 로
    균일하게 PLOT_DOWNSAMPLE 개만 추출해 시각화한다.
    스케일링/PCA 는 모두 Sample 기준 스케일러(파이프라인과 동일) 위에서 수행한다.

실행:  python coverage_visualize.py
"""

import os

import matplotlib
matplotlib.use("Agg")                 # 헤드리스(디스플레이 없는) 환경용 백엔드
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from common import (
    DummyDataGenerator, FEATURE_COLS, N_FEATURES,
    fit_scaler, reservoir_sample_reference,
)

# =============================================================================
# CONFIG  ── 실행 파라미터를 여기서 직접 정의한다 (argparse 미사용)
# =============================================================================
SAMPLE_PATH     = "dummy_sample.csv"
REF_PATH        = "dummy_reference.csv"
FMT             = "csv"          # 'csv' 또는 'parquet'
SCALE_METHOD    = "standard"     # PCA 는 표준화(standard)와 궁합이 좋다
CHUNKSIZE       = 200_000
PLOT_DOWNSAMPLE = 20_000         # 시각화용 Reference 표본 수 (너무 크면 산점도가 뭉갬)
OUTPUT_DIR      = "coverage_plots"
POINT_ALPHA     = 0.25           # 산점도 투명도 (겹침 가독성)

GEN_DUMMY       = True
REF_ROWS        = 2_000_000


# =============================================================================
# 데이터 준비: Sample 로드 → Sample 기준 스케일러 → Reference 다운샘플
# =============================================================================
def prepare_data():
    """스케일링된 (sample_scaled, ref_scaled) 를 반환한다."""
    sample_df = pd.read_csv(SAMPLE_PATH)
    scaler = fit_scaler(sample_df, SCALE_METHOD)
    sample_scaled = scaler.transform(sample_df[FEATURE_COLS].values)
    ref_scaled = reservoir_sample_reference(
        REF_PATH, scaler, PLOT_DOWNSAMPLE, CHUNKSIZE, FMT
    )
    return sample_scaled, ref_scaled


def _labeled_frame(ref_2d, sample_2d, xcol, ycol):
    """산점도용 long-form DataFrame (Reference/Sample 라벨 포함) 생성."""
    df_ref = pd.DataFrame(ref_2d, columns=[xcol, ycol]);  df_ref["set"] = "Reference"
    df_sam = pd.DataFrame(sample_2d, columns=[xcol, ycol]); df_sam["set"] = "Sample"
    # Reference 를 아래에 깔고 Sample 을 위에 그리도록 concat 순서 유지
    return pd.concat([df_ref, df_sam], ignore_index=True)


# =============================================================================
# [Plot 1] PCA 2D 산점도 + KDE 등고선 (겹침/비겹침)
# =============================================================================
def plot_pca_2d(sample_scaled, ref_scaled, out_path):
    """
    15D → PCA 2D 로 투영해 Reference 와 Sample 을 겹쳐 그린다.
    PCA 는 '모집단' 인 Reference 표본으로 학습하고, Sample 을 같은 축에 투영한다.
    """
    pca = PCA(n_components=2).fit(ref_scaled)
    ref_2d, sample_2d = pca.transform(ref_scaled), pca.transform(sample_scaled)
    evr = pca.explained_variance_ratio_
    df = _labeled_frame(ref_2d, sample_2d, "PC1", "PC2")

    palette = {"Reference": "#4C72B0", "Sample": "#DD8452"}
    plt.figure(figsize=(9, 8))

    # (1) 산점도: Reference(옅게) 위에 Sample(진하게) 겹쳐 → 겹침 영역이 보임
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="set", palette=palette,
                    s=14, alpha=POINT_ALPHA, edgecolor="none")
    # (2) 각 집단의 2D KDE 등고선 → 분포의 '핵심 영역' 윤곽 비교
    for name, color in palette.items():
        sub = df[df["set"] == name]
        sns.kdeplot(data=sub, x="PC1", y="PC2", levels=5, color=color,
                    linewidths=1.3, alpha=0.9)

    plt.title(f"PCA 2D: Reference vs Sample  "
              f"(PC1 {evr[0]*100:.1f}% + PC2 {evr[1]*100:.1f}% = {evr[:2].sum()*100:.1f}%)")
    plt.legend(title="", loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"[Plot1] 저장: {out_path}  (PC1+PC2 설명분산 {evr[:2].sum()*100:.1f}%)")


# =============================================================================
# [Plot 2] 차원별 Factor 영향력 — d 는 raw, 나머지 14D 는 PCA 1D 압축 (15장)
# =============================================================================
def plot_per_dimension(sample_scaled, ref_scaled, out_dir):
    """
    각 차원 d 마다:
        x축 = 차원 d 원본값(scaled), y축 = 나머지 14차원의 PC1
    Reference vs Sample 산점도를 한 장씩 저장한다.
    """
    os.makedirs(out_dir, exist_ok=True)
    palette = {"Reference": "#4C72B0", "Sample": "#DD8452"}

    for d in range(N_FEATURES):
        others = [j for j in range(N_FEATURES) if j != d]   # d 를 뺀 14개 축

        # 나머지 14D 를 PC1 로 압축 (Reference 로 학습 후 양쪽 투영)
        pca = PCA(n_components=1).fit(ref_scaled[:, others])
        ref_rest = pca.transform(ref_scaled[:, others]).ravel()
        sam_rest = pca.transform(sample_scaled[:, others]).ravel()
        evr = pca.explained_variance_ratio_[0]

        # x=원본 d, y=나머지 PC1 로 long-form 구성
        xcol, ycol = f"f{d:02d} (raw)", "rest-PC1"
        ref_2d = np.column_stack([ref_scaled[:, d], ref_rest])
        sam_2d = np.column_stack([sample_scaled[:, d], sam_rest])
        df = _labeled_frame(ref_2d, sam_2d, xcol, ycol)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=xcol, y=ycol, hue="set", palette=palette,
                        s=14, alpha=POINT_ALPHA, edgecolor="none")
        # x축(차원 d) 방향 분포 차이를 보조로: 상단에 rug/marginal 대신 경량 KDE 등고선
        for name, color in palette.items():
            sub = df[df["set"] == name]
            sns.kdeplot(data=sub, x=xcol, y=ycol, levels=4, color=color,
                        linewidths=1.0, alpha=0.8)

        # 제목은 폰트 호환을 위해 ASCII 로 유지 (한글 폰트 미설치 환경에서도 안 깨짐)
        plt.title(f"Factor f{d:02d}:  raw f{d:02d}  vs  rest-14D PC1 "
                  f"(rest EVR {evr*100:.1f}%)")
        plt.legend(title="", loc="best")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"factor_f{d:02d}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[Plot2] 저장: {out_path}")

    print(f"[Plot2] 차원별 그림 {N_FEATURES}장 저장 완료 → {out_dir}/")


# =============================================================================
# 메인
# =============================================================================
def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sample_scaled, ref_scaled = prepare_data()

    # [Plot 1] PCA 2D 겹침
    plot_pca_2d(sample_scaled, ref_scaled, os.path.join(OUTPUT_DIR, "pca_2d_overlap.png"))

    # [Plot 2] 차원별 Factor 영향력 (15장)
    plot_per_dimension(sample_scaled, ref_scaled, os.path.join(OUTPUT_DIR, "per_dimension"))

    print(f"\n[완료] 모든 그림 저장 위치: {OUTPUT_DIR}/")


if __name__ == "__main__":
    if GEN_DUMMY or not (os.path.exists(SAMPLE_PATH) and os.path.exists(REF_PATH)):
        gen = DummyDataGenerator(seed=42)
        gen.make_sample(SAMPLE_PATH, n_rows=1800)
        gen.make_reference(REF_PATH, total_rows=REF_ROWS, fmt=FMT)
    run()
