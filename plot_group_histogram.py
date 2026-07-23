"""
plot_group_histogram.py — 탭(\\t) 구분 txt 를 읽어 group 별 histogram 비교

지정한 COLUMNS 의 각 column 에 대해, GROUP_COL 값별로 분포를 겹쳐 그려
(seaborn histplot, hue=group) 그룹 간 분포 차이를 한눈에 비교한다.

설정은 아래 [설정] 블록의 값만 고치고 그냥 실행하면 된다:
  python plot_group_histogram.py
"""

import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")          # 화면 없는 환경(서버)에서도 PNG 저장 가능
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
# [설정] — 여기 값만 고치면 됨
# ══════════════════════════════════════════════════════════════════
FILE      = "data.txt"                 # \t 구분 txt/tsv 경로
GROUP_COL = "label"                    # 비교 기준 group column (범주형)
COLUMNS   = ["width", "height", "area"]  # histogram 그릴 수치형 column 목록

BINS      = 30            # 히스토그램 구간 수
STAT      = "count"       # "count" / "density" / "probability" / "frequency"
                         #   (그룹 크기 다르면 "density" 추천)
KDE       = False         # True 면 KDE 곡선도 겹쳐 그림
SEPARATE  = False         # True 면 column 별 PNG 개별 저장, False 면 격자 한 장
OUT_DIR   = "group_hist"  # PNG 저장 폴더
OUT_NAME  = "group_hist.png"  # 격자 모드 저장 파일명


# ══════════════════════════════════════════════════════════════════
# [Load]
# ══════════════════════════════════════════════════════════════════
def load_tsv(path: str) -> pd.DataFrame:
    """\\t 구분 txt 를 DataFrame 으로. (파이썬 엔진으로 관대하게 파싱)"""
    df = pd.read_csv(path, sep="\t", engine="python")
    df.columns = [str(c).strip() for c in df.columns]   # 헤더 앞뒤 공백 제거
    return df


def validate_columns(df: pd.DataFrame, group_col: str, columns: list) -> None:
    """group / value column 이 실제로 있는지, value 가 수치형인지 확인."""
    missing = [c for c in [group_col, *columns] if c not in df.columns]
    if missing:
        raise KeyError(
            f"파일에 없는 column: {missing}\n  사용 가능한 column: {list(df.columns)}"
        )
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            f"histogram 을 그리려면 수치형이어야 함. 수치형이 아닌 column: {non_numeric}"
        )


# ══════════════════════════════════════════════════════════════════
# [Plot]
# ══════════════════════════════════════════════════════════════════
def _draw_one(ax, df: pd.DataFrame, col: str, group_col: str) -> None:
    """한 축(ax)에 col 의 group 별 겹친 histogram 을 그린다."""
    sns.histplot(
        data=df,
        x=col,
        hue=group_col,
        bins=BINS,
        stat=STAT,
        kde=KDE,
        element="step",      # 여러 그룹이 겹쳐도 윤곽이 보이게
        common_norm=False,   # density/probability 를 그룹별로 각각 정규화 → 공정 비교
        alpha=0.45,
        ax=ax,
    )
    ax.set_title(f"{col}  (by {group_col})")
    ax.set_xlabel(col)
    ax.grid(True, ls=":", alpha=0.4)


def plot_grid(df: pd.DataFrame, group_col: str, columns: list, out_path: str) -> None:
    """column 들을 격자로 배치해 한 장의 PNG 로 저장."""
    n = len(columns)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows),
                             squeeze=False)
    for i, col in enumerate(columns):
        _draw_one(axes[i // ncols][i % ncols], df, col, group_col)
    for j in range(n, nrows * ncols):          # 남는 빈 축 숨김
        axes[j // ncols][j % ncols].axis("off")

    n_groups = df[group_col].nunique()
    fig.suptitle(f"Group histogram  |  group='{group_col}' ({n_groups} groups)  "
                 f"|  n={len(df)}  stat={STAT}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_separate(df: pd.DataFrame, group_col: str, columns: list, out_dir: str) -> None:
    """column 마다 PNG 를 따로 저장."""
    for col in columns:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        _draw_one(ax, df, col, group_col)
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in col)
        out_path = os.path.join(out_dir, f"hist_{safe}.png")
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[saved] {out_path}")


# ══════════════════════════════════════════════════════════════════
# [Main]
# ══════════════════════════════════════════════════════════════════
def main():
    df = load_tsv(FILE)
    validate_columns(df, GROUP_COL, COLUMNS)
    print(f"[load] {FILE}  | rows={len(df)}  cols={len(df.columns)}  "
          f"groups={df[GROUP_COL].nunique()} {sorted(df[GROUP_COL].unique().tolist())}")

    sns.set_theme(style="whitegrid")
    os.makedirs(OUT_DIR, exist_ok=True)

    if SEPARATE:
        plot_separate(df, GROUP_COL, COLUMNS, OUT_DIR)
    else:
        plot_grid(df, GROUP_COL, COLUMNS, os.path.join(OUT_DIR, OUT_NAME))


if __name__ == "__main__":
    main()
