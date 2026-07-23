"""
plot_group_histogram.py — 탭(\\t) 구분 txt 를 읽어 group 별 histogram 비교

지정한 column list 의 각 column 에 대해, group column 값별로 분포를 겹쳐 그려
(seaborn histplot, hue=group) 그룹 간 분포 차이를 한눈에 비교한다.

  - 입력    : \\t 로 구분된 txt/tsv (헤더 있음)
  - group   : 비교 기준이 되는 범주형 column (예: label, class, batch ...)
  - columns : 분포를 볼 수치형 column 들. 각 column 마다 subplot 1개.
  - 출력    : column 들을 격자(grid)로 배치한 PNG 한 장 (column 별 개별 저장도 가능)

사용 예:
  python plot_group_histogram.py --file data.txt --group-col label \\
         --columns width height area
  python plot_group_histogram.py -f data.tsv -g class -c f1 f2 f3 f4 \\
         --bins 40 --stat density --kde --separate --out-dir hist_out
"""

import os
import argparse

import pandas as pd

import matplotlib
matplotlib.use("Agg")          # 화면 없는 환경(서버)에서도 PNG 저장 가능
import matplotlib.pyplot as plt
import seaborn as sns


# ══════════════════════════════════════════════════════════════════
# [Load]
# ══════════════════════════════════════════════════════════════════
def load_tsv(path: str) -> pd.DataFrame:
    """\\t 구분 txt 를 DataFrame 으로. (파이썬 엔진으로 관대하게 파싱)"""
    df = pd.read_csv(path, sep="\t", engine="python")
    # 헤더 앞뒤 공백 제거 (엑셀/수기 편집 대비)
    df.columns = [str(c).strip() for c in df.columns]
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
def _draw_one(ax, df: pd.DataFrame, col: str, group_col: str,
              bins: int, stat: str, kde: bool) -> None:
    """한 축(ax)에 col 의 group 별 겹친 histogram 을 그린다."""
    sns.histplot(
        data=df,
        x=col,
        hue=group_col,
        bins=bins,
        stat=stat,           # count / density / probability / frequency
        kde=kde,
        element="step",      # 여러 그룹이 겹쳐도 윤곽이 보이게
        common_norm=False,   # density/probability 를 그룹별로 각각 정규화 → 공정 비교
        alpha=0.45,
        ax=ax,
    )
    ax.set_title(f"{col}  (by {group_col})")
    ax.set_xlabel(col)
    ax.grid(True, ls=":", alpha=0.4)


def plot_grid(df: pd.DataFrame, group_col: str, columns: list,
              out_path: str, bins: int, stat: str, kde: bool) -> None:
    """column 들을 격자로 배치해 한 장의 PNG 로 저장."""
    n = len(columns)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows),
                             squeeze=False)
    for i, col in enumerate(columns):
        ax = axes[i // ncols][i % ncols]
        _draw_one(ax, df, col, group_col, bins, stat, kde)
    # 남는 빈 축 숨김
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    n_groups = df[group_col].nunique()
    fig.suptitle(f"Group histogram  |  group='{group_col}' ({n_groups} groups)  "
                 f"|  n={len(df)}  stat={stat}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_separate(df: pd.DataFrame, group_col: str, columns: list,
                  out_dir: str, bins: int, stat: str, kde: bool) -> None:
    """column 마다 PNG 를 따로 저장."""
    for col in columns:
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        _draw_one(ax, df, col, group_col, bins, stat, kde)
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
    ap = argparse.ArgumentParser(
        description="\\t 구분 txt 를 읽어 지정한 column 들의 group 별 histogram 비교",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-f", "--file", required=True, help="\\t 구분 txt/tsv 경로")
    ap.add_argument("-g", "--group-col", required=True,
                    help="비교 기준 group column (범주형)")
    ap.add_argument("-c", "--columns", required=True, nargs="+",
                    help="histogram 을 그릴 수치형 column 목록 (공백 구분)")
    ap.add_argument("--bins", type=int, default=30, help="히스토그램 구간 수")
    ap.add_argument("--stat", default="count",
                    choices=["count", "density", "probability", "frequency"],
                    help="y축 통계량 (그룹 크기 다르면 density/probability 추천)")
    ap.add_argument("--kde", action="store_true", help="KDE 곡선 겹쳐 그리기")
    ap.add_argument("--separate", action="store_true",
                    help="column 별로 PNG 를 따로 저장 (기본은 격자 한 장)")
    ap.add_argument("--out-dir", default="group_hist", help="PNG 저장 폴더")
    ap.add_argument("--out", default=None,
                    help="격자 모드일 때 저장 파일명 (기본: <out-dir>/group_hist.png)")
    args = ap.parse_args()

    df = load_tsv(args.file)
    validate_columns(df, args.group_col, args.columns)
    print(f"[load] {args.file}  | rows={len(df)}  cols={len(df.columns)}  "
          f"groups={df[args.group_col].nunique()} {sorted(df[args.group_col].unique().tolist())}")

    sns.set_theme(style="whitegrid")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.separate:
        plot_separate(df, args.group_col, args.columns,
                      args.out_dir, args.bins, args.stat, args.kde)
    else:
        out_path = args.out or os.path.join(args.out_dir, "group_hist.png")
        plot_grid(df, args.group_col, args.columns,
                  out_path, args.bins, args.stat, args.kde)


if __name__ == "__main__":
    main()
