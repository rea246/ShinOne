import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# ==================== 설정 ====================
TXT_PATH = "data.txt"       # 읽을 tab 구분 txt 파일 경로

# 파일에 컬럼이 여러 개 있어도 아래 3개만 뽑아서 사용합니다.
GROUP_COL = "group"
X_COL = "X"
Y_COL = "Y"

X_LIM = None                 # 예: (0, 100) / None 이면 자동
Y_LIM = None                 # 예: (0, 100) / None 이면 자동

MARKER_SIZE = 60
ALPHA = 0.75

WIDE_CSV_PATH = "group_wide.csv"   # 그룹별 x,y로 전개해서 저장할 csv 경로
# ================================================

# 그룹 구분용 고정 팔레트 (최대 8개 그룹까지 색이 뚜렷하게 구분됨)
PALETTE = [
    "#2a78d6",  # blue
    "#008300",  # green
    "#e87ba4",  # magenta
    "#eda100",  # yellow
    "#1baf7a",  # aqua
    "#eb6834",  # orange
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def _use_korean_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic", "Noto Sans CJK KR"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def save_wide_csv(df, groups):
    """group, X, Y 형태를 g1_x, g1_y, g2_x, g2_y ... 형태로 전개해서 저장."""
    wide_cols = {}
    for group in groups:
        sub = df[df[GROUP_COL] == group]
        wide_cols[f"{group}_{X_COL}"] = sub[X_COL].reset_index(drop=True)
        wide_cols[f"{group}_{Y_COL}"] = sub[Y_COL].reset_index(drop=True)

    wide_df = pd.DataFrame(wide_cols)
    wide_df.to_csv(WIDE_CSV_PATH, index=False)


def main():
    sns.set_theme(style="whitegrid", context="talk")
    _use_korean_font()

    df = pd.read_csv(TXT_PATH, sep="\t", header=0, usecols=[GROUP_COL, X_COL, Y_COL])

    groups = list(pd.unique(df[GROUP_COL]))

    save_wide_csv(df, groups)

    if len(groups) > len(PALETTE):
        raise ValueError(
            f"그룹이 {len(groups)}개라 팔레트({len(PALETTE)}개)로는 색을 뚜렷하게 구분할 수 없습니다. "
            "PALETTE를 늘리거나 그룹 수를 줄이세요."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    for group, color in zip(groups, PALETTE):
        sub = df[df[GROUP_COL] == group]
        ax.scatter(
            sub[X_COL], sub[Y_COL],
            color=color, s=MARKER_SIZE, alpha=ALPHA,
            edgecolor="white", linewidth=0.5,
        )

    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=str(group))
        for group, color in zip(groups, PALETTE)
    ]
    ax.legend(handles=legend_handles, title=GROUP_COL, frameon=False, loc="best")
    sns.despine(fig=fig, ax=ax)

    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
