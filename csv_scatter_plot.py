import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==================== 설정 ====================
CSV_PATH = "data.csv"      # 읽을 csv 파일 경로
DELIMITER = ","             # 구분자

X_LIM = None                # 예: (0, 100) / None 이면 자동
Y_LIM = None                # 예: (0, 100) / None 이면 자동

MARKER_SIZE = 60
ALPHA = 0.75

HIGH_VALUE_THRESHOLD = 2.2   # 이 값보다 큰 y값은 회색으로 표기
HIGH_VALUE_COLOR = "#898781"  # 회색
# ================================================

# 카테고리 구분용 고정 팔레트 (최대 8개 컬럼까지 색이 뚜렷하게 구분됨)
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


def main():
    sns.set_theme(style="whitegrid", context="talk")

    df = pd.read_csv(CSV_PATH, sep=DELIMITER, header=0)  # 첫 번째 행 = 컬럼명

    x_col = df.columns[0]
    y_cols = list(df.columns[1:])

    if len(y_cols) > len(PALETTE):
        raise ValueError(
            f"컬럼이 {len(y_cols)}개라 팔레트({len(PALETTE)}개)로는 색을 뚜렷하게 구분할 수 없습니다. "
            "PALETTE를 늘리거나 컬럼 수를 줄이세요."
        )

    fig, ax = plt.subplots(figsize=(9, 6))

    scatter_kwargs = dict(s=MARKER_SIZE, alpha=ALPHA, edgecolor="white", linewidth=0.5)

    for col, color in zip(y_cols, PALETTE):
        is_high = df[col] > HIGH_VALUE_THRESHOLD
        ax.scatter(df.loc[~is_high, x_col], df.loc[~is_high, col], color=color, **scatter_kwargs)
        ax.scatter(df.loc[is_high, x_col], df.loc[is_high, col], color=HIGH_VALUE_COLOR, **scatter_kwargs)

    ax.set_xlabel(str(x_col))
    ax.set_ylabel("value")

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=str(col))
        for col, color in zip(y_cols, PALETTE)
    ]
    legend_handles.append(
        Line2D([0], [0], marker="o", linestyle="", color=HIGH_VALUE_COLOR, label=f"> {HIGH_VALUE_THRESHOLD}")
    )
    ax.legend(handles=legend_handles, title=None, frameon=False, loc="best")
    sns.despine(fig=fig, ax=ax)

    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
