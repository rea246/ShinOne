import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==================== 설정 ====================
CSV_PATH = "data.csv"      # 읽을 csv 파일 경로
DELIMITER = ","             # 구분자
HAS_HEADER = True           # 첫 번째 행이 컬럼명이면 True

X_LIM = None                # 예: (0, 100) / None 이면 자동
Y_LIM = None                # 예: (0, 100) / None 이면 자동

MARKER_SIZE = 60
ALPHA = 0.75
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

    df = pd.read_csv(CSV_PATH, sep=DELIMITER, header=0 if HAS_HEADER else None)

    x_col = df.columns[0]
    y_cols = list(df.columns[1:])

    if len(y_cols) > len(PALETTE):
        raise ValueError(
            f"컬럼이 {len(y_cols)}개라 팔레트({len(PALETTE)}개)로는 색을 뚜렷하게 구분할 수 없습니다. "
            "PALETTE를 늘리거나 컬럼 수를 줄이세요."
        )

    long_df = df.melt(id_vars=x_col, value_vars=y_cols, var_name="series", value_name="value")

    fig, ax = plt.subplots(figsize=(9, 6))

    sns.scatterplot(
        data=long_df,
        x=x_col,
        y="value",
        hue="series",
        hue_order=y_cols,
        palette=PALETTE[: len(y_cols)],
        s=MARKER_SIZE,
        alpha=ALPHA,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    ax.set_xlabel(str(x_col))
    ax.set_ylabel("value")
    ax.legend(title=None, frameon=False, loc="best")
    sns.despine(fig=fig, ax=ax)

    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
