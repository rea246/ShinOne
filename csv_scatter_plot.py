import pandas as pd
import matplotlib.pyplot as plt

# ==================== 설정 ====================
CSV_PATH = "data.csv"      # 읽을 csv 파일 경로
DELIMITER = ","             # 구분자
HAS_HEADER = True           # 첫 번째 행이 컬럼명이면 True

X_LIM = None                # 예: (0, 100) / None 이면 자동
Y_LIM = None                # 예: (0, 100) / None 이면 자동

MARKER_SIZE = 20
ALPHA = 0.7
# ================================================


def main():
    df = pd.read_csv(CSV_PATH, sep=DELIMITER, header=0 if HAS_HEADER else None)

    x_col = df.columns[0]
    y_cols = df.columns[1:]

    fig, ax = plt.subplots(figsize=(8, 6))

    for col in y_cols:
        ax.scatter(df[x_col], df[col], s=MARKER_SIZE, alpha=ALPHA, label=str(col))

    ax.set_xlabel(str(x_col))
    ax.set_ylabel("value")
    ax.legend()

    if X_LIM is not None:
        ax.set_xlim(*X_LIM)
    if Y_LIM is not None:
        ax.set_ylim(*Y_LIM)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
