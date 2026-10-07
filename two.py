import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import platform
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # 정확도
from sklearn.dummy import DummyClassifier  # 베이스라인 dummy
from sklearn.metrics import confusion_matrix  # 혼동행렬
from sklearn.metrics import ConfusionMatrixDisplay  # 혼동행렬 heatmap
from sklearn.metrics import precision_score  # 정밀도
from sklearn.metrics import recall_score  # 재현율
from sklearn.metrics import f1_score, precision_score, recall_score  # 조화평균
from sklearn.metrics import classification_report  # 한눈에
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.font_manager as fm

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux
    plt.rcParams["font.family"] = "NanumGothic"

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

df_temp = pd.read_csv("data/T-CR1-CAL01_온도.csv")
df_repair = pd.read_csv("data/G-02_정비이력.csv")
df_event = pd.read_csv("data/G-02_조업이벤트.csv")
df_welding = pd.read_csv("data/T-CR1-WLD01_용접.csv")

###  datasets은 사용하는 모든 데이터프레임이 들어있음
datasets = {
    "온도 데이터": df_temp,
    "정비이력 데이터": df_repair,
    "조업이벤트 데이터": df_event,
    "용접 데이터": df_welding,
}

for name, df in datasets.items():
    print("=" * 80)
    print(f"[{name}]")

    print("\n1. 데이터 크기")
    print(f"행(row): {df.shape[0]:,}")
    print(f"열(column): {df.shape[1]:,}")

    print("\n2. 컬럼 목록")
    print(df.columns.tolist())

    print("\n3. 데이터 타입 및 결측치")
    info_df = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "non_null_count": df.notnull().sum(),
            "na_count": df.isna().sum(),
            "na_ratio": (df.isna().mean() * 100).round(2),
        }
    )
    print(info_df)

    print("\n4. 기본 통계량")
    print(df.describe(include="all"))

    print("\n5. 상위 5개 행")
    print(df.head())
    print("=========================================")

    for name, df in datasets.items():
        print("=" * 80)
        print(f"[{name}] info")
        df.info()
    print("=========================================")
## 문자열 MEAS_DT 컬럼을 시간 인덱스로 정렬함
df_temp["MEAS_DT"] = pd.to_datetime(df_temp["MEAS_DT"])  # 소결로 온도
df_temp = df_temp.set_index("MEAS_DT").sort_index()
df_welding["MEAS_DT"] = pd.to_datetime(df_welding["MEAS_DT"])  # 상공정 용접
df_welding = df_welding.set_index("MEAS_DT").sort_index()
df_event["EVT_DT"] = pd.to_datetime(
    df_event["EVT_DT"]
)  # 조업 이벤트 간격이 일정하지 않음
df_event = df_event.set_index("EVT_DT").sort_index()
df_repair["DETC_DT"] = pd.to_datetime(
    df_repair["DETC_DT"]
)  # 정비이력 간격이 일정하지 않음
df_repair = df_repair.set_index("DETC_DT").sort_index()  # 정기점검 기간도 포함


print(df_temp["LOT_NO"].value_counts())
# LOT_NO를 고유값으로 변환
LOT_NO = df_temp["LOT_NO"].unique()


#### 온도 데이터셋을 로트번호로 나눈 그래프

# 가열·균열존 온도 컬럼
heating_temp_cols = ["TC-Z3", "TC-Z4", "TC-Z5", "TC-Z6"]

# 가열·균열존 제어출력 컬럼
heating_op_cols = ["OP-Z3", "OP-Z4", "OP-Z5", "OP-Z6"]

zone_colors = {
    "Z3": "tab:red",
    "Z4": "tab:orange",
    "Z5": "tab:purple",
    "Z6": "tab:brown",
}

zone_linestyles = {
    "Z3": "-",
    "Z4": "--",
    "Z5": ":",
    "Z6": "-.",
}

LOT_NO = sorted(LOT_NO)

lots_per_page = 2  # 로트 2개 x 그래프 2개 = 한 화면 4개 subplot

for start in range(0, len(LOT_NO), lots_per_page):
    lot_batch = LOT_NO[start : start + lots_per_page]

    fig, axes = plt.subplots(
        nrows=lots_per_page,
        ncols=2,
        figsize=(18, 8),
        sharex=False,
        sharey=False,
    )

    axes = np.array(axes).reshape(lots_per_page, 2)

    for row_idx, lot_no in enumerate(lot_batch):
        g = df_temp[df_temp["LOT_NO"] == lot_no]

        ax_temp = axes[row_idx, 0]
        ax_op = axes[row_idx, 1]

        # 왼쪽: 온도 그래프
        for col in heating_temp_cols:
            zone = col.split("-")[-1]

            ax_temp.plot(
                g.index,
                g[col],
                color=zone_colors[zone],
                linestyle=zone_linestyles[zone],
                linewidth=0.9,
                alpha=0.85,
                label=col,
            )

        ax_temp.set_title(f"LOT_NO: {lot_no} | 가열·균열존 온도")
        ax_temp.set_xlabel("Time")
        ax_temp.set_ylabel("Temperature (℃)")
        ax_temp.grid(True, alpha=0.3)
        ax_temp.legend(fontsize=8, ncol=4)

        # 오른쪽: 제어출력 그래프
        for col in heating_op_cols:
            zone = col.split("-")[-1]

            ax_op.plot(
                g.index,
                g[col],
                color=zone_colors[zone],
                linestyle=zone_linestyles[zone],
                linewidth=0.9,
                alpha=0.85,
                label=col,
            )

        ax_op.set_title(f"LOT_NO: {lot_no} | 가열·균열존 제어출력")
        ax_op.set_xlabel("Time")
        ax_op.set_ylabel("Control Output (%)")
        ax_op.set_ylim(0, 100)
        ax_op.grid(True, alpha=0.3)
        ax_op.legend(fontsize=8, ncol=4)

    # 마지막 페이지에서 로트가 1개만 남는 경우 빈 row 제거
    if len(lot_batch) < lots_per_page:
        for row_idx in range(len(lot_batch), lots_per_page):
            axes[row_idx, 0].remove()
            axes[row_idx, 1].remove()

    plt.suptitle(
        f"LOT_NO별 가열·균열존 온도 / 제어출력 추이 "
        f"({start + 1}~{start + len(lot_batch)} / {len(LOT_NO)})",
        fontsize=15,
    )

    plt.tight_layout()
    plt.show()
## 유독 가열존 1존이 2~4존에 비해 온도 변동이 심하고 제어출력도 높다. 공정상 자연스러운 결과일 수 있다
