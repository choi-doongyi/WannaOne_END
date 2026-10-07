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
df_temp["MEAS_DT"] = pd.to_datetime(df_temp["MEAS_DT"])  # 소결로 온도 1초 간격
df_temp = df_temp.set_index("MEAS_DT").sort_index()
df_welding["MEAS_DT"] = pd.to_datetime(df_welding["MEAS_DT"])  # 상공정 용접 3초 간격
df_welding = df_welding.set_index("MEAS_DT").sort_index()
df_event["EVT_DT"] = pd.to_datetime(
    df_event["EVT_DT"]
)  # 조업 이벤트 간격이 일정하지 않음
df_event = df_event.set_index("EVT_DT").sort_index()
df_repair["DETC_DT"] = pd.to_datetime(
    df_repair["DETC_DT"]
)  # 정비이력 간격이 일정하지 않음
df_repair = df_repair.set_index("DETC_DT").sort_index()  # 정기점검 기간도 포함
##
