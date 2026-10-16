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
from collections import Counter
from pathlib import Path

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux
    plt.rcParams["font.family"] = "NanumGothic"


# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


project = Path(r"C:\Users\user\Documents\GitHub\WannaOne_END")

casting_csv = (
    project / "data" / "POSKO-PdM-Data-pdm-v2.2.0-parquet" / "casting_all_scenarios.csv"
)
source_root = (
    project
    / "data"
    / "데이터셋-20261013T011038Z-1-001"
    / "데이터셋"
    / "합성데이터_POSKO"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "POSKO-PdM-Data-pdm-v2.2.0"
)
labels_dir = source_root / "labels"

# EDA는 학습 시나리오만 사용
train_ids = {f"S{i:04d}" for i in range(1, 217)}
numeric_cols = [
    "LOAD",
    "WD-SPD",
    "WD-LOD",
    "OSC-FRQ",
    "OSC-STK",
    "MLD-LVL",
    "CONTROL_DELAY",
]

scenario_counts = Counter()
quality_counts = Counter()
running_counts = Counter()
missing_counts = Counter()
value_sums = Counter()
value_counts = Counter()
sample_parts = []

for chunk in pd.read_csv(casting_csv, chunksize=200_000):
    df = chunk[
        chunk["SCENARIO_ID"].isin(train_ids) & chunk["EQP_CD"].eq("P1-CC1-INS01")
    ]

    scenario_counts.update(df["SCENARIO_ID"].value_counts().to_dict())
    quality_counts.update(df["QUALITY_FLAG"].value_counts().to_dict())
    running_counts.update(df["RUNNING"].value_counts().to_dict())

    for col in numeric_cols:
        values = df[col].dropna()
        missing_counts[col] += df[col].isna().sum()
        value_sums[col] += values.sum()
        value_counts[col] += len(values)

    # 시계열 그래프용으로 한 시나리오만 보관
    example = df[df["SCENARIO_ID"] == "S0002"]
    if not example.empty:
        sample_parts.append(example.copy())

print("학습 시나리오 수:", len(scenario_counts))
print("학습 데이터 행 수:", sum(scenario_counts.values()))
print("운전 상태별 행 수:", running_counts)
print("측정 품질별 행 수:", quality_counts)

sensor_summary = pd.DataFrame(
    {
        col: {
            "평균": value_sums[col] / value_counts[col],
            "결측 행 수": missing_counts[col],
        }
        for col in numeric_cols
    }
).T
print("\n센서 요약")
print(sensor_summary)

# 고장위험 정답도 CSV에서 읽기
labels = pd.concat(
    [pd.read_csv(labels_dir / f"S{i:04d}.csv") for i in range(1, 217)],
    ignore_index=True,
)
labels = labels[labels["EQP_CD"] == "P1-CC1-INS01"]
valid_labels = labels[labels["CENSORED_24H"] == 0]

print("\n검열되지 않은 24시간 고장위험 정답")
print(valid_labels["RISK_24H"].value_counts().sort_index())

# S0002 센서값을 1시간 간격으로 요약해 그래프 표시
sample = pd.concat(sample_parts, ignore_index=True)
sample["MEAS_DT"] = pd.to_datetime(sample["MEAS_DT"], utc=True)
sample = sample[sample["RUNNING"] == 1]

hourly = (
    sample.set_index("MEAS_DT")
    .resample("1h")
    .agg(
        {
            "MLD-LVL": "std",
            "CONTROL_DELAY": "mean",
            "WD-LOD": "mean",
            "OSC-STK": "mean",
        }
    )
)

axes = hourly.plot(
    subplots=True,
    figsize=(12, 9),
    legend=False,
    title=["몰드 레벨 변동", "제어 지연", "인발 부하", "진동 폭"],
)
plt.rcParams["font.family"] = "Malgun Gothic"
for ax in axes:
    ax.set_xlabel("관측 시각 (UTC)")
plt.tight_layout()
plt.show()

print("================================================")

from pathlib import Path
import pandas as pd

data_dir = Path(r"C:\Users\user\Documents\GitHub\WannaOne_END\data")
csv_path = data_dir / "POSKO-PdM-Data-pdm-v2.2.0-parquet" / "casting_all_scenarios.csv"
labels_dir = (
    data_dir
    / "데이터셋-20261013T011038Z-1-001"
    / "데이터셋"
    / "합성데이터_POSKO"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "labels"
)
output_path = csv_path.with_name("casting_with_failer_soon.csv")

labels = pd.concat(
    (
        pd.read_csv(
            path,
            usecols=[
                "SCENARIO_ID",
                "EQP_CD",
                "OBSERVED_AT",
                "RISK_24H",
                "CENSORED_24H",
            ],
        )
        for path in sorted(labels_dir.glob("S*.csv"))
    ),
    ignore_index=True,
)

labels = labels[labels["EQP_CD"] == "P1-CC1-INS01"].copy()
labels["_time"] = pd.to_datetime(labels["OBSERVED_AT"], utc=True)

# 검열된 정답은 0으로 취급하지 않는다.
labels["failer_soon"] = (
    labels["RISK_24H"].where(labels["CENSORED_24H"].eq(0)).astype("Int8")
)
lookup = labels[["SCENARIO_ID", "EQP_CD", "_time", "failer_soon"]]

first_chunk = True
for df in pd.read_csv(csv_path, chunksize=200_000):
    df["_time"] = pd.to_datetime(df["MEAS_DT"], utc=True)

    df = df.merge(
        lookup,
        on=["SCENARIO_ID", "EQP_CD", "_time"],
        how="left",
        validate="many_to_one",
    )
    df = df.drop(columns="_time")

    df.to_csv(
        output_path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
    )
    first_chunk = False

print(f"저장 완료:", df.columns)
print(df["failer_soon"])

# 랜덤포레스트 모델

print("================================================")

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

# 데이터 파일과 시나리오별 학습/검증 구분 정보의 위치
data_dir = Path(r"C:\Users\user\Documents\GitHub\WannaOne_END\data")
csv_path = (
    data_dir / "POSKO-PdM-Data-pdm-v2.2.0-parquet" / "casting_with_failer_soon.csv"
)
manifest_path = (
    data_dir
    / "데이터셋-20261013T011038Z-1-001"
    / "데이터셋"
    / "합성데이터_POSKO"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "manifest.json"
)

# 예측에 사용할 센서값: 정답이나 시나리오 ID는 포함하지 않음
features = [
    "LOAD",
    "WD-SPD",
    "WD-LOD",
    "OSC-FRQ",
    "OSC-STK",
    "MLD-LVL",
    "CONTROL_DELAY",
    "RUNNING",
]

# 큰 CSV를 나눠 읽고, 정답이 있는 6시간 간격의 행만 수집
parts = []
for chunk in pd.read_csv(
    csv_path,
    usecols=["SCENARIO_ID", "failer_soon", *features],
    chunksize=200_000,
):
    labeled = chunk[chunk["failer_soon"].notna()]
    if not labeled.empty:
        parts.append(labeled)

df = pd.concat(parts, ignore_index=True)

# 정답: 0은 24시간 내 고장 없음, 1은 24시간 내 고장
df["failer_soon"] = df["failer_soon"].astype(int)

# 데이터셋이 지정한 시나리오별 학습/검증 구분을 읽음
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
split_by_id = {scenario["id"]: scenario["split"] for scenario in manifest["scenarios"]}

# 같은 시나리오가 학습과 검증 양쪽에 들어가지 않도록 분리
train = df[df["SCENARIO_ID"].map(split_by_id) == "train"]
valid = df[df["SCENARIO_ID"].map(split_by_id) == "validation"]

# X는 센서 입력값, y는 실제 정답
X_train, y_train = train[features], train["failer_soon"]
X_valid, y_valid = valid[features], valid["failer_soon"]

# 결측 센서값은 학습 데이터의 중앙값으로 대체한 뒤 모델 학습
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=300,  # 생성할 결정트리 개수
        min_samples_leaf=3,  # 말단 노드의 최소 샘플 수
        class_weight="balanced_subsample",  # 적은 고장 사례에 가중치 부여
        random_state=42,  # 실행 결과 재현
        n_jobs=-1,  # 사용 가능한 CPU 코어 활용
    ),
)
model.fit(X_train, y_train)

# 각 검증 행의 '24시간 내 고장' 예측 점수(0~1)
risk_score = model.predict_proba(X_valid)[:, 1]

# ★ 분류 임계값: 이 값을 바꿔도 학습된 모델 자체는 바뀌지 않음
threshold = 0.5
predicted = (risk_score >= threshold).astype(int)

# 정답 분포와 임계값을 적용한 분류 결과 출력
print("학습 정답 분포:", y_train.value_counts().to_dict())
print("검증 정답 분포:", y_valid.value_counts().to_dict())
print("임계값:", threshold)
print(
    classification_report(
        y_valid,
        predicted,
        target_names=["24시간 내 고장 없음", "24시간 내 고장"],
        zero_division=0,
    )
)

# TN/FP/FN/TP는 현재 선택한 임계값에 따라 달라짐
tn, fp, fn, tp = confusion_matrix(y_valid, predicted, labels=[0, 1]).ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

# PR-AUC와 ROC-AUC는 분류 임계값이 아닌 전체 예측 점수로 계산
print("PR-AUC:", average_precision_score(y_valid, risk_score))
print("ROC-AUC:", roc_auc_score(y_valid, risk_score))


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Windows에서 그래프의 한글 표시
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

thresholds = [0.3, 0.5, 0.7]
fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

for ax, threshold in zip(axes, thresholds):
    # 같은 모델 점수에 서로 다른 임계값 적용
    predicted_at_threshold = (risk_score >= threshold).astype(int)

    # 행: 실제 정답(0, 1), 열: 모델 예측(0, 1)
    cm = confusion_matrix(y_valid, predicted_at_threshold, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_valid, predicted_at_threshold, zero_division=0)
    recall = recall_score(y_valid, predicted_at_threshold, zero_division=0)

    ax.imshow(cm, cmap="Blues", vmin=0, vmax=len(y_valid))

    # 각 칸에 TN, FP, FN, TP와 건수 표시
    cell_names = [["TN", "FP"], ["FN", "TP"]]
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                f"{cell_names[row][col]}\n{cm[row, col]}",
                ha="center",
                va="center",
                fontsize=15,
                color="white" if cm[row, col] > len(y_valid) / 2 else "black",
            )

    ax.set_xticks([0, 1], ["고장 없음", "24시간 내 고장"])
    ax.set_yticks([0, 1], ["고장 없음", "24시간 내 고장"])
    ax.set_xlabel("모델 예측")
    ax.set_ylabel("실제 정답")
    ax.set_title(
        f"임계값 {threshold:.1f}\n" f"정밀도 {precision:.2f}  |  재현율 {recall:.2f}"
    )

plt.show()
