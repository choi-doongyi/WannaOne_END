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
from pathlib import Path
import sys

# Allow direct execution with `python src/two2.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.eda import (
    load_scenario,
    load_slope,
    add_changes,
    add_score,
    compare,
    plot_example,
    main,
)

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
elif platform.system() == "Darwin":  # macOS
    plt.rcParams["font.family"] = "AppleGothic"
else:  # Linux
    plt.rcParams["font.family"] = "NanumGothic"


# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


df = pd.read_parquet(
    r"C:\Users\user\Documents\GitHub\WannaOne_END\data\POSKO-PdM-Data-pdm-v2.2.0-parquet\scenarios\S0001\casting.parquet"
)


print(df.head(3))

import json

root = Path(
    r"C:\Users\user\Documents\GitHub\WannaOne_END\data\데이터셋-20261013T011038Z-1-001\데이터셋\합성데이터_POSKO\POSKO-PdM-Data-pdm-v2.2.0\POSKO-PdM-Data-pdm-v2.2.0"
)
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

source = next(
    item
    for item in manifest["files"]
    if item["scenarioId"] == "S0001" and item["id"] == "casting"
)

df = pd.concat(
    [pd.read_csv(root / "chunks" / chunk["sha256"]) for chunk in source["chunks"]],
    ignore_index=True,
)

print(df.head())
print(df.tail())

import subprocess

project = Path(r"C:\Users\user\Documents\GitHub\WannaOne_END")

# 실행 후 생성된 결과를 df로 불러오기
df = pd.read_csv(project / "analysis" / "output" / "train_hourly_features.csv")
print(df.head(7))

print(df.columns)
