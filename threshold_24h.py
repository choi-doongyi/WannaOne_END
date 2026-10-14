from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
)

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

TRAIN_PATH = ROOT / "outputs" / "casting_train_features.csv"

VALID_PATH = ROOT / "outputs" / "casting_validation_features.csv"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 데이터 읽기
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("DATA LOAD")
print("=" * 80)

print("Train:", train_df.shape)

print("Validation:", valid_df.shape)


# ============================================================
# 2. 24H 검열 제거
# ============================================================

train = train_df[(train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()].copy()


valid = valid_df[(valid_df["CENSORED_24H"] == 0) & valid_df["RISK_24H"].notna()].copy()


train["RISK_24H"] = train["RISK_24H"].astype(int)

valid["RISK_24H"] = valid["RISK_24H"].astype(int)


print()
print("=" * 80)
print("24H DATA")
print("=" * 80)

print("Train:", train.shape)

print("Validation:", valid.shape)

print("Train positive:", int(train["RISK_24H"].sum()))

print("Validation positive:", int(valid["RISK_24H"].sum()))


# ============================================================
# 3. Feature 선택
# ============================================================

EXCLUDE_COLS = {
    "SCENARIO_ID",
    "OBSERVED_AT",
    "LAST_MEAS_DT",
    "SPLIT",
    "RISK_6H",
    "RISK_24H",
    "CENSORED_6H",
    "CENSORED_24H",
}


feature_cols = [col for col in train.columns if col not in EXCLUDE_COLS]


feature_cols = [
    col for col in feature_cols if pd.api.types.is_numeric_dtype(train[col])
]


print()
print("Feature 수:", len(feature_cols))


X_train = train[feature_cols]

y_train = train["RISK_24H"]


X_valid = valid[feature_cols]

y_valid = valid["RISK_24H"]


# ============================================================
# 4. 모델
# ============================================================

models = {
    "LogisticRegression": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=42
                ),
            ),
        ]
    ),
    "RandomForest": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ============================================================
# 5. Threshold 평가 함수
# ============================================================


def threshold_table(model_name, model, X_train, y_train, X_valid, y_valid):

    print()
    print("=" * 80)
    print(model_name, "학습")
    print("=" * 80)

    model.fit(X_train, y_train)

    prob = model.predict_proba(X_valid)[:, 1]

    # threshold에 영향을 받지 않는 지표
    roc_auc = roc_auc_score(y_valid, prob)

    pr_auc = average_precision_score(y_valid, prob)

    print(f"ROC-AUC : {roc_auc:.4f}")

    print(f"PR-AUC  : {pr_auc:.4f}")

    rows = []

    # 0.05 ~ 0.90
    thresholds = np.arange(0.05, 0.91, 0.01)

    for threshold in thresholds:

        pred = (prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()

        precision = precision_score(y_valid, pred, zero_division=0)

        recall = recall_score(y_valid, pred, zero_division=0)

        f1 = f1_score(y_valid, pred, zero_division=0)

        # Recall을 Precision보다 중요하게 보는 F2
        f2 = fbeta_score(y_valid, pred, beta=2, zero_division=0)

        rows.append(
            {
                "MODEL": model_name,
                "THRESHOLD": round(threshold, 2),
                "PRECISION": precision,
                "RECALL": recall,
                "F1": f1,
                "F2": f2,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    result = pd.DataFrame(rows)

    return (result, roc_auc, pr_auc)


# ============================================================
# 6. 전체 모델 Threshold 탐색
# ============================================================

all_results = []


for model_name, model in models.items():

    result, roc_auc, pr_auc = threshold_table(
        model_name, model, X_train, y_train, X_valid, y_valid
    )

    result["ROC_AUC"] = roc_auc

    result["PR_AUC"] = pr_auc

    all_results.append(result)


threshold_df = pd.concat(all_results, ignore_index=True)


# ============================================================
# 7. 모델별 F2 최적 Threshold
# ============================================================

print()
print("=" * 80)
print("BEST THRESHOLD - F2")
print("=" * 80)


best_f2_rows = []


for model_name in threshold_df["MODEL"].unique():

    model_df = threshold_df[threshold_df["MODEL"] == model_name]

    best_idx = model_df["F2"].idxmax()

    best = threshold_df.loc[best_idx]

    best_f2_rows.append(best)


best_f2_df = pd.DataFrame(best_f2_rows)


print(
    best_f2_df[
        [
            "MODEL",
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 8. F1 기준도 같이 확인
# ============================================================

print()
print("=" * 80)
print("BEST THRESHOLD - F1")
print("=" * 80)


best_f1_rows = []


for model_name in threshold_df["MODEL"].unique():

    model_df = threshold_df[threshold_df["MODEL"] == model_name]

    best_idx = model_df["F1"].idxmax()

    best = threshold_df.loc[best_idx]

    best_f1_rows.append(best)


best_f1_df = pd.DataFrame(best_f1_rows)


print(
    best_f1_df[
        [
            "MODEL",
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 9. RandomForest 상세 Threshold 확인
# ============================================================

print()
print("=" * 80)
print("RANDOM FOREST THRESHOLD TABLE")
print("=" * 80)


rf_result = threshold_df[
    (threshold_df["MODEL"] == "RandomForest")
    & (threshold_df["THRESHOLD"].between(0.20, 0.70))
].copy()


# 너무 길지 않게 0.05 단위 출력
rf_result = rf_result[((rf_result["THRESHOLD"] * 100).round().astype(int) % 5) == 0]


print(
    rf_result[
        [
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. 저장
# ============================================================

all_path = OUTPUT_DIR / "casting_24h_threshold_all.csv"


best_f2_path = OUTPUT_DIR / "casting_24h_best_f2.csv"


best_f1_path = OUTPUT_DIR / "casting_24h_best_f1.csv"


threshold_df.to_csv(all_path, index=False, encoding="utf-8-sig")


best_f2_df.to_csv(best_f2_path, index=False, encoding="utf-8-sig")


best_f1_df.to_csv(best_f1_path, index=False, encoding="utf-8-sig")


print()
print("=" * 80)
print("저장 완료")
print("=" * 80)

print(all_path)

print(best_f2_path)

print(best_f1_path)
