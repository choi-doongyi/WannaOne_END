# boosting_24h.py

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 0. 경로 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "casting_train_features.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("DATA LOAD")
print("=" * 80)

print("Train:", train_df.shape)

print("Validation:", valid_df.shape)


# ============================================================
# 2. 24H 학습 데이터 생성
# ============================================================

# 검열된 데이터 제외
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


print()

print("Train Label")

print(train["RISK_24H"].value_counts())


print()

print("Validation Label")

print(valid["RISK_24H"].value_counts())


# ============================================================
# 3. Feature 선택
# ============================================================

EXCLUDE_COLS = {
    # 식별자
    "SCENARIO_ID",
    # 시간
    "OBSERVED_AT",
    "LAST_MEAS_DT",
    # split
    "SPLIT",
    # 정답
    "RISK_6H",
    "RISK_24H",
    # 검열정보
    "CENSORED_6H",
    "CENSORED_24H",
}


feature_cols = [col for col in train.columns if col not in EXCLUDE_COLS]


# 숫자형 Feature만
feature_cols = [
    col for col in feature_cols if pd.api.types.is_numeric_dtype(train[col])
]


print()
print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print("Feature 수:", len(feature_cols))


X_train = train[feature_cols]

y_train = train["RISK_24H"]


X_valid = valid[feature_cols]

y_valid = valid["RISK_24H"]


# ============================================================
# 4. Class Imbalance 확인
# ============================================================

negative_count = (y_train == 0).sum()


positive_count = (y_train == 1).sum()


scale_pos_weight = negative_count / positive_count


print()
print("=" * 80)
print("CLASS IMBALANCE")
print("=" * 80)

print("Negative:", negative_count)

print("Positive:", positive_count)

print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 5. 모델 정의
# ============================================================

models = {
    # --------------------------------------------------------
    # Random Forest
    # 기존 baseline과 동일한 조건
    # --------------------------------------------------------
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
    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------
    "XGBoost": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------
    "LightGBM": Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    num_leaves=15,
                    max_depth=-1,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    ),
}


# ============================================================
# 6. Threshold 평가
# ============================================================


def evaluate_thresholds(model_name, model):

    print()
    print("=" * 80)
    print(f"{model_name} 학습")
    print("=" * 80)

    # --------------------------------------------------------
    # 학습
    # --------------------------------------------------------

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # Validation 확률
    # --------------------------------------------------------

    probability = model.predict_proba(X_valid)[:, 1]

    # --------------------------------------------------------
    # Threshold 독립 지표
    # --------------------------------------------------------

    roc_auc = roc_auc_score(y_valid, probability)

    pr_auc = average_precision_score(y_valid, probability)

    print(f"ROC-AUC : {roc_auc:.4f}")

    print(f"PR-AUC  : {pr_auc:.4f}")

    # --------------------------------------------------------
    # Threshold 탐색
    # --------------------------------------------------------

    rows = []

    for threshold in np.arange(0.05, 0.91, 0.01):

        pred = (probability >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()

        precision = precision_score(y_valid, pred, zero_division=0)

        recall = recall_score(y_valid, pred, zero_division=0)

        f1 = f1_score(y_valid, pred, zero_division=0)

        # Recall에 더 큰 비중
        f2 = fbeta_score(y_valid, pred, beta=2, zero_division=0)

        rows.append(
            {
                "MODEL": model_name,
                "THRESHOLD": round(float(threshold), 2),
                "PRECISION": precision,
                "RECALL": recall,
                "F1": f1,
                "F2": f2,
                "ROC_AUC": roc_auc,
                "PR_AUC": pr_auc,
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 7. 세 모델 실행
# ============================================================

results = []


for model_name, model in models.items():

    result = evaluate_thresholds(model_name, model)

    results.append(result)


all_results = pd.concat(results, ignore_index=True)


# ============================================================
# 8. Threshold = 0.50 비교
# ============================================================

baseline = all_results[all_results["THRESHOLD"] == 0.50].copy()


print()
print("=" * 80)
print("MODEL COMPARISON - THRESHOLD 0.50")
print("=" * 80)


print(
    baseline[
        [
            "MODEL",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "ROC_AUC",
            "PR_AUC",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 9. 모델별 F2 최적값
# ============================================================

best_f2_rows = []


for model_name in all_results["MODEL"].unique():

    temp = all_results[all_results["MODEL"] == model_name]

    best_index = temp["F2"].idxmax()

    best_f2_rows.append(all_results.loc[best_index])


best_f2 = pd.DataFrame(best_f2_rows)


# F2 높은 순
best_f2 = best_f2.sort_values("F2", ascending=False).reset_index(drop=True)


print()
print("=" * 80)
print("BEST THRESHOLD - F2")
print("=" * 80)


print(
    best_f2[
        [
            "MODEL",
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "ROC_AUC",
            "PR_AUC",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 10. 모델별 F1 최적값
# ============================================================

best_f1_rows = []


for model_name in all_results["MODEL"].unique():

    temp = all_results[all_results["MODEL"] == model_name]

    best_index = temp["F1"].idxmax()

    best_f1_rows.append(all_results.loc[best_index])


best_f1 = pd.DataFrame(best_f1_rows)


best_f1 = best_f1.sort_values("F1", ascending=False).reset_index(drop=True)


print()
print("=" * 80)
print("BEST THRESHOLD - F1")
print("=" * 80)


print(
    best_f1[
        [
            "MODEL",
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "ROC_AUC",
            "PR_AUC",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 11. PR-AUC 기준 모델 순위
# ============================================================

model_summary = (
    all_results[["MODEL", "ROC_AUC", "PR_AUC"]]
    .drop_duplicates(subset=["MODEL"])
    .sort_values("PR_AUC", ascending=False)
    .reset_index(drop=True)
)


print()
print("=" * 80)
print("MODEL RANKING - PR-AUC")
print("=" * 80)


print(model_summary.round(4).to_string(index=False))


# ============================================================
# 12. 저장
# ============================================================

all_path = OUTPUT_DIR / "casting_24h_boosting_threshold_all.csv"


baseline_path = OUTPUT_DIR / "casting_24h_boosting_baseline.csv"


f2_path = OUTPUT_DIR / "casting_24h_boosting_best_f2.csv"


f1_path = OUTPUT_DIR / "casting_24h_boosting_best_f1.csv"


ranking_path = OUTPUT_DIR / "casting_24h_model_ranking.csv"


all_results.to_csv(all_path, index=False, encoding="utf-8-sig")


baseline.to_csv(baseline_path, index=False, encoding="utf-8-sig")


best_f2.to_csv(f2_path, index=False, encoding="utf-8-sig")


best_f1.to_csv(f1_path, index=False, encoding="utf-8-sig")


model_summary.to_csv(ranking_path, index=False, encoding="utf-8-sig")


print()
print("=" * 80)
print("저장 완료")
print("=" * 80)


print(all_path)

print(baseline_path)

print(f2_path)

print(f1_path)

print(ranking_path)
