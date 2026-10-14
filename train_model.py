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
    roc_auc_score,
    average_precision_score,
    classification_report,
)

# ============================================================
# 0. 경로 설정
# ============================================================

ROOT = Path(".")

TRAIN_PATH = ROOT / "outputs" / "casting_train_features.csv"

VALID_PATH = ROOT / "outputs" / "casting_validation_features.csv"


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

train_24 = train_df[
    (train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()
].copy()


valid_24 = valid_df[
    (valid_df["CENSORED_24H"] == 0) & valid_df["RISK_24H"].notna()
].copy()


train_24["RISK_24H"] = train_24["RISK_24H"].astype(int)

valid_24["RISK_24H"] = valid_24["RISK_24H"].astype(int)


print()
print("=" * 80)
print("24H LABEL CHECK")
print("=" * 80)

print("Train usable:", train_24.shape)

print(train_24["RISK_24H"].value_counts())

print("Train positive rate:", round(train_24["RISK_24H"].mean() * 100, 3), "%")

print()

print("Validation usable:", valid_24.shape)

print(valid_24["RISK_24H"].value_counts())

print("Validation positive rate:", round(valid_24["RISK_24H"].mean() * 100, 3), "%")


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


feature_cols = [col for col in train_24.columns if col not in EXCLUDE_COLS]


# 숫자형 Feature만 사용
feature_cols = [
    col for col in feature_cols if pd.api.types.is_numeric_dtype(train_24[col])
]


print()
print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print("Feature 수:", len(feature_cols))


X_train = train_24[feature_cols]

y_train = train_24["RISK_24H"]


X_valid = valid_24[feature_cols]

y_valid = valid_24["RISK_24H"]


# ============================================================
# 4. 결측치 확인
# ============================================================

missing_rate = X_train.isna().mean().sort_values(ascending=False)


print()
print("=" * 80)
print("MISSING VALUE CHECK")
print("=" * 80)

print(missing_rate.head(20))


# ============================================================
# 5. 평가 함수
# ============================================================


def evaluate_model(
    model_name, model, X_train, y_train, X_valid, y_valid, threshold=0.5
):

    print()
    print("=" * 80)
    print(model_name)
    print("=" * 80)

    # 학습
    model.fit(X_train, y_train)

    # 고장 확률
    prob = model.predict_proba(X_valid)[:, 1]

    # threshold 기준 예측
    pred = (prob >= threshold).astype(int)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_valid, pred, labels=[0, 1]).ravel()

    precision = precision_score(y_valid, pred, zero_division=0)

    recall = recall_score(y_valid, pred, zero_division=0)

    f1 = f1_score(y_valid, pred, zero_division=0)

    roc_auc = roc_auc_score(y_valid, prob)

    pr_auc = average_precision_score(y_valid, prob)

    print(f"Threshold : {threshold:.2f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1        : {f1:.4f}")

    print(f"ROC-AUC   : {roc_auc:.4f}")

    print(f"PR-AUC    : {pr_auc:.4f}")

    print()

    print("Confusion Matrix")

    print(f"TN={tn}  FP={fp}")

    print(f"FN={fn}  TP={tp}")

    print()

    print(classification_report(y_valid, pred, digits=4, zero_division=0))

    return {
        "model": model_name,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ============================================================
# 6. Logistic Regression
# ============================================================

logistic_model = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
        ),
    ]
)


# ============================================================
# 7. Random Forest
# ============================================================

rf_model = Pipeline(
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
)


# ============================================================
# 8. 모델 평가
# ============================================================

results = []


results.append(
    evaluate_model(
        "LogisticRegression",
        logistic_model,
        X_train,
        y_train,
        X_valid,
        y_valid,
        threshold=0.5,
    )
)


results.append(
    evaluate_model(
        "RandomForest", rf_model, X_train, y_train, X_valid, y_valid, threshold=0.5
    )
)


# ============================================================
# 9. 결과 비교
# ============================================================

result_df = pd.DataFrame(results)


print()
print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(result_df.round(4).to_string(index=False))


# ============================================================
# 10. 저장
# ============================================================

save_path = ROOT / "outputs" / "casting_24h_baseline_metrics.csv"


result_df.to_csv(save_path, index=False, encoding="utf-8-sig")


print()
print("저장 완료:", save_path)
