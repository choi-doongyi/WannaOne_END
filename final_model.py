# ============================================================
# final_model.py
#
# Casting Predictive Maintenance Final Model
#
# 24H : RandomForest / threshold = 0.47
# 6H  : RandomForest / threshold = 0.26
#
# 결과:
# 1. 24H 모델 저장
# 2. 6H 모델 저장
# 3. Validation 위험도 예측
# 4. risk6h <= risk24h 일관성 보정
# 5. 정상 / 주의 / 위험 경보 생성
# 6. Feature Importance 저장
# ============================================================


from pathlib import Path
import json

import joblib
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

# ============================================================
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features.csv"


# ------------------------------------------------------------
# 우리가 앞에서 선택한 최종 후보 Threshold
# ------------------------------------------------------------

THRESHOLD_24H = 0.47
THRESHOLD_6H = 0.26


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("FINAL MODEL")
print("=" * 80)

print()

print("Train:", train_df.shape)

print("Validation:", valid_df.shape)


# ============================================================
# 2. Feature 선택
# ============================================================

EXCLUDE_COLS = {
    # 식별자
    "SCENARIO_ID",
    # 시간
    "OBSERVED_AT",
    "LAST_MEAS_DT",
    # 데이터 Split
    "SPLIT",
    # 정답
    "RISK_6H",
    "RISK_24H",
    # 검열
    "CENSORED_6H",
    "CENSORED_24H",
}


feature_cols = [
    col
    for col in train_df.columns
    if (col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(train_df[col]))
]


print()

print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print("Feature 수:", len(feature_cols))


if len(feature_cols) != 237:

    print("[WARNING] 예상 Feature 수 237개와 다릅니다.")


# ============================================================
# 3. 24H 학습 데이터
# ============================================================

train_24h = train_df[
    (train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()
].copy()


valid_24h = valid_df[
    (valid_df["CENSORED_24H"] == 0) & valid_df["RISK_24H"].notna()
].copy()


train_24h["RISK_24H"] = train_24h["RISK_24H"].astype(int)


valid_24h["RISK_24H"] = valid_24h["RISK_24H"].astype(int)


X_train_24h = train_24h[feature_cols]

y_train_24h = train_24h["RISK_24H"]


X_valid_24h = valid_24h[feature_cols]

y_valid_24h = valid_24h["RISK_24H"]


# ============================================================
# 4. 6H 학습 데이터
# ============================================================

train_6h = train_df[(train_df["CENSORED_6H"] == 0) & train_df["RISK_6H"].notna()].copy()


valid_6h = valid_df[(valid_df["CENSORED_6H"] == 0) & valid_df["RISK_6H"].notna()].copy()


train_6h["RISK_6H"] = train_6h["RISK_6H"].astype(int)


valid_6h["RISK_6H"] = valid_6h["RISK_6H"].astype(int)


X_train_6h = train_6h[feature_cols]

y_train_6h = train_6h["RISK_6H"]


X_valid_6h = valid_6h[feature_cols]

y_valid_6h = valid_6h["RISK_6H"]


print()

print("=" * 80)
print("TRAIN DATA")
print("=" * 80)

print("24H Train:", X_train_24h.shape, "| Positive:", int(y_train_24h.sum()))

print("6H Train :", X_train_6h.shape, "| Positive:", int(y_train_6h.sum()))


# ============================================================
# 5. 최종 RandomForest 모델
# ============================================================

# ------------------------------------------------------------
# 24H
#
# 기존 24H 실험과 동일한 설정
# ------------------------------------------------------------

model_24h = Pipeline(
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


# ------------------------------------------------------------
# 6H
#
# 기존 6H 실험과 동일한 설정
# ------------------------------------------------------------

model_6h = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# ============================================================
# 6. 모델 학습
# ============================================================

print()

print("=" * 80)
print("MODEL TRAIN")
print("=" * 80)


print("24H RandomForest 학습...")

model_24h.fit(X_train_24h, y_train_24h)


print("6H RandomForest 학습...")

model_6h.fit(X_train_6h, y_train_6h)


print("학습 완료!")


# ============================================================
# 7. 평가 함수
# ============================================================


def evaluate_model(name, model, X, y, threshold):

    probability = model.predict_proba(X)[:, 1]

    prediction = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    precision = precision_score(y, prediction, zero_division=0)

    recall = recall_score(y, prediction, zero_division=0)

    f1 = f1_score(y, prediction, zero_division=0)

    f2 = fbeta_score(y, prediction, beta=2, zero_division=0)

    roc_auc = roc_auc_score(y, probability)

    pr_auc = average_precision_score(y, probability)

    result = {
        "MODEL": name,
        "THRESHOLD": threshold,
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

    return result


# ============================================================
# 8. Validation 평가
# ============================================================

result_24h = evaluate_model(
    "RandomForest_24H", model_24h, X_valid_24h, y_valid_24h, THRESHOLD_24H
)


result_6h = evaluate_model(
    "RandomForest_6H", model_6h, X_valid_6h, y_valid_6h, THRESHOLD_6H
)


metrics_df = pd.DataFrame([result_24h, result_6h])


print()

print("=" * 80)
print("FINAL MODEL VALIDATION")
print("=" * 80)


print(metrics_df.round(4).to_string(index=False))


# ============================================================
# 9. 전체 Validation 위험도 예측
#
# 검열 여부와 관계없이 센서 Feature가 있으면
# 모델 확률 자체는 계산 가능
# ============================================================

X_valid_all = valid_df[feature_cols]


raw_risk_24h = model_24h.predict_proba(X_valid_all)[:, 1]


raw_risk_6h = model_6h.predict_proba(X_valid_all)[:, 1]


# ============================================================
# 10. 확률 일관성 보정
#
# 논리적으로:
#
# P(6시간 내 고장)
# <=
# P(24시간 내 고장)
#
# 이어야 함
#
# 독립 모델이라 이를 위반할 수 있으므로
# 24H를 max(24H, 6H)로 보정
# ============================================================

risk_6h = raw_risk_6h.copy()


risk_24h = np.maximum(raw_risk_24h, raw_risk_6h)


violation_count = int((raw_risk_6h > raw_risk_24h).sum())


print()

print("=" * 80)
print("RISK CONSISTENCY CHECK")
print("=" * 80)

print("보정 전 risk6h > risk24h 건수:", violation_count)

print("보정 후 위반 건수:", int((risk_6h > risk_24h).sum()))


# ============================================================
# 11. 경보 등급 생성
#
# 우선순위:
#
# 6H >= 0.26 → DANGER
#
# 아니면서
#
# 24H >= 0.47 → CAUTION
#
# 둘 다 아니면 NORMAL
# ============================================================


def make_alert(risk6, risk24):

    if risk6 >= THRESHOLD_6H:

        return "DANGER"

    elif risk24 >= THRESHOLD_24H:

        return "CAUTION"

    else:

        return "NORMAL"


alerts = [make_alert(r6, r24) for r6, r24 in zip(risk_6h, risk_24h)]


# ============================================================
# 12. Validation 결과 Dataset
# ============================================================

prediction_df = pd.DataFrame(
    {
        "SCENARIO_ID": valid_df["SCENARIO_ID"],
        "OBSERVED_AT": valid_df["OBSERVED_AT"],
        # 원래 모델 출력
        "RAW_RISK_6H": raw_risk_6h,
        "RAW_RISK_24H": raw_risk_24h,
        # 논리 일관성 보정 후
        "RISK_6H": risk_6h,
        "RISK_24H": risk_24h,
        "ALERT": alerts,
        # 정답 확인용
        "TRUE_RISK_6H": valid_df["RISK_6H"],
        "TRUE_RISK_24H": valid_df["RISK_24H"],
        "CENSORED_6H": valid_df["CENSORED_6H"],
        "CENSORED_24H": valid_df["CENSORED_24H"],
    }
)


# ============================================================
# 13. 경보 분포 확인
# ============================================================

print()

print("=" * 80)
print("ALERT DISTRIBUTION")
print("=" * 80)


print(prediction_df["ALERT"].value_counts())


print()

print("경보 비율 (%)")


print((prediction_df["ALERT"].value_counts(normalize=True) * 100).round(2))


# ============================================================
# 14. Feature Importance 함수
# ============================================================


def get_feature_importance(pipeline, feature_names, model_name):

    rf = pipeline.named_steps["model"]

    importance = pd.DataFrame(
        {
            "FEATURE": feature_names,
            "IMPORTANCE": rf.feature_importances_,
        }
    )

    importance["MODEL"] = model_name

    importance = importance.sort_values("IMPORTANCE", ascending=False).reset_index(
        drop=True
    )

    return importance


# ============================================================
# 15. Feature Importance 생성
# ============================================================

importance_24h = get_feature_importance(model_24h, feature_cols, "24H")


importance_6h = get_feature_importance(model_6h, feature_cols, "6H")


print()

print("=" * 80)
print("24H TOP 20 FEATURES")
print("=" * 80)


print(
    importance_24h[["FEATURE", "IMPORTANCE"]].head(20).round(6).to_string(index=False)
)


print()

print("=" * 80)
print("6H TOP 20 FEATURES")
print("=" * 80)


print(importance_6h[["FEATURE", "IMPORTANCE"]].head(20).round(6).to_string(index=False))


# ============================================================
# 16. 모델 저장
# ============================================================

MODEL_24H_PATH = MODEL_DIR / "casting_rf_24h.joblib"


MODEL_6H_PATH = MODEL_DIR / "casting_rf_6h.joblib"


joblib.dump(model_24h, MODEL_24H_PATH)


joblib.dump(model_6h, MODEL_6H_PATH)


# ============================================================
# 17. Feature 목록 저장
# ============================================================

FEATURE_PATH = MODEL_DIR / "casting_feature_columns.json"


with FEATURE_PATH.open("w", encoding="utf-8") as f:

    json.dump(feature_cols, f, ensure_ascii=False, indent=2)


# ============================================================
# 18. 모델 설정 저장
# ============================================================

CONFIG = {
    "equipment": "P1-CC1-INS01",
    "source": "casting",
    "feature_count": len(feature_cols),
    "model_24h": "RandomForest",
    "threshold_24h": THRESHOLD_24H,
    "model_6h": "RandomForest",
    "threshold_6h": THRESHOLD_6H,
    "alert_rule": {
        "DANGER": "risk6h >= 0.26",
        "CAUTION": "risk6h < 0.26 and risk24h >= 0.47",
        "NORMAL": "otherwise",
    },
    "probability_rule": "risk24h = max(raw_risk24h, raw_risk6h)",
}


CONFIG_PATH = MODEL_DIR / "casting_model_config.json"


with CONFIG_PATH.open("w", encoding="utf-8") as f:

    json.dump(CONFIG, f, ensure_ascii=False, indent=2)


# ============================================================
# 19. 결과 CSV 저장
# ============================================================

METRICS_PATH = OUTPUT_DIR / "casting_final_model_metrics.csv"


PREDICTION_PATH = OUTPUT_DIR / "casting_validation_alerts.csv"


IMPORTANCE_24H_PATH = OUTPUT_DIR / "casting_24h_feature_importance.csv"


IMPORTANCE_6H_PATH = OUTPUT_DIR / "casting_6h_feature_importance.csv"


metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")


prediction_df.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")


importance_24h.to_csv(IMPORTANCE_24H_PATH, index=False, encoding="utf-8-sig")


importance_6h.to_csv(IMPORTANCE_6H_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 20. 최종 확인
# ============================================================

print()

print("=" * 80)
print("FINAL MODEL SAVE COMPLETE")
print("=" * 80)


print()

print("24H 모델:", MODEL_24H_PATH)

print("6H 모델:", MODEL_6H_PATH)

print("Feature 목록:", FEATURE_PATH)

print("모델 설정:", CONFIG_PATH)


print()

print("최종 Metrics:", METRICS_PATH)

print("Validation 경보:", PREDICTION_PATH)

print("24H Importance:", IMPORTANCE_24H_PATH)

print("6H Importance:", IMPORTANCE_6H_PATH)


print()

print("=" * 80)
print("예지보전 모델 구축 완료")
print("=" * 80)
