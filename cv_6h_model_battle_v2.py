# ============================================================
# cv_6h_model_battle_v2.py
#
# 6H DANGER - FAIR MODEL BATTLE
#
# Feature:
# BASE 240 고정
#
# 비교 모델:
# 1. RandomForest
# 2. XGBoost
# 3. LightGBM
#
# CV:
# StratifiedGroupKFold 5-Fold
# SCENARIO_ID 기준 그룹 분리
#
# 모델 선택:
# Train OOF PR-AUC
#
# Threshold:
# 선택 모델의 OOF F2 최대
#
# Validation:
# 모델 + Threshold LOCK 이후 마지막 평가
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

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
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v25.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features_v25.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("6H FAIR MODEL BATTLE V2")
print("RF VS XGB VS LIGHTGBM")
print("=" * 80)

print()

print("Train:", train_df.shape)

print("Validation:", valid_df.shape)


# ============================================================
# 2. 6H Label
# ============================================================

train = train_df[(train_df["CENSORED_6H"] == 0) & train_df["RISK_6H"].notna()].copy()


valid = valid_df[(valid_df["CENSORED_6H"] == 0) & valid_df["RISK_6H"].notna()].copy()


train["RISK_6H"] = train["RISK_6H"].astype(int)

valid["RISK_6H"] = valid["RISK_6H"].astype(int)


print()

print("=" * 80)
print("6H LABEL CHECK")
print("=" * 80)

print("Train usable:", train.shape)

print(train["RISK_6H"].value_counts())

print()

print("Validation usable:", valid.shape)

print(valid["RISK_6H"].value_counts())


# ============================================================
# 3. 모델 제외 컬럼
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


# ============================================================
# 4. 정비 Feature
# ============================================================

ALL_MAINTENANCE_COLS = [
    "MNT_DOWN_SUM",
    "H_SINCE_MNT",
    "LAST_MNT_EXISTS",
    "LAST_DOWN_MIN",
    "LAST_COST_SCORE",
    "LAST_MNT_IS_PARTIAL",
    "LAST_MNT_IS_REPAIR",
    "LAST_MNT_IS_REPLACE",
    "LAST_PART_UNKNOWN",
    "LAST_PART_OSCILLATOR",
    "LAST_PART_VALVE",
]


SELECTED_MAINTENANCE_COLS = [
    "MNT_DOWN_SUM",
    "LAST_DOWN_MIN",
    "LAST_MNT_IS_PARTIAL",
]


DROP_MAINTENANCE_COLS = [
    col for col in ALL_MAINTENANCE_COLS if col not in SELECTED_MAINTENANCE_COLS
]


# ============================================================
# 5. V2.5 Trend Feature
#
# 6H BASE 240에서는 전부 제외
# ============================================================

TREND_SENSORS = [
    "CONTROL_DELAY",
    "MLD-LVL",
    "OSC-STK",
    "OSC-FRQ",
]


WINDOWS = [
    "1H",
    "3H",
    "6H",
    "12H",
]


trend_per_hour_features = []


for sensor in TREND_SENSORS:

    for window in WINDOWS:

        trend_per_hour_features.append(f"{window}_" f"{sensor}_TREND_PER_H")


# ============================================================
# 6. Mean Diff 12개
#
# 6H에서 성능 하락했으므로 전부 제외
# ============================================================

mean_diff_features = [
    "CONTROL_DELAY_1H_MINUS_3H_MEAN",
    "CONTROL_DELAY_3H_MINUS_6H_MEAN",
    "CONTROL_DELAY_6H_MINUS_12H_MEAN",
    "MLD-LVL_1H_MINUS_3H_MEAN",
    "MLD-LVL_3H_MINUS_6H_MEAN",
    "MLD-LVL_6H_MINUS_12H_MEAN",
    "OSC-STK_1H_MINUS_3H_MEAN",
    "OSC-STK_3H_MINUS_6H_MEAN",
    "OSC-STK_6H_MINUS_12H_MEAN",
    "OSC-FRQ_1H_MINUS_3H_MEAN",
    "OSC-FRQ_3H_MINUS_6H_MEAN",
    "OSC-FRQ_6H_MINUS_12H_MEAN",
]


# ============================================================
# 7. BASE 240 Feature
# ============================================================

base_features = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE_COLS
        and col not in DROP_MAINTENANCE_COLS
        and col not in trend_per_hour_features
        and col not in mean_diff_features
        and pd.api.types.is_numeric_dtype(train[col])
    )
]


print()

print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print("BASE Feature 수:", len(base_features))


if len(base_features) != 240:

    print("[WARNING] 예상 240개인데 실제:", len(base_features))


print()

print("정비 Feature:")

for col in SELECTED_MAINTENANCE_COLS:

    print(f"{col:<25}", "→", col in base_features)


# ============================================================
# 8. X / y / Group
# ============================================================

X = train[base_features].copy()


X_valid = valid[base_features].copy()


y = train["RISK_6H"].copy()


y_valid = valid["RISK_6H"].copy()


groups = train["SCENARIO_ID"].copy()


# ============================================================
# 9. Class Imbalance
# ============================================================

negative_count = int((y == 0).sum())

positive_count = int((y == 1).sum())


scale_pos_weight = negative_count / positive_count


print()

print("=" * 80)
print("CLASS IMBALANCE")
print("=" * 80)

print("Negative:", negative_count)

print("Positive:", positive_count)

print("Positive Rate:", round(positive_count / len(y), 6))

print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 10. 모델 생성
# ============================================================


def make_model(model_name):

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if model_name == "RF":

        return RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    # --------------------------------------------------------
    # XGBoost
    #
    # 기존 Fair XGB와 같은 조건
    # --------------------------------------------------------

    elif model_name == "XGB":

        return XGBClassifier(
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
        )

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

    elif model_name == "LGBM":

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    else:

        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 11. Threshold 평가
# ============================================================


def evaluate_threshold(y_true, probability, threshold):

    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, pred, zero_division=0)

    recall = recall_score(y_true, pred, zero_division=0)

    f1 = f1_score(y_true, pred, zero_division=0)

    f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)

    return {
        "THRESHOLD": float(threshold),
        "PRECISION": precision,
        "RECALL": recall,
        "F1": f1,
        "F2": f2,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# 12. CV 설정
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


MODEL_NAMES = [
    "RF",
    "XGB",
    "LGBM",
]


# ============================================================
# 13. 저장 구조
# ============================================================

model_results = {}

fold_rows = []

oof_prediction_df = train[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
    ]
].copy()


# ============================================================
# 14. 모델별 5-Fold CV
# ============================================================

for model_name in MODEL_NAMES:

    print()

    print("=" * 80)

    print(f"{model_name} - 5 FOLD CV")

    print("=" * 80)

    oof_probability = np.full(len(train), np.nan, dtype=float)

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        # ====================================================
        # Fold 데이터
        # ====================================================

        X_fold_train = X.iloc[train_idx].copy()

        X_fold_valid = X.iloc[fold_valid_idx].copy()

        y_fold_train = y.iloc[train_idx].copy()

        y_fold_valid = y.iloc[fold_valid_idx].copy()

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        # ====================================================
        # Scenario leakage 검사
        # ====================================================

        if train_groups & valid_groups:

            raise RuntimeError(
                f"{model_name} " f"Fold {fold}: " f"Scenario leakage 발생"
            )

        # ====================================================
        # Fold Train 안에서만 Imputer fit
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_fold_train_imp = imputer.fit_transform(X_fold_train)

        X_fold_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # 모델 학습
        # ====================================================

        model = make_model(model_name)

        model.fit(X_fold_train_imp, y_fold_train)

        # ====================================================
        # OOF Probability
        # ====================================================

        fold_probability = model.predict_proba(X_fold_valid_imp)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # Fold Metric
        # ====================================================

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "MODEL": model_name,
                "FOLD": fold,
                "FEATURE_COUNT": len(base_features),
                "TRAIN_SCENARIOS": len(train_groups),
                "VALID_SCENARIOS": len(valid_groups),
                "TRAIN_POSITIVE": int(y_fold_train.sum()),
                "VALID_POSITIVE": int(y_fold_valid.sum()),
                "ROC_AUC": fold_roc,
                "PR_AUC": fold_pr,
            }
        )

        print(
            f"Fold {fold}"
            f" | Positive "
            f"{int(y_fold_valid.sum())}"
            f" | ROC "
            f"{fold_roc:.4f}"
            f" | PR-AUC "
            f"{fold_pr:.4f}"
        )

    # ========================================================
    # OOF 검사
    # ========================================================

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{model_name}: " f"OOF NaN 발생")

    # ========================================================
    # 전체 OOF 평가
    # ========================================================

    oof_roc = roc_auc_score(y, oof_probability)

    oof_pr = average_precision_score(y, oof_probability)

    print()

    print(f"{model_name} " f"OOF ROC-AUC: " f"{oof_roc:.4f}")

    print(f"{model_name} " f"OOF PR-AUC : " f"{oof_pr:.4f}")

    # ========================================================
    # 저장
    # ========================================================

    model_results[model_name] = {
        "oof_probability": oof_probability,
        "roc_auc": oof_roc,
        "pr_auc": oof_pr,
    }

    oof_prediction_df[f"{model_name}_PROBA"] = oof_probability


# ============================================================
# 15. 모델 Ranking
# ============================================================

ranking_rows = []


for model_name, result in model_results.items():

    ranking_rows.append(
        {
            "MODEL": model_name,
            "FEATURE_COUNT": len(base_features),
            "OOF_ROC_AUC": result["roc_auc"],
            "OOF_PR_AUC": result["pr_auc"],
        }
    )


ranking_df = pd.DataFrame(ranking_rows)


ranking_df = ranking_df.sort_values("OOF_PR_AUC", ascending=False).reset_index(
    drop=True
)


print()

print("=" * 80)

print("6H MODEL RANKING")

print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 16. Best Model 선택
# ============================================================

best_model_name = str(ranking_df.iloc[0]["MODEL"])


best_oof_probability = model_results[best_model_name]["oof_probability"]


print()

print("선택 Model:", best_model_name)


print("선택 OOF PR-AUC:", round(model_results[best_model_name]["pr_auc"], 4))


# ============================================================
# 17. Threshold Search
#
# 6H는 양성이 매우 적어서
# 0.01부터 탐색
# ============================================================

threshold_rows = []


for threshold in np.arange(0.01, 0.81, 0.01):

    threshold = round(float(threshold), 2)

    metrics = evaluate_threshold(
        y_true=y, probability=best_oof_probability, threshold=threshold
    )

    threshold_rows.append(metrics)


threshold_df = pd.DataFrame(threshold_rows)


# ============================================================
# 18. F2 최대 Threshold
# ============================================================

best_threshold_idx = threshold_df["F2"].idxmax()


best_threshold_row = threshold_df.loc[best_threshold_idx]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)

print("BEST MODEL - OOF BEST THRESHOLD")

print("=" * 80)


print(
    best_threshold_row[
        [
            "THRESHOLD",
            "PRECISION",
            "RECALL",
            "F1",
            "F2",
            "TN",
            "FP",
            "FN",
            "TP",
        ]
    ]
    .round(4)
    .to_string()
)


# ============================================================
# 19. 모델 LOCK
# ============================================================

print()

print("=" * 80)

print("6H FINAL LOCK")

print("=" * 80)


print("Model:", best_model_name)


print("Feature Count:", len(base_features))


print("Threshold:", best_threshold)


# ============================================================
# 20. Full Train Imputer
# ============================================================

final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X)


X_validation_imp = final_imputer.transform(X_valid)


# ============================================================
# 21. Full Train 모델
# ============================================================

final_model = make_model(best_model_name)


final_model.fit(X_train_imp, y)


# ============================================================
# 22. Validation Prediction
# ============================================================

valid_probability = final_model.predict_proba(X_validation_imp)[:, 1]


# ============================================================
# 23. Validation Metric
# ============================================================

valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


print()

print("=" * 80)

print("FINAL VALIDATION - 6H MODEL BATTLE")

print("=" * 80)


print("Model:", best_model_name)


print("Feature Count:", len(base_features))


print("Threshold:", best_threshold)


print()


print(f"ROC-AUC   : " f"{valid_roc_auc:.4f}")


print(f"PR-AUC    : " f"{valid_pr_auc:.4f}")


print(f"Precision : " f"{valid_metrics['PRECISION']:.4f}")


print(f"Recall    : " f"{valid_metrics['RECALL']:.4f}")


print(f"F1        : " f"{valid_metrics['F1']:.4f}")


print(f"F2        : " f"{valid_metrics['F2']:.4f}")


print()


print(f"TN={valid_metrics['TN']} " f"FP={valid_metrics['FP']}")


print(f"FN={valid_metrics['FN']} " f"TP={valid_metrics['TP']}")


# ============================================================
# 24. Validation 모델별 PR-AUC도 참고용 계산
#
# 중요:
# 모델 선택에는 사용하지 않는다.
# 이미 Train OOF에서 Best Model을 LOCK한 상태.
# ============================================================

validation_model_rows = []


for model_name in MODEL_NAMES:

    model = make_model(model_name)

    model.fit(X_train_imp, y)

    probability = model.predict_proba(X_validation_imp)[:, 1]

    roc_value = roc_auc_score(y_valid, probability)

    pr_value = average_precision_score(y_valid, probability)

    validation_model_rows.append(
        {
            "MODEL": model_name,
            "VAL_ROC_AUC": roc_value,
            "VAL_PR_AUC": pr_value,
            "SELECTED_BY_OOF": (model_name == best_model_name),
        }
    )


validation_model_df = pd.DataFrame(validation_model_rows)


print()

print("=" * 80)

print("VALIDATION MODEL AP - 참고용")

print("※ 모델 선택에는 사용하지 않음")

print("=" * 80)


print(validation_model_df.round(4).to_string(index=False))


# ============================================================
# 25. Feature Importance
# ============================================================

if hasattr(final_model, "feature_importances_"):

    importance_df = pd.DataFrame(
        {
            "FEATURE": base_features,
            "IMPORTANCE": final_model.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        "IMPORTANCE", ascending=False
    ).reset_index(drop=True)


else:

    importance_df = pd.DataFrame(
        columns=[
            "FEATURE",
            "IMPORTANCE",
        ]
    )


print()

print("=" * 80)

print("TOP 50 FEATURE IMPORTANCE")

print("=" * 80)


print(importance_df.head(50).round(6).to_string(index=False))


# ============================================================
# 26. 정비 Feature Importance
# ============================================================

maintenance_importance_df = importance_df[
    importance_df["FEATURE"].isin(SELECTED_MAINTENANCE_COLS)
].copy()


print()

print("=" * 80)

print("MAINTENANCE FEATURE IMPORTANCE")

print("=" * 80)


print(maintenance_importance_df.round(6).to_string(index=False))


# ============================================================
# 27. Validation Prediction 저장
# ============================================================

validation_prediction_df = valid[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
    ]
].copy()


validation_prediction_df["PRED_PROBA_6H"] = valid_probability


validation_prediction_df["PRED_6H"] = (valid_probability >= best_threshold).astype(int)


# ============================================================
# 28. Best OOF Prediction 저장
# ============================================================

oof_prediction_df["BEST_MODEL"] = best_model_name


oof_prediction_df["BEST_PROBA"] = best_oof_probability


oof_prediction_df["BEST_PRED"] = (best_oof_probability >= best_threshold).astype(int)


# ============================================================
# 29. Fold DataFrame
# ============================================================

fold_df = pd.DataFrame(fold_rows)


# ============================================================
# 30. 저장
# ============================================================

ranking_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_ranking_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_folds_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_thresholds_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


validation_model_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_validation_models_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


importance_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_importance_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


maintenance_importance_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_maintenance_importance_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


validation_prediction_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_validation_predictions_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


oof_prediction_df.to_csv(
    OUTPUT_DIR / "casting_6h_model_battle_oof_predictions_v2.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 31. 완료
# ============================================================

print()

print("=" * 80)

print("SAVE COMPLETE - 6H MODEL BATTLE V2")

print("=" * 80)
