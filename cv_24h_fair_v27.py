# ============================================================
# cv_24h_fair_v27.py
#
# V2.7
#
# 비교:
# 1. BASE_240
# 2. MEAN_DIFF_9 = 249 Features
#
# V2.6에서 중요도 0이었던
# OSC-FRQ Mean Diff 3개 제거
#
# 유지 Mean Diff:
# CONTROL_DELAY 3개
# MLD-LVL 3개
# OSC-STK 3개
#
# Model:
# XGBoost
#
# CV:
# StratifiedGroupKFold 5-Fold
#
# 모델 선택:
# OOF PR-AUC
#
# Threshold:
# OOF F2 최대
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

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
print("24H FAIR MODEL V2.7")
print("BASE 240 VS MEAN DIFF 9")
print("=" * 80)

print()
print("Train:", train_df.shape)
print("Validation:", valid_df.shape)


# ============================================================
# 2. 유효 Label
# ============================================================

train = train_df[(train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()].copy()


valid = valid_df[(valid_df["CENSORED_24H"] == 0) & valid_df["RISK_24H"].notna()].copy()


train["RISK_24H"] = train["RISK_24H"].astype(int)

valid["RISK_24H"] = valid["RISK_24H"].astype(int)


print()

print("=" * 80)
print("LABEL CHECK")
print("=" * 80)

print("Train usable:", train.shape)

print(train["RISK_24H"].value_counts())

print()

print("Validation usable:", valid.shape)

print(valid["RISK_24H"].value_counts())


# ============================================================
# 3. 제외 컬럼
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
# 5. TREND_PER_H Feature
#
# V2.7에서는 전부 제외
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
# 6. 전체 Mean Diff 12개
# ============================================================

all_mean_diff_features = [
    # --------------------------------------------------------
    # CONTROL_DELAY
    # --------------------------------------------------------
    "CONTROL_DELAY_1H_MINUS_3H_MEAN",
    "CONTROL_DELAY_3H_MINUS_6H_MEAN",
    "CONTROL_DELAY_6H_MINUS_12H_MEAN",
    # --------------------------------------------------------
    # MLD-LVL
    # --------------------------------------------------------
    "MLD-LVL_1H_MINUS_3H_MEAN",
    "MLD-LVL_3H_MINUS_6H_MEAN",
    "MLD-LVL_6H_MINUS_12H_MEAN",
    # --------------------------------------------------------
    # OSC-STK
    # --------------------------------------------------------
    "OSC-STK_1H_MINUS_3H_MEAN",
    "OSC-STK_3H_MINUS_6H_MEAN",
    "OSC-STK_6H_MINUS_12H_MEAN",
    # --------------------------------------------------------
    # OSC-FRQ
    # --------------------------------------------------------
    "OSC-FRQ_1H_MINUS_3H_MEAN",
    "OSC-FRQ_3H_MINUS_6H_MEAN",
    "OSC-FRQ_6H_MINUS_12H_MEAN",
]


# ============================================================
# 7. V2.7에서 유지할 Mean Diff 9개
#
# OSC-FRQ 3개 제외
# ============================================================

selected_mean_diff_features = [
    # CONTROL_DELAY
    "CONTROL_DELAY_1H_MINUS_3H_MEAN",
    "CONTROL_DELAY_3H_MINUS_6H_MEAN",
    "CONTROL_DELAY_6H_MINUS_12H_MEAN",
    # MLD-LVL
    "MLD-LVL_1H_MINUS_3H_MEAN",
    "MLD-LVL_3H_MINUS_6H_MEAN",
    "MLD-LVL_6H_MINUS_12H_MEAN",
    # OSC-STK
    "OSC-STK_1H_MINUS_3H_MEAN",
    "OSC-STK_3H_MINUS_6H_MEAN",
    "OSC-STK_6H_MINUS_12H_MEAN",
]


# ============================================================
# 8. 제거 Mean Diff
# ============================================================

drop_mean_diff_features = [
    col for col in all_mean_diff_features if col not in selected_mean_diff_features
]


# ============================================================
# 9. BASE 240
# ============================================================

base_features = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE_COLS
        and col not in DROP_MAINTENANCE_COLS
        and col not in trend_per_hour_features
        and col not in all_mean_diff_features
        and pd.api.types.is_numeric_dtype(train[col])
    )
]


# ============================================================
# 10. V2.7 Feature
# ============================================================

v27_features = base_features + selected_mean_diff_features


print()

print("=" * 80)
print("FEATURE CHECK V2.7")
print("=" * 80)


print("BASE Feature 수:", len(base_features))


print("추가 Mean Diff:", len(selected_mean_diff_features))


print("V2.7 Feature 수:", len(v27_features))


print()

print("유지 Mean Diff:")


for col in selected_mean_diff_features:

    print(f"{col:<45}", "→", col in train.columns)


print()

print("제외 Mean Diff:")


for col in drop_mean_diff_features:

    print(f"{col:<45}", "→ 제외")


if len(base_features) != 240:

    print("[WARNING] BASE 예상 240개")


if len(v27_features) != 249:

    print("[WARNING] V2.7 예상 249개")


# ============================================================
# 11. y / groups
# ============================================================

y = train["RISK_24H"].copy()


y_valid = valid["RISK_24H"].copy()


groups = train["SCENARIO_ID"].copy()


# ============================================================
# 12. Class imbalance
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


print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 13. XGBoost 생성
# ============================================================


def make_xgb_model():

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


# ============================================================
# 14. Threshold 평가
# ============================================================


def evaluate_threshold(y_true, probability, threshold):

    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, pred, zero_division=0)

    recall = recall_score(y_true, pred, zero_division=0)

    f1 = f1_score(y_true, pred, zero_division=0)

    f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)

    return {
        "THRESHOLD": threshold,
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
# 15. 실험 구성
# ============================================================

EXPERIMENTS = {
    "BASE_240": base_features,
    "MEAN_DIFF_9_249": v27_features,
}


# ============================================================
# 16. CV
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


experiment_results = {}

fold_rows = []


# ============================================================
# 17. 실험별 CV
# ============================================================

for experiment_name, feature_cols in EXPERIMENTS.items():

    print()

    print("=" * 80)

    print(f"{experiment_name} - 5 FOLD CV")

    print("=" * 80)

    X = train[feature_cols].copy()

    oof_probability = np.full(len(train), np.nan, dtype=float)

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        X_fold_train = X.iloc[train_idx].copy()

        y_fold_train = y.iloc[train_idx].copy()

        X_fold_valid = X.iloc[fold_valid_idx].copy()

        y_fold_valid = y.iloc[fold_valid_idx].copy()

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        overlap = train_groups & valid_groups

        if overlap:

            raise RuntimeError(
                f"{experiment_name} " f"Fold {fold}: " f"Scenario leakage 발생"
            )

        # ====================================================
        # Imputer
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_fold_train)

        X_fold_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # 모델
        # ====================================================

        model = make_xgb_model()

        model.fit(X_train_imp, y_fold_train)

        # ====================================================
        # 예측
        # ====================================================

        fold_probability = model.predict_proba(X_fold_valid_imp)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # 평가
        # ====================================================

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "EXPERIMENT": experiment_name,
                "FOLD": fold,
                "FEATURE_COUNT": len(feature_cols),
                "TRAIN_SCENARIOS": len(train_groups),
                "VALID_SCENARIOS": len(valid_groups),
                "VALID_POSITIVE": int(y_fold_valid.sum()),
                "ROC_AUC": fold_roc,
                "PR_AUC": fold_pr,
            }
        )

        print(
            f"Fold {fold}"
            f" | Feature "
            f"{len(feature_cols)}"
            f" | PR-AUC "
            f"{fold_pr:.4f}"
        )

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{experiment_name}: " f"OOF NaN 발생")

    # ========================================================
    # OOF 전체 평가
    # ========================================================

    oof_roc = roc_auc_score(y, oof_probability)

    oof_pr = average_precision_score(y, oof_probability)

    print()

    print(f"{experiment_name} " f"OOF ROC-AUC: " f"{oof_roc:.4f}")

    print(f"{experiment_name} " f"OOF PR-AUC : " f"{oof_pr:.4f}")

    experiment_results[experiment_name] = {
        "feature_cols": feature_cols,
        "oof_probability": oof_probability,
        "roc_auc": oof_roc,
        "pr_auc": oof_pr,
    }


# ============================================================
# 18. Ranking
# ============================================================

ranking_rows = []


for experiment_name, result in experiment_results.items():

    ranking_rows.append(
        {
            "EXPERIMENT": experiment_name,
            "FEATURE_COUNT": len(result["feature_cols"]),
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
print("V2.7 OOF RANKING")
print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 19. Best Experiment
# ============================================================

best_experiment_name = str(ranking_df.iloc[0]["EXPERIMENT"])


best_features = experiment_results[best_experiment_name]["feature_cols"]


best_oof_probability = experiment_results[best_experiment_name]["oof_probability"]


print()

print("선택 실험:", best_experiment_name)


print("선택 Feature 수:", len(best_features))


# ============================================================
# 20. OOF Threshold 탐색
# ============================================================

threshold_rows = []


for threshold in np.arange(0.05, 0.91, 0.01):

    threshold = round(float(threshold), 2)

    result = evaluate_threshold(
        y_true=y, probability=best_oof_probability, threshold=threshold
    )

    threshold_rows.append(result)


threshold_df = pd.DataFrame(threshold_rows)


best_threshold_idx = threshold_df["F2"].idxmax()


best_threshold_row = threshold_df.loc[best_threshold_idx]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)
print("OOF BEST THRESHOLD V2.7")
print("=" * 80)


print(
    best_threshold_row[
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
    .to_string()
)


# ============================================================
# 21. LOCK
# ============================================================

print()

print("=" * 80)
print("V2.7 LOCK")
print("=" * 80)


print("Model: XGBoost")


print("Experiment:", best_experiment_name)


print("Feature Count:", len(best_features))


print("Threshold:", best_threshold)


# ============================================================
# 22. Full Train
# ============================================================

X_train_final = train[best_features].copy()


X_valid_final = valid[best_features].copy()


final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X_train_final)


X_valid_imp = final_imputer.transform(X_valid_final)


final_model = make_xgb_model()


final_model.fit(X_train_imp, y)


# ============================================================
# 23. Validation
# ============================================================

valid_probability = final_model.predict_proba(X_valid_imp)[:, 1]


valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


print()

print("=" * 80)
print("FINAL VALIDATION V2.7")
print("=" * 80)


print("Experiment:", best_experiment_name)


print("Feature Count:", len(best_features))


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
# 24. V2.3 / V2.6 / V2.7 비교
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "VERSION": "V2.3_BASE_240",
            "FEATURE_COUNT": 240,
            "OOF_PR_AUC": 0.8094,
            "THRESHOLD": 0.14,
            "VAL_PR_AUC": 0.7797,
            "PRECISION": 0.5859,
            "RECALL": 0.8286,
            "F1": 0.6864,
            "F2": 0.7652,
            "TN": 528,
            "FP": 41,
            "FN": 12,
            "TP": 58,
        },
        {
            "VERSION": "V2.6_MEAN_DIFF_12",
            "FEATURE_COUNT": 252,
            "OOF_PR_AUC": 0.8164,
            "THRESHOLD": 0.14,
            "VAL_PR_AUC": 0.7825,
            "PRECISION": 0.6333,
            "RECALL": 0.8143,
            "F1": 0.7125,
            "F2": 0.7703,
            "TN": 536,
            "FP": 33,
            "FN": 13,
            "TP": 57,
        },
        {
            "VERSION": "V2.7",
            "FEATURE_COUNT": len(best_features),
            "OOF_PR_AUC": experiment_results[best_experiment_name]["pr_auc"],
            "THRESHOLD": best_threshold,
            "VAL_PR_AUC": valid_pr_auc,
            "PRECISION": valid_metrics["PRECISION"],
            "RECALL": valid_metrics["RECALL"],
            "F1": valid_metrics["F1"],
            "F2": valid_metrics["F2"],
            "TN": valid_metrics["TN"],
            "FP": valid_metrics["FP"],
            "FN": valid_metrics["FN"],
            "TP": valid_metrics["TP"],
        },
    ]
)


print()

print("=" * 80)
print("V2.3 VS V2.6 VS V2.7")
print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 25. Feature Importance
# ============================================================

importance_df = pd.DataFrame(
    {
        "FEATURE": best_features,
        "IMPORTANCE": final_model.feature_importances_,
    }
)


importance_df = importance_df.sort_values("IMPORTANCE", ascending=False).reset_index(
    drop=True
)


print()

print("=" * 80)
print("TOP 50 FEATURE IMPORTANCE V2.7")
print("=" * 80)


print(importance_df.head(50).round(6).to_string(index=False))


# ============================================================
# 26. Mean Diff 9 Importance
# ============================================================

mean_diff_importance_df = importance_df[
    importance_df["FEATURE"].isin(selected_mean_diff_features)
].copy()


print()

print("=" * 80)
print("SELECTED MEAN DIFF IMPORTANCE")
print("=" * 80)


print(mean_diff_importance_df.round(6).to_string(index=False))


# ============================================================
# 27. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


ranking_df.to_csv(
    OUTPUT_DIR / "casting_24h_v27_oof_ranking.csv", index=False, encoding="utf-8-sig"
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_24h_v27_folds.csv", index=False, encoding="utf-8-sig"
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_24h_v27_thresholds.csv", index=False, encoding="utf-8-sig"
)


comparison_df.to_csv(
    OUTPUT_DIR / "casting_24h_v23_v26_v27_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_v27_importance.csv", index=False, encoding="utf-8-sig"
)


mean_diff_importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_v27_mean_diff_importance.csv",
    index=False,
    encoding="utf-8-sig",
)


print()

print("=" * 80)
print("SAVE COMPLETE V2.7")
print("=" * 80)
