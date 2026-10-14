# ============================================================
# cv_24h_fair_v25.py
#
# V2.5
#
# 비교:
# 1. BASE_240
#    - 센서 237
#    - 정비 3
#
# 2. TREND_268
#    - BASE_240
#    - 추세 Feature 28개 추가
#
# Model:
# XGBoost
#
# CV:
# StratifiedGroupKFold 5-Fold
# Scenario 단위 분리
#
# 선택 기준:
# OOF PR-AUC
#
# Threshold:
# OOF F2 최대
#
# Validation:
# 마지막 평가
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

print("24H FAIR MODEL V2.5")

print("BASE 240 VS TREND 268")

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
# 4. V2.2 정비 전체
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


# ============================================================
# 5. V2.3에서 유지한 정비 3개
# ============================================================

SELECTED_MAINTENANCE_COLS = [
    "MNT_DOWN_SUM",
    "LAST_DOWN_MIN",
    "LAST_MNT_IS_PARTIAL",
]


DROP_MAINTENANCE_COLS = [
    col for col in ALL_MAINTENANCE_COLS if col not in SELECTED_MAINTENANCE_COLS
]


# ============================================================
# 6. V2.5 추세 Feature 28개
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


trend_features = []


for sensor in TREND_SENSORS:

    # --------------------------------------------------------
    # 시간당 변화량
    # --------------------------------------------------------

    for window in WINDOWS:

        trend_features.append(f"{window}_" f"{sensor}_TREND_PER_H")

    # --------------------------------------------------------
    # 구간 평균 차이
    # --------------------------------------------------------

    trend_features.extend(
        [
            (f"{sensor}_" f"1H_MINUS_3H_MEAN"),
            (f"{sensor}_" f"3H_MINUS_6H_MEAN"),
            (f"{sensor}_" f"6H_MINUS_12H_MEAN"),
        ]
    )


# ============================================================
# 7. BASE 240 Feature
# ============================================================

base_features = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE_COLS
        and col not in DROP_MAINTENANCE_COLS
        and col not in trend_features
        and pd.api.types.is_numeric_dtype(train[col])
    )
]


# ============================================================
# 8. TREND 268 Feature
# ============================================================

trend_model_features = base_features + trend_features


print()

print("=" * 80)

print("FEATURE CHECK")

print("=" * 80)


print("BASE Feature 수:", len(base_features))


print("Trend 추가 Feature 수:", len(trend_features))


print("TREND 모델 Feature 수:", len(trend_model_features))


if len(base_features) != 240:

    print("[WARNING] BASE 예상 240개")


if len(trend_model_features) != 268:

    print("[WARNING] TREND 예상 268개")


# ============================================================
# 9. 추세 Feature 존재 확인
# ============================================================

print()

print("Trend Feature check")


for col in trend_features:

    print(f"{col:<45}", "→", col in train.columns)


# ============================================================
# 10. y / groups
# ============================================================

y = train["RISK_24H"].copy()


y_valid = valid["RISK_24H"].copy()


groups = train["SCENARIO_ID"].copy()


# ============================================================
# 11. 불균형 비율
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
# 12. 모델 생성 함수
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
# 13. Threshold 평가
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
# 14. 실험 구성
# ============================================================

EXPERIMENTS = {
    "BASE_240": base_features,
    "TREND_268": trend_model_features,
}


# ============================================================
# 15. CV
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 16. 결과 저장
# ============================================================

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

        # ====================================================
        # Fold 데이터
        # ====================================================

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
        # Fold Train만 학습
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_fold_train)

        X_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # 모델
        # ====================================================

        model = make_xgb_model()

        model.fit(X_train_imp, y_fold_train)

        # ====================================================
        # Fold 예측
        # ====================================================

        fold_probability = model.predict_proba(X_valid_imp)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # Fold 평가
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
            f" | Train Scenario "
            f"{len(train_groups)}"
            f" | Valid Scenario "
            f"{len(valid_groups)}"
            f" | Positive "
            f"{int(y_fold_valid.sum())}"
            f" | PR-AUC "
            f"{fold_pr:.4f}"
        )

    # ========================================================
    # OOF 검사
    # ========================================================

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{experiment_name}: " f"OOF NaN 발생")

    # ========================================================
    # OOF 전체 성능
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
# 18. 실험 Ranking
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

print("V2.5 OOF RANKING")

print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 19. Best 실험 선택
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

print("OOF BEST THRESHOLD V2.5")

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

print("V2.5 LOCK")

print("=" * 80)


print("Model: XGBoost")


print("Experiment:", best_experiment_name)


print("Feature Count:", len(best_features))


print("Threshold:", best_threshold)


# ============================================================
# 22. 전체 Train 재학습
# ============================================================

X_train_final = train[best_features].copy()


X_valid_final = valid[best_features].copy()


final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X_train_final)


X_valid_imp = final_imputer.transform(X_valid_final)


final_model = make_xgb_model()


final_model.fit(X_train_imp, y)


# ============================================================
# 23. Validation 예측
# ============================================================

valid_probability = final_model.predict_proba(X_valid_imp)[:, 1]


# ============================================================
# 24. Validation 평가
# ============================================================

valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


print()

print("=" * 80)

print("FINAL VALIDATION V2.5")

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
# 25. V2.3 vs V2.5 비교
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
            "VERSION": "V2.5",
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

print("V2.3 VS V2.5")

print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 26. Feature Importance
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

print("TOP 50 FEATURE IMPORTANCE V2.5")

print("=" * 80)


print(importance_df.head(50).round(6).to_string(index=False))


# ============================================================
# 27. 새 Trend Feature Importance
# ============================================================

trend_importance_df = importance_df[
    importance_df["FEATURE"].isin(trend_features)
].copy()


print()

print("=" * 80)

print("TREND FEATURE IMPORTANCE")

print("=" * 80)


if trend_importance_df.empty:

    print("선택 모델에 Trend Feature 없음")


else:

    print(trend_importance_df.round(6).to_string(index=False))


# ============================================================
# 28. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


ranking_df.to_csv(
    OUTPUT_DIR / "casting_24h_v25_oof_ranking.csv", index=False, encoding="utf-8-sig"
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_24h_v25_folds.csv", index=False, encoding="utf-8-sig"
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_24h_v25_thresholds.csv", index=False, encoding="utf-8-sig"
)


comparison_df.to_csv(
    OUTPUT_DIR / "casting_24h_v23_v25_comparison.csv", index=False, encoding="utf-8-sig"
)


importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_v25_importance.csv", index=False, encoding="utf-8-sig"
)


trend_importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_v25_trend_importance.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 29. 완료
# ============================================================

print()

print("=" * 80)

print("SAVE COMPLETE V2.5")

print("=" * 80)
