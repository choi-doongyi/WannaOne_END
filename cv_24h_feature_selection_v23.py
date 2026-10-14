# ============================================================
# cv_24h_feature_selection_v23.py
#
# V2.3
#
# Sensor 237
# +
# Maintenance 3
#
# = Candidate Feature 240개
#
# Feature 축소 실험:
# - ALL 240
# - TOP 100
# - TOP 50
# - TOP 30
#
# 중요:
# 각 Fold의 Train 데이터 안에서만
# Feature Importance를 계산하고 Top K 선택
#
# 따라서 Fold Validation 정보는
# Feature 선택에 사용되지 않음
#
# Model:
# XGBoost
#
# Model version 선택:
# Train OOF PR-AUC
#
# Threshold 선택:
# Train OOF F2
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


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


VALID_PATH = OUTPUT_DIR / "casting_validation_features_v22.csv"


# ============================================================
# 1. 데이터 읽기
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)

print("24H FEATURE SELECTION V2.3")

print("SENSOR + SELECTED MAINTENANCE")

print("=" * 80)


print()

print("Train:", train_df.shape)


print("Validation:", valid_df.shape)


# ============================================================
# 2. 24H Label
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
# 4. V2.2 정비 Feature 11개
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
# 5. V2.3에서 유지할 정비 Feature
# ============================================================

SELECTED_MAINTENANCE_COLS = [
    "MNT_DOWN_SUM",
    "LAST_DOWN_MIN",
    "LAST_MNT_IS_PARTIAL",
]


# ============================================================
# 6. 제거할 정비 Feature
# ============================================================

DROP_MAINTENANCE_COLS = [
    col for col in ALL_MAINTENANCE_COLS if col not in SELECTED_MAINTENANCE_COLS
]


# ============================================================
# 7. Candidate Feature
# ============================================================

candidate_features = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE_COLS
        and col not in DROP_MAINTENANCE_COLS
        and pd.api.types.is_numeric_dtype(train[col])
    )
]


print()

print("=" * 80)

print("V2.3 FEATURE POOL")

print("=" * 80)


print("Candidate Feature 수:", len(candidate_features))


print()

print("정비 Feature 유지:")


for col in SELECTED_MAINTENANCE_COLS:

    print(f"{col:<25}", "→", col in candidate_features)


print()

print("정비 Feature 제외:")


for col in DROP_MAINTENANCE_COLS:

    print(f"{col:<25}", "→", col not in candidate_features)


if len(candidate_features) != 240:

    print()

    print(
        "[WARNING] 예상 Candidate Feature는 " "240개인데 실제는",
        len(candidate_features),
        "개입니다.",
    )


# ============================================================
# 8. X / y / groups
# ============================================================

X = train[candidate_features].copy()


y = train["RISK_24H"].copy()


groups = train["SCENARIO_ID"].copy()


X_valid = valid[candidate_features].copy()


y_valid = valid["RISK_24H"].copy()


# ============================================================
# 9. Class imbalance
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
# 10. XGBoost 생성 함수
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
# 12. Feature 개수 실험
# ============================================================

FEATURE_COUNTS = [
    240,
    100,
    50,
    30,
]


# ============================================================
# 13. Scenario CV
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 14. 결과 저장용
# ============================================================

experiment_results = {}

fold_rows = []

feature_frequency_rows = []


# ============================================================
# 15. Feature Count별 CV
# ============================================================

for top_k in FEATURE_COUNTS:

    print()

    print("=" * 80)

    print(f"TOP {top_k} FEATURE CV")

    print("=" * 80)

    oof_probability = np.full(len(train), np.nan, dtype=float)

    selected_feature_counter = {feature: 0 for feature in candidate_features}

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        # ====================================================
        # Fold Train / Valid
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
                f"TOP {top_k} " f"Fold {fold}: " f"Scenario leakage 발생"
            )

        # ====================================================
        # Fold Train에서만 Imputer 학습
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_fold_train)

        X_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # ALL 240은 Feature 선택 없이 그대로
        # ====================================================

        if top_k >= len(candidate_features):

            selected_features = candidate_features.copy()

            selected_indices = np.arange(len(candidate_features))

        else:

            # =================================================
            # 중요:
            #
            # Feature selector도
            # Fold Train만 사용
            # =================================================

            selector_model = make_xgb_model()

            selector_model.fit(X_train_imp, y_fold_train)

            importances = selector_model.feature_importances_

            ranking = np.argsort(importances)[::-1]

            selected_indices = ranking[:top_k]

            selected_features = [candidate_features[idx] for idx in selected_indices]

        # ====================================================
        # Feature 선택 빈도 저장
        # ====================================================

        for feature in selected_features:

            selected_feature_counter[feature] += 1

        # ====================================================
        # 선택된 Feature만 사용
        # ====================================================

        X_train_selected = X_train_imp[:, selected_indices]

        X_valid_selected = X_valid_imp[:, selected_indices]

        # ====================================================
        # 최종 Fold 모델
        # ====================================================

        model = make_xgb_model()

        model.fit(X_train_selected, y_fold_train)

        fold_probability = model.predict_proba(X_valid_selected)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # Fold 성능
        # ====================================================

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "TOP_K": top_k,
                "FOLD": fold,
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
            f"{len(selected_features)}"
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
    # OOF 완성 확인
    # ========================================================

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"TOP {top_k}: " f"OOF NaN 발생")

    # ========================================================
    # 전체 OOF 성능
    # ========================================================

    oof_roc = roc_auc_score(y, oof_probability)

    oof_pr = average_precision_score(y, oof_probability)

    print()

    print(f"TOP {top_k} " f"OOF ROC-AUC: " f"{oof_roc:.4f}")

    print(f"TOP {top_k} " f"OOF PR-AUC : " f"{oof_pr:.4f}")

    experiment_results[top_k] = {
        "oof_probability": oof_probability,
        "roc_auc": oof_roc,
        "pr_auc": oof_pr,
        "feature_counter": selected_feature_counter,
    }

    # ========================================================
    # Feature 선택 빈도
    # ========================================================

    for feature, count in selected_feature_counter.items():

        if count > 0:

            feature_frequency_rows.append(
                {
                    "TOP_K": top_k,
                    "FEATURE": feature,
                    "SELECTED_FOLDS": count,
                }
            )


# ============================================================
# 16. Feature Count Ranking
# ============================================================

ranking_rows = []


for top_k, result in experiment_results.items():

    ranking_rows.append(
        {
            "TOP_K": top_k,
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

print("FEATURE COUNT RANKING")

print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 17. Best Feature Count
# ============================================================

best_top_k = int(ranking_df.iloc[0]["TOP_K"])


best_oof_probability = experiment_results[best_top_k]["oof_probability"]


print()

print("선택 Feature 수:", best_top_k)


# ============================================================
# 18. OOF Threshold 탐색
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

print("OOF BEST THRESHOLD V2.3")

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
# 19. Model/Feature/Threshold LOCK
# ============================================================

print()

print("=" * 80)

print("V2.3 LOCK")

print("=" * 80)


print("Model: XGBoost")


print("Feature Count:", best_top_k)


print("Threshold:", best_threshold)


print()

print("이제 전체 Train으로 최종 Feature를 선정합니다.")


# ============================================================
# 20. Full Train Imputer
# ============================================================

final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X)


X_validation_imp = final_imputer.transform(X_valid)


# ============================================================
# 21. Full Train에서 최종 Feature ranking
#
# Validation은 사용하지 않음
# ============================================================

if best_top_k >= len(candidate_features):

    final_selected_indices = np.arange(len(candidate_features))

    final_selected_features = candidate_features.copy()


else:

    selector_model = make_xgb_model()

    selector_model.fit(X_train_imp, y)

    final_importances = selector_model.feature_importances_

    final_ranking = np.argsort(final_importances)[::-1]

    final_selected_indices = final_ranking[:best_top_k]

    final_selected_features = [
        candidate_features[idx] for idx in final_selected_indices
    ]


# ============================================================
# 22. 최종 Feature 출력
# ============================================================

print()

print("=" * 80)

print("FINAL SELECTED FEATURES")

print("=" * 80)


for i, feature in enumerate(final_selected_features, start=1):

    print(f"{i:3d}. " f"{feature}")


# ============================================================
# 23. 최종 모델 학습
# ============================================================

X_train_selected = X_train_imp[:, final_selected_indices]


X_validation_selected = X_validation_imp[:, final_selected_indices]


final_model = make_xgb_model()


final_model.fit(X_train_selected, y)


# ============================================================
# 24. Validation Probability
# ============================================================

valid_probability = final_model.predict_proba(X_validation_selected)[:, 1]


# ============================================================
# 25. Validation 성능
# ============================================================

valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


print()

print("=" * 80)

print("FINAL VALIDATION V2.3")

print("=" * 80)


print("Feature Count:", best_top_k)


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
# 26. V1 / V2.2 / V2.3 비교
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "VERSION": "V1_SENSOR",
            "FEATURE_COUNT": 237,
            "THRESHOLD": 0.10,
            "PR_AUC": 0.7780,
            "PRECISION": 0.5481,
            "RECALL": 0.8143,
            "F1": 0.6552,
            "F2": 0.7422,
            "TN": 522,
            "FP": 47,
            "FN": 13,
            "TP": 57,
        },
        {
            "VERSION": "V2.2_LAST_MNT",
            "FEATURE_COUNT": 248,
            "THRESHOLD": 0.10,
            "PR_AUC": 0.7758,
            "PRECISION": 0.5743,
            "RECALL": 0.8286,
            "F1": 0.6784,
            "F2": 0.7612,
            "TN": 526,
            "FP": 43,
            "FN": 12,
            "TP": 58,
        },
        {
            "VERSION": "V2.3_FEATURE_SELECTION",
            "FEATURE_COUNT": best_top_k,
            "THRESHOLD": best_threshold,
            "PR_AUC": valid_pr_auc,
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

print("V1 VS V2.2 VS V2.3")

print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 27. Fold에서 자주 선택된 Feature
# ============================================================

frequency_df = pd.DataFrame(feature_frequency_rows)


best_frequency_df = (
    frequency_df[frequency_df["TOP_K"] == best_top_k]
    .sort_values(
        [
            "SELECTED_FOLDS",
            "FEATURE",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .reset_index(drop=True)
)


print()

print("=" * 80)

print("BEST K - FEATURE SELECTION FREQUENCY")

print("=" * 80)


print(best_frequency_df.head(50).to_string(index=False))


# ============================================================
# 28. 최종 Feature Importance
# ============================================================

final_importance_df = pd.DataFrame(
    {
        "FEATURE": final_selected_features,
        "IMPORTANCE": final_model.feature_importances_,
    }
)


final_importance_df = final_importance_df.sort_values(
    "IMPORTANCE", ascending=False
).reset_index(drop=True)


print()

print("=" * 80)

print("FINAL MODEL FEATURE IMPORTANCE")

print("=" * 80)


print(final_importance_df.head(50).round(6).to_string(index=False))


# ============================================================
# 29. 정비 Feature 확인
# ============================================================

maintenance_final_df = final_importance_df[
    final_importance_df["FEATURE"].isin(SELECTED_MAINTENANCE_COLS)
].copy()


print()

print("=" * 80)

print("FINAL MAINTENANCE FEATURES")

print("=" * 80)


if maintenance_final_df.empty:

    print("최종 선택 Feature에 " "정비 Feature가 없습니다.")


else:

    print(maintenance_final_df.round(6).to_string(index=False))


# ============================================================
# 30. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


selected_features_df = pd.DataFrame(
    {
        "RANK": np.arange(1, len(final_selected_features) + 1),
        "FEATURE": final_selected_features,
    }
)


ranking_df.to_csv(
    OUTPUT_DIR / "casting_24h_feature_count_ranking_v23.csv",
    index=False,
    encoding="utf-8-sig",
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_24h_feature_selection_folds_v23.csv",
    index=False,
    encoding="utf-8-sig",
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_24h_thresholds_v23.csv", index=False, encoding="utf-8-sig"
)


comparison_df.to_csv(
    OUTPUT_DIR / "casting_24h_v1_v22_v23_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)


frequency_df.to_csv(
    OUTPUT_DIR / "casting_24h_feature_frequency_v23.csv",
    index=False,
    encoding="utf-8-sig",
)


selected_features_df.to_csv(
    OUTPUT_DIR / "casting_24h_selected_features_v23.csv",
    index=False,
    encoding="utf-8-sig",
)


final_importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_final_importance_v23.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 31. 완료
# ============================================================

print()

print("=" * 80)

print("SAVE COMPLETE V2.3")

print("=" * 80)


print()

print(OUTPUT_DIR / "casting_24h_feature_count_ranking_v23.csv")


print(OUTPUT_DIR / "casting_24h_selected_features_v23.csv")


print(OUTPUT_DIR / "casting_24h_final_importance_v23.csv")
