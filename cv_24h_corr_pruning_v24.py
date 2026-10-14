# ============================================================
# cv_24h_corr_pruning_v24.py
#
# V2.4 - Correlation Pruning
#
# Candidate:
# Sensor 237
# +
# Maintenance 3
#
# = 240 Features
#
# 비교:
# - NONE : 상관 제거 없음
# - 0.98
# - 0.95
# - 0.90
#
# 중요:
# 각 Fold의 Train 데이터 안에서만
# 1. Imputer 학습
# 2. 상관계수 계산
# 3. 중복 Feature 제거
#
# Fold Validation은 Feature 선택에 사용하지 않음
#
# 모델:
# XGBoost
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


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


VALID_PATH = OUTPUT_DIR / "casting_validation_features_v22.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)

print("24H CORRELATION PRUNING V2.4")

print("SENSOR + SELECTED MAINTENANCE")

print("=" * 80)


print()

print("Train:", train_df.shape)


print("Validation:", valid_df.shape)


# ============================================================
# 2. 24H 유효 Label
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
# 4. V2.2 정비 Feature 전체
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
# 5. V2.3에서 유지한 정비 Feature
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
# 6. Candidate Feature
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

print("V2.4 FEATURE POOL")

print("=" * 80)


print("Candidate Feature 수:", len(candidate_features))


print()

print("유지 정비 Feature:")


for col in SELECTED_MAINTENANCE_COLS:

    print(f"{col:<25}", "→", col in candidate_features)


if len(candidate_features) != 240:

    print()

    print("[WARNING] 예상 240개인데 실제는", len(candidate_features))


# ============================================================
# 7. X / y / group
# ============================================================

X = train[candidate_features].copy()


y = train["RISK_24H"].copy()


groups = train["SCENARIO_ID"].copy()


X_valid = valid[candidate_features].copy()


y_valid = valid["RISK_24H"].copy()


# ============================================================
# 8. Class imbalance
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
# 9. XGBoost 생성
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
# 10. Threshold 평가
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
# 11. 상관 높은 Feature 제거 함수
#
# threshold=None이면 제거 안 함
#
# 주의:
# 이 함수에 들어오는 데이터는
# 반드시 해당 Fold Train 데이터여야 함
# ============================================================


def find_correlated_features(X_train_df, threshold):

    # --------------------------------------------------------
    # NONE
    # --------------------------------------------------------

    if threshold is None:

        return []

    # --------------------------------------------------------
    # 절대 상관계수
    # --------------------------------------------------------

    corr_matrix = X_train_df.corr().abs()

    # --------------------------------------------------------
    # 상삼각 행렬만 사용
    #
    # 같은 쌍을 두 번 검사하지 않기 위함
    # --------------------------------------------------------

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # --------------------------------------------------------
    # threshold보다 큰 상관관계가 하나라도 있으면
    # 뒤쪽 Feature를 제거
    # --------------------------------------------------------

    drop_cols = [
        column for column in upper.columns if (upper[column] > threshold).any()
    ]

    return drop_cols


# ============================================================
# 12. 실험할 상관계수 기준
# ============================================================

CORR_CONFIGS = {
    "NONE": None,
    "CORR_098": 0.98,
    "CORR_095": 0.95,
    "CORR_090": 0.90,
}


# ============================================================
# 13. Scenario CV
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 14. 결과 저장용
# ============================================================

experiment_results = {}

fold_rows = []

drop_frequency_rows = []


# ============================================================
# 15. 상관 제거 기준별 CV
# ============================================================

for config_name, corr_threshold in CORR_CONFIGS.items():

    print()

    print("=" * 80)

    print(f"{config_name} - 5 FOLD CV")

    print("=" * 80)

    oof_probability = np.full(len(train), np.nan, dtype=float)

    drop_counter = {feature: 0 for feature in candidate_features}

    feature_counts = []

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        # ====================================================
        # Fold 분리
        # ====================================================

        X_fold_train = X.iloc[train_idx].copy()

        y_fold_train = y.iloc[train_idx].copy()

        X_fold_valid = X.iloc[fold_valid_idx].copy()

        y_fold_valid = y.iloc[fold_valid_idx].copy()

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        # ====================================================
        # Scenario leakage 검사
        # ====================================================

        overlap = train_groups & valid_groups

        if overlap:

            raise RuntimeError(
                f"{config_name} " f"Fold {fold}: " f"Scenario leakage 발생"
            )

        # ====================================================
        # Imputer는 Fold Train에서만 학습
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_fold_train_imp = imputer.fit_transform(X_fold_train)

        X_fold_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # DataFrame 복원
        # ====================================================

        X_fold_train_imp_df = pd.DataFrame(
            X_fold_train_imp, columns=candidate_features, index=X_fold_train.index
        )

        X_fold_valid_imp_df = pd.DataFrame(
            X_fold_valid_imp, columns=candidate_features, index=X_fold_valid.index
        )

        # ====================================================
        # Fold Train에서만 상관관계 계산
        # ====================================================

        drop_cols = find_correlated_features(X_fold_train_imp_df, corr_threshold)

        selected_features = [col for col in candidate_features if col not in drop_cols]

        if len(selected_features) == 0:

            raise RuntimeError(
                f"{config_name} " f"Fold {fold}: " f"모든 Feature 제거됨"
            )

        # ====================================================
        # 제거 빈도 저장
        # ====================================================

        for col in drop_cols:

            drop_counter[col] += 1

        feature_counts.append(len(selected_features))

        # ====================================================
        # 선택 Feature만 사용
        # ====================================================

        X_fold_train_selected = X_fold_train_imp_df[selected_features]

        X_fold_valid_selected = X_fold_valid_imp_df[selected_features]

        # ====================================================
        # 모델
        # ====================================================

        model = make_xgb_model()

        model.fit(X_fold_train_selected, y_fold_train)

        fold_probability = model.predict_proba(X_fold_valid_selected)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # Fold 성능
        # ====================================================

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "CONFIG": config_name,
                "CORR_THRESHOLD": (
                    np.nan if corr_threshold is None else corr_threshold
                ),
                "FOLD": fold,
                "FEATURE_COUNT": len(selected_features),
                "DROPPED_COUNT": len(drop_cols),
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
            f" | Drop "
            f"{len(drop_cols)}"
            f" | PR-AUC "
            f"{fold_pr:.4f}"
        )

    # ========================================================
    # OOF 검사
    # ========================================================

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{config_name}: " f"OOF NaN 발생")

    # ========================================================
    # OOF 전체 성능
    # ========================================================

    oof_roc = roc_auc_score(y, oof_probability)

    oof_pr = average_precision_score(y, oof_probability)

    mean_feature_count = float(np.mean(feature_counts))

    min_feature_count = int(np.min(feature_counts))

    max_feature_count = int(np.max(feature_counts))

    print()

    print(f"{config_name} " f"OOF ROC-AUC: " f"{oof_roc:.4f}")

    print(f"{config_name} " f"OOF PR-AUC : " f"{oof_pr:.4f}")

    print(f"{config_name} " f"평균 Feature 수: " f"{mean_feature_count:.1f}")

    experiment_results[config_name] = {
        "corr_threshold": corr_threshold,
        "oof_probability": oof_probability,
        "roc_auc": oof_roc,
        "pr_auc": oof_pr,
        "mean_feature_count": mean_feature_count,
        "min_feature_count": min_feature_count,
        "max_feature_count": max_feature_count,
        "drop_counter": drop_counter,
    }

    # ========================================================
    # 제거 Feature 빈도
    # ========================================================

    for feature, count in drop_counter.items():

        if count > 0:

            drop_frequency_rows.append(
                {
                    "CONFIG": config_name,
                    "FEATURE": feature,
                    "DROPPED_FOLDS": count,
                }
            )


# ============================================================
# 16. OOF Ranking
# ============================================================

ranking_rows = []


for config_name, result in experiment_results.items():

    ranking_rows.append(
        {
            "CONFIG": config_name,
            "CORR_THRESHOLD": (
                np.nan if result["corr_threshold"] is None else result["corr_threshold"]
            ),
            "MEAN_FEATURE_COUNT": result["mean_feature_count"],
            "MIN_FEATURE_COUNT": result["min_feature_count"],
            "MAX_FEATURE_COUNT": result["max_feature_count"],
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

print("CORRELATION PRUNING RANKING")

print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 17. Best Config
# ============================================================

best_config_name = str(ranking_df.iloc[0]["CONFIG"])


best_corr_threshold = experiment_results[best_config_name]["corr_threshold"]


best_oof_probability = experiment_results[best_config_name]["oof_probability"]


print()

print("선택 Config:", best_config_name)


print("선택 Correlation Threshold:", best_corr_threshold)


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

print("OOF BEST THRESHOLD V2.4")

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
# 19. 설정 LOCK
# ============================================================

print()

print("=" * 80)

print("V2.4 LOCK")

print("=" * 80)


print("Model: XGBoost")


print("Correlation Config:", best_config_name)


print("Correlation Threshold:", best_corr_threshold)


print("Decision Threshold:", best_threshold)


# ============================================================
# 20. 전체 Train Imputer
# ============================================================

final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X)


X_validation_imp = final_imputer.transform(X_valid)


X_train_imp_df = pd.DataFrame(X_train_imp, columns=candidate_features, index=X.index)


X_validation_imp_df = pd.DataFrame(
    X_validation_imp, columns=candidate_features, index=X_valid.index
)


# ============================================================
# 21. 전체 Train에서 최종 상관 제거
#
# Validation은 절대 사용하지 않음
# ============================================================

final_drop_cols = find_correlated_features(X_train_imp_df, best_corr_threshold)


final_selected_features = [
    col for col in candidate_features if col not in final_drop_cols
]


print()

print("=" * 80)

print("FINAL CORRELATION PRUNING")

print("=" * 80)


print("원래 Feature 수:", len(candidate_features))


print("제거 Feature 수:", len(final_drop_cols))


print("최종 Feature 수:", len(final_selected_features))


# ============================================================
# 22. 제거 Feature 출력
# ============================================================

print()

print("=" * 80)

print("FINAL DROPPED FEATURES")

print("=" * 80)


if len(final_drop_cols) == 0:

    print("제거된 Feature 없음")


else:

    for i, feature in enumerate(final_drop_cols, start=1):

        print(f"{i:3d}. " f"{feature}")


# ============================================================
# 23. 최종 Feature
# ============================================================

print()

print("=" * 80)

print("FINAL SELECTED FEATURES")

print("=" * 80)


for i, feature in enumerate(final_selected_features, start=1):

    print(f"{i:3d}. " f"{feature}")


# ============================================================
# 24. 최종 모델 학습
# ============================================================

final_model = make_xgb_model()


final_model.fit(X_train_imp_df[final_selected_features], y)


# ============================================================
# 25. Validation 예측
# ============================================================

valid_probability = final_model.predict_proba(
    X_validation_imp_df[final_selected_features]
)[:, 1]


# ============================================================
# 26. Validation 평가
# ============================================================

valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


print()

print("=" * 80)

print("FINAL VALIDATION V2.4")

print("=" * 80)


print("Config:", best_config_name)


print("Feature Count:", len(final_selected_features))


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
# 27. V2.3 vs V2.4
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "VERSION": "V2.3_240",
            "FEATURE_COUNT": 240,
            "CORR_CONFIG": "NONE",
            "THRESHOLD": 0.14,
            "PR_AUC": 0.7797,
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
            "VERSION": "V2.4_CORR",
            "FEATURE_COUNT": len(final_selected_features),
            "CORR_CONFIG": best_config_name,
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

print("V2.3 VS V2.4")

print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 28. 최종 Feature Importance
# ============================================================

importance_df = pd.DataFrame(
    {
        "FEATURE": final_selected_features,
        "IMPORTANCE": final_model.feature_importances_,
    }
)


importance_df = importance_df.sort_values("IMPORTANCE", ascending=False).reset_index(
    drop=True
)


print()

print("=" * 80)

print("TOP 50 FINAL FEATURE IMPORTANCE")

print("=" * 80)


print(importance_df.head(50).round(6).to_string(index=False))


# ============================================================
# 29. 정비 Feature 확인
# ============================================================

maintenance_importance_df = importance_df[
    importance_df["FEATURE"].isin(SELECTED_MAINTENANCE_COLS)
].copy()


print()

print("=" * 80)

print("FINAL MAINTENANCE FEATURES")

print("=" * 80)


if maintenance_importance_df.empty:

    print("정비 Feature가 모두 제거됐습니다.")


else:

    print(maintenance_importance_df.round(6).to_string(index=False))


# ============================================================
# 30. Drop Frequency
# ============================================================

drop_frequency_df = pd.DataFrame(drop_frequency_rows)


if not drop_frequency_df.empty:

    best_drop_frequency_df = (
        drop_frequency_df[drop_frequency_df["CONFIG"] == best_config_name]
        .sort_values(
            [
                "DROPPED_FOLDS",
                "FEATURE",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


else:

    best_drop_frequency_df = pd.DataFrame()


print()

print("=" * 80)

print("BEST CONFIG - DROP FREQUENCY")

print("=" * 80)


if best_drop_frequency_df.empty:

    print("제거 Feature 없음")


else:

    print(best_drop_frequency_df.head(100).to_string(index=False))


# ============================================================
# 31. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


selected_features_df = pd.DataFrame({"FEATURE": final_selected_features})


dropped_features_df = pd.DataFrame({"FEATURE": final_drop_cols})


ranking_df.to_csv(
    OUTPUT_DIR / "casting_24h_corr_ranking_v24.csv", index=False, encoding="utf-8-sig"
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_24h_corr_folds_v24.csv", index=False, encoding="utf-8-sig"
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_24h_thresholds_v24.csv", index=False, encoding="utf-8-sig"
)


comparison_df.to_csv(
    OUTPUT_DIR / "casting_24h_v23_v24_comparison.csv", index=False, encoding="utf-8-sig"
)


selected_features_df.to_csv(
    OUTPUT_DIR / "casting_24h_selected_features_v24.csv",
    index=False,
    encoding="utf-8-sig",
)


dropped_features_df.to_csv(
    OUTPUT_DIR / "casting_24h_dropped_features_v24.csv",
    index=False,
    encoding="utf-8-sig",
)


importance_df.to_csv(
    OUTPUT_DIR / "casting_24h_importance_v24.csv", index=False, encoding="utf-8-sig"
)


drop_frequency_df.to_csv(
    OUTPUT_DIR / "casting_24h_drop_frequency_v24.csv", index=False, encoding="utf-8-sig"
)


# ============================================================
# 32. 완료
# ============================================================

print()

print("=" * 80)

print("SAVE COMPLETE V2.4")

print("=" * 80)
