# ============================================================
# cv_24h_fair_v2.py
#
# 24H 공정 평가 - SENSOR + MAINTENANCE V2
#
# 원칙
# 1. Train에서만 모델 선택
# 2. Train에서만 Threshold 선택
# 3. Scenario 단위 5-Fold CV
# 4. Validation은 모든 선택이 끝난 뒤 평가
#
# 비교 모델
# - RandomForest
# - XGBoost
# - LightGBM
#
# 모델 선택
# - OOF PR-AUC 최대
#
# Threshold 선택
# - OOF F2 최대
#
# V1:
# 센서 237개
#
# V2:
# 센서 237개 + 정비 9개
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

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
from lightgbm import LGBMClassifier

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# V2 데이터
# ============================================================

TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v2.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features_v2.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)

print("24H FAIR MODEL SELECTION V2")

print("SENSOR + MAINTENANCE")

print("=" * 80)


print()

print("Train:", train_df.shape)

print("Validation:", valid_df.shape)


# ============================================================
# 2. 24H 유효 Label만 사용
# ============================================================

train = train_df[(train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()].copy()


valid = valid_df[(valid_df["CENSORED_24H"] == 0) & valid_df["RISK_24H"].notna()].copy()


train["RISK_24H"] = train["RISK_24H"].astype(int)


valid["RISK_24H"] = valid["RISK_24H"].astype(int)


print()

print("=" * 80)

print("24H LABEL CHECK")

print("=" * 80)


print("Train usable:", train.shape)

print(train["RISK_24H"].value_counts())


print()


print("Validation usable:", valid.shape)

print(valid["RISK_24H"].value_counts())


# ============================================================
# 3. Feature 선택
#
# 정답 / 메타데이터 제외
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


feature_cols = [
    col
    for col in train.columns
    if (col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(train[col]))
]


print()

print("=" * 80)

print("FEATURE CHECK")

print("=" * 80)


print("전체 모델 Feature 수:", len(feature_cols))


# ============================================================
# 정비 Feature 확인
# ============================================================

maintenance_cols = [
    "MNT_TOTAL_COUNT",
    "MNT_24H_COUNT",
    "MNT_72H_COUNT",
    "MNT_LAST_GAP_H",
    "MNT_DOWN_SUM",
    "MNT_DOWN_MEAN",
    "MNT_REPLACE_COUNT",
    "MNT_COST_SUM",
    "MNT_COST_MEAN",
]


print()

print("정비 Feature:")


for col in maintenance_cols:

    print(f"{col:<22}", "→", col in feature_cols)


maintenance_feature_count = sum(col in feature_cols for col in maintenance_cols)


print()

print("정비 Feature 수:", maintenance_feature_count)


print("센서 및 기타 Feature 수:", len(feature_cols) - maintenance_feature_count)


# ============================================================
# 4. Train / Validation 구성
# ============================================================

X = train[feature_cols]


y = train["RISK_24H"]


groups = train["SCENARIO_ID"]


X_valid = valid[feature_cols]


y_valid = valid["RISK_24H"]


# ============================================================
# 5. Class imbalance
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
# 6. 모델 정의
# ============================================================

models = {
    # ========================================================
    # Random Forest
    # ========================================================
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
    # ========================================================
    # XGBoost
    # ========================================================
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
    # ========================================================
    # LightGBM
    # ========================================================
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
# 7. Scenario 단위 Cross Validation
#
# 같은 Scenario의 행들이
# 서로 다른 Fold로 섞이지 않도록 함
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


# ============================================================
# 8. Threshold 평가 함수
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
# 9. 모델별 OOF 예측
# ============================================================

model_oof_results = {}

fold_rows = []


for model_name, base_model in models.items():

    print()

    print("=" * 80)

    print(f"{model_name} - " f"5 FOLD CV")

    print("=" * 80)

    # --------------------------------------------------------
    # 모든 Train 행의 OOF 확률 저장
    # --------------------------------------------------------

    oof_probability = np.full(len(train), np.nan, dtype=float)

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        X_fold_train = X.iloc[train_idx]

        y_fold_train = y.iloc[train_idx]

        X_fold_valid = X.iloc[fold_valid_idx]

        y_fold_valid = y.iloc[fold_valid_idx]

        # ----------------------------------------------------
        # Scenario 검사
        # ----------------------------------------------------

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        overlap = train_groups & valid_groups

        if overlap:

            raise RuntimeError(
                f"{model_name} " f"Fold {fold}: " f"Scenario leakage 발생 " f"{overlap}"
            )

        # ----------------------------------------------------
        # 모델 복사
        # ----------------------------------------------------

        model = clone(base_model)

        # ----------------------------------------------------
        # 학습
        # ----------------------------------------------------

        model.fit(X_fold_train, y_fold_train)

        # ----------------------------------------------------
        # Fold Validation 확률
        # ----------------------------------------------------

        fold_probability = model.predict_proba(X_fold_valid)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ----------------------------------------------------
        # Fold 성능
        # ----------------------------------------------------

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "MODEL": model_name,
                "FOLD": fold,
                "TRAIN_ROWS": len(train_idx),
                "VALID_ROWS": len(fold_valid_idx),
                "TRAIN_SCENARIOS": len(train_groups),
                "VALID_SCENARIOS": len(valid_groups),
                "VALID_POSITIVE": int(y_fold_valid.sum()),
                "ROC_AUC": fold_roc,
                "PR_AUC": fold_pr,
            }
        )

        print(
            f"Fold {fold}"
            f" | train scenario "
            f"{len(train_groups)}"
            f" | valid scenario "
            f"{len(valid_groups)}"
            f" | positive "
            f"{int(y_fold_valid.sum())}"
            f" | PR-AUC "
            f"{fold_pr:.4f}"
        )

    # --------------------------------------------------------
    # OOF 누락 검사
    # --------------------------------------------------------

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{model_name}: " f"OOF prediction에 " f"NaN 존재")

    # --------------------------------------------------------
    # 전체 OOF 성능
    # --------------------------------------------------------

    oof_roc_auc = roc_auc_score(y, oof_probability)

    oof_pr_auc = average_precision_score(y, oof_probability)

    print()

    print(f"{model_name} " f"OOF ROC-AUC: " f"{oof_roc_auc:.4f}")

    print(f"{model_name} " f"OOF PR-AUC : " f"{oof_pr_auc:.4f}")

    model_oof_results[model_name] = {
        "probability": oof_probability,
        "roc_auc": oof_roc_auc,
        "pr_auc": oof_pr_auc,
    }


# ============================================================
# 10. OOF 모델 순위
#
# Validation은 아직 사용하지 않음
# ============================================================

ranking_rows = []


for model_name, result in model_oof_results.items():

    ranking_rows.append(
        {
            "MODEL": model_name,
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

print("OOF MODEL RANKING V2")

print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 11. 최종 모델 선택
# ============================================================

best_model_name = ranking_df.iloc[0]["MODEL"]


print()


print("선택 모델:", best_model_name)


best_oof_probability = model_oof_results[best_model_name]["probability"]


# ============================================================
# 12. Threshold 탐색
#
# Train OOF에서만 선택
# ============================================================

threshold_rows = []


for threshold in np.arange(0.05, 0.91, 0.01):

    threshold = round(float(threshold), 2)

    result = evaluate_threshold(
        y_true=y, probability=best_oof_probability, threshold=threshold
    )

    threshold_rows.append(result)


threshold_df = pd.DataFrame(threshold_rows)


# ============================================================
# 13. F2 최대 Threshold 선택
# ============================================================

best_threshold_idx = threshold_df["F2"].idxmax()


best_threshold_row = threshold_df.loc[best_threshold_idx]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)

print("OOF BEST THRESHOLD V2 - F2")

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
# Validation 사용 전 모델 / Threshold LOCK
# ============================================================

print()

print("=" * 80)

print("MODEL / THRESHOLD LOCK V2")

print("=" * 80)


print("Final model:", best_model_name)


print("Final threshold:", best_threshold)


print()

print("여기까지 Validation 정답은 " "모델/Threshold 선택에 사용하지 않았습니다.")


# ============================================================
# 14. 전체 Train으로 최종 모델 재학습
# ============================================================

final_model = clone(models[best_model_name])


final_model.fit(X, y)


# ============================================================
# 15. Validation 최종 평가
# ============================================================

valid_probability = final_model.predict_proba(X_valid)[:, 1]


valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


# ============================================================
# 16. Validation 결과
# ============================================================

print()

print("=" * 80)

print("FINAL VALIDATION V2 - ONE SHOT")

print("=" * 80)


print("Model:", best_model_name)


print("Threshold:", best_threshold)


print()


print(f"ROC-AUC   : " f"{valid_roc_auc:.4f}")


print(f"PR-AUC    : " f"{valid_pr_auc:.4f}")


print(f"Precision : " f"{valid_metrics['PRECISION']:.4f}")


print(f"Recall    : " f"{valid_metrics['RECALL']:.4f}")


print(f"F1        : " f"{valid_metrics['F1']:.4f}")


print(f"F2        : " f"{valid_metrics['F2']:.4f}")


print()


print("Confusion Matrix")


print(f"TN={valid_metrics['TN']}" f"  FP={valid_metrics['FP']}")


print(f"FN={valid_metrics['FN']}" f"  TP={valid_metrics['TP']}")


# ============================================================
# 17. V1 vs V2 비교
#
# V1 공정평가 결과:
#
# XGBoost
# threshold = 0.10
#
# PR-AUC 0.7780
# Precision 0.5481
# Recall 0.8143
# F1 0.6552
# F2 0.7422
# TN 522
# FP 47
# FN 13
# TP 57
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "VERSION": "V1_SENSOR_ONLY",
            "FEATURE_COUNT": 237,
            "MODEL": "XGBoost",
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
            "VERSION": "V2_SENSOR_MAINTENANCE",
            "FEATURE_COUNT": len(feature_cols),
            "MODEL": best_model_name,
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

print("V1 SENSOR ONLY " "VS " "V2 SENSOR + MAINTENANCE")

print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 18. 성능 변화 계산
# ============================================================

delta_pr = valid_pr_auc - 0.7780


delta_precision = valid_metrics["PRECISION"] - 0.5481


delta_recall = valid_metrics["RECALL"] - 0.8143


delta_f1 = valid_metrics["F1"] - 0.6552


delta_f2 = valid_metrics["F2"] - 0.7422


delta_fp = valid_metrics["FP"] - 47


delta_fn = valid_metrics["FN"] - 13


print()

print("=" * 80)

print("V2 CHANGE FROM V1")

print("=" * 80)


print(f"PR-AUC 변화    : " f"{delta_pr:+.4f}")


print(f"Precision 변화 : " f"{delta_precision:+.4f}")


print(f"Recall 변화    : " f"{delta_recall:+.4f}")


print(f"F1 변화        : " f"{delta_f1:+.4f}")


print(f"F2 변화        : " f"{delta_f2:+.4f}")


print(f"FP 변화        : " f"{delta_fp:+d}")


print(f"FN 변화        : " f"{delta_fn:+d}")


# ============================================================
# 19. 정비 Feature Importance
#
# 최종 선택 모델이 Tree 기반 모델이므로
# feature_importances_ 확인 가능
# ============================================================

try:

    model_step = final_model.named_steps["model"]

    importances = model_step.feature_importances_

    importance_df = pd.DataFrame(
        {
            "FEATURE": feature_cols,
            "IMPORTANCE": importances,
        }
    )

    importance_df = importance_df.sort_values(
        "IMPORTANCE", ascending=False
    ).reset_index(drop=True)

    print()

    print("=" * 80)

    print("TOP 30 FEATURE IMPORTANCE")

    print("=" * 80)

    print(importance_df.head(30).round(6).to_string(index=False))

    # --------------------------------------------------------
    # 정비 Feature만 확인
    # --------------------------------------------------------

    maintenance_importance_df = importance_df[
        importance_df["FEATURE"].isin(maintenance_cols)
    ].copy()

    print()

    print("=" * 80)

    print("MAINTENANCE FEATURE IMPORTANCE")

    print("=" * 80)

    print(maintenance_importance_df.round(6).to_string(index=False))


except Exception as e:

    print()

    print("[WARNING] " "Feature importance 계산 실패:")

    print(type(e).__name__, ":", e)


# ============================================================
# 20. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


RANKING_PATH = OUTPUT_DIR / "casting_24h_cv_model_ranking_v2.csv"


FOLD_PATH = OUTPUT_DIR / "casting_24h_cv_fold_metrics_v2.csv"


THRESHOLD_PATH = OUTPUT_DIR / "casting_24h_cv_thresholds_v2.csv"


COMPARISON_PATH = OUTPUT_DIR / "casting_24h_v1_vs_v2_comparison.csv"


IMPORTANCE_PATH = OUTPUT_DIR / "casting_24h_feature_importance_v2.csv"


MAINTENANCE_IMPORTANCE_PATH = OUTPUT_DIR / "casting_24h_maintenance_importance_v2.csv"


ranking_df.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")


fold_df.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")


threshold_df.to_csv(THRESHOLD_PATH, index=False, encoding="utf-8-sig")


comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")


try:

    importance_df.to_csv(IMPORTANCE_PATH, index=False, encoding="utf-8-sig")

    maintenance_importance_df.to_csv(
        MAINTENANCE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig"
    )


except NameError:

    pass


print()

print("=" * 80)

print("SAVE COMPLETE V2")

print("=" * 80)


print(RANKING_PATH)


print(FOLD_PATH)


print(THRESHOLD_PATH)


print(COMPARISON_PATH)


if "importance_df" in globals():

    print(IMPORTANCE_PATH)

    print(MAINTENANCE_IMPORTANCE_PATH)
