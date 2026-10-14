# ============================================================
# cv_24h_fair.py
#
# 24H 모델 공정 평가 버전
#
# 원칙:
# 1. 모델 선택은 Train에서만
# 2. Threshold 선택도 Train에서만
# 3. Scenario 단위 Cross Validation
# 4. Validation은 모든 선택이 끝난 후 딱 한 번 평가
#
# 비교:
# - RandomForest
# - XGBoost
# - LightGBM
#
# Model selection:
# - OOF PR-AUC 우선
#
# Threshold selection:
# - OOF F2 최대
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


TRAIN_PATH = OUTPUT_DIR / "casting_train_features.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("24H FAIR MODEL SELECTION")
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

print("Feature 수:", len(feature_cols))


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
    # --------------------------------------------------------
    # Random Forest
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
# 7. Scenario 단위 Cross Validation
#
# 중요:
# 같은 Scenario의 여러 OBSERVED_AT이
# Train과 CV Validation에 동시에 들어가지 않게 함
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
# 9. 모델별 OOF 예측 생성
# ============================================================

model_oof_results = {}

fold_rows = []


for model_name, base_model in models.items():

    print()

    print("=" * 80)
    print(f"{model_name} - 5 FOLD CV")
    print("=" * 80)

    # --------------------------------------------------------
    # 전체 Train 행에 대한 Out-Of-Fold 확률
    # --------------------------------------------------------

    oof_probability = np.full(len(train), np.nan, dtype=float)

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        X_fold_train = X.iloc[train_idx]

        y_fold_train = y.iloc[train_idx]

        X_fold_valid = X.iloc[fold_valid_idx]

        y_fold_valid = y.iloc[fold_valid_idx]

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        overlap = train_groups & valid_groups

        # ----------------------------------------------------
        # Scenario 누수 검사
        # ----------------------------------------------------

        if overlap:

            raise RuntimeError(
                f"{model_name} Fold {fold}: " f"Scenario leakage 발생 " f"{overlap}"
            )

        model = clone(base_model)

        model.fit(X_fold_train, y_fold_train)

        fold_probability = model.predict_proba(X_fold_valid)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

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
    # OOF가 모두 채워졌는지 검사
    # --------------------------------------------------------

    if np.isnan(oof_probability).any():

        raise RuntimeError(f"{model_name}: " f"OOF prediction에 NaN 존재")

    # --------------------------------------------------------
    # 전체 OOF 성능
    # --------------------------------------------------------

    oof_roc_auc = roc_auc_score(y, oof_probability)

    oof_pr_auc = average_precision_score(y, oof_probability)

    print()

    print(f"{model_name} OOF ROC-AUC:" f" {oof_roc_auc:.4f}")

    print(f"{model_name} OOF PR-AUC :" f" {oof_pr_auc:.4f}")

    model_oof_results[model_name] = {
        "probability": oof_probability,
        "roc_auc": oof_roc_auc,
        "pr_auc": oof_pr_auc,
    }


# ============================================================
# 10. 모델 순위
#
# 모델 선택은 Train OOF PR-AUC로만!
# Validation 아직 사용 안 함.
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
print("OOF MODEL RANKING")
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
# 12. Threshold는 OOF Train 예측으로만 선택
# ============================================================

threshold_rows = []


for threshold in np.arange(0.05, 0.91, 0.01):

    result = evaluate_threshold(
        y_true=y, probability=best_oof_probability, threshold=round(float(threshold), 2)
    )

    threshold_rows.append(result)


threshold_df = pd.DataFrame(threshold_rows)


# ------------------------------------------------------------
# F2 최대 Threshold
# ------------------------------------------------------------

best_threshold_idx = threshold_df["F2"].idxmax()


best_threshold_row = threshold_df.loc[best_threshold_idx]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)
print("OOF BEST THRESHOLD - F2")
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
# 여기까지 Validation 정답은 모델/Threshold 선택에
# 단 한 번도 사용하지 않았음.
# ============================================================


print()

print("=" * 80)
print("MODEL / THRESHOLD LOCK")
print("=" * 80)

print("Final model:", best_model_name)

print("Final threshold:", best_threshold)

print()

print("이제부터 Validation을 딱 한 번 평가합니다.")


# ============================================================
# 13. 선택된 모델을 전체 Train 216개로 재학습
# ============================================================

final_model = clone(models[best_model_name])


final_model.fit(X, y)


# ============================================================
# 14. Validation 딱 한 번 평가
# ============================================================

valid_probability = final_model.predict_proba(X_valid)[:, 1]


valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=best_threshold
)


# ============================================================
# 15. 최종 Validation 출력
# ============================================================

print()

print("=" * 80)
print("FINAL VALIDATION - ONE SHOT")
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
# 16. 기존 결과와 비교용 표
# ============================================================

comparison_df = pd.DataFrame(
    [
        {
            "VERSION": "OLD_VALIDATION_SELECTED",
            "MODEL": "RandomForest",
            "THRESHOLD": 0.47,
            "PR_AUC": 0.7614,
            "PRECISION": 0.7073,
            "RECALL": 0.8286,
            "F1": 0.7632,
            "F2": 0.8011,
            "FP": 24,
            "FN": 12,
            "TP": 58,
        },
        {
            "VERSION": "FAIR_TRAIN_CV",
            "MODEL": best_model_name,
            "THRESHOLD": best_threshold,
            "PR_AUC": valid_pr_auc,
            "PRECISION": valid_metrics["PRECISION"],
            "RECALL": valid_metrics["RECALL"],
            "F1": valid_metrics["F1"],
            "F2": valid_metrics["F2"],
            "FP": valid_metrics["FP"],
            "FN": valid_metrics["FN"],
            "TP": valid_metrics["TP"],
        },
    ]
)


print()

print("=" * 80)
print("OLD VS FAIR EVALUATION")
print("=" * 80)


print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 17. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


RANKING_PATH = OUTPUT_DIR / "casting_24h_cv_model_ranking.csv"


FOLD_PATH = OUTPUT_DIR / "casting_24h_cv_fold_metrics.csv"


THRESHOLD_PATH = OUTPUT_DIR / "casting_24h_cv_thresholds.csv"


COMPARISON_PATH = OUTPUT_DIR / "casting_24h_fair_comparison.csv"


ranking_df.to_csv(RANKING_PATH, index=False, encoding="utf-8-sig")


fold_df.to_csv(FOLD_PATH, index=False, encoding="utf-8-sig")


threshold_df.to_csv(THRESHOLD_PATH, index=False, encoding="utf-8-sig")


comparison_df.to_csv(COMPARISON_PATH, index=False, encoding="utf-8-sig")


print()

print("=" * 80)
print("SAVE COMPLETE")
print("=" * 80)

print(RANKING_PATH)

print(FOLD_PATH)

print(THRESHOLD_PATH)

print(COMPARISON_PATH)
