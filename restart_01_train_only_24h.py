# ============================================================
# restart_01_train_only_24h.py
#
# STRICT RESTART - 24H
#
# 중요 원칙
# ------------------------------------------------------------
# 1. TRAIN ONLY
# 2. Validation 파일 절대 읽지 않음
# 3. Test 파일 절대 읽지 않음
# 4. Scenario Group CV
# 5. Feature family도 Train OOF로 선택
# 6. Model도 Train OOF로 선택
# 7. Threshold도 Train OOF로 선택
#
# 최종 선택 기준
# ------------------------------------------------------------
# Feature + Model : OOF PR-AUC 최대
# Threshold       : OOF F2 최대
#
# 입력
# ------------------------------------------------------------
# outputs/casting_train_features_v22.csv
#
# Validation 관련 경로 없음.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


# ============================================================
# 1. Train ONLY 로드
#
# 여기에는 Validation 경로가 아예 없다.
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


print("=" * 80)
print("STRICT RESTART - TRAIN ONLY 24H")
print("=" * 80)

print()

print("Train raw shape:", train_df.shape)


# ============================================================
# 2. 24H usable label
# ============================================================

train = train_df[(train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()].copy()


train["RISK_24H"] = train["RISK_24H"].astype(int)


print()

print("=" * 80)
print("24H LABEL CHECK")
print("=" * 80)

print("Train usable:", train.shape)

print()

print(train["RISK_24H"].value_counts())

print()

print("Scenario 수:", train["SCENARIO_ID"].nunique())


# ============================================================
# 3. Metadata / label 제외
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
# 4. Maintenance 11개
#
# 이전 중요도 기반 선택 폐기.
# 11개 전체를 하나의 candidate family로 취급.
# ============================================================

MAINTENANCE_FEATURES = [
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
# 5. 기존 Sensor Feature
#
# Maintenance와 Metadata를 제외하면
# 기존 약 237개 Sensor/Window Feature
# ============================================================

sensor_features = [
    col
    for col in train.columns
    if (
        col not in EXCLUDE_COLS
        and col not in MAINTENANCE_FEATURES
        and pd.api.types.is_numeric_dtype(train[col])
    )
]


print()

print("=" * 80)
print("BASE SENSOR FEATURE")
print("=" * 80)

print("Sensor Feature 수:", len(sensor_features))


# ============================================================
# 6. Mean Diff 12개 생성
#
# Validation 결과를 보지 않고,
# 사전에 정의한 후보 family 전체를 생성.
#
# 최근 평균 - 장기 평균
# ============================================================

MEAN_DIFF_SENSORS = [
    "CONTROL_DELAY",
    "MLD-LVL",
    "OSC-STK",
    "OSC-FRQ",
]


mean_diff_features = []


for sensor in MEAN_DIFF_SENSORS:

    definitions = [
        (
            f"{sensor}_1H_MINUS_3H_MEAN",
            f"1H_{sensor}_MEAN",
            f"3H_{sensor}_MEAN",
        ),
        (
            f"{sensor}_3H_MINUS_6H_MEAN",
            f"3H_{sensor}_MEAN",
            f"6H_{sensor}_MEAN",
        ),
        (
            f"{sensor}_6H_MINUS_12H_MEAN",
            f"6H_{sensor}_MEAN",
            f"12H_{sensor}_MEAN",
        ),
    ]

    for new_col, short_col, long_col in definitions:

        if short_col not in train.columns:

            raise KeyError(f"컬럼 없음: " f"{short_col}")

        if long_col not in train.columns:

            raise KeyError(f"컬럼 없음: " f"{long_col}")

        train[new_col] = train[short_col] - train[long_col]

        mean_diff_features.append(new_col)


print()

print("=" * 80)
print("MEAN DIFF FEATURE")
print("=" * 80)

print("Mean Diff 수:", len(mean_diff_features))

for col in mean_diff_features:

    print(col)


# ============================================================
# 7. Maintenance 존재 검사
# ============================================================

missing_mnt = [col for col in MAINTENANCE_FEATURES if col not in train.columns]


if missing_mnt:

    raise KeyError(f"Maintenance Feature 누락: " f"{missing_mnt}")


# ============================================================
# 8. Feature Family 정의
#
# 이전 Validation 기반 선택 폐기.
# Family 전체를 Train OOF에서 경쟁시킨다.
# ============================================================

FEATURE_SETS = {
    "SENSOR_ONLY": sensor_features,
    "SENSOR_MAINT": sensor_features + MAINTENANCE_FEATURES,
    "SENSOR_MEAN_DIFF": sensor_features + mean_diff_features,
    "SENSOR_MAINT_MEAN_DIFF": sensor_features
    + MAINTENANCE_FEATURES
    + mean_diff_features,
}


print()

print("=" * 80)
print("FEATURE FAMILY CHECK")
print("=" * 80)


for name, cols in FEATURE_SETS.items():

    print(f"{name:<25}" f"{len(cols)}")


# ============================================================
# 9. Label / Group
# ============================================================

y = train["RISK_24H"].copy()


groups = train["SCENARIO_ID"].copy()


negative_count = int((y == 0).sum())


positive_count = int((y == 1).sum())


scale_pos_weight = negative_count / positive_count


print()

print("=" * 80)
print("CLASS BALANCE")
print("=" * 80)

print("Negative:", negative_count)

print("Positive:", positive_count)

print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 10. 모델
#
# Hyperparameter도 이번 실행 전에 고정.
# 결과 보고 변경하지 않는다.
# ============================================================


def make_model(model_name):

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

    if model_name == "XGB":

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

    if model_name == "LGBM":

        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

    raise ValueError(model_name)


MODEL_NAMES = [
    "RF",
    "XGB",
    "LGBM",
]


# ============================================================
# 11. Scenario Group CV
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# ============================================================
# 12. OOF 저장
# ============================================================

experiment_results = {}

fold_rows = []


# ============================================================
# 13. Feature Family × Model Battle
# ============================================================

for feature_set_name, feature_cols in FEATURE_SETS.items():

    X = train[feature_cols].copy()

    for model_name in MODEL_NAMES:

        experiment_name = f"{feature_set_name}" f"__" f"{model_name}"

        print()

        print("=" * 80)

        print(experiment_name)

        print("=" * 80)

        oof_probability = np.full(
            len(train),
            np.nan,
            dtype=float,
        )

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(
                X,
                y,
                groups,
            ),
            start=1,
        ):

            X_fold_train = X.iloc[train_idx].copy()

            X_fold_valid = X.iloc[valid_idx].copy()

            y_fold_train = y.iloc[train_idx].copy()

            y_fold_valid = y.iloc[valid_idx].copy()

            train_groups = set(groups.iloc[train_idx])

            valid_groups = set(groups.iloc[valid_idx])

            overlap = train_groups & valid_groups

            if overlap:

                raise RuntimeError("Scenario Leakage 발생")

            # =================================================
            # Imputer도 Fold Train에서만 fit
            # =================================================

            imputer = SimpleImputer(strategy="median")

            X_fold_train_imp = imputer.fit_transform(X_fold_train)

            X_fold_valid_imp = imputer.transform(X_fold_valid)

            model = make_model(model_name)

            model.fit(
                X_fold_train_imp,
                y_fold_train,
            )

            fold_probability = model.predict_proba(X_fold_valid_imp)[:, 1]

            oof_probability[valid_idx] = fold_probability

            fold_pr = average_precision_score(
                y_fold_valid,
                fold_probability,
            )

            fold_roc = roc_auc_score(
                y_fold_valid,
                fold_probability,
            )

            fold_rows.append(
                {
                    "EXPERIMENT": experiment_name,
                    "FEATURE_SET": feature_set_name,
                    "MODEL": model_name,
                    "FEATURE_COUNT": len(feature_cols),
                    "FOLD": fold,
                    "TRAIN_SCENARIOS": len(train_groups),
                    "VALID_SCENARIOS": len(valid_groups),
                    "VALID_POSITIVE": int(y_fold_valid.sum()),
                    "PR_AUC": fold_pr,
                    "ROC_AUC": fold_roc,
                }
            )

            print(
                f"Fold {fold}"
                f" | Positive "
                f"{int(y_fold_valid.sum())}"
                f" | PR-AUC "
                f"{fold_pr:.4f}"
            )

        if np.isnan(oof_probability).any():

            raise RuntimeError(f"{experiment_name}: " f"OOF NaN")

        oof_pr = average_precision_score(
            y,
            oof_probability,
        )

        oof_roc = roc_auc_score(
            y,
            oof_probability,
        )

        experiment_results[experiment_name] = {
            "feature_set": feature_set_name,
            "model": model_name,
            "feature_cols": feature_cols,
            "oof_probability": oof_probability,
            "oof_pr": oof_pr,
            "oof_roc": oof_roc,
        }

        print()

        print("OOF PR-AUC:", round(oof_pr, 4))

        print("OOF ROC-AUC:", round(oof_roc, 4))


# ============================================================
# 14. Ranking
#
# 오직 OOF PR-AUC.
# ============================================================

ranking_rows = []


for experiment_name, result in experiment_results.items():

    ranking_rows.append(
        {
            "EXPERIMENT": experiment_name,
            "FEATURE_SET": result["feature_set"],
            "MODEL": result["model"],
            "FEATURE_COUNT": len(result["feature_cols"]),
            "OOF_PR_AUC": result["oof_pr"],
            "OOF_ROC_AUC": result["oof_roc"],
        }
    )


ranking_df = pd.DataFrame(ranking_rows)


ranking_df = ranking_df.sort_values(
    [
        "OOF_PR_AUC",
        "OOF_ROC_AUC",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


print()

print("=" * 80)
print("TRAIN ONLY 24H RANKING")
print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 15. Best experiment LOCK 후보
# ============================================================

best_experiment = str(ranking_df.iloc[0]["EXPERIMENT"])


best_result = experiment_results[best_experiment]


best_probability = best_result["oof_probability"]


print()

print("=" * 80)
print("24H BEST TRAIN-ONLY EXPERIMENT")
print("=" * 80)

print("Experiment:", best_experiment)

print("Feature Set:", best_result["feature_set"])

print("Model:", best_result["model"])

print("Feature Count:", len(best_result["feature_cols"]))

print("OOF PR-AUC:", round(best_result["oof_pr"], 4))


# ============================================================
# 16. Threshold evaluator
# ============================================================


def evaluate_threshold(probability, threshold):

    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    precision = precision_score(
        y,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0,
    )

    f2 = fbeta_score(
        y,
        pred,
        beta=2,
        zero_division=0,
    )

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
# 17. Threshold는 OOF에서만 선택
# ============================================================

threshold_rows = []


for threshold in np.arange(
    0.01,
    0.81,
    0.01,
):

    threshold = round(float(threshold), 2)

    threshold_rows.append(
        evaluate_threshold(
            probability=best_probability,
            threshold=threshold,
        )
    )


threshold_df = pd.DataFrame(threshold_rows)


best_threshold_idx = threshold_df["F2"].idxmax()


best_threshold_row = threshold_df.loc[best_threshold_idx]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)
print("24H TRAIN-ONLY OOF THRESHOLD")
print("=" * 80)


print(best_threshold_row.round(4).to_string())


# ============================================================
# 18. LOCK 정보 저장
#
# 아직 Validation 평가 X
# ============================================================

lock_info = {
    "protocol": "TRAIN_ONLY_CV",
    "target": "RISK_24H",
    "censor": "CENSORED_24H",
    "feature_set": best_result["feature_set"],
    "model": best_result["model"],
    "feature_count": len(best_result["feature_cols"]),
    "features": best_result["feature_cols"],
    "threshold": best_threshold,
    "oof_pr_auc": float(best_result["oof_pr"]),
    "oof_roc_auc": float(best_result["oof_roc"]),
    "oof_precision": float(best_threshold_row["PRECISION"]),
    "oof_recall": float(best_threshold_row["RECALL"]),
    "oof_f1": float(best_threshold_row["F1"]),
    "oof_f2": float(best_threshold_row["F2"]),
    "oof_tn": int(best_threshold_row["TN"]),
    "oof_fp": int(best_threshold_row["FP"]),
    "oof_fn": int(best_threshold_row["FN"]),
    "oof_tp": int(best_threshold_row["TP"]),
}


# ============================================================
# 19. 저장
# ============================================================

ranking_df.to_csv(
    OUTPUT_DIR / "restart_24h_train_only_ranking.csv",
    index=False,
    encoding="utf-8-sig",
)


pd.DataFrame(fold_rows).to_csv(
    OUTPUT_DIR / "restart_24h_train_only_folds.csv",
    index=False,
    encoding="utf-8-sig",
)


threshold_df.to_csv(
    OUTPUT_DIR / "restart_24h_train_only_thresholds.csv",
    index=False,
    encoding="utf-8-sig",
)


oof_df = train[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_24H",
    ]
].copy()


oof_df["PRED_PROBA_24H"] = best_probability


oof_df["PRED_24H"] = (best_probability >= best_threshold).astype(int)


oof_df.to_csv(
    OUTPUT_DIR / "restart_24h_train_only_oof.csv",
    index=False,
    encoding="utf-8-sig",
)


with (OUTPUT_DIR / "restart_24h_lock.json").open("w", encoding="utf-8") as f:

    json.dump(
        lock_info,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 20. 완료
# ============================================================

print()

print("=" * 80)
print("24H TRAIN-ONLY LOCK COMPLETE")
print("=" * 80)

print("Feature Set:", lock_info["feature_set"])

print("Model:", lock_info["model"])

print("Feature Count:", lock_info["feature_count"])

print("Threshold:", lock_info["threshold"])

print("OOF PR-AUC:", round(lock_info["oof_pr_auc"], 4))

print("OOF Recall:", round(lock_info["oof_recall"], 4))

print("OOF F2:", round(lock_info["oof_f2"], 4))


print()

print("※ Validation은 사용하지 않았습니다.")

print("※ 다음 단계도 Validation 사용 금지.")
