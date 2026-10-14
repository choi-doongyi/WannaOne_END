# ============================================================
# restart_02_train_only_6h.py
#
# STRICT RESTART - 6H DANGER
#
# 원칙
# ------------------------------------------------------------
# 1. TRAIN ONLY
# 2. Validation 파일 절대 읽지 않음
# 3. Test 파일 절대 읽지 않음
# 4. Scenario 단위 5-Fold CV
# 5. Feature family = Train OOF PR-AUC로 선택
# 6. Model = Train OOF PR-AUC로 선택
#
# Threshold 정책 (사전 고정)
# ------------------------------------------------------------
# OOF Recall >= 0.75를 만족하는 후보 중
# Precision 최대
#
# tie-break:
# 1) FP 적은 것
# 2) F2 높은 것
# 3) Threshold 높은 것
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
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


# 6H DANGER 운영 정책
RECALL_TARGET = 0.75


# ============================================================
# 1. TRAIN ONLY 데이터 로드
#
# Validation 경로 자체가 없다.
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


print("=" * 80)
print("STRICT RESTART - TRAIN ONLY 6H")
print("=" * 80)

print()

print("Train raw shape:", train_df.shape)


# ============================================================
# 2. 6H usable label
# ============================================================

train = train_df[(train_df["CENSORED_6H"] == 0) & train_df["RISK_6H"].notna()].copy()


train["RISK_6H"] = train["RISK_6H"].astype(int)


print()

print("=" * 80)
print("6H LABEL CHECK")
print("=" * 80)


print("Train usable:", train.shape)


print()

print(train["RISK_6H"].value_counts())


print()

print("Scenario 수:", train["SCENARIO_ID"].nunique())


# ============================================================
# 3. Metadata / Label 제외
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
# 4. Maintenance Feature 11개
#
# 이전 중요도 기반 선택은 전부 폐기.
# 전체 11개를 하나의 Feature Family로 취급.
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
# 5. Sensor Feature
#
# Metadata와 Maintenance를 제외한 기존 수치형 Feature.
#
# 예상:
# 237개
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


if len(sensor_features) != 237:

    print("[WARNING] Sensor Feature 예상 237개")


# ============================================================
# 6. Mean Diff 12개 생성
#
# 이전 결과와 무관하게
# 사전에 정의한 Feature Family 전체 생성.
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
# 7. Maintenance Feature 존재 확인
# ============================================================

missing_mnt = [col for col in MAINTENANCE_FEATURES if col not in train.columns]


if missing_mnt:

    raise KeyError(f"Maintenance Feature 누락: " f"{missing_mnt}")


# ============================================================
# 8. Feature Family
#
# 24H와 동일한 후보 구조.
#
# Validation 결과로 가지치기하지 않는다.
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


for feature_set_name, feature_cols in FEATURE_SETS.items():

    print(f"{feature_set_name:<25}" f"{len(feature_cols)}")


# ============================================================
# 9. Label / Scenario Group
# ============================================================

y = train["RISK_6H"].copy()


groups = train["SCENARIO_ID"].copy()


negative_count = int((y == 0).sum())


positive_count = int((y == 1).sum())


scale_pos_weight = negative_count / positive_count


positive_rate = positive_count / len(y)


print()

print("=" * 80)
print("CLASS BALANCE")
print("=" * 80)


print("Negative:", negative_count)


print("Positive:", positive_count)


print("Positive Rate:", round(positive_rate, 6))


print("scale_pos_weight:", round(scale_pos_weight, 4))


# ============================================================
# 10. Model Factory
#
# Hyperparameter는 실행 전에 고정.
# 결과를 보고 변경하지 않는다.
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
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LightGBM
    # --------------------------------------------------------

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

    raise ValueError(f"Unknown model: " f"{model_name}")


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
# 12. 결과 저장
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

            # =================================================
            # Fold 데이터
            # =================================================

            X_fold_train = X.iloc[train_idx].copy()

            X_fold_valid = X.iloc[valid_idx].copy()

            y_fold_train = y.iloc[train_idx].copy()

            y_fold_valid = y.iloc[valid_idx].copy()

            train_groups = set(groups.iloc[train_idx])

            valid_groups = set(groups.iloc[valid_idx])

            # =================================================
            # Scenario Leakage 검사
            # =================================================

            overlap = train_groups & valid_groups

            if overlap:

                raise RuntimeError(
                    f"{experiment_name} " f"Fold {fold}: " f"Scenario Leakage"
                )

            # =================================================
            # Imputer
            #
            # Fold Train에서만 fit.
            # =================================================

            imputer = SimpleImputer(strategy="median")

            X_fold_train_imp = imputer.fit_transform(X_fold_train)

            X_fold_valid_imp = imputer.transform(X_fold_valid)

            # =================================================
            # Model
            # =================================================

            model = make_model(model_name)

            model.fit(
                X_fold_train_imp,
                y_fold_train,
            )

            # =================================================
            # OOF Probability
            # =================================================

            fold_probability = model.predict_proba(X_fold_valid_imp)[:, 1]

            oof_probability[valid_idx] = fold_probability

            # =================================================
            # Fold Metrics
            # =================================================

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
                    "TRAIN_POSITIVE": int(y_fold_train.sum()),
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
                f" | ROC-AUC "
                f"{fold_roc:.4f}"
            )

        # =====================================================
        # OOF Check
        # =====================================================

        if np.isnan(oof_probability).any():

            raise RuntimeError(f"{experiment_name}: " f"OOF NaN 발생")

        # =====================================================
        # 전체 OOF Metric
        # =====================================================

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
# 14. Train-only Ranking
#
# Feature + Model 선택 기준:
#
# OOF PR-AUC 최대
#
# ROC는 tie 참고용.
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
print("TRAIN ONLY 6H RANKING")
print("=" * 80)


print(ranking_df.round(4).to_string(index=False))


# ============================================================
# 15. Best Experiment
# ============================================================

best_experiment = str(ranking_df.iloc[0]["EXPERIMENT"])


best_result = experiment_results[best_experiment]


best_probability = best_result["oof_probability"]


print()

print("=" * 80)
print("6H BEST TRAIN-ONLY EXPERIMENT")
print("=" * 80)


print("Experiment:", best_experiment)


print("Feature Set:", best_result["feature_set"])


print("Model:", best_result["model"])


print("Feature Count:", len(best_result["feature_cols"]))


print("OOF PR-AUC:", round(best_result["oof_pr"], 4))


print("OOF ROC-AUC:", round(best_result["oof_roc"], 4))


# ============================================================
# 16. Threshold 평가 함수
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
# 17. Threshold Grid
#
# 6H는 Positive가 매우 적기 때문에
# 아주 낮은 구간까지 미리 탐색.
#
# Validation 결과를 보고 범위를 변경하지 않는다.
# ============================================================

threshold_grid = np.unique(
    np.concatenate(
        [
            # 0.0001 ~ 0.0009
            np.arange(0.0001, 0.0010, 0.0001),
            # 0.001 ~ 0.009
            np.arange(0.001, 0.010, 0.001),
            # 0.01 ~ 0.095
            np.arange(0.01, 0.10, 0.005),
            # 0.10 ~ 0.80
            np.arange(0.10, 0.81, 0.01),
        ]
    )
)


threshold_grid = [round(float(x), 4) for x in threshold_grid]


# ============================================================
# 18. OOF Threshold 전체 평가
# ============================================================

threshold_rows = []


for threshold in threshold_grid:

    threshold_rows.append(
        evaluate_threshold(
            probability=best_probability,
            threshold=threshold,
        )
    )


threshold_df = pd.DataFrame(threshold_rows)


# ============================================================
# 19. 참고용 MAX F2
#
# 출력은 하지만 LOCK 기준은 아님.
# ============================================================

max_f2_idx = threshold_df["F2"].idxmax()


max_f2_row = threshold_df.loc[max_f2_idx]


print()

print("=" * 80)
print("6H OOF MAX F2 - REFERENCE")
print("=" * 80)


print(max_f2_row.round(4).to_string())


# ============================================================
# 20. 사전 고정 운영정책
#
# Recall >= 0.75
# 중 Precision 최대
#
# tie-break:
# FP 최소
# F2 최대
# threshold 최대
# ============================================================

policy_candidates = threshold_df[threshold_df["RECALL"] >= RECALL_TARGET].copy()


if policy_candidates.empty:

    raise RuntimeError(
        f"Recall >= "
        f"{RECALL_TARGET:.2f} "
        f"조건을 만족하는 "
        f"Threshold가 없습니다."
    )


policy_candidates = policy_candidates.sort_values(
    by=[
        "PRECISION",
        "FP",
        "F2",
        "THRESHOLD",
    ],
    ascending=[
        False,
        True,
        False,
        False,
    ],
).reset_index(drop=True)


best_threshold_row = policy_candidates.iloc[0]


best_threshold = float(best_threshold_row["THRESHOLD"])


print()

print("=" * 80)
print("6H TRAIN-ONLY OPERATING THRESHOLD")
print("=" * 80)


print(f"Policy: " f"Recall >= " f"{RECALL_TARGET:.2f}")


print()


print(best_threshold_row.round(4).to_string())


# ============================================================
# 21. LOCK 정보
# ============================================================

lock_info = {
    "protocol": "TRAIN_ONLY_CV",
    "target": "RISK_6H",
    "censor": "CENSORED_6H",
    "feature_set": best_result["feature_set"],
    "model": best_result["model"],
    "feature_count": len(best_result["feature_cols"]),
    "features": best_result["feature_cols"],
    "threshold_policy": "RECALL_TARGET",
    "recall_target": RECALL_TARGET,
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
# 22. OOF 결과 저장
# ============================================================

oof_df = train[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
    ]
].copy()


oof_df["PRED_PROBA_6H"] = best_probability


oof_df["PRED_6H"] = (best_probability >= best_threshold).astype(int)


# ============================================================
# 23. 파일 저장
# ============================================================

ranking_df.to_csv(
    OUTPUT_DIR / "restart_6h_train_only_ranking.csv",
    index=False,
    encoding="utf-8-sig",
)


pd.DataFrame(fold_rows).to_csv(
    OUTPUT_DIR / "restart_6h_train_only_folds.csv",
    index=False,
    encoding="utf-8-sig",
)


threshold_df.to_csv(
    OUTPUT_DIR / "restart_6h_train_only_thresholds.csv",
    index=False,
    encoding="utf-8-sig",
)


oof_df.to_csv(
    OUTPUT_DIR / "restart_6h_train_only_oof.csv",
    index=False,
    encoding="utf-8-sig",
)


with (OUTPUT_DIR / "restart_6h_lock.json").open("w", encoding="utf-8") as f:

    json.dump(
        lock_info,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 24. 완료
# ============================================================

print()

print("=" * 80)
print("6H TRAIN-ONLY LOCK COMPLETE")
print("=" * 80)


print("Feature Set:", lock_info["feature_set"])


print("Model:", lock_info["model"])


print("Feature Count:", lock_info["feature_count"])


print("Threshold Policy:", f"Recall >= " f"{lock_info['recall_target']}")


print("Threshold:", lock_info["threshold"])


print("OOF PR-AUC:", round(lock_info["oof_pr_auc"], 4))


print("OOF Precision:", round(lock_info["oof_precision"], 4))


print("OOF Recall:", round(lock_info["oof_recall"], 4))


print("OOF F2:", round(lock_info["oof_f2"], 4))


print("OOF FP:", lock_info["oof_fp"])


print("OOF FN:", lock_info["oof_fn"])


print()

print("※ Validation은 사용하지 않았습니다.")

print("※ 결과가 마음에 안 들어도 " "Validation로 설정을 바꾸면 안 됩니다.")
