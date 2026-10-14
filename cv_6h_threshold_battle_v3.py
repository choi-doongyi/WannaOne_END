# ============================================================
# cv_6h_threshold_battle_v3.py
#
# 6H DANGER - THRESHOLD BATTLE V3
#
# Feature:
# BASE 240 고정
#
# Model:
# RF / XGB / LGBM
#
# 목적:
# 모델별 OOF prediction을 이용해서
# threshold를 촘촘하게 탐색
#
# 비교 정책:
# 1. MAX_F2
# 2. RECALL_075
#    - Recall >= 0.75 조건 중 Precision 최대
# 3. RECALL_080
#    - Recall >= 0.80 조건 중 Precision 최대
#
# Validation:
# Train OOF에서 정책/threshold 결정 후
# 참고용 최종 평가
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

TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v25.csv"

VALID_PATH = OUTPUT_DIR / "casting_validation_features_v25.csv"


# ============================================================
# 1. 데이터 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)

valid_df = pd.read_csv(VALID_PATH)


print("=" * 80)
print("6H THRESHOLD BATTLE V3")
print("RF VS XGB VS LGBM")
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
print("LABEL CHECK")
print("=" * 80)

print("Train:", train["RISK_6H"].value_counts())

print()

print("Validation:", valid["RISK_6H"].value_counts())


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
# 5. Trend / Mean Diff 제외
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
# 6. BASE 240
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


# ============================================================
# 7. X / y / groups
# ============================================================

X = train[base_features].copy()

X_valid = valid[base_features].copy()

y = train["RISK_6H"].copy()

y_valid = valid["RISK_6H"].copy()

groups = train["SCENARIO_ID"].copy()


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
# 9. 모델 생성
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

    elif model_name == "LGBM":

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

    else:

        raise ValueError(f"Unknown model: {model_name}")


# ============================================================
# 10. 평가 함수
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
# 11. Threshold grid
#
# 매우 낮은 값까지 촘촘하게 탐색
# ============================================================

threshold_grid = np.unique(
    np.concatenate(
        [
            # 매우 낮은 구간
            np.arange(0.0001, 0.0010, 0.0001),
            # 낮은 구간
            np.arange(0.001, 0.010, 0.001),
            # 중간 구간
            np.arange(0.01, 0.101, 0.005),
            # 높은 구간
            np.arange(0.11, 0.51, 0.01),
        ]
    )
)


threshold_grid = [round(float(x), 4) for x in threshold_grid]


# ============================================================
# 12. CV
# ============================================================

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


MODEL_NAMES = [
    "RF",
    "XGB",
    "LGBM",
]


# ============================================================
# 13. OOF 생성
# ============================================================

model_results = {}

fold_rows = []


for model_name in MODEL_NAMES:

    print()

    print("=" * 80)
    print(f"{model_name} - 5 FOLD CV")
    print("=" * 80)

    oof_probability = np.full(len(train), np.nan, dtype=float)

    for fold, (train_idx, fold_valid_idx) in enumerate(cv.split(X, y, groups), start=1):

        X_fold_train = X.iloc[train_idx].copy()

        X_fold_valid = X.iloc[fold_valid_idx].copy()

        y_fold_train = y.iloc[train_idx].copy()

        y_fold_valid = y.iloc[fold_valid_idx].copy()

        train_groups = set(groups.iloc[train_idx])

        valid_groups = set(groups.iloc[fold_valid_idx])

        if train_groups & valid_groups:

            raise RuntimeError(f"{model_name} " f"Fold {fold}: " f"Scenario Leakage")

        # ====================================================
        # Imputer
        # ====================================================

        imputer = SimpleImputer(strategy="median")

        X_train_imp = imputer.fit_transform(X_fold_train)

        X_fold_valid_imp = imputer.transform(X_fold_valid)

        # ====================================================
        # 모델
        # ====================================================

        model = make_model(model_name)

        model.fit(X_train_imp, y_fold_train)

        # ====================================================
        # 예측
        # ====================================================

        fold_probability = model.predict_proba(X_fold_valid_imp)[:, 1]

        oof_probability[fold_valid_idx] = fold_probability

        # ====================================================
        # Fold 성능
        # ====================================================

        fold_pr = average_precision_score(y_fold_valid, fold_probability)

        fold_roc = roc_auc_score(y_fold_valid, fold_probability)

        fold_rows.append(
            {
                "MODEL": model_name,
                "FOLD": fold,
                "PR_AUC": fold_pr,
                "ROC_AUC": fold_roc,
                "VALID_POSITIVE": int(y_fold_valid.sum()),
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

        raise RuntimeError(f"{model_name}: OOF NaN")

    oof_pr = average_precision_score(y, oof_probability)

    oof_roc = roc_auc_score(y, oof_probability)

    print()

    print(f"{model_name} " f"OOF PR-AUC: " f"{oof_pr:.4f}")

    print(f"{model_name} " f"OOF ROC-AUC: " f"{oof_roc:.4f}")

    model_results[model_name] = {
        "oof_probability": oof_probability,
        "oof_pr": oof_pr,
        "oof_roc": oof_roc,
    }


# ============================================================
# 14. 모델별 Threshold 전체 탐색
# ============================================================

threshold_rows = []


for model_name in MODEL_NAMES:

    probability = model_results[model_name]["oof_probability"]

    for threshold in threshold_grid:

        metrics = evaluate_threshold(
            y_true=y, probability=probability, threshold=threshold
        )

        threshold_rows.append(
            {
                "MODEL": model_name,
                **metrics,
            }
        )


threshold_df = pd.DataFrame(threshold_rows)


# ============================================================
# 15. 모델별 MAX F2
# ============================================================

max_f2_rows = []


for model_name in MODEL_NAMES:

    model_table = threshold_df[threshold_df["MODEL"] == model_name].copy()

    best_idx = model_table["F2"].idxmax()

    row = model_table.loc[best_idx].copy()

    row["POLICY"] = "MAX_F2"

    max_f2_rows.append(row)


max_f2_df = pd.DataFrame(max_f2_rows)


# ============================================================
# 16. Recall >= target 정책
# ============================================================


def select_recall_policy(threshold_df, model_name, recall_target):

    table = threshold_df[
        (threshold_df["MODEL"] == model_name)
        & (threshold_df["RECALL"] >= recall_target)
    ].copy()

    if table.empty:

        return None

    # --------------------------------------------------------
    # Recall 조건을 만족하면서
    # Precision 최대
    #
    # Precision 동일하면
    # FP 적은 것
    #
    # 그래도 동일하면
    # Threshold 높은 것
    # --------------------------------------------------------

    table = table.sort_values(
        by=[
            "PRECISION",
            "FP",
            "THRESHOLD",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    ).reset_index(drop=True)

    row = table.iloc[0].copy()

    row["POLICY"] = f"RECALL_{int(recall_target * 100):02d}"

    return row


recall_policy_rows = []


for recall_target in [
    0.75,
    0.80,
]:

    for model_name in MODEL_NAMES:

        row = select_recall_policy(
            threshold_df=threshold_df,
            model_name=model_name,
            recall_target=recall_target,
        )

        if row is not None:

            recall_policy_rows.append(row)


recall_policy_df = pd.DataFrame(recall_policy_rows)


# ============================================================
# 17. 정책 통합
# ============================================================

policy_df = pd.concat(
    [
        max_f2_df,
        recall_policy_df,
    ],
    ignore_index=True,
)


policy_df = policy_df[
    [
        "MODEL",
        "POLICY",
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


print()

print("=" * 80)
print("6H THRESHOLD POLICY TABLE")
print("=" * 80)


print(
    policy_df.sort_values(
        [
            "POLICY",
            "F2",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 18. 모델 AP Ranking
# ============================================================

ranking_rows = []


for model_name in MODEL_NAMES:

    ranking_rows.append(
        {
            "MODEL": model_name,
            "OOF_PR_AUC": model_results[model_name]["oof_pr"],
            "OOF_ROC_AUC": model_results[model_name]["oof_roc"],
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
# 19. 운영 후보 선정
#
# 기본 후보:
# Recall >= 0.75 정책 중
# F2 최대
#
# 이유:
# 6H는 고장 임박 경보이므로
# Recall을 최소 75% 확보
# ============================================================

candidate_table = policy_df[policy_df["POLICY"] == "RECALL_75"].copy()


if candidate_table.empty:

    raise RuntimeError("Recall 0.75 이상 후보 없음")


candidate_table = candidate_table.sort_values(
    [
        "F2",
        "PRECISION",
    ],
    ascending=[
        False,
        False,
    ],
).reset_index(drop=True)


selected_row = candidate_table.iloc[0]


selected_model_name = str(selected_row["MODEL"])


selected_threshold = float(selected_row["THRESHOLD"])


selected_policy = str(selected_row["POLICY"])


print()

print("=" * 80)
print("6H SELECTED OPERATING POLICY")
print("=" * 80)


print("Policy:", selected_policy)


print("Model:", selected_model_name)


print("Threshold:", selected_threshold)


print("OOF Precision:", round(float(selected_row["PRECISION"]), 4))


print("OOF Recall:", round(float(selected_row["RECALL"]), 4))


print("OOF F2:", round(float(selected_row["F2"]), 4))


print("OOF FP:", int(selected_row["FP"]))


print("OOF FN:", int(selected_row["FN"]))


# ============================================================
# 20. Full Train 학습
# ============================================================

final_imputer = SimpleImputer(strategy="median")


X_train_imp = final_imputer.fit_transform(X)


X_valid_imp = final_imputer.transform(X_valid)


final_model = make_model(selected_model_name)


final_model.fit(X_train_imp, y)


# ============================================================
# 21. Validation
# ============================================================

valid_probability = final_model.predict_proba(X_valid_imp)[:, 1]


valid_pr_auc = average_precision_score(y_valid, valid_probability)


valid_roc_auc = roc_auc_score(y_valid, valid_probability)


valid_metrics = evaluate_threshold(
    y_true=y_valid, probability=valid_probability, threshold=selected_threshold
)


print()

print("=" * 80)
print("FINAL VALIDATION - SELECTED 6H POLICY")
print("=" * 80)


print("Model:", selected_model_name)


print("Policy:", selected_policy)


print("Threshold:", selected_threshold)


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
# 22. Validation Predictions 저장
# ============================================================

validation_prediction_df = valid[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
    ]
].copy()


validation_prediction_df["PRED_PROBA_6H"] = valid_probability


validation_prediction_df["PRED_6H"] = (valid_probability >= selected_threshold).astype(
    int
)


# ============================================================
# 23. OOF Prediction 저장
# ============================================================

oof_prediction_df = train[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
    ]
].copy()


for model_name in MODEL_NAMES:

    oof_prediction_df[f"{model_name}_PROBA"] = model_results[model_name][
        "oof_probability"
    ]


# ============================================================
# 24. 저장
# ============================================================

fold_df = pd.DataFrame(fold_rows)


ranking_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_model_ranking.csv", index=False, encoding="utf-8-sig"
)


threshold_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_all_thresholds.csv", index=False, encoding="utf-8-sig"
)


policy_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_policy_table.csv", index=False, encoding="utf-8-sig"
)


fold_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_folds.csv", index=False, encoding="utf-8-sig"
)


validation_prediction_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_validation_predictions.csv",
    index=False,
    encoding="utf-8-sig",
)


oof_prediction_df.to_csv(
    OUTPUT_DIR / "casting_6h_v3_oof_predictions.csv", index=False, encoding="utf-8-sig"
)


print()

print("=" * 80)
print("SAVE COMPLETE - 6H V3")
print("=" * 80)
