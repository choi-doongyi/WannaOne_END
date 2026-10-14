# ============================================================
# restart_05_final_validation.py
#
# FINAL VALIDATION - ONE SHOT
#
# ------------------------------------------------------------
# 중요
# ------------------------------------------------------------
# 이 파일에서는:
#
# - Feature 선택 X
# - Model 선택 X
# - Threshold 선택 X
# - State Machine 튜닝 X
#
# Train-only 단계에서 LOCK한 설정을
# Validation에 단 한 번 적용한다.
#
# ------------------------------------------------------------
# 24H
# ------------------------------------------------------------
# Model      : XGBoost
# Feature    : SENSOR + MEAN_DIFF
# Threshold  : restart_24h_lock.json
#
# ------------------------------------------------------------
# 6H
# ------------------------------------------------------------
# Model      : XGBoost
# Feature    : SENSOR_ONLY
# Threshold  : restart_6h_lock.json
#
# ------------------------------------------------------------
# State Machine
# ------------------------------------------------------------
# CAUTION HOLD = 6H
# DANGER HOLD  = 6H
#
# Validation 결과를 본 뒤
# 어떠한 설정도 변경하지 않는다.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

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

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


VALID_PATH = OUTPUT_DIR / "casting_validation_features_v22.csv"


LOCK_24H_PATH = OUTPUT_DIR / "restart_24h_lock.json"


LOCK_6H_PATH = OUTPUT_DIR / "restart_6h_lock.json"


# ============================================================
# 1. State Machine LOCK
# ============================================================

CAUTION_HOLD_HOURS = 6

DANGER_HOLD_HOURS = 6


# ============================================================
# 2. 파일 존재 여부 사전 검사
#
# 결과 보기 전에 오류를 최대한 잡는다.
# ============================================================

required_paths = [
    TRAIN_PATH,
    VALID_PATH,
    LOCK_24H_PATH,
    LOCK_6H_PATH,
]


for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(f"필수 파일 없음: {path}")


# ============================================================
# 3. LOCK 정보 로드
# ============================================================

with LOCK_24H_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    lock_24h = json.load(f)


with LOCK_6H_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    lock_6h = json.load(f)


# ============================================================
# 4. LOCK 검사
#
# 예상했던 최종 설정과 다르면
# Validation 실행 전에 중단한다.
# ============================================================

if lock_24h["model"] != "XGB":

    raise RuntimeError("24H LOCK Model이 XGB가 아닙니다.")


if lock_6h["model"] != "XGB":

    raise RuntimeError("6H LOCK Model이 XGB가 아닙니다.")


threshold_24h = float(lock_24h["threshold"])


threshold_6h = float(lock_6h["threshold"])


features_24h = list(lock_24h["features"])


features_6h = list(lock_6h["features"])


print("=" * 80)
print("FINAL VALIDATION - LOCK CHECK")
print("=" * 80)


print()

print("24H Model:", lock_24h["model"])


print("24H Feature Set:", lock_24h["feature_set"])


print("24H Feature Count:", len(features_24h))


print("24H Threshold:", threshold_24h)


print()

print("6H Model:", lock_6h["model"])


print("6H Feature Set:", lock_6h["feature_set"])


print("6H Feature Count:", len(features_6h))


print("6H Threshold:", threshold_6h)


print()

print("CAUTION HOLD:", CAUTION_HOLD_HOURS, "H")


print("DANGER HOLD:", DANGER_HOLD_HOURS, "H")


# ============================================================
# 5. Train / Validation 로드
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


valid_df = pd.read_csv(VALID_PATH)


print()

print("=" * 80)
print("DATA CHECK")
print("=" * 80)


print("Train:", train_df.shape)


print("Validation:", valid_df.shape)


print("Train Scenario:", train_df["SCENARIO_ID"].nunique())


print("Validation Scenario:", valid_df["SCENARIO_ID"].nunique())


# ============================================================
# 6. Train / Validation Scenario 겹침 검사
# ============================================================

train_scenarios = set(train_df["SCENARIO_ID"].unique())


valid_scenarios = set(valid_df["SCENARIO_ID"].unique())


scenario_overlap = train_scenarios & valid_scenarios


if scenario_overlap:

    raise RuntimeError(
        f"Train / Validation Scenario Leakage: " f"{sorted(scenario_overlap)[:10]}"
    )


# ============================================================
# 7. 시간 변환
# ============================================================

for df in [
    train_df,
    valid_df,
]:

    df["OBSERVED_AT"] = pd.to_datetime(
        df["OBSERVED_AT"],
        format="mixed",
        utc=True,
        errors="coerce",
    )

    if df["OBSERVED_AT"].isna().any():

        raise RuntimeError("OBSERVED_AT 변환 실패")


# ============================================================
# 8. Mean Diff Feature 생성
#
# Train-only 단계와 정확히 동일한 정의.
# Validation 결과에 따라 바꾸지 않는다.
# ============================================================

MEAN_DIFF_SENSORS = [
    "CONTROL_DELAY",
    "MLD-LVL",
    "OSC-STK",
    "OSC-FRQ",
]


def add_mean_diff_features(df):

    df = df.copy()

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

        for (
            new_col,
            short_col,
            long_col,
        ) in definitions:

            if short_col not in df.columns:

                raise KeyError(f"Missing column: " f"{short_col}")

            if long_col not in df.columns:

                raise KeyError(f"Missing column: " f"{long_col}")

            df[new_col] = df[short_col] - df[long_col]

    return df


train_df = add_mean_diff_features(train_df)


valid_df = add_mean_diff_features(valid_df)


# ============================================================
# 9. LOCK Feature 존재 여부 검사
# ============================================================

missing_24h_train = [col for col in features_24h if col not in train_df.columns]


missing_24h_valid = [col for col in features_24h if col not in valid_df.columns]


missing_6h_train = [col for col in features_6h if col not in train_df.columns]


missing_6h_valid = [col for col in features_6h if col not in valid_df.columns]


if missing_24h_train:

    raise KeyError(f"24H Train Feature 누락: " f"{missing_24h_train}")


if missing_24h_valid:

    raise KeyError(f"24H Validation Feature 누락: " f"{missing_24h_valid}")


if missing_6h_train:

    raise KeyError(f"6H Train Feature 누락: " f"{missing_6h_train}")


if missing_6h_valid:

    raise KeyError(f"6H Validation Feature 누락: " f"{missing_6h_valid}")


print()

print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)


print("24H Feature:", len(features_24h))


print("6H Feature:", len(features_6h))


# ============================================================
# 10. 24H Train Label
# ============================================================

train_24h = train_df[
    (train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()
].copy()


y_train_24h = train_24h["RISK_24H"].astype(int)


# ============================================================
# 11. 6H Train Label
# ============================================================

train_6h = train_df[(train_df["CENSORED_6H"] == 0) & train_df["RISK_6H"].notna()].copy()


y_train_6h = train_6h["RISK_6H"].astype(int)


print()

print("=" * 80)
print("TRAIN LABEL CHECK")
print("=" * 80)


print("24H Train usable:", len(train_24h))


print("24H Positive:", int(y_train_24h.sum()))


print("6H Train usable:", len(train_6h))


print("6H Positive:", int(y_train_6h.sum()))


# ============================================================
# 12. XGBoost 생성
#
# restart_01 / restart_02와
# 동일한 Hyperparameter.
# ============================================================


def make_xgb(y):

    negative_count = int((y == 0).sum())

    positive_count = int((y == 1).sum())

    scale_pos_weight = negative_count / positive_count

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
# 13. 24H 최종 Train 전체 학습
#
# Imputer도 Train에서만 fit.
# ============================================================

X_train_24h = train_24h[features_24h].copy()


X_valid_24h_all = valid_df[features_24h].copy()


imputer_24h = SimpleImputer(strategy="median")


X_train_24h_imp = imputer_24h.fit_transform(X_train_24h)


X_valid_24h_imp = imputer_24h.transform(X_valid_24h_all)


model_24h = make_xgb(y_train_24h)


model_24h.fit(
    X_train_24h_imp,
    y_train_24h,
)


valid_prob_24h = model_24h.predict_proba(X_valid_24h_imp)[:, 1]


# ============================================================
# 14. 6H 최종 Train 전체 학습
# ============================================================

X_train_6h = train_6h[features_6h].copy()


X_valid_6h_all = valid_df[features_6h].copy()


imputer_6h = SimpleImputer(strategy="median")


X_train_6h_imp = imputer_6h.fit_transform(X_train_6h)


X_valid_6h_imp = imputer_6h.transform(X_valid_6h_all)


model_6h = make_xgb(y_train_6h)


model_6h.fit(
    X_train_6h_imp,
    y_train_6h,
)


valid_prob_6h = model_6h.predict_proba(X_valid_6h_imp)[:, 1]


# ============================================================
# 15. Validation Prediction Table
# ============================================================

result_df = valid_df[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_24H",
        "CENSORED_24H",
        "RISK_6H",
        "CENSORED_6H",
    ]
].copy()


result_df["PRED_PROBA_24H"] = valid_prob_24h


result_df["PRED_PROBA_6H"] = valid_prob_6h


result_df["CAUTION_TRIGGER"] = result_df["PRED_PROBA_24H"] >= threshold_24h


result_df["DANGER_TRIGGER"] = result_df["PRED_PROBA_6H"] >= threshold_6h


# ============================================================
# 16. RAW Warning
#
# DANGER > CAUTION > NORMAL
# ============================================================

result_df["RAW_WARNING"] = np.select(
    [
        result_df["DANGER_TRIGGER"],
        result_df["CAUTION_TRIGGER"],
    ],
    [
        "DANGER",
        "CAUTION",
    ],
    default="NORMAL",
)


# ============================================================
# 17. State Machine
# ============================================================


def apply_state_machine(scenario_df):

    scenario_df = scenario_df.sort_values("OBSERVED_AT").copy()

    state_list = []

    caution_until = None

    danger_until = None

    for row in scenario_df.itertuples():

        current_time = row.OBSERVED_AT

        # ----------------------------------------------------
        # DANGER Trigger
        # ----------------------------------------------------

        if bool(row.DANGER_TRIGGER):

            new_until = current_time + pd.Timedelta(hours=DANGER_HOLD_HOURS)

            if danger_until is None or new_until > danger_until:

                danger_until = new_until

        # ----------------------------------------------------
        # CAUTION Trigger
        # ----------------------------------------------------

        if bool(row.CAUTION_TRIGGER):

            new_until = current_time + pd.Timedelta(hours=CAUTION_HOLD_HOURS)

            if caution_until is None or new_until > caution_until:

                caution_until = new_until

        # ----------------------------------------------------
        # 활성 상태 확인
        # ----------------------------------------------------

        danger_active = danger_until is not None and current_time <= danger_until

        caution_active = caution_until is not None and current_time <= caution_until

        # ----------------------------------------------------
        # Priority
        # ----------------------------------------------------

        if danger_active:

            state = "DANGER"

        elif caution_active:

            state = "CAUTION"

        else:

            state = "NORMAL"

        state_list.append(state)

    scenario_df["STATE_WARNING"] = state_list

    return scenario_df


# ============================================================
# 18. Scenario별 State Machine 적용
#
# groupby.apply 대신 명시적 loop 사용.
# ============================================================

state_parts = []


for scenario_id, scenario_df in result_df.groupby(
    "SCENARIO_ID",
    sort=False,
):

    temp = apply_state_machine(scenario_df)

    temp["SCENARIO_ID"] = scenario_id

    state_parts.append(temp)


result_df = pd.concat(
    state_parts,
    ignore_index=True,
)


result_df = result_df.sort_values(
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
    ]
).reset_index(drop=True)


# ============================================================
# 19. Point Metric 함수
# ============================================================


def evaluate_point(
    df,
    target_col,
    censor_col,
    prob_col,
    threshold,
):

    usable = df[(df[censor_col] == 0) & df[target_col].notna()].copy()

    y_true = usable[target_col].astype(int)

    probability = usable[prob_col].astype(float)

    y_pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f2 = fbeta_score(
        y_true,
        y_pred,
        beta=2,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_true,
        probability,
    )

    pr_auc = average_precision_score(
        y_true,
        probability,
    )

    return {
        "N": len(usable),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
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
# 20. FINAL Validation Point Metrics
# ============================================================

metric_24h = evaluate_point(
    result_df,
    target_col="RISK_24H",
    censor_col="CENSORED_24H",
    prob_col="PRED_PROBA_24H",
    threshold=threshold_24h,
)


metric_6h = evaluate_point(
    result_df,
    target_col="RISK_6H",
    censor_col="CENSORED_6H",
    prob_col="PRED_PROBA_6H",
    threshold=threshold_6h,
)


metric_df = pd.DataFrame(
    [
        {
            "TARGET": "24H_CAUTION",
            **metric_24h,
        },
        {
            "TARGET": "6H_DANGER",
            **metric_6h,
        },
    ]
)


# ============================================================
# 21. 경보 분포
# ============================================================


def make_distribution(
    df,
    col,
    system_name,
):

    count = (
        df[col]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "CAUTION",
                "DANGER",
            ],
            fill_value=0,
        )
    )

    return pd.DataFrame(
        {
            "SYSTEM": system_name,
            "WARNING_LEVEL": count.index,
            "COUNT": count.values,
            "RATIO": (count.values / len(df)),
        }
    )


raw_distribution = make_distribution(
    result_df,
    "RAW_WARNING",
    "RAW",
)


state_distribution = make_distribution(
    result_df,
    "STATE_WARNING",
    "STATE_MACHINE",
)


distribution_df = pd.concat(
    [
        raw_distribution,
        state_distribution,
    ],
    ignore_index=True,
)


# ============================================================
# 22. Transition Count
# ============================================================


def count_transitions(
    df,
    col,
):

    temp = df[
        [
            "SCENARIO_ID",
            "OBSERVED_AT",
            col,
        ]
    ].copy()

    temp["PREV"] = temp.groupby("SCENARIO_ID")[col].shift(1)

    changed = temp["PREV"].notna() & (temp[col] != temp["PREV"])

    return int(changed.sum())


raw_transitions = count_transitions(
    result_df,
    "RAW_WARNING",
)


state_transitions = count_transitions(
    result_df,
    "STATE_WARNING",
)


transition_reduction = raw_transitions - state_transitions


transition_reduction_ratio = (
    transition_reduction / raw_transitions if raw_transitions > 0 else 0
)


# ============================================================
# 23. FINAL 결과 출력
#
# 여기서 처음 Validation 성능을 본다.
# ============================================================

print()

print("=" * 80)
print("FINAL VALIDATION POINT METRICS")
print("=" * 80)


print(metric_df.round(4).to_string(index=False))


print()

print("=" * 80)
print("FINAL VALIDATION WARNING DISTRIBUTION")
print("=" * 80)


print(distribution_df.round(4).to_string(index=False))


print()

print("=" * 80)
print("FINAL VALIDATION TRANSITION")
print("=" * 80)


print("RAW transitions:", raw_transitions)


print("STATE transitions:", state_transitions)


print("Reduction:", transition_reduction)


print("Reduction Ratio:", round(transition_reduction_ratio, 4))


# ============================================================
# 24. 저장
# ============================================================

prediction_path = OUTPUT_DIR / "restart_final_validation_predictions.csv"


metric_path = OUTPUT_DIR / "restart_final_validation_point_metrics.csv"


distribution_path = OUTPUT_DIR / "restart_final_validation_warning_distribution.csv"


result_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig",
)


metric_df.to_csv(
    metric_path,
    index=False,
    encoding="utf-8-sig",
)


distribution_df.to_csv(
    distribution_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 25. 완료
# ============================================================

print()

print("=" * 80)
print("FINAL VALIDATION COMPLETE")
print("=" * 80)


print("Prediction:", prediction_path)


print("Metrics:", metric_path)


print("Distribution:", distribution_path)


print()

print("※ Validation 평가 완료")


print("※ 이 결과 이후 " "Feature / Model / Threshold / Hold Time 변경 금지")


print("※ 다음 단계는 Test 예측뿐입니다.")
