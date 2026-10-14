# ============================================================
# restart_06_predict_test.py
#
# FINAL TEST PREDICTION
#
# ------------------------------------------------------------
# 중요
# ------------------------------------------------------------
# - Validation 사용 X
# - Test label 사용 X
# - Feature 선택 X
# - Model 선택 X
# - Threshold 변경 X
# - Hold Time 변경 X
#
# Train-only 단계에서 LOCK한 모델을
# Test 72개 Scenario에 그대로 적용한다.
#
# 최종 출력:
# - 24H probability
# - 6H probability
# - RAW WARNING
# - STATE WARNING
#
# Test는 정답이 없으므로
# Accuracy / Recall / PR-AUC 등을 계산하지 않는다.
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


TEST_PATH = OUTPUT_DIR / "casting_test_features_v22.csv"


LOCK_24H_PATH = OUTPUT_DIR / "restart_24h_lock.json"


LOCK_6H_PATH = OUTPUT_DIR / "restart_6h_lock.json"


# ============================================================
# 1. LOCKED State Machine
# ============================================================

CAUTION_HOLD_HOURS = 6

DANGER_HOLD_HOURS = 6


# ============================================================
# 2. 파일 존재 검사
# ============================================================

required_paths = [
    TRAIN_PATH,
    TEST_PATH,
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
# 4. LOCK 설정 확인
# ============================================================

if lock_24h["model"] != "XGB":

    raise RuntimeError("24H LOCK 모델이 XGB가 아닙니다.")


if lock_6h["model"] != "XGB":

    raise RuntimeError("6H LOCK 모델이 XGB가 아닙니다.")


features_24h = list(lock_24h["features"])


features_6h = list(lock_6h["features"])


threshold_24h = float(lock_24h["threshold"])


threshold_6h = float(lock_6h["threshold"])


print("=" * 80)
print("FINAL TEST PREDICTION - LOCK CHECK")
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
# 5. Train / Test 로드
#
# Validation은 절대 읽지 않는다.
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)


test_df = pd.read_csv(TEST_PATH)


print()

print("=" * 80)
print("DATA CHECK")
print("=" * 80)


print("Train:", train_df.shape)

print("Test:", test_df.shape)


print("Train Scenario:", train_df["SCENARIO_ID"].nunique())

print("Test Scenario:", test_df["SCENARIO_ID"].nunique())


# ============================================================
# 6. Train / Test Scenario Leakage 검사
# ============================================================

train_scenarios = set(train_df["SCENARIO_ID"].unique())


test_scenarios = set(test_df["SCENARIO_ID"].unique())


overlap = train_scenarios & test_scenarios


if overlap:

    raise RuntimeError(f"Train/Test Scenario 겹침: " f"{sorted(overlap)[:10]}")


# ============================================================
# 7. 시간 변환
# ============================================================

for df in [
    train_df,
    test_df,
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
# Train-only 개발과 동일한 정의.
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

        for new_col, short_col, long_col in definitions:

            if short_col not in df.columns:

                raise KeyError(f"컬럼 없음: {short_col}")

            if long_col not in df.columns:

                raise KeyError(f"컬럼 없음: {long_col}")

            df[new_col] = df[short_col] - df[long_col]

    return df


train_df = add_mean_diff_features(train_df)


test_df = add_mean_diff_features(test_df)


# ============================================================
# 9. Feature 존재 검사
# ============================================================

for name, feature_list in [
    (
        "24H",
        features_24h,
    ),
    (
        "6H",
        features_6h,
    ),
]:

    missing_train = [col for col in feature_list if col not in train_df.columns]

    missing_test = [col for col in feature_list if col not in test_df.columns]

    if missing_train:

        raise KeyError(f"{name} Train Feature 누락: " f"{missing_train}")

    if missing_test:

        raise KeyError(f"{name} Test Feature 누락: " f"{missing_test}")


print()

print("=" * 80)
print("FEATURE CHECK")
print("=" * 80)


print("24H Feature:", len(features_24h))


print("6H Feature:", len(features_6h))


# ============================================================
# 10. Train 24H usable
# ============================================================

train_24h = train_df[
    (train_df["CENSORED_24H"] == 0) & train_df["RISK_24H"].notna()
].copy()


y_train_24h = train_24h["RISK_24H"].astype(int)


# ============================================================
# 11. Train 6H usable
# ============================================================

train_6h = train_df[(train_df["CENSORED_6H"] == 0) & train_df["RISK_6H"].notna()].copy()


y_train_6h = train_6h["RISK_6H"].astype(int)


print()

print("=" * 80)
print("TRAIN LABEL CHECK")
print("=" * 80)


print("24H usable:", len(train_24h))


print("24H Positive:", int(y_train_24h.sum()))


print("6H usable:", len(train_6h))


print("6H Positive:", int(y_train_6h.sum()))


# ============================================================
# 12. XGBoost
#
# Validation 때 사용한 것과 완전히 동일.
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
# 13. 24H 최종 모델
# ============================================================

X_train_24h = train_24h[features_24h].copy()


X_test_24h = test_df[features_24h].copy()


imputer_24h = SimpleImputer(strategy="median")


X_train_24h_imp = imputer_24h.fit_transform(X_train_24h)


X_test_24h_imp = imputer_24h.transform(X_test_24h)


model_24h = make_xgb(y_train_24h)


model_24h.fit(
    X_train_24h_imp,
    y_train_24h,
)


test_prob_24h = model_24h.predict_proba(X_test_24h_imp)[:, 1]


# ============================================================
# 14. 6H 최종 모델
# ============================================================

X_train_6h = train_6h[features_6h].copy()


X_test_6h = test_df[features_6h].copy()


imputer_6h = SimpleImputer(strategy="median")


X_train_6h_imp = imputer_6h.fit_transform(X_train_6h)


X_test_6h_imp = imputer_6h.transform(X_test_6h)


model_6h = make_xgb(y_train_6h)


model_6h.fit(
    X_train_6h_imp,
    y_train_6h,
)


test_prob_6h = model_6h.predict_proba(X_test_6h_imp)[:, 1]


# ============================================================
# 15. Test Prediction Table
# ============================================================

result_df = test_df[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
    ]
].copy()


result_df["PRED_PROBA_24H"] = test_prob_24h


result_df["PRED_PROBA_6H"] = test_prob_6h


result_df["CAUTION_TRIGGER"] = result_df["PRED_PROBA_24H"] >= threshold_24h


result_df["DANGER_TRIGGER"] = result_df["PRED_PROBA_6H"] >= threshold_6h


# ============================================================
# 16. RAW WARNING
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
        # DANGER
        # ----------------------------------------------------

        if bool(row.DANGER_TRIGGER):

            new_until = current_time + pd.Timedelta(hours=DANGER_HOLD_HOURS)

            if danger_until is None or new_until > danger_until:

                danger_until = new_until

        # ----------------------------------------------------
        # CAUTION
        # ----------------------------------------------------

        if bool(row.CAUTION_TRIGGER):

            new_until = current_time + pd.Timedelta(hours=CAUTION_HOLD_HOURS)

            if caution_until is None or new_until > caution_until:

                caution_until = new_until

        # ----------------------------------------------------
        # Active 상태
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
# 19. RAW / STATE 분포
# ============================================================


def make_distribution(
    df,
    col,
    system_name,
):

    counts = (
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
            "WARNING_LEVEL": counts.index,
            "COUNT": counts.values,
            "RATIO": counts.values / len(df),
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
# 20. 상태 전환 횟수
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


# ============================================================
# 21. Scenario Summary
# ============================================================

warning_score = {
    "NORMAL": 0,
    "CAUTION": 1,
    "DANGER": 2,
}


result_df["STATE_SCORE"] = result_df["STATE_WARNING"].map(warning_score).astype(int)


scenario_summary = result_df.groupby(
    "SCENARIO_ID",
    as_index=False,
).agg(
    TOTAL_POINTS=(
        "STATE_WARNING",
        "size",
    ),
    CAUTION_TRIGGER_COUNT=(
        "CAUTION_TRIGGER",
        "sum",
    ),
    DANGER_TRIGGER_COUNT=(
        "DANGER_TRIGGER",
        "sum",
    ),
    MAX_STATE_SCORE=(
        "STATE_SCORE",
        "max",
    ),
    MAX_P24H=(
        "PRED_PROBA_24H",
        "max",
    ),
    MAX_P6H=(
        "PRED_PROBA_6H",
        "max",
    ),
)


scenario_summary["MAX_WARNING"] = scenario_summary["MAX_STATE_SCORE"].map(
    {
        0: "NORMAL",
        1: "CAUTION",
        2: "DANGER",
    }
)


# ============================================================
# 22. 출력
# ============================================================

print()

print("=" * 80)
print("TEST WARNING DISTRIBUTION")
print("=" * 80)


print(distribution_df.round(4).to_string(index=False))


print()

print("=" * 80)
print("TEST SCENARIO SUMMARY")
print("=" * 80)


print("Total Scenario:", len(scenario_summary))


print("NORMAL only Scenario:", int((scenario_summary["MAX_WARNING"] == "NORMAL").sum()))


print(
    "CAUTION max Scenario:", int((scenario_summary["MAX_WARNING"] == "CAUTION").sum())
)


print("DANGER Scenario:", int((scenario_summary["MAX_WARNING"] == "DANGER").sum()))


print()

print("=" * 80)
print("TEST TRANSITION")
print("=" * 80)


print("RAW transitions:", raw_transitions)


print("STATE transitions:", state_transitions)


if raw_transitions > 0:

    reduction_ratio = (raw_transitions - state_transitions) / raw_transitions

else:

    reduction_ratio = 0


print("Reduction Ratio:", round(reduction_ratio, 4))


# ============================================================
# 23. 저장
# ============================================================

prediction_path = OUTPUT_DIR / "restart_final_test_predictions.csv"


distribution_path = OUTPUT_DIR / "restart_final_test_warning_distribution.csv"


scenario_path = OUTPUT_DIR / "restart_final_test_scenario_summary.csv"


result_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig",
)


distribution_df.to_csv(
    distribution_path,
    index=False,
    encoding="utf-8-sig",
)


scenario_summary.to_csv(
    scenario_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 24. 완료
# ============================================================

print()

print("=" * 80)
print("FINAL TEST PREDICTION COMPLETE")
print("=" * 80)


print("Prediction:", prediction_path)


print("Distribution:", distribution_path)


print("Scenario Summary:", scenario_path)


print()

print("※ Test 정답을 사용하지 않았습니다.")


print("※ Test 성능 수치는 계산하지 않습니다.")


print(
    "※ 모델 / Feature / Threshold / Hold Time은 "
    "Final Validation 이후 변경하지 않았습니다."
)
