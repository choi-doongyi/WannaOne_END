# ============================================================
# restart_04_train_oof_state_machine.py
#
# STRICT RESTART - TRAIN OOF STATE MACHINE
#
# Validation 사용 X
# Test 사용 X
#
# State Machine 규칙은 결과를 보고 튜닝하지 않는다.
#
# 사전 정의:
# ------------------------------------------------------------
# Prediction cadence = 6H
#
# CAUTION 발생
# -> 최소 다음 판단 시점까지 유지
# -> 6H HOLD
#
# DANGER 발생
# -> 최소 다음 판단 시점까지 유지
# -> 6H HOLD
#
# 우선순위:
# DANGER > CAUTION > NORMAL
#
# 목적:
# RAW 경보의 순간적인 ON/OFF를 줄인다.
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"


INPUT_PATH = OUTPUT_DIR / "restart_train_oof_integrated_warning.csv"


# ------------------------------------------------------------
# 사전에 고정한 운영 규칙
#
# 모델 예측 간격이 6시간이므로
# 최소 다음 판단 시점까지 상태 유지
# ------------------------------------------------------------

CAUTION_HOLD_HOURS = 6

DANGER_HOLD_HOURS = 6


# ============================================================
# 1. 데이터 로드
# ============================================================

df = pd.read_csv(INPUT_PATH)


print("=" * 80)
print("STRICT RESTART - TRAIN OOF STATE MACHINE")
print("=" * 80)


print()

print("Input shape:", df.shape)


# ============================================================
# 2. 시간형 변환
# ============================================================

df["OBSERVED_AT"] = pd.to_datetime(
    df["OBSERVED_AT"],
    format="mixed",
    utc=True,
    errors="coerce",
)


if df["OBSERVED_AT"].isna().any():

    raise RuntimeError("OBSERVED_AT 변환 실패")


df = df.sort_values(
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
    ]
).reset_index(drop=True)


# ============================================================
# 3. RAW Warning 확인
# ============================================================

required_cols = [
    "SCENARIO_ID",
    "OBSERVED_AT",
    "RAW_WARNING",
    "CAUTION_TRIGGER",
    "DANGER_TRIGGER",
]


missing_cols = [col for col in required_cols if col not in df.columns]


if missing_cols:

    raise KeyError(f"필수 컬럼 누락: " f"{missing_cols}")


print()

print("=" * 80)
print("STATE MACHINE POLICY")
print("=" * 80)


print("CAUTION HOLD:", CAUTION_HOLD_HOURS, "hours")


print("DANGER HOLD:", DANGER_HOLD_HOURS, "hours")


# ============================================================
# 4. Scenario별 State Machine
# ============================================================


def apply_state_machine(scenario_df):

    scenario_df = scenario_df.sort_values("OBSERVED_AT").copy()

    state_list = []

    caution_until = None

    danger_until = None

    for row in scenario_df.itertuples():

        current_time = row.OBSERVED_AT

        caution_trigger = bool(row.CAUTION_TRIGGER)

        danger_trigger = bool(row.DANGER_TRIGGER)

        # =====================================================
        # DANGER Trigger
        #
        # 새로 발생하면
        # current_time + 6H까지 유지
        # =====================================================

        if danger_trigger:

            new_danger_until = current_time + pd.Timedelta(hours=DANGER_HOLD_HOURS)

            if danger_until is None or new_danger_until > danger_until:

                danger_until = new_danger_until

        # =====================================================
        # CAUTION Trigger
        # =====================================================

        if caution_trigger:

            new_caution_until = current_time + pd.Timedelta(hours=CAUTION_HOLD_HOURS)

            if caution_until is None or new_caution_until > caution_until:

                caution_until = new_caution_until

        # =====================================================
        # 현재 State 결정
        #
        # Priority:
        #
        # DANGER
        # >
        # CAUTION
        # >
        # NORMAL
        # =====================================================

        danger_active = danger_until is not None and current_time <= danger_until

        caution_active = caution_until is not None and current_time <= caution_until

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
# 5. 전체 Scenario 적용
#
# pandas groupby.apply에서 그룹 컬럼이 사라지는 문제를
# 피하기 위해 Scenario별로 직접 처리 후 concat한다.
# ============================================================

state_parts = []


for scenario_id, scenario_df in df.groupby(
    "SCENARIO_ID",
    sort=False,
):

    temp = apply_state_machine(scenario_df)

    # 혹시 pandas/groupby 동작으로
    # SCENARIO_ID가 빠져도 다시 명시적으로 넣는다.
    temp["SCENARIO_ID"] = scenario_id

    state_parts.append(temp)


state_df = pd.concat(
    state_parts,
    ignore_index=True,
)


# Scenario / 시간 순으로 다시 정렬
state_df = state_df.sort_values(
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
    ]
).reset_index(drop=True)


# 안전 검사
if "SCENARIO_ID" not in state_df.columns:

    raise RuntimeError("State Machine 결과에서 " "SCENARIO_ID가 사라졌습니다.")


print()

print("State Machine rows:", len(state_df))

print("State Machine scenarios:", state_df["SCENARIO_ID"].nunique())

# ============================================================
# 6. 숫자 Score
# ============================================================

score_map = {
    "NORMAL": 0,
    "CAUTION": 1,
    "DANGER": 2,
}


state_df["RAW_SCORE"] = state_df["RAW_WARNING"].map(score_map).astype(int)


state_df["STATE_SCORE"] = state_df["STATE_WARNING"].map(score_map).astype(int)


# ============================================================
# 7. 경보 분포 비교
# ============================================================


def warning_distribution(
    data,
    column,
    name,
):

    count = (
        data[column]
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

    result = pd.DataFrame(
        {
            "SYSTEM": name,
            "WARNING_LEVEL": count.index,
            "COUNT": count.values,
            "RATIO": (count.values / len(data)),
        }
    )

    return result


raw_distribution = warning_distribution(
    state_df,
    column="RAW_WARNING",
    name="RAW",
)


state_distribution = warning_distribution(
    state_df,
    column="STATE_WARNING",
    name="STATE_MACHINE",
)


distribution_df = pd.concat(
    [
        raw_distribution,
        state_distribution,
    ],
    ignore_index=True,
)


print()

print("=" * 80)
print("WARNING DISTRIBUTION COMPARISON")
print("=" * 80)


print(distribution_df.round(4).to_string(index=False))


# ============================================================
# 8. 상태 변경 계산
# ============================================================


def count_transitions(
    data,
    column,
):

    temp = data[
        [
            "SCENARIO_ID",
            "OBSERVED_AT",
            column,
        ]
    ].copy()

    temp["PREVIOUS"] = temp.groupby("SCENARIO_ID")[column].shift(1)

    changed = temp["PREVIOUS"].notna() & (temp[column] != temp["PREVIOUS"])

    return int(changed.sum())


raw_transition_count = count_transitions(
    state_df,
    "RAW_WARNING",
)


state_transition_count = count_transitions(
    state_df,
    "STATE_WARNING",
)


transition_reduction = raw_transition_count - state_transition_count


transition_reduction_ratio = (
    transition_reduction / raw_transition_count if raw_transition_count > 0 else 0
)


print()

print("=" * 80)
print("TRANSITION COMPARISON")
print("=" * 80)


print("RAW transitions:", raw_transition_count)


print("STATE transitions:", state_transition_count)


print("Reduction:", transition_reduction)


print("Reduction Ratio:", round(transition_reduction_ratio, 4))


# ============================================================
# 9. State가 실제로 바뀐 Point
# ============================================================

state_df["STATE_CHANGED_FROM_RAW"] = (
    state_df["RAW_WARNING"] != state_df["STATE_WARNING"]
)


changed_count = int(state_df["STATE_CHANGED_FROM_RAW"].sum())


changed_ratio = changed_count / len(state_df)


print()

print("=" * 80)
print("RAW VS STATE CHANGE")
print("=" * 80)


print("Changed Point:", changed_count)


print("Changed Ratio:", round(changed_ratio, 4))


# ============================================================
# 10. 변경 유형
# ============================================================

change_type_df = (
    state_df[state_df["STATE_CHANGED_FROM_RAW"]]
    .groupby(
        [
            "RAW_WARNING",
            "STATE_WARNING",
        ]
    )
    .size()
    .reset_index(name="COUNT")
    .sort_values("COUNT", ascending=False)
)


print()

print("=" * 80)
print("CHANGE TYPE")
print("=" * 80)


if len(change_type_df) > 0:

    print(change_type_df.to_string(index=False))

else:

    print("변경 없음")


# ============================================================
# 11. Direct NORMAL -> DANGER Entry
#
# State Machine 이후
# 이전 상태 NORMAL에서
# 현재 DANGER로 바로 올라가는 경우.
#
# 이것은 허용한다.
# 6H 모델이 독립적인 임박위험 신호이기 때문.
# ============================================================

state_df["PREV_STATE_WARNING"] = state_df.groupby("SCENARIO_ID")["STATE_WARNING"].shift(
    1
)


direct_normal_to_danger = (state_df["PREV_STATE_WARNING"] == "NORMAL") & (
    state_df["STATE_WARNING"] == "DANGER"
)


direct_normal_to_danger_count = int(direct_normal_to_danger.sum())


print()

print("=" * 80)
print("DIRECT NORMAL -> DANGER")
print("=" * 80)


print("Count:", direct_normal_to_danger_count)


# ============================================================
# 12. Scenario별 State 요약
# ============================================================

scenario_state_summary = state_df.groupby("SCENARIO_ID", as_index=False).agg(
    TOTAL_POINTS=(
        "STATE_WARNING",
        "size",
    ),
    MAX_RAW_SCORE=(
        "RAW_SCORE",
        "max",
    ),
    MAX_STATE_SCORE=(
        "STATE_SCORE",
        "max",
    ),
)


scenario_state_summary["MAX_RAW_WARNING"] = scenario_state_summary["MAX_RAW_SCORE"].map(
    {
        0: "NORMAL",
        1: "CAUTION",
        2: "DANGER",
    }
)


scenario_state_summary["MAX_STATE_WARNING"] = scenario_state_summary[
    "MAX_STATE_SCORE"
].map(
    {
        0: "NORMAL",
        1: "CAUTION",
        2: "DANGER",
    }
)


# ============================================================
# 13. 저장
# ============================================================

state_output_path = OUTPUT_DIR / "restart_train_oof_state_machine.csv"


distribution_output_path = OUTPUT_DIR / "restart_train_oof_state_distribution.csv"


scenario_output_path = OUTPUT_DIR / "restart_train_oof_state_scenario_summary.csv"


state_df.to_csv(
    state_output_path,
    index=False,
    encoding="utf-8-sig",
)


distribution_df.to_csv(
    distribution_output_path,
    index=False,
    encoding="utf-8-sig",
)


scenario_state_summary.to_csv(
    scenario_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 14. 완료
# ============================================================

print()

print("=" * 80)
print("TRAIN OOF STATE MACHINE COMPLETE")
print("=" * 80)


print("CAUTION HOLD:", CAUTION_HOLD_HOURS, "H")


print("DANGER HOLD:", DANGER_HOLD_HOURS, "H")


print()


print("RAW transitions:", raw_transition_count)


print("STATE transitions:", state_transition_count)


print("Transition reduction:", round(transition_reduction_ratio * 100, 2), "%")


print()


print("State output:", state_output_path)


print()

print("※ Validation 사용하지 않음")


print("※ Hold Time은 Validation 성능으로 선택하지 않음")


print("※ 6시간 예측 cadence에 맞춰 " "6H로 사전 정의")
