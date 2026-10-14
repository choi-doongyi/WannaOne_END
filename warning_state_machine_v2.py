# ============================================================
# warning_state_machine_v2.py
#
# INTEGRATED WARNING STATE MACHINE V2
#
# 입력:
# outputs/casting_integrated_warning_validation_v1.csv
# outputs/casting_validation_failure_events_v1.csv
#
# 모델 재학습 X
# Threshold 변경 X
#
# 기존 LOCK:
#
# 24H CAUTION
# XGBoost / 252 Features / Threshold 0.14
#
# 6H DANGER
# XGBoost / 240 Features / Threshold 0.035
#
#
# State Machine:
#
# CAUTION Trigger
#   → 최소 12시간 유지
#
# DANGER Trigger
#   → 최소 6시간 유지
#
# Priority:
#   DANGER > CAUTION > NORMAL
#
#
# 평가:
# 1. RAW vs State Machine 경보 분포
# 2. 상태 전환 횟수
# 3. Episode 수
# 4. False Alarm Episode
# 5. Failure Event 탐지율
# 6. Lead Time
# ============================================================


from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"


WARNING_PATH = OUTPUT_DIR / "casting_integrated_warning_validation_v1.csv"


FAILURE_PATH = OUTPUT_DIR / "casting_validation_failure_events_v1.csv"


# ============================================================
# 1. State Machine 설정
# ============================================================

CAUTION_HOLD_H = 12

DANGER_HOLD_H = 6


# Episode는 6시간 간격이 이어지면
# 하나의 연속 Episode로 본다.
EPISODE_GAP_H = 6


# ============================================================
# 2. 데이터 로드
# ============================================================


def load_data():

    warning_df = pd.read_csv(WARNING_PATH)

    failure_df = pd.read_csv(FAILURE_PATH)

    warning_df["OBSERVED_AT"] = pd.to_datetime(
        warning_df["OBSERVED_AT"], utc=True, errors="coerce"
    )

    failure_df["FAILURE_DT"] = pd.to_datetime(
        failure_df["FAILURE_DT"], format="mixed", utc=True, errors="coerce"
    )

    warning_df = (
        warning_df.dropna(subset=["OBSERVED_AT"])
        .sort_values(
            [
                "SCENARIO_ID",
                "OBSERVED_AT",
            ]
        )
        .reset_index(drop=True)
    )

    failure_df = (
        failure_df.dropna(subset=["FAILURE_DT"])
        .sort_values(
            [
                "SCENARIO_ID",
                "FAILURE_DT",
            ]
        )
        .reset_index(drop=True)
    )

    return (
        warning_df,
        failure_df,
    )


# ============================================================
# 3. 경보 단계 숫자화
# ============================================================

LEVEL_ORDER = {
    "NORMAL": 0,
    "CAUTION": 1,
    "DANGER": 2,
}


# ============================================================
# 4. State Machine
#
# 핵심:
#
# CAUTION과 DANGER의 유지 만료시간을
# 각각 따로 관리한다.
#
# 따라서:
#
# CAUTION
# → DANGER
# → DANGER 유지 종료
# → 기존 CAUTION 시간이 남아 있으면
# → 다시 CAUTION
#
# 형태가 가능하다.
# ============================================================


def apply_state_machine(warning_df):

    result_rows = []

    # ========================================================
    # Scenario별 독립 처리
    # ========================================================

    for scenario_id, group in warning_df.groupby("SCENARIO_ID"):

        group = group.sort_values("OBSERVED_AT").reset_index(drop=True)

        # ----------------------------------------------------
        # 각각의 유지 종료 시간
        # ----------------------------------------------------

        caution_until = pd.NaT

        danger_until = pd.NaT

        previous_state = "NORMAL"

        for _, row in group.iterrows():

            current_time = row["OBSERVED_AT"]

            raw_caution = int(row["PRED_24H"])

            raw_danger = int(row["PRED_6H"])

            # =================================================
            # 1. CAUTION Trigger
            #
            # 새로운 CAUTION 신호가 오면
            # 현재 시점 + 12H까지 유지
            # =================================================

            if raw_caution == 1:

                new_caution_until = current_time + pd.Timedelta(hours=CAUTION_HOLD_H)

                if pd.isna(caution_until) or new_caution_until > caution_until:

                    caution_until = new_caution_until

            # =================================================
            # 2. DANGER Trigger
            #
            # 새로운 DANGER 신호가 오면
            # 현재 시점 + 6H까지 유지
            # =================================================

            if raw_danger == 1:

                new_danger_until = current_time + pd.Timedelta(hours=DANGER_HOLD_H)

                if pd.isna(danger_until) or new_danger_until > danger_until:

                    danger_until = new_danger_until

            # =================================================
            # 3. 현재 State 결정
            #
            # Priority:
            # DANGER > CAUTION > NORMAL
            # =================================================

            danger_active = pd.notna(danger_until) and current_time <= danger_until

            caution_active = pd.notna(caution_until) and current_time <= caution_until

            if danger_active:

                state = "DANGER"

            elif caution_active:

                state = "CAUTION"

            else:

                state = "NORMAL"

            # =================================================
            # 4. State Transition
            # =================================================

            transition = state != previous_state

            # =================================================
            # 5. 상태 변화 종류
            # =================================================

            if not transition:

                transition_type = "STAY"

            else:

                previous_value = LEVEL_ORDER[previous_state]

                current_value = LEVEL_ORDER[state]

                if current_value > previous_value:

                    transition_type = "UP"

                else:

                    transition_type = "DOWN"

            # =================================================
            # 저장
            # =================================================

            new_row = row.to_dict()

            new_row["RAW_WARNING_LEVEL"] = row["WARNING_LEVEL"]

            new_row["STATE_WARNING_LEVEL"] = state

            new_row["CAUTION_ACTIVE"] = int(caution_active)

            new_row["DANGER_ACTIVE"] = int(danger_active)

            new_row["CAUTION_UNTIL"] = caution_until

            new_row["DANGER_UNTIL"] = danger_until

            new_row["STATE_TRANSITION"] = int(transition)

            new_row["TRANSITION_TYPE"] = transition_type

            result_rows.append(new_row)

            previous_state = state

    result_df = pd.DataFrame(result_rows)

    return result_df


# ============================================================
# 5. 경보 분포
# ============================================================


def warning_distribution(df, column_name):

    result = (
        df[column_name]
        .value_counts()
        .reindex(
            [
                "NORMAL",
                "CAUTION",
                "DANGER",
            ],
            fill_value=0,
        )
        .rename("COUNT")
        .reset_index()
    )

    result.columns = [
        "WARNING_LEVEL",
        "COUNT",
    ]

    result["RATE"] = result["COUNT"] / len(df)

    return result


# ============================================================
# 6. Episode 생성
#
# NORMAL 제외 후
# 동일 Scenario에서
#
# 다음 경보와의 간격 <= 6H
#
# 이면 하나의 Episode
# ============================================================


def build_episodes(df, warning_col, version_name):

    alert_df = (
        df[df[warning_col] != "NORMAL"]
        .sort_values(
            [
                "SCENARIO_ID",
                "OBSERVED_AT",
            ]
        )
        .copy()
    )

    if alert_df.empty:

        return pd.DataFrame()

    episode_rows = []

    for scenario_id, group in alert_df.groupby("SCENARIO_ID"):

        group = group.sort_values("OBSERVED_AT").reset_index(drop=True)

        episode_number = 1

        start_dt = None

        end_dt = None

        max_level = None

        point_count = 0

        for _, row in group.iterrows():

            current_dt = row["OBSERVED_AT"]

            current_level = row[warning_col]

            # =================================================
            # 첫 경보
            # =================================================

            if start_dt is None:

                start_dt = current_dt

                end_dt = current_dt

                max_level = current_level

                point_count = 1

                continue

            # =================================================
            # 이전 경보와의 시간차
            # =================================================

            gap_h = (current_dt - end_dt).total_seconds() / 3600

            # =================================================
            # 같은 Episode
            # =================================================

            if gap_h <= EPISODE_GAP_H:

                end_dt = current_dt

                point_count += 1

                if current_level == "DANGER":

                    max_level = "DANGER"

            # =================================================
            # 새로운 Episode
            # =================================================

            else:

                episode_rows.append(
                    {
                        "VERSION": version_name,
                        "SCENARIO_ID": scenario_id,
                        "EPISODE_ID": episode_number,
                        "START_DT": start_dt,
                        "END_DT": end_dt,
                        "MAX_LEVEL": max_level,
                        "POINT_COUNT": point_count,
                    }
                )

                episode_number += 1

                start_dt = current_dt

                end_dt = current_dt

                max_level = current_level

                point_count = 1

        # =====================================================
        # 마지막 Episode
        # =====================================================

        episode_rows.append(
            {
                "VERSION": version_name,
                "SCENARIO_ID": scenario_id,
                "EPISODE_ID": episode_number,
                "START_DT": start_dt,
                "END_DT": end_dt,
                "MAX_LEVEL": max_level,
                "POINT_COUNT": point_count,
            }
        )

    return pd.DataFrame(episode_rows)


# ============================================================
# 7. Episode 평가
#
# 통합 Episode:
#
# START ~ END 이후 24H 내 Failure가 있으면
# Failure 관련 Episode
#
# DANGER Episode는 별도로
# 6H 내 Failure 여부 확인
# ============================================================


def evaluate_episodes(episode_df, failure_df):

    if episode_df.empty:

        return episode_df

    rows = []

    for _, episode in episode_df.iterrows():

        scenario_id = episode["SCENARIO_ID"]

        start_dt = episode["START_DT"]

        end_dt = episode["END_DT"]

        max_level = episode["MAX_LEVEL"]

        failures = failure_df[failure_df["SCENARIO_ID"] == scenario_id].copy()

        # =====================================================
        # 24H 관련성
        # =====================================================

        related_24 = failures[
            (failures["FAILURE_DT"] >= start_dt)
            & (failures["FAILURE_DT"] <= (end_dt + pd.Timedelta(hours=24)))
        ]

        related_24h = not related_24.empty

        # =====================================================
        # 6H 관련성
        # =====================================================

        related_6 = failures[
            (failures["FAILURE_DT"] >= start_dt)
            & (failures["FAILURE_DT"] <= (end_dt + pd.Timedelta(hours=6)))
        ]

        related_6h = not related_6.empty

        rows.append(
            {
                **episode.to_dict(),
                "RELATED_FAILURE_24H": int(related_24h),
                "FALSE_ALARM_24H": int(not related_24h),
                "RELATED_FAILURE_6H": int(related_6h),
                "DANGER_TRUE_6H": (
                    int(related_6h) if (max_level == "DANGER") else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 8. 실제 Failure Event 평가
#
# CAUTION:
# Failure 전 24H 안에서
# CAUTION 또는 DANGER 존재 여부
#
# DANGER:
# Failure 전 6H 안에서
# DANGER 존재 여부
#
# State Machine 적용 전/후 비교
# ============================================================


def evaluate_failure_events(warning_df, failure_df, warning_col, version_name):

    rows = []

    for _, failure in failure_df.iterrows():

        scenario_id = failure["SCENARIO_ID"]

        failure_dt = failure["FAILURE_DT"]

        scenario_df = (
            warning_df[warning_df["SCENARIO_ID"] == scenario_id]
            .sort_values("OBSERVED_AT")
            .copy()
        )

        # =====================================================
        # 24H
        # =====================================================

        window_24 = scenario_df[
            (scenario_df["OBSERVED_AT"] >= (failure_dt - pd.Timedelta(hours=24)))
            & (scenario_df["OBSERVED_AT"] <= failure_dt)
        ].copy()

        alert_24 = window_24[
            window_24[warning_col].isin(
                [
                    "CAUTION",
                    "DANGER",
                ]
            )
        ]

        detected_24 = not alert_24.empty

        if detected_24:

            first_warning = alert_24["OBSERVED_AT"].min()

            lead_24 = (failure_dt - first_warning).total_seconds() / 3600

        else:

            first_warning = pd.NaT

            lead_24 = np.nan

        # =====================================================
        # 6H
        # =====================================================

        window_6 = scenario_df[
            (scenario_df["OBSERVED_AT"] >= (failure_dt - pd.Timedelta(hours=6)))
            & (scenario_df["OBSERVED_AT"] <= failure_dt)
        ].copy()

        alert_6 = window_6[window_6[warning_col] == "DANGER"]

        detected_6 = not alert_6.empty

        if detected_6:

            first_danger = alert_6["OBSERVED_AT"].min()

            lead_6 = (failure_dt - first_danger).total_seconds() / 3600

        else:

            first_danger = pd.NaT

            lead_6 = np.nan

        rows.append(
            {
                "VERSION": version_name,
                "SCENARIO_ID": scenario_id,
                "FAILURE_DT": failure_dt,
                "DETECTED_24H": int(detected_24),
                "FIRST_WARNING_DT": first_warning,
                "LEAD_24H": lead_24,
                "DETECTED_6H": int(detected_6),
                "FIRST_DANGER_DT": first_danger,
                "LEAD_6H": lead_6,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 9. Event Summary
# ============================================================


def event_summary(event_df, version):

    total = len(event_df)

    detected_24 = int(event_df["DETECTED_24H"].sum())

    detected_6 = int(event_df["DETECTED_6H"].sum())

    lead_24 = event_df["LEAD_24H"].dropna()

    lead_6 = event_df["LEAD_6H"].dropna()

    return {
        "VERSION": version,
        "FAILURE_EVENTS": total,
        "DETECTED_24H": detected_24,
        "DETECTION_RATE_24H": (detected_24 / total if total else np.nan),
        "MEAN_LEAD_24H": (lead_24.mean() if not lead_24.empty else np.nan),
        "MEDIAN_LEAD_24H": (lead_24.median() if not lead_24.empty else np.nan),
        "DETECTED_6H": detected_6,
        "DETECTION_RATE_6H": (detected_6 / total if total else np.nan),
        "MEAN_LEAD_6H": (lead_6.mean() if not lead_6.empty else np.nan),
        "MEDIAN_LEAD_6H": (lead_6.median() if not lead_6.empty else np.nan),
    }


# ============================================================
# 10. Episode Summary
# ============================================================


def episode_summary(episode_eval_df, version):

    if episode_eval_df.empty:

        return {
            "VERSION": version,
            "TOTAL_EPISODES": 0,
            "TRUE_24H_EPISODES": 0,
            "FALSE_24H_EPISODES": 0,
            "DANGER_EPISODES": 0,
            "TRUE_DANGER_6H": 0,
            "FALSE_DANGER_6H": 0,
        }

    total = len(episode_eval_df)

    true_24 = int(episode_eval_df["RELATED_FAILURE_24H"].sum())

    false_24 = int(episode_eval_df["FALSE_ALARM_24H"].sum())

    danger_df = episode_eval_df[episode_eval_df["MAX_LEVEL"] == "DANGER"]

    danger_total = len(danger_df)

    true_danger = int(danger_df["RELATED_FAILURE_6H"].sum()) if danger_total else 0

    false_danger = danger_total - true_danger

    return {
        "VERSION": version,
        "TOTAL_EPISODES": total,
        "TRUE_24H_EPISODES": true_24,
        "FALSE_24H_EPISODES": false_24,
        "DANGER_EPISODES": danger_total,
        "TRUE_DANGER_6H": true_danger,
        "FALSE_DANGER_6H": false_danger,
    }


# ============================================================
# 11. MAIN
# ============================================================


def main():

    print("=" * 80)

    print("WARNING STATE MACHINE V2")

    print("=" * 80)

    warning_df, failure_df = load_data()

    print()

    print("Warning rows:", len(warning_df))

    print("Failure events:", len(failure_df))

    # ========================================================
    # State Machine
    # ========================================================

    state_df = apply_state_machine(warning_df)

    # ========================================================
    # RAW 분포
    # ========================================================

    raw_distribution = warning_distribution(state_df, "RAW_WARNING_LEVEL")

    raw_distribution["VERSION"] = "RAW"

    # ========================================================
    # State 분포
    # ========================================================

    state_distribution = warning_distribution(state_df, "STATE_WARNING_LEVEL")

    state_distribution["VERSION"] = "STATE_MACHINE"

    distribution_df = pd.concat(
        [
            raw_distribution,
            state_distribution,
        ],
        ignore_index=True,
    )

    print()

    print("=" * 80)

    print("WARNING DISTRIBUTION")

    print("=" * 80)

    print(
        distribution_df[
            [
                "VERSION",
                "WARNING_LEVEL",
                "COUNT",
                "RATE",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # 상태 전환
    # ========================================================

    transitions = state_df["STATE_TRANSITION"].sum()

    up_transitions = state_df["TRANSITION_TYPE"].eq("UP").sum()

    down_transitions = state_df["TRANSITION_TYPE"].eq("DOWN").sum()

    print()

    print("=" * 80)

    print("STATE TRANSITIONS")

    print("=" * 80)

    print("전체 State 변경:", int(transitions))

    print("상향 전환:", int(up_transitions))

    print("하향 전환:", int(down_transitions))

    # ========================================================
    # Episode - RAW
    # ========================================================

    raw_episode = build_episodes(
        df=state_df, warning_col="RAW_WARNING_LEVEL", version_name="RAW"
    )

    raw_episode_eval = evaluate_episodes(episode_df=raw_episode, failure_df=failure_df)

    # ========================================================
    # Episode - State Machine
    # ========================================================

    state_episode = build_episodes(
        df=state_df, warning_col="STATE_WARNING_LEVEL", version_name="STATE_MACHINE"
    )

    state_episode_eval = evaluate_episodes(
        episode_df=state_episode, failure_df=failure_df
    )

    # ========================================================
    # Episode 비교
    # ========================================================

    episode_summary_df = pd.DataFrame(
        [
            episode_summary(raw_episode_eval, "RAW"),
            episode_summary(state_episode_eval, "STATE_MACHINE"),
        ]
    )

    print()

    print("=" * 80)

    print("EPISODE COMPARISON")

    print("=" * 80)

    print(episode_summary_df.to_string(index=False))

    # ========================================================
    # Failure Event - RAW
    # ========================================================

    raw_event_df = evaluate_failure_events(
        warning_df=state_df,
        failure_df=failure_df,
        warning_col="RAW_WARNING_LEVEL",
        version_name="RAW",
    )

    # ========================================================
    # Failure Event - State Machine
    # ========================================================

    state_event_df = evaluate_failure_events(
        warning_df=state_df,
        failure_df=failure_df,
        warning_col="STATE_WARNING_LEVEL",
        version_name="STATE_MACHINE",
    )

    event_summary_df = pd.DataFrame(
        [
            event_summary(raw_event_df, "RAW"),
            event_summary(state_event_df, "STATE_MACHINE"),
        ]
    )

    print()

    print("=" * 80)

    print("FAILURE EVENT COMPARISON")

    print("=" * 80)

    print(event_summary_df.round(4).to_string(index=False))

    # ========================================================
    # Raw와 State 경보가 달라진 Point 수
    # ========================================================

    changed_points = state_df["RAW_WARNING_LEVEL"] != state_df["STATE_WARNING_LEVEL"]

    changed_count = int(changed_points.sum())

    print()

    print("=" * 80)

    print("STATE MACHINE EFFECT")

    print("=" * 80)

    print("상태가 변경된 Point:", changed_count, "/", len(state_df))

    print("변경 비율:", f"{changed_count / len(state_df):.2%}")

    # ========================================================
    # 변경 유형
    # ========================================================

    changed_detail_df = state_df.loc[
        changed_points,
        [
            "SCENARIO_ID",
            "OBSERVED_AT",
            "RAW_WARNING_LEVEL",
            "STATE_WARNING_LEVEL",
        ],
    ].copy()

    if not changed_detail_df.empty:

        change_type_summary = (
            changed_detail_df.groupby(
                [
                    "RAW_WARNING_LEVEL",
                    "STATE_WARNING_LEVEL",
                ]
            )
            .size()
            .reset_index(name="COUNT")
            .sort_values("COUNT", ascending=False)
        )

        print()

        print("변경 유형:")

        print(change_type_summary.to_string(index=False))

    else:

        change_type_summary = pd.DataFrame()

    # ========================================================
    # State Machine 이후
    # DANGER인데 바로 전 State가 NORMAL이었던 경우
    # ========================================================

    danger_entries = []

    for scenario_id, group in state_df.groupby("SCENARIO_ID"):

        group = group.sort_values("OBSERVED_AT").reset_index(drop=True)

        previous_state = "NORMAL"

        for _, row in group.iterrows():

            current_state = row["STATE_WARNING_LEVEL"]

            if current_state == "DANGER" and previous_state == "NORMAL":

                danger_entries.append(
                    {
                        "SCENARIO_ID": scenario_id,
                        "OBSERVED_AT": row["OBSERVED_AT"],
                        "DIRECT_NORMAL_TO_DANGER": 1,
                    }
                )

            previous_state = current_state

    danger_entry_df = pd.DataFrame(danger_entries)

    print()

    print("=" * 80)

    print("DIRECT NORMAL -> DANGER")

    print("=" * 80)

    print("건수:", len(danger_entry_df))

    # ========================================================
    # 저장
    # ========================================================

    state_df.to_csv(
        OUTPUT_DIR / "casting_warning_state_machine_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    distribution_df.to_csv(
        OUTPUT_DIR / "casting_warning_distribution_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    episode_summary_df.to_csv(
        OUTPUT_DIR / "casting_warning_episode_comparison_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_summary_df.to_csv(
        OUTPUT_DIR / "casting_warning_event_comparison_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_episode_eval.to_csv(
        OUTPUT_DIR / "casting_warning_raw_episodes_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_episode_eval.to_csv(
        OUTPUT_DIR / "casting_warning_state_episodes_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_event_df.to_csv(
        OUTPUT_DIR / "casting_warning_raw_events_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    state_event_df.to_csv(
        OUTPUT_DIR / "casting_warning_state_events_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    changed_detail_df.to_csv(
        OUTPUT_DIR / "casting_warning_changed_points_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    danger_entry_df.to_csv(
        OUTPUT_DIR / "casting_warning_direct_danger_v2.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 완료
    # ========================================================

    print()

    print("=" * 80)

    print("STATE MACHINE V2 COMPLETE")

    print("=" * 80)

    print()

    print("CAUTION Hold:", CAUTION_HOLD_H, "hours")

    print("DANGER Hold:", DANGER_HOLD_H, "hours")

    print("Episode Gap:", EPISODE_GAP_H, "hours")


# ============================================================
# 12. RUN
# ============================================================

if __name__ == "__main__":

    main()
