# ============================================================
# integrated_warning_system_v1.py
#
# CASTING INTEGRATED EARLY WARNING SYSTEM V1
#
# 최종 LOCK
#
# 24H CAUTION
# ------------------------------------------------------------
# Model      : XGBoost
# Feature    : BASE 240 + Mean Diff 12 = 252
# Threshold  : 0.14
#
# 6H DANGER
# ------------------------------------------------------------
# Model      : XGBoost
# Feature    : BASE 240
# Threshold  : 0.035
# Policy     : Recall >= 0.75
#
#
# 최종 경보:
#
# if P6H >= 0.035:
#     DANGER
#
# elif P24H >= 0.14:
#     CAUTION
#
# else:
#     NORMAL
#
#
# 평가:
# 1. Point-level
# 2. 통합 경보 분포
# 3. 실제 FAILURE_DT 기반 Event-level 탐지율
# 4. 최초 경보 Lead Time
# 5. 경보 Episode / 오경보
#
#
# 중요:
# FAILURE_DT는 절대 Feature로 사용하지 않음.
# 오직 최종 retrospective evaluation에서만 사용.
# ============================================================


import csv
import json

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

# ============================================================
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v25.csv"


VALID_PATH = OUTPUT_DIR / "casting_validation_features_v25.csv"


MANIFEST_PATH = ROOT / "manifest.json"


CASTING_EQP = "P1-CC1-INS01"


# ============================================================
# 1. LOCKED THRESHOLD
# ============================================================

THRESHOLD_24H = 0.14

THRESHOLD_6H = 0.035


# ============================================================
# 2. CSV Stream
# ============================================================


def read_rows(path):

    with Path(path).open(encoding="utf-8-sig", newline="") as f:

        yield from csv.DictReader(f)


# ============================================================
# 3. Manifest
# ============================================================


def load_manifest():

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(f"manifest 없음: " f"{MANIFEST_PATH}")

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ============================================================
# 4. Scenario Split
# ============================================================


def get_split_scenarios(manifest, split_name):

    ids = []

    for scenario in manifest["scenarios"]:

        if scenario["split"] == split_name:

            ids.append(scenario["id"])

    return sorted(ids)


# ============================================================
# 5. Manifest Source Reader
# ============================================================


def source_rows(scenario_id, file_id, manifest):

    candidates = [
        f
        for f in manifest["files"]
        if (f["scenarioId"] == scenario_id and f["id"] == file_id)
    ]

    if not candidates:

        return

    source = candidates[0]

    for chunk in source["chunks"]:

        chunk_path = ROOT / "chunks" / chunk["sha256"]

        if not chunk_path.exists():

            raise FileNotFoundError(f"Chunk 없음: " f"{chunk_path}")

        for row in read_rows(chunk_path):

            yield row


# ============================================================
# 6. Validation FAILURE_DT 읽기
#
# 주의:
# 이것은 X Feature 생성에 절대 사용하지 않는다.
#
# Event-level 평가용 Truth만 가져온다.
# ============================================================


def load_failure_events(manifest, scenario_ids):

    rows = []

    for scenario_id in scenario_ids:

        maintenance_rows = source_rows(
            scenario_id=scenario_id, file_id="maintenance", manifest=manifest
        )

        if maintenance_rows is None:

            continue

        for row in maintenance_rows:

            # ------------------------------------------------
            # 연주기 CC1만
            # ------------------------------------------------

            if row.get("EQP_TAG") != CASTING_EQP:

                continue

            failure_dt_raw = row.get("FAILURE_DT")

            # ------------------------------------------------
            # FAILURE_DT 없는 정비는 실제 고장 Event 아님
            # ------------------------------------------------

            if failure_dt_raw is None or str(failure_dt_raw).strip() == "":

                continue

            failure_dt = pd.to_datetime(failure_dt_raw, utc=True, errors="coerce")

            if pd.isna(failure_dt):

                continue

            rows.append(
                {
                    "SCENARIO_ID": scenario_id,
                    "FAILURE_DT": failure_dt,
                    "WO_NO": row.get("WO_NO"),
                    "PART_CD": row.get("PART_CD"),
                    "MNT_TYPE": row.get("MNT_TYPE"),
                }
            )

    failure_df = pd.DataFrame(rows)

    if failure_df.empty:

        return pd.DataFrame(
            columns=[
                "SCENARIO_ID",
                "FAILURE_DT",
                "WO_NO",
                "PART_CD",
                "MNT_TYPE",
            ]
        )

    failure_df = (
        failure_df.drop_duplicates(
            subset=[
                "SCENARIO_ID",
                "FAILURE_DT",
            ]
        )
        .sort_values(
            [
                "SCENARIO_ID",
                "FAILURE_DT",
            ]
        )
        .reset_index(drop=True)
    )

    return failure_df


# ============================================================
# 7. Threshold Metrics
# ============================================================


def evaluate_binary(y_true, probability, threshold):

    pred = (probability >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[
            0,
            1,
        ],
    ).ravel()

    return {
        "THRESHOLD": threshold,
        "PRECISION": precision_score(y_true, pred, zero_division=0),
        "RECALL": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "F2": fbeta_score(y_true, pred, beta=2, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ============================================================
# 8. Feature 정의
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
# 9. Maintenance Feature
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
# 10. TREND_PER_H 제외
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


# ============================================================
# 11. 24H Mean Diff Feature 12개
# ============================================================

mean_diff_features = [
    # CONTROL_DELAY
    "CONTROL_DELAY_1H_MINUS_3H_MEAN",
    "CONTROL_DELAY_3H_MINUS_6H_MEAN",
    "CONTROL_DELAY_6H_MINUS_12H_MEAN",
    # MLD-LVL
    "MLD-LVL_1H_MINUS_3H_MEAN",
    "MLD-LVL_3H_MINUS_6H_MEAN",
    "MLD-LVL_6H_MINUS_12H_MEAN",
    # OSC-STK
    "OSC-STK_1H_MINUS_3H_MEAN",
    "OSC-STK_3H_MINUS_6H_MEAN",
    "OSC-STK_6H_MINUS_12H_MEAN",
    # OSC-FRQ
    "OSC-FRQ_1H_MINUS_3H_MEAN",
    "OSC-FRQ_3H_MINUS_6H_MEAN",
    "OSC-FRQ_6H_MINUS_12H_MEAN",
]


# ============================================================
# 12. Feature List 생성
# ============================================================


def build_feature_lists(df):

    # --------------------------------------------------------
    # BASE 240
    # --------------------------------------------------------

    base_features = [
        col
        for col in df.columns
        if (
            col not in EXCLUDE_COLS
            and col not in DROP_MAINTENANCE_COLS
            and col not in trend_per_hour_features
            and col not in mean_diff_features
            and pd.api.types.is_numeric_dtype(df[col])
        )
    ]

    # --------------------------------------------------------
    # 24H = BASE + Mean Diff
    # --------------------------------------------------------

    features_24h = base_features + mean_diff_features

    # --------------------------------------------------------
    # 6H = BASE만
    # --------------------------------------------------------

    features_6h = base_features.copy()

    return (
        base_features,
        features_24h,
        features_6h,
    )


# ============================================================
# 13. XGBoost 생성
# ============================================================


def make_xgb(scale_pos_weight):

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
# 14. 하나의 모델 학습
# ============================================================


def train_locked_model(train_df, feature_cols, target_col, censor_col):

    usable = train_df[(train_df[censor_col] == 0) & train_df[target_col].notna()].copy()

    usable[target_col] = usable[target_col].astype(int)

    y = usable[target_col]

    negative = int((y == 0).sum())

    positive = int((y == 1).sum())

    scale_pos_weight = negative / positive

    imputer = SimpleImputer(strategy="median")

    X_imp = imputer.fit_transform(usable[feature_cols])

    model = make_xgb(scale_pos_weight=scale_pos_weight)

    model.fit(X_imp, y)

    return {
        "model": model,
        "imputer": imputer,
        "train_rows": len(usable),
        "negative": negative,
        "positive": positive,
        "scale_pos_weight": scale_pos_weight,
    }


# ============================================================
# 15. Point-level Evaluation
# ============================================================


def point_level_evaluation(df, target, censor, probability, threshold):

    mask = (df[censor] == 0) & df[target].notna()

    y_true = df.loc[mask, target].astype(int)

    y_prob = probability[mask.to_numpy()]

    metrics = evaluate_binary(y_true=y_true, probability=y_prob, threshold=threshold)

    metrics["ROC_AUC"] = roc_auc_score(y_true, y_prob)

    metrics["PR_AUC"] = average_precision_score(y_true, y_prob)

    metrics["N"] = len(y_true)

    return metrics


# ============================================================
# 16. Event Detection
#
# 각 실제 FAILURE_DT 기준:
#
# CAUTION:
# failure 전 24H 안에
# PRED_24H == 1이 있었는가?
#
# DANGER:
# failure 전 6H 안에
# PRED_6H == 1이 있었는가?
#
# Lead time:
# 해당 Window 안에서 최초 경보 시점
# ============================================================


def evaluate_failure_events(warning_df, failure_df):

    rows = []

    for _, failure in failure_df.iterrows():

        scenario_id = failure["SCENARIO_ID"]

        failure_dt = failure["FAILURE_DT"]

        scenario_pred = (
            warning_df[warning_df["SCENARIO_ID"] == scenario_id]
            .sort_values("OBSERVED_AT")
            .copy()
        )

        # ====================================================
        # 과거 시점만
        # ====================================================

        before_failure = scenario_pred[
            scenario_pred["OBSERVED_AT"] <= failure_dt
        ].copy()

        # ====================================================
        # 24H Window
        # ====================================================

        start_24h = failure_dt - pd.Timedelta(hours=24)

        window_24h = before_failure[before_failure["OBSERVED_AT"] >= start_24h].copy()

        alerts_24h = window_24h[window_24h["PRED_24H"] == 1].copy()

        caution_detected = not alerts_24h.empty

        if caution_detected:

            first_caution_dt = alerts_24h["OBSERVED_AT"].min()

            caution_lead_h = (failure_dt - first_caution_dt).total_seconds() / 3600

        else:

            first_caution_dt = pd.NaT

            caution_lead_h = np.nan

        # ====================================================
        # 6H Window
        # ====================================================

        start_6h = failure_dt - pd.Timedelta(hours=6)

        window_6h = before_failure[before_failure["OBSERVED_AT"] >= start_6h].copy()

        alerts_6h = window_6h[window_6h["PRED_6H"] == 1].copy()

        danger_detected = not alerts_6h.empty

        if danger_detected:

            first_danger_dt = alerts_6h["OBSERVED_AT"].min()

            danger_lead_h = (failure_dt - first_danger_dt).total_seconds() / 3600

        else:

            first_danger_dt = pd.NaT

            danger_lead_h = np.nan

        # ====================================================
        # Integrated Warning
        #
        # CAUTION 또는 DANGER가 하나라도 있으면
        # 24H 전체 조기경보 성공으로 볼 수 있음
        # ====================================================

        integrated_alerts = window_24h[
            window_24h["WARNING_LEVEL"].isin(
                [
                    "CAUTION",
                    "DANGER",
                ]
            )
        ].copy()

        integrated_detected = not integrated_alerts.empty

        if integrated_detected:

            first_integrated_dt = integrated_alerts["OBSERVED_AT"].min()

            integrated_lead_h = (
                failure_dt - first_integrated_dt
            ).total_seconds() / 3600

        else:

            first_integrated_dt = pd.NaT

            integrated_lead_h = np.nan

        rows.append(
            {
                "SCENARIO_ID": scenario_id,
                "FAILURE_DT": failure_dt,
                "PART_CD": failure.get("PART_CD"),
                "MNT_TYPE": failure.get("MNT_TYPE"),
                "CAUTION_DETECTED": int(caution_detected),
                "FIRST_CAUTION_DT": first_caution_dt,
                "CAUTION_LEAD_H": caution_lead_h,
                "DANGER_DETECTED": int(danger_detected),
                "FIRST_DANGER_DT": first_danger_dt,
                "DANGER_LEAD_H": danger_lead_h,
                "INTEGRATED_DETECTED": int(integrated_detected),
                "FIRST_WARNING_DT": first_integrated_dt,
                "FIRST_WARNING_LEAD_H": integrated_lead_h,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 17. Alert Episode 생성
#
# OBSERVED_AT 간격이 6H이므로
# 6H 이하 간격의 연속 Alert를 같은 Episode로 묶는다.
# ============================================================


def build_alert_episodes(warning_df):

    alert_df = (
        warning_df[warning_df["WARNING_LEVEL"] != "NORMAL"]
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

        episode_id = 1

        start_dt = None

        end_dt = None

        max_level = None

        row_count = 0

        for _, row in group.iterrows():

            current_dt = row["OBSERVED_AT"]

            current_level = row["WARNING_LEVEL"]

            # ================================================
            # 첫 번째
            # ================================================

            if start_dt is None:

                start_dt = current_dt

                end_dt = current_dt

                max_level = current_level

                row_count = 1

                continue

            gap = current_dt - end_dt

            # ================================================
            # 6시간 이하 → 동일 Episode
            # ================================================

            if gap <= pd.Timedelta(hours=6):

                end_dt = current_dt

                row_count += 1

                if current_level == "DANGER":

                    max_level = "DANGER"

            # ================================================
            # 새로운 Episode
            # ================================================

            else:

                episode_rows.append(
                    {
                        "SCENARIO_ID": scenario_id,
                        "EPISODE_ID": episode_id,
                        "START_DT": start_dt,
                        "END_DT": end_dt,
                        "MAX_LEVEL": max_level,
                        "POINT_COUNT": row_count,
                    }
                )

                episode_id += 1

                start_dt = current_dt

                end_dt = current_dt

                max_level = current_level

                row_count = 1

        # ====================================================
        # 마지막 Episode
        # ====================================================

        episode_rows.append(
            {
                "SCENARIO_ID": scenario_id,
                "EPISODE_ID": episode_id,
                "START_DT": start_dt,
                "END_DT": end_dt,
                "MAX_LEVEL": max_level,
                "POINT_COUNT": row_count,
            }
        )

    return pd.DataFrame(episode_rows)


# ============================================================
# 18. Episode가 실제 Failure와 관련 있는지 평가
#
# Episode 내 Alert 중:
#
# CAUTION이면 이후 24H 내 Failure
# DANGER면 이후 6H 내 Failure
#
# 여기서는 통합 Episode 기준으로
# Episode 시작~끝 중 어느 시점이라도
# 24H 내 Failure가 있으면 RELATED=1
#
# DANGER Episode는 별도로 6H 여부도 표시
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

        # ====================================================
        # Episode 종료 이후 24H 안의 Failure
        #
        # Episode가 길 경우 시작 시점도 포함해서 평가
        # ====================================================

        future_24 = failures[
            (failures["FAILURE_DT"] >= start_dt)
            & (failures["FAILURE_DT"] <= (end_dt + pd.Timedelta(hours=24)))
        ]

        related_24h = not future_24.empty

        # ====================================================
        # DANGER용 6H
        # ====================================================

        future_6 = failures[
            (failures["FAILURE_DT"] >= start_dt)
            & (failures["FAILURE_DT"] <= (end_dt + pd.Timedelta(hours=6)))
        ]

        related_6h = not future_6.empty

        rows.append(
            {
                **episode.to_dict(),
                "RELATED_FAILURE_24H": int(related_24h),
                "RELATED_FAILURE_6H": int(related_6h),
                "FALSE_ALARM_24H": int(not related_24h),
                "DANGER_TRUE_6H": (
                    int(related_6h) if (max_level == "DANGER") else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 19. MAIN
# ============================================================


def main():

    print("=" * 80)

    print("INTEGRATED WARNING SYSTEM V1")

    print("=" * 80)

    # ========================================================
    # Data
    # ========================================================

    train_df = pd.read_csv(TRAIN_PATH)

    valid_df = pd.read_csv(VALID_PATH)

    # 시간형
    valid_df["OBSERVED_AT"] = pd.to_datetime(
        valid_df["OBSERVED_AT"], utc=True, errors="coerce"
    )

    print()

    print("Train:", train_df.shape)

    print("Validation:", valid_df.shape)

    # ========================================================
    # Feature
    # ========================================================

    (
        base_features,
        features_24h,
        features_6h,
    ) = build_feature_lists(train_df)

    print()

    print("=" * 80)

    print("FEATURE CHECK")

    print("=" * 80)

    print("BASE:", len(base_features))

    print("24H Feature:", len(features_24h))

    print("6H Feature:", len(features_6h))

    if len(base_features) != 240:

        raise RuntimeError(f"BASE Feature가 " f"240개가 아님: " f"{len(base_features)}")

    if len(features_24h) != 252:

        raise RuntimeError(f"24H Feature가 " f"252개가 아님: " f"{len(features_24h)}")

    if len(features_6h) != 240:

        raise RuntimeError(f"6H Feature가 " f"240개가 아님")

    # ========================================================
    # 24H Model
    # ========================================================

    result_24h = train_locked_model(
        train_df=train_df,
        feature_cols=features_24h,
        target_col="RISK_24H",
        censor_col="CENSORED_24H",
    )

    print()

    print("=" * 80)

    print("24H TRAIN")

    print("=" * 80)

    print("Rows:", result_24h["train_rows"])

    print("Negative:", result_24h["negative"])

    print("Positive:", result_24h["positive"])

    # ========================================================
    # 6H Model
    # ========================================================

    result_6h = train_locked_model(
        train_df=train_df,
        feature_cols=features_6h,
        target_col="RISK_6H",
        censor_col="CENSORED_6H",
    )

    print()

    print("=" * 80)

    print("6H TRAIN")

    print("=" * 80)

    print("Rows:", result_6h["train_rows"])

    print("Negative:", result_6h["negative"])

    print("Positive:", result_6h["positive"])

    # ========================================================
    # Validation 전체 936행 Probability
    #
    # Censor 여부와 무관하게 실제 운영처럼
    # 모든 시점에서 Prediction을 생성
    # ========================================================

    X_valid_24h = result_24h["imputer"].transform(valid_df[features_24h])

    prob_24h = result_24h["model"].predict_proba(X_valid_24h)[:, 1]

    X_valid_6h = result_6h["imputer"].transform(valid_df[features_6h])

    prob_6h = result_6h["model"].predict_proba(X_valid_6h)[:, 1]

    # ========================================================
    # Integrated DataFrame
    # ========================================================

    warning_df = valid_df[
        [
            "SCENARIO_ID",
            "OBSERVED_AT",
            "RISK_24H",
            "CENSORED_24H",
            "RISK_6H",
            "CENSORED_6H",
        ]
    ].copy()

    warning_df["PROB_24H"] = prob_24h

    warning_df["PROB_6H"] = prob_6h

    warning_df["PRED_24H"] = (warning_df["PROB_24H"] >= THRESHOLD_24H).astype(int)

    warning_df["PRED_6H"] = (warning_df["PROB_6H"] >= THRESHOLD_6H).astype(int)

    # ========================================================
    # Warning Level
    # ========================================================

    warning_df["WARNING_LEVEL"] = np.select(
        [
            warning_df["PRED_6H"] == 1,
            warning_df["PRED_24H"] == 1,
        ],
        [
            "DANGER",
            "CAUTION",
        ],
        default="NORMAL",
    )

    # ========================================================
    # 논리 불일치
    #
    # 6H = DANGER인데
    # 24H 모델은 CAUTION을 안 준 경우
    # ========================================================

    warning_df["DANGER_WITHOUT_CAUTION"] = (
        (warning_df["PRED_6H"] == 1) & (warning_df["PRED_24H"] == 0)
    ).astype(int)

    # ========================================================
    # Point-level 평가
    # ========================================================

    point_24h = point_level_evaluation(
        df=valid_df,
        target="RISK_24H",
        censor="CENSORED_24H",
        probability=prob_24h,
        threshold=THRESHOLD_24H,
    )

    point_6h = point_level_evaluation(
        df=valid_df,
        target="RISK_6H",
        censor="CENSORED_6H",
        probability=prob_6h,
        threshold=THRESHOLD_6H,
    )

    point_df = pd.DataFrame(
        [
            {
                "HORIZON": "24H",
                **point_24h,
            },
            {
                "HORIZON": "6H",
                **point_6h,
            },
        ]
    )

    print()

    print("=" * 80)

    print("POINT LEVEL VALIDATION")

    print("=" * 80)

    print(
        point_df[
            [
                "HORIZON",
                "N",
                "THRESHOLD",
                "ROC_AUC",
                "PR_AUC",
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
        .round(4)
        .to_string(index=False)
    )

    # ========================================================
    # Warning 분포
    # ========================================================

    warning_counts = (
        warning_df["WARNING_LEVEL"]
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

    warning_summary_df = (
        warning_counts.rename("COUNT")
        .reset_index()
        .rename(columns={"index": "WARNING_LEVEL"})
    )

    warning_summary_df["RATE"] = warning_summary_df["COUNT"] / len(warning_df)

    print()

    print("=" * 80)

    print("INTEGRATED WARNING DISTRIBUTION")

    print("=" * 80)

    print(warning_summary_df.round(4).to_string(index=False))

    inconsistency_count = int(warning_df["DANGER_WITHOUT_CAUTION"].sum())

    print()

    print("DANGER인데 24H CAUTION 모델은 음성:", inconsistency_count)

    # ========================================================
    # Failure Truth
    # ========================================================

    manifest = load_manifest()

    validation_scenarios = get_split_scenarios(
        manifest=manifest, split_name="validation"
    )

    failure_df = load_failure_events(
        manifest=manifest, scenario_ids=validation_scenarios
    )

    print()

    print("=" * 80)

    print("VALIDATION FAILURE EVENTS")

    print("=" * 80)

    print("Failure Event 수:", len(failure_df))

    print(
        "Failure Scenario 수:",
        failure_df["SCENARIO_ID"].nunique() if not failure_df.empty else 0,
    )

    # ========================================================
    # Event-level
    # ========================================================

    event_df = evaluate_failure_events(warning_df=warning_df, failure_df=failure_df)

    if not event_df.empty:

        n_events = len(event_df)

        caution_detected = int(event_df["CAUTION_DETECTED"].sum())

        danger_detected = int(event_df["DANGER_DETECTED"].sum())

        integrated_detected = int(event_df["INTEGRATED_DETECTED"].sum())

        print()

        print("=" * 80)

        print("EVENT LEVEL DETECTION")

        print("=" * 80)

        print(f"전체 Failure: " f"{n_events}")

        print()

        print("24H CAUTION 감지:")

        print(
            f"{caution_detected}"
            f" / "
            f"{n_events}"
            f" = "
            f"{caution_detected / n_events:.2%}"
        )

        print()

        print("6H DANGER 감지:")

        print(
            f"{danger_detected}"
            f" / "
            f"{n_events}"
            f" = "
            f"{danger_detected / n_events:.2%}"
        )

        print()

        print("통합 경보 감지:")

        print(
            f"{integrated_detected}"
            f" / "
            f"{n_events}"
            f" = "
            f"{integrated_detected / n_events:.2%}"
        )

        # ====================================================
        # Lead Time
        # ====================================================

        caution_leads = event_df["CAUTION_LEAD_H"].dropna()

        danger_leads = event_df["DANGER_LEAD_H"].dropna()

        integrated_leads = event_df["FIRST_WARNING_LEAD_H"].dropna()

        print()

        print("=" * 80)

        print("LEAD TIME")

        print("=" * 80)

        if not caution_leads.empty:

            print()

            print("[24H CAUTION]")

            print(f"Mean   : " f"{caution_leads.mean():.2f} h")

            print(f"Median : " f"{caution_leads.median():.2f} h")

            print(f"Min    : " f"{caution_leads.min():.2f} h")

            print(f"Max    : " f"{caution_leads.max():.2f} h")

        if not danger_leads.empty:

            print()

            print("[6H DANGER]")

            print(f"Mean   : " f"{danger_leads.mean():.2f} h")

            print(f"Median : " f"{danger_leads.median():.2f} h")

            print(f"Min    : " f"{danger_leads.min():.2f} h")

            print(f"Max    : " f"{danger_leads.max():.2f} h")

        if not integrated_leads.empty:

            print()

            print("[FIRST WARNING]")

            print(f"Mean   : " f"{integrated_leads.mean():.2f} h")

            print(f"Median : " f"{integrated_leads.median():.2f} h")

            print(f"Min    : " f"{integrated_leads.min():.2f} h")

            print(f"Max    : " f"{integrated_leads.max():.2f} h")

    # ========================================================
    # Alert Episode
    # ========================================================

    episode_df = build_alert_episodes(warning_df)

    episode_eval_df = evaluate_episodes(episode_df=episode_df, failure_df=failure_df)

    print()

    print("=" * 80)

    print("ALERT EPISODE SUMMARY")

    print("=" * 80)

    print("전체 경보 Episode:", len(episode_eval_df))

    if not episode_eval_df.empty:

        true_24 = int(episode_eval_df["RELATED_FAILURE_24H"].sum())

        false_24 = int(episode_eval_df["FALSE_ALARM_24H"].sum())

        print("24H 내 실제 Failure 관련 Episode:", true_24)

        print("24H 기준 False Alarm Episode:", false_24)

        danger_episodes = episode_eval_df[episode_eval_df["MAX_LEVEL"] == "DANGER"]

        if not danger_episodes.empty:

            danger_true = int(danger_episodes["RELATED_FAILURE_6H"].sum())

            danger_false = len(danger_episodes) - danger_true

            print()

            print("DANGER Episode:", len(danger_episodes))

            print("6H 내 Failure 관련 DANGER:", danger_true)

            print("6H 기준 False DANGER:", danger_false)

    # ========================================================
    # 저장
    # ========================================================

    warning_df.to_csv(
        OUTPUT_DIR / "casting_integrated_warning_validation_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    point_df.to_csv(
        OUTPUT_DIR / "casting_integrated_point_metrics_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    warning_summary_df.to_csv(
        OUTPUT_DIR / "casting_integrated_warning_summary_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    failure_df.to_csv(
        OUTPUT_DIR / "casting_validation_failure_events_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    event_df.to_csv(
        OUTPUT_DIR / "casting_integrated_event_metrics_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    episode_eval_df.to_csv(
        OUTPUT_DIR / "casting_integrated_alert_episodes_v1.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("=" * 80)

    print("FINAL LOCK")

    print("=" * 80)

    print("24H CAUTION:")

    print("XGB / 252 Features / Threshold 0.14")

    print()

    print("6H DANGER:")

    print("XGB / 240 Features / Threshold 0.035")

    print()

    print("WARNING RULE:")

    print("DANGER > CAUTION > NORMAL")

    print()

    print("=" * 80)

    print("SAVE COMPLETE")

    print("=" * 80)


# ============================================================
# 20. RUN
# ============================================================

if __name__ == "__main__":

    main()
