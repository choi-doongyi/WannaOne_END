# ============================================================
# early_warning_analysis.py
#
# Casting Predictive Maintenance
# 실제 고장 시점 기준 조기경보 Lead Time 분석
#
# 평가용 Truth:
# maintenance.FAILURE_DT
#
# 주의:
# FAILURE_DT는 모델 입력 Feature로 사용하지 않음.
# 오직 사후 평가에만 사용.
# ============================================================


import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


MANIFEST_PATH = ROOT / "manifest.json"


PREDICTION_PATH = OUTPUT_DIR / "casting_validation_alerts.csv"


CASTING_EQP = "P1-CC1-INS01"

MAINTENANCE_SOURCE_ID = "maintenance"


# ------------------------------------------------------------
# 최종 모델 Threshold
# ------------------------------------------------------------

THRESHOLD_24H = 0.47
THRESHOLD_6H = 0.26


# ------------------------------------------------------------
# 평가 Window
# ------------------------------------------------------------

WINDOW_24H = pd.Timedelta(hours=24)

WINDOW_6H = pd.Timedelta(hours=6)


# ============================================================
# 1. CSV 읽기
# ============================================================


def read_rows(path):

    with Path(path).open(encoding="utf-8-sig", newline="") as f:

        yield from csv.DictReader(f)


# ============================================================
# 2. Manifest 읽기
# ============================================================


def load_manifest():

    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ============================================================
# 3. Scenario별 Source 읽기
# ============================================================


def source_rows(scenario_id, file_id, manifest):

    source = next(
        (
            f
            for f in manifest["files"]
            if (f["scenarioId"] == scenario_id and f["id"] == file_id)
        ),
        None,
    )

    if source is None:
        return

    for chunk in source["chunks"]:

        chunk_path = ROOT / "chunks" / chunk["sha256"]

        for row in read_rows(chunk_path):

            yield row


# ============================================================
# 4. Validation Scenario 목록
# ============================================================


def get_validation_scenarios(manifest):

    return sorted(
        [
            scenario["id"]
            for scenario in manifest["scenarios"]
            if scenario["split"] == "validation"
        ]
    )


# ============================================================
# 5. Validation 예측 결과 읽기
# ============================================================


def load_predictions():

    df = pd.read_csv(PREDICTION_PATH)

    df["OBSERVED_AT"] = pd.to_datetime(df["OBSERVED_AT"], utc=True, errors="coerce")

    numeric_cols = [
        "RAW_RISK_6H",
        "RAW_RISK_24H",
        "RISK_6H",
        "RISK_24H",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["SCENARIO_ID", "OBSERVED_AT"])
        .sort_values(["SCENARIO_ID", "OBSERVED_AT"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 6. Maintenance에서 실제 고장시점 추출
# ============================================================


def load_failure_events(scenario_id, manifest):

    rows = list(source_rows(scenario_id, MAINTENANCE_SOURCE_ID, manifest) or [])

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 필수 컬럼 검사
    # --------------------------------------------------------

    if "FAILURE_DT" not in df.columns:

        raise RuntimeError(f"{scenario_id}: " f"maintenance에 FAILURE_DT가 없습니다.")

    # --------------------------------------------------------
    # 연주기 설비만
    # --------------------------------------------------------

    if "EQP_TAG" in df.columns:

        df = df[df["EQP_TAG"] == CASTING_EQP].copy()

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # 실제 고장시점
    # --------------------------------------------------------

    df["FAILURE_DT"] = pd.to_datetime(df["FAILURE_DT"], utc=True, errors="coerce")

    df = df[df["FAILURE_DT"].notna()].copy()

    if df.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # 같은 고장시각 중복 제거
    #
    # 하나의 고장에 여러 WO가 연결된 경우
    # 중복 평가되는 것을 방지
    # --------------------------------------------------------

    df = (
        df.sort_values("FAILURE_DT")
        .drop_duplicates(subset=["FAILURE_DT"])
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 7. 최초 경보 찾기
# ============================================================


def find_first_signal(scenario_pred, failure_dt, horizon, risk_column, threshold):

    start_time = failure_dt - horizon

    # --------------------------------------------------------
    # 고장 직전 Horizon 안의 예측만
    #
    # failure_dt와 정확히 같은 시각도 포함.
    # 이런 경우 Lead Time = 0H
    # --------------------------------------------------------

    window = scenario_pred[
        (scenario_pred["OBSERVED_AT"] >= start_time)
        & (scenario_pred["OBSERVED_AT"] <= failure_dt)
    ].copy()

    if window.empty:

        return {
            "detected": False,
            "first_time": pd.NaT,
            "lead_hours": np.nan,
            "risk": np.nan,
            "window_rows": 0,
        }

    signals = window[window[risk_column] >= threshold].copy()

    if signals.empty:

        return {
            "detected": False,
            "first_time": pd.NaT,
            "lead_hours": np.nan,
            "risk": np.nan,
            "window_rows": len(window),
        }

    # --------------------------------------------------------
    # 가장 먼저 발생한 경보
    # = 고장으로부터 가장 긴 리드타임
    # --------------------------------------------------------

    first = signals.sort_values("OBSERVED_AT").iloc[0]

    lead_hours = (failure_dt - first["OBSERVED_AT"]).total_seconds() / 3600

    return {
        "detected": True,
        "first_time": first["OBSERVED_AT"],
        "lead_hours": lead_hours,
        "risk": first[risk_column],
        "window_rows": len(window),
    }


# ============================================================
# 8. 고장 하나 분석
# ============================================================


def analyze_failure(scenario_id, failure_dt, predictions):

    scenario_pred = predictions[predictions["SCENARIO_ID"] == scenario_id].copy()

    # --------------------------------------------------------
    # 24H 조기경보
    #
    # 보정 전 모델 확률 사용
    # threshold 실험과 동일 조건 유지
    # --------------------------------------------------------

    result_24h = find_first_signal(
        scenario_pred=scenario_pred,
        failure_dt=failure_dt,
        horizon=WINDOW_24H,
        risk_column="RAW_RISK_24H",
        threshold=THRESHOLD_24H,
    )

    # --------------------------------------------------------
    # 6H 임박경보
    # --------------------------------------------------------

    result_6h = find_first_signal(
        scenario_pred=scenario_pred,
        failure_dt=failure_dt,
        horizon=WINDOW_6H,
        risk_column="RAW_RISK_6H",
        threshold=THRESHOLD_6H,
    )

    # --------------------------------------------------------
    # 고장 직전 최대 위험도
    # --------------------------------------------------------

    before_failure = scenario_pred[scenario_pred["OBSERVED_AT"] <= failure_dt].copy()

    last_24h = before_failure[
        before_failure["OBSERVED_AT"] >= (failure_dt - WINDOW_24H)
    ]

    last_6h = before_failure[before_failure["OBSERVED_AT"] >= (failure_dt - WINDOW_6H)]

    max_24h = last_24h["RAW_RISK_24H"].max() if not last_24h.empty else np.nan

    max_6h = last_6h["RAW_RISK_6H"].max() if not last_6h.empty else np.nan

    return {
        "SCENARIO_ID": scenario_id,
        "FAILURE_DT": failure_dt,
        # ================================================
        # 24H
        # ================================================
        "DETECTED_24H": result_24h["detected"],
        "FIRST_24H_SIGNAL": result_24h["first_time"],
        "LEAD_TIME_24H_HOURS": result_24h["lead_hours"],
        "FIRST_24H_RISK": result_24h["risk"],
        "MAX_24H_RISK": max_24h,
        "OBS_COUNT_24H": result_24h["window_rows"],
        # ================================================
        # 6H
        # ================================================
        "DETECTED_6H": result_6h["detected"],
        "FIRST_6H_SIGNAL": result_6h["first_time"],
        "LEAD_TIME_6H_HOURS": result_6h["lead_hours"],
        "FIRST_6H_RISK": result_6h["risk"],
        "MAX_6H_RISK": max_6h,
        "OBS_COUNT_6H": result_6h["window_rows"],
    }


# ============================================================
# 9. 요약 통계
# ============================================================


def make_summary(detail):

    rows = []

    # --------------------------------------------------------
    # 24H
    # --------------------------------------------------------

    detected_24 = detail[detail["DETECTED_24H"] == True]

    total = len(detail)

    rows.append(
        {
            "HORIZON": "24H",
            "TOTAL_FAILURES": total,
            "DETECTED": len(detected_24),
            "MISSED": total - len(detected_24),
            "DETECTION_RATE": (len(detected_24) / total if total > 0 else np.nan),
            "MEAN_LEAD_HOURS": detected_24["LEAD_TIME_24H_HOURS"].mean(),
            "MEDIAN_LEAD_HOURS": detected_24["LEAD_TIME_24H_HOURS"].median(),
            "MIN_LEAD_HOURS": detected_24["LEAD_TIME_24H_HOURS"].min(),
            "MAX_LEAD_HOURS": detected_24["LEAD_TIME_24H_HOURS"].max(),
        }
    )

    # --------------------------------------------------------
    # 6H
    # --------------------------------------------------------

    detected_6 = detail[detail["DETECTED_6H"] == True]

    rows.append(
        {
            "HORIZON": "6H",
            "TOTAL_FAILURES": total,
            "DETECTED": len(detected_6),
            "MISSED": total - len(detected_6),
            "DETECTION_RATE": (len(detected_6) / total if total > 0 else np.nan),
            "MEAN_LEAD_HOURS": detected_6["LEAD_TIME_6H_HOURS"].mean(),
            "MEDIAN_LEAD_HOURS": detected_6["LEAD_TIME_6H_HOURS"].median(),
            "MIN_LEAD_HOURS": detected_6["LEAD_TIME_6H_HOURS"].min(),
            "MAX_LEAD_HOURS": detected_6["LEAD_TIME_6H_HOURS"].max(),
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# 10. MAIN
# ============================================================


def main():

    print("=" * 80)
    print("EARLY WARNING LEAD TIME ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    manifest = load_manifest()

    validation_ids = get_validation_scenarios(manifest)

    print()

    print("Validation scenarios:", len(validation_ids))

    # --------------------------------------------------------
    # 모델 예측 결과
    # --------------------------------------------------------

    predictions = load_predictions()

    print("Prediction rows:", len(predictions))

    # --------------------------------------------------------
    # 실제 고장 목록
    # --------------------------------------------------------

    failure_records = []

    print()

    print("=" * 80)
    print("FAILURE EVENT LOAD")
    print("=" * 80)

    for i, scenario_id in enumerate(validation_ids, start=1):

        failures = load_failure_events(scenario_id, manifest)

        if not failures.empty:

            for _, row in failures.iterrows():

                failure_records.append(
                    {
                        "SCENARIO_ID": scenario_id,
                        "FAILURE_DT": row["FAILURE_DT"],
                    }
                )

        if i % 10 == 0 or i == len(validation_ids):

            print(
                f"{i}/{len(validation_ids)} "
                f"| 누적 고장 이벤트 "
                f"{len(failure_records)}"
            )

    failures_df = pd.DataFrame(failure_records)

    # --------------------------------------------------------
    # 실제 고장이 하나도 없다면
    # --------------------------------------------------------

    if failures_df.empty:

        raise RuntimeError(
            "Validation maintenance에서 "
            "연주기 FAILURE_DT를 찾지 못했습니다.\n"
            "EQP_TAG 또는 FAILURE_DT 구조를 확인해야 합니다."
        )

    failures_df = (
        failures_df.drop_duplicates(subset=["SCENARIO_ID", "FAILURE_DT"])
        .sort_values(["SCENARIO_ID", "FAILURE_DT"])
        .reset_index(drop=True)
    )

    print()

    print("실제 연주기 고장 이벤트:", len(failures_df))

    print("고장 Scenario 수:", failures_df["SCENARIO_ID"].nunique())

    # ========================================================
    # 고장별 리드타임 분석
    # ========================================================

    results = []

    for _, failure in failures_df.iterrows():

        result = analyze_failure(
            scenario_id=failure["SCENARIO_ID"],
            failure_dt=failure["FAILURE_DT"],
            predictions=predictions,
        )

        results.append(result)

    detail = pd.DataFrame(results)

    # ========================================================
    # 요약
    # ========================================================

    summary = make_summary(detail)

    # ========================================================
    # 출력
    # ========================================================

    print()

    print("=" * 80)
    print("EARLY WARNING SUMMARY")
    print("=" * 80)

    print(summary.round(4).to_string(index=False))

    # --------------------------------------------------------
    # 24H 탐지
    # --------------------------------------------------------

    print()

    print("=" * 80)
    print("24H EARLY WARNING")
    print("=" * 80)

    columns_24 = [
        "SCENARIO_ID",
        "FAILURE_DT",
        "DETECTED_24H",
        "FIRST_24H_SIGNAL",
        "LEAD_TIME_24H_HOURS",
        "FIRST_24H_RISK",
        "MAX_24H_RISK",
    ]

    print(detail[columns_24].round(4).to_string(index=False))

    # --------------------------------------------------------
    # 6H 탐지
    # --------------------------------------------------------

    print()

    print("=" * 80)
    print("6H IMMINENT WARNING")
    print("=" * 80)

    columns_6 = [
        "SCENARIO_ID",
        "FAILURE_DT",
        "DETECTED_6H",
        "FIRST_6H_SIGNAL",
        "LEAD_TIME_6H_HOURS",
        "FIRST_6H_RISK",
        "MAX_6H_RISK",
    ]

    print(detail[columns_6].round(4).to_string(index=False))

    # ========================================================
    # Missed Failure
    # ========================================================

    missed_24 = detail[detail["DETECTED_24H"] == False]

    missed_6 = detail[detail["DETECTED_6H"] == False]

    print()

    print("=" * 80)
    print("MISSED FAILURE CHECK")
    print("=" * 80)

    print("24H 놓친 고장:", len(missed_24))

    print("6H 놓친 고장:", len(missed_6))

    # ========================================================
    # 저장
    # ========================================================

    DETAIL_PATH = OUTPUT_DIR / "casting_early_warning_detail.csv"

    SUMMARY_PATH = OUTPUT_DIR / "casting_early_warning_summary.csv"

    FAILURES_PATH = OUTPUT_DIR / "casting_validation_failures.csv"

    detail.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")

    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    failures_df.to_csv(FAILURES_PATH, index=False, encoding="utf-8-sig")

    print()

    print("=" * 80)
    print("SAVE COMPLETE")
    print("=" * 80)

    print("고장 이벤트:", FAILURES_PATH)

    print("고장별 분석:", DETAIL_PATH)

    print("요약:", SUMMARY_PATH)


# ============================================================
# 11. 실행
# ============================================================

if __name__ == "__main__":

    main()
