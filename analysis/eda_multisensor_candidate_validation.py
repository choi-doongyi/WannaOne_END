from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ============================================================
# 0. 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

RESULT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. 원본 온도 데이터
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))


temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])

event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])


# ============================================================
# 2. 강종 연결
# ============================================================

cr1_event = event[
    (event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")
].sort_values("EVT_DT")


lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")


grade_event = cr1_event[["EVT_DT", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "AFT_VAL": "STEEL_GRADE"}
)


lot_grade = lot_start.merge(grade_event, on="start_time", how="left")


temp = temp.merge(lot_grade[["LOT_NO", "STEEL_GRADE"]], on="LOT_NO", how="left")


# ============================================================
# 3. Cycle 시간 정보
# ============================================================

cycle_info = pd.read_csv(RESULT_DIR / "tc_out1_cycle_maintenance_analysis.csv")


cycle_info["cycle_start"] = pd.to_datetime(cycle_info["cycle_start"])

cycle_info["cycle_end"] = pd.to_datetime(cycle_info["cycle_end"])


# ============================================================
# 4. 분석 센서
# ============================================================

exclude_cols = {"MEAS_DT", "LOT_NO", "STEEL_GRADE"}


sensor_cols = [
    col
    for col in temp.columns
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(temp[col])
]


print("\n센서 개수:", len(sensor_cols))
print(sensor_cols)


# ============================================================
# 5. 각 Cycle 센서 특징 계산
# ============================================================

features = []


for _, row in cycle_info.iterrows():

    lot_no = row["LOT_NO"]
    grade = row["STEEL_GRADE"]

    start = row["cycle_start"]
    end = row["cycle_end"]

    segment = temp[
        (temp["LOT_NO"] == lot_no)
        & (temp["MEAS_DT"] >= start)
        & (temp["MEAS_DT"] < end)
    ]

    if len(segment) == 0:
        continue

    result = {
        "LOT_NO": lot_no,
        "STEEL_GRADE": grade,
        "cycle_no": row["cycle_no"],
        "anomaly_score": row["anomaly_score"],
        "is_boundary_cycle": row["is_boundary_cycle"],
        "distance_days": row["distance_days"],
    }

    for sensor in sensor_cols:

        values = segment[sensor].dropna()

        if len(values) == 0:
            continue

        # 평균 수준
        result[f"{sensor}__mean"] = values.mean()

        # 표준편차
        result[f"{sensor}__std"] = values.std()

        # Robust 범위
        result[f"{sensor}__range90"] = values.quantile(0.95) - values.quantile(0.05)

    features.append(result)


feature_df = pd.DataFrame(features)


# ============================================================
# 6. Reference Pool
# ============================================================
#
# LOT 내부
# +
# 정기수리에서 7일 이상 떨어진 Cycle
# ============================================================

reference_mask = (feature_df["is_boundary_cycle"] == False) & (
    feature_df["distance_days"] > 7
)


# ============================================================
# 7. Robust Z-score
# ============================================================

feature_columns = [c for c in feature_df.columns if "__" in c]


z_results = feature_df[["LOT_NO", "STEEL_GRADE", "cycle_no", "anomaly_score"]].copy()


for feature in feature_columns:

    z_results[feature + "__z"] = np.nan


# 강종별 계산
for grade in feature_df["STEEL_GRADE"].dropna().unique():

    grade_reference = feature_df[reference_mask & (feature_df["STEEL_GRADE"] == grade)]

    grade_target_idx = feature_df["STEEL_GRADE"] == grade

    for feature in feature_columns:

        ref = grade_reference[feature].dropna()

        if len(ref) < 3:
            continue

        median = ref.median()

        mad = (ref - median).abs().median()

        if mad == 0:
            continue

        z = 0.6745 * (feature_df.loc[grade_target_idx, feature] - median) / mad

        z_results.loc[grade_target_idx, feature + "__z"] = z


# ============================================================
# 8. Residual 후보
# ============================================================

candidate_keys = [(240091, 4), (240091, 5), (240037, 9), (240121, 12)]


candidate_rows = []


for lot_no, cycle_no in candidate_keys:

    row = z_results[
        (z_results["LOT_NO"] == lot_no) & (z_results["cycle_no"] == cycle_no)
    ]

    if len(row) == 0:
        continue

    row = row.iloc[0]

    for feature in feature_columns:

        z_col = feature + "__z"

        z_value = row[z_col]

        if pd.isna(z_value):
            continue

        sensor = feature.split("__")[0]

        statistic = feature.split("__")[1]

        candidate_rows.append(
            {
                "LOT_NO": lot_no,
                "cycle_no": cycle_no,
                "STEEL_GRADE": row["STEEL_GRADE"],
                "sensor": sensor,
                "statistic": statistic,
                "robust_z": z_value,
                "abs_z": abs(z_value),
            }
        )


candidate_df = pd.DataFrame(candidate_rows)


# ============================================================
# 9. 후보별 TOP 센서
# ============================================================

print("\n" + "=" * 120)
print("[Residual 후보별 다중센서 편차 TOP 15]")
print("=" * 120)


for (lot_no, cycle_no), group in candidate_df.groupby(["LOT_NO", "cycle_no"]):

    print(f"\nLOT {lot_no} " f"/ Cycle {cycle_no}")

    print(
        group.sort_values("abs_z", ascending=False)
        .head(15)[["sensor", "statistic", "robust_z"]]
        .round(3)
        .to_string(index=False)
    )


# ============================================================
# 10. 센서 단위로 통합
# ============================================================

sensor_summary = (
    candidate_df.groupby(["LOT_NO", "cycle_no", "sensor"])["abs_z"]
    .max()
    .reset_index(name="max_abs_z")
)


print("\n" + "=" * 120)
print("[후보별 영향 센서 TOP 10]")
print("=" * 120)


for (lot_no, cycle_no), group in sensor_summary.groupby(["LOT_NO", "cycle_no"]):

    print(f"\nLOT {lot_no} " f"/ Cycle {cycle_no}")

    print(
        group.sort_values("max_abs_z", ascending=False)
        .head(10)
        .round(3)
        .to_string(index=False)
    )


# ============================================================
# 11. 여러 후보에서 반복 등장하는 센서
# ============================================================

high_sensor = sensor_summary[sensor_summary["max_abs_z"] >= 3]


repeat_sensor = (
    high_sensor.groupby("sensor")
    .agg(
        candidate_count=("LOT_NO", "count"),
        max_abs_z=("max_abs_z", "max"),
        mean_abs_z=("max_abs_z", "mean"),
    )
    .reset_index()
    .sort_values(["candidate_count", "max_abs_z"], ascending=False)
)


print("\n" + "=" * 120)
print("[여러 후보에서 반복적으로 변한 센서]")
print("=" * 120)

print(repeat_sensor.round(3).to_string(index=False))


# ============================================================
# 12. 저장
# ============================================================

feature_df.to_csv(
    RESULT_DIR / "cycle_multisensor_features.csv", index=False, encoding="utf-8-sig"
)

candidate_df.to_csv(
    RESULT_DIR / "residual_candidate_multisensor_z.csv",
    index=False,
    encoding="utf-8-sig",
)

sensor_summary.to_csv(
    RESULT_DIR / "residual_candidate_sensor_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n결과 저장 완료")
