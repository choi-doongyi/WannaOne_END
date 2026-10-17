from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ============================================================
# 0. 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

RESULT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. 데이터 로드
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))

maintenance = load_data(find_file("G-01_정기수리캘린더"))


temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])

event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])

maintenance["STRT_DT"] = pd.to_datetime(maintenance["STRT_DT"])

maintenance["END_DT"] = pd.to_datetime(maintenance["END_DT"])


# ============================================================
# 2. CR1 강종 매핑
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
# 3. CR1 정기수리
# ============================================================

cr1_maintenance = (
    maintenance[maintenance["PLANT_CD"] == "CR1"].copy().sort_values("STRT_DT")
)


print("\n" + "=" * 100)
print("[CR1 정기수리]")
print("=" * 100)

print(cr1_maintenance[["PLAN_ID", "STRT_DT", "END_DT", "SCOPE"]].to_string(index=False))


# ============================================================
# 4. Cycle 시작/종료 시각 다시 생성
# ============================================================

cycle_time_results = []


for lot_no in sorted(temp["LOT_NO"].unique()):

    lot = temp[temp["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    grade = lot["STEEL_GRADE"].iloc[0]

    series = (
        lot.set_index("MEAS_DT")["TC-OUT1"]
        .resample("1min")
        .mean()
        .interpolate(method="time", limit=2)
    )

    smooth = series.rolling(window=5, center=True, min_periods=3).mean().dropna()

    peaks, _ = find_peaks(smooth.values, distance=30, prominence=5)

    peak_times = smooth.index[peaks]

    if len(peak_times) < 2:
        continue

    for i in range(len(peak_times) - 1):

        cycle_start = peak_times[i]
        cycle_end = peak_times[i + 1]

        cycle_time_results.append(
            {
                "LOT_NO": lot_no,
                "STEEL_GRADE": grade,
                "cycle_no": i + 1,
                "cycle_start": cycle_start,
                "cycle_end": cycle_end,
                "cycle_mid": cycle_start + (cycle_end - cycle_start) / 2,
            }
        )


cycle_time = pd.DataFrame(cycle_time_results)


# ============================================================
# 5. 이상점수 연결
# ============================================================

score = pd.read_csv(RESULT_DIR / "tc_out1_cycle_anomaly_score.csv")


merged = score.merge(cycle_time, on=["LOT_NO", "STEEL_GRADE", "cycle_no"], how="left")


# ============================================================
# 6. 정기수리와 시간거리 계산
# ============================================================


def maintenance_relation(row):

    t = row["cycle_mid"]

    if pd.isna(t):
        return pd.Series(
            {
                "maintenance_relation": "UNKNOWN",
                "nearest_plan": np.nan,
                "distance_days": np.nan,
            }
        )

    best_relation = "NONE"
    best_plan = None
    best_distance = np.inf

    for _, m in cr1_maintenance.iterrows():

        start = m["STRT_DT"]
        end = m["END_DT"]

        # 정기수리 기간 내부
        if start <= t <= end:

            relation = "DURING"
            distance = 0

        # 정기수리 직전
        elif t < start:

            distance = (start - t).total_seconds() / 86400

            relation = "BEFORE"

        # 정기수리 직후
        else:

            distance = (t - end).total_seconds() / 86400

            relation = "AFTER"

        if distance < best_distance:

            best_distance = distance
            best_relation = relation
            best_plan = m["PLAN_ID"]

    return pd.Series(
        {
            "maintenance_relation": best_relation,
            "nearest_plan": best_plan,
            "distance_days": best_distance,
        }
    )


relation = merged.apply(maintenance_relation, axis=1)


merged = pd.concat([merged, relation], axis=1)


# ============================================================
# 7. 정기수리 근접 여부
# ============================================================

merged["near_maintenance_3d"] = merged["distance_days"] <= 3

merged["near_maintenance_7d"] = merged["distance_days"] <= 7


# ============================================================
# 8. 상위 이상 Cycle 확인
# ============================================================

print("\n" + "=" * 120)
print("[이상점수 상위 Cycle + 정기수리 관계]")
print("=" * 120)

print(
    merged[
        [
            "LOT_NO",
            "STEEL_GRADE",
            "cycle_no",
            "is_boundary_cycle",
            "anomaly_score",
            "cycle_start",
            "cycle_end",
            "maintenance_relation",
            "distance_days",
            "near_maintenance_3d",
        ]
    ]
    .sort_values("anomaly_score", ascending=False)
    .head(30)
    .round({"anomaly_score": 3, "distance_days": 2})
    .to_string(index=False)
)


# ============================================================
# 9. 정기수리 근접 여부별 이상점수
# ============================================================

print("\n" + "=" * 120)
print("[정기수리 ±3일 여부별 이상점수]")
print("=" * 120)

print(
    merged.groupby("near_maintenance_3d")["anomaly_score"]
    .agg(count="count", mean="mean", median="median", max="max")
    .round(3)
)


print("\n" + "=" * 120)
print("[정기수리 전 / 후 / 기타]")
print("=" * 120)

print(
    merged.groupby("maintenance_relation")["anomaly_score"]
    .agg(count="count", mean="mean", median="median", max="max")
    .round(3)
)


# ============================================================
# 10. 내부 Cycle만 별도로 확인
# ============================================================

internal = merged[~merged["is_boundary_cycle"]].copy()


print("\n" + "=" * 120)
print("[LOT 내부 이상점수 TOP 15]")
print("=" * 120)

print(
    internal[
        [
            "LOT_NO",
            "STEEL_GRADE",
            "cycle_no",
            "anomaly_score",
            "maintenance_relation",
            "distance_days",
        ]
    ]
    .sort_values("anomaly_score", ascending=False)
    .head(15)
    .round({"anomaly_score": 3, "distance_days": 2})
    .to_string(index=False)
)


# ============================================================
# 11. 저장
# ============================================================

merged.to_csv(
    RESULT_DIR / "tc_out1_cycle_maintenance_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)


print("\n결과 저장 완료")
