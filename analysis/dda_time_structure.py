from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

# ============================================================
# 1. 데이터 불러오기
# ============================================================

temp_path = find_file("T-CR1-CAL01_온도")
event_path = find_file("G-02_조업이벤트")
repair_path = find_file("G-01_정기수리캘린더")

temp = load_data(temp_path)
event = load_data(event_path)
repair = load_data(repair_path)


# ============================================================
# 2. 시간형 변환
# ============================================================

temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])

event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])

repair["STRT_DT"] = pd.to_datetime(repair["STRT_DT"])
repair["END_DT"] = pd.to_datetime(repair["END_DT"])


# ============================================================
# 3. CR1 이벤트 구조 확인
# ============================================================

print("\n" + "=" * 80)
print("[1] 공장별 이벤트 종류")
print("=" * 80)

print(pd.crosstab(event["PLANT_CD"], event["EVT_TYPE"]))


# ============================================================
# 4. CR1 이벤트만 추출
# ============================================================

cr1_event = (
    event[event["PLANT_CD"] == "CR1"].sort_values("EVT_DT").reset_index(drop=True)
)

print("\n" + "=" * 80)
print("[2] CR1 이벤트")
print("=" * 80)

print(cr1_event)

print("\nCR1 이벤트 개수:", len(cr1_event))


# ============================================================
# 5. LOT별 시간 범위
# ============================================================

lot_summary = (
    temp.groupby("LOT_NO")
    .agg(
        start_time=("MEAS_DT", "min"),
        end_time=("MEAS_DT", "max"),
        n_rows=("MEAS_DT", "size"),
    )
    .reset_index()
)

lot_summary["duration"] = lot_summary["end_time"] - lot_summary["start_time"]

print("\n" + "=" * 80)
print("[3] LOT별 시간 범위")
print("=" * 80)

print(lot_summary.to_string(index=False))


# ============================================================
# 6. LOT 변경 시점
# ============================================================

temp = temp.sort_values("MEAS_DT").reset_index(drop=True)

lot_change = temp["LOT_NO"] != temp["LOT_NO"].shift()

lot_change_df = temp.loc[lot_change, ["MEAS_DT", "LOT_NO"]].copy()

lot_change_df["prev_LOT_NO"] = temp["LOT_NO"].shift().loc[lot_change].values

print("\n" + "=" * 80)
print("[4] LOT 변경 시점")
print("=" * 80)

print(lot_change_df.to_string(index=False))


# ============================================================
# 7. 시간 간격 확인
# ============================================================

temp["time_diff"] = temp["MEAS_DT"].diff().dt.total_seconds()

print("\n" + "=" * 80)
print("[5] 온도 데이터 시간 간격")
print("=" * 80)

print(temp["time_diff"].describe())

print("\n시간 간격 상위 20개")

print(
    temp[["MEAS_DT", "LOT_NO", "time_diff"]]
    .sort_values("time_diff", ascending=False)
    .head(20)
    .to_string(index=False)
)


# ============================================================
# 8. CR1 정기수리만 확인
# ============================================================

cr1_repair = repair[repair["PLANT_CD"] == "CR1"].copy()

print("\n" + "=" * 80)
print("[6] CR1 정기수리")
print("=" * 80)

print(cr1_repair.to_string(index=False))
