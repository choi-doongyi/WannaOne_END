from pathlib import Path
import sys
import pandas as pd

# ============================================================
# 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

# ============================================================
# 1. 데이터 불러오기
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))


temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])

event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])


# ============================================================
# 2. CR1 강종변경 이벤트
# ============================================================

cr1_event = event[
    (event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")
].sort_values("EVT_DT")


# ============================================================
# 3. LOT 시작시간
# ============================================================

lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")


# ============================================================
# 4. LOT - 강종 매핑
# ============================================================

grade_event = cr1_event[["EVT_DT", "BEF_VAL", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "AFT_VAL": "STEEL_GRADE"}
)


lot_grade = lot_start.merge(grade_event, on="start_time", how="left")


print("\n" + "=" * 100)
print("[1] LOT - 강종 매핑")
print("=" * 100)

print(lot_grade.to_string(index=False))


# ============================================================
# 5. 온도 데이터에 강종 추가
# ============================================================

temp = temp.merge(lot_grade[["LOT_NO", "STEEL_GRADE"]], on="LOT_NO", how="left")


print("\n" + "=" * 100)
print("[2] 강종 매핑 확인")
print("=" * 100)

print(temp[["MEAS_DT", "LOT_NO", "STEEL_GRADE"]].head())


print("\n강종 미매핑 행:", temp["STEEL_GRADE"].isna().sum())


# ============================================================
# 6. 강종별 데이터 수
# ============================================================

print("\n" + "=" * 100)
print("[3] 강종별 데이터 수")
print("=" * 100)

print(temp["STEEL_GRADE"].value_counts())


# ============================================================
# 7. 분석 센서
# ============================================================

sensor_cols = [
    col for col in temp.columns if col not in ["MEAS_DT", "LOT_NO", "STEEL_GRADE"]
]


# ============================================================
# 8. 강종별 평균
# ============================================================

grade_mean = temp.groupby("STEEL_GRADE")[sensor_cols].mean()

print("\n" + "=" * 100)
print("[4] 강종별 평균")
print("=" * 100)

print(grade_mean.round(3))


# ============================================================
# 9. 강종별 표준편차
# ============================================================

grade_std = temp.groupby("STEEL_GRADE")[sensor_cols].std()

print("\n" + "=" * 100)
print("[5] 강종별 표준편차")
print("=" * 100)

print(grade_std.round(3))


# ============================================================
# 10. LOT별 평균
# ============================================================

lot_mean = temp.groupby(["STEEL_GRADE", "LOT_NO"])[sensor_cols].mean().reset_index()


print("\n" + "=" * 100)
print("[6] 강종 / LOT별 평균")
print("=" * 100)

print(lot_mean.round(3).to_string(index=False))
