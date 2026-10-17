from pathlib import Path
import sys
import pandas as pd
import numpy as np

# ============================================================
# 0. 프로젝트 경로
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
# 3. LOT - 강종 매핑
# ============================================================

lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")

grade_event = cr1_event[["EVT_DT", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "AFT_VAL": "STEEL_GRADE"}
)

lot_grade = lot_start.merge(grade_event, on="start_time", how="left")

temp = temp.merge(lot_grade[["LOT_NO", "STEEL_GRADE"]], on="LOT_NO", how="left")


# ============================================================
# 4. 센서 컬럼
# ============================================================

sensor_cols = [
    col for col in temp.select_dtypes(include="number").columns if col != "LOT_NO"
]


# ============================================================
# 5. LOT별 평균
# ============================================================

lot_mean = temp.groupby(["STEEL_GRADE", "LOT_NO"])[sensor_cols].mean().reset_index()


# ============================================================
# 6. LOT 기준 강종 평균
# ============================================================

grade_mean = lot_mean.groupby("STEEL_GRADE")[sensor_cols].mean()


# ============================================================
# 7. 반복 LOT가 존재하는 강종
# ============================================================

grade_lot_count = lot_mean.groupby("STEEL_GRADE")["LOT_NO"].nunique()

repeated_grades = grade_lot_count[grade_lot_count >= 2].index.tolist()

print("\n반복 LOT가 존재하는 강종:")
print(repeated_grades)


# ============================================================
# 8. 센서별 강종 차이 vs 동일 강종 LOT 차이
# ============================================================

results = []


for sensor in sensor_cols:

    # --------------------------------------------------------
    # 강종 간 평균 범위
    # --------------------------------------------------------

    between_grade_range = grade_mean[sensor].max() - grade_mean[sensor].min()

    # --------------------------------------------------------
    # 같은 강종 내부 LOT 평균 범위
    # --------------------------------------------------------

    within_ranges = []

    for grade in repeated_grades:

        values = lot_mean.loc[lot_mean["STEEL_GRADE"] == grade, sensor]

        within_range = values.max() - values.min()

        within_ranges.append(within_range)

    mean_within_range = np.mean(within_ranges)

    max_within_range = np.max(within_ranges)

    # --------------------------------------------------------
    # 강종 분리 정도
    # --------------------------------------------------------

    separation_ratio = between_grade_range / (mean_within_range + 1e-9)

    results.append(
        {
            "sensor": sensor,
            "between_grade_range": between_grade_range,
            "mean_within_grade_range": mean_within_range,
            "max_within_grade_range": max_within_range,
            "separation_ratio": separation_ratio,
        }
    )


result_df = pd.DataFrame(results)


# ============================================================
# 9. 강종 영향이 상대적으로 큰 센서
# ============================================================

result_df = result_df.sort_values("separation_ratio", ascending=False)


print("\n" + "=" * 100)
print("[강종 간 차이 / 동일 강종 LOT 차이]")
print("=" * 100)

print(result_df.round(4).to_string(index=False))


# ============================================================
# 10. 상위 센서
# ============================================================

print("\n" + "=" * 100)
print("[강종 구분력이 상대적으로 높은 센서 TOP 10]")
print("=" * 100)

print(result_df.head(10).round(4).to_string(index=False))


# ============================================================
# 11. LOT 변동이 상대적으로 큰 센서
# ============================================================

print("\n" + "=" * 100)
print("[동일 강종 내 LOT 변동이 상대적으로 큰 센서]")
print("=" * 100)

print(
    result_df.sort_values("max_within_grade_range", ascending=False)
    .head(10)
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 12. 결과 저장
# ============================================================

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

result_df.to_csv(
    RESULT_DIR / "grade_variability.csv", index=False, encoding="utf-8-sig"
)

lot_mean.to_csv(RESULT_DIR / "lot_sensor_mean.csv", index=False, encoding="utf-8-sig")

print("\n결과 저장 완료")
