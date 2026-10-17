from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

# ============================================================
# 1. 데이터 로드
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))

temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])
event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])


# ============================================================
# 2. LOT - 강종 매핑
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
# 3. SPHC만 추출
# ============================================================

sphc = temp[temp["STEEL_GRADE"] == "SPHC"].sort_values(["LOT_NO", "MEAS_DT"]).copy()


# ============================================================
# 4. 분석 변수
# ============================================================

sensors = ["TC-OUT1", "TC-OUT2", "OP-Z3", "OP-Z4"]

target_lots = [240044, 240061, 240069, 240091]


# ============================================================
# 5. Rolling 분석
# ============================================================

# 1초 데이터 기준 약 5분
window = 300


FIGURE_DIR = PROJECT_ROOT / "figures" / "change_point"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


for lot_no in target_lots:

    lot = sphc[sphc["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    start_time = lot["MEAS_DT"].min()

    # 실제 경과시간(분)
    lot["elapsed_min"] = (lot["MEAS_DT"] - start_time).dt.total_seconds() / 60

    for sensor in sensors:

        # -----------------------------------------
        # 5분 Rolling 평균
        # -----------------------------------------

        lot[f"{sensor}_roll_mean"] = (
            lot[sensor].rolling(window=window, min_periods=60).mean()
        )

        # -----------------------------------------
        # 5분 Rolling 표준편차
        # -----------------------------------------

        lot[f"{sensor}_roll_std"] = (
            lot[sensor].rolling(window=window, min_periods=60).std()
        )

        # =========================================
        # 그래프 1 : Raw + Rolling Mean
        # =========================================

        plt.figure(figsize=(13, 6))

        plt.plot(lot["elapsed_min"], lot[sensor], alpha=0.25, label="Raw")

        plt.plot(
            lot["elapsed_min"],
            lot[f"{sensor}_roll_mean"],
            linewidth=2,
            label="5-min Rolling Mean",
        )

        plt.xlabel("Elapsed Time (min)")

        plt.ylabel(sensor)

        plt.title(f"SPHC LOT {lot_no} - {sensor}")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{lot_no}_{sensor}_rolling_mean.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()

        # =========================================
        # 그래프 2 : Rolling Std
        # =========================================

        plt.figure(figsize=(13, 5))

        plt.plot(lot["elapsed_min"], lot[f"{sensor}_roll_std"])

        plt.xlabel("Elapsed Time (min)")

        plt.ylabel("Rolling Std")

        plt.title(f"SPHC LOT {lot_no} - {sensor} Rolling Std")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{lot_no}_{sensor}_rolling_std.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.show()


print("\n저장 완료")
print(FIGURE_DIR)
