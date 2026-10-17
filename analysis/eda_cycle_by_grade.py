from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.signal import find_peaks

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
# 2. LOT - 강종 연결
# ============================================================

cr1_event = event[
    (event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")
].sort_values("EVT_DT")

lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")

grade_event = cr1_event[["EVT_DT", "BEF_VAL", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "BEF_VAL": "BEF_GRADE", "AFT_VAL": "STEEL_GRADE"}
)

lot_grade = lot_start.merge(grade_event, on="start_time", how="left")

temp = temp.merge(
    lot_grade[["LOT_NO", "BEF_GRADE", "STEEL_GRADE"]], on="LOT_NO", how="left"
)


# ============================================================
# 3. 분석 센서
# ============================================================

sensors = ["TC-OUT1", "TC-OUT2"]


# ============================================================
# 4. Peak 검출 함수
# ============================================================


def detect_peaks(series, min_distance_min=30, prominence=5):

    peaks, properties = find_peaks(
        series.values, distance=min_distance_min, prominence=prominence
    )

    return peaks, properties


# ============================================================
# 5. 전체 LOT 분석
# ============================================================

lot_results = []
cycle_results = []


for lot_no in sorted(temp["LOT_NO"].unique()):

    lot = temp[temp["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    steel_grade = lot["STEEL_GRADE"].iloc[0]

    bef_grade = lot["BEF_GRADE"].iloc[0]

    # --------------------------------------------------------
    # LOT 시간 정보
    # --------------------------------------------------------

    start_time = lot["MEAS_DT"].min()
    end_time = lot["MEAS_DT"].max()

    duration_min = (end_time - start_time).total_seconds() / 60

    # --------------------------------------------------------
    # 1분 단위 평균
    # --------------------------------------------------------

    lot_min = lot.set_index("MEAS_DT")[sensors].resample("1min").mean()

    # 짧은 공백만 보간
    lot_min = lot_min.interpolate(method="time", limit=2)

    for sensor in sensors:

        series = lot_min[sensor]

        # ----------------------------------------------------
        # 5분 이동평균
        # ----------------------------------------------------

        smooth = series.rolling(window=5, center=True, min_periods=3).mean().dropna()

        # ----------------------------------------------------
        # Peak 검출
        # ----------------------------------------------------

        peaks, properties = detect_peaks(smooth, min_distance_min=30, prominence=5)

        peak_times = smooth.index[peaks]

        # ----------------------------------------------------
        # Peak 간격
        # ----------------------------------------------------

        if len(peak_times) >= 2:

            cycle_min = (
                pd.Series(peak_times).diff().dt.total_seconds().div(60).dropna().values
            )

        else:

            cycle_min = np.array([])

        # ----------------------------------------------------
        # LOT 수준 통계
        # ----------------------------------------------------

        lot_results.append(
            {
                "LOT_NO": lot_no,
                "BEF_GRADE": bef_grade,
                "STEEL_GRADE": steel_grade,
                "sensor": sensor,
                "duration_min": duration_min,
                "n_peaks": len(peak_times),
                "n_cycles": len(cycle_min),
                "mean_cycle_min": (
                    np.mean(cycle_min) if len(cycle_min) > 0 else np.nan
                ),
                "std_cycle_min": (
                    np.std(cycle_min, ddof=1) if len(cycle_min) > 1 else np.nan
                ),
                # 신호 중심 수준
                "mean_level": smooth.mean(),
                # robust amplitude 후보
                "q05": smooth.quantile(0.05),
                "q95": smooth.quantile(0.95),
                "q95_q05_range": (smooth.quantile(0.95) - smooth.quantile(0.05)),
            }
        )

        # ----------------------------------------------------
        # 개별 사이클 저장
        # ----------------------------------------------------

        for i in range(1, len(peak_times)):

            cycle_results.append(
                {
                    "LOT_NO": lot_no,
                    "STEEL_GRADE": steel_grade,
                    "sensor": sensor,
                    "cycle_no": i,
                    "cycle_min": (peak_times[i] - peak_times[i - 1]).total_seconds()
                    / 60,
                }
            )


# ============================================================
# 6. DataFrame
# ============================================================

lot_df = pd.DataFrame(lot_results)

cycle_df = pd.DataFrame(cycle_results)


# ============================================================
# 7. LOT별 결과
# ============================================================

print("\n" + "=" * 120)
print("[전체 LOT 주기 분석]")
print("=" * 120)

print(
    lot_df[
        [
            "LOT_NO",
            "BEF_GRADE",
            "STEEL_GRADE",
            "sensor",
            "duration_min",
            "n_peaks",
            "n_cycles",
            "mean_cycle_min",
            "std_cycle_min",
            "mean_level",
            "q95_q05_range",
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 8. 강종별 사이클 요약
# ============================================================

# 사이클이 실제로 1개 이상 측정된 경우만
valid_cycle = cycle_df.copy()


grade_cycle_summary = (
    valid_cycle.groupby(["STEEL_GRADE", "sensor"])["cycle_min"]
    .agg(
        n_cycles="count",
        mean_cycle_min="mean",
        std_cycle_min="std",
        min_cycle_min="min",
        max_cycle_min="max",
    )
    .reset_index()
)


print("\n" + "=" * 120)
print("[강종별 사이클 길이]")
print("=" * 120)

print(grade_cycle_summary.round(3).to_string(index=False))


# ============================================================
# 9. 강종별 신호 수준 / 진폭
# ============================================================

grade_level_summary = (
    lot_df.groupby(["STEEL_GRADE", "sensor"])
    .agg(
        n_lots=("LOT_NO", "nunique"),
        mean_level=("mean_level", "mean"),
        mean_amplitude=("q95_q05_range", "mean"),
    )
    .reset_index()
)


print("\n" + "=" * 120)
print("[강종별 TC-OUT 수준 / 진폭]")
print("=" * 120)

print(grade_level_summary.round(3).to_string(index=False))


# ============================================================
# 10. 저장
# ============================================================

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)


lot_df.to_csv(
    RESULT_DIR / "cycle_by_grade_lot_summary.csv", index=False, encoding="utf-8-sig"
)

cycle_df.to_csv(
    RESULT_DIR / "cycle_by_grade_detail.csv", index=False, encoding="utf-8-sig"
)

grade_cycle_summary.to_csv(
    RESULT_DIR / "cycle_by_grade_summary.csv", index=False, encoding="utf-8-sig"
)

grade_level_summary.to_csv(
    RESULT_DIR / "cycle_grade_level_summary.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
