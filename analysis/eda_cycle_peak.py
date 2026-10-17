from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

sphc = temp[temp["STEEL_GRADE"] == "SPHC"].copy()


target_lots = [240044, 240061, 240069, 240091]

sensors = ["TC-OUT1", "TC-OUT2"]


# ============================================================
# 4. 저장 폴더
# ============================================================

FIGURE_DIR = PROJECT_ROOT / "figures" / "cycle_peak"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# 5. Peak 검출 함수
# ============================================================


def detect_peaks(series, min_distance_min=30, prominence=5):

    values = series.values

    peaks, properties = find_peaks(
        values,
        # 서로 너무 가까운 peak 제거
        distance=min_distance_min,
        # 어느 정도 뚜렷한 peak만 사용
        prominence=prominence,
    )

    return peaks, properties


# ============================================================
# 6. LOT별 Peak 분석
# ============================================================

peak_summary = []
cycle_detail = []


for lot_no in target_lots:

    lot = (
        sphc[sphc["LOT_NO"] == lot_no]
        .copy()
        .sort_values("MEAS_DT")
        .set_index("MEAS_DT")
    )

    # --------------------------------------------------------
    # 1분 단위 평균
    # --------------------------------------------------------

    lot_min = lot[sensors].resample("1min").mean()

    # --------------------------------------------------------
    # 혹시 1분 bin이 비어 있으면
    # 2분 이하의 짧은 공백만 보간
    # --------------------------------------------------------

    lot_min = lot_min.interpolate(method="time", limit=2)

    for sensor in sensors:

        series = lot_min[sensor].copy()

        # ====================================================
        # 5분 이동평균
        # ====================================================

        smooth = series.rolling(window=5, center=True, min_periods=3).mean()

        # 남아 있는 NaN 제거
        valid = smooth.notna()

        smooth_valid = smooth[valid]

        # ====================================================
        # Peak 검출
        # ====================================================

        peaks, properties = detect_peaks(
            smooth_valid, min_distance_min=30, prominence=5
        )

        peak_times = smooth_valid.index[peaks]

        peak_values = smooth_valid.iloc[peaks].values

        # ====================================================
        # Peak 간격 계산
        # ====================================================

        if len(peak_times) >= 2:

            cycle_minutes = (
                pd.Series(peak_times).diff().dt.total_seconds().div(60).dropna().values
            )

        else:

            cycle_minutes = np.array([])

        # ====================================================
        # 요약 저장
        # ====================================================

        peak_summary.append(
            {
                "LOT_NO": lot_no,
                "sensor": sensor,
                "n_peaks": len(peak_times),
                "mean_cycle_min": (
                    np.mean(cycle_minutes) if len(cycle_minutes) > 0 else np.nan
                ),
                "std_cycle_min": (
                    np.std(cycle_minutes, ddof=1) if len(cycle_minutes) > 1 else np.nan
                ),
                "min_cycle_min": (
                    np.min(cycle_minutes) if len(cycle_minutes) > 0 else np.nan
                ),
                "max_cycle_min": (
                    np.max(cycle_minutes) if len(cycle_minutes) > 0 else np.nan
                ),
            }
        )

        # ====================================================
        # 각 사이클 상세 저장
        # ====================================================

        for i in range(1, len(peak_times)):

            cycle_detail.append(
                {
                    "LOT_NO": lot_no,
                    "sensor": sensor,
                    "cycle_no": i,
                    "peak_start": peak_times[i - 1],
                    "peak_end": peak_times[i],
                    "cycle_min": (peak_times[i] - peak_times[i - 1]).total_seconds()
                    / 60,
                }
            )

        # ====================================================
        # 그래프
        # ====================================================

        plt.figure(figsize=(13, 6))

        plt.plot(smooth_valid.index, smooth_valid.values, label="5-min Rolling Mean")

        plt.scatter(peak_times, peak_values, s=80, label="Detected Peak")

        # Peak 번호 표시
        for i, (peak_time, peak_value) in enumerate(
            zip(peak_times, peak_values), start=1
        ):

            plt.annotate(
                str(i),
                (peak_time, peak_value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
            )

        plt.xlabel("Time")

        plt.ylabel(sensor)

        plt.title(f"SPHC LOT {lot_no} - " f"{sensor} Peak Detection")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIR / f"{lot_no}_{sensor}_peak.png", dpi=200, bbox_inches="tight"
        )

        plt.show()


# ============================================================
# 7. DataFrame 생성
# ============================================================

peak_summary_df = pd.DataFrame(peak_summary)

cycle_detail_df = pd.DataFrame(cycle_detail)


# ============================================================
# 8. 결과 출력
# ============================================================

print("\n" + "=" * 110)
print("[SPHC Peak 기반 사이클 요약]")
print("=" * 110)

print(peak_summary_df.round(3).to_string(index=False))


print("\n" + "=" * 110)
print("[개별 사이클 길이]")
print("=" * 110)

print(cycle_detail_df.round(3).to_string(index=False))


# ============================================================
# 9. 전체 센서별 평균
# ============================================================

overall = cycle_detail_df.groupby("sensor")["cycle_min"].agg(
    ["count", "mean", "std", "min", "max"]
)


print("\n" + "=" * 110)
print("[센서별 전체 사이클 길이]")
print("=" * 110)

print(overall.round(3))


# ============================================================
# 10. 결과 저장
# ============================================================

peak_summary_df.to_csv(
    RESULT_DIR / "sphc_peak_cycle_summary.csv", index=False, encoding="utf-8-sig"
)

cycle_detail_df.to_csv(
    RESULT_DIR / "sphc_peak_cycle_detail.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
print(FIGURE_DIR)
