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
# 1. 데이터 로드
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

grade_event = cr1_event[["EVT_DT", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "AFT_VAL": "STEEL_GRADE"}
)

lot_grade = lot_start.merge(grade_event, on="start_time", how="left")

temp = temp.merge(lot_grade[["LOT_NO", "STEEL_GRADE"]], on="LOT_NO", how="left")


# ============================================================
# 3. 저장 폴더
# ============================================================

RESULT_DIR = PROJECT_ROOT / "results"
FIGURE_DIR = PROJECT_ROOT / "figures" / "cycle_phase"

RESULT_DIR.mkdir(exist_ok=True)

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. 결과 저장 리스트
# ============================================================

cycle_profiles = []


# ============================================================
# 5. LOT별 Cycle 추출
# ============================================================

for lot_no in sorted(temp["LOT_NO"].unique()):

    lot = temp[temp["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    steel_grade = lot["STEEL_GRADE"].iloc[0]

    # --------------------------------------------------------
    # TC-OUT1 1분 평균
    # --------------------------------------------------------

    series = (
        lot.set_index("MEAS_DT")["TC-OUT1"]
        .resample("1min")
        .mean()
        .interpolate(method="time", limit=2)
    )

    # --------------------------------------------------------
    # 5분 이동평균
    # --------------------------------------------------------

    smooth = series.rolling(window=5, center=True, min_periods=3).mean().dropna()

    # --------------------------------------------------------
    # Peak 검출
    # --------------------------------------------------------

    peaks, _ = find_peaks(smooth.values, distance=30, prominence=5)

    peak_times = smooth.index[peaks]

    # Peak가 2개 미만이면
    # 완전한 사이클이 없음
    if len(peak_times) < 2:
        continue

    # ========================================================
    # Peak → Peak 사이클
    # ========================================================

    for cycle_no in range(len(peak_times) - 1):

        start = peak_times[cycle_no]
        end = peak_times[cycle_no + 1]

        cycle = smooth.loc[start:end].copy()

        if len(cycle) < 10:
            continue

        # ----------------------------------------------------
        # 사이클 Phase 0 ~ 100
        # ----------------------------------------------------

        original_phase = np.linspace(0, 100, len(cycle))

        # 모든 사이클을 0~100의
        # 101개 점으로 동일하게 맞춤
        target_phase = np.linspace(0, 100, 101)

        values_interp = np.interp(target_phase, original_phase, cycle.values)

        # ----------------------------------------------------
        # 사이클 평균 / 진폭
        # ----------------------------------------------------

        cycle_mean = np.mean(values_interp)

        amplitude = np.quantile(values_interp, 0.95) - np.quantile(values_interp, 0.05)

        # ----------------------------------------------------
        # 중심 제거 버전
        # ----------------------------------------------------

        centered = values_interp - cycle_mean

        # ----------------------------------------------------
        # 저장
        # ----------------------------------------------------

        for phase, raw, center in zip(target_phase, values_interp, centered):

            cycle_profiles.append(
                {
                    "LOT_NO": lot_no,
                    "STEEL_GRADE": steel_grade,
                    "cycle_no": cycle_no + 1,
                    "phase": phase,
                    "TC_OUT1": raw,
                    "centered_TC_OUT1": center,
                    "cycle_mean": cycle_mean,
                    "cycle_amplitude": amplitude,
                }
            )


# ============================================================
# 6. DataFrame
# ============================================================

profile_df = pd.DataFrame(cycle_profiles)


# ============================================================
# 7. 강종별 Cycle Template
# ============================================================

grade_template = (
    profile_df.groupby(["STEEL_GRADE", "phase"])
    .agg(
        mean_TC_OUT1=("TC_OUT1", "mean"),
        std_TC_OUT1=("TC_OUT1", "std"),
        centered_mean=("centered_TC_OUT1", "mean"),
        n_cycles=("cycle_no", "count"),
    )
    .reset_index()
)


# ============================================================
# 8. Raw Template 그래프
# ============================================================

plt.figure(figsize=(13, 7))

for grade, grade_data in grade_template.groupby("STEEL_GRADE"):

    plt.plot(grade_data["phase"], grade_data["mean_TC_OUT1"], label=grade)


plt.xlabel("Cycle Phase (%)")

plt.ylabel("TC-OUT1")

plt.title("TC-OUT1 Cycle Template by Steel Grade")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "tc_out1_cycle_template_by_grade.png", dpi=200, bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. 중심 제거 Template
# ============================================================

plt.figure(figsize=(13, 7))

for grade, grade_data in grade_template.groupby("STEEL_GRADE"):

    plt.plot(grade_data["phase"], grade_data["centered_mean"], label=grade)


plt.axhline(0, linestyle="--")

plt.xlabel("Cycle Phase (%)")

plt.ylabel("Centered TC-OUT1")

plt.title("Centered TC-OUT1 Cycle Shape by Steel Grade")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "tc_out1_centered_cycle_template.png", dpi=200, bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. 사이클별 통계
# ============================================================

cycle_summary = profile_df[
    ["LOT_NO", "STEEL_GRADE", "cycle_no", "cycle_mean", "cycle_amplitude"]
].drop_duplicates()


print("\n" + "=" * 110)
print("[사이클별 평균 / 진폭]")
print("=" * 110)

print(cycle_summary.round(3).to_string(index=False))


grade_summary = (
    cycle_summary.groupby("STEEL_GRADE")
    .agg(
        n_cycles=("cycle_no", "count"),
        mean_level=("cycle_mean", "mean"),
        std_level=("cycle_mean", "std"),
        mean_amplitude=("cycle_amplitude", "mean"),
        std_amplitude=("cycle_amplitude", "std"),
    )
    .reset_index()
)


print("\n" + "=" * 110)
print("[강종별 Cycle 수준 / 진폭]")
print("=" * 110)

print(grade_summary.round(3).to_string(index=False))


# ============================================================
# 11. 저장
# ============================================================

profile_df.to_csv(
    RESULT_DIR / "tc_out1_cycle_profiles.csv", index=False, encoding="utf-8-sig"
)

grade_template.to_csv(
    RESULT_DIR / "tc_out1_cycle_template_by_grade.csv",
    index=False,
    encoding="utf-8-sig",
)

cycle_summary.to_csv(
    RESULT_DIR / "tc_out1_cycle_summary.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
