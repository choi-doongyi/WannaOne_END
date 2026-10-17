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
# 설정
# ============================================================

TARGET_LOT = 240091
SENSOR = "TC-OUT1"

PHASE_POINTS = 101

MIN_PEAK_DISTANCE = 30
PEAK_PROMINENCE = 5

MAINTENANCE_WINDOW_DAYS = 7

PERCENTILES = [
    0.90,
    0.95,
    0.975,
    0.99,
]


# ============================================================
# 1. 데이터 불러오기
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))

maintenance = load_data(find_file("G-01_정기수리캘린더"))


temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])

event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])

maintenance["STRT_DT"] = pd.to_datetime(maintenance["STRT_DT"])

maintenance["END_DT"] = pd.to_datetime(maintenance["END_DT"])


# ============================================================
# 2. LOT ↔ 강종 연결
# ============================================================

cr1_grade_event = (
    event[(event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")]
    .copy()
    .sort_values("EVT_DT")
)


lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")


grade_event = cr1_grade_event[["EVT_DT", "BEF_VAL", "AFT_VAL"]].rename(
    columns={
        "EVT_DT": "start_time",
        "BEF_VAL": "BEF_GRADE",
        "AFT_VAL": "STEEL_GRADE",
    }
)


lot_grade = lot_start.merge(grade_event, on="start_time", how="left")


temp = temp.merge(
    lot_grade[
        [
            "LOT_NO",
            "BEF_GRADE",
            "STEEL_GRADE",
        ]
    ],
    on="LOT_NO",
    how="left",
)


# ============================================================
# 3. CR1 정기수리만 추출
# ============================================================

cr1_maintenance = maintenance[maintenance["PLANT_CD"] == "CR1"].copy()


# ============================================================
# 4. 정기수리까지 거리 계산 함수
# ============================================================


def distance_to_maintenance_days(timestamp):

    distances = []

    for _, row in cr1_maintenance.iterrows():

        start = row["STRT_DT"]
        end = row["END_DT"]

        # 정비기간 내부
        if start <= timestamp <= end:
            distance = 0.0

        # 정비 이전
        elif timestamp < start:
            distance = (start - timestamp).total_seconds() / 86400

        # 정비 이후
        else:
            distance = (timestamp - end).total_seconds() / 86400

        distances.append(distance)

    if len(distances) == 0:
        return np.nan

    return min(distances)


# ============================================================
# 5. Cycle 추출
# ============================================================

cycle_rows = []
cycle_shapes = []


for lot_no in sorted(temp["LOT_NO"].unique()):

    lot = temp[temp["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    steel_grade = lot["STEEL_GRADE"].iloc[0]

    # --------------------------------------------------------
    # 1분 평균
    # --------------------------------------------------------

    series = (
        lot.set_index("MEAS_DT")[SENSOR]
        .resample("1min")
        .mean()
        .interpolate(method="time", limit=2)
    )

    # --------------------------------------------------------
    # 5분 이동평균
    # --------------------------------------------------------

    smooth = series.rolling(window=5, center=True, min_periods=3).mean().dropna()

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peaks, _ = find_peaks(
        smooth.values,
        distance=MIN_PEAK_DISTANCE,
        prominence=PEAK_PROMINENCE,
    )

    peak_times = smooth.index[peaks]

    n_cycles = max(len(peak_times) - 1, 0)

    # --------------------------------------------------------
    # Peak → Peak Cycle
    # --------------------------------------------------------

    for i in range(n_cycles):

        start = peak_times[i]
        end = peak_times[i + 1]

        cycle = smooth.loc[start:end].dropna()

        if len(cycle) < 3:
            continue

        values = cycle.values

        # ----------------------------------------------------
        # Cycle 기본 통계
        # ----------------------------------------------------

        cycle_mean = float(np.mean(values))

        q05 = np.quantile(values, 0.05)

        q95 = np.quantile(values, 0.95)

        amplitude = float(q95 - q05)

        # ----------------------------------------------------
        # Cycle Phase 0~100%
        # ----------------------------------------------------

        original_phase = np.linspace(0, 100, len(values))

        target_phase = np.linspace(0, 100, PHASE_POINTS)

        interp_values = np.interp(target_phase, original_phase, values)

        # 평균 Level 제거
        centered_shape = interp_values - cycle_mean

        cycle_mid = start + (end - start) / 2

        distance_days = distance_to_maintenance_days(cycle_mid)

        cycle_no = i + 1

        is_first = cycle_no == 1

        is_last = cycle_no == n_cycles

        is_boundary = is_first or is_last

        cycle_rows.append(
            {
                "LOT_NO": lot_no,
                "STEEL_GRADE": steel_grade,
                "cycle_no": cycle_no,
                "cycle_start": start,
                "cycle_end": end,
                "cycle_mid": cycle_mid,
                "cycle_mean": cycle_mean,
                "cycle_amplitude": amplitude,
                "n_cycles_in_lot": n_cycles,
                "is_first_cycle": is_first,
                "is_last_cycle": is_last,
                "is_boundary_cycle": is_boundary,
                "distance_days": distance_days,
            }
        )

        cycle_shapes.append(centered_shape)


cycles = pd.DataFrame(cycle_rows)

shapes = np.vstack(cycle_shapes)


print("=" * 80)
print("LEAVE-ONE-LOT-OUT VALIDATION")
print("=" * 80)

print("\n전체 Cycle 수 :", len(cycles))


# ============================================================
# 6. Target LOT 정보
# ============================================================

target_mask = cycles["LOT_NO"].astype(str) == str(TARGET_LOT)

target_df = cycles[target_mask].copy()


if len(target_df) == 0:

    raise ValueError(f"{TARGET_LOT} LOT를 찾을 수 없습니다.")


target_grade = target_df["STEEL_GRADE"].iloc[0]


print("\nTarget LOT :", TARGET_LOT)

print("Target Grade :", target_grade)


# ============================================================
# 7. Strict Same-Grade LOLO Reference
#
# Target LOT 완전 제외
# + 동일 강종만 사용
# + Boundary 제외
# + 정기수리 ±7일 제외
# ============================================================

reference_mask = (
    (~target_mask)
    & (cycles["STEEL_GRADE"] == target_grade)
    & (cycles["is_boundary_cycle"] == False)
    & (cycles["distance_days"] > MAINTENANCE_WINDOW_DAYS)
)


reference_df = cycles[reference_mask].copy()


reference_shapes = shapes[reference_mask.to_numpy()]


print("\n" + "=" * 80)
print("STRICT SAME-GRADE LOLO REFERENCE")
print("=" * 80)

print("Reference Cycle 수 :", len(reference_df))

print("Reference LOT 수 :", reference_df["LOT_NO"].nunique())

print("Reference LOT :", sorted(reference_df["LOT_NO"].unique().tolist()))


if len(reference_df) < 5:

    raise ValueError("동일 강종 Reference Cycle이 너무 적습니다.")


# ============================================================
# 8. Target 제외 정상 기준 생성
# ============================================================

level_median = reference_df["cycle_mean"].median()

amplitude_median = reference_df["cycle_amplitude"].median()

common_template = np.median(reference_shapes, axis=0)


print("\n정상 기준")

print("Level Median :", level_median)

print("Amplitude Median :", amplitude_median)


# ============================================================
# 9. Feature 계산
# ============================================================


def calculate_features(data, data_shapes):

    result = data.copy()

    result["level_deviation"] = result["cycle_mean"] - level_median

    result["amplitude_deviation"] = result["cycle_amplitude"] - amplitude_median

    result["shape_rmse"] = np.sqrt(
        np.mean((data_shapes - common_template) ** 2, axis=1)
    )

    return result


reference_features = calculate_features(reference_df, reference_shapes)


target_shapes = shapes[target_mask.to_numpy()]


target_features = calculate_features(target_df, target_shapes)


# ============================================================
# 10. Robust Z Parameter
#
# ★ 반드시 동일 강종 Reference에서만 계산
# ============================================================

feature_cols = [
    "level_deviation",
    "amplitude_deviation",
    "shape_rmse",
]


def robust_params(series):

    values = pd.Series(series).dropna().to_numpy()

    median = np.median(values)

    mad = np.median(np.abs(values - median))

    return median, mad


params = {}


for col in feature_cols:

    median, mad = robust_params(reference_features[col])

    if mad == 0:

        raise ValueError(f"{col} MAD가 0입니다.")

    params[col] = (median, mad)


# ============================================================
# 11. Robust Z 적용
# ============================================================


def apply_score(data):

    result = data.copy()

    mapping = {
        "level_deviation": "level_z",
        "amplitude_deviation": "amplitude_z",
        "shape_rmse": "shape_z",
    }

    for col, z_col in mapping.items():

        median, mad = params[col]

        result[z_col] = 0.67448975 * (result[col] - median) / mad

    result["anomaly_score_lolo"] = (
        result["level_z"].abs() + result["amplitude_z"].abs() + result["shape_z"].abs()
    )

    return result


reference_scored = apply_score(reference_features)

target_scored = apply_score(target_features)


# ============================================================
# 12. Threshold
# ============================================================

threshold_rows = []


for percentile in PERCENTILES:

    threshold = reference_scored["anomaly_score_lolo"].quantile(percentile)

    threshold_rows.append(
        {
            "percentile": percentile,
            "threshold": threshold,
        }
    )


threshold_df = pd.DataFrame(threshold_rows)


print("\n" + "=" * 80)
print("STRICT LOLO THRESHOLDS")
print("=" * 80)

print(threshold_df.to_string(index=False))


# ============================================================
# 13. Target 평가
# ============================================================

for _, row in threshold_df.iterrows():

    col = f"above_" f"{row['percentile'] * 100:g}" f"pct"

    target_scored[col] = target_scored["anomaly_score_lolo"] >= row["threshold"]


reference_scores = reference_scored["anomaly_score_lolo"].to_numpy()


def percentile_rank(score):

    return np.mean(reference_scores <= score) * 100


target_scored["percentile_rank"] = target_scored["anomaly_score_lolo"].apply(
    percentile_rank
)


# ============================================================
# 14. 결과
# ============================================================

show_cols = [
    "LOT_NO",
    "STEEL_GRADE",
    "cycle_no",
    "cycle_mean",
    "cycle_amplitude",
    "level_z",
    "amplitude_z",
    "shape_z",
    "anomaly_score_lolo",
    "percentile_rank",
    "is_boundary_cycle",
    "above_90pct",
    "above_95pct",
    "above_97.5pct",
    "above_99pct",
]


print("\n" + "=" * 80)
print("240091 STRICT LOLO RESULT")
print("=" * 80)

print(
    target_scored[show_cols]
    .sort_values("anomaly_score_lolo", ascending=False)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("CYCLE 4~5 FOCUS")
print("=" * 80)

print(
    target_scored[target_scored["cycle_no"].isin([4, 5])][show_cols].to_string(
        index=False
    )
)
