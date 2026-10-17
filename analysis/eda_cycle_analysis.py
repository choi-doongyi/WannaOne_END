from pathlib import Path
import sys
import numpy as np
import pandas as pd

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
# 3. SPHC 추출
# ============================================================

sphc = temp[temp["STEEL_GRADE"] == "SPHC"].copy()


target_lots = [240044, 240061, 240069, 240091]

sensors = ["TC-OUT1", "TC-OUT2"]


# ============================================================
# 4. 자기상관으로 주기 추정
# ============================================================


def estimate_period(series, min_lag=20, max_lag=80, min_overlap=20):

    series = series.dropna()

    x = series.values.astype(float)

    x = x - np.mean(x)

    std = np.std(x)

    if std == 0:
        return np.nan, np.nan

    x = x / std

    # 실제 데이터 길이를 고려한 최대 lag
    valid_max_lag = min(max_lag, len(x) - min_overlap)

    if valid_max_lag <= min_lag:
        return np.nan, np.nan

    lags = range(min_lag, valid_max_lag + 1)

    autocorr = []

    for lag in lags:

        x1 = x[:-lag]
        x2 = x[lag:]

        if len(x1) < min_overlap:
            autocorr.append(np.nan)
            continue

        corr = np.corrcoef(x1, x2)[0, 1]

        autocorr.append(corr)

    autocorr = np.array(autocorr)

    if np.all(np.isnan(autocorr)):
        return np.nan, np.nan

    best_idx = np.nanargmax(autocorr)

    best_lag = list(lags)[best_idx]

    best_corr = autocorr[best_idx]

    return best_lag, best_corr


# ============================================================
# 5. LOT별 분석
# ============================================================

results = []


for lot_no in target_lots:

    lot = sphc[sphc["LOT_NO"] == lot_no].copy().sort_values("MEAS_DT")

    # -----------------------------------------
    # 1분 단위 평균
    # -----------------------------------------

    lot_min = lot.set_index("MEAS_DT")[sensors].resample("1min").mean()

    for sensor in sensors:

        period, acf = estimate_period(lot_min[sensor], min_lag=20, max_lag=80)

        results.append(
            {
                "LOT_NO": lot_no,
                "sensor": sensor,
                "estimated_period_min": period,
                "autocorr": acf,
                "mean": lot_min[sensor].mean(),
                "std": lot_min[sensor].std(),
                "min": lot_min[sensor].min(),
                "max": lot_min[sensor].max(),
            }
        )


result_df = pd.DataFrame(results)


# ============================================================
# 6. TC-OUT1 / TC-OUT2 상관
# ============================================================

correlation_results = []


for lot_no in target_lots:

    lot = sphc[sphc["LOT_NO"] == lot_no].copy().set_index("MEAS_DT")

    lot_min = lot[["TC-OUT1", "TC-OUT2"]].resample("1min").mean()

    corr = lot_min["TC-OUT1"].corr(lot_min["TC-OUT2"])

    correlation_results.append({"LOT_NO": lot_no, "TC_OUT1_OUT2_corr": corr})


corr_df = pd.DataFrame(correlation_results)


# ============================================================
# 7. 출력
# ============================================================

print("\n" + "=" * 100)
print("[SPHC 주기 추정]")
print("=" * 100)

print(result_df.round(4).to_string(index=False))


print("\n" + "=" * 100)
print("[TC-OUT1 / TC-OUT2 상관]")
print("=" * 100)

print(corr_df.round(4).to_string(index=False))


# ============================================================
# 8. 저장
# ============================================================

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

result_df.to_csv(
    RESULT_DIR / "sphc_cycle_analysis.csv", index=False, encoding="utf-8-sig"
)

corr_df.to_csv(
    RESULT_DIR / "sphc_tc_out_correlation.csv", index=False, encoding="utf-8-sig"
)


print("\n저장 완료")
