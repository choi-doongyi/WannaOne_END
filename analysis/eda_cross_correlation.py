from pathlib import Path
import sys
import numpy as np
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
# 3. SPHC 데이터
# ============================================================

sphc = temp[temp["STEEL_GRADE"] == "SPHC"].copy()


target_lots = [240044, 240061, 240069, 240091]


# ============================================================
# 4. Cross-correlation 함수
# ============================================================


def cross_correlation(x, y, max_lag=20):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    results = []

    for lag in range(-max_lag, max_lag + 1):

        # ----------------------------------------------------
        # lag > 0
        # OUT2가 OUT1보다 lag분 늦게 나타난다고 가정
        # ----------------------------------------------------

        if lag > 0:

            x_part = x[:-lag]
            y_part = y[lag:]

        # ----------------------------------------------------
        # lag < 0
        # OUT1이 OUT2보다 |lag|분 늦게 나타남
        # ----------------------------------------------------

        elif lag < 0:

            x_part = x[-lag:]
            y_part = y[:lag]

        # ----------------------------------------------------
        # lag = 0
        # 동일 시점
        # ----------------------------------------------------

        else:

            x_part = x
            y_part = y

        valid = ~np.isnan(x_part) & ~np.isnan(y_part)

        x_valid = x_part[valid]
        y_valid = y_part[valid]

        if len(x_valid) < 20:

            corr = np.nan

        else:

            corr = np.corrcoef(x_valid, y_valid)[0, 1]

        results.append({"lag_min": lag, "correlation": corr})

    result = pd.DataFrame(results)

    best_idx = result["correlation"].idxmax()

    best_lag = result.loc[best_idx, "lag_min"]

    best_corr = result.loc[best_idx, "correlation"]

    return result, best_lag, best_corr


# ============================================================
# 5. LOT별 Cross-correlation
# ============================================================

summary = []


FIGURE_DIR = PROJECT_ROOT / "figures" / "cross_correlation"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


for lot_no in target_lots:

    lot = sphc[sphc["LOT_NO"] == lot_no].copy().set_index("MEAS_DT")

    # --------------------------------------------------------
    # 1분 단위 평균
    # --------------------------------------------------------

    lot_min = lot[["TC-OUT1", "TC-OUT2"]].resample("1min").mean()

    # --------------------------------------------------------
    # 동시점 상관
    # --------------------------------------------------------

    zero_corr = lot_min["TC-OUT1"].corr(lot_min["TC-OUT2"])

    # --------------------------------------------------------
    # Cross-correlation
    # --------------------------------------------------------

    corr_df, best_lag, best_corr = cross_correlation(
        lot_min["TC-OUT1"].values, lot_min["TC-OUT2"].values, max_lag=20
    )

    summary.append(
        {
            "LOT_NO": lot_no,
            "zero_lag_corr": zero_corr,
            "best_lag_min": best_lag,
            "best_corr": best_corr,
        }
    )

    # ========================================================
    # 그래프
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(corr_df["lag_min"], corr_df["correlation"], marker="o")

    plt.axvline(best_lag, linestyle="--")

    plt.xlabel("Lag (min)")

    plt.ylabel("Correlation")

    plt.title(f"SPHC LOT {lot_no} " f"TC-OUT1 vs TC-OUT2 Cross-Correlation")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{lot_no}_tc_out_crosscorr.png", dpi=200, bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 6. 결과
# ============================================================

summary_df = pd.DataFrame(summary)


print("\n" + "=" * 100)
print("[TC-OUT1 / TC-OUT2 Cross-Correlation]")
print("=" * 100)

print(summary_df.round(4).to_string(index=False))


# ============================================================
# 7. 저장
# ============================================================

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)

summary_df.to_csv(
    RESULT_DIR / "sphc_cross_correlation.csv", index=False, encoding="utf-8-sig"
)


print("\n저장 완료")
