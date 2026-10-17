from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 0. 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. 데이터 로드
# ============================================================

profile = pd.read_csv(RESULT_DIR / "tc_out1_cycle_profiles.csv")

cycle = pd.read_csv(RESULT_DIR / "tc_out1_cycle_summary.csv")


# ============================================================
# 2. 공통 Centered Cycle Template
# ============================================================

global_template = profile.groupby("phase")["centered_TC_OUT1"].median()


# ============================================================
# 3. 강종별 기준 Level / Amplitude
# ============================================================

grade_baseline = (
    cycle.groupby("STEEL_GRADE")
    .agg(
        level_median=("cycle_mean", "median"),
        amplitude_median=("cycle_amplitude", "median"),
    )
    .reset_index()
)


cycle = cycle.merge(grade_baseline, on="STEEL_GRADE", how="left")


# ============================================================
# 4. Level / Amplitude 편차
# ============================================================

cycle["level_deviation"] = cycle["cycle_mean"] - cycle["level_median"]

cycle["amplitude_deviation"] = cycle["cycle_amplitude"] - cycle["amplitude_median"]


# ============================================================
# 5. Cycle Shape RMSE
# ============================================================

shape_results = []


for (lot_no, grade, cycle_no), group in profile.groupby(
    ["LOT_NO", "STEEL_GRADE", "cycle_no"]
):

    group = group.sort_values("phase")

    template = global_template.reindex(group["phase"].values).values

    actual = group["centered_TC_OUT1"].values

    rmse = np.sqrt(np.mean((actual - template) ** 2))

    shape_results.append(
        {
            "LOT_NO": lot_no,
            "STEEL_GRADE": grade,
            "cycle_no": cycle_no,
            "shape_rmse": rmse,
        }
    )


shape_df = pd.DataFrame(shape_results)


cycle = cycle.merge(shape_df, on=["LOT_NO", "STEEL_GRADE", "cycle_no"], how="left")


# ============================================================
# 6. LOT 내부 Cycle 위치
# ============================================================

cycle["n_cycles_in_lot"] = cycle.groupby("LOT_NO")["cycle_no"].transform("max")


cycle["is_first_cycle"] = cycle["cycle_no"] == 1


cycle["is_last_cycle"] = cycle["cycle_no"] == cycle["n_cycles_in_lot"]


cycle["is_boundary_cycle"] = cycle["is_first_cycle"] | cycle["is_last_cycle"]


# ============================================================
# 7. Robust Z-score 함수
# ============================================================


def robust_z(series):

    median = series.median()

    mad = (series - median).abs().median()

    if mad == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return 0.6745 * (series - median) / mad


# ============================================================
# 8. 편차 표준화
# ============================================================

cycle["level_z"] = cycle.groupby("STEEL_GRADE")["cycle_mean"].transform(robust_z)


cycle["amplitude_z"] = cycle.groupby("STEEL_GRADE")["cycle_amplitude"].transform(
    robust_z
)


cycle["shape_z"] = robust_z(cycle["shape_rmse"])


# ============================================================
# 9. 종합 이상 점수
# ============================================================

cycle["anomaly_score"] = (
    cycle["level_z"].abs() + cycle["amplitude_z"].abs() + cycle["shape_z"].abs()
)


# ============================================================
# 10. 우선순위
# ============================================================

cycle = cycle.sort_values("anomaly_score", ascending=False)


print("\n" + "=" * 120)
print("[Cycle 이상 후보]")
print("=" * 120)

print(
    cycle[
        [
            "LOT_NO",
            "STEEL_GRADE",
            "cycle_no",
            "is_boundary_cycle",
            "cycle_mean",
            "cycle_amplitude",
            "shape_rmse",
            "level_z",
            "amplitude_z",
            "shape_z",
            "anomaly_score",
        ]
    ]
    .head(30)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 11. 경계 / 내부 비교
# ============================================================

print("\n" + "=" * 120)
print("[경계 Cycle 평균 이상점수]")
print("=" * 120)

print(
    cycle.groupby("is_boundary_cycle")["anomaly_score"]
    .agg(["count", "mean", "median", "max"])
    .round(3)
)


# ============================================================
# 12. 저장
# ============================================================

cycle.to_csv(
    RESULT_DIR / "tc_out1_cycle_anomaly_score.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
