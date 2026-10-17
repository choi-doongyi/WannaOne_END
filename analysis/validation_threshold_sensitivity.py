from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_PATH = RESULTS_DIR / "tc_out1_cycle_maintenance_analysis.csv"


# ============================================================
# 1. 데이터 불러오기
# ============================================================

df = pd.read_csv(INPUT_PATH)


print("=" * 70)
print("THRESHOLD SENSITIVITY VALIDATION")
print("=" * 70)

print("\n데이터 크기 :", df.shape)

print("\n컬럼")
print(df.columns.tolist())


# ============================================================
# 2. 필요한 컬럼 확인
# ============================================================

required_columns = [
    "LOT_NO",
    "STEEL_GRADE",
    "cycle_no",
    "anomaly_score",
    "is_boundary_cycle",
    "distance_days",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"필요한 컬럼이 없습니다: {missing_columns}")


# ============================================================
# 3. 데이터 타입 정리
# ============================================================

df["is_boundary_cycle"] = (
    df["is_boundary_cycle"]
    .astype(str)
    .str.lower()
    .map({"true": True, "false": False})
    .fillna(df["is_boundary_cycle"])
)


df["distance_days"] = pd.to_numeric(df["distance_days"], errors="coerce")

df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")


# ============================================================
# 4. Reference Pool 정의
#
# INTERNAL_FAR
# = LOT 경계가 아니고
# + 정기수리에서 7일 이상 떨어진 Cycle
# ============================================================

df["abs_distance_days"] = df["distance_days"].abs()

reference_mask = (df["is_boundary_cycle"] == False) & (df["abs_distance_days"] > 7)

reference_df = df[reference_mask].copy()

print("\n" + "=" * 70)
print("REFERENCE POOL")
print("=" * 70)

print("INTERNAL_FAR Cycle 수 :", len(reference_df))

print("\nAnomaly Score 요약")
print(reference_df["anomaly_score"].describe())


# ============================================================
# 5. Threshold 계산
# ============================================================

percentiles = [
    0.90,
    0.95,
    0.975,
    0.99,
]

threshold_rows = []


for percentile in percentiles:

    threshold = reference_df["anomaly_score"].quantile(percentile)

    candidates = (
        reference_df[reference_df["anomaly_score"] >= threshold]
        .copy()
        .sort_values("anomaly_score", ascending=False)
    )

    threshold_rows.append(
        {
            "percentile": percentile,
            "percentile_label": (f"{percentile * 100:g}%"),
            "threshold": threshold,
            "n_candidates": len(candidates),
        }
    )


threshold_df = pd.DataFrame(threshold_rows)


print("\n" + "=" * 70)
print("THRESHOLD SUMMARY")
print("=" * 70)

print(threshold_df.to_string(index=False))


# ============================================================
# 6. Threshold별 후보 출력
# ============================================================

candidate_results = []


for percentile in percentiles:

    threshold = reference_df["anomaly_score"].quantile(percentile)

    candidates = (
        reference_df[reference_df["anomaly_score"] >= threshold]
        .copy()
        .sort_values("anomaly_score", ascending=False)
    )

    candidates["percentile"] = percentile
    candidates["threshold"] = threshold

    candidate_results.append(candidates)

    print("\n" + "-" * 70)

    print(f"{percentile * 100:g}% Threshold")

    print(f"기준값 : {threshold:.4f}")

    print(f"후보 수 : {len(candidates)}")

    show_cols = [
        "LOT_NO",
        "STEEL_GRADE",
        "cycle_no",
        "anomaly_score",
        "abs_distance_days",
    ]

    print(candidates[show_cols].to_string(index=False))


all_candidates = pd.concat(candidate_results, ignore_index=True)


# ============================================================
# 7. 후보 안정성 계산
#
# 몇 개 Threshold에서 계속 살아남았는지 확인
# ============================================================

stability = all_candidates.groupby(
    ["LOT_NO", "STEEL_GRADE", "cycle_no"], as_index=False
).agg(
    anomaly_score=("anomaly_score", "first"),
    survived_thresholds=("percentile", "nunique"),
    max_percentile=("percentile", "max"),
)


stability["survival_rate"] = stability["survived_thresholds"] / len(percentiles)


stability = stability.sort_values(
    [
        "survived_thresholds",
        "anomaly_score",
    ],
    ascending=[
        False,
        False,
    ],
)


print("\n" + "=" * 70)
print("CANDIDATE STABILITY")
print("=" * 70)

print(stability.to_string(index=False))


# ============================================================
# 8. 기존 핵심 후보 240091 확인
# ============================================================

target = stability[stability["LOT_NO"].astype(str) == "240091"].copy()


print("\n" + "=" * 70)
print("240091 VALIDATION")
print("=" * 70)

if len(target) == 0:

    print("240091 후보가 Threshold 후보군에 없습니다.")

else:

    print(target.to_string(index=False))


# ============================================================
# 9. 결과 저장
# ============================================================

threshold_save_path = RESULTS_DIR / "validation_threshold_sensitivity.csv"

stability_save_path = RESULTS_DIR / "validation_threshold_candidate_stability.csv"


threshold_df.to_csv(threshold_save_path, index=False, encoding="utf-8-sig")

stability.to_csv(stability_save_path, index=False, encoding="utf-8-sig")


print("\n" + "=" * 70)
print("SAVE")
print("=" * 70)

print("Threshold 결과 :", threshold_save_path)

print("후보 안정성 결과 :", stability_save_path)
