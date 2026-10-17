from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"


# ============================================================
# 1. 기존 결과 로드
# ============================================================

df = pd.read_csv(RESULT_DIR / "tc_out1_cycle_maintenance_analysis.csv")


# datetime 변환
datetime_cols = ["cycle_start", "cycle_end", "cycle_mid"]

for col in datetime_cols:

    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


# ============================================================
# 2. 정기수리 근접 구간 다시 정의
# ============================================================

df["near_maintenance_3d"] = df["distance_days"] <= 3

df["near_maintenance_7d"] = df["distance_days"] <= 7


# ------------------------------------------------------------
# 7일 이상 떨어지면 BEFORE/AFTER가 아니라 FAR
# ------------------------------------------------------------

df["maintenance_window"] = "FAR"


df.loc[
    (df["distance_days"] <= 7) & (df["maintenance_relation"] == "BEFORE"),
    "maintenance_window",
] = "PRE_7D"


df.loc[
    (df["distance_days"] <= 7) & (df["maintenance_relation"] == "AFTER"),
    "maintenance_window",
] = "POST_7D"


df.loc[df["maintenance_relation"] == "DURING", "maintenance_window"] = "DURING"


# ============================================================
# 3. Boundary × Maintenance 조합
# ============================================================


def make_condition(row):

    boundary = bool(row["is_boundary_cycle"])

    near = bool(row["near_maintenance_7d"])

    if boundary and near:

        return "BOUNDARY_NEAR_MAINT"

    elif boundary and not near:

        return "BOUNDARY_FAR"

    elif (not boundary) and near:

        return "INTERNAL_NEAR_MAINT"

    else:

        return "INTERNAL_FAR"


df["condition_group"] = df.apply(make_condition, axis=1)


# ============================================================
# 4. 그룹별 이상점수 비교
# ============================================================

condition_summary = (
    df.groupby("condition_group")["anomaly_score"]
    .agg(count="count", mean="mean", median="median", max="max")
    .reset_index()
)


print("\n" + "=" * 120)
print("[Boundary × 정기수리 근접 여부]")
print("=" * 120)

print(condition_summary.round(3).to_string(index=False))


# ============================================================
# 5. 정상 기준 Pool 정의
# ============================================================
#
# LOT 내부이며
# 정기수리에서 7일 이상 떨어진 Cycle
#
# → 비교적 공정전환 영향이 적은 Reference Pool 후보
# ============================================================

reference = df[
    (df["is_boundary_cycle"] == False) & (df["near_maintenance_7d"] == False)
].copy()


print("\nReference Cycle 수:")
print(len(reference))


# ============================================================
# 6. Reference Pool 기반 탐색 임계값
# ============================================================
#
# 여기서는 모델 최적 임계값이 아니라
# 이상 후보 탐색을 위한 95 percentile 사용
# ============================================================

candidate_threshold = reference["anomaly_score"].quantile(0.95)


print("\n" + "=" * 120)
print("[Reference Pool 이상점수 기준]")
print("=" * 120)

print(f"Median : " f"{reference['anomaly_score'].median():.3f}")

print(f"95%    : " f"{candidate_threshold:.3f}")

print(f"Max    : " f"{reference['anomaly_score'].max():.3f}")


# ============================================================
# 7. 후보 유형 분류
# ============================================================


def classify_candidate(row):

    high_score = row["anomaly_score"] >= candidate_threshold

    # --------------------------------------------------------
    # 정기수리 + 경계
    # --------------------------------------------------------

    if row["is_boundary_cycle"] and row["near_maintenance_7d"]:

        return "MAINTENANCE_BOUNDARY_TRANSITION"

    # --------------------------------------------------------
    # 수리 근처의 내부 Cycle
    # --------------------------------------------------------

    elif not row["is_boundary_cycle"] and row["near_maintenance_7d"] and high_score:

        return "MAINTENANCE_STABILIZATION_CANDIDATE"

    # --------------------------------------------------------
    # 수리와 멀고 LOT 내부인데 점수가 높음
    # --------------------------------------------------------

    elif not row["is_boundary_cycle"] and not row["near_maintenance_7d"] and high_score:

        return "RESIDUAL_ANOMALY_CANDIDATE"

    # --------------------------------------------------------
    # 수리와 멀지만 경계 Cycle
    # --------------------------------------------------------

    elif row["is_boundary_cycle"] and not row["near_maintenance_7d"] and high_score:

        return "LOT_BOUNDARY_TRANSITION"

    else:

        return "NORMAL_LIKE"


df["candidate_type"] = df.apply(classify_candidate, axis=1)


# ============================================================
# 8. 후보 유형별 요약
# ============================================================

candidate_summary = (
    df.groupby("candidate_type")["anomaly_score"]
    .agg(count="count", mean="mean", median="median", max="max")
    .reset_index()
    .sort_values("max", ascending=False)
)


print("\n" + "=" * 120)
print("[후보 유형별 요약]")
print("=" * 120)

print(candidate_summary.round(3).to_string(index=False))


# ============================================================
# 9. 최종 Residual 이상 후보
# ============================================================

residual_candidates = df[
    df["candidate_type"] == "RESIDUAL_ANOMALY_CANDIDATE"
].sort_values("anomaly_score", ascending=False)


print("\n" + "=" * 120)
print("[정기수리 / LOT 경계로 설명되지 않는 이상 후보]")
print("=" * 120)


if len(residual_candidates) == 0:

    print("조건을 만족하는 후보가 없습니다.")

else:

    print(
        residual_candidates[
            [
                "LOT_NO",
                "STEEL_GRADE",
                "cycle_no",
                "cycle_start",
                "cycle_end",
                "anomaly_score",
                "level_z",
                "amplitude_z",
                "shape_z",
                "distance_days",
            ]
        ]
        .round(
            {
                "anomaly_score": 3,
                "level_z": 3,
                "amplitude_z": 3,
                "shape_z": 3,
                "distance_days": 2,
            }
        )
        .to_string(index=False)
    )


# ============================================================
# 10. 정기수리 후 안정화 후보
# ============================================================

maintenance_candidates = df[
    df["candidate_type"].isin(
        ["MAINTENANCE_BOUNDARY_TRANSITION", "MAINTENANCE_STABILIZATION_CANDIDATE"]
    )
].sort_values(["LOT_NO", "cycle_no"])


print("\n" + "=" * 120)
print("[정기수리 근처 Transition / Stabilization 후보]")
print("=" * 120)

print(
    maintenance_candidates[
        [
            "LOT_NO",
            "STEEL_GRADE",
            "cycle_no",
            "is_boundary_cycle",
            "maintenance_window",
            "distance_days",
            "anomaly_score",
            "candidate_type",
        ]
    ]
    .round({"distance_days": 2, "anomaly_score": 3})
    .to_string(index=False)
)


# ============================================================
# 11. 저장
# ============================================================

df.to_csv(
    RESULT_DIR / "tc_out1_cycle_candidate_classification.csv",
    index=False,
    encoding="utf-8-sig",
)

condition_summary.to_csv(
    RESULT_DIR / "cycle_condition_group_summary.csv", index=False, encoding="utf-8-sig"
)

residual_candidates.to_csv(
    RESULT_DIR / "residual_anomaly_candidates.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
