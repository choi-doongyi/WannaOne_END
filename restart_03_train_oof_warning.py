# ============================================================
# restart_03_train_oof_warning.py
#
# STRICT RESTART - TRAIN OOF INTEGRATED WARNING
#
# 목적
# ------------------------------------------------------------
# 이미 Train-only CV로 LOCK한
#
# 24H CAUTION
# 6H DANGER
#
# OOF 예측을 통합한다.
#
# 중요:
# Validation 사용 X
# Test 사용 X
#
# 이번 단계에서는
# State Machine / Hold Time을 아직 적용하지 않는다.
#
# RAW WARNING만 만든다.
#
# ------------------------------------------------------------
# 24H LOCK
# P24H >= 0.10 -> CAUTION 후보
#
# 6H LOCK
# P6H >= 0.03 -> DANGER 후보
#
# 우선순위
# DANGER > CAUTION > NORMAL
# ============================================================


from pathlib import Path
import json

import numpy as np
import pandas as pd

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"


PATH_24H_OOF = OUTPUT_DIR / "restart_24h_train_only_oof.csv"


PATH_6H_OOF = OUTPUT_DIR / "restart_6h_train_only_oof.csv"


PATH_24H_LOCK = OUTPUT_DIR / "restart_24h_lock.json"


PATH_6H_LOCK = OUTPUT_DIR / "restart_6h_lock.json"


# ============================================================
# 1. LOCK 정보 로드
# ============================================================

with PATH_24H_LOCK.open("r", encoding="utf-8") as f:

    lock_24h = json.load(f)


with PATH_6H_LOCK.open("r", encoding="utf-8") as f:

    lock_6h = json.load(f)


threshold_24h = float(lock_24h["threshold"])


threshold_6h = float(lock_6h["threshold"])


print("=" * 80)
print("STRICT RESTART - TRAIN OOF WARNING SYSTEM")
print("=" * 80)


print()

print("24H Model:", lock_24h["model"])

print("24H Feature Set:", lock_24h["feature_set"])

print("24H Threshold:", threshold_24h)


print()

print("6H Model:", lock_6h["model"])

print("6H Feature Set:", lock_6h["feature_set"])

print("6H Threshold:", threshold_6h)


# ============================================================
# 2. OOF 로드
# ============================================================

oof_24h = pd.read_csv(PATH_24H_OOF)


oof_6h = pd.read_csv(PATH_6H_OOF)


print()

print("=" * 80)
print("OOF INPUT CHECK")
print("=" * 80)


print("24H OOF:", oof_24h.shape)

print("6H OOF:", oof_6h.shape)


# ============================================================
# 3. 시간 변환
# ============================================================

oof_24h["OBSERVED_AT"] = pd.to_datetime(
    oof_24h["OBSERVED_AT"],
    format="mixed",
    utc=True,
    errors="coerce",
)


oof_6h["OBSERVED_AT"] = pd.to_datetime(
    oof_6h["OBSERVED_AT"],
    format="mixed",
    utc=True,
    errors="coerce",
)


if oof_24h["OBSERVED_AT"].isna().any():

    raise RuntimeError("24H OBSERVED_AT 변환 실패")


if oof_6h["OBSERVED_AT"].isna().any():

    raise RuntimeError("6H OBSERVED_AT 변환 실패")


# ============================================================
# 4. 필요한 컬럼만 정리
# ============================================================

oof_24h = oof_24h[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_24H",
        "PRED_PROBA_24H",
        "PRED_24H",
    ]
].copy()


oof_6h = oof_6h[
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
        "RISK_6H",
        "PRED_PROBA_6H",
        "PRED_6H",
    ]
].copy()


# ============================================================
# 5. 24H / 6H OOF 병합
#
# 두 타깃의 censor 조건이 다르므로
# usable row 수가 다르다.
#
# 따라서 outer merge 사용.
# ============================================================

warning_df = pd.merge(
    oof_24h,
    oof_6h,
    on=[
        "SCENARIO_ID",
        "OBSERVED_AT",
    ],
    how="outer",
)


warning_df = warning_df.sort_values(
    [
        "SCENARIO_ID",
        "OBSERVED_AT",
    ]
).reset_index(drop=True)


print()

print("=" * 80)
print("MERGE CHECK")
print("=" * 80)


print("Merged rows:", len(warning_df))


print("Scenario 수:", warning_df["SCENARIO_ID"].nunique())


print("24H Probability 존재:", warning_df["PRED_PROBA_24H"].notna().sum())


print("6H Probability 존재:", warning_df["PRED_PROBA_6H"].notna().sum())


# ============================================================
# 6. Threshold 재계산
#
# 저장된 PRED_*도 있지만,
# JSON LOCK threshold를 기준으로 다시 계산한다.
# ============================================================

warning_df["CAUTION_TRIGGER"] = warning_df["PRED_PROBA_24H"] >= threshold_24h


warning_df["DANGER_TRIGGER"] = warning_df["PRED_PROBA_6H"] >= threshold_6h


# NaN 비교 결과는 False 처리
warning_df["CAUTION_TRIGGER"] = warning_df["CAUTION_TRIGGER"].fillna(False).astype(bool)


warning_df["DANGER_TRIGGER"] = warning_df["DANGER_TRIGGER"].fillna(False).astype(bool)


# ============================================================
# 7. RAW WARNING 생성
#
# 우선순위:
#
# DANGER
# >
# CAUTION
# >
# NORMAL
# ============================================================

warning_df["RAW_WARNING"] = np.select(
    [
        warning_df["DANGER_TRIGGER"],
        warning_df["CAUTION_TRIGGER"],
    ],
    [
        "DANGER",
        "CAUTION",
    ],
    default="NORMAL",
)


# ============================================================
# 8. Warning Score
#
# 시각화나 정렬용 숫자
#
# NORMAL  = 0
# CAUTION = 1
# DANGER  = 2
# ============================================================

warning_score_map = {
    "NORMAL": 0,
    "CAUTION": 1,
    "DANGER": 2,
}


warning_df["WARNING_SCORE"] = (
    warning_df["RAW_WARNING"].map(warning_score_map).astype(int)
)


# ============================================================
# 9. 기본 경보 분포
# ============================================================

warning_count = (
    warning_df["RAW_WARNING"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "CAUTION",
            "DANGER",
        ],
        fill_value=0,
    )
)


warning_ratio = warning_count / len(warning_df)


summary_df = pd.DataFrame(
    {
        "WARNING_LEVEL": warning_count.index,
        "COUNT": warning_count.values,
        "RATIO": warning_ratio.values,
    }
)


print()

print("=" * 80)
print("RAW WARNING DISTRIBUTION")
print("=" * 80)


print(summary_df.round(4).to_string(index=False))


# ============================================================
# 10. Trigger 개별 확인
# ============================================================

caution_trigger_count = int(warning_df["CAUTION_TRIGGER"].sum())


danger_trigger_count = int(warning_df["DANGER_TRIGGER"].sum())


both_trigger_count = int(
    (warning_df["CAUTION_TRIGGER"] & warning_df["DANGER_TRIGGER"]).sum()
)


danger_only_count = int(
    (~warning_df["CAUTION_TRIGGER"] & warning_df["DANGER_TRIGGER"]).sum()
)


print()

print("=" * 80)
print("TRIGGER CHECK")
print("=" * 80)


print("CAUTION Trigger:", caution_trigger_count)


print("DANGER Trigger:", danger_trigger_count)


print("CAUTION + DANGER 동시:", both_trigger_count)


print("DANGER only:", danger_only_count)


# ============================================================
# 11. Point Metric
#
# 각각 usable label이 존재하는 row에서만 계산.
# ============================================================


def point_metric(
    df,
    target_col,
    pred_col,
):

    usable = df[df[target_col].notna()].copy()

    y_true = usable[target_col].astype(int)

    y_pred = usable[pred_col].astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    fp = int(((y_true == 0) & (y_pred == 1)).sum())

    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    beta2 = 4

    f2 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if (beta2 * precision + recall) > 0
        else 0
    )

    return {
        "N": len(usable),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "PRECISION": precision,
        "RECALL": recall,
        "F1": f1,
        "F2": f2,
    }


# ============================================================
# 12. 24H / 6H Point Metric
# ============================================================

warning_df["PRED_24H_LOCK"] = warning_df["CAUTION_TRIGGER"].astype(int)


warning_df["PRED_6H_LOCK"] = warning_df["DANGER_TRIGGER"].astype(int)


metric_24h = point_metric(
    warning_df,
    target_col="RISK_24H",
    pred_col="PRED_24H_LOCK",
)


metric_6h = point_metric(
    warning_df,
    target_col="RISK_6H",
    pred_col="PRED_6H_LOCK",
)


point_metric_df = pd.DataFrame(
    [
        {
            "TARGET": "24H_CAUTION",
            "THRESHOLD": threshold_24h,
            **metric_24h,
        },
        {
            "TARGET": "6H_DANGER",
            "THRESHOLD": threshold_6h,
            **metric_6h,
        },
    ]
)


print()

print("=" * 80)
print("TRAIN OOF POINT METRICS")
print("=" * 80)


print(point_metric_df.round(4).to_string(index=False))


# ============================================================
# 13. Scenario별 경보 통계
# ============================================================

scenario_summary = warning_df.groupby("SCENARIO_ID", as_index=False).agg(
    TOTAL_POINTS=(
        "RAW_WARNING",
        "size",
    ),
    CAUTION_POINTS=(
        "CAUTION_TRIGGER",
        "sum",
    ),
    DANGER_POINTS=(
        "DANGER_TRIGGER",
        "sum",
    ),
    MAX_WARNING_SCORE=(
        "WARNING_SCORE",
        "max",
    ),
)


scenario_summary["MAX_WARNING"] = scenario_summary["MAX_WARNING_SCORE"].map(
    {
        0: "NORMAL",
        1: "CAUTION",
        2: "DANGER",
    }
)


scenario_summary["HAS_CAUTION"] = scenario_summary["CAUTION_POINTS"] > 0


scenario_summary["HAS_DANGER"] = scenario_summary["DANGER_POINTS"] > 0


print()

print("=" * 80)
print("SCENARIO WARNING SUMMARY")
print("=" * 80)


print("Scenario:", len(scenario_summary))


print(
    "CAUTION 이상 발생 Scenario:",
    int((scenario_summary["MAX_WARNING_SCORE"] >= 1).sum()),
)


print("DANGER 발생 Scenario:", int((scenario_summary["MAX_WARNING_SCORE"] >= 2).sum()))


# ============================================================
# 14. Warning Transition 계산
#
# 아직 State Machine은 아니고,
# RAW 경보가 얼마나 자주 바뀌는지만 확인.
# ============================================================

warning_df["PREV_WARNING"] = warning_df.groupby("SCENARIO_ID")["RAW_WARNING"].shift(1)


warning_df["WARNING_CHANGED"] = warning_df["PREV_WARNING"].notna() & (
    warning_df["RAW_WARNING"] != warning_df["PREV_WARNING"]
)


transition_count = int(warning_df["WARNING_CHANGED"].sum())


print()

print("=" * 80)
print("RAW WARNING TRANSITION")
print("=" * 80)


print("총 상태 변경 횟수:", transition_count)


# ============================================================
# 15. 저장
# ============================================================

warning_output_path = OUTPUT_DIR / "restart_train_oof_integrated_warning.csv"


summary_output_path = OUTPUT_DIR / "restart_train_oof_warning_summary.csv"


point_metric_output_path = OUTPUT_DIR / "restart_train_oof_point_metrics.csv"


scenario_output_path = OUTPUT_DIR / "restart_train_oof_scenario_summary.csv"


warning_df.to_csv(
    warning_output_path,
    index=False,
    encoding="utf-8-sig",
)


summary_df.to_csv(
    summary_output_path,
    index=False,
    encoding="utf-8-sig",
)


point_metric_df.to_csv(
    point_metric_output_path,
    index=False,
    encoding="utf-8-sig",
)


scenario_summary.to_csv(
    scenario_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. 완료
# ============================================================

print()

print("=" * 80)
print("TRAIN OOF RAW WARNING COMPLETE")
print("=" * 80)


print("24H Threshold:", threshold_24h)


print("6H Threshold:", threshold_6h)


print()


print("Warning file:", warning_output_path)


print("Point metric file:", point_metric_output_path)


print("Scenario summary file:", scenario_output_path)


print()

print("※ Validation 사용하지 않음")

print("※ State Machine은 아직 적용하지 않음")

print("※ 다음 단계에서 Train OOF만으로 " "경보 지속 정책을 결정")
