import csv
import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 기본 설정
# ============================================================

ROOT = Path(".")

CASTING_EQP = "P1-CC1-INS01"
SOURCE_ID = "casting"
MAINTENANCE_SOURCE_ID = "maintenance"

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 연주기 센서
# ============================================================

BASE_FEATURES = [
    "LOAD",
    "WD-SPD",
    "WD-LOD",
    "OSC-FRQ",
    "OSC-STK",
    "MLD-LVL",
    "CONTROL_DELAY",
    "RUNNING",
]


# ============================================================
# 과거 시간창
# ============================================================

WINDOWS = {
    "1H": pd.Timedelta(hours=1),
    "3H": pd.Timedelta(hours=3),
    "6H": pd.Timedelta(hours=6),
    "12H": pd.Timedelta(hours=12),
}


# ============================================================
# 시간창별 통계
# ============================================================

STATS = [
    "MEAN",
    "STD",
    "MIN",
    "MAX",
    "RANGE",
    "MEDIAN",
    "CHANGE",
]


# ============================================================
# 정비 Feature
# ============================================================

MAINTENANCE_FEATURES = [
    "MNT_TOTAL_COUNT",
    "MNT_24H_COUNT",
    "MNT_72H_COUNT",
    "MNT_LAST_GAP_H",
    "MNT_DOWN_SUM",
    "MNT_DOWN_MEAN",
    "MNT_REPLACE_COUNT",
    "MNT_COST_SUM",
    "MNT_COST_MEAN",
]


# 기존 244 + 정비 Feature 9
EXPECTED_COLS_BEFORE_SPLIT = 253

# SPLIT 추가 후
EXPECTED_COLS_AFTER_SPLIT = 254


# ============================================================
# 1. CSV 스트리밍
# ============================================================


def read_rows(path):

    with Path(path).open(encoding="utf-8-sig", newline="") as f:

        yield from csv.DictReader(f)


# ============================================================
# 2. manifest.json 읽기
# ============================================================


def load_manifest(root):

    path = Path(root) / "manifest.json"

    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 3. Scenario Split 분리
# ============================================================


def get_scenario_splits(manifest):

    train_ids = []
    validation_ids = []
    test_ids = []

    for scenario in manifest["scenarios"]:

        sid = scenario["id"]
        split = scenario["split"]

        if split == "train":

            train_ids.append(sid)

        elif split == "validation":

            validation_ids.append(sid)

        elif split == "test":

            test_ids.append(sid)

    return (
        sorted(train_ids),
        sorted(validation_ids),
        sorted(test_ids),
    )


# ============================================================
# 4. Source 데이터 스트리밍
# ============================================================


def source_rows(root, scenario_id, file_id, manifest):

    root = Path(root)

    source = next(
        (
            f
            for f in manifest["files"]
            if (f["scenarioId"] == scenario_id and f["id"] == file_id)
        ),
        None,
    )

    # 해당 Scenario에 Source가 없는 경우
    if source is None:

        return

    for chunk in source["chunks"]:

        chunk_path = root / "chunks" / chunk["sha256"]

        for index, row in enumerate(read_rows(chunk_path)):

            yield {
                **row,
                "_SOURCE_CHUNK": chunk["sha256"],
                "_SOURCE_ROW": chunk["rowStart"] + index,
            }


# ============================================================
# 5. 연주기 Label 읽기
# ============================================================


def load_labels(root, scenario_id):

    path = Path(root) / "labels" / f"{scenario_id}.csv"

    if not path.exists():

        return None

    df = pd.DataFrame(read_rows(path))

    if df.empty:

        return None

    # --------------------------------------------------------
    # CC1만
    # --------------------------------------------------------

    df = df[df["EQP_CD"] == CASTING_EQP].copy()

    if df.empty:

        return None

    # --------------------------------------------------------
    # 시간 변환
    # --------------------------------------------------------

    df["OBSERVED_AT"] = pd.to_datetime(df["OBSERVED_AT"], utc=True, errors="coerce")

    # --------------------------------------------------------
    # Label 숫자형 변환
    # --------------------------------------------------------

    numeric_cols = [
        "RISK_6H",
        "RISK_24H",
        "CENSORED_6H",
        "CENSORED_24H",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["OBSERVED_AT"])
        .sort_values("OBSERVED_AT")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# 6. 연주기 Raw 읽기
# ============================================================


def load_casting(root, scenario_id, manifest):

    rows = []

    for row in source_rows(root, scenario_id, SOURCE_ID, manifest):

        # ----------------------------------------------------
        # CC1만
        # ----------------------------------------------------

        if row.get("EQP_CD") != CASTING_EQP:

            continue

        # ----------------------------------------------------
        # 정상 측정 품질만
        # ----------------------------------------------------

        if row.get("QUALITY_FLAG") != "ok":

            continue

        rows.append(row)

    if not rows:

        return None

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 시간 변환
    # --------------------------------------------------------

    df["MEAS_DT"] = pd.to_datetime(df["MEAS_DT"], utc=True, errors="coerce")

    # --------------------------------------------------------
    # 센서 숫자형 변환
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "SCENARIO_ID",
        "MEAS_DT",
        "SLAB_NO",
    ] + BASE_FEATURES

    df = df[keep_cols].copy()

    # --------------------------------------------------------
    # 메모리 절약
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        if col == "RUNNING":

            df[col] = pd.to_numeric(df[col], downcast="integer")

        else:

            df[col] = pd.to_numeric(df[col], downcast="float")

    df = df.dropna(subset=["MEAS_DT"]).sort_values("MEAS_DT").reset_index(drop=True)

    return df


# ============================================================
# 7. 정비 Raw 읽기
# ============================================================


def load_maintenance(root, scenario_id, manifest):

    rows = []

    for row in source_rows(root, scenario_id, MAINTENANCE_SOURCE_ID, manifest):

        # ----------------------------------------------------
        # 우리가 분석하는 연주 설비만
        # ----------------------------------------------------

        if row.get("EQP_TAG") != CASTING_EQP:

            continue

        rows.append(row)

    # --------------------------------------------------------
    # 해당 Scenario에 정비 기록이 없는 경우
    # --------------------------------------------------------

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # 시간 컬럼
    # --------------------------------------------------------

    time_cols = [
        "DETC_DT",
        "FAILURE_DT",
        "STRT_DT",
        "END_DT",
    ]

    for col in time_cols:

        if col in df.columns:

            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # --------------------------------------------------------
    # 숫자 컬럼
    # --------------------------------------------------------

    numeric_cols = [
        "DOWN_MIN",
        "COST_SCORE",
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # 정비 종료 시각 기준 정렬
    # --------------------------------------------------------

    if "END_DT" in df.columns:

        df = df.sort_values("END_DT").reset_index(drop=True)

    return df


# ============================================================
# 8. 한 Window의 센서 통계 Feature 생성
# ============================================================


def summarize_window(window_df, result, prefix):

    # --------------------------------------------------------
    # 관측 개수
    # --------------------------------------------------------

    result[f"{prefix}_ROW_COUNT"] = len(window_df)

    # --------------------------------------------------------
    # 모든 센서 Feature를 우선 NaN으로 생성
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        for stat in STATS:

            result[f"{prefix}_{col}_{stat}"] = np.nan

    if window_df.empty:

        return

    # --------------------------------------------------------
    # 실제 통계 계산
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        s = window_df[col].dropna()

        if s.empty:

            continue

        result[f"{prefix}_{col}_MEAN"] = float(s.mean())

        if len(s) >= 2:

            result[f"{prefix}_{col}_STD"] = float(s.std())

        result[f"{prefix}_{col}_MIN"] = float(s.min())

        result[f"{prefix}_{col}_MAX"] = float(s.max())

        result[f"{prefix}_{col}_RANGE"] = float(s.max() - s.min())

        result[f"{prefix}_{col}_MEDIAN"] = float(s.median())

        result[f"{prefix}_{col}_CHANGE"] = float(s.iloc[-1] - s.iloc[0])


# ============================================================
# 9. OBSERVED_AT 기준 과거 정비 Feature 생성
# ============================================================


def make_maintenance_features(maintenance, observed_at):

    # --------------------------------------------------------
    # 기본값
    # --------------------------------------------------------

    result = {
        "MNT_TOTAL_COUNT": 0,
        "MNT_24H_COUNT": 0,
        "MNT_72H_COUNT": 0,
        "MNT_LAST_GAP_H": np.nan,
        "MNT_DOWN_SUM": 0.0,
        "MNT_DOWN_MEAN": np.nan,
        "MNT_REPLACE_COUNT": 0,
        "MNT_COST_SUM": 0.0,
        "MNT_COST_MEAN": np.nan,
    }

    # --------------------------------------------------------
    # 정비 기록 없음
    # --------------------------------------------------------

    if maintenance is None or maintenance.empty:

        return result

    if "END_DT" not in maintenance.columns:

        return result

    # ========================================================
    # 매우 중요
    #
    # OBSERVED_AT 시점에서
    # 이미 완료된 정비만 사용
    #
    # 미래 정비 데이터 사용 금지
    # ========================================================

    past = maintenance[
        maintenance["END_DT"].notna() & (maintenance["END_DT"] <= observed_at)
    ].copy()

    if past.empty:

        return result

    # --------------------------------------------------------
    # 누적 완료 정비 횟수
    # --------------------------------------------------------

    result["MNT_TOTAL_COUNT"] = int(len(past))

    # --------------------------------------------------------
    # 마지막 완료 정비 후 경과시간
    # --------------------------------------------------------

    last_end_dt = past["END_DT"].max()

    result["MNT_LAST_GAP_H"] = float((observed_at - last_end_dt).total_seconds() / 3600)

    # --------------------------------------------------------
    # 최근 24시간 완료 정비 횟수
    # --------------------------------------------------------

    start_24h = observed_at - pd.Timedelta(hours=24)

    recent_24h = past[past["END_DT"] >= start_24h]

    result["MNT_24H_COUNT"] = int(len(recent_24h))

    # --------------------------------------------------------
    # 최근 72시간 완료 정비 횟수
    # --------------------------------------------------------

    start_72h = observed_at - pd.Timedelta(hours=72)

    recent_72h = past[past["END_DT"] >= start_72h]

    result["MNT_72H_COUNT"] = int(len(recent_72h))

    # --------------------------------------------------------
    # 완료된 과거 정비 DOWN_MIN
    # --------------------------------------------------------

    if "DOWN_MIN" in past.columns:

        down = past["DOWN_MIN"].dropna()

        if not down.empty:

            result["MNT_DOWN_SUM"] = float(down.sum())

            result["MNT_DOWN_MEAN"] = float(down.mean())

    # --------------------------------------------------------
    # replace 정비 횟수
    # --------------------------------------------------------

    if "MNT_TYPE" in past.columns:

        mnt_type = past["MNT_TYPE"].fillna("").astype(str).str.lower()

        result["MNT_REPLACE_COUNT"] = int((mnt_type == "replace").sum())

    # --------------------------------------------------------
    # 완료된 과거 정비 비용
    # --------------------------------------------------------

    if "COST_SCORE" in past.columns:

        cost = past["COST_SCORE"].dropna()

        if not cost.empty:

            result["MNT_COST_SUM"] = float(cost.sum())

            result["MNT_COST_MEAN"] = float(cost.mean())

    # ========================================================
    # FAILURE_DT는 절대 Feature로 넣지 않음
    # ========================================================

    return result


# ============================================================
# 10. OBSERVED_AT별 Feature 생성
# ============================================================


def make_features(raw, labels, maintenance, scenario_id):

    results = []

    raw = raw.sort_values("MEAS_DT").reset_index(drop=True)

    for _, label in labels.iterrows():

        observed_at = label["OBSERVED_AT"]

        # ====================================================
        # 예측 시점 이전/현재 센서만 사용
        # ====================================================

        past = raw[raw["MEAS_DT"] <= observed_at]

        if past.empty:

            continue

        current = past.iloc[-1]

        # ====================================================
        # 기본 정보
        # ====================================================

        row = {
            "SCENARIO_ID": scenario_id,
            "OBSERVED_AT": observed_at,
            "LAST_MEAS_DT": current["MEAS_DT"],
            "LAST_MEAS_GAP_MIN": (observed_at - current["MEAS_DT"]).total_seconds()
            / 60,
            "RISK_6H": label["RISK_6H"],
            "RISK_24H": label["RISK_24H"],
            "CENSORED_6H": label["CENSORED_6H"],
            "CENSORED_24H": label["CENSORED_24H"],
        }

        # ====================================================
        # 현재 센서값
        # ====================================================

        for col in BASE_FEATURES:

            row[f"CUR_{col}"] = current[col]

        # ====================================================
        # 센서 과거 Window
        # ====================================================

        for window_name, window_size in WINDOWS.items():

            start_time = observed_at - window_size

            window_df = past[past["MEAS_DT"] >= start_time]

            summarize_window(window_df, row, window_name)

        # ====================================================
        # 완료된 과거 정비 Feature
        # ====================================================

        maintenance_features = make_maintenance_features(maintenance, observed_at)

        row.update(maintenance_features)

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 11. Scenario Feature 검증
# ============================================================


def validate_features(features, scenario_id):

    if features.empty:

        raise RuntimeError(f"{scenario_id}: " f"Feature Dataset이 비어 있습니다.")

    # --------------------------------------------------------
    # 센서 핵심 Feature
    # --------------------------------------------------------

    required_cols = [
        "CUR_LOAD",
        "1H_LOAD_MEAN",
        "3H_LOAD_MEAN",
        "6H_LOAD_MEAN",
        "12H_LOAD_MEAN",
        "1H_WD-LOD_STD",
        "6H_CONTROL_DELAY_MAX",
        "12H_MLD-LVL_CHANGE",
    ] + MAINTENANCE_FEATURES

    missing = [col for col in required_cols if col not in features.columns]

    if missing:

        raise RuntimeError(f"{scenario_id}: " f"Feature 누락\n" f"{missing}")

    # --------------------------------------------------------
    # 컬럼 수
    # --------------------------------------------------------

    if len(features.columns) != EXPECTED_COLS_BEFORE_SPLIT:

        raise RuntimeError(
            f"{scenario_id}: "
            f"Feature 컬럼 수 이상 "
            f"({len(features.columns)}개, "
            f"예상 "
            f"{EXPECTED_COLS_BEFORE_SPLIT}개)"
        )

    # --------------------------------------------------------
    # Window ROW_COUNT 확인
    # --------------------------------------------------------

    row_count_cols = [
        "1H_ROW_COUNT",
        "3H_ROW_COUNT",
        "6H_ROW_COUNT",
        "12H_ROW_COUNT",
    ]

    max_count = features[row_count_cols].max().max()

    if max_count <= 0:

        raise RuntimeError(f"{scenario_id}: " f"모든 Window ROW_COUNT가 0입니다.")

    # --------------------------------------------------------
    # 정비 Feature에서 음수 금지
    # --------------------------------------------------------

    count_cols = [
        "MNT_TOTAL_COUNT",
        "MNT_24H_COUNT",
        "MNT_72H_COUNT",
        "MNT_REPLACE_COUNT",
    ]

    for col in count_cols:

        if features[col].fillna(0).lt(0).any():

            raise RuntimeError(f"{scenario_id}: " f"{col}에 음수 존재")

    # --------------------------------------------------------
    # 마지막 정비 경과시간 음수 금지
    # --------------------------------------------------------

    gap = features["MNT_LAST_GAP_H"].dropna()

    if (gap < 0).any():

        raise RuntimeError(f"{scenario_id}: " f"MNT_LAST_GAP_H 음수 발생")


# ============================================================
# 12. Scenario 하나 처리
# ============================================================


def process_scenario(root, scenario_id, manifest):

    labels = load_labels(root, scenario_id)

    if labels is None:

        return None

    raw = load_casting(root, scenario_id, manifest)

    if raw is None:

        return None

    maintenance = load_maintenance(root, scenario_id, manifest)

    features = make_features(raw, labels, maintenance, scenario_id)

    validate_features(features, scenario_id)

    del raw
    del labels
    del maintenance

    gc.collect()

    return features


# ============================================================
# 13. S0001 사전 테스트
# ============================================================


def test_single_scenario(root, manifest):

    print()

    print("=" * 80)

    print("S0001 PRE-CHECK V2")

    print("=" * 80)

    features = process_scenario(root, "S0001", manifest)

    print("S0001 shape:", features.shape)

    print("S0001 컬럼 수:", len(features.columns))

    # --------------------------------------------------------
    # Window 검사
    # --------------------------------------------------------

    cols = [
        "OBSERVED_AT",
        "1H_ROW_COUNT",
        "3H_ROW_COUNT",
        "6H_ROW_COUNT",
        "12H_ROW_COUNT",
    ]

    print()

    print(features[cols].to_string(index=False))

    max_counts = features[
        [
            "1H_ROW_COUNT",
            "3H_ROW_COUNT",
            "6H_ROW_COUNT",
            "12H_ROW_COUNT",
        ]
    ].max()

    print()

    print("S0001 최대 Window Row Count")

    print(max_counts)

    if (
        max_counts["1H_ROW_COUNT"] < 500
        or max_counts["3H_ROW_COUNT"] < 1500
        or max_counts["6H_ROW_COUNT"] < 3000
        or max_counts["12H_ROW_COUNT"] < 6000
    ):

        raise RuntimeError(
            "S0001 Window 추출이 비정상입니다. " "전체 Dataset 생성을 중단합니다."
        )

    # --------------------------------------------------------
    # 정비 Feature 검사
    # --------------------------------------------------------

    print()

    print("S0001 정비 Feature")

    print(
        features[
            [
                "OBSERVED_AT",
                "MNT_TOTAL_COUNT",
                "MNT_24H_COUNT",
                "MNT_72H_COUNT",
                "MNT_LAST_GAP_H",
                "MNT_DOWN_SUM",
                "MNT_DOWN_MEAN",
                "MNT_REPLACE_COUNT",
                "MNT_COST_SUM",
                "MNT_COST_MEAN",
            ]
        ].to_string(index=False)
    )

    print()

    print("S0001 사전 검증 성공!")


# ============================================================
# 14. Split 전체 처리
# ============================================================


def process_split(root, scenario_ids, split_name, manifest):

    all_features = []

    failed = []

    success_count = 0

    total = len(scenario_ids)

    print()

    print("=" * 80)

    print(f"{split_name.upper()} " f"FEATURE V2 생성 시작")

    print("=" * 80)

    for idx, scenario_id in enumerate(scenario_ids, start=1):

        try:

            features = process_scenario(root, scenario_id, manifest)

            if features is not None and not features.empty:

                features["SPLIT"] = split_name

                all_features.append(features)

                success_count += 1

            else:

                failed.append(scenario_id)

                print(f"[WARNING] " f"{scenario_id}: " f"Feature 없음")

        except Exception as e:

            failed.append(scenario_id)

            print()

            print(f"[ERROR] " f"{scenario_id}")

            print(type(e).__name__, ":", e)

        # ----------------------------------------------------
        # 진행률
        # ----------------------------------------------------

        if idx % 10 == 0 or idx == total:

            print(
                f"[{split_name}] "
                f"처리 "
                f"{idx}/{total}"
                f" | 성공 "
                f"{success_count}"
                f" | 실패 "
                f"{len(failed)}"
            )

        gc.collect()

    if not all_features:

        return (pd.DataFrame(), failed)

    result = pd.concat(all_features, ignore_index=True)

    return (result, failed)


# ============================================================
# 15. 전체 Dataset 검사
# ============================================================


def check_dataset(df, name):

    print()

    print("=" * 80)

    print(f"{name} DATASET CHECK V2")

    print("=" * 80)

    print("shape:", df.shape)

    print("scenario 수:", df["SCENARIO_ID"].nunique())

    print("전체 컬럼 수:", len(df.columns))

    # --------------------------------------------------------
    # Label 확인
    # --------------------------------------------------------

    print()

    print("RISK_24H")

    print(df["RISK_24H"].value_counts(dropna=False))

    print()

    print("RISK_6H")

    print(df["RISK_6H"].value_counts(dropna=False))

    # --------------------------------------------------------
    # 센서 Feature
    # --------------------------------------------------------

    print()

    print("시계열 Feature 존재 확인")

    check_cols = [
        "1H_LOAD_MEAN",
        "3H_LOAD_MEAN",
        "6H_LOAD_MEAN",
        "12H_LOAD_MEAN",
        "1H_WD-LOD_STD",
        "6H_CONTROL_DELAY_MAX",
        "12H_MLD-LVL_CHANGE",
    ]

    for col in check_cols:

        print(col, "→", col in df.columns)

    # --------------------------------------------------------
    # 정비 Feature
    # --------------------------------------------------------

    print()

    print("정비 Feature 존재 확인")

    for col in MAINTENANCE_FEATURES:

        print(col, "→", col in df.columns)

    # --------------------------------------------------------
    # 정비 Feature 요약
    # --------------------------------------------------------

    print()

    print("정비 Feature 요약")

    print(df[MAINTENANCE_FEATURES].describe().round(3))

    # --------------------------------------------------------
    # 정비 기록이 반영된 행 수
    # --------------------------------------------------------

    print()

    used_mnt_rows = int((df["MNT_TOTAL_COUNT"] > 0).sum())

    print(
        "과거 완료 정비가 존재하는 예측행:",
        used_mnt_rows,
        "/",
        len(df),
        f"({used_mnt_rows / len(df):.2%})",
    )

    # --------------------------------------------------------
    # Window ROW COUNT
    # --------------------------------------------------------

    print()

    print("Window ROW_COUNT 요약")

    row_count_cols = [
        "1H_ROW_COUNT",
        "3H_ROW_COUNT",
        "6H_ROW_COUNT",
        "12H_ROW_COUNT",
    ]

    summary = df[row_count_cols].describe().round(2)

    print(summary)

    if df[row_count_cols].max().max() <= 0:

        raise RuntimeError(f"{name}: " f"Window ROW_COUNT가 모두 0입니다.")

    # --------------------------------------------------------
    # 컬럼 수
    # --------------------------------------------------------

    if len(df.columns) != EXPECTED_COLS_AFTER_SPLIT:

        raise RuntimeError(
            f"{name}: "
            f"컬럼 수 "
            f"{len(df.columns)}개, "
            f"예상 "
            f"{EXPECTED_COLS_AFTER_SPLIT}개"
        )


# ============================================================
# 16. 안전 저장
# ============================================================


def safe_save(df, final_path):

    temp_path = final_path.parent / (final_path.stem + "_TEMP.csv")

    # --------------------------------------------------------
    # TEMP 저장
    # --------------------------------------------------------

    df.to_csv(temp_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 저장 후 다시 확인
    # --------------------------------------------------------

    check = pd.read_csv(temp_path, nrows=10)

    if len(check.columns) != EXPECTED_COLS_AFTER_SPLIT:

        raise RuntimeError(
            f"{final_path.name}: "
            f"저장 후 컬럼 수 "
            f"{len(check.columns)}개, "
            f"예상 "
            f"{EXPECTED_COLS_AFTER_SPLIT}개"
        )

    required_cols = [
        "1H_LOAD_MEAN",
        "3H_LOAD_MEAN",
        "6H_LOAD_MEAN",
        "12H_LOAD_MEAN",
    ] + MAINTENANCE_FEATURES

    for col in required_cols:

        if col not in check.columns:

            raise RuntimeError(f"{final_path.name}: " f"{col} 누락")

    # --------------------------------------------------------
    # 기존 파일 삭제
    # --------------------------------------------------------

    if final_path.exists():

        final_path.unlink()

    # --------------------------------------------------------
    # TEMP → 최종 파일
    # --------------------------------------------------------

    temp_path.rename(final_path)


# ============================================================
# 17. MAIN
# ============================================================


def main():

    print("=" * 80)

    print("CASTING FEATURE DATASET BUILDER V4")

    print("SENSOR + MAINTENANCE")

    print("=" * 80)

    manifest = load_manifest(ROOT)

    train_ids, validation_ids, test_ids = get_scenario_splits(manifest)

    print()

    print("Train scenarios:", len(train_ids))

    print("Validation scenarios:", len(validation_ids))

    print("Test scenarios:", len(test_ids))

    # --------------------------------------------------------
    # Split 검사
    # --------------------------------------------------------

    if len(train_ids) != 216 or len(validation_ids) != 72 or len(test_ids) != 72:

        raise RuntimeError("Scenario Split 개수가 " "예상과 다릅니다.")

    # ========================================================
    # S0001 먼저 검증
    # ========================================================

    test_single_scenario(ROOT, manifest)

    # ========================================================
    # Train
    # ========================================================

    train_df, train_failed = process_split(ROOT, train_ids, "train", manifest)

    # ========================================================
    # Validation
    # ========================================================

    validation_df, validation_failed = process_split(
        ROOT, validation_ids, "validation", manifest
    )

    # --------------------------------------------------------
    # 실패 검사
    # --------------------------------------------------------

    if train_failed or validation_failed:

        print()

        print("=" * 80)

        print("FEATURE 생성 실패")

        print("=" * 80)

        print("Train 실패:", train_failed)

        print("Validation 실패:", validation_failed)

        raise RuntimeError("실패 Scenario가 있으므로 " "최종 CSV를 저장하지 않습니다.")

    # ========================================================
    # Dataset 점검
    # ========================================================

    check_dataset(train_df, "TRAIN")

    check_dataset(validation_df, "VALIDATION")

    # ========================================================
    # 저장
    # ========================================================

    train_path = OUTPUT_DIR / "casting_train_features_v2.csv"

    validation_path = OUTPUT_DIR / "casting_validation_features_v2.csv"

    safe_save(train_df, train_path)

    safe_save(validation_df, validation_path)

    # ========================================================
    # 완료
    # ========================================================

    print()

    print("=" * 80)

    print("Sensor + Maintenance " "Feature 생성 및 검증 성공!")

    print("=" * 80)

    print()

    print("Train:", train_df.shape)

    print("Validation:", validation_df.shape)

    print("Train 컬럼:", len(train_df.columns))

    print("Validation 컬럼:", len(validation_df.columns))

    print()

    print("저장:", train_path)

    print("저장:", validation_path)


# ============================================================
# 18. 실행
# ============================================================

if __name__ == "__main__":

    main()
