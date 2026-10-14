import csv
import json
import gc
import re
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
# 센서 과거 시간창
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
    # CC1만 사용
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
# 6. 연주기 Sensor Raw 읽기
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
        # 정상 품질 데이터만
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
    # 센서 숫자 변환
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
        # 분석 대상 연주 설비만
        # ----------------------------------------------------

        if row.get("EQP_TAG") != CASTING_EQP:

            continue

        rows.append(row)

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
    # 정비 종료 시각 순으로 정렬
    # --------------------------------------------------------

    if "END_DT" in df.columns:

        df = df.sort_values("END_DT").reset_index(drop=True)

    return df


# ============================================================
# 8. Feature 이름용 문자열 정리
# ============================================================


def clean_feature_name(value):

    value = str(value).strip().upper()

    value = re.sub(r"[^A-Z0-9]+", "_", value)

    value = value.strip("_")

    if not value:

        value = "UNKNOWN"

    return value


# ============================================================
# 9. Train 데이터에서만 정비 카테고리 탐색
#
# Validation 정보로 카테고리를 만들지 않음
# ============================================================


def discover_train_maintenance_categories(root, train_ids, manifest):

    parts = set()
    mnt_types = set()

    print()

    print("=" * 80)

    print("TRAIN MAINTENANCE CATEGORY DISCOVERY")

    print("=" * 80)

    for idx, scenario_id in enumerate(train_ids, start=1):

        maintenance = load_maintenance(root, scenario_id, manifest)

        if not maintenance.empty:

            # ------------------------------------------------
            # PART_CD
            # ------------------------------------------------

            if "PART_CD" in maintenance.columns:

                values = maintenance["PART_CD"].dropna().astype(str).str.strip()

                for value in values:

                    if value:

                        parts.add(value)

            # ------------------------------------------------
            # MNT_TYPE
            # ------------------------------------------------

            if "MNT_TYPE" in maintenance.columns:

                values = maintenance["MNT_TYPE"].dropna().astype(str).str.strip()

                for value in values:

                    if value:

                        mnt_types.add(value)

        if idx % 50 == 0 or idx == len(train_ids):

            print(f"처리 " f"{idx}/{len(train_ids)}")

    parts = sorted(parts)

    mnt_types = sorted(mnt_types)

    print()

    print("Train PART_CD 종류:", len(parts))

    for part in parts:

        print(" -", part)

    print()

    print("Train MNT_TYPE 종류:", len(mnt_types))

    for mnt_type in mnt_types:

        print(" -", mnt_type)

    return (parts, mnt_types)


# ============================================================
# 10. 한 Sensor Window의 통계 생성
# ============================================================


def summarize_window(window_df, result, prefix):

    # --------------------------------------------------------
    # 관측 행 수
    # --------------------------------------------------------

    result[f"{prefix}_ROW_COUNT"] = len(window_df)

    # --------------------------------------------------------
    # 모든 통계 컬럼 먼저 생성
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        for stat in STATS:

            result[f"{prefix}_{col}_{stat}"] = np.nan

    if window_df.empty:

        return

    # --------------------------------------------------------
    # 실제 계산
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
# 11. 최근 정비 맥락 Feature 생성
# ============================================================


def make_maintenance_context_features(maintenance, observed_at, train_parts):

    # ========================================================
    # 기본 Feature
    # ========================================================

    result = {
        # 과거 완료 정비의 총 정지시간
        "MNT_DOWN_SUM": 0.0,
        # 마지막 완료 정비 후 경과시간
        "H_SINCE_MNT": np.nan,
        # 과거 완료 정비 존재 여부
        "LAST_MNT_EXISTS": 0,
        # 가장 최근 완료 정비의 정지시간
        "LAST_DOWN_MIN": np.nan,
        # 가장 최근 완료 정비의 비용점수
        "LAST_COST_SCORE": np.nan,
        # 최근 정비 유형
        "LAST_MNT_IS_PARTIAL": 0,
        "LAST_MNT_IS_REPAIR": 0,
        "LAST_MNT_IS_REPLACE": 0,
        # Train에 없던 PART
        "LAST_PART_UNKNOWN": 0,
    }

    # ========================================================
    # 최근 정비 PART One-hot
    # ========================================================

    for part in train_parts:

        clean_part = clean_feature_name(part)

        result[f"LAST_PART_{clean_part}"] = 0

    # ========================================================
    # 정비 데이터 없음
    # ========================================================

    if maintenance is None or maintenance.empty:

        return result

    if "END_DT" not in maintenance.columns:

        return result

    # ========================================================
    # 매우 중요
    #
    # OBSERVED_AT 이전에
    # 이미 끝난 정비만 사용
    #
    # 미래 정비 정보 금지
    # ========================================================

    past = maintenance[
        maintenance["END_DT"].notna() & (maintenance["END_DT"] <= observed_at)
    ].copy()

    if past.empty:

        return result

    # ========================================================
    # 완료 정비 존재
    # ========================================================

    result["LAST_MNT_EXISTS"] = 1

    # ========================================================
    # 누적 DOWN_MIN
    # ========================================================

    if "DOWN_MIN" in past.columns:

        down = past["DOWN_MIN"].dropna()

        if not down.empty:

            result["MNT_DOWN_SUM"] = float(down.sum())

    # ========================================================
    # 가장 최근 완료된 정비
    # ========================================================

    last_idx = past["END_DT"].idxmax()

    last = past.loc[last_idx]

    last_end_dt = last["END_DT"]

    # ========================================================
    # 마지막 정비 후 경과시간
    # ========================================================

    result["H_SINCE_MNT"] = float((observed_at - last_end_dt).total_seconds() / 3600)

    # ========================================================
    # 최근 정비 DOWN_MIN
    # ========================================================

    value = last.get("DOWN_MIN", np.nan)

    if pd.notna(value):

        result["LAST_DOWN_MIN"] = float(value)

    # ========================================================
    # 최근 정비 COST_SCORE
    # ========================================================

    value = last.get("COST_SCORE", np.nan)

    if pd.notna(value):

        result["LAST_COST_SCORE"] = float(value)

    # ========================================================
    # 최근 정비 유형
    # ========================================================

    last_type = str(last.get("MNT_TYPE", "")).strip().lower()

    if last_type == "partial":

        result["LAST_MNT_IS_PARTIAL"] = 1

    elif last_type == "repair":

        result["LAST_MNT_IS_REPAIR"] = 1

    elif last_type == "replace":

        result["LAST_MNT_IS_REPLACE"] = 1

    # ========================================================
    # 최근 정비 PART
    # ========================================================

    last_part = str(last.get("PART_CD", "")).strip()

    if last_part:

        if last_part in train_parts:

            clean_part = clean_feature_name(last_part)

            result[f"LAST_PART_{clean_part}"] = 1

        else:

            result["LAST_PART_UNKNOWN"] = 1

    # ========================================================
    # FAILURE_DT는 Feature로 절대 사용하지 않음
    # ========================================================

    return result


# ============================================================
# 12. OBSERVED_AT별 전체 Feature 생성
# ============================================================


def make_features(raw, labels, maintenance, scenario_id, train_parts):

    results = []

    raw = raw.sort_values("MEAS_DT").reset_index(drop=True)

    for _, label in labels.iterrows():

        observed_at = label["OBSERVED_AT"]

        # ====================================================
        # 예측 시점 이전 Sensor만 사용
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
        # 현재 Sensor 값
        # ====================================================

        for col in BASE_FEATURES:

            row[f"CUR_{col}"] = current[col]

        # ====================================================
        # 1H / 3H / 6H / 12H Sensor Window
        # ====================================================

        for window_name, window_size in WINDOWS.items():

            start_time = observed_at - window_size

            window_df = past[past["MEAS_DT"] >= start_time]

            summarize_window(window_df, row, window_name)

        # ====================================================
        # 최근 정비 맥락 Feature
        # ====================================================

        maintenance_features = make_maintenance_context_features(
            maintenance, observed_at, train_parts
        )

        row.update(maintenance_features)

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 13. Scenario Feature 검증
# ============================================================


def validate_features(features, scenario_id, maintenance_feature_cols):

    if features.empty:

        raise RuntimeError(f"{scenario_id}: " f"Feature Dataset 비어 있음")

    # --------------------------------------------------------
    # 반드시 있어야 할 Sensor + Maintenance Feature
    # --------------------------------------------------------

    required_cols = [
        "CUR_LOAD",
        "1H_LOAD_MEAN",
        "3H_LOAD_MEAN",
        "6H_LOAD_MEAN",
        "12H_LOAD_MEAN",
        "12H_CONTROL_DELAY_CHANGE",
    ] + maintenance_feature_cols

    missing = [col for col in required_cols if col not in features.columns]

    if missing:

        raise RuntimeError(f"{scenario_id}: " f"Feature 누락 " f"{missing}")

    # --------------------------------------------------------
    # Window 정상 여부
    # --------------------------------------------------------

    row_count_cols = [
        "1H_ROW_COUNT",
        "3H_ROW_COUNT",
        "6H_ROW_COUNT",
        "12H_ROW_COUNT",
    ]

    if features[row_count_cols].max().max() <= 0:

        raise RuntimeError(f"{scenario_id}: " f"Window 추출 실패")

    # --------------------------------------------------------
    # 정비 경과시간 음수 금지
    # --------------------------------------------------------

    gap = features["H_SINCE_MNT"].dropna()

    if (gap < 0).any():

        raise RuntimeError(f"{scenario_id}: " f"H_SINCE_MNT 음수 발생")

    # --------------------------------------------------------
    # 정비 유형 One-hot 중복 확인
    # --------------------------------------------------------

    type_cols = [
        "LAST_MNT_IS_PARTIAL",
        "LAST_MNT_IS_REPAIR",
        "LAST_MNT_IS_REPLACE",
    ]

    type_sum = features[type_cols].sum(axis=1)

    if (type_sum > 1).any():

        raise RuntimeError(f"{scenario_id}: " f"정비 유형 One-hot 중복")

    # --------------------------------------------------------
    # PART One-hot 중복 확인
    # --------------------------------------------------------

    part_cols = [
        col for col in maintenance_feature_cols if col.startswith("LAST_PART_")
    ]

    if part_cols:

        part_sum = features[part_cols].sum(axis=1)

        if (part_sum > 1).any():

            raise RuntimeError(f"{scenario_id}: " f"정비 PART One-hot 중복")


# ============================================================
# 14. Scenario 하나 처리
# ============================================================


def process_scenario(
    root, scenario_id, manifest, train_parts, maintenance_feature_cols
):

    labels = load_labels(root, scenario_id)

    if labels is None:

        return None

    raw = load_casting(root, scenario_id, manifest)

    if raw is None:

        return None

    maintenance = load_maintenance(root, scenario_id, manifest)

    features = make_features(raw, labels, maintenance, scenario_id, train_parts)

    validate_features(features, scenario_id, maintenance_feature_cols)

    del raw
    del labels
    del maintenance

    gc.collect()

    return features


# ============================================================
# 15. S0001 사전 테스트
# ============================================================


def test_single_scenario(root, manifest, train_parts, maintenance_feature_cols):

    print()

    print("=" * 80)

    print("S0001 PRE-CHECK V2.2")

    print("=" * 80)

    features = process_scenario(
        root, "S0001", manifest, train_parts, maintenance_feature_cols
    )

    print("shape:", features.shape)

    print("컬럼 수:", len(features.columns))

    show_cols = [
        "OBSERVED_AT",
        "MNT_DOWN_SUM",
        "H_SINCE_MNT",
        "LAST_MNT_EXISTS",
        "LAST_DOWN_MIN",
        "LAST_COST_SCORE",
        "LAST_MNT_IS_PARTIAL",
        "LAST_MNT_IS_REPAIR",
        "LAST_MNT_IS_REPLACE",
    ]

    part_cols = [
        col for col in maintenance_feature_cols if col.startswith("LAST_PART_")
    ]

    show_cols += part_cols

    print()

    print(features[show_cols].to_string(index=False))

    print()

    print("S0001 V2.2 검증 성공!")


# ============================================================
# 16. Split 전체 처리
# ============================================================


def process_split(
    root, scenario_ids, split_name, manifest, train_parts, maintenance_feature_cols
):

    all_features = []

    failed = []

    success_count = 0

    total = len(scenario_ids)

    print()

    print("=" * 80)

    print(f"{split_name.upper()} " f"FEATURE V2.2 생성 시작")

    print("=" * 80)

    for idx, scenario_id in enumerate(scenario_ids, start=1):

        try:

            features = process_scenario(
                root, scenario_id, manifest, train_parts, maintenance_feature_cols
            )

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
# 17. 전체 Dataset 검사
# ============================================================


def check_dataset(df, name, maintenance_feature_cols):

    print()

    print("=" * 80)

    print(f"{name} DATASET CHECK V2.2")

    print("=" * 80)

    print("shape:", df.shape)

    print("scenario 수:", df["SCENARIO_ID"].nunique())

    print("전체 컬럼 수:", len(df.columns))

    print()

    print("정비 Feature 수:", len(maintenance_feature_cols))

    print()

    print("정비 Feature 목록")

    for col in maintenance_feature_cols:

        print(col, "→", col in df.columns)

    # --------------------------------------------------------
    # 과거 완료 정비 존재 행
    # --------------------------------------------------------

    count = int((df["LAST_MNT_EXISTS"] == 1).sum())

    print()

    print("과거 완료 정비 존재 행:", count, "/", len(df), f"({count / len(df):.2%})")

    # --------------------------------------------------------
    # 최근 정비 유형 분포
    # --------------------------------------------------------

    type_cols = [
        "LAST_MNT_IS_PARTIAL",
        "LAST_MNT_IS_REPAIR",
        "LAST_MNT_IS_REPLACE",
    ]

    print()

    print("최근 정비 유형 합계")

    print(df[type_cols].sum().sort_values(ascending=False))

    # --------------------------------------------------------
    # 최근 정비 PART 분포
    # --------------------------------------------------------

    part_cols = [
        col for col in maintenance_feature_cols if col.startswith("LAST_PART_")
    ]

    if part_cols:

        print()

        print("최근 정비 부품 One-hot 합계")

        print(df[part_cols].sum().sort_values(ascending=False))

    # --------------------------------------------------------
    # 정비 숫자 Feature 요약
    # --------------------------------------------------------

    numeric_mnt_cols = [
        "MNT_DOWN_SUM",
        "H_SINCE_MNT",
        "LAST_DOWN_MIN",
        "LAST_COST_SCORE",
    ]

    print()

    print("정비 숫자 Feature 요약")

    print(df[numeric_mnt_cols].describe().round(3))


# ============================================================
# 18. 안전 저장
# ============================================================


def safe_save(df, final_path):

    temp_path = final_path.parent / (final_path.stem + "_TEMP.csv")

    # --------------------------------------------------------
    # 임시 저장
    # --------------------------------------------------------

    df.to_csv(temp_path, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # 다시 열어서 확인
    # --------------------------------------------------------

    check = pd.read_csv(temp_path, nrows=10)

    if len(check.columns) != len(df.columns):

        raise RuntimeError(f"{final_path.name}: " f"저장 컬럼 수 불일치")

    # --------------------------------------------------------
    # 기존 최종 파일 삭제
    # --------------------------------------------------------

    if final_path.exists():

        final_path.unlink()

    # --------------------------------------------------------
    # TEMP → 최종
    # --------------------------------------------------------

    temp_path.rename(final_path)


# ============================================================
# 19. MAIN
# ============================================================


def main():

    print("=" * 80)

    print("CASTING FEATURE DATASET BUILDER V2.2")

    print("SENSOR + LAST MAINTENANCE CONTEXT")

    print("=" * 80)

    manifest = load_manifest(ROOT)

    train_ids, validation_ids, test_ids = get_scenario_splits(manifest)

    print()

    print("Train scenarios:", len(train_ids))

    print("Validation scenarios:", len(validation_ids))

    print("Test scenarios:", len(test_ids))

    # ========================================================
    # Split 안전 검사
    # ========================================================

    if len(train_ids) != 216 or len(validation_ids) != 72 or len(test_ids) != 72:

        raise RuntimeError("Scenario Split 개수가 " "예상과 다릅니다.")

    # ========================================================
    # Train에서만 정비 Category 탐색
    # ========================================================

    train_parts, train_mnt_types = discover_train_maintenance_categories(
        ROOT, train_ids, manifest
    )

    # ========================================================
    # V2.2 정비 Feature 목록
    # ========================================================

    maintenance_feature_cols = [
        "MNT_DOWN_SUM",
        "H_SINCE_MNT",
        "LAST_MNT_EXISTS",
        "LAST_DOWN_MIN",
        "LAST_COST_SCORE",
        "LAST_MNT_IS_PARTIAL",
        "LAST_MNT_IS_REPAIR",
        "LAST_MNT_IS_REPLACE",
        "LAST_PART_UNKNOWN",
    ]

    for part in train_parts:

        maintenance_feature_cols.append("LAST_PART_" + clean_feature_name(part))

    print()

    print("=" * 80)

    print("V2.2 MAINTENANCE FEATURES")

    print("=" * 80)

    print("총 정비 Feature:", len(maintenance_feature_cols))

    for col in maintenance_feature_cols:

        print(col)

    # ========================================================
    # S0001 사전 테스트
    # ========================================================

    test_single_scenario(ROOT, manifest, train_parts, maintenance_feature_cols)

    # ========================================================
    # Train 생성
    # ========================================================

    train_df, train_failed = process_split(
        ROOT, train_ids, "train", manifest, train_parts, maintenance_feature_cols
    )

    # ========================================================
    # Validation 생성
    # ========================================================

    validation_df, validation_failed = process_split(
        ROOT,
        validation_ids,
        "validation",
        manifest,
        train_parts,
        maintenance_feature_cols,
    )

    # ========================================================
    # 실패 검사
    # ========================================================

    if train_failed or validation_failed:

        print()

        print("=" * 80)

        print("FEATURE 생성 실패")

        print("=" * 80)

        print("Train 실패:", train_failed)

        print("Validation 실패:", validation_failed)

        raise RuntimeError("실패 Scenario가 존재합니다.")

    # ========================================================
    # 최종 Dataset 검사
    # ========================================================

    check_dataset(train_df, "TRAIN", maintenance_feature_cols)

    check_dataset(validation_df, "VALIDATION", maintenance_feature_cols)

    # ========================================================
    # 저장
    # ========================================================

    train_path = OUTPUT_DIR / "casting_train_features_v22.csv"

    validation_path = OUTPUT_DIR / "casting_validation_features_v22.csv"

    safe_save(train_df, train_path)

    safe_save(validation_df, validation_path)

    # ========================================================
    # 완료
    # ========================================================

    print()

    print("=" * 80)

    print("V2.2 Feature 생성 완료!")

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
# 20. 실행
# ============================================================

if __name__ == "__main__":

    main()
