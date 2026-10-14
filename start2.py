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

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# 연주기 입력 후보
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


# 과거 시간창
WINDOWS = {
    "1H": pd.Timedelta(hours=1),
    "3H": pd.Timedelta(hours=3),
    "6H": pd.Timedelta(hours=6),
    "12H": pd.Timedelta(hours=12),
}


# 시간창별 생성 통계
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
        f
        for f in manifest["files"]
        if (f["scenarioId"] == scenario_id and f["id"] == file_id)
    )

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

    # CC1만
    df = df[df["EQP_CD"] == CASTING_EQP].copy()

    if df.empty:
        return None

    # UTC 통일
    df["OBSERVED_AT"] = pd.to_datetime(df["OBSERVED_AT"], utc=True, errors="coerce")

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

        # CC1만
        if row.get("EQP_CD") != CASTING_EQP:
            continue

        # 측정 품질 정상만
        if row.get("QUALITY_FLAG") != "ok":
            continue

        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows)

    # 시간 변환
    df["MEAS_DT"] = pd.to_datetime(df["MEAS_DT"], utc=True, errors="coerce")

    # 센서 숫자형 변환
    for col in BASE_FEATURES:

        df[col] = pd.to_numeric(df[col], errors="coerce")

    keep_cols = [
        "SCENARIO_ID",
        "MEAS_DT",
        "SLAB_NO",
    ] + BASE_FEATURES

    df = df[keep_cols].copy()

    # 메모리 절약
    for col in BASE_FEATURES:

        if col == "RUNNING":

            df[col] = pd.to_numeric(df[col], downcast="integer")

        else:

            df[col] = pd.to_numeric(df[col], downcast="float")

    df = df.dropna(subset=["MEAS_DT"]).sort_values("MEAS_DT").reset_index(drop=True)

    return df


# ============================================================
# 7. 한 Window의 통계 Feature 생성
# ============================================================


def summarize_window(window_df, result, prefix):

    # --------------------------------------------------------
    # 관측 개수
    # --------------------------------------------------------

    result[f"{prefix}_ROW_COUNT"] = len(window_df)

    # --------------------------------------------------------
    # 먼저 모든 통계 Feature를 NaN으로 생성
    #
    # 특정 센서가 모두 결측이어도
    # 컬럼 자체는 유지되도록 함
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        for stat in STATS:

            result[f"{prefix}_{col}_{stat}"] = np.nan

    if window_df.empty:
        return

    # --------------------------------------------------------
    # 실제 값 계산
    # --------------------------------------------------------

    for col in BASE_FEATURES:

        s = window_df[col].dropna()

        if s.empty:
            continue

        result[f"{prefix}_{col}_MEAN"] = float(s.mean())

        # 1개뿐이면 표준편차 계산 불가
        if len(s) >= 2:

            result[f"{prefix}_{col}_STD"] = float(s.std())

        result[f"{prefix}_{col}_MIN"] = float(s.min())

        result[f"{prefix}_{col}_MAX"] = float(s.max())

        result[f"{prefix}_{col}_RANGE"] = float(s.max() - s.min())

        result[f"{prefix}_{col}_MEDIAN"] = float(s.median())

        result[f"{prefix}_{col}_CHANGE"] = float(s.iloc[-1] - s.iloc[0])


# ============================================================
# 8. OBSERVED_AT별 Feature 생성
# ============================================================


def make_features(raw, labels, scenario_id):

    results = []

    raw = raw.sort_values("MEAS_DT").reset_index(drop=True)

    for _, label in labels.iterrows():

        observed_at = label["OBSERVED_AT"]

        # ====================================================
        # 예측시점 이전/현재 데이터만 사용
        # 미래 데이터 금지
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
        # 현재값 Feature
        # ====================================================

        for col in BASE_FEATURES:

            row[f"CUR_{col}"] = current[col]

        # ====================================================
        # 과거 Window
        # ====================================================

        for window_name, window_size in WINDOWS.items():

            start_time = observed_at - window_size

            window_df = past[past["MEAS_DT"] >= start_time]

            summarize_window(window_df, row, window_name)

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 9. Scenario Feature 검증
# ============================================================


def validate_features(features, scenario_id):

    if features.empty:

        raise RuntimeError(f"{scenario_id}: " f"Feature Dataset이 비어 있습니다.")

    # --------------------------------------------------------
    # 핵심 Feature 존재 확인
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
    ]

    missing = [col for col in required_cols if col not in features.columns]

    if missing:

        raise RuntimeError(f"{scenario_id}: " f"시계열 Feature 누락\n" f"{missing}")

    # --------------------------------------------------------
    # 컬럼 수 확인
    # SPLIT 넣기 전 약 244개
    # --------------------------------------------------------

    if len(features.columns) < 240:

        raise RuntimeError(
            f"{scenario_id}: " f"Feature 컬럼 수 이상 " f"({len(features.columns)}개)"
        )

    # --------------------------------------------------------
    # ROW_COUNT 실제 생성 여부 확인
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


# ============================================================
# 10. Scenario 하나 처리
# ============================================================


def process_scenario(root, scenario_id, manifest):

    labels = load_labels(root, scenario_id)

    if labels is None:
        return None

    raw = load_casting(root, scenario_id, manifest)

    if raw is None:
        return None

    features = make_features(raw, labels, scenario_id)

    validate_features(features, scenario_id)

    del raw
    del labels

    gc.collect()

    return features


# ============================================================
# 11. S0001 사전 테스트
# ============================================================


def test_single_scenario(root, manifest):

    print()
    print("=" * 80)
    print("S0001 PRE-CHECK")
    print("=" * 80)

    features = process_scenario(root, "S0001", manifest)

    print("S0001 shape:", features.shape)

    print("S0001 컬럼 수:", len(features.columns))

    cols = [
        "OBSERVED_AT",
        "1H_ROW_COUNT",
        "3H_ROW_COUNT",
        "6H_ROW_COUNT",
        "12H_ROW_COUNT",
    ]

    print()

    print(features[cols].to_string(index=False))

    # --------------------------------------------------------
    # 최소한 정상적인 최대값 범위 확인
    # --------------------------------------------------------

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

    print()

    print("S0001 사전 검증 성공!")


# ============================================================
# 12. Split 전체 처리
# ============================================================


def process_split(root, scenario_ids, split_name, manifest):

    all_features = []

    failed = []

    success_count = 0

    total = len(scenario_ids)

    print()
    print("=" * 80)

    print(f"{split_name.upper()} " f"FEATURE 생성 시작")

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
                f"처리 {idx}/{total}"
                f" | 성공 {success_count}"
                f" | 실패 {len(failed)}"
            )

        gc.collect()

    if not all_features:

        return (pd.DataFrame(), failed)

    result = pd.concat(all_features, ignore_index=True)

    return (result, failed)


# ============================================================
# 13. 전체 Dataset 검사
# ============================================================


def check_dataset(df, name):

    print()
    print("=" * 80)

    print(f"{name} DATASET CHECK")

    print("=" * 80)

    print("shape:", df.shape)

    print("scenario 수:", df["SCENARIO_ID"].nunique())

    print("전체 컬럼 수:", len(df.columns))

    print()
    print("RISK_24H")

    print(df["RISK_24H"].value_counts(dropna=False))

    print()
    print("RISK_6H")

    print(df["RISK_6H"].value_counts(dropna=False))

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

    # --------------------------------------------------------
    # 최종 안전검사
    # --------------------------------------------------------

    if df[row_count_cols].max().max() <= 0:

        raise RuntimeError(f"{name}: " f"Window ROW_COUNT가 모두 0입니다.")


# ============================================================
# 14. 안전 저장
# ============================================================


def safe_save(df, final_path):

    temp_path = final_path.parent / (final_path.stem + "_TEMP.csv")

    # 임시 저장
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")

    # 다시 열어서 검사
    check = pd.read_csv(temp_path, nrows=10)

    if len(check.columns) < 240:

        raise RuntimeError(
            f"{final_path.name}: "
            f"저장 후 컬럼 수가 "
            f"{len(check.columns)}개뿐입니다."
        )

    required_cols = [
        "1H_LOAD_MEAN",
        "3H_LOAD_MEAN",
        "6H_LOAD_MEAN",
        "12H_LOAD_MEAN",
    ]

    for col in required_cols:

        if col not in check.columns:

            raise RuntimeError(f"{final_path.name}: " f"{col} 누락")

    # 기존 최종본 삭제
    if final_path.exists():

        final_path.unlink()

    # TEMP → 최종 파일
    temp_path.rename(final_path)


# ============================================================
# 15. MAIN
# ============================================================


def main():

    print("=" * 80)
    print("CASTING FEATURE DATASET BUILDER V3")
    print("=" * 80)

    manifest = load_manifest(ROOT)

    train_ids, validation_ids, test_ids = get_scenario_splits(manifest)

    print()

    print("Train scenarios:", len(train_ids))

    print("Validation scenarios:", len(validation_ids))

    print("Test scenarios:", len(test_ids))

    # --------------------------------------------------------
    # Split 검증
    # --------------------------------------------------------

    if len(train_ids) != 216 or len(validation_ids) != 72 or len(test_ids) != 72:

        raise RuntimeError("Scenario Split 개수가 " "예상과 다릅니다.")

    # ========================================================
    # 전체 실행 전에 S0001 먼저 확인
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
    # 최종 점검
    # ========================================================

    check_dataset(train_df, "TRAIN")

    check_dataset(validation_df, "VALIDATION")

    # --------------------------------------------------------
    # 컬럼 수 최종 확인
    # --------------------------------------------------------

    if len(train_df.columns) < 240:

        raise RuntimeError(f"Train 컬럼 수 이상: " f"{len(train_df.columns)}")

    if len(validation_df.columns) < 240:

        raise RuntimeError(f"Validation 컬럼 수 이상: " f"{len(validation_df.columns)}")

    # ========================================================
    # 저장
    # ========================================================

    train_path = OUTPUT_DIR / "casting_train_features.csv"

    validation_path = OUTPUT_DIR / "casting_validation_features.csv"

    safe_save(train_df, train_path)

    safe_save(validation_df, validation_path)

    # ========================================================
    # 완료
    # ========================================================

    print()
    print("=" * 80)
    print("모든 Feature 생성 및 검증 성공!")
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
# 16. 실행
# ============================================================

if __name__ == "__main__":

    main()
