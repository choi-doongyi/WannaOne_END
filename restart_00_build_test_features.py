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


WINDOWS = {
    "1H": pd.Timedelta(hours=1),
    "3H": pd.Timedelta(hours=3),
    "6H": pd.Timedelta(hours=6),
    "12H": pd.Timedelta(hours=12),
}


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
# 1. CSV 읽기
# ============================================================


def read_rows(path):

    with Path(path).open(encoding="utf-8-sig", newline="") as f:

        yield from csv.DictReader(f)


# ============================================================
# 2. manifest
# ============================================================


def load_manifest(root):

    path = Path(root) / "manifest.json"

    return json.loads(path.read_text(encoding="utf-8"))


# ============================================================
# 3. Test Scenario ID
# ============================================================


def get_test_ids(manifest):

    test_ids = []

    for scenario in manifest["scenarios"]:

        if scenario["split"] == "test":

            test_ids.append(scenario["id"])

    return sorted(test_ids)


# ============================================================
# 4. Source Row Streaming
# ============================================================


def source_rows(
    root,
    scenario_id,
    file_id,
    manifest,
):

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
# 5. Casting Raw
# ============================================================


def load_casting(
    root,
    scenario_id,
    manifest,
):

    rows = []

    for row in source_rows(
        root,
        scenario_id,
        SOURCE_ID,
        manifest,
    ):

        if row.get("EQP_CD") != CASTING_EQP:

            continue

        if row.get("QUALITY_FLAG") != "ok":

            continue

        rows.append(row)

    if not rows:

        return None

    df = pd.DataFrame(rows)

    df["MEAS_DT"] = pd.to_datetime(
        df["MEAS_DT"],
        utc=True,
        errors="coerce",
    )

    for col in BASE_FEATURES:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    keep_cols = [
        "SCENARIO_ID",
        "MEAS_DT",
        "SLAB_NO",
    ] + BASE_FEATURES

    df = df[keep_cols].copy()

    for col in BASE_FEATURES:

        if col == "RUNNING":

            df[col] = pd.to_numeric(
                df[col],
                downcast="integer",
            )

        else:

            df[col] = pd.to_numeric(
                df[col],
                downcast="float",
            )

    df = df.dropna(subset=["MEAS_DT"]).sort_values("MEAS_DT").reset_index(drop=True)

    return df


# ============================================================
# 6. Window 통계
# ============================================================


def summarize_window(
    window_df,
    result,
    prefix,
):

    result[f"{prefix}_ROW_COUNT"] = len(window_df)

    for col in BASE_FEATURES:

        for stat in STATS:

            result[f"{prefix}_{col}_{stat}"] = np.nan

    if window_df.empty:

        return

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
# 7. Test OBSERVED_AT 생성
#
# 72시간 Scenario 기준
# 0, 6, 12, ... 72H
#
# Train / Validation과 동일하게
# Scenario당 13개 시점
# ============================================================


def make_test_observation_times(raw):

    start_time = raw["MEAS_DT"].min()

    observed_times = [
        start_time + pd.Timedelta(hours=h)
        for h in range(
            0,
            73,
            6,
        )
    ]

    return observed_times


# ============================================================
# 8. Test Feature 생성
# ============================================================


def make_test_features(
    raw,
    scenario_id,
):

    results = []

    raw = raw.sort_values("MEAS_DT").reset_index(drop=True)

    observed_times = make_test_observation_times(raw)

    for observed_at in observed_times:

        past = raw[raw["MEAS_DT"] <= observed_at].copy()

        if past.empty:

            continue

        current = past.iloc[-1]

        row = {
            "SCENARIO_ID": scenario_id,
            "OBSERVED_AT": observed_at,
            "LAST_MEAS_DT": current["MEAS_DT"],
            "LAST_MEAS_GAP_MIN": (
                (observed_at - current["MEAS_DT"]).total_seconds() / 60
            ),
            # Test는 정답 없음
            "RISK_6H": np.nan,
            "RISK_24H": np.nan,
            "CENSORED_6H": np.nan,
            "CENSORED_24H": np.nan,
        }

        # ----------------------------------------------------
        # 현재값
        # ----------------------------------------------------

        for col in BASE_FEATURES:

            row[f"CUR_{col}"] = current[col]

        # ----------------------------------------------------
        # Window
        # ----------------------------------------------------

        for window_name, window_size in WINDOWS.items():

            start_time = observed_at - window_size

            window_df = past[past["MEAS_DT"] >= start_time]

            summarize_window(
                window_df,
                row,
                window_name,
            )

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 9. Scenario 하나 처리
# ============================================================


def process_test_scenario(
    root,
    scenario_id,
    manifest,
):

    raw = load_casting(
        root,
        scenario_id,
        manifest,
    )

    if raw is None:

        return None

    features = make_test_features(
        raw,
        scenario_id,
    )

    del raw

    gc.collect()

    return features


# ============================================================
# 10. 전체 Test 처리
# ============================================================


def main():

    print("=" * 80)

    print("TEST FEATURE DATASET BUILDER")

    print("=" * 80)

    manifest = load_manifest(ROOT)

    test_ids = get_test_ids(manifest)

    print()

    print("Test scenarios:", len(test_ids))

    if len(test_ids) != 72:

        raise RuntimeError("Test Scenario 수가 72가 아닙니다.")

    all_features = []

    failed = []

    for i, scenario_id in enumerate(
        test_ids,
        start=1,
    ):

        print(f"[{i:02d}/72] " f"{scenario_id}")

        try:

            features = process_test_scenario(
                ROOT,
                scenario_id,
                manifest,
            )

            if features is None or features.empty:

                failed.append(scenario_id)

                continue

            if len(features) != 13:

                raise RuntimeError(
                    f"{scenario_id}: " f"예상 13개 시점인데 " f"{len(features)}개"
                )

            all_features.append(features)

        except Exception as e:

            print("ERROR:", e)

            failed.append(scenario_id)

    # ========================================================
    # 실패 검사
    # ========================================================

    if failed:

        print()

        print("실패 Scenario:")

        print(failed)

        raise RuntimeError("Test Feature 생성 실패 Scenario 존재")

    # ========================================================
    # 병합
    # ========================================================

    test_df = pd.concat(
        all_features,
        ignore_index=True,
    )

    test_df["SPLIT"] = "test"

    print()

    print("=" * 80)

    print("TEST FEATURE CHECK")

    print("=" * 80)

    print("Shape:", test_df.shape)

    print("Scenario:", test_df["SCENARIO_ID"].nunique())

    print("Rows per Scenario:")

    print(test_df.groupby("SCENARIO_ID").size().value_counts().sort_index())

    # ========================================================
    # 저장
    # ========================================================

    output_path = OUTPUT_DIR / "casting_test_features_v22.csv"

    test_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()

    print("=" * 80)

    print("TEST FEATURE COMPLETE")

    print("=" * 80)

    print("Saved:", output_path)

    print("Final Shape:", test_df.shape)


if __name__ == "__main__":

    main()
