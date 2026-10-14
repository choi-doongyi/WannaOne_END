import csv
import json

from pathlib import Path

import pandas as pd

# ============================================================
# 0. 설정
# ============================================================

ROOT = Path(".")

CASTING_EQP = "P1-CC1-INS01"
SOURCE_ID = "casting"


# ============================================================
# 1. CSV 스트리밍
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
# 3. Source 데이터 읽기
# ============================================================


def source_rows(
    root,
    scenario_id,
    file_id,
    manifest,
):

    source = next(
        f
        for f in manifest["files"]
        if (f["scenarioId"] == scenario_id and f["id"] == file_id)
    )

    for chunk in source["chunks"]:

        chunk_path = Path(root) / "chunks" / chunk["sha256"]

        for row in read_rows(chunk_path):

            yield row


# ============================================================
# 4. Scenario Split
# ============================================================


def get_all_scenario_ids(manifest):

    return sorted(scenario["id"] for scenario in manifest["scenarios"])


# ============================================================
# 5. Casting 시간만 읽기
#
# 메모리 절약을 위해
# 센서값 전체를 읽지 않고 시간만 사용
# ============================================================


def load_casting_time(
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

        # CC1만
        if row.get("EQP_CD") != CASTING_EQP:

            continue

        # 정상 품질만
        if row.get("QUALITY_FLAG") != "ok":

            continue

        rows.append(
            {
                "SCENARIO_ID": scenario_id,
                "MEAS_DT": row.get("MEAS_DT"),
            }
        )

    if not rows:

        return None

    df = pd.DataFrame(rows)

    df["MEAS_DT"] = pd.to_datetime(
        df["MEAS_DT"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["MEAS_DT"]).sort_values("MEAS_DT").reset_index(drop=True)

    return df


# ============================================================
# 6. 전체 시나리오 검사
# ============================================================

manifest = load_manifest(ROOT)


scenario_ids = get_all_scenario_ids(manifest)


print("=" * 80)
print("5-SECOND TIME INTERVAL QUALITY CHECK")
print("=" * 80)

print("Scenario:", len(scenario_ids))


all_summary = []

all_diff_values = []


for i, scenario_id in enumerate(
    scenario_ids,
    start=1,
):

    df = load_casting_time(
        ROOT,
        scenario_id,
        manifest,
    )

    if df is None or df.empty:

        continue

    # --------------------------------------------------------
    # 시간 차이
    # --------------------------------------------------------

    diff_sec = df["MEAS_DT"].diff().dt.total_seconds().dropna()

    if diff_sec.empty:

        continue

    total = len(diff_sec)

    exact_5 = int((diff_sec == 5).sum())

    duplicate = int((diff_sec == 0).sum())

    under_5 = int(((diff_sec > 0) & (diff_sec < 5)).sum())

    over_5 = int((diff_sec > 5).sum())

    negative = int((diff_sec < 0).sum())

    all_summary.append(
        {
            "SCENARIO_ID": scenario_id,
            "TOTAL_INTERVALS": total,
            "EXACT_5S": exact_5,
            "DUPLICATE_0S": duplicate,
            "UNDER_5S": under_5,
            "OVER_5S": over_5,
            "NEGATIVE": negative,
            "MAX_GAP_SEC": float(diff_sec.max()),
        }
    )

    all_diff_values.append(diff_sec)

    if i % 20 == 0 or i == len(scenario_ids):

        print(f"{i}/{len(scenario_ids)} 완료")


# ============================================================
# 7. 전체 결과
# ============================================================

summary_df = pd.DataFrame(all_summary)


all_diff = pd.concat(
    all_diff_values,
    ignore_index=True,
)


total = len(all_diff)


exact_5 = int((all_diff == 5).sum())


duplicate = int((all_diff == 0).sum())


under_5 = int(((all_diff > 0) & (all_diff < 5)).sum())


over_5 = int((all_diff > 5).sum())


negative = int((all_diff < 0).sum())


print()

print("=" * 80)
print("OVERALL RESULT")
print("=" * 80)


print(f"전체 시간 간격 : " f"{total:,}")


print(f"정확히 5초     : " f"{exact_5:,} " f"({exact_5 / total:.6%})")


print(f"중복 0초       : " f"{duplicate:,} " f"({duplicate / total:.6%})")


print(f"0~5초 미만     : " f"{under_5:,} " f"({under_5 / total:.6%})")


print(f"5초 초과       : " f"{over_5:,} " f"({over_5 / total:.6%})")


print(f"음수 간격      : " f"{negative:,} " f"({negative / total:.6%})")


print()

print("=" * 80)
print("5초 초과 Gap 요약")
print("=" * 80)


if over_5 > 0:

    print(all_diff[all_diff > 5].describe())

else:

    print("5초 초과 간격 없음")


print()

print("=" * 80)
print("가장 큰 Gap Top 20")
print("=" * 80)


print(all_diff.sort_values(ascending=False).head(20).to_string(index=False))


print()

print("=" * 80)
print("비정상 간격이 많은 Scenario Top 20")
print("=" * 80)


summary_df["ABNORMAL_COUNT"] = (
    summary_df["DUPLICATE_0S"]
    + summary_df["UNDER_5S"]
    + summary_df["OVER_5S"]
    + summary_df["NEGATIVE"]
)


print(
    summary_df.sort_values(
        "ABNORMAL_COUNT",
        ascending=False,
    )
    .head(20)
    .to_string(index=False)
)
