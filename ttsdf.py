from pathlib import Path
import pandas as pd

OUTPUT_DIR = Path("outputs")

failure_path = OUTPUT_DIR / "casting_validation_failure_events_v1.csv"

df = pd.read_csv(failure_path)

print("=" * 80)
print("FAILURE FILE CHECK")
print("=" * 80)

print("shape:", df.shape)

print("\ncolumns:")
print(df.columns.tolist())

print("\nSCENARIO_ID unique:")
print(df["SCENARIO_ID"].nunique())

print("\nFAILURE_DT raw null:")
print(df["FAILURE_DT"].isna().sum())

print("\n전체:")
print(
    df[
        [
            "SCENARIO_ID",
            "FAILURE_DT",
            "PART_CD",
            "MNT_TYPE",
        ]
    ]
    .sort_values("SCENARIO_ID")
    .to_string(index=False)
)

df["FAILURE_DT_PARSED"] = pd.to_datetime(df["FAILURE_DT"], utc=True, errors="coerce")

print("\n시간 변환 실패:")
print(df[df["FAILURE_DT_PARSED"].isna()].to_string(index=False))

print("\n최종 유효 Failure:")
print(df["FAILURE_DT_PARSED"].notna().sum())
