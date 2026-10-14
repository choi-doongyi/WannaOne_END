from pathlib import Path
import pandas as pd

CHUNK_DIR = Path("chunks")

files = list(CHUNK_DIR.glob("*"))

for f in files:
    try:
        df = pd.read_csv(f, nrows=1000)

        if "EQP_TAG" not in df.columns:
            continue

        cc1 = df[df["EQP_TAG"].astype(str).str.contains("CC1", na=False)]

        if not cc1.empty:
            print("\n" + "=" * 100)
            print("파일:", f.name)
            print("컬럼:")
            print(df.columns.tolist())

            print("\nCC1 샘플:")
            print(cc1.head(10).to_string(index=False))

            break

    except Exception:
        continue
