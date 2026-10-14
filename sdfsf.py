import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(".")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def load_manifest(root):
    path = Path(root) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


manifest = load_manifest(ROOT)


rows = list(source_rows(ROOT, "S0001", "maintenance", manifest))


maintenance = pd.DataFrame(rows)


print("shape:", maintenance.shape)

print()

print("columns:")

print(maintenance.columns.tolist())

print()

print(maintenance.head(20).to_string(index=False))


print(maintenance.columns.tolist())
print()
print(maintenance.to_string(index=False))
