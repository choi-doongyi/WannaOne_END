"""Focused checks for the reusable casting random-forest helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.casting_rf import (
    FEATURES,
    evaluate_thresholds,
    load_splits,
    prepare_labeled_csv,
    select_threshold,
)


class CastingRfTests(unittest.TestCase):
    def test_extracts_only_known_labels_and_reuses_cache(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            large_csv = Path(directory) / "large.csv"
            small_csv = Path(directory) / "small.csv"
            rows = pd.DataFrame({
                "SCENARIO_ID": ["S0001", "S0001", "S0002"],
                "failer_soon": [0, None, 1],
                **{feature: [1, 2, 3] for feature in FEATURES},
            })
            rows.to_csv(large_csv, index=False)

            path, created = prepare_labeled_csv(
                large_csv, small_csv, chunksize=2
            )
            self.assertTrue(created)
            self.assertEqual(path, small_csv)
            self.assertEqual(pd.read_csv(path)["failer_soon"].tolist(), [0, 1])
            self.assertEqual(prepare_labeled_csv(large_csv, small_csv), (path, False))

    def test_scenario_split_uses_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            directory = Path(directory)
            small_csv = directory / "small.csv"
            manifest_path = directory / "manifest.json"
            pd.DataFrame({
                "SCENARIO_ID": ["S0001", "S0002", "S0003", "S0004"],
                "failer_soon": [0, 1, 0, 1],
            }).to_csv(small_csv, index=False)
            manifest_path.write_text(json.dumps({"scenarios": [
                {"id": f"S{i:04d}", "split": "train" if i <= 2 else "validation"}
                for i in range(1, 5)
            ]}), encoding="utf-8")

            train, valid = load_splits(small_csv, manifest_path)
            self.assertEqual(train["SCENARIO_ID"].tolist(), ["S0001", "S0002"])
            self.assertEqual(valid["SCENARIO_ID"].tolist(), ["S0003", "S0004"])

    def test_threshold_cost_and_tie_break(self):
        results = evaluate_thresholds(
            pd.Series([0, 0, 1, 1]),
            pd.Series([0.1, 0.7, 0.4, 0.9]),
            fn_cost=200,
            fp_cost=20,
        )
        self.assertEqual(len(results), 101)
        best = select_threshold(results)
        self.assertEqual(best["손실점수"], 20)
        self.assertEqual(best["FN"], 0)
        self.assertAlmostEqual(best["임계값"], 0.11)
        with self.assertRaises(ValueError):
            evaluate_thresholds(pd.Series([0, 1]), pd.Series([0.2, 0.8]), -1, 20)


if __name__ == "__main__":
    unittest.main()
