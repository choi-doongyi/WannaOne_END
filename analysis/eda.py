"""연주 설비(P1-CC1-INS01) 조기경보 EDA.

개인 탐색 노트북(eda.ipynb)에서 확정한 내용을 처음부터 끝까지 다시 계산한다.
절 번호는 노트북·팀 공유 문서(reports/연주_조기경보_EDA_근거.md)의 근거 번호와 함께 적혀 있다.

    python analysis/01_eda.py

- 데이터: src/posko_data.py로 변환한 data/posko/ Parquet (변환 방법은 src/posko_data.py 참고)
- 범위: 학습 세트 216개 시나리오만 쓴다. 검증·테스트 세트는 보지 않는다(분할 원칙).
- 결과: 표는 화면에 출력하고, 그림은 figures/에 저장한다.
- 실행 시간: 약 20초 (학습 세트 원자료 약 1,120만 행을 한 번에 불러온다)

절 구성
  1. 데이터 기본 정보        2. 시나리오와 원인          3·5. 대표 시나리오 비교 그래프
  6. 센서별 열화 변화량       7. 조기경보의 한계          8. 운전조건 이벤트
  9. 센서 간 관계            10. 원인 구분 가능성         11. 정비 효과
  (노트북 4절 '대표 시나리오 기본 점검'은 1-3~1-5와 결과가 같아 생략)

진단 정답과 고장 시각은 비교 대상을 나누는 데만 쓰고, 특징량으로 쓰지 않는다.
"""

"""Training-only casting EDA using the scenario-oriented POSKO Parquet files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / "data" / "POSKO-PdM-Data-pdm-v2.2.0-parquet"
EQUIPMENT = "P1-CC1-INS01"
SIGNALS = [("MLD_AMP_D", 1), ("CONTROL_DELAY_D", 1), ("WDL_C_D", 1), ("OSC_STK_D", -1)]
SENSORS = ["MLD_AMP", "CONTROL_DELAY", "WDL_C", "OSC_STK"]


def load_scenario(root, scenario_id, start_at):
    folder = root / "scenarios" / scenario_id
    columns = [
        "MEAS_DT",
        "EQP_CD",
        "LOAD",
        "WD-LOD",
        "OSC-STK",
        "MLD-LVL",
        "CONTROL_DELAY",
        "RUNNING",
        "QUALITY_FLAG",
    ]
    raw = pd.read_parquet(folder / "casting.parquet", columns=columns)
    raw = raw.loc[raw["EQP_CD"].eq(EQUIPMENT)].copy()
    start = pd.Timestamp(start_at)
    raw["HOUR"] = (
        (pd.to_datetime(raw["MEAS_DT"], utc=True) - start).dt.total_seconds() // 3600
    ).astype(int)
    running = raw["RUNNING"].eq(1)
    raw["SEG"] = (running & ~running.shift(fill_value=False)).astype("int64").cumsum()
    raw["WDL_C"] = (raw["WD-LOD"] - 3.06) / raw["LOAD"]
    active = raw.loc[running & raw["HOUR"].between(0, 71)]
    hourly = active.groupby(["SEG", "HOUR"], as_index=False).agg(
        N_RUN=("RUNNING", "size"),
        LOAD=("LOAD", "mean"),
        MLD_AMP=("MLD-LVL", "std"),
        CONTROL_DELAY=("CONTROL_DELAY", "mean"),
        WDL_C=("WDL_C", "mean"),
        OSC_STK=("OSC-STK", "mean"),
    )
    hourly.insert(0, "SCENARIO_ID", scenario_id)
    hourly["SEG_HOUR"] = hourly.groupby("SEG").cumcount()
    hourly["HOUR_END"] = start + pd.to_timedelta(hourly["HOUR"] + 1, unit="h")

    maintenance = pd.read_parquet(folder / "maintenance.parquet")
    maintenance = maintenance.loc[maintenance["EQP_TAG"].eq(EQUIPMENT)]
    failures = pd.to_datetime(maintenance["FAILURE_DT"].dropna(), utc=True).tolist()
    diagnosis = pd.read_parquet(folder / "diagnostics" / "diagnostics.parquet")
    diagnosis = diagnosis.loc[diagnosis["EQP_CD"].eq(EQUIPMENT)]
    normal = bool(~diagnosis["HEALTH_STATE"].isin(["degraded", "failed"]).any())
    labels = pd.read_parquet(folder / "labels" / "labels.parquet")
    labels = labels.loc[labels["EQP_CD"].eq(EQUIPMENT)]
    uncensored = labels.loc[labels["CENSORED_24H"].eq(0)]
    info = {
        "raw_rows": int(len(raw)),
        "stopped_rows": int((~running).sum()),
        "missing_mld_rows": int(raw["MLD-LVL"].isna().sum()),
        "quality_missing_rows": int(raw["QUALITY_FLAG"].eq("missing").sum()),
        "failure_events": len(failures),
        "normal": normal,
        "uncensored_labels": int(len(uncensored)),
        "positive_24h_labels": int(uncensored["RISK_24H"].eq(1).sum()),
    }
    return hourly, failures, info


def load_slope(frame, sensor):
    x = frame["LOAD"] - frame.groupby("SCENARIO_ID")["LOAD"].transform("mean")
    y = frame[sensor] - frame.groupby("SCENARIO_ID")[sensor].transform("mean")
    denominator = float((x * x).sum())
    return float((x * y).sum() / denominator) if denominator else 0.0


def add_changes(hourly):
    for sensor in SENSORS:
        baseline = (
            hourly.loc[hourly["SEG_HOUR"] < 2]
            .groupby(["SCENARIO_ID", "SEG"])[sensor]
            .median()
            .rename(sensor + "_BASE")
        )
        hourly = hourly.join(baseline, on=["SCENARIO_ID", "SEG"])
        hourly.loc[hourly["SEG_HOUR"] < 2, sensor + "_BASE"] = np.nan
        hourly[sensor + "_D"] = hourly[sensor] - hourly[sensor + "_BASE"]

        first = (
            hourly.loc[(hourly["SEG"] == 1) & (hourly["SEG_HOUR"] < 2)]
            .groupby("SCENARIO_ID")[sensor]
            .median()
            .rename(sensor + "_INITIAL_BASE")
        )
        hourly = hourly.join(first, on="SCENARIO_ID")
        hourly.loc[hourly["HOUR"] < 2, sensor + "_INITIAL_BASE"] = np.nan
        hourly[sensor + "_INITIAL_D"] = (
            hourly[sensor] - hourly[sensor + "_INITIAL_BASE"]
        )
    return hourly


def add_score(hourly, normal_ids):
    parameters, columns = {}, []
    for name, direction in SIGNALS:
        reference = (
            hourly.loc[hourly["SCENARIO_ID"].isin(normal_ids), name].dropna()
            * direction
        )
        mean, std = float(reference.mean()), float(reference.std())
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"No usable reference variance for {name}")
        parameters[name] = {"direction": direction, "mean": mean, "std": std}
        columns.append((hourly[name] * direction - mean) / std)
    components = pd.concat(columns, axis=1)
    hourly["SCORE"] = components.mean(axis=1)
    hourly.loc[components.isna().any(axis=1), "SCORE"] = np.nan
    return hourly, parameters


def compare(hourly, failures, normal_ids):
    normal = hourly.loc[hourly["SCENARIO_ID"].isin(normal_ids)]
    before = []
    for scenario_id, dates in failures.items():
        if not dates:
            continue
        group = hourly.loc[hourly["SCENARIO_ID"].eq(scenario_id)].copy()
        lead = np.full(len(group), np.inf)
        for date in dates:
            hours = (date - group["HOUR_END"]).dt.total_seconds().to_numpy() / 3600
            lead = np.minimum(lead, np.where(hours >= 0, hours, np.inf))
        before.append(group.loc[(lead >= 24) & (lead < 36)])
    pre_failure = pd.concat(before, ignore_index=True) if before else hourly.iloc[:0]
    result = {}
    for name, direction in SIGNALS + [("SCORE", 1)]:
        a = (normal[name] * direction).dropna()
        b = (pre_failure[name] * direction).dropna()
        auc = (
            roc_auc_score(
                np.r_[np.zeros(len(a)), np.ones(len(b))],
                np.r_[a.to_numpy(), b.to_numpy()],
            )
            if len(a) and len(b)
            else None
        )
        result[name] = {
            "normal_hours": int(len(a)),
            "pre_failure_hours": int(len(b)),
            "normal_median": float(a.median()) if len(a) else None,
            "pre_failure_median": float(b.median()) if len(b) else None,
            "eda_auc": float(auc) if auc is not None else None,
        }
    return result


def plot_example(hourly, scenario_id, dates, path):
    group = hourly.loc[hourly["SCENARIO_ID"].eq(scenario_id)]
    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True)
    for ax, (name, direction) in zip(axes, SIGNALS):
        ax.plot(group["HOUR"], group[name] * direction, linewidth=1.4)
        ax.set_ylabel(name)
    axes[-1].plot(group["HOUR"], group["SCORE"], color="black", linewidth=1.4)
    axes[-1].set_ylabel("SCORE")
    axes[-1].set_xlabel("Hour in scenario")
    start = group["HOUR_END"].min() - pd.Timedelta(hours=int(group["HOUR"].min()) + 1)
    for date in dates:
        x = (date - start).total_seconds() / 3600
        for ax in axes:
            ax.axvline(x, color="red", linestyle="--", linewidth=1)
    fig.suptitle(f"Casting signals: {scenario_id} (red = failure)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT / "analysis" / "output"
    )
    parser.add_argument(
        "--limit", type=int, help="Use only the first N train scenarios"
    )
    args = parser.parse_args()
    index = pd.read_parquet(args.data_root / "scenario_index.parquet")
    train = index.loc[index["split"].eq("train")].sort_values("scenario_id")
    if args.limit is not None:
        train = train.head(args.limit)
    if train.empty:
        raise ValueError("No training scenarios found")

    parts, failures, details = [], {}, {}
    for position, row in enumerate(train.itertuples(index=False), 1):
        part, dates, info = load_scenario(args.data_root, row.scenario_id, row.start_at)
        parts.append(part)
        failures[row.scenario_id] = dates
        details[row.scenario_id] = info
        if position % 25 == 0 or position == len(train):
            print(f"Read {position}/{len(train)} train scenarios", flush=True)
    hourly = pd.concat(parts, ignore_index=True)
    normal_ids = {key for key, value in details.items() if value["normal"]}
    if not normal_ids:
        raise ValueError("No normal training scenarios for calibration")
    normal = hourly.loc[hourly["SCENARIO_ID"].isin(normal_ids)]
    slope_stk = load_slope(normal, "OSC_STK")
    slope_delay = load_slope(normal, "CONTROL_DELAY")
    reference_load = float(normal["LOAD"].median())
    hourly["OSC_STK"] -= slope_stk * (hourly["LOAD"] - reference_load)
    hourly["CONTROL_DELAY"] -= slope_delay * (hourly["LOAD"] - reference_load)
    hourly = add_changes(hourly)
    hourly, score_parameters = add_score(hourly, normal_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(
        args.output_dir / "train_hourly_features.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "train_scenarios": int(len(train)),
        "normal_scenarios": len(normal_ids),
        "failure_scenarios": sum(bool(v) for v in failures.values()),
        "failure_events": sum(len(v) for v in failures.values()),
        "hourly_rows": int(len(hourly)),
        "raw_rows": sum(v["raw_rows"] for v in details.values()),
        "stopped_rows": sum(v["stopped_rows"] for v in details.values()),
        "missing_mld_rows": sum(v["missing_mld_rows"] for v in details.values()),
        "quality_missing_rows": sum(
            v["quality_missing_rows"] for v in details.values()
        ),
        "uncensored_24h_labels": sum(v["uncensored_labels"] for v in details.values()),
        "positive_24h_labels": sum(v["positive_24h_labels"] for v in details.values()),
        "reference_load": reference_load,
        "load_slopes": {"OSC_STK": slope_stk, "CONTROL_DELAY": slope_delay},
        "score_parameters": score_parameters,
        "signal_comparison": compare(hourly, failures, normal_ids),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    failure_example = next((key for key, value in failures.items() if value), None)
    if failure_example:
        plot_example(
            hourly,
            failure_example,
            failures[failure_example],
            args.output_dir / "example_failure.png",
        )
    normal_example = next(iter(sorted(normal_ids)))
    plot_example(hourly, normal_example, [], args.output_dir / "example_normal.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
