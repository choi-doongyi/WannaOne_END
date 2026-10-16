"""CSV-based 24-hour casting failure classification."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline


FEATURES = (
    "LOAD", "WD-SPD", "WD-LOD", "OSC-FRQ", "OSC-STK", "MLD-LVL",
    "CONTROL_DELAY", "RUNNING",
)
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOURCE_DIR = (
    "데이터셋-20261013T011038Z-1-001/데이터셋/합성데이터_POSKO/"
    "POSKO-PdM-Data-pdm-v2.2.0/POSKO-PdM-Data-pdm-v2.2.0"
)


def data_paths(data_dir=DEFAULT_DATA_DIR):
    data_dir = Path(data_dir)
    csv_dir = data_dir / "POSKO-PdM-Data-pdm-v2.2.0-parquet"
    return (
        csv_dir / "casting_with_failer_soon.csv",
        csv_dir / "casting_labeled_only.csv",
        data_dir / SOURCE_DIR / "manifest.json",
    )


def prepare_labeled_csv(large_csv, small_csv, *, force=False, chunksize=200_000):
    """Cache only rows with a known label, without changing the large CSV."""
    large_csv, small_csv = Path(large_csv), Path(small_csv)
    if not large_csv.exists():
        raise FileNotFoundError(f"라벨이 추가된 CSV가 없습니다: {large_csv}")
    if (not force and small_csv.exists()
            and small_csv.stat().st_mtime >= large_csv.stat().st_mtime):
        return small_csv, False

    temp_csv = small_csv.with_name(small_csv.name + ".tmp")
    first_chunk = True
    for chunk in pd.read_csv(
        large_csv,
        usecols=["SCENARIO_ID", "failer_soon", *FEATURES],
        chunksize=chunksize,
    ):
        labeled = chunk.dropna(subset=["failer_soon"]).copy()
        if labeled.empty:
            continue
        labeled["failer_soon"] = labeled["failer_soon"].astype("int8")
        labeled.to_csv(
            temp_csv, mode="w" if first_chunk else "a",
            header=first_chunk, index=False,
        )
        first_chunk = False
    if first_chunk:
        raise ValueError("큰 CSV에서 정답이 있는 행을 찾지 못했습니다.")
    temp_csv.replace(small_csv)
    return small_csv, True


def load_splits(small_csv, manifest_path):
    """Keep independent scenarios apart for training and validation."""
    df = pd.read_csv(small_csv)
    df["failer_soon"] = df["failer_soon"].astype(int)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    split_by_id = {
        scenario["id"]: scenario["split"] for scenario in manifest["scenarios"]
    }
    splits = df["SCENARIO_ID"].map(split_by_id)
    if splits.isna().any():
        raise ValueError("manifest.json에 없는 SCENARIO_ID가 CSV에 있습니다.")
    train = df.loc[splits.eq("train")].copy()
    valid = df.loc[splits.eq("validation")].copy()
    if train.empty or valid.empty:
        raise ValueError("학습 또는 검증 시나리오가 없습니다.")
    if train["failer_soon"].nunique() != 2 or valid["failer_soon"].nunique() != 2:
        raise ValueError("학습과 검증 데이터에 정답 0과 1이 모두 필요합니다.")
    return train, valid


def train_model(train):
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        ),
    )
    model.fit(train[list(FEATURES)], train["failer_soon"])
    return model


def evaluate_thresholds(y_true, risk_score, fn_cost, fp_cost):
    """Score 101 hundredth-step thresholds by cost per labeled row."""
    if fn_cost < 0 or fp_cost < 0:
        raise ValueError("FN/FP 비용은 0 이상이어야 합니다.")
    results = []
    for i in range(101):
        threshold = i / 100
        predicted = (risk_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_true, predicted, labels=[0, 1]
        ).ravel()
        results.append({
            "임계값": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "정밀도": precision_score(y_true, predicted, zero_division=0),
            "재현율": recall_score(y_true, predicted, zero_division=0),
            "F1": f1_score(y_true, predicted, zero_division=0),
            "손실점수": fn_cost * fn + fp_cost * fp,
        })
    return pd.DataFrame(results)


def select_threshold(results):
    return results.sort_values(
        ["손실점수", "FN", "임계값"], ascending=[True, True, True]
    ).iloc[0]


def plot_results(results, y_true, risk_score, best_threshold):
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    predicted = (risk_score >= best_threshold).astype(int)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    axes[0].plot(results["임계값"], results["손실점수"])
    axes[0].axvline(best_threshold, color="red", linestyle="--")
    axes[0].set(
        title=f"최소 손실 임계값: {best_threshold:.2f}",
        xlabel="임계값", ylabel="손실점수 (낮을수록 좋음)",
    )
    axes[0].grid(alpha=0.3)

    for metric in ("정밀도", "재현율", "F1"):
        axes[1].plot(results["임계값"], results[metric], label=metric)
    axes[1].axvline(best_threshold, color="red", linestyle="--")
    axes[1].set(title="임계값별 분류 성능", xlabel="임계값", ylabel="점수", ylim=(0, 1))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=len(y_true))
    cell_names = (("TN", "FP"), ("FN", "TP"))
    for row in range(2):
        for col in range(2):
            axes[2].text(
                col, row, f"{cell_names[row][col]}\n{cm[row, col]}",
                ha="center", va="center", fontsize=15,
                color="white" if cm[row, col] > len(y_true) / 2 else "black",
            )
    axes[2].set_xticks([0, 1], ["고장 없음", "24시간 내 고장"])
    axes[2].set_yticks([0, 1], ["고장 없음", "24시간 내 고장"])
    axes[2].set(
        title=f"임계값 {best_threshold:.2f} 혼동행렬",
        xlabel="모델 예측", ylabel="실제 정답",
    )
    return fig


def run_analysis(data_dir, fn_cost, fp_cost, *, plot=True, force_extract=False):
    large_csv, small_csv, manifest_path = data_paths(data_dir)
    small_csv, created = prepare_labeled_csv(
        large_csv, small_csv, force=force_extract
    )
    print(f"{'학습용 CSV 생성' if created else '기존 학습용 CSV 사용'}: {small_csv}")
    train, valid = load_splits(small_csv, manifest_path)
    print("학습 정답 분포:", train["failer_soon"].value_counts().to_dict())
    print("검증 정답 분포:", valid["failer_soon"].value_counts().to_dict())

    model = train_model(train)
    y_valid = valid["failer_soon"]
    risk_score = model.predict_proba(valid[list(FEATURES)])[:, 1]
    print("PR-AUC:", average_precision_score(y_valid, risk_score))
    print("ROC-AUC:", roc_auc_score(y_valid, risk_score))

    results = evaluate_thresholds(y_valid, risk_score, fn_cost, fp_cost)
    best = select_threshold(results)
    best_threshold = float(best["임계값"])
    best_f1 = results.loc[results["F1"].idxmax()]
    print(
        f"최소 손실 임계값: {best_threshold:.2f} "
        f"(손실 {int(best['손실점수']):,}점, "
        f"FP {int(best['FP'])}건, FN {int(best['FN'])}건)"
    )
    print(f"최대 F1 임계값: {best_f1['임계값']:.2f} (F1 {best_f1['F1']:.3f})")
    print("\n손실이 낮은 상위 10개 임계값")
    print(results.nsmallest(10, "손실점수").to_string(
        index=False, float_format=lambda value: f"{value:.3f}"
    ))
    predicted_best = (risk_score >= best_threshold).astype(int)
    print("\n최소 손실 임계값의 분류 결과")
    print(classification_report(
        y_valid, predicted_best,
        target_names=["24시간 내 고장 없음", "24시간 내 고장"],
        zero_division=0,
    ))
    if plot:
        plot_results(results, y_valid, risk_score, best_threshold)
        plt.show()
    return model, results, best


def main(argv=None):
    parser = argparse.ArgumentParser(description="연주 설비 24시간 고장 조기경보")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--fn-cost", type=int, help="고장을 놓친 경우의 비용")
    parser.add_argument("--fp-cost", type=int, help="헛정비 비용")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    fn_cost = args.fn_cost if args.fn_cost is not None else int(input("고장발생시 비용 입력: "))
    fp_cost = args.fp_cost if args.fp_cost is not None else int(input("헛정비 비용 입력: "))
    run_analysis(
        args.data_dir, fn_cost, fp_cost,
        plot=not args.no_plot, force_extract=args.force_extract,
    )


if __name__ == "__main__":
    main()
