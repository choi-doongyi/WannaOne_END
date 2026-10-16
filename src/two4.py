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
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

# --------------------------------------------------
# 1. 파일 경로와 입력 변수
# --------------------------------------------------
data_dir = Path(r"C:\Users\user\Documents\GitHub\WannaOne_END\data")
csv_dir = data_dir / "POSKO-PdM-Data-pdm-v2.2.0-parquet"

large_csv = csv_dir / "casting_with_failer_soon.csv"
small_csv = csv_dir / "casting_labeled_only.csv"

manifest_path = (
    data_dir
    / "데이터셋-20261013T011038Z-1-001"
    / "데이터셋"
    / "합성데이터_POSKO"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "POSKO-PdM-Data-pdm-v2.2.0"
    / "manifest.json"
)

features = [
    "LOAD",
    "WD-SPD",
    "WD-LOD",
    "OSC-FRQ",
    "OSC-STK",
    "MLD-LVL",
    "CONTROL_DELAY",
    "RUNNING",
]
columns = ["SCENARIO_ID", "failer_soon", *features]

# --------------------------------------------------
# 2. 정답이 있는 행만 작은 CSV로 저장
#    큰 CSV가 바뀌지 않았다면 다음 실행부터 건너뜀
# --------------------------------------------------
if not large_csv.exists():
    raise FileNotFoundError(f"라벨이 추가된 CSV가 없습니다: {large_csv}")

needs_extract = (
    not small_csv.exists() or small_csv.stat().st_mtime < large_csv.stat().st_mtime
)

if needs_extract:
    temp_csv = small_csv.with_name(small_csv.name + ".tmp")
    first_chunk = True
    saved_rows = 0

    for chunk in pd.read_csv(
        large_csv,
        usecols=columns,
        chunksize=200_000,
    ):
        labeled = chunk.dropna(subset=["failer_soon"]).copy()
        if labeled.empty:
            continue

        labeled["failer_soon"] = labeled["failer_soon"].astype("int8")
        labeled.to_csv(
            temp_csv,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        saved_rows += len(labeled)

    if first_chunk:
        raise ValueError("큰 CSV에서 정답이 있는 행을 찾지 못했습니다.")

    # 추출이 끝난 후에만 기존 작은 CSV를 교체
    temp_csv.replace(small_csv)
    print(f"학습용 CSV 생성: {small_csv} ({saved_rows:,}행)")
else:
    print(f"기존 학습용 CSV 사용: {small_csv}")

# --------------------------------------------------
# 3. 작은 CSV를 읽고 시나리오별 학습/검증 분리
# --------------------------------------------------
df = pd.read_csv(small_csv)
df["failer_soon"] = df["failer_soon"].astype(int)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
split_by_id = {scenario["id"]: scenario["split"] for scenario in manifest["scenarios"]}

train = df[df["SCENARIO_ID"].map(split_by_id) == "train"]
valid = df[df["SCENARIO_ID"].map(split_by_id) == "validation"]

X_train, y_train = train[features], train["failer_soon"]
X_valid, y_valid = valid[features], valid["failer_soon"]

if y_train.nunique() != 2 or y_valid.nunique() != 2:
    raise ValueError("학습 또는 검증 데이터에 정답 0과 1이 모두 있는지 확인하세요.")

print("학습 정답 분포:", y_train.value_counts().to_dict())
print("검증 정답 분포:", y_valid.value_counts().to_dict())

# --------------------------------------------------
# 4. 랜덤포레스트 학습
# --------------------------------------------------
model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    ),
)
model.fit(X_train, y_train)

# 검증 데이터의 '24시간 내 고장' 예측 점수
risk_score = model.predict_proba(X_valid)[:, 1]

print("PR-AUC:", average_precision_score(y_valid, risk_score))
print("ROC-AUC:", roc_auc_score(y_valid, risk_score))

# --------------------------------------------------
from sklearn.metrics import f1_score

# --------------------------------------------------
# 5. 비용에 따른 임계값 0 ~ 1을 0.01 간격으로 비교
# --------------------------------------------------
FN_COST = int(input("고장발생시 비용 입력: "))
FP_COST = int(input("헛정비 비용 입력: "))

results = []

for i in range(101):
    threshold = i / 100
    predicted = (risk_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_valid, predicted, labels=[0, 1]).ravel()

    results.append(
        {
            "임계값": threshold,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "정밀도": precision_score(y_valid, predicted, zero_division=0),
            "재현율": recall_score(y_valid, predicted, zero_division=0),
            "F1": f1_score(y_valid, predicted, zero_division=0),
            "손실점수": FN_COST * fn + FP_COST * fp,
        }
    )

result_df = pd.DataFrame(results)

# 손실이 같으면 FN이 적은 값, 그래도 같으면 낮은 임계값 선택
best = result_df.sort_values(
    ["손실점수", "FN", "임계값"],
    ascending=[True, True, True],
).iloc[0]
best_threshold = float(best["임계값"])

# F1 기준 최선도 별도로 확인
best_f1 = result_df.loc[result_df["F1"].idxmax()]

print(
    f"최소 손실 임계값: {best_threshold:.2f} "
    f"(손실 {int(best['손실점수']):,}점, "
    f"FP {int(best['FP'])}건, FN {int(best['FN'])}건)"
)
print(f"최대 F1 임계값: {best_f1['임계값']:.2f} " f"(F1 {best_f1['F1']:.3f})")
print("\n손실이 낮은 상위 10개 임계값")
print(
    result_df.nsmallest(10, "손실점수").to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    )
)

# 선택된 임계값으로 상세 분류 결과 출력
predicted_best = (risk_score >= best_threshold).astype(int)
print("\n최소 손실 임계값의 분류 결과")
print(
    classification_report(
        y_valid,
        predicted_best,
        target_names=["24시간 내 고장 없음", "24시간 내 고장"],
        zero_division=0,
    )
)

# --------------------------------------------------
# 6. 전체 임계값 곡선과 선택 결과 시각화
# --------------------------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

axes[0].plot(result_df["임계값"], result_df["손실점수"])
axes[0].axvline(best_threshold, color="red", linestyle="--")
axes[0].set(
    title=f"최소 손실 임계값: {best_threshold:.2f}",
    xlabel="임계값",
    ylabel="손실점수 (낮을수록 좋음)",
)
axes[0].grid(alpha=0.3)

for metric in ["정밀도", "재현율", "F1"]:
    axes[1].plot(result_df["임계값"], result_df[metric], label=metric)
axes[1].axvline(best_threshold, color="red", linestyle="--")
axes[1].set(
    title="임계값별 분류 성능",
    xlabel="임계값",
    ylabel="점수",
    ylim=(0, 1),
)
axes[1].legend()
axes[1].grid(alpha=0.3)

cm = confusion_matrix(y_valid, predicted_best, labels=[0, 1])
cell_names = [["TN", "FP"], ["FN", "TP"]]
axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=len(y_valid))

for row in range(2):
    for col in range(2):
        axes[2].text(
            col,
            row,
            f"{cell_names[row][col]}\n{cm[row, col]}",
            ha="center",
            va="center",
            fontsize=15,
            color="white" if cm[row, col] > len(y_valid) / 2 else "black",
        )

axes[2].set_xticks([0, 1], ["고장 없음", "24시간 내 고장"])
axes[2].set_yticks([0, 1], ["고장 없음", "24시간 내 고장"])
axes[2].set(
    title=f"임계값 {best_threshold:.2f} 혼동행렬",
    xlabel="모델 예측",
    ylabel="실제 정답",
)

plt.show()
