from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 0. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

# ============================================================
# 1. 데이터 불러오기
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

event = load_data(find_file("G-02_조업이벤트"))

temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])
event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])


# ============================================================
# 2. CR1 강종변경 이벤트
# ============================================================

cr1_event = event[
    (event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")
].sort_values("EVT_DT")


# ============================================================
# 3. LOT - 강종 매핑
# ============================================================

lot_start = temp.groupby("LOT_NO")["MEAS_DT"].min().reset_index(name="start_time")

grade_event = cr1_event[["EVT_DT", "AFT_VAL"]].rename(
    columns={"EVT_DT": "start_time", "AFT_VAL": "STEEL_GRADE"}
)

lot_grade = lot_start.merge(grade_event, on="start_time", how="left")

temp = temp.merge(lot_grade[["LOT_NO", "STEEL_GRADE"]], on="LOT_NO", how="left")


# ============================================================
# 4. LOT별 평균
# ============================================================

selected_sensors = ["TC-OUT1", "TC-OUT2", "OP-Z3", "TC-Z7", "TC-Z8", "OP-Z4"]

lot_mean = (
    temp.groupby(["STEEL_GRADE", "LOT_NO"])[selected_sensors].mean().reset_index()
)


# ============================================================
# 5. 결과 폴더
# ============================================================

FIGURE_DIR = PROJECT_ROOT / "figures" / "grade_analysis"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 6. 강종별 LOT 평균 시각화
# ============================================================

grade_order = ["SGCC", "DP590", "SPCD", "SPCE", "SPCEN", "SPFH590", "SPHC"]


for sensor in selected_sensors:

    fig, ax = plt.subplots(figsize=(10, 6))

    for x, grade in enumerate(grade_order):

        grade_data = lot_mean[lot_mean["STEEL_GRADE"] == grade]

        if grade_data.empty:
            continue

        # LOT별 평균값 표시
        ax.scatter([x] * len(grade_data), grade_data[sensor], s=80)

        # LOT 번호 표시
        for _, row in grade_data.iterrows():

            ax.annotate(
                str(int(row["LOT_NO"])),
                (x, row[sensor]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_xticks(range(len(grade_order)))

    ax.set_xticklabels(grade_order)

    ax.set_xlabel("Steel Grade")

    ax.set_ylabel(f"{sensor} LOT Mean")

    ax.set_title(f"{sensor} - LOT Mean by Steel Grade")

    ax.grid(alpha=0.3)

    plt.tight_layout()

    save_path = FIGURE_DIR / f"{sensor}_lot_mean_by_grade.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


print("\n시각화 저장 완료:")
print(FIGURE_DIR)
