from pathlib import Path
import sys
import pandas as pd
import numpy as np
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
# 2. CR1 강종 변경 이벤트
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
# 4. SPHC 데이터만 추출
# ============================================================

target_grade = "SPHC"

sphc = (
    temp[temp["STEEL_GRADE"] == target_grade].copy().sort_values(["LOT_NO", "MEAS_DT"])
)


print("\n" + "=" * 100)
print("[SPHC LOT 목록]")
print("=" * 100)

print(
    sphc.groupby("LOT_NO").agg(
        start_time=("MEAS_DT", "min"),
        end_time=("MEAS_DT", "max"),
        n_rows=("MEAS_DT", "size"),
    )
)

# ============================================================
# 5. LOT별 진행률 계산
# ============================================================

sphc = sphc.sort_values(["LOT_NO", "MEAS_DT"]).copy()


# 각 행에 해당 LOT의 시작시간 / 종료시간 붙이기
sphc["lot_start"] = sphc.groupby("LOT_NO")["MEAS_DT"].transform("min")

sphc["lot_end"] = sphc.groupby("LOT_NO")["MEAS_DT"].transform("max")


# LOT 시작 이후 경과시간
sphc["elapsed_seconds"] = (sphc["MEAS_DT"] - sphc["lot_start"]).dt.total_seconds()


# LOT 전체 시간
sphc["total_seconds"] = (sphc["lot_end"] - sphc["lot_start"]).dt.total_seconds()


# LOT 진행률 0 ~ 100%
sphc["progress"] = sphc["elapsed_seconds"] / sphc["total_seconds"] * 100


print("\n진행률 확인")

print(sphc[["LOT_NO", "MEAS_DT", "progress"]].head())

print("\n컬럼 확인")
print(sphc.columns.tolist())

# ============================================================
# 6. 진행률 구간 생성
# ============================================================

# 0~100%를 100개 구간으로 나눔
sphc["progress_bin"] = np.floor(sphc["progress"]).clip(0, 99).astype(int)


# ============================================================
# 7. 비교할 변수
# ============================================================

selected_sensors = ["TC-OUT1", "TC-OUT2", "OP-Z3", "OP-Z4"]


# ============================================================
# 8. LOT × 진행률 평균 계산
# ============================================================

progress_mean = (
    sphc.groupby(["LOT_NO", "progress_bin"])[selected_sensors].mean().reset_index()
)


# 그래프에서는 각 bin 가운데 값을 사용
progress_mean["progress_pct"] = progress_mean["progress_bin"] + 0.5


# ============================================================
# 9. 결과 저장 폴더
# ============================================================

FIGURE_DIR = PROJECT_ROOT / "figures" / "sphc_progress"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(exist_ok=True)


# ============================================================
# 10. 센서별 LOT 진행률 그래프
# ============================================================

for sensor in selected_sensors:

    plt.figure(figsize=(12, 6))

    for lot_no, lot_data in progress_mean.groupby("LOT_NO"):

        plt.plot(lot_data["progress_pct"], lot_data[sensor], label=str(lot_no))

    plt.xlabel("LOT Progress (%)")

    plt.ylabel(sensor)

    plt.title(f"{target_grade} - {sensor} by LOT Progress")

    plt.legend(title="LOT")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    save_path = FIGURE_DIR / f"{target_grade}_{sensor}_progress.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


# ============================================================
# 11. 초반 / 중반 / 후반 평균 비교
# ============================================================


def progress_section(progress):

    if progress < 20:
        return "START_0_20"

    elif progress < 80:
        return "MIDDLE_20_80"

    else:
        return "END_80_100"


sphc["section"] = sphc["progress"].apply(progress_section)


section_mean = (
    sphc.groupby(["LOT_NO", "section"])[selected_sensors].mean().reset_index()
)


print("\n" + "=" * 100)
print("[SPHC LOT별 초반 / 중반 / 후반 평균]")
print("=" * 100)

print(section_mean.round(3).to_string(index=False))


# ============================================================
# 12. 결과 저장
# ============================================================

progress_mean.to_csv(
    RESULT_DIR / "sphc_progress_mean.csv", index=False, encoding="utf-8-sig"
)

section_mean.to_csv(
    RESULT_DIR / "sphc_section_mean.csv", index=False, encoding="utf-8-sig"
)


print("\n결과 저장 완료")
print(FIGURE_DIR)
