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

RESULT_DIR = PROJECT_ROOT / "results"

FIGURE_DIR = PROJECT_ROOT / "figures" / "final_candidates"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 원본 데이터
# ============================================================

temp = load_data(find_file("T-CR1-CAL01_온도"))

temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])


# ============================================================
# 2. Cycle 시간 정보
# ============================================================

cycle_info = pd.read_csv(RESULT_DIR / "tc_out1_cycle_maintenance_analysis.csv")

cycle_info["cycle_start"] = pd.to_datetime(cycle_info["cycle_start"])

cycle_info["cycle_end"] = pd.to_datetime(cycle_info["cycle_end"])


# ============================================================
# 3. 최종 후보 정의
# ============================================================

candidate_config = {
    # --------------------------------------------------------
    # 후보 A
    # --------------------------------------------------------
    "candidate_A_240091": {
        "LOT_NO": 240091,
        # Cycle 4~5가 연속 후보
        "start_cycle": 4,
        "end_cycle": 5,
        "sensors": ["TC-Z7", "TC-Z8", "TC-OUT1", "OP-Z6"],
    },
    # --------------------------------------------------------
    # 후보 B
    # --------------------------------------------------------
    "candidate_B_240121": {
        "LOT_NO": 240121,
        "start_cycle": 12,
        "end_cycle": 12,
        "sensors": ["TC-Z3", "OP-Z3", "CP-Z4", "TC-Z8", "OP-Z8"],
    },
}


# ============================================================
# 4. 후보별 분석
# ============================================================

for candidate_name, config in candidate_config.items():

    lot_no = config["LOT_NO"]

    start_cycle = config["start_cycle"]
    end_cycle = config["end_cycle"]

    sensors = config["sensors"]

    # --------------------------------------------------------
    # 해당 LOT Cycle 정보
    # --------------------------------------------------------

    lot_cycle = (
        cycle_info[cycle_info["LOT_NO"] == lot_no].sort_values("cycle_no").copy()
    )

    # --------------------------------------------------------
    # 후보 시작 / 종료시간
    # --------------------------------------------------------

    candidate_cycles = lot_cycle[
        (lot_cycle["cycle_no"] >= start_cycle) & (lot_cycle["cycle_no"] <= end_cycle)
    ]

    candidate_start = candidate_cycles["cycle_start"].min()

    candidate_end = candidate_cycles["cycle_end"].max()

    # ========================================================
    # 후보 앞뒤 Cycle까지 같이 보기
    # ========================================================

    context_start_cycle = max(1, start_cycle - 2)

    context_end_cycle = min(int(lot_cycle["cycle_no"].max()), end_cycle + 2)

    context_cycles = lot_cycle[
        (lot_cycle["cycle_no"] >= context_start_cycle)
        & (lot_cycle["cycle_no"] <= context_end_cycle)
    ]

    plot_start = context_cycles["cycle_start"].min()

    plot_end = context_cycles["cycle_end"].max()

    # --------------------------------------------------------
    # 원본 데이터 추출
    # --------------------------------------------------------

    lot_data = temp[
        (temp["LOT_NO"] == lot_no)
        & (temp["MEAS_DT"] >= plot_start)
        & (temp["MEAS_DT"] <= plot_end)
    ].copy()

    # ========================================================
    # 5. 1분 평균
    # ========================================================

    data_min = lot_data.set_index("MEAS_DT")[sensors].resample("1min").mean()

    # 짧은 결측만 보간
    data_min = data_min.interpolate(method="time", limit=2)

    # ========================================================
    # 6. 센서별 그래프
    # ========================================================

    for sensor in sensors:

        # 5분 rolling
        rolling = data_min[sensor].rolling(window=5, center=True, min_periods=3).mean()

        plt.figure(figsize=(13, 6))

        # Raw 1분 평균
        plt.plot(data_min.index, data_min[sensor], alpha=0.3, label="1-min Mean")

        # Rolling
        plt.plot(rolling.index, rolling.values, linewidth=2, label="5-min Rolling Mean")

        # ----------------------------------------------------
        # 후보구간 표시
        # ----------------------------------------------------

        plt.axvspan(
            candidate_start, candidate_end, alpha=0.2, label="Anomaly Candidate"
        )

        # ----------------------------------------------------
        # Cycle 경계 표시
        # ----------------------------------------------------

        for _, row in context_cycles.iterrows():

            plt.axvline(row["cycle_start"], linestyle="--", alpha=0.4)

        plt.xlabel("Time")

        plt.ylabel(sensor)

        plt.title(f"{candidate_name} | " f"LOT {lot_no} | " f"{sensor}")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        save_path = FIGURE_DIR / f"{candidate_name}_{sensor}.png"

        plt.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()

    # ========================================================
    # 7. 후보 Cycle 정보 출력
    # ========================================================

    print("\n" + "=" * 110)

    print(f"[{candidate_name}]")

    print("=" * 110)

    print(
        candidate_cycles[
            [
                "LOT_NO",
                "STEEL_GRADE",
                "cycle_no",
                "cycle_start",
                "cycle_end",
                "anomaly_score",
                "distance_days",
            ]
        ]
        .round({"anomaly_score": 3, "distance_days": 2})
        .to_string(index=False)
    )


print("\n시각화 저장 완료")
print(FIGURE_DIR)
