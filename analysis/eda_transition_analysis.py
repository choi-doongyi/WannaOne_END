import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = PROJECT_ROOT / "results"

section = pd.read_csv(RESULT_DIR / "sphc_section_mean.csv")


# ============================================================
# 1. long → wide
# ============================================================

sensors = ["TC-OUT1", "TC-OUT2", "OP-Z3", "OP-Z4"]

wide = section.pivot(index="LOT_NO", columns="section", values=sensors)


# ============================================================
# 2. 변화량 계산
# ============================================================

results = []

for lot_no in wide.index:

    for sensor in sensors:

        start = wide.loc[lot_no, (sensor, "START_0_20")]

        middle = wide.loc[lot_no, (sensor, "MIDDLE_20_80")]

        end = wide.loc[lot_no, (sensor, "END_80_100")]

        results.append(
            {
                "LOT_NO": lot_no,
                "sensor": sensor,
                "START": start,
                "MIDDLE": middle,
                "END": end,
                # 초반이 중간 안정구간과 얼마나 다른지
                "START_vs_MIDDLE": start - middle,
                # 후반이 중간과 얼마나 다른지
                "END_vs_MIDDLE": end - middle,
                # LOT 전체 방향성
                "END_vs_START": end - start,
                # 세 구간 평균 중 최대 차이
                "section_range": (max(start, middle, end) - min(start, middle, end)),
            }
        )


result = pd.DataFrame(results)


# ============================================================
# 3. 결과 출력
# ============================================================

print("\n" + "=" * 100)
print("[SPHC 구간별 변화량]")
print("=" * 100)

print(result.round(3).to_string(index=False))


# ============================================================
# 4. 변화가 큰 LOT / 센서
# ============================================================

print("\n" + "=" * 100)
print("[초반 변화가 큰 순서]")
print("=" * 100)

print(
    result.assign(abs_shift=lambda x: x["START_vs_MIDDLE"].abs())
    .sort_values("abs_shift", ascending=False)
    .head(15)
    .round(3)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("[후반 변화가 큰 순서]")
print("=" * 100)

print(
    result.assign(abs_shift=lambda x: x["END_vs_MIDDLE"].abs())
    .sort_values("abs_shift", ascending=False)
    .head(15)
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 5. 저장
# ============================================================

result.to_csv(
    RESULT_DIR / "sphc_transition_analysis.csv", index=False, encoding="utf-8-sig"
)

print("\n저장 완료")
