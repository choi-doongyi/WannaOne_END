from pathlib import Path
import sys
import pandas as pd

# ============================================================
# 0. 프로젝트 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import find_file, load_data

# ============================================================
# 1. 데이터 불러오기
# ============================================================

temp_path = find_file("T-CR1-CAL01_온도")
event_path = find_file("G-02_조업이벤트")

temp = load_data(temp_path)
event = load_data(event_path)


# ============================================================
# 2. 시간형 변환
# ============================================================

temp["MEAS_DT"] = pd.to_datetime(temp["MEAS_DT"])
event["EVT_DT"] = pd.to_datetime(event["EVT_DT"])


# ============================================================
# 3. CR1 강종변경 이벤트만 추출
# ============================================================

cr1_event = (
    event[(event["PLANT_CD"] == "CR1") & (event["EVT_TYPE"] == "강종변경")]
    .sort_values("EVT_DT")
    .reset_index(drop=True)
)


# ============================================================
# 4. LOT별 시간 범위 생성
# ============================================================

lot_summary = (
    temp.groupby("LOT_NO")
    .agg(
        start_time=("MEAS_DT", "min"),
        end_time=("MEAS_DT", "max"),
        n_rows=("MEAS_DT", "size"),
    )
    .reset_index()
    .sort_values("start_time")
)


# ============================================================
# 5. LOT별 강종 정보 확인
# ============================================================

results = []


for _, lot in lot_summary.iterrows():

    lot_no = lot["LOT_NO"]
    start_time = lot["start_time"]
    end_time = lot["end_time"]

    # --------------------------------------------------------
    # LOT 시작 시점 이전 또는 동일 시점의 마지막 강종변경
    # --------------------------------------------------------

    before_start = cr1_event[cr1_event["EVT_DT"] <= start_time]

    if len(before_start) > 0:

        start_event = before_start.iloc[-1]

        start_event_time = start_event["EVT_DT"]
        start_before_grade = start_event["BEF_VAL"]
        start_grade = start_event["AFT_VAL"]

        start_event_gap_sec = (start_time - start_event_time).total_seconds()

    else:

        start_event_time = pd.NaT
        start_before_grade = None
        start_grade = None
        start_event_gap_sec = None

    # --------------------------------------------------------
    # LOT 내부 강종변경 이벤트
    # --------------------------------------------------------

    inside_events = cr1_event[
        (cr1_event["EVT_DT"] >= start_time) & (cr1_event["EVT_DT"] <= end_time)
    ].copy()

    inside_event_count = len(inside_events)

    # --------------------------------------------------------
    # LOT 종료시점의 강종
    # --------------------------------------------------------

    before_end = cr1_event[cr1_event["EVT_DT"] <= end_time]

    if len(before_end) > 0:

        end_event = before_end.iloc[-1]

        end_grade = end_event["AFT_VAL"]

    else:

        end_grade = None

    # --------------------------------------------------------
    # LOT 시작시점과 이벤트가 정확히 같은지
    # --------------------------------------------------------

    exact_start_match = False

    if pd.notna(start_event_time):

        exact_start_match = start_event_time == start_time

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    results.append(
        {
            "LOT_NO": lot_no,
            "start_time": start_time,
            "end_time": end_time,
            "n_rows": lot["n_rows"],
            "event_time_before_start": start_event_time,
            "event_gap_sec": start_event_gap_sec,
            "exact_start_match": exact_start_match,
            "BEF_GRADE": start_before_grade,
            "START_GRADE": start_grade,
            "END_GRADE": end_grade,
            "events_inside_lot": inside_event_count,
        }
    )


lot_grade_summary = pd.DataFrame(results)


# ============================================================
# 6. 결과 출력
# ============================================================

print("\n" + "=" * 110)
print("[1] LOT별 강종 연결 결과")
print("=" * 110)

print(lot_grade_summary.to_string(index=False))


# ============================================================
# 7. LOT 시작시점과 강종변경 시점 일치 여부
# ============================================================

print("\n" + "=" * 110)
print("[2] LOT 시작 = 강종변경 시점 여부")
print("=" * 110)

print(
    lot_grade_summary[
        [
            "LOT_NO",
            "start_time",
            "event_time_before_start",
            "event_gap_sec",
            "exact_start_match",
            "BEF_GRADE",
            "START_GRADE",
        ]
    ].to_string(index=False)
)


# ============================================================
# 8. LOT 내부 강종변경 존재 여부
# ============================================================

print("\n" + "=" * 110)
print("[3] LOT 내부 강종변경 횟수")
print("=" * 110)

print(
    lot_grade_summary[
        ["LOT_NO", "START_GRADE", "END_GRADE", "events_inside_lot"]
    ].to_string(index=False)
)


# ============================================================
# 9. 시작 강종 분포
# ============================================================

print("\n" + "=" * 110)
print("[4] LOT별 시작 강종 분포")
print("=" * 110)

print(lot_grade_summary["START_GRADE"].value_counts(dropna=False))


# ============================================================
# 10. 정확히 일치한 LOT 개수
# ============================================================

exact_count = lot_grade_summary["exact_start_match"].sum()

print("\n" + "=" * 110)
print("[5] LOT 시작시점과 강종변경 시점 정확 일치")
print("=" * 110)

print(f"{exact_count} / " f"{len(lot_grade_summary)} LOT")


# ============================================================
# 11. LOT 내부에 강종이 2개 이상 존재하는 경우 상세 출력
# ============================================================

print("\n" + "=" * 110)
print("[6] LOT 내부 강종변경 상세")
print("=" * 110)

for _, lot in lot_summary.iterrows():

    start_time = lot["start_time"]
    end_time = lot["end_time"]

    inside_events = cr1_event[
        (cr1_event["EVT_DT"] >= start_time) & (cr1_event["EVT_DT"] <= end_time)
    ]

    if len(inside_events) > 0:

        print(f"\nLOT {lot['LOT_NO']}")

        print(inside_events[["EVT_DT", "BEF_VAL", "AFT_VAL"]].to_string(index=False))
