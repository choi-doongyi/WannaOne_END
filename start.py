from pathlib import Path
import pandas as pd

# =========================
# 1. 경로 설정
# =========================
LABEL_DIR = Path("labels")

files = sorted(LABEL_DIR.glob("*.csv"))

print(f"[INFO] label 파일 수: {len(files)}")


# =========================
# 2. 전체 label 파일 불러오기
# =========================
dfs = []

use_cols = [
    "SCENARIO_ID",
    "EQP_CD",
    "OBSERVED_AT",
    "RISK_6H",
    "RISK_24H",
    "CENSORED_6H",
    "CENSORED_24H",
]

for f in files:
    df = pd.read_csv(f, usecols=use_cols)
    dfs.append(df)

labels = pd.concat(dfs, ignore_index=True)

print(f"[INFO] 전체 shape: {labels.shape}")
print(labels.head())


# =========================
# 3. 기본 타입 정리
# =========================
labels["OBSERVED_AT"] = pd.to_datetime(labels["OBSERVED_AT"], errors="coerce")

numeric_cols = ["RISK_6H", "RISK_24H", "CENSORED_6H", "CENSORED_24H"]

for col in numeric_cols:
    labels[col] = pd.to_numeric(labels[col], errors="coerce")


# =========================
# 4. 6시간 위험 요약
# =========================
# 검열되지 않은 샘플만 사용
risk6 = labels[labels["CENSORED_6H"] == 0].copy()

summary6 = risk6.groupby("EQP_CD")["RISK_6H"].agg(
    유효표본="count", 고장양성="sum", 고장비율="mean"
)

# 양성 시나리오 개수
positive_scenarios_6h = (
    risk6[risk6["RISK_6H"] == 1]
    .groupby("EQP_CD")["SCENARIO_ID"]
    .nunique()
    .rename("양성시나리오수")
)

summary6 = summary6.join(positive_scenarios_6h).fillna({"양성시나리오수": 0})

summary6["고장비율(%)"] = summary6["고장비율"] * 100

summary6 = summary6[
    ["유효표본", "고장양성", "고장비율(%)", "양성시나리오수"]
].sort_values(["고장양성", "양성시나리오수"], ascending=False)


# =========================
# 5. 24시간 위험 요약
# =========================
risk24 = labels[labels["CENSORED_24H"] == 0].copy()

summary24 = risk24.groupby("EQP_CD")["RISK_24H"].agg(
    유효표본="count", 고장양성="sum", 고장비율="mean"
)

positive_scenarios_24h = (
    risk24[risk24["RISK_24H"] == 1]
    .groupby("EQP_CD")["SCENARIO_ID"]
    .nunique()
    .rename("양성시나리오수")
)

summary24 = summary24.join(positive_scenarios_24h).fillna({"양성시나리오수": 0})

summary24["고장비율(%)"] = summary24["고장비율"] * 100

summary24 = summary24[
    ["유효표본", "고장양성", "고장비율(%)", "양성시나리오수"]
].sort_values(["고장양성", "양성시나리오수"], ascending=False)


# =========================
# 6. 전체 시나리오 개수도 같이 확인
# =========================
scenario_count = (
    labels.groupby("EQP_CD")["SCENARIO_ID"].nunique().rename("전체시나리오수")
)

summary6 = summary6.join(scenario_count)
summary24 = summary24.join(scenario_count)


# =========================
# 7. 결과 출력
# =========================
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("\n" + "=" * 80)
print("6시간 고장 위험 요약")
print("=" * 80)
print(summary6.round(3))

print("\n" + "=" * 80)
print("24시간 고장 위험 요약")
print("=" * 80)
print(summary24.round(3))


# =========================
# 8. 연주기 계열만 따로 보기
# =========================
# EQP_CD 이름에 연주기 코드가 어떻게 들어가는지 모를 경우
# 아래처럼 전체 코드부터 확인
print("\n[설비 코드 목록]")
print(sorted(labels["EQP_CD"].dropna().unique()))

# 예시:
# casting_codes = ["CC1", "CC2"]
# print(summary24.loc[casting_codes])


# =========================
# 9. CSV 저장
# =========================
summary6.to_csv("risk_6h_equipment_summary.csv", encoding="utf-8-sig")

summary24.to_csv("risk_24h_equipment_summary.csv", encoding="utf-8-sig")

print("\n[INFO] 저장 완료")
print("- risk_6h_equipment_summary.csv")
print("- risk_24h_equipment_summary.csv")
