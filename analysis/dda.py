from pathlib import Path
import sys
import pandas as pd

# ============================================================
# 0. 프로젝트 경로 설정
# ============================================================

# 현재 파일:
# WannaOne_Final/analysis/dda.py

# 프로젝트 최상위:
# WannaOne_Final/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Python이 src 폴더를 찾을 수 있도록 경로 추가
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# src 모듈 import
from src.data_loader import find_file, load_data

# ============================================================
# 1. 분석 대상 파일
# ============================================================

TARGET_FILES = {
    "온도": "T-CR1-CAL01_온도",
    "조업이벤트": "G-02_조업이벤트",
    "정기수리": "G-01_정기수리캘린더",
}


# ============================================================
# 2. 기본 DDA 함수
# ============================================================


def basic_dda(name, df, path):

    print("\n")
    print("=" * 100)
    print(f"[ {name} 데이터 ]")
    print("=" * 100)

    # --------------------------------------------------------
    # 파일 정보
    # --------------------------------------------------------

    print("\n[1] 파일 정보")

    print("파일명 :", path.name)
    print("파일경로 :", path)

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    print("\n[2] 데이터 크기")

    print(f"행 개수 : {df.shape[0]:,}")
    print(f"열 개수 : {df.shape[1]:,}")

    print("Shape :", df.shape)

    # --------------------------------------------------------
    # 컬럼명
    # --------------------------------------------------------

    print("\n[3] 컬럼명")

    for i, column in enumerate(df.columns, start=1):

        print(f"{i:>3}. {column}")

    # --------------------------------------------------------
    # 데이터 타입
    # --------------------------------------------------------

    print("\n[4] 데이터 타입")

    print(df.dtypes)

    # --------------------------------------------------------
    # 결측치
    # --------------------------------------------------------

    print("\n[5] 결측치")

    missing_count = df.isna().sum()

    missing_ratio = df.isna().mean() * 100

    missing_df = pd.DataFrame(
        {"missing_count": missing_count, "missing_ratio(%)": missing_ratio.round(2)}
    )

    missing_df = missing_df.sort_values("missing_count", ascending=False)

    print(missing_df)

    # --------------------------------------------------------
    # 완전 중복
    # --------------------------------------------------------

    print("\n[6] 완전 중복 행")

    duplicate_count = df.duplicated().sum()

    print(f"중복 행 개수 : {duplicate_count:,}")

    if len(df) > 0:

        duplicate_ratio = duplicate_count / len(df) * 100

        print(f"중복 비율 : " f"{duplicate_ratio:.4f}%")

    # --------------------------------------------------------
    # 데이터 앞부분
    # --------------------------------------------------------

    print("\n[7] 데이터 앞부분")

    print(df.head())

    # --------------------------------------------------------
    # 데이터 뒷부분
    # --------------------------------------------------------

    print("\n[8] 데이터 뒷부분")

    print(df.tail())

    # --------------------------------------------------------
    # Unique 개수
    # --------------------------------------------------------

    print("\n[9] 컬럼별 고유값 개수")

    unique_count = df.nunique(dropna=False).sort_values()

    print(unique_count)

    # --------------------------------------------------------
    # 기초 통계량
    # --------------------------------------------------------

    print("\n[10] 수치형 데이터 기초 통계량")

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] > 0:

        print(numeric_df.describe().T)

    else:

        print("수치형 컬럼이 없습니다.")

    # --------------------------------------------------------
    # 문자열 / 범주형 컬럼 확인
    # --------------------------------------------------------

    print("\n[11] 문자열 / 범주형 컬럼")

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns

    if len(categorical_columns) == 0:

        print("문자열/범주형 컬럼이 없습니다.")

    else:

        for column in categorical_columns:

            print("\n" + "-" * 80)

            print(f"컬럼 : {column}")

            print(f"고유값 개수 : " f"{df[column].nunique(dropna=False)}")

            # 고유값이 너무 많으면 상위 10개만
            print(df[column].value_counts(dropna=False).head(10))

    # --------------------------------------------------------
    # 메모리 사용량
    # --------------------------------------------------------

    print("\n[12] 메모리 사용량")

    memory_mb = df.memory_usage(deep=True).sum() / 1024**2

    print(f"{memory_mb:.2f} MB")


# ============================================================
# 3. 메인 실행
# ============================================================


def main():

    print("\n")
    print("#" * 100)
    print("연속소둔로 프로젝트 - 1차 DDA")
    print("#" * 100)

    print("\n프로젝트 경로:")
    print(PROJECT_ROOT)

    print("\n분석 대상:")

    for name, keyword in TARGET_FILES.items():

        print(f"- {name}: {keyword}")

    # 데이터 저장용
    datasets = {}

    # --------------------------------------------------------
    # 각 데이터 파일 DDA 실행
    # --------------------------------------------------------

    for name, keyword in TARGET_FILES.items():

        print("\n")
        print("#" * 100)
        print(f"{name} 데이터 로드")
        print("#" * 100)

        try:

            # 파일 찾기
            path = find_file(keyword)

            print(f"찾은 파일: {path.name}")

            # 데이터 로드
            df = load_data(path)

            # 딕셔너리에 저장
            datasets[name] = df

            # DDA
            basic_dda(name=name, df=df, path=path)

        except Exception as e:

            print(f"\n[ERROR] " f"{name} 데이터를 처리하는 중 " f"오류가 발생했습니다.")

            print("오류 내용:", e)

    # --------------------------------------------------------
    # 데이터 전체 요약
    # --------------------------------------------------------

    print("\n")
    print("=" * 100)
    print("[전체 데이터 요약]")
    print("=" * 100)

    summary = []

    for name, df in datasets.items():

        summary.append(
            {
                "데이터": name,
                "행": len(df),
                "열": df.shape[1],
                "결측치": int(df.isna().sum().sum()),
                "중복행": int(df.duplicated().sum()),
            }
        )

    if len(summary) > 0:

        summary_df = pd.DataFrame(summary)

        print(summary_df.to_string(index=False))

    print("\n")
    print("#" * 100)
    print("1차 DDA 완료")
    print("#" * 100)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()

# 연속소둔로 온도 데이터는 약 34.5만 건의 시계열 데이터로 구성되며 결측률과 중복 문제가 크지 않았다. 조업이벤트에는 강종변경 정보가 존재하여 시간 기준으로 온도 데이터와 연결할 수 있는 가능성이 확인되었다. 향후 LOT와 강종변경 시점의 관계, 데이터 시간 공백, CR1 이벤트 구조를 추가 확인한 뒤 강종별 정상 온도 기준의 차이를 분석한다.
