# ============================================================
# build_features_v25.py
#
# V2.5 Trend Feature Builder
#
# 기존:
# casting_train_features_v22.csv
# casting_validation_features_v22.csv
#
# 추가:
# 핵심 센서 4개에 대한
#
# 1. 시간당 변화량
#    - 1H_TREND_PER_H
#    - 3H_TREND_PER_H
#    - 6H_TREND_PER_H
#    - 12H_TREND_PER_H
#
# 2. 최근 평균 vs 장기 평균 차이
#    - 1H_MINUS_3H_MEAN
#    - 3H_MINUS_6H_MEAN
#    - 6H_MINUS_12H_MEAN
#
# 총 4센서 × 7개 = 28 Feature
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 경로
# ============================================================

ROOT = Path(".")

OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "casting_train_features_v22.csv"


VALID_PATH = OUTPUT_DIR / "casting_validation_features_v22.csv"


TRAIN_OUTPUT_PATH = OUTPUT_DIR / "casting_train_features_v25.csv"


VALID_OUTPUT_PATH = OUTPUT_DIR / "casting_validation_features_v25.csv"


# ============================================================
# 1. 추세 분석 대상 핵심 센서
# ============================================================

TREND_SENSORS = [
    "CONTROL_DELAY",
    "MLD-LVL",
    "OSC-STK",
    "OSC-FRQ",
]


# ============================================================
# 2. 시간창
# ============================================================

WINDOW_HOURS = {
    "1H": 1.0,
    "3H": 3.0,
    "6H": 6.0,
    "12H": 12.0,
}


# ============================================================
# 3. Feature 생성 함수
# ============================================================


def add_trend_features(df):

    df = df.copy()

    # ========================================================
    # 센서별 추세 Feature
    # ========================================================

    for sensor in TREND_SENSORS:

        # ====================================================
        # 1. 시간당 변화량
        #
        # 기존 CHANGE:
        # 마지막 값 - 첫 값
        #
        # 이를 시간으로 나눠서
        # 단위 시간당 변화량으로 변환
        # ====================================================

        for window, hours in WINDOW_HOURS.items():

            change_col = f"{window}_" f"{sensor}_CHANGE"

            new_col = f"{window}_" f"{sensor}_TREND_PER_H"

            if change_col not in df.columns:

                raise KeyError(f"필수 컬럼 없음: " f"{change_col}")

            df[new_col] = df[change_col] / hours

        # ====================================================
        # 2. 단기 평균 - 장기 평균
        #
        # 최근 구간이
        # 과거 긴 구간보다 얼마나 상승/하락했는지
        # ====================================================

        comparisons = [
            ("1H", "3H"),
            ("3H", "6H"),
            ("6H", "12H"),
        ]

        for short_window, long_window in comparisons:

            short_col = f"{short_window}_" f"{sensor}_MEAN"

            long_col = f"{long_window}_" f"{sensor}_MEAN"

            new_col = f"{sensor}_" f"{short_window}_MINUS_" f"{long_window}_MEAN"

            if short_col not in df.columns:

                raise KeyError(f"필수 컬럼 없음: " f"{short_col}")

            if long_col not in df.columns:

                raise KeyError(f"필수 컬럼 없음: " f"{long_col}")

            df[new_col] = df[short_col] - df[long_col]

    return df


# ============================================================
# 4. 생성된 V2.5 Feature 목록
# ============================================================


def get_v25_feature_names():

    features = []

    for sensor in TREND_SENSORS:

        # ----------------------------------------------------
        # 시간당 변화
        # ----------------------------------------------------

        for window in WINDOW_HOURS:

            features.append(f"{window}_" f"{sensor}_TREND_PER_H")

        # ----------------------------------------------------
        # Window 평균 차이
        # ----------------------------------------------------

        features.extend(
            [
                (f"{sensor}_" f"1H_MINUS_3H_MEAN"),
                (f"{sensor}_" f"3H_MINUS_6H_MEAN"),
                (f"{sensor}_" f"6H_MINUS_12H_MEAN"),
            ]
        )

    return features


# ============================================================
# 5. Dataset 검사
# ============================================================


def check_dataset(old_df, new_df, name):

    v25_features = get_v25_feature_names()

    print()

    print("=" * 80)

    print(f"{name} V2.5 CHECK")

    print("=" * 80)

    print("기존 shape:", old_df.shape)

    print("V2.5 shape:", new_df.shape)

    print("추가 Feature 수:", len(v25_features))

    print()

    for feature in v25_features:

        print(f"{feature:<45}", "→", feature in new_df.columns)

    # ========================================================
    # 예상 컬럼 수 검사
    # ========================================================

    expected_cols = len(old_df.columns) + len(v25_features)

    if len(new_df.columns) != expected_cols:

        raise RuntimeError(
            f"{name}: "
            f"예상 컬럼 "
            f"{expected_cols}개인데 "
            f"실제 "
            f"{len(new_df.columns)}개"
        )

    # ========================================================
    # 무한대 값 검사
    # ========================================================

    trend_values = new_df[v25_features].select_dtypes(include=[np.number])

    inf_count = np.isinf(trend_values).sum().sum()

    print()

    print("Inf 개수:", int(inf_count))

    if inf_count > 0:

        raise RuntimeError(f"{name}: " f"Inf 값 발생")

    # ========================================================
    # Feature 요약
    # ========================================================

    print()

    print("추가 Feature 통계")

    print(new_df[v25_features].describe().T.round(4).to_string())


# ============================================================
# 6. 안전 저장
# ============================================================


def safe_save(df, path):

    temp_path = path.parent / (path.stem + "_TEMP.csv")

    df.to_csv(temp_path, index=False, encoding="utf-8-sig")

    check = pd.read_csv(temp_path, nrows=5)

    if len(check.columns) != len(df.columns):

        raise RuntimeError(f"{path.name}: " f"저장 컬럼 불일치")

    if path.exists():

        path.unlink()

    temp_path.rename(path)


# ============================================================
# 7. MAIN
# ============================================================


def main():

    print("=" * 80)

    print("CASTING FEATURE BUILDER V2.5")

    print("TREND FEATURE ENGINEERING")

    print("=" * 80)

    # ========================================================
    # 데이터 읽기
    # ========================================================

    train_df = pd.read_csv(TRAIN_PATH)

    valid_df = pd.read_csv(VALID_PATH)

    print()

    print("Train:", train_df.shape)

    print("Validation:", valid_df.shape)

    # ========================================================
    # Feature 추가
    # ========================================================

    train_v25 = add_trend_features(train_df)

    valid_v25 = add_trend_features(valid_df)

    # ========================================================
    # 검사
    # ========================================================

    check_dataset(train_df, train_v25, "TRAIN")

    check_dataset(valid_df, valid_v25, "VALIDATION")

    # ========================================================
    # 저장
    # ========================================================

    safe_save(train_v25, TRAIN_OUTPUT_PATH)

    safe_save(valid_v25, VALID_OUTPUT_PATH)

    # ========================================================
    # 완료
    # ========================================================

    print()

    print("=" * 80)

    print("V2.5 FEATURE 생성 완료")

    print("=" * 80)

    print()

    print("Train:", train_v25.shape)

    print("Validation:", valid_v25.shape)

    print()

    print("추가 Feature:", len(get_v25_feature_names()))

    print()

    print("저장:", TRAIN_OUTPUT_PATH)

    print("저장:", VALID_OUTPUT_PATH)


# ============================================================
# 8. 실행
# ============================================================

if __name__ == "__main__":

    main()
