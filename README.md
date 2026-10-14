# 연주기 다변량 시계열 기반 24시간 조기경보 및 6시간 임박고장 예측

## 1. 프로젝트 개요

본 프로젝트는 연속주조(연주) 설비의 다변량 센서 시계열 데이터를 활용하여 고장 위험을 조기에 탐지하고, 실제 운영에 적용 가능한 단계형 경보 시스템을 구성하는 것을 목표로 한다.

단순히 고장/정상을 한 번에 분류하는 대신 고장까지 남은 시간에 따라 경보를 두 단계로 구분하였다.

- **CAUTION**: 24시간 이내 고장 위험
- **DANGER**: 6시간 이내 임박 고장 위험
- **NORMAL**: 위 조건에 해당하지 않는 상태

최종적으로 24H 모델과 6H 모델의 예측 결과를 통합하고, 경보 상태가 지나치게 자주 바뀌는 문제를 완화하기 위해 State Machine을 적용하였다.

---

## 2. 데이터 구성

전체 데이터는 360개의 독립 시나리오로 구성되어 있으며, 각 시나리오는 약 72시간의 설비 상태를 포함한다.

| 구분 | 시나리오 수 | 용도 |
|---|---:|---|
| Train | 216 | 학습, 교차검증, Feature/Model/Threshold 결정 |
| Validation | 72 | 최종 성능 평가 |
| Test | 72 | 최종 예측 생성 |

Test 데이터에는 정답을 사용하지 않았으며, Test 성능 수치는 계산하지 않았다.

### 대상 설비

- 설비: 연주기
- Equipment Code: `P1-CC1-INS01`
- Raw Source: `casting`
- 기본 Sampling Interval: 5초

### 사용 센서

- `LOAD`
- `WD-SPD`
- `WD-LOD`
- `OSC-FRQ`
- `OSC-STK`
- `MLD-LVL`
- `CONTROL_DELAY`
- `RUNNING`

---

## 3. 시간 데이터 품질 점검

원본 센서의 실제 시간 간격을 전체 360개 시나리오에 대해 검사하였다.

| 항목 | 결과 |
|---|---:|
| 전체 시간 간격 | 18,660,960 |
| 정확히 5초 | 18,659,880 (99.994213%) |
| 중복 0초 | 0 |
| 0~5초 미만 | 0 |
| 5초 초과 | 1,080 (0.005787%) |
| 음수 간격 | 0 |
| 최대 Gap | 10초 |

5초를 초과한 모든 간격은 10초였으며, 전체에서 차지하는 비율이 매우 낮았다.

따라서 별도의 5초 단위 강제 Resampling 또는 Interpolation은 수행하지 않고 실제 관측값을 그대로 사용하였다.

---

## 4. 전처리

전처리 과정은 다음 기준을 적용하였다.

- `QUALITY_FLAG == "ok"` 데이터만 사용
- 시간 컬럼을 UTC 기준 datetime으로 변환
- 시간 변환 실패값 제거
- `MEAS_DT` 기준 시간순 정렬
- 예측 시점보다 미래의 데이터는 사용하지 않음
- 센서 결측이 존재할 경우 통계 Feature는 NaN으로 유지
- 모델 입력 전 Median Imputation 적용
- 교차검증에서는 Imputer를 각 Fold의 Train 데이터에만 Fit

### 데이터 누수 방지

각 예측 시점 `OBSERVED_AT`에 대해 다음 조건을 만족하는 데이터만 사용하였다.

```python
MEAS_DT <= OBSERVED_AT
```

즉, 미래 센서값이나 미래 정비 정보는 Feature로 사용하지 않았다.

---

## 5. Feature Engineering

예측 시점마다 현재값과 과거 Window 통계 Feature를 생성하였다.

### Current Feature

- `CUR_LOAD`
- `CUR_WD-SPD`
- `CUR_WD-LOD`
- `CUR_OSC-FRQ`
- `CUR_OSC-STK`
- `CUR_MLD-LVL`
- `CUR_CONTROL_DELAY`
- `CUR_RUNNING`

### Window

- 1H
- 3H
- 6H
- 12H

### Window Statistics

각 센서별로 다음 통계량을 생성하였다.

- Mean
- Standard Deviation
- Min
- Max
- Range
- Median
- Change

추가로 각 Window에 대해 `ROW_COUNT`를 생성하였다.

### Mean Difference Feature

24H 모델에서는 일부 센서에 대해 장기/단기 평균 변화량을 추가하였다.

사용 센서:

- `CONTROL_DELAY`
- `MLD-LVL`
- `OSC-STK`
- `OSC-FRQ`

예시:

```text
1H_MEAN - 3H_MEAN
3H_MEAN - 6H_MEAN
6H_MEAN - 12H_MEAN
```

총 12개의 Mean Difference Feature가 추가되었다.

---

## 6. 학습 및 검증 전략

최종 개발 절차에서는 Validation 결과를 보고 Feature, Model, Threshold를 다시 수정하지 않도록 하였다.

### Train 단계

Train 데이터 내부에서 `StratifiedGroupKFold` 5-Fold 교차검증을 사용하였다.

Scenario 단위로 Group을 분리하여 동일 Scenario가 Train/Valid Fold에 동시에 들어가지 않도록 하였다.

Train CV에서 다음 항목을 결정하였다.

- Feature Set
- Model
- Threshold

### Validation 단계

위 설정을 모두 고정한 뒤 Validation 데이터에 한 번 적용하여 최종 성능을 확인하였다.

### Test 단계

Validation 이후 Model, Feature, Threshold, Hold Time을 변경하지 않고 Test 예측을 생성하였다.

---

## 7. 24H CAUTION 모델

### 목적

24시간 이내 고장 위험을 조기에 탐지하여 점검, 정비 준비, 부품 확인 등 사전 대응 시간을 확보한다.

### 최종 설정

- Model: **XGBoost**
- Feature Set: **SENSOR + MEAN_DIFF**
- Feature Count: **249**
- Threshold: **0.10**

### Train OOF

| Metric | Score |
|---|---:|
| PR-AUC | 0.8163 |
| ROC-AUC | 0.9296 |
| Precision | 0.6233 |
| Recall | 0.8387 |
| F1 | 0.7151 |
| F2 | 0.7845 |

Confusion Matrix:

```text
TN = 1592
FP = 110
FN = 35
TP = 182
```

### Final Validation

| Metric | Score |
|---|---:|
| PR-AUC | 0.7779 |
| ROC-AUC | 0.9190 |
| Precision | 0.5283 |
| Recall | 0.8000 |
| F1 | 0.6364 |
| F2 | 0.7254 |

Confusion Matrix:

```text
TN = 519
FP = 50
FN = 14
TP = 56
```

---

## 8. 6H DANGER 모델

### 목적

6시간 이내 임박 고장을 탐지하여 즉시 점검 또는 정비 우선순위 상향이 필요한 상태를 구분한다.

6H Target은 양성 비율이 매우 낮은 불균형 문제를 갖는다.

Train 기준:

```text
Usable Samples = 2,508
Positive = 52
Positive Ratio ≈ 2.07%
```

### 최종 설정

- Model: **XGBoost**
- Feature Set: **SENSOR ONLY**
- Feature Count: **237**
- Threshold: **0.03**

Threshold는 Train OOF 기준으로 Recall 0.75 이상을 만족하는 후보 중 Precision을 우선하여 선택하였다.

### Train OOF

| Metric | Score |
|---|---:|
| PR-AUC | 0.4340 |
| ROC-AUC | 0.8861 |
| Precision | 0.2335 |
| Recall | 0.7500 |
| F1 | 0.3562 |
| F2 | 0.5200 |

Confusion Matrix:

```text
TN = 2328
FP = 128
FN = 13
TP = 39
```

### Final Validation

| Metric | Score |
|---|---:|
| PR-AUC | 0.4474 |
| ROC-AUC | 0.9199 |
| Precision | 0.2766 |
| Recall | 0.8125 |
| F1 | 0.4127 |
| F2 | 0.5856 |

Confusion Matrix:

```text
TN = 787
FP = 34
FN = 3
TP = 13
```

---

## 9. 통합 경보 시스템

두 모델의 확률을 다음 기준으로 통합하였다.

```text
P(6H) >= 0.03
    → DANGER

else if P(24H) >= 0.10
    → CAUTION

else
    → NORMAL
```

우선순위는 다음과 같다.

```text
DANGER > CAUTION > NORMAL
```

---

## 10. State Machine

Raw Prediction은 시점마다 확률이 변하면서 경보 상태가 자주 바뀔 수 있다.

이를 완화하기 위해 Hold Rule을 적용하였다.

- CAUTION Hold: 6H
- DANGER Hold: 6H

### Train OOF

```text
RAW transitions   = 268
STATE transitions = 229
Reduction         = 14.55%
```

### Validation

```text
RAW transitions   = 126
STATE transitions = 112
Reduction         = 11.11%
```

### Test

```text
RAW transitions   = 110
STATE transitions = 86
Reduction         = 21.82%
```

이 수치는 오경보 감소율이 아니라 **경보 상태 전환 횟수 감소율**이다.

---

## 11. Final Test Prediction

### Data Check

```text
Train Shape : (2808, 256)
Test Shape  : (936, 245)

Train Scenario : 216
Test Scenario  : 72
```

### Test Warning Distribution

| System | Level | Count | Ratio |
|---|---|---:|---:|
| RAW | NORMAL | 786 | 0.8397 |
| RAW | CAUTION | 81 | 0.0865 |
| RAW | DANGER | 69 | 0.0737 |
| STATE_MACHINE | NORMAL | 753 | 0.8045 |
| STATE_MACHINE | CAUTION | 92 | 0.0983 |
| STATE_MACHINE | DANGER | 91 | 0.0972 |

### Scenario Summary

```text
Total Scenario       : 72
NORMAL only Scenario : 47
CAUTION max Scenario : 9
DANGER Scenario      : 16
```

Test 정답은 사용하지 않았으므로 정확도, Recall, PR-AUC 등의 Test 성능 수치는 계산하지 않는다.

---

## 12. 시스템 동작 예시

```text
센서 데이터 입력
        ↓
과거 1H / 3H / 6H / 12H Feature 생성
        ↓
24H XGBoost ──→ 24H Risk Probability
        ↓
6H XGBoost  ──→ 6H Risk Probability
        ↓
경보 Rule 적용
        ↓
NORMAL / CAUTION / DANGER
        ↓
State Machine
        ↓
최종 운영 경보
```

운영 관점에서는 다음과 같이 활용할 수 있다.

```text
NORMAL
→ 정상 운전

CAUTION
→ 24시간 내 위험 가능
→ 설비 점검 / 정비 준비 / 부품 확인

DANGER
→ 6시간 내 임박 위험 가능
→ 즉시 점검 / 정비 우선순위 상승
```

---

## 13. 주요 결과

1. 24H 조기경보와 6H 임박경보를 별도 모델로 구성하였다.
2. 24H 모델은 Validation Recall 0.8000을 기록하였다.
3. 6H 모델은 Validation Recall 0.8125를 기록하였다.
4. 6H Target의 매우 낮은 양성 비율에도 불구하고 임박고장 탐지 성능을 확보하였다.
5. State Machine 적용으로 Validation 경보 전환 횟수를 11.11% 감소시켰다.
6. Test 72개 시나리오에 최종 고정 모델을 적용하여 운영형 3단계 경보 결과를 생성하였다.
7. 원본 센서의 99.9942%가 정확한 5초 Sampling Interval을 유지함을 확인하였다.

---

## 14. 한계

- 합성 데이터 기반이므로 실제 설비 환경과 분포 차이가 존재할 수 있다.
- 6H Target의 Positive Sample이 적어 성능 변동 가능성이 크다.
- Test Label이 제공되지 않아 Test 성능을 직접 평가할 수 없다.
- State Machine은 경보 안정화를 위한 운영 규칙이며 자체적으로 예측 정확도를 향상시키는 모델은 아니다.
- 현재 최종 재구성 결과에는 Event-level Detection Rate 및 Lead Time 분석이 포함되어 있지 않다.

---

## 15. 향후 개선 방향

- Event-level 고장 탐지율 및 Lead Time 평가
- 실제 설비 데이터 기반 재학습 및 Threshold 재보정
- 설비별 Probability Calibration
- False Alarm Episode 기반 운영 평가
- 경보별 점검/정비 Action Recommendation 연결
- 실시간 Dashboard 및 모니터링 서비스 구현

---

## 16. 최종 요약

본 프로젝트는 단일 고장 분류 모델이 아니라,

> **24시간 조기경보 + 6시간 임박경보 + State Machine**

구조를 통해 실제 설비 운영에 적용 가능한 단계형 예지보전 경보 시스템을 설계하였다.

최종 시스템은 다음 상태를 제공한다.

```text
NORMAL
   ↓
CAUTION
   ↓
DANGER
```

이를 통해 단순한 고장 여부 예측을 넘어, 고장까지 남은 시간에 따라 대응 우선순위를 구분할 수 있도록 구성하였다.
