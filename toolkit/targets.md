# pdm-v2.1 예측정비 목표와 정답 사용법

360개 독립 72시간 시나리오: 학습 216개, 검증 72개, 테스트 72개(시작 시각순). 연주 설비가 주요 열화 대상인 시나리오는 120개이며 모든 시나리오에 A~G 관측이 포함됩니다.

## 세 가지 정답

| 경로 | 목표 | 결합 기준 |
|---|---|---|
| labels/Sxxxx.csv | 추가 정비가 없을 때 6·24시간 내 고장 여부 | SCENARIO_ID, EQP_CD, OBSERVED_AT |
| diagnostics/Sxxxx.csv | 정상·열화·고장·정비 상태와 이상 원인 부품·모드 | SCENARIO_ID, EQP_CD, START_AT ≤ MEAS_DT < END_AT |
| targets/Sxxxx.csv | 잔여수명, 검열, 유한 기간 비용 기준 교체 시점 | SCENARIO_ID, EQP_CD, OBSERVED_AT |

학습·검증 정답만 포함됩니다. 이 세 폴더의 열과 원자료의 `SCENARIO_ID`·`SLAB_NO` 같은 식별자는 예측 특징량에서 제외하세요. `QUALITY_FLAG`는 측정 품질이며 고장 라벨이 아닙니다.

`RISK_6H`·`RISK_24H`는 해당 기간 안의 고장 여부(1/0)입니다. 관측 종료까지 평가 기간이 부족하거나 설비가 고장·정비 중이면 위험 라벨은 공란이고 해당 `CENSORED_6H`·`CENSORED_24H`가 1입니다.

진단 구간은 `START_AT ≤ MEAS_DT < END_AT`에 적용합니다. `normal`: 모든 부품 열화도 0.35 미만, `degraded`: 한 부품 이상 0.35 이상, `failed`: 고장, `maintenance`: 정지 정비 중. `normal`과 일부 정비 구간에서는 원인 부품·모드가 공란일 수 있습니다.

| 설비 | FAULT_MODE |
|---|---|
| A 소둔로 | heater-loss: 히터 성능 저하, sensor-drift: 온도 센서 드리프트 |
| B 용접기 | electrode-wear: 전극 마모, power-loss: 전원 성능 저하 |
| C 연주 | friction: 진동 구동 마찰, sticking: 밸브 고착, control-delay: 밸브 제어 지연 |
| D 조질압연 | roll-wear: 롤 마모, bearing-damage: 베어링 손상 |
| E 배풍기 | imbalance: 불평형, bearing-damage: 베어링 손상, looseness: 체결부 풀림 |
| F 교반기 | deposit: 부착, wear: 마모, rotor-damage: 회전체 손상 |
| 공통 | background-wear: 주 고장 원인 이외 부품의 완만한 열화 |

`posko.py`로 원자료에 구간 라벨을 붙일 수 있습니다. 반환되는 `GT_*` 열은 정답입니다.

```python
from posko import source_rows, supervision_rows, annotate_diagnosis
root = '../'  # toolkit 폴더에서 실행할 때의 상위 데이터 폴더
diagnosis = supervision_rows(root, 'S0001', 'diagnostics')
rows = annotate_diagnosis(source_rows(root, 'S0001', 'casting'), diagnosis)
first_labeled_row = next(rows)
```

## 잔여수명과 검열

- `AT_RISK=1`: 현재 운전 상태에서 신규 고장을 평가할 수 있습니다. 고장·정비 중이거나 관측 종료 시점이면 0입니다.
- `RUL_EVENT=1`: 현재 상태에서 추가 정비를 하지 않을 때 72시간 관측 종료 전에 고장이 발생합니다. `RUL_HOURS`가 그 고장까지의 시간이며 `FOLLOWUP_HOURS`와 같습니다.
- `RUL_EVENT=0`: 관측 종료까지 고장이 없습니다. `RUL_HOURS`는 공란이고 `FOLLOWUP_HOURS`는 남은 관측 시간입니다. 잔여수명을 0 또는 72시간으로 채우지 마세요.
- `AT_RISK=0`: 사건 여부와 잔여수명은 공란입니다. 정비 완료 후 새 관측에서 다시 판단합니다.

RUL은 예측 시각까지의 정비를 반영하고 이후 추가 정비가 없다고 가정합니다. 향후 시행된 예방 정비와 무관하게 계산됩니다. 사건이 관측된 표본만으로 회귀 오차를 계산할 수 있으며, 검열을 함께 활용하려면 생존 분석 또는 검열 목적함수가 필요합니다.

## 정비 시점 정답

정비 시점 목표는 예측 시각부터 72시간 관측 종료까지 아래 비용 점수를 최소화한 정책입니다.

- 후보: 대기 또는 현재부터 6시간 간격으로 원인 부품을 한 번 교체. 관측 종료 전에 4시간 교체가 완료되어야 합니다. 고장이 발생하면 예정 교체 전후 모두 즉시 8시간 복구합니다. 복구 중 예정 교체 시각을 지나면 그 교체는 취소합니다.
- 점수: 조치 비용 + 정지시간×2 + 고장 횟수×20 + 불필요 정비 횟수×8. 교체 비용은 12, 복구 비용은 30입니다. 열화 0.35 미만 부품의 교체는 불필요 정비로 셉니다.
- `POLICY_VALID=1`: 정비팀을 사용할 수 있어 비교가 가능합니다. 현재 정비팀이 작업 중이면 0이고 정책 정답은 공란입니다.
- `POLICY_ACTION`: `wait` 또는 `replace`. `POLICY_PART`는 교체 대상, `POLICY_DELAY_H`는 관측 시점에서 교체 시작까지의 시간입니다. 대기에는 교체 시간이 없습니다.
- `POLICY_SCORE`: 선택한 정책의 추가 점수. `WAIT_SCORE`: 대기 후 고장 복구만 했을 때의 추가 점수입니다. 점수가 같으면 대기를 우선하며, 교체 후보끼리 최소 점수가 같으면 가장 늦은 시점을 택합니다.

교체 시점 회귀는 `POLICY_ACTION=replace`와 교체 부품이 알려진 표본에서 평가합니다. 전체 정책 비교에는 부품 선택·오경보·미탐지·검열·정지시간·비용을 포함하세요.

`POLICY_DELAY_H`는 최초 고장 예방 마감 시각이 아니며 `RUL_HOURS`보다 길 수 있습니다. 최초 고장 경보는 RUL과 6·24시간 위험으로 별도 평가하세요. 관측 종료 전에 시작한 복구의 비용은 전액, 정지시간은 관측된 기간까지만 셉니다.

## 재현 가능한 연주 예제

Python, NumPy, scikit-learn을 설치한 환경에서 실행합니다.

재현 환경: Python 3.13.7, NumPy 2.5.1, scikit-learn 1.9.1.

```sh
python toolkit/casting_example.py --data-root . --out casting-validation-results.json
```

예제는 현재값과 과거 12시간 특징으로 24시간 고장위험·RUL·교체 시점을 비교합니다. 공정 간 소재 흐름과 하류 품질 목표는 [연결 생산 흐름](flow.md)을 참조하세요.
