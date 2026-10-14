# 연결 생산 흐름

한 `FLOW_ID`는 교반기(F) → 소결 배풍기 1·2(E, 병렬) → 연주(C) → 용접(B) → 소둔(A) → 조질압연(D)의 작업을 연결합니다. G는 공정 단계가 아니라 각 설비의 정비 이력입니다. 시나리오마다 2시간 간격으로 36개 흐름이 시작됩니다. 각 작업은 2시간 슬롯을 차지하며 설비 정지 또는 앞 작업 지연 시 다음 가용 슬롯으로 이동합니다. 관측 종료 전에 끝나지 않은 흐름은 미완료로 표시합니다.

## 파일과 결합

| 경로 | 내용 | 결합 키 |
|---|---|---|
| `flow/operations/Sxxxx.csv` | 작업 순서, 설비, 시작·종료, 원자료 레코드 ID, 대기시간 | `SCENARIO_ID`, `FLOW_ID`, `STAGE` |
| `flow/inspections/Sxxxx.csv` | F·E·C·B·A 작업 종료 후 품질 지표 | `SCENARIO_ID`, `FLOW_ID`, `STAGE`, `OBSERVED_AT` |
| `flow/quality/Sxxxx.csv` | 단계별 품질·최종 결함·원인 설비 정답(학습·검증만) | `SCENARIO_ID`, `FLOW_ID`, `STAGE` |

`operations`의 `SOURCE_ID`는 작업 시작 표본의 F·D `BURST_ID`, E `SESS_ID`, C `SLAB_NO`, B `COIL_ID`, A `LOT_NO`와 연결합니다. 작업 구간의 전체 관측은 `SCENARIO_ID`, `EQP_CD`, `START_AT ≤ MEAS_DT < END_AT`으로 연결하세요. 두 배풍기 작업은 같은 흐름에 병렬로 속하며, 연주 작업은 둘 다 완료된 뒤 시작합니다.

`quality`의 `INPUT_SCORE`는 앞 단계 품질 영향을 이어받고 `LOCAL_IMPACT`는 해당 설비 상태의 추가 영향입니다. `OUTPUT_SCORE`와 `DEFECT_FLAG`는 정답이며 예측 특징량으로 사용하지 않습니다. `inspections`는 해당 작업이 끝난 시각부터 이용할 수 있으며 D 최종 검사값은 포함하지 않습니다. 예측 기준시각 이후의 작업·검사·정비 행을 특징량에 연결하지 마세요.

최종 작업(D)이 72시간 안에 끝나지 않으면 최종 결함은 공란이고 `FINAL_CENSORED=1`입니다. 완료된 흐름의 최종 결함은 `FINAL_DEFECT_FLAG`에서 평가합니다. 원인 설비는 품질 영향이 가장 큰 단계로 정의합니다.
