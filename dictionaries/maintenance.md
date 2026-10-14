# maintenance · G

| 컬럼 | 정의·단위 |
|---|---|
| SCENARIO_ID | 독립 시나리오 ID; 모든 표의 연결 키 |
| WO_NO | 정비 작업 지시 ID |
| EQP_TAG | 설비 ID; EQP_CD와 동일한 값 |
| PART_CD | 설비 부품 코드 |
| DETC_DT | 이상 인지 시각(UTC ISO 8601) |
| FAILURE_DT | 고장 발생 시각(UTC ISO 8601); 고장이 없으면 공란 |
| STRT_DT | 시작 시각(UTC ISO 8601) |
| END_DT | 종료 시각(UTC ISO 8601); 미완료 작업은 공란 |
| MNT_TYPE | 정비 유형: inspect, partial, replace, repair |
| DOWN_MIN | 정비 정지시간(분) |
| COST_SCORE | 정비 조치 비용 점수 |
