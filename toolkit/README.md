# 분석 안내

`manifest.json`에 시나리오 분할, 원자료 파일, 청크 순서와 SHA-256이 있습니다. CSV는 시나리오·원자료 종류별 청크로 나뉩니다. 대용량 원자료는 `posko.read_rows()` 또는 `pandas.read_csv(chunksize=...)`로 읽으세요.

| 원자료 | 관측 간격 | 세부 표본 |
|---|---|---|
| A 소둔 | 30초 | — |
| B 용접 | 10분 | 코일당 13행 |
| C 연주 | 5초 | — |
| D 조질압연 | 5분 | 10Hz × 18표본 |
| E 배풍기 | 2시간 | 1400Hz × 700표본 |
| F 교반기 | 30분 | 256Hz × 128표본 |

시나리오 시작 시각에 따른 train/validation/test 분할을 유지하고 전처리·임계값은 train/validation에서 결정하세요. 정비 이력은 예측 기준시각까지 알려진 행만 연결하세요. `labels/`, `diagnostics/`, `targets/`, `flow/quality/`는 정답 파일이며 예측 특징량에 넣지 않습니다. 검열된 표본은 미고장으로 치환하지 마세요.

[정답·검열·정비 정책 정의](targets.md)를 확인한 뒤 `casting_example.py`로 현재값과 과거 12시간 특징의 예측 결과를 비교할 수 있습니다. 예측 JSON을 앱으로 가져올 때는 관측 교환본의 `scenarioId`, `branchId`, `datasetVersion`, `observedAt`, `observationHash`를 그대로 사용하세요.

[연결 생산 흐름과 품질 정답](flow.md)을 확인하세요.
