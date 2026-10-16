# 연주 조기경보 EDA 실행

프로젝트 폴더 `WannaOne_END`에서 실행합니다.

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/eda.py
```

이미 만들어 둔 `data/POSKO-PdM-Data-pdm-v2.2.0-parquet/`의 시나리오별
Parquet을 읽습니다. 기본 실행은 학습 시나리오 216개만 사용하며, 빠르게 확인할 때는
`python analysis/eda.py --limit 2 --output-dir analysis/output-smoke`를 실행하세요. 데이터와 결과 경로는
`--data-root`, `--output-dir`로 바꿀 수 있습니다.

결과는 `analysis/output/`에 저장됩니다.

- `train_hourly_features.csv`: 1시간 단위 센서 요약과 기준 대비 변화량
- `summary.json`: 행 수, 고장 건수, 신호 분포 비교
- `example_normal.png`, `example_failure.png`: 정상·고장 사례 그래프

결과를 데이터프레임으로 읽는 예시:

```python
import pandas as pd

df = pd.read_csv("analysis/output/train_hourly_features.csv")
print(df.head())
```

`labels`와 `diagnostics`는 EDA의 비교 집단을 정할 때만 사용합니다. 결과 CSV의
모델 입력 열에는 정답 열을 넣지 않았습니다. `summary.json`의 AUC는 학습 세트의
정상 시간과 고장 24~36시간 전을 비교한 탐색 지표이며, 모델 성능이 아닙니다.
