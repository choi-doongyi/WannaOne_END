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

## CSV 기반 24시간 고장 조기경보

`src/two4.py`는 실행용 파일이고, 데이터 추출·학습·평가·그래프 함수는
`analysis/casting_rf.py`에 있습니다. 이미 생성한
`casting_with_failer_soon.csv`가 필요하며, 처음 실행할 때만 정답이 있는 행을
`casting_labeled_only.csv`로 추출합니다. 이후에는 작은 CSV를 재사용합니다.

```powershell
python src/two4.py --fn-cost 200 --fp-cost 20
python src/two4.py --fn-cost 200 --fp-cost 20 --no-plot
```

비용 옵션을 생략하면 실행 중 입력받습니다. `--data-dir`로 데이터 폴더를 바꾸거나
`--force-extract`로 작은 CSV를 다시 만들 수 있습니다. 다른 Python 파일에서는
프로젝트 루트를 import 경로에 둔 뒤 다음처럼 함수별로 사용할 수 있습니다.

```python
from analysis.casting_rf import (
    FEATURES, data_paths, evaluate_thresholds, load_splits,
    prepare_labeled_csv, select_threshold, train_model,
)

large_csv, small_csv, manifest_path = data_paths()
prepare_labeled_csv(large_csv, small_csv)
train, valid = load_splits(small_csv, manifest_path)
model = train_model(train)
scores = model.predict_proba(valid[list(FEATURES)])[:, 1]
results = evaluate_thresholds(valid["failer_soon"], scores, 200, 20)
best = select_threshold(results)
print(best["임계값"], best["손실점수"])
```

임계값은 검증 데이터의 6시간 간격 라벨 행마다 FN/FP 비용을 부과해 고릅니다.
같은 고장 사건이 여러 행에 반영될 수 있으므로 사건별 운영 비용과는 구분하세요.
