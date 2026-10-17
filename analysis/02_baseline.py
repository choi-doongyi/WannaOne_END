"""기준 예제(toolkit/casting_example.py)의 24시간 고장위험 예측을 Parquet 데이터로 재현한다.

데이터셋에 들어 있는 연주 예제를 같은 규칙으로 다시 구현해, 우리 모델이 넘어야 할 기준점을 만든다.
원본 예제는 CSV 조각을 직접 읽고, 여기서는 src/posko_data.py가 만든 Parquet을 읽는다.

    python analysis/02_baseline.py

기준 예제의 방법
- 관측: 연주 센서 6개를 30분 간격(정각·30분)으로 뽑는다
- 예측 시점: targets의 6시간 간격 시점 중 AT_RISK=1인 시점
- 정답: 24시간 안에 고장(RUL_EVENT=1이고 FOLLOWUP_HOURS ≤ 24)이면 1,
        고장이 24시간 넘게 남았거나 24시간 넘게 고장이 없음이 확인되면 0,
        그 외(관측 종료로 모름)는 제외
- 특징: current = 예측 시점의 센서값 6개
        temporal = current + 최근 12시간의 변화량·기울기·평균·표준편차 + 관측 길이
        (12시간 안에 정지가 있으면 정지 이후만 쓴다. 예측 시점에 정지 중이면 제외)
- 모델: 랜덤포레스트(180그루, 깊이 7, class_weight=balanced, random_state=31)
- 임계값: 검증 세트에서 F1이 가장 큰 값(0.05~0.95, 0.05 간격)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import evaluate as evaluate_alarms, print_report
from src.posko_data import load_answer, load_scenarios, load_source

EQP = 'P1-CC1-INS01'
COLS = ['LOAD', 'WD-SPD', 'WD-LOD', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY']   # 예제가 쓰는 센서 6개


# 1. 데이터 불러오기: 학습·검증 시나리오의 30분 간격 관측과 예측 시점 정답

def load_observations(scen):
    """시나리오마다 시작 후 0, 0.5, ..., 71.5시간 시각의 관측 행을 뽑는다."""
    cast = load_source('casting', scenarios=scen['SCENARIO_ID'].tolist(),
                       columns=['SCENARIO_ID', 'MEAS_DT', 'RUNNING'] + COLS)
    cast = cast.merge(scen[['SCENARIO_ID', 'START_AT']], on='SCENARIO_ID')
    cast['h'] = (cast['MEAS_DT'] - cast['START_AT']).dt.total_seconds() / 3600
    obs = cast[np.isclose(cast['h'] * 2, np.round(cast['h'] * 2))]   # 30분 배수 시각만
    return obs.sort_values(['SCENARIO_ID', 'h'], ignore_index=True)


def load_targets(scen, keep_unlabeled=False):
    """예측 시점별 정답. 예제와 같은 규칙으로 24시간 위험(risk24)을 만든다.

    keep_unlabeled=True면 정답을 알 수 없는 시점(risk24 = -1)도 남긴다 (운영 비용의 미확정 경보를 셀 때)
    """
    tg = load_answer('targets').query('EQP_CD == @EQP')
    tg = tg.merge(scen[['SCENARIO_ID', 'START_AT', 'SPLIT']], on='SCENARIO_ID')
    tg['h'] = (tg['OBSERVED_AT'] - tg['START_AT']).dt.total_seconds() / 3600
    tg = tg[tg['AT_RISK'] == 1].copy()
    event = tg['RUL_EVENT'] == 1
    # 고장 예정이면 24시간 안인지로 1/0, 고장 예정이 없으면 남은 관측이 24시간 이상일 때만 0
    #   (고장이 24시간 넘게 남은 시점도 '24시간 안엔 고장 안 남'이 확실하므로 0)
    tg['risk24'] = np.select([event & (tg['FOLLOWUP_HOURS'] <= 24), event,
                              ~event & (tg['FOLLOWUP_HOURS'] >= 24)],
                             [1, 0, 0], default=-1)
    return tg if keep_unlabeled else tg[tg['risk24'] >= 0]   # 관측 종료로 24시간 결과를 모르는 시점은 뺀다


# 2. 특징 만들기: 예측 시점까지의 최근 12시간만 쓴다

def window_features(g, h):
    """예제의 window_features와 같은 계산. g는 한 시나리오의 30분 관측(h 오름차순)."""
    recent = g[(g['h'] >= max(0, h - 12)) & (g['h'] <= h)]
    if recent.empty or recent['RUNNING'].iloc[-1] != 1:
        return None                                       # 예측 시점에 정지 중이면 제외
    stops = np.flatnonzero(recent['RUNNING'].values == 0)
    if len(stops):
        recent = recent.iloc[stops[-1] + 1:]              # 마지막 정지 이후만 쓴다
    history = recent[COLS].values
    current = history[-1]
    elapsed = recent['h'].values
    centered = elapsed - elapsed.mean()
    slope = ((history - history.mean(axis=0)) * centered[:, None]).sum(axis=0) / max(float((centered ** 2).sum()), 1e-9)
    delta = history[-1] - history[0] if len(recent) > 1 else np.full(len(COLS), np.nan)
    temporal = np.concatenate([current, delta, slope, history.mean(axis=0), history.std(axis=0),
                               [elapsed[-1] - elapsed[0]]])
    return current, temporal


def make_records(obs, tg):
    rows = []
    obs_by = dict(tuple(obs.groupby('SCENARIO_ID')))
    for r in tg.itertuples():
        f = window_features(obs_by[r.SCENARIO_ID], r.h)
        if f is None:
            continue
        rows.append({'SCENARIO_ID': r.SCENARIO_ID, 'SPLIT': r.SPLIT, 'h': r.h, 'OBSERVED_AT': r.OBSERVED_AT,
                     'risk24': r.risk24,
                     'RUL_EVENT': r.RUL_EVENT, 'RUL_HOURS': r.RUL_HOURS,
                     'current': f[0], 'temporal': f[1]})
    return pd.DataFrame(rows)


# 3. 학습·평가: 학습 세트로 학습, 검증 세트에서 임계값을 고르고 평가한다

def evaluate(rec):
    train, valid = rec[rec['SPLIT'] == 'train'], rec[rec['SPLIT'] == 'validation']
    report, probs = {}, {}
    for mode in ['current', 'temporal']:
        model = make_pipeline(SimpleImputer(add_indicator=True),
                              RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=5,
                                                     class_weight='balanced', random_state=31, n_jobs=2))
        model.fit(np.stack(train[mode]), train['risk24'])
        p = model.predict_proba(np.stack(valid[mode]))[:, 1]
        y = valid['risk24'].values
        th = float(max(np.arange(.05, .96, .05), key=lambda t: f1_score(y, p >= t, zero_division=0)))
        report[mode] = {
            '예측 시점 수': len(valid), '양성 시점': int(y.sum()),
            '양성 시나리오': valid.loc[valid['risk24'] == 1, 'SCENARIO_ID'].nunique(),
            'AP': average_precision_score(y, p), '양성 비율': y.mean(), '임계값': th,
            '정밀도': precision_score(y, p >= th, zero_division=0),
            '재현율': recall_score(y, p >= th, zero_division=0),
        }
        probs[mode] = p
    return pd.DataFrame(report), valid.assign(**{f'p_{m}': v for m, v in probs.items()})


if __name__ == '__main__':
    scen = load_scenarios().query('SPLIT != "test"')
    obs = load_observations(scen)
    tg = load_targets(scen)
    rec = make_records(obs, tg)
    print(f'예측 시점: 학습 {int((rec.SPLIT == "train").sum())}개, 검증 {int((rec.SPLIT == "validation").sum())}개')
    report, valid = evaluate(rec)
    print(report.round(3).to_string())
    print()
    print('주의: 임계값을 검증 세트에서 고르고 같은 검증 세트로 평가하므로 정밀도·재현율은 낙관적이다.')
    print()
    # 4. 평가 함수(src/evaluate.py)로 오경보·놓친 고장을 나누고 고장 단위·비용까지 본다
    print_report(evaluate_alarms(valid, valid['p_temporal'].values, report.loc['임계값', 'temporal']),
                 '기준 예제 temporal (검증 세트)')
