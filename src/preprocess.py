"""연주 조기경보 전처리: 5초 원자료 → 1시간 요약 → 부하 보정 → 기준 대비 변화량.

EDA(analysis/01_eda.py, reports/연주_조기경보_결과보고서.md 부록 A)에서 확정한 결정을 함수로 옮긴 것이다.

    from src.preprocess import build_hourly, fit_params, apply_params

    hr = build_hourly(scenario_ids)                    # 1시간 요약 (원자료를 나눠 읽는다)
    params = fit_params(hr, normal_train_ids)          # 학습 세트 정상 시나리오에서 보정값을 정한다
    hr = apply_params(hr, params)                      # 보정값·변화량·표준화 점수를 붙인다

미래 정보 누설 방지
- 1시간 구간 [h, h+1)의 값은 그 구간 안의 원자료로만 계산한다. 예측 시점 t에는 h+1 ≤ t인 구간만 쓴다(features.py).
- 기준 수준은 운전 구간(정비 후 재가동부터)의 첫 2시간 중앙값이다. 기준이 정해지기 전인 첫 2시간은 변화량을 비워 둔다.
- 보정값(부하 기울기·절편, 표준화 기준)은 학습 세트에서 한 번 정해 검증·테스트에 그대로 쓴다.
"""

import numpy as np
import pandas as pd

from src.posko_data import load_scenarios, load_source

EQP = 'P1-CC1-INS01'
RAW_COLS = ['LOAD', 'WD-LOD', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY']   # OSC-FRQ(상수)·WD-SPD(LOAD와 같음) 제외
MIN_RUN_ROWS = 360        # 1시간(720행) 중 절반 이상 운전한 구간만 쓴다
BASE_HOURS = 2            # 기준 수준을 잡는 운전 구간 첫 시간 수

# 조기경보 신호 4개와 이상 방향 (+1: 커지면 이상, -1: 작아지면 이상)
SIGNALS = {'MLD-AMP': 1, 'CONTROL_DELAY': 1, 'WDL': 1, 'OSC-STK': -1}


# 1. 1시간 요약

def hourly_summary(cast, start_at):
    """한 묶음 시나리오의 5초 원자료를 1시간 구간으로 요약한다.

    cast: SCENARIO_ID, MEAS_DT, RUNNING, RAW_COLS 열이 있는 원자료
    start_at: 시나리오 ID → 시작 시각
    반환: 시나리오 × 시간 구간 표
      HOUR       시작 후 몇 번째 시간(0~71)
      SEG        운전 구간 번호 (정지 후 재가동할 때마다 1씩 증가)
      N_RUN      구간 안 운전 행 수
      MLD-AMP    MLD-LVL의 1시간 표준편차(진폭). MLD-LVL 결측 행은 계산에서 빠진다
      나머지     운전 중 평균
    """
    cast = cast.sort_values(['SCENARIO_ID', 'MEAS_DT'], ignore_index=True)
    h = (cast['MEAS_DT'] - cast['SCENARIO_ID'].map(start_at)).dt.total_seconds() / 3600
    cast['HOUR'] = h.astype(int)
    restart = (cast['RUNNING'] == 1) & (cast.groupby('SCENARIO_ID')['RUNNING'].shift() == 0)
    cast['SEG'] = restart.astype(int).groupby(cast['SCENARIO_ID']).cumsum()

    run = cast[cast['RUNNING'] == 1]   # 정지 중에는 센서가 0이라 요약에서 뺀다
    hr = (run.groupby(['SCENARIO_ID', 'HOUR'])
             .agg(**{c: (c, 'mean') for c in RAW_COLS},
                  **{'MLD-AMP': ('MLD-LVL', 'std')},
                  SEG=('SEG', 'max'), N_RUN=('RUNNING', 'size'))
             .reset_index())
    hr = hr[hr['N_RUN'] >= MIN_RUN_ROWS].drop(columns='MLD-LVL')   # MLD-LVL 평균은 신호가 없다(EDA 6절)
    hr['HOUR_START'] = (hr['SCENARIO_ID'].map(start_at)
                        + pd.to_timedelta(hr['HOUR'], unit='h')).astype('datetime64[ms, UTC]')
    hr['SEG_HOUR'] = hr.groupby(['SCENARIO_ID', 'SEG']).cumcount()   # 운전 구간 안에서 몇 번째 시간
    return hr.reset_index(drop=True)


def build_hourly(scenario_ids, chunk=60):
    """시나리오를 chunk개씩 나눠 원자료를 읽고 1시간 요약을 이어 붙인다 (메모리 절약)."""
    start_at = load_scenarios().set_index('SCENARIO_ID')['START_AT']
    ids = list(scenario_ids)
    parts = []
    for i in range(0, len(ids), chunk):
        cast = load_source('casting', scenarios=ids[i:i + chunk],
                           columns=['SCENARIO_ID', 'MEAS_DT', 'RUNNING'] + RAW_COLS)
        parts.append(hourly_summary(cast, start_at))
    return pd.concat(parts, ignore_index=True)


# 2. 보정값 정하기 (학습 세트 정상 시나리오)

def _level(hr, params):
    """부하 영향을 뺀 신호 수준 (EDA 6-2, 6-6)"""
    out = pd.DataFrame(index=hr.index)
    for c in ['MLD-AMP', 'CONTROL_DELAY', 'OSC-STK']:
        out[c] = hr[c] - params['slope'][c] * (hr['LOAD'] - params['load_ref'])      # 공통 기울기 차감
    out['WDL'] = (hr['WD-LOD'] - params['wdl_intercept']) / hr['LOAD']               # 절편 보정 비율
    return out


def _change(hr, level):
    """기준 대비 변화량
    _D: 운전 구간 첫 2시간 중앙값 대비 (이번 운전 구간에서 얼마나 나빠졌나)
    _F: 시나리오 첫 운전 구간의 첫 2시간 중앙값 대비 (처음부터 얼마나 나빠졌나, EDA 11절)
    기준이 정해지기 전(운전 구간 첫 2시간)은 비워 둔다
    """
    key = [hr['SCENARIO_ID'], hr['SEG']]
    is_base = hr['SEG_HOUR'] < BASE_HOURS
    seg_base = level.where(is_base).groupby(key).transform('median')
    first = is_base & (hr['SEG'] == 0)
    first_base = level.where(first).groupby(hr['SCENARIO_ID']).transform('median')
    d = level - seg_base
    d.loc[is_base] = np.nan                    # 기준 구간 자체는 비운다
    f = level - first_base                     # 첫 운전 구간이 짧아 기준이 없으면 NaN
    f.loc[first] = np.nan
    return d.add_suffix('_D'), f.add_suffix('_F')


def fit_params(hr, normal_ids):
    """학습 세트 정상 시나리오(normal_ids)에서 보정값을 정한다.

    slope         MLD-AMP·CONTROL_DELAY·OSC-STK의 부하 기울기 (시나리오 평균을 뺀 값으로 최소제곱)
    load_ref      기준 부하
    wdl_intercept WD-LOD = 절편 + 기울기 × LOAD 를 시나리오마다 맞춘 절편의 중앙값
    med, scale    변화량(_D) 표준화 기준: 정상 중앙값, (정상 95% 분위수 − 중앙값) (이상 방향 기준)
    """
    nrm = hr[hr['SCENARIO_ID'].isin(normal_ids)]
    cols = ['MLD-AMP', 'CONTROL_DELAY', 'OSC-STK', 'LOAD']
    dm = nrm[cols] - nrm.groupby('SCENARIO_ID')[cols].transform('mean')
    params = {
        'slope': {c: float((dm[c] * dm['LOAD']).sum() / (dm['LOAD'] ** 2).sum()) for c in cols[:-1]},
        'load_ref': float(hr['LOAD'].mean()),
        'wdl_intercept': float(nrm.groupby('SCENARIO_ID')[['LOAD', 'WD-LOD']]
                               .apply(lambda g: np.polyfit(g['LOAD'], g['WD-LOD'], 1)[1]).median()),
    }
    d, _ = _change(nrm, _level(nrm, params))
    params['med'], params['scale'] = {}, {}
    for s, sgn in SIGNALS.items():
        v = d[s + '_D'].dropna() * sgn
        params['med'][s] = float(v.median() * sgn)
        params['scale'][s] = float(v.quantile(0.95) - v.median())
    return params


# 3. 보정값 적용

def apply_params(hr, params):
    """1시간 요약에 보정 수준, 변화량(_D, _F), 표준화 점수(Z_), 평균 점수(SCORE)를 붙인다.

    Z_신호 = (변화량 − 정상 중앙값) × 방향 ÷ 정상 범위 폭. 1이면 정상 범위 경계(정상 95%).
    SCORE  = 네 신호 Z의 평균 (EDA 9-3: 오경보를 크게 줄이는 결합 점수)
    SCORE_F = 처음 수준 대비 변화량(_F)으로 같은 방식으로 계산한 평균 점수
    """
    hr = hr.copy()
    level = _level(hr, params)
    d, f = _change(hr, level)
    hr[[c + '_LVL' for c in level.columns]] = level.values
    hr[d.columns] = d.values
    hr[f.columns] = f.values
    for s, sgn in SIGNALS.items():
        hr['Z_' + s] = (hr[s + '_D'] - params['med'][s]) * sgn / params['scale'][s]
        hr['ZF_' + s] = (hr[s + '_F'] - params['med'][s]) * sgn / params['scale'][s]
    hr['SCORE'] = hr[['Z_' + s for s in SIGNALS]].mean(axis=1, skipna=False)
    hr['SCORE_F'] = hr[['ZF_' + s for s in SIGNALS]].mean(axis=1, skipna=False)
    return hr
