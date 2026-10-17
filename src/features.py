"""연주 조기경보 특징·정답 만들기: 예측 시점마다 그 시점까지의 정보만으로 특징을 계산한다.

    from src.features import build_dataset, FEATURE_SETS

    X, params, hr = build_dataset()                    # 학습·검증 세트 특징·정답 표 (한 번에)
    cols = FEATURE_SETS['sensor']                      # 쓸 특징 묶음을 고른다

    # 단계별로 하려면
    from src.features import make_targets, make_features
    tg = make_targets(scenario_ids)                    # 예측 시점(6시간마다)과 정답
    X = make_features(hr, tg)                          # hr = preprocess.apply_params 결과

예측 시점 t에서 쓰는 정보
- 센서: 끝난 1시간 구간(구간 시작 + 1시간 ≤ t)만
- 정비 기록: t까지 끝난 정비만

정답은 기준 예제(analysis/02_baseline.py)와 같은 규칙이다. 그래서 기준 예제와 같은 시점에서 비교할 수 있다.
"""

import numpy as np
import pandas as pd

from src.posko_data import load_answer, load_scenarios, load_source
from src.preprocess import EQP, SIGNALS

# 특징 묶음. 'sensor'가 기본이다. 정비 이력은 쓰고(팀 결정 F), 경과 시간은 비교용으로만 둔다(팀 결정 E)
_SIG_FEATS = [f'{s}_{k}' for s in SIGNALS for k in ['D', 'D_MEAN3', 'D_SLOPE6', 'F']]
FEATURE_SETS = {
    'sensor': _SIG_FEATS + [
        'SCORE', 'SCORE_MAX', 'SCORE_2ND', 'SCORE_F', 'SCORE_SLOPE6',   # 결합 점수 (EDA 9-3)
        'PERSIST_H',                                                    # 평균 점수가 1을 넘은 연속 시간 (EDA 7-3)
        'GAP_H',                                                        # 마지막 운전 구간 이후 공백
    ],
    'maintenance': ['N_MNT', 'H_SINCE_MNT', 'LAST_PARTIAL', 'LAST_REPAIR', 'LAST_REPLACE'],   # 정비 이력 (팀 결정 F: 사용)
    # 시간을 세는 특징 (팀 결정 E: 사용 안 함). 계산은 하지만 모델에는 넣지 않는다.
    #   ELAPSED_H  시나리오 시작 후 경과 시간
    #   SEG_HOURS  재가동 후 지난 시간. 정비가 없던 시나리오에서는 경과 시간과 같아 같은 착시를 배운다
    #              (학습 세트 54시간 이후 시점은 관측 종료로 정답 0이 빠져 모두 1이다. 결과보고서 부록 B-4 E)
    'elapsed': ['ELAPSED_H', 'SEG_HOURS'],
}

# 예측 시점 표의 열 (정답 열은 특징이 아니다)
TARGET_COLS = ['SCENARIO_ID', 'SPLIT', 'OBSERVED_AT', 'T_H', 'risk24', 'RUL_EVENT', 'RUL_HOURS']

# 마지막 정비 유형 특징으로 만드는 정비 유형. 연주 설비 정비 기록에는 이 셋만 있다(점검 inspect는 없음)
MNT_TYPES = ['partial', 'repair', 'replace']


# 1. 예측 시점과 정답

def make_targets(scenario_ids):
    """targets의 6시간 간격 예측 시점 중 AT_RISK=1인 시점과 정답.

    risk24  24시간 안 고장이면 1, 고장이 24시간 넘게 남았거나 24시간 넘게 고장이 없음이 확인되면 0.
            관측 종료로 알 수 없는 시점은 뺀다 (기준 예제와 같은 규칙)
    RUL_EVENT, RUL_HOURS  평가에서 '이른 경보'를 나눠 보기 위한 열 (특징 아님)
    """
    scen = load_scenarios().set_index('SCENARIO_ID')
    tg = load_answer('targets').query('EQP_CD == @EQP and SCENARIO_ID in @scenario_ids').copy()
    tg = tg[tg['AT_RISK'] == 1]
    tg['SPLIT'] = tg['SCENARIO_ID'].map(scen['SPLIT'])
    tg['T_H'] = (tg['OBSERVED_AT'] - tg['SCENARIO_ID'].map(scen['START_AT'])).dt.total_seconds() / 3600
    event = tg['RUL_EVENT'] == 1
    tg['risk24'] = np.select([event & (tg['FOLLOWUP_HOURS'] <= 24), event, tg['FOLLOWUP_HOURS'] >= 24],
                             [1, 0, 0], default=-1)
    tg = tg[tg['risk24'] >= 0]
    return tg[['SCENARIO_ID', 'SPLIT', 'OBSERVED_AT', 'T_H', 'risk24', 'RUL_EVENT', 'RUL_HOURS']].reset_index(drop=True)


def state_outcomes(scenario_ids, step_h=1):
    """예측 시점마다 '정비가 없었다면' 고장 날 시각 (make_targets_hourly와 DSS 정책 시뮬레이션이 같이 쓴다).

    정답 파일의 잔여수명은 "예측 시각까지의 정비를 반영하고, 이후 추가 정비가 없다고 가정"한 값이다.
    그래서 정기 교체로 실제 고장이 막힌 경우에도 '정비가 없었다면 났을 고장'이 정답에 들어 있고,
    정비 기록만으로는 같은 정답을 만들 수 없다. 대신
    - 같은 정비 상태(마지막으로 시작된 정비가 같은 구간) 안의 6시간 정답 행에서 가상 고장 시각
      (관측 시각 + RUL_HOURS)을 읽는다. 상태 안의 6시간 행끼리 이 시각과 RUL_EVENT는 항상 같다(2026-10-15 확인)
    - 6시간 행이 없는 상태는 그 뒤에 정비가 없을 때만 '관측 끝까지 고장 없음'으로 둔다 (그 밖은 알 수 없음 NaN)
    반환: make_prediction_points 결과(정답 열 제외) + STATE(정비 상태 번호), EV(관측 안 고장 1/0/NaN),
          TSTAR(가상 고장 시각), FOLLOW_H(관측 끝까지 남은 시간). 정답을 알 수 없는 시점도 모두 들어 있다
    """
    scen = load_scenarios().set_index('SCENARIO_ID')
    end = scen['START_AT'] + pd.to_timedelta(scen['DURATION_H'], unit='h')
    mnt = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids')
    tg = load_answer('targets').query('EQP_CD == @EQP and SCENARIO_ID in @scenario_ids and AT_RISK == 1').copy()
    pts = make_prediction_points(scenario_ids, step_h=step_h)

    def n_started(df):   # 정비 상태 번호: 그 시각까지 시작된 정비 수
        m = df[['SCENARIO_ID', 'OBSERVED_AT']].reset_index().merge(mnt[['SCENARIO_ID', 'STRT_DT']], on='SCENARIO_ID', how='left')
        return (m['STRT_DT'] <= m['OBSERVED_AT']).groupby(m['index']).sum().reindex(df.index).fillna(0).astype(int)

    tg['STATE'] = n_started(tg)
    tg['TSTAR'] = tg['OBSERVED_AT'] + pd.to_timedelta(tg['RUL_HOURS'], unit='h')
    st = tg.groupby(['SCENARIO_ID', 'STATE']).agg(EV=('RUL_EVENT', 'max'), EV_MIN=('RUL_EVENT', 'min'),
                                                  TSTAR=('TSTAR', 'min'), TSTAR_MAX=('TSTAR', 'max'))
    spread = (st['TSTAR_MAX'] - st['TSTAR']).dt.total_seconds() > 1
    if (st['EV'] != st['EV_MIN']).any() or spread.any():
        raise ValueError('같은 정비 상태 안의 6시간 정답이 서로 다르다. state_outcomes의 가정을 다시 확인할 것')

    pts['STATE'] = n_started(pts)
    pts = pts.drop(columns=['risk24', 'RUL_EVENT', 'RUL_HOURS']).merge(
        st[['EV', 'TSTAR']].reset_index(), on=['SCENARIO_ID', 'STATE'], how='left')
    later = pts[['SCENARIO_ID', 'OBSERVED_AT']].merge(mnt[['SCENARIO_ID', 'STRT_DT']], on='SCENARIO_ID')
    later = later[later['STRT_DT'] > later['OBSERVED_AT']]
    has_later = set(zip(later['SCENARIO_ID'], later['OBSERVED_AT']))
    quiet = pts['EV'].isna() & np.array([k not in has_later for k in zip(pts['SCENARIO_ID'], pts['OBSERVED_AT'])], dtype=bool)
    pts.loc[quiet, 'EV'] = 0
    pts['FOLLOW_H'] = (pts['SCENARIO_ID'].map(end) - pts['OBSERVED_AT']).dt.total_seconds() / 3600
    return pts


def make_targets_hourly(scenario_ids, step_h=1):
    """step_h시간 간격 예측 시점의 정답 (정답 파일 targets는 6시간 간격이라 state_outcomes로 만든다).

    정답 규칙(24시간 안 고장 1 / 그 밖 0 / 알 수 없음 제외)은 make_targets와 같다.
    6시간 시점에서는 make_targets와 정답·잔여수명이 모두 같다 (tests/check_regression.py).
    """
    pts = state_outcomes(scenario_ids, step_h=step_h)
    pts['RUL_EVENT'] = pts['EV']
    pts['RUL_HOURS'] = ((pts['TSTAR'] - pts['OBSERVED_AT']).dt.total_seconds() / 3600).where(pts['EV'] == 1)
    event = pts['EV'] == 1
    pts['risk24'] = np.select([event & (pts['RUL_HOURS'] <= 24), event, (pts['EV'] == 0) & (pts['FOLLOW_H'] >= 24)],
                              [1, 0, 0], default=-1)
    pts = pts[pts['risk24'] >= 0].astype({'T_H': float})   # make_targets와 같은 형식
    return pts[TARGET_COLS].reset_index(drop=True)


def make_prediction_points(scenario_ids, step_h=6):
    """정답 파일 없이 예측 시점을 만든다 (테스트 세트용).

    시나리오 시작 후 0, 6, ..., 66시간 시점 중 그 시각에 정비 중이 아닌 시점.
    정비 중인지는 정비 기록의 시작 시각으로 판단한다(시작된 정비는 그 시각에 이미 알려진 정보).
    학습·검증 세트에서 이 규칙은 targets의 AT_RISK=1과 정확히 일치한다(analysis/05_predict_test.py에서 확인).
    """
    scen = load_scenarios().set_index('SCENARIO_ID')
    pts = pd.DataFrame([(s, h) for s in scenario_ids for h in range(0, 72, step_h)], columns=['SCENARIO_ID', 'T_H'])
    pts['OBSERVED_AT'] = (pts['SCENARIO_ID'].map(scen['START_AT'])
                          + pd.to_timedelta(pts['T_H'], unit='h')).astype('datetime64[ms, UTC]')
    mnt = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids')
    m = pts.merge(mnt[['SCENARIO_ID', 'STRT_DT', 'END_DT']], on='SCENARIO_ID')
    busy = m[(m['STRT_DT'] <= m['OBSERVED_AT']) & ((m['END_DT'] > m['OBSERVED_AT']) | m['END_DT'].isna())]
    busy_keys = set(zip(busy['SCENARIO_ID'], busy['OBSERVED_AT']))
    pts = pts[[k not in busy_keys for k in zip(pts['SCENARIO_ID'], pts['OBSERVED_AT'])]]
    pts['SPLIT'] = pts['SCENARIO_ID'].map(scen['SPLIT'])
    for c in ['risk24', 'RUL_EVENT', 'RUL_HOURS']:   # 정답 열 자리 (테스트 세트에는 정답이 없다)
        pts[c] = np.nan
    return pts[TARGET_COLS].reset_index(drop=True)


# 2. 특징

def _slope(y):
    """시간 간격 1인 값들의 최소제곱 기울기 (값이 2개 미만이거나 결측이 있으면 NaN)"""
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or np.isnan(y).any():
        return np.nan
    x = np.arange(len(y)) - (len(y) - 1) / 2
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def _sensor_features(g, t_h):
    """한 시나리오의 1시간 표(g, HOUR 오름차순)에서 예측 시점 t_h(경과 시간)의 센서 특징"""
    past = g[g['HOUR'] + 1 <= t_h]                      # 끝난 구간만
    out = {}
    if past.empty:
        return out
    last = past.iloc[-1]
    out['GAP_H'] = t_h - (last['HOUR'] + 1)             # 마지막으로 운전한 1시간 구간 이후 지난 시간
    if out['GAP_H'] >= 1:
        # 그 사이에 정지(정비)가 있었고 재가동 후 끝난 구간이 아직 없다.
        # 정비 전 값은 지금 상태가 아니므로 센서 특징을 비워 둔다
        out['SEG_HOURS'] = 0
        return out
    seg = past[past['SEG'] == last['SEG']]              # 현재 운전 구간 (정비 이전 기록은 섞지 않는다)
    out['SEG_HOURS'] = len(seg)
    for s in SIGNALS:
        d = seg[s + '_D']
        out[f'{s}_D'] = last[s + '_D']
        out[f'{s}_D_MEAN3'] = d.tail(3).mean() if d.tail(3).notna().any() else np.nan
        out[f'{s}_D_SLOPE6'] = _slope(d.dropna().tail(6))
        out[f'{s}_F'] = last[s + '_F']
    z = np.array([last['Z_' + s] for s in SIGNALS], dtype=float)
    zs = np.sort(z)
    out['SCORE'] = last['SCORE']
    out['SCORE_MAX'] = zs[-1] if not np.isnan(z).any() else np.nan
    out['SCORE_2ND'] = zs[-2] if not np.isnan(z).any() else np.nan
    out['SCORE_F'] = last['SCORE_F']
    out['SCORE_SLOPE6'] = _slope(seg['SCORE'].dropna().tail(6))
    over = (seg['SCORE'] > 1).values[::-1]              # 최근 시간부터 거꾸로
    out['PERSIST_H'] = int(np.argmin(over)) if not over.all() else len(over)
    return out


def _maintenance_features(m, t):
    """한 시나리오의 정비 기록(m)에서 예측 시각 t까지 끝난 정비로 만든 특징"""
    done = m[m['END_DT'] <= t].sort_values('END_DT')
    out = {'N_MNT': len(done), 'H_SINCE_MNT': np.nan, 'LAST_PARTIAL': 0, 'LAST_REPAIR': 0, 'LAST_REPLACE': 0}
    if len(done):
        lastm = done.iloc[-1]
        out['H_SINCE_MNT'] = (t - lastm['END_DT']).total_seconds() / 3600
        if lastm['MNT_TYPE'] not in MNT_TYPES:
            # 모르는 유형을 조용히 버리면 '마지막 정비 없음'과 구분되지 않는다
            raise ValueError(f'모르는 정비 유형: {lastm["MNT_TYPE"]} (MNT_TYPES에 추가하고 모델을 다시 학습할 것)')
        out['LAST_' + lastm['MNT_TYPE'].upper()] = 1
    return out


def make_features(hr, tg):
    """예측 시점(tg의 각 행)마다 특징을 계산해 tg 옆에 붙인다.

    hr: preprocess.apply_params 결과 (tg의 시나리오를 모두 포함)
    """
    mnt = load_source('maintenance').query('EQP_TAG == @EQP')
    hr_by = dict(tuple(hr.sort_values(['SCENARIO_ID', 'HOUR']).groupby('SCENARIO_ID')))
    mnt_by = dict(tuple(mnt.groupby('SCENARIO_ID')))
    empty_m = mnt.iloc[:0]
    rows = []
    for r in tg.itertuples():
        row = {'ELAPSED_H': r.T_H}
        row.update(_sensor_features(hr_by.get(r.SCENARIO_ID, hr.iloc[:0]), r.T_H))
        row.update(_maintenance_features(mnt_by.get(r.SCENARIO_ID, empty_m), r.OBSERVED_AT))
        rows.append(row)
    feats = pd.DataFrame(rows, index=tg.index)
    all_cols = [c for cols in FEATURE_SETS.values() for c in cols]
    return pd.concat([tg, feats.reindex(columns=all_cols)], axis=1)


# 3. 한 번에 만들기


def normal_scenarios(scenario_ids):
    """주 고장 원인(마찰·고착·제어 지연)이 없는 시나리오. 보정값을 정하는 기준으로만 쓴다.

    진단 정답을 쓰므로, 보정값을 정하는 시나리오(학습 세트, 교차검증에서는 그 폴드의 학습 부분) 안에서만 고른다.
    """
    diag = load_answer('diagnostics').query('EQP_CD == @EQP and SCENARIO_ID in @scenario_ids')
    fault = set(diag.loc[diag['FAULT_MODE'].isin(['friction', 'sticking', 'control-delay']), 'SCENARIO_ID'])
    return [s for s in scenario_ids if s not in fault]


def build_dataset(step_h=6):
    """학습·검증 세트의 예측 시점별 특징·정답 표를 만든다 (6시간 간격 약 15초, 1시간 간격 약 1분).

    보정값은 학습 세트의 정상 시나리오(주 고장 원인이 없는 시나리오)에서 정한다.
    step_h  예측 간격. 6이면 정답 파일(targets)의 시점, 그 밖이면 make_targets_hourly
    반환: (특징·정답 표, 보정값 params, 1시간 요약 hr)
    """
    from src.preprocess import apply_params, build_hourly, fit_params

    scen = load_scenarios()
    ids = scen.loc[scen['SPLIT'] != 'test', 'SCENARIO_ID'].tolist()
    train = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID'].tolist()
    hr = build_hourly(ids)
    params = fit_params(hr[hr['SCENARIO_ID'].isin(train)], normal_scenarios(train))
    hr = apply_params(hr, params)
    tg = make_targets(ids) if step_h == 6 else make_targets_hourly(ids, step_h=step_h)
    return make_features(hr, tg), params, hr
