"""전처리·특징 함수(src/preprocess.py, src/features.py) 점검.

모델에 넣기 전에 특징이 의도대로 계산됐는지 여러 방향에서 맞춰 본다.
전처리 코드를 고칠 때마다 다시 실행해 망가진 곳이 없는지 확인한다.

    python analysis/03_check_features.py           # 전체 점검 (약 3~5분)
    python analysis/03_check_features.py --quick   # 누설 검사를 표본 100개로 (약 30초)

점검 항목 (하나라도 실패하면 종료 코드 1)
  1. 정답: 예측 시점과 정답이 기준 예제(02_baseline.py)와 같은가
  2. 보정값: EDA에서 확정한 값과 같은가 (절편 3.06, 정상 구간 경계 초과 비율 약 5% / 평균 점수 1% 미만)
  3. 미래 정보 누설: 원자료를 예측 시점에서 잘라 다시 계산해도 특징이 같은가
  4. 경계 상황: 기준 구간·정비 직후·시작 시점·고장 2번인 시나리오가 의도대로 처리되는가
  5. 손 계산: 원자료에서 직접 계산한 값과 함수 결과가 같은가
  6. 신호 방향: 고장 전 시점의 점수가 정상보다 높은가 (그림: figures/11_check_고장시나리오_점수흐름.png)
  7. 분포: 정상 시나리오에서 학습·검증 세트의 특징 분포가 비슷한가 (보정값이 학습 세트에 과하게 맞춰지지 않았나)
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import FEATURE_SETS, build_dataset, make_features
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_answer, load_scenarios, load_source
from src.preprocess import EQP, RAW_COLS, SIGNALS, apply_params, hourly_summary

setup_korean_font()

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(f'  [{"통과" if ok else "실패"}] {name}' + (f' — {detail}' if detail else ''))


def section(title):
    print(f'\n{title}')


# 0. 특징 표 만들기

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true', help='누설 검사를 표본 100개로 줄인다')
args = parser.parse_args()

print('특징 표를 만드는 중...')
X, params, hr = build_dataset()
scen = load_scenarios()
START = scen.set_index('SCENARIO_ID')['START_AT']
SENSOR = FEATURE_SETS['sensor']
ALL = [c for cols in FEATURE_SETS.values() for c in cols]
diag = load_answer('diagnostics').query('EQP_CD == @EQP')
fault = set(diag.loc[diag['FAULT_MODE'].isin(['friction', 'sticking', 'control-delay']), 'SCENARIO_ID'])
mnt = load_source('maintenance').query('EQP_TAG == @EQP')
print(f'예측 시점 {len(X):,}개 × 특징 {len(ALL)}개')


# 1. 정답이 기준 예제와 같은가

section('1. 정답')
spec = importlib.util.spec_from_file_location('baseline', ROOT / 'analysis' / '02_baseline.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
bt = baseline.load_targets(scen[scen['SPLIT'] != 'test'])
ours = X.set_index(['SCENARIO_ID', 'OBSERVED_AT'])['risk24'].sort_index()
theirs = bt.set_index(['SCENARIO_ID', 'OBSERVED_AT'])['risk24'].sort_index()
check('예측 시점 목록이 기준 예제와 같다', ours.index.equals(theirs.index), f'{len(ours)} vs {len(theirs)}')
check('정답(risk24)이 기준 예제와 같다', ours.index.equals(theirs.index) and (ours.values == theirs.values).all(),
      f'양성 {int(ours.sum())} vs {int(theirs.sum())}')
check('검열·관측 종료로 모르는 시점이 없다', X['risk24'].isin([0, 1]).all())


# 2. 보정값이 EDA와 같은가

section('2. 보정값')
check('WD-LOD 절편이 EDA 값(3.06)과 같다', abs(params['wdl_intercept'] - 3.06) < 0.01, f'{params["wdl_intercept"]:.3f}')
train = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID']
n = hr[hr['SCENARIO_ID'].isin(set(train) - fault)]
share = {s: float((n['Z_' + s] > 1).mean()) for s in SIGNALS}
check('학습 정상 구간에서 신호별 경계 초과 비율이 약 5%다', all(0.04 <= v <= 0.06 for v in share.values()),
      ', '.join(f'{s} {v:.3f}' for s, v in share.items()))
check('학습 정상 구간에서 평균 점수 경계 초과 비율이 1% 미만이다', (n['SCORE'] > 1).mean() < 0.01,
      f'{(n["SCORE"] > 1).mean():.4f}')


# 3. 미래 정보 누설: 원자료를 예측 시점에서 잘라 다시 계산

section('3. 미래 정보 누설')
target = X.sample(100, random_state=0) if args.quick else X
mismatch = []
for sid, pts in target.groupby('SCENARIO_ID'):
    raw = load_source('casting', scenarios=[sid], columns=['SCENARIO_ID', 'MEAS_DT', 'RUNNING'] + RAW_COLS)
    for idx, r in pts.iterrows():
        cut = raw[raw['MEAS_DT'] < r['OBSERVED_AT']]
        h2 = apply_params(hourly_summary(cut.copy(), START), params) if len(cut) else hr.iloc[:0]
        x2 = make_features(h2, X.loc[[idx], ['SCENARIO_ID', 'SPLIT', 'OBSERVED_AT', 'T_H', 'risk24', 'RUL_EVENT', 'RUL_HOURS']])
        a = X.loc[idx, ALL].astype(float).values
        b = x2[ALL].astype(float).values[0]
        if not np.allclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9):
            mismatch.append((sid, r['T_H'], [c for c, u, v in zip(ALL, a, b) if not np.isclose(u, v, equal_nan=True)][:5]))
check(f'예측 시점 {len(target):,}개 모두 잘라서 다시 계산한 특징과 같다', not mismatch,
      f'다른 시점 {len(mismatch)}개 {mismatch[:3]}' if mismatch else '')


# 4. 경계 상황

section('4. 경계 상황')
base_rows = hr[hr['SEG_HOUR'] < 2]
check('기준 구간(운전 구간 첫 2시간)의 변화량(_D)은 비어 있다', base_rows[[s + '_D' for s in SIGNALS]].isna().all().all(),
      f'{len(base_rows)}개 구간')
first_rows = hr[(hr['SEG'] == 0) & (hr['SEG_HOUR'] < 2)]
check('시나리오 첫 2시간의 처음 대비 변화량(_F)은 비어 있다', first_rows[[s + '_F' for s in SIGNALS]].isna().all().all())

t0 = X[X['T_H'] == 0]
check('시작 시점(t=0)의 센서 특징은 모두 비어 있다', t0[SENSOR].isna().all().all(), f'{len(t0)}개 시점')

# 정비 직후: 예측 시점 직전에 정비가 끝났으면 재가동 후 시간이 짧고 _D가 비어 있어야 한다
just = X[(X['H_SINCE_MNT'] < 2) & (X['N_MNT'] > 0)]
check('정비 끝난 지 2시간 안인 시점은 재가동 후 시간이 2 이하이고 _D가 비어 있다',
      (just['SEG_HOURS'].fillna(0) <= 2).all() and just[[s + '_D' for s in SIGNALS]].isna().all().all(),
      f'{len(just)}개 시점')

# 재가동 횟수 = 정지 후 다시 돈 횟수: 운전 구간 번호(SEG)는 정비가 끝날 때마다 늘어난다
#   고장 복구 직후 부분 정비처럼 쉬는 틈 없이 이어진 정비는 정지 한 번으로 센다
m = mnt.sort_values(['SCENARIO_ID', 'STRT_DT'])
cont = (m['SCENARIO_ID'] == m['SCENARIO_ID'].shift()) & (m['STRT_DT'] <= m['END_DT'].shift())
stops_done = m[~cont.shift(-1, fill_value=False)]          # 이어진 정비 묶음의 마지막 정비
stops_done = stops_done[stops_done['END_DT'] <= stops_done['SCENARIO_ID'].map(START) + pd.Timedelta(hours=71)]
seg_n = hr.groupby('SCENARIO_ID')['SEG'].max()
cmp = pd.concat([seg_n, stops_done.groupby('SCENARIO_ID').size().reindex(seg_n.index).fillna(0)], axis=1,
                keys=['SEG', 'STOP'])
check('시나리오별 운전 구간 번호의 최댓값이 관측 중 끝난 정지(정비 묶음) 횟수와 같다', (cmp['SEG'] == cmp['STOP']).all(),
      f'다른 시나리오 {int((cmp["SEG"] != cmp["STOP"]).sum())}개, 이어진 정비 {int(cont.sum())}건')

two = mnt[mnt['FAILURE_DT'].notna()].groupby('SCENARIO_ID').size()
two = two[two >= 2].index
pos_two = X[X['SCENARIO_ID'].isin(two) & (X['risk24'] == 1)]
check('고장이 2번인 시나리오에서도 양성 시점의 센서 특징이 계산된다',
      len(two) > 0 and pos_two['SCORE'].notna().mean() > 0.8, f'시나리오 {len(two)}개, 양성 시점 {len(pos_two)}개')


# 5. 손 계산

section('5. 손 계산 (원자료에서 직접)')


def manual(sid, t):
    """원자료로 예측 시점 t의 마지막 끝난 1시간 구간 변화량(_D)을 직접 계산한다"""
    raw = load_source('casting', scenarios=[sid], columns=['MEAS_DT', 'RUNNING'] + RAW_COLS)
    raw = raw[raw['MEAS_DT'] < t].sort_values('MEAS_DT').reset_index(drop=True)
    raw['HOUR'] = ((raw['MEAS_DT'] - START[sid]).dt.total_seconds() // 3600).astype(int)
    running = raw['RUNNING'].values == 1
    restart = running & np.r_[False, raw['RUNNING'].values[:-1] == 0]
    raw['SEG'] = np.cumsum(restart)
    hours = []
    for hh, g in raw[running].groupby('HOUR'):
        if len(g) >= 360:
            hours.append({'HOUR': hh, 'SEG': g['SEG'].max(), 'LOAD': g['LOAD'].mean(), 'WD-LOD': g['WD-LOD'].mean(),
                          'OSC-STK': g['OSC-STK'].mean(), 'CONTROL_DELAY': g['CONTROL_DELAY'].mean(),
                          'MLD-AMP': np.nanstd(g['MLD-LVL'].values, ddof=1)})
    hs = pd.DataFrame(hours)
    seg = hs[hs['SEG'] == hs['SEG'].iloc[-1]]
    if len(seg) <= 2:
        return {s: np.nan for s in SIGNALS}
    p = params

    def level(row):
        return {'MLD-AMP': row['MLD-AMP'] - p['slope']['MLD-AMP'] * (row['LOAD'] - p['load_ref']),
                'CONTROL_DELAY': row['CONTROL_DELAY'] - p['slope']['CONTROL_DELAY'] * (row['LOAD'] - p['load_ref']),
                'OSC-STK': row['OSC-STK'] - p['slope']['OSC-STK'] * (row['LOAD'] - p['load_ref']),
                'WDL': (row['WD-LOD'] - p['wdl_intercept']) / row['LOAD']}

    lv = pd.DataFrame([level(r) for _, r in seg.iterrows()])
    return {s: lv[s].iloc[-1] - np.median(lv[s].iloc[:2]) for s in SIGNALS}


picks = pd.concat([
    X[(X['risk24'] == 1) & X['SCORE'].notna()].sample(2, random_state=1),                      # 고장 24시간 안
    X[(X['N_MNT'] > 0) & (X['SEG_HOURS'] > 4) & X['SCORE'].notna()].sample(2, random_state=1),  # 정비 뒤 운전 구간
    X[X['SCENARIO_ID'].isin(set(train) - fault) & X['SCORE'].notna()].sample(1, random_state=1),  # 정상
])
for _, r in picks.iterrows():
    m = manual(r['SCENARIO_ID'], r['OBSERVED_AT'])
    ok = all(np.isclose(m[s], r[s + '_D'], rtol=1e-6, atol=1e-9) for s in SIGNALS)
    check(f'{r["SCENARIO_ID"]} {r["T_H"]:.0f}h: 손 계산 = 함수 결과', ok,
          ', '.join(f'{s} {m[s]:.4f}/{r[s + "_D"]:.4f}' for s in SIGNALS))


# 6. 신호 방향

section('6. 신호 방향 (학습 세트)')
tr = X[(X['SPLIT'] == 'train') & X['SCORE'].notna()]
auc = roc_auc_score(tr['risk24'], tr['SCORE'])
check('평균 점수(SCORE)로 24시간 위험을 잘 구분한다 (AUC > 0.9)', auc > 0.9, f'AUC {auc:.3f}')
pos_med = tr.loc[tr['risk24'] == 1, 'SCORE'].median()
nrm_med = tr.loc[tr['SCENARIO_ID'].isin(fault) == False, 'SCORE'].median()
check('양성 시점의 점수 중앙값이 정상 범위 경계(1)를 넘고, 정상 시나리오 중앙값은 1 미만이다',
      pos_med > 1 > nrm_med, f'양성 {pos_med:.2f}, 정상 {nrm_med:.2f}')

fails = mnt[mnt['FAILURE_DT'].notna() & mnt['SCENARIO_ID'].isin(train)]
show_ids = fails['SCENARIO_ID'].drop_duplicates().iloc[:6].tolist()
fig, axes = plt.subplots(len(show_ids), 1, figsize=(11, 2.2 * len(show_ids)), sharex=True)
for ax, sid in zip(axes, show_ids):
    g = X[X['SCENARIO_ID'] == sid]
    ax.plot(g['T_H'], g['SCORE'], marker='o', label='SCORE (재가동 후 기준)')
    ax.plot(g['T_H'], g['SCORE_F'], marker='s', label='SCORE_F (처음 수준 기준)')
    for m in mnt[mnt['SCENARIO_ID'] == sid].itertuples():
        a = (m.STRT_DT - START[sid]).total_seconds() / 3600
        b = (m.END_DT - START[sid]).total_seconds() / 3600 if pd.notna(m.END_DT) else 72
        ax.axvspan(a, b, color='#eb5757' if pd.notna(m.FAILURE_DT) else '#bdbdbd', alpha=0.3, lw=0)
    ax.scatter(g.loc[g['risk24'] == 1, 'T_H'], np.zeros((g['risk24'] == 1).sum()), color='#eb5757', marker='^',
               zorder=3, label='정답 1')
    ax.axhline(1, color='k', lw=0.5, ls='--')
    ax.set_ylabel(sid, rotation=0, ha='right')
axes[0].legend(ncol=3, fontsize=8)
axes[0].set_title('고장 시나리오의 예측 시점별 점수 (빨강 띠: 고장 후 복구, 회색 띠: 그 밖의 정비, ▲: 정답 1)')
axes[-1].set_xlabel('경과 시간(h)')
fig.tight_layout()
save_figure(fig, '11_check_고장시나리오_점수흐름.png')
print('    (점수가 고장 전에 오르고 정비 뒤 떨어지는지 눈으로 확인)')


# 7. 분포: 정상 시나리오에서 학습 vs 검증

section('7. 분포 (정상 시나리오, 학습 vs 검증)')
nx = X[~X['SCENARIO_ID'].isin(fault)]
drift = {}
for c in SENSOR:
    a, b = nx.loc[nx['SPLIT'] == 'train', c].dropna(), nx.loc[nx['SPLIT'] == 'validation', c].dropna()
    iqr = a.quantile(0.75) - a.quantile(0.25)
    drift[c] = abs(a.median() - b.median()) / iqr if iqr > 0 else 0.0
worst = sorted(drift.items(), key=lambda kv: -kv[1])[:3]
check('센서 특징의 중앙값 차이가 학습 세트 사분위 범위의 0.5배 미만이다', all(v < 0.5 for v in drift.values()),
      '가장 큰 차이: ' + ', '.join(f'{c} {v:.2f}' for c, v in worst))


# 요약

n_fail = sum(not ok for _, ok in results)
print(f'\n결과: {len(results) - n_fail}/{len(results)}개 통과' + (f', {n_fail}개 실패' if n_fail else ''))
for name, ok in results:
    if not ok:
        print(f'  실패: {name}')
sys.exit(1 if n_fail else 0)
