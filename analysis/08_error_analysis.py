"""오류 분석: 검증 세트의 오탐(오경보)·미탐(놓친 고장)을 센서 시계열 위에 표시하고 원인을 적는다.

    python analysis/08_error_analysis.py      # 먼저 05_predict_test.py로 모델을 저장해 둘 것

고른 사례 (검증 세트, 최종 모델·임계값 그대로)
  S0248  오탐 - 진짜 오경보: 고장 예정이 없는데 울린 경보
  S0277  오탐 - 이른 경보: 고장 24시간보다 일찍 울려 정답상 오경보로 센 경보
  S0235  미탐 - 돌발 고장: 신호 없이 갑자기 고장
  S0234  미탐 - 늦은 감지: 점진 고장인데 첫 경보가 고장 16.75시간 전 (중간발표 6시간 예측에서는 13.75시간 전)
예측 간격은 저장된 모델(models/risk24.joblib)의 step_h를 따른다 (개선 1부터 1시간)

그림 (figures/, 저장소에는 올리지 않음)
  19_error_검증_오탐미탐_시계열.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import build_dataset
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_answer
from src.preprocess import EQP
from src.serving import failure_hours, load_bundle, score_scenarios

setup_korean_font()
CASES = [
    ('S0248', '오탐: 진짜 오경보'),
    ('S0277', '오탐: 이른 경보'),
    ('S0235', '미탐: 돌발 고장'),
    ('S0234', '미탐: 늦은 감지 (점진 고장)'),
]
SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연', 'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
KIND = {'TP': ('#eb5757', 'o', '맞은 경보 (TP)'), 'FP': ('#f2994a', 'X', '오경보 (FP)'),
        'FN': ('#2f5d8a', 's', '놓친 시점 (FN)'), 'TN': ('#bdbdbd', 'o', '맞은 정상 (TN)'),
        '정답 없음': ('white', 'o', '정답을 알 수 없는 시점')}

bundle = load_bundle()
ids = [c for c, _ in CASES]
pts, hr = score_scenarios(ids, bundle)
X, _, _ = build_dataset(step_h=bundle.get('step_h', 6))   # 모델과 같은 예측 간격의 정답 (결과를 알 수 없는 시점은 없다)
lab = X[X['SPLIT'] == 'validation'][['SCENARIO_ID', 'T_H', 'risk24', 'RUL_EVENT', 'RUL_HOURS']]
pts = pts.drop(columns=['risk24', 'RUL_EVENT', 'RUL_HOURS']).merge(lab, on=['SCENARIO_ID', 'T_H'], how='left')
y = pts['risk24']
pts['KIND'] = np.select([y.isna(), pts['ALARM'] & (y == 1), pts['ALARM'] & (y == 0), ~pts['ALARM'] & (y == 1)],
                        ['정답 없음', 'TP', 'FP', 'FN'], default='TN')
mnt = failure_hours(ids)
diag = load_answer('diagnostics').query('EQP_CD == @EQP and SCENARIO_ID in @ids')
start = pts.groupby('SCENARIO_ID')['OBSERVED_AT'].min() - pd.to_timedelta(pts.groupby('SCENARIO_ID')['T_H'].min(), unit='h')
diag = diag.assign(S=(diag['START_AT'] - diag['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600,
                   E=(diag['END_AT'] - diag['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600)
TH = bundle['threshold']

print(f'임계값 {TH:.2f} · 사례별 예측 시점')
for sid, title in CASES:
    p = pts[pts['SCENARIO_ID'] == sid]
    print(f'\n■ {sid} ({title})')
    print(p[['T_H', 'PROB', 'ALARM', 'risk24', 'RUL_HOURS', 'KIND', 'SCORE_F']].round(2).to_string(index=False))
    d = diag[diag['SCENARIO_ID'] == sid]
    print('  진단:', ', '.join(f'{r.HEALTH_STATE} {r.S:.2f}~{r.E:.2f}h' for r in d.itertuples()))
    m = mnt[mnt['SCENARIO_ID'] == sid]
    print('  정비:', ', '.join(f'{r.PART_CD} {r.종류} {r.START_H:.2f}~{r.END_H:.2f}h' + (f' (고장 {r.FAIL_H:.2f}h)' if pd.notna(r.FAIL_H) else '')
                             for r in m.itertuples()) or '없음')

# 그림: 사례마다 위 = 네 신호 표준화 점수(1시간), 아래 = 24시간 고장 확률과 예측 시점 판정
fig, axes = plt.subplots(len(CASES) * 2, 1, figsize=(11, 3.1 * len(CASES) * 2 / 1.25),
                         gridspec_kw={'height_ratios': [1.3, 1] * len(CASES)}, sharex=True)
for i, (sid, title) in enumerate(CASES):
    a1, a2 = axes[2 * i], axes[2 * i + 1]
    h = hr[hr['SCENARIO_ID'] == sid]
    for col, name in SIGNAL_NAMES.items():
        a1.plot(h['HOUR'] + 1, h[col], lw=1, alpha=0.8, label=name)
    a1.plot(h['HOUR'] + 1, h['SCORE'], color='k', lw=2, label='네 신호 평균')
    a1.axhline(1, color='k', ls=':', lw=0.8)
    a1.set_ylabel('표준화 점수')
    p = pts[pts['SCENARIO_ID'] == sid]
    a2.plot(p['T_H'], p['PROB'], color='#757575', lw=1)
    for k, (c, mk, lb) in KIND.items():
        q = p[p['KIND'] == k]
        a2.scatter(q['T_H'], q['PROB'], color=c, marker=mk, s=70, edgecolor='k', zorder=3, label=lb)
    a2.axhline(TH, color='#eb5757', ls='--', lw=1)
    a2.set_ylim(-0.05, 1.05)
    a2.set_ylabel('24시간 고장 확률')
    for ax in (a1, a2):
        for r in diag[(diag['SCENARIO_ID'] == sid) & (diag['HEALTH_STATE'] == 'degraded')].itertuples():
            ax.axvspan(r.S, r.E, color='#c9b8e8', alpha=0.45, lw=0)
        for r in mnt[mnt['SCENARIO_ID'] == sid].itertuples():
            ax.axvspan(r.START_H, min(r.END_H, 72), color='#bdbdbd', alpha=0.5, lw=0)
            if pd.notna(r.FAIL_H):
                ax.axvline(r.FAIL_H, color='#d64545', lw=2)
    a1.set_title(f'{sid} · {title}', loc='left', fontsize=11)
    if i == 0:
        a1.legend(fontsize=8, ncol=5, loc='upper left')
        a2.legend(fontsize=8, ncol=5, loc='upper left')
axes[-1].set_xlim(0, 72)
axes[-1].set_xlabel('시나리오 시작 후 경과 시간 (h) · 보라 띠: 진단상 열화 · 회색 띠: 정비 · 빨간 실선: 고장 · '
                    f'빨간 점선: 경보 기준 {TH:.2f} · 점선(위): 정상 경계 1')
fig.suptitle('검증 세트 오탐·미탐 사례: 센서 시계열과 예측 시점 판정', fontsize=12)
fig.tight_layout()
save_figure(fig, '19_error_검증_오탐미탐_시계열.png')
