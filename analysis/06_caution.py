"""2단계 경보의 주의 단계: 기준을 학습 세트 교차검증으로 정하고, 검증 세트에서 한 번 확인한다.

    python analysis/06_caution.py

왜 필요한가 (결과보고서 5절)
- 현장에서는 부품 조달 때문에 24시간 경보도 빠듯하다. 경보보다 먼저 "부품 재고를 확인하라"는 주의 단계를 둔다.
- 주의는 교체가 아니라 재고 확인·발주라서 헛걸음 비용이 작다. 대신 너무 자주 울리면 아무도 믿지 않는다.

두 방법을 비교한다
- 방법 A: 경보 모델(24시간 안 고장 확률)의 임계값만 낮춘다
- 방법 B: '지금 열화 중인가'를 맞히는 주의 모델을 따로 둔다 (정답: 진단 정답의 열화 구간)

주의 기준 (src/evaluate.py의 CAUTION_MIN_PRECISION)
- 주의 묶음 중 85% 이상은 실제로 열화 중인 설비여야 한다(주의 정밀도 ≥ 0.85). 그 안에서 가장 낮은 임계값(가장 일찍 주의)
- 처음에는 0.9였다. 현장에서는 주의를 받으면 장비 상태를 확인하고, 설비가 돌아가면 다음 정비 때 그 부분까지 정비한다.
  설비를 세우거나 바로 발주하지 않으므로 헛걸음 비용은 점검 한 번이라 0.85로 낮췄다. 0.8 아래로는 내리지 않는다
- 경보 단계(임계값 0.67, 팀 결정 D)는 그대로 두고, 주의 단계 = 주의 확률 ≥ 주의 임계값 또는 경보

그림 (figures/, 저장소에는 올리지 않음)
  16_caution_주의임계값별_결과.png        주의 임계값을 고른 근거 (학습 세트 교차검증, 방법 A·B)
  17_caution_검증_고장별_주의경보타임라인.png  검증 세트 점진 고장마다 주의·경보가 언제 켜졌는지
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import (CAUTION_MIN_PRECISION, _alarm_episodes, attach_failure_info, caution_report, caution_sweep, choose_caution_threshold,
                          choose_threshold, degraded_now, failure_events)
from src.features import build_dataset
from src.model import CAUTION_FINAL, FEATURES, FINAL, MODELS, fold_features, oof_from_folds
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_answer
from src.preprocess import EQP

setup_korean_font()
A_COLOR, B_COLOR, ALARM_COLOR, CAUTION_COLOR = '#9e9e9e', '#2f5d8a', '#eb5757', '#f2b84a'


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def show(summary, title):
    s = summary
    print(f'\n■ {title}')
    print(f'  주의 임계값 {s["주의 임계값"]:.2f} · 주의 묶음 {s["주의 묶음"]}개 중 열화 아닌 설비 {s["열화 아닌 설비의 주의"]}개'
          f' (주의 정밀도 {s["주의 정밀도"]:.3f}) · 주의 시점 {s["주의 시점"]}개')
    print(f'  점진 고장 {s["점진 고장"]}건 중 주의로 감지 {s["주의로 감지"]}건 · 리드타임 중앙값: 주의 {s["주의 리드타임 중앙값(h)"]:.2f}h,'
          f' 경보 {s["경보 리드타임 중앙값(h)"]:.2f}h')
    print(f'  추가 준비 시간(주의 − 경보): 평균 {s["추가 준비 시간 평균(h)"]:.1f}h · 중앙값 {s["추가 준비 시간 중앙값(h)"]:.1f}h'
          f' · 경보보다 12시간 이상 먼저 {s["경보보다 12시간 이상 먼저"]}건')


# 1. 데이터

section('1. 데이터')
X, params, hr = build_dataset()
train = X[X['SPLIT'] == 'train'].reset_index(drop=True)
valid = X[X['SPLIT'] == 'validation'].reset_index(drop=True)
ev_train = failure_events(train['SCENARIO_ID'].unique().tolist())
ev_valid = failure_events(valid['SCENARIO_ID'].unique().tolist())
deg_tr, deg_va = degraded_now(train), degraded_now(valid)
print(f'학습 {len(train)}개 시점 중 열화 중 {deg_tr.sum()}개 ({deg_tr.mean():.1%}) · '
      f'검증 {len(valid)}개 시점 중 열화 중 {deg_va.sum()}개 ({deg_va.mean():.1%})')
print(f'24시간 안 고장(정답 1) 시점 중 열화 중인 비율: 학습 {deg_tr[train["risk24"] == 1].mean():.1%}')

# 열화가 고장 몇 시간 전에 시작하나 (주의로 얻을 수 있는 준비 시간의 상한)
d_tr = attach_failure_info(train, ev_train)
dg = load_answer('diagnostics').query('EQP_CD == @EQP and HEALTH_STATE == "degraded"')
g_tr = ev_train[ev_train['TYPE'] == '점진 고장'].merge(dg[['SCENARIO_ID', 'START_AT', 'END_AT']],
                                                     left_on=['SCENARIO_ID', 'FAILURE_DT'], right_on=['SCENARIO_ID', 'END_AT'])
deg_lead = (g_tr['FAILURE_DT'] - g_tr['START_AT']).dt.total_seconds() / 3600
print(f'학습 세트 점진 고장 {len(g_tr)}건: 열화 시작 → 고장 {deg_lead.min():.1f}~{deg_lead.max():.1f}h (중앙값 {deg_lead.median():.1f}h)')


# 2. 경보 단계 (04_model.py와 같은 모델·같은 규칙)

section('2. 경보 단계 (고정: 04_model.py와 같은 모델, 임계값 규칙 D)')
folds = fold_features(hr, train)
alarm_cols = FEATURES[FINAL['features']]
a_oof = oof_from_folds(MODELS[FINAL['model']], alarm_cols, train['risk24'].values, folds)
t_alarm = choose_threshold(train, a_oof, criterion='earliest', events=ev_train)
print(f'경보 모델: {FINAL["model"]} · {FINAL["features"]} · 임계값 {t_alarm:.2f}')


# 3. 방법 A: 경보 모델의 임계값만 낮춘다

section('3. 방법 A: 경보 모델의 임계값을 낮춘다 (학습 세트 교차검증)')
sweep_a = caution_sweep(train, a_oof, a_oof, t_alarm, ev_train, grid=np.round(np.arange(0.05, t_alarm + 0.001, 0.01), 2))
t_a = choose_caution_threshold(sweep_a)
sa = caution_report(train, a_oof, t_a, a_oof, t_alarm, ev_train)[0]
show(sa, f'방법 A, 주의 정밀도 ≥ {CAUTION_MIN_PRECISION} 중 가장 낮은 임계값')


# 4. 방법 B: 열화 감지 모델을 따로 둔다

section('4. 방법 B: 열화 감지 모델 (정답 = 지금 열화 중, 학습 세트 교차검증)')
rows, c_oofs = [], {}
for fname, cols in FEATURES.items():
    for mname, make_model in MODELS.items():
        oof = oof_from_folds(make_model, cols, deg_tr.astype(int), folds)
        c_oofs[(mname, fname)] = oof
        rows.append({'모델': mname, '특징': fname, 'CV AP (열화 중)': average_precision_score(deg_tr, oof)})
cv = pd.DataFrame(rows).set_index(['모델', '특징'])
print(cv.round(3).to_string())
best = cv['CV AP (열화 중)'].idxmax()
print(f'\n→ 선택: {best[0]} · {best[1]} (교차검증 AP 최대)')
if best != (CAUTION_FINAL['model'], CAUTION_FINAL['features']):
    print(f'  주의: src/model.py의 CAUTION_FINAL({CAUTION_FINAL})과 다르다. CAUTION_FINAL을 고칠 것')
c_oof = c_oofs[best]
sweep_b = caution_sweep(train, c_oof, a_oof, t_alarm, ev_train)
t_b = choose_caution_threshold(sweep_b)
sb = caution_report(train, c_oof, t_b, a_oof, t_alarm, ev_train)[0]
show(sb, f'방법 B, 주의 정밀도 ≥ {CAUTION_MIN_PRECISION} 중 가장 낮은 임계값')


# 5. 두 방법 비교와 다른 기준

section('5. 비교 (학습 세트 교차검증)')
cmp = pd.DataFrame({'방법 A (경보 임계값 낮춤)': sa, '방법 B (열화 감지 모델)': sb}).drop(['점진 고장'])
print(cmp.to_string())
print('\n참고: 정밀도 기준을 바꾸면 (방법 B)')
for mp in [0.8, 0.85, 0.9, 0.95]:
    t = choose_caution_threshold(sweep_b, mp)
    r = sweep_b.set_index('주의 임계값').loc[t]
    print(f'  정밀도 ≥ {mp}: 임계값 {t:.2f} · 열화 아닌 설비 주의 {int(r["열화 아닌 설비의 주의"])}개 · '
          f'추가 준비 시간 평균 {r["추가 준비 시간 평균(h)"]:.1f}h · 경보보다 12시간 이상 먼저 {int(r["경보보다 12시간 이상 먼저"])}건')


# 6. 검증 세트 (한 번)

section('6. 검증 세트 (교차검증에서 정한 모델·임계값 그대로)')
alarm_model = MODELS[FINAL['model']]().fit(train[alarm_cols].values.astype(float), train['risk24'].values)
c_cols = FEATURES[best[1]]
caution_model = MODELS[best[0]]().fit(train[c_cols].values.astype(float), deg_tr.astype(int))
pa_va = alarm_model.predict_proba(valid[alarm_cols].values.astype(float))[:, 1]
pc_va = caution_model.predict_proba(valid[c_cols].values.astype(float))[:, 1]
print(f'열화 감지 AP (검증): {average_precision_score(deg_va, pc_va):.3f}')
va_b, ev_b = caution_report(valid, pc_va, t_b, pa_va, t_alarm, ev_valid)
va_a, _ = caution_report(valid, pa_va, t_a, pa_va, t_alarm, ev_valid)
show(va_b, f'방법 B (주의 임계값 {t_b:.2f})')
show(va_a, f'참고: 방법 A (주의 임계값 {t_a:.2f})')
print('\n점진 고장별 (방법 B):')
print(ev_b.sort_values('추가 준비 시간(h)', ascending=False).to_string(index=False))

# 열화 아닌 설비의 주의 묶음: 어떤 헛걸음이었나 (다음 열화 시작까지 몇 시간 남았었나)
def false_cautions(d, stage_on, label):
    ep = _alarm_episodes(d, stage_on)
    bad = ep.groupby('EP').filter(lambda x: not x['DEG'].any())
    first = bad.groupby('EP').agg(SCENARIO_ID=('SCENARIO_ID', 'first'), 시작=('OBSERVED_AT', 'min'),
                                  끝=('OBSERVED_AT', 'max'), 시점수=('T_H', 'size'), T_H=('T_H', 'min'))
    nxt = first.merge(dg[['SCENARIO_ID', 'START_AT']], on='SCENARIO_ID', how='left')
    nxt = nxt[nxt['START_AT'] > nxt['끝']].groupby(['SCENARIO_ID', 'T_H'])['START_AT'].min()
    first = first.set_index(['SCENARIO_ID', 'T_H'])
    first['다음 열화 시작까지(h)'] = ((nxt - first['끝']).dt.total_seconds() / 3600).reindex(first.index)
    print(f'\n{label}: 열화 아닌 설비의 주의 묶음 {len(first)}개')
    print(first[['시점수', '다음 열화 시작까지(h)']].reset_index().rename(columns={'T_H': '첫 주의(h)'}).to_string(index=False))
    h = first['다음 열화 시작까지(h)']
    print(f'  → {int((h <= 12).sum())}개는 12시간 안에 실제로 열화가 시작됐다 (열화 직전에 켜진 주의)')
    return first


d_va = attach_failure_info(valid, ev_valid)
d_va['DEG'] = deg_va
d_va['STAGE'] = np.where(pa_va >= t_alarm, '경보', np.where(pc_va >= t_b, '주의', ''))
fc_va = false_cautions(d_va, (d_va['STAGE'] != '').values, '검증 세트')
d_tr['DEG'] = deg_tr
fc_tr = false_cautions(d_tr, (c_oof >= t_b) | (a_oof >= t_alarm), '학습 세트 교차검증')


# 7. 근거 그림

section('7. 근거 그림')

# 16. 주의 임계값별 결과 (학습 세트 교차검증): 위 = 헛걸음, 아래 = 추가 준비 시간
fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
for sw, t, color, name, offset in [(sweep_a, t_a, A_COLOR, 'A: 경보 임계값 낮춤', (-48, -16)),
                                   (sweep_b, t_b, B_COLOR, 'B: 열화 감지 모델', (6, 6))]:
    axes[0].plot(sw['주의 임계값'], sw['열화 아닌 설비의 주의'], color=color, marker='.', label=name)
    axes[1].plot(sw['주의 임계값'], sw['추가 준비 시간 평균(h)'], color=color, marker='.', label=name)
    for ax, col in [(axes[0], '열화 아닌 설비의 주의'), (axes[1], '추가 준비 시간 평균(h)')]:
        v = sw.set_index('주의 임계값').loc[t, col]
        ax.scatter(t, v, color=color, s=80, edgecolor='k', zorder=3)
        ax.annotate(f'{t:.2f}: {v:.0f}' if col.startswith('열화') else f'{t:.2f}: {v:.1f}h', (t, v),
                    textcoords='offset points', xytext=offset, fontsize=9, color=color)
axes[0].set_ylim(0, 40)
axes[0].set_ylabel('열화 아닌 설비에 켜진 주의 (묶음)')
axes[0].legend(fontsize=9)
axes[1].set_ylabel('경보보다 먼저 켜진 시간\n(점진 고장 평균, h)')
axes[1].set_xlabel('주의 임계값')
axes[0].set_title(f'주의 임계값별 결과 (학습 세트 교차검증, 점진 고장 {int(sb["점진 고장"])}건)\n'
                  f'점: 주의 중 {CAUTION_MIN_PRECISION:.0%} 이상 실제 열화(정밀도 ≥ {CAUTION_MIN_PRECISION}) 중 가장 낮은 임계값', fontsize=10)
save_figure(fig, '16_caution_주의임계값별_결과.png')

# 17. 검증 세트 점진 고장마다 주의·경보 타임라인
ev_plot = (ev_b.merge(ev_valid[['WO_NO', 'FAILURE_DT']], on='WO_NO')
           .rename(columns={'주의 리드타임(h)': 'LEAD', '추가 준비 시간(h)': 'EXTRA'})
           .sort_values('LEAD', ascending=False))
deg_start = dg.set_index(['SCENARIO_ID', 'END_AT'])['START_AT']
fig, ax = plt.subplots(figsize=(9, 0.42 * len(ev_plot) + 1.6))
for i, e in enumerate(ev_plot.itertuples()):
    g = d_va[d_va['FAIL_WO'] == e.WO_NO]
    ax.plot([-g['HTF'].max(), 0], [i, i], color='#e0e0e0', lw=1, zorder=1)
    s0 = deg_start.get((e.SCENARIO_ID, e.FAILURE_DT))
    if s0 is not None:
        ax.plot([-(e.FAILURE_DT - s0).total_seconds() / 3600, 0], [i, i], color='#c9b8e8', lw=6, zorder=0, solid_capstyle='butt')
    for stage, kw in [('', dict(facecolor='white', edgecolor='#757575')), ('주의', dict(color=CAUTION_COLOR, edgecolor='#a07a1e')),
                      ('경보', dict(color=ALARM_COLOR))]:
        x = g[g['STAGE'] == stage]
        ax.scatter(-x['HTF'], [i] * len(x), s=40, zorder=3, **kw)
    ax.scatter(0, i, marker='x', color='k', s=40, zorder=3)
ax.set_yticks(range(len(ev_plot)))
ax.set_yticklabels([f'{e.SCENARIO_ID} (+{e.EXTRA:.0f}h)' if pd.notna(e.EXTRA) else e.SCENARIO_ID for e in ev_plot.itertuples()], fontsize=9)
ax.invert_yaxis()
ax.axvline(-24, color='k', lw=0.6, ls=':')
ax.set_xlabel('고장까지 남은 시간 (h) · 보라 띠: 진단상 열화 구간 · 점선: 24시간 전 · ×: 고장 · 괄호: 경보보다 먼저 켜진 시간')
ax.scatter([], [], color=CAUTION_COLOR, edgecolor='#a07a1e', label=f'주의 (열화 확률 ≥ {t_b:.2f})')
ax.scatter([], [], color=ALARM_COLOR, label=f'경보 (24시간 고장 확률 ≥ {t_alarm:.2f})')
ax.scatter([], [], facecolor='white', edgecolor='#757575', label='정상')
ax.legend(loc='lower left', fontsize=9)
ax.set_title(f'검증 세트 점진 고장 {len(ev_plot)}건의 주의·경보 (같은 운전 구간 안 예측 시점만)')
save_figure(fig, '17_caution_검증_고장별_주의경보타임라인.png')
