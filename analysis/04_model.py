"""연주 24시간 조기경보 모델: 학습 세트 교차검증으로 고르고, 검증 세트에서 평가한다.

    python analysis/04_model.py

팀 결정 A~G (reports/연주_조기경보_결과보고서.md 2-1절, 근거는 부록 B-4)
- 모델·특징 묶음·임계값은 모두 학습 세트의 시나리오 단위 5겹 교차검증으로 정한다. 검증 세트는 평가에만 쓰고 선택에는 쓰지 않는다.
- 임계값: 교차검증 예측에서 점진 고장을 모두 잡고 진짜 오경보가 0인 임계값 중 가장 낮은 것 (오경보 없이 가장 일찍 경보)
  처음에는 "두 부품 교체" 비용 최소로 골랐으나, 비용 규칙이 리드타임을 반영하지 못해 늦은 경보를 골라서 바꿨다
- 모델 선택: 교차검증 AP가 가장 높은 조합 (임계값과 상관없는 순위 능력). 비용은 함께 보고한다
- 특징: 센서 특징만 / 센서 + 정비 이력을 비교한다. 경과 시간은 쓰지 않는다.
- 기준 모델: 학습 없이 네 신호 Z-score 평균(SCORE_F)을 위험도로 쓰는 Z-score 기준 모델을 먼저 같은 규칙으로 평가한다.
- 기준 예제(02_baseline.py)도 같은 규칙으로 다시 평가해 공정하게 비교한다.
- 확인: 시간 순서 교차검증(앞 시나리오로 학습 → 뒤 시나리오 평가)으로도 돌려 선택이 얼마나 흔들리는지 본다(선택에는 쓰지 않음).
- 지표: 혼동행렬, 정밀도, 재현율, F1 (정확도는 쓰지 않는다)
- 대표 숫자는 이른 경보도 오경보로 센다. 재현율은 점진 고장 기준(돌발 고장은 신호가 없어 제외, 결정 B)이고, 돌발 고장을 포함한 값도 함께 출력한다.

그림 (figures/, 저장소에는 올리지 않음)
  12_model_교차검증_모델비교.png      모델·특징 묶음을 고른 근거 (학습 세트 교차검증 AP)
  13_model_검증_정밀도재현율곡선.png  우리 모델과 기준 예제의 검증 세트 비교
  14_model_임계값별_경보결과.png      임계값을 고른 근거 (학습 세트 교차검증)
  15_model_검증_고장별_경보타임라인.png  검증 세트 고장마다 언제 경보가 났는지
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import (attach_failure_info, choose_threshold, evaluate, event_and_cost, failure_events,
                          point_metrics, print_report, unlabeled_alarms)
from src.features import build_dataset, make_features, make_prediction_points
from src.model import FEATURES, MODELS, fold_features, oof_from_folds, oof_predict, time_fold_features
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_scenarios

PART_RULE = '두 부품'
setup_korean_font()
OURS, BASE = '#2f5d8a', '#9e9e9e'          # 그림 색: 우리 모델, 기준 예제


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


# 1. 데이터

section('1. 데이터')
X, params, hr = build_dataset()
train = X[X['SPLIT'] == 'train'].reset_index(drop=True)
valid = X[X['SPLIT'] == 'validation'].reset_index(drop=True)
ev_train = failure_events(train['SCENARIO_ID'].unique().tolist())
ev_valid = failure_events(valid['SCENARIO_ID'].unique().tolist())
print(f'학습 {len(train)}개 시점 (양성 {int(train["risk24"].sum())}, 고장 {len(ev_train)}건) · '
      f'검증 {len(valid)}개 시점 (양성 {int(valid["risk24"].sum())}, 고장 {len(ev_valid)}건)')

def cv_summary(df, oof, events):
    """교차검증 예측으로 임계값을 고르고, 그 임계값에서의 교차검증 성능"""
    th = choose_threshold(df, oof, criterion='earliest', events=events)
    res = evaluate(df, oof, th, events)
    g = res['event_by_type']
    return th, {
        'CV AP': average_precision_score(df['risk24'], oof),
        '임계값': th,
        f'CV 비용 ({PART_RULE})': res['cost'][PART_RULE],
        'CV 정밀도': res['point']['정밀도'], 'CV 재현율': res['point']['재현율'],
        'CV 점진 고장 감지': f'{int(g.loc["점진 고장", "감지"])}/{int(g.loc["점진 고장", "고장"])}',
        'CV 리드타임 중앙값(h)': res['event']['리드타임 중앙값(h)'],
    }


# 2. 학습 세트 교차검증으로 모델·특징 묶음 고르기

section('2. 학습 세트 교차검증 (시나리오 단위 5겹)')
y_tr = train['risk24'].values
folds = fold_features(hr, train)          # 폴드마다 학습 부분만으로 보정값을 다시 정한 특징
print('보정값은 폴드마다 그 폴드의 학습 부분(정상 시나리오)에서 다시 정했다')
rows, oofs = [], {}
for fname, cols in FEATURES.items():
    for mname, make_model in MODELS.items():
        oof = oof_from_folds(make_model, cols, y_tr, folds)
        th, summ = cv_summary(train, oof, ev_train)
        oofs[(mname, fname)] = (oof, th)
        rows.append({'모델': mname, '특징': fname, **summ})
cv = pd.DataFrame(rows).set_index(['모델', '특징'])
print(cv.round(3).to_string())

best = cv['CV AP'].idxmax()
best_model, best_feat = best
print(f'\n→ 선택: {best_model} · {best_feat} (교차검증 AP 최대)')
print('  (src/model.py의 FINAL에 이 조합을 적어 두고 05_predict_test.py가 그대로 쓴다)')

# 확인: 시간 순서 교차검증 (앞 시나리오로 학습 → 바로 뒤 시나리오로 평가). 선택은 위의 5겹 교차검증으로 한다
section('2-1. 확인: 시간 순서 교차검증 (시나리오 6묶음, 앞 묶음들로 학습 → 다음 묶음 평가)')
t_folds, t_mask = time_fold_features(hr, train)
t_sub = train[t_mask].reset_index(drop=True)
ev_t = failure_events(t_sub['SCENARIO_ID'].unique().tolist())
t_rows = []
for fname, cols_ in FEATURES.items():
    for mname, make_model in MODELS.items():
        o = oof_from_folds(make_model, cols_, y_tr, t_folds)[t_mask]
        t_th, t_summ = cv_summary(t_sub, o, ev_t)
        t_rows.append({'모델': mname, '특징': fname, **t_summ})
tcv = pd.DataFrame(t_rows).set_index(['모델', '특징'])
print(f'평가된 시나리오 {t_sub["SCENARIO_ID"].nunique()}개 (첫 묶음 36개는 학습에만 쓰임)')
print(tcv.round(3).to_string())
print(f'\n→ 시간 순서 교차검증 AP 최대: {" · ".join(tcv["CV AP"].idxmax())} ({tcv["CV AP"].max():.3f}) · '
      f'우리 선택({best_model} · {best_feat}) {tcv.loc[best, "CV AP"]:.3f}, 임계값 {tcv.loc[best, "임계값"]:.2f}')


# 3. 기준 예제를 같은 규칙으로 다시 평가

section('3. 기준 예제 (같은 규칙: 학습 세트 교차검증으로 임계값)')
spec = importlib.util.spec_from_file_location('baseline', ROOT / 'analysis' / '02_baseline.py')
bl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bl)
scen = load_scenarios().query('SPLIT != "test"')
obs = bl.load_observations(scen)
rec = bl.make_records(obs, bl.load_targets(scen))
rec_tr = rec[rec['SPLIT'] == 'train'].reset_index(drop=True)
rec_va = rec[rec['SPLIT'] == 'validation'].reset_index(drop=True)


def make_baseline():
    return make_pipeline(SimpleImputer(add_indicator=True),
                         RandomForestClassifier(n_estimators=180, max_depth=7, min_samples_leaf=5,
                                                class_weight='balanced', random_state=31, n_jobs=2))


Btr = np.stack(rec_tr['temporal'])
b_oof = oof_predict(make_baseline, Btr, rec_tr['risk24'].values, rec_tr['SCENARIO_ID'].values)
b_th, b_summ = cv_summary(rec_tr, b_oof, ev_train)
print(pd.Series(b_summ).to_string())


# 3-1. Z-score 기준 모델: 학습 없이 네 신호 Z-score 평균(처음 수준 대비, SCORE_F)을 위험도로 쓴다

section('3-1. Z-score 기준 모델 (학습 없음, 같은 임계값 규칙)')
# 점수의 표준화 기준(정상 분포)도 폴드마다 학습 부분으로 정한 값을 쓴다 (교차검증과 같은 조건)
z_oof = np.full(len(train), np.nan)
for _, te_idx, F in folds:
    z_oof[te_idx] = F['SCORE_F'].values[te_idx]
LOW = -1e9                                         # 시작 직후·정비 직후처럼 점수가 비면 가장 낮은 위험
z_oof = np.nan_to_num(z_oof, nan=LOW)
z_grid = np.unique(np.round(np.quantile(z_oof[z_oof > LOW], np.linspace(0.5, 0.995, 200)), 2))
z_th = choose_threshold(train, z_oof, criterion='earliest', events=ev_train, grid=z_grid)
z_res_cv = evaluate(train, z_oof, z_th, ev_train)
print(f'임계값(점수) {z_th:.2f}: 네 신호의 정상 대비 표준화 점수 평균이 이 값 이상이면 경보')
print(f'교차검증 AP {average_precision_score(train["risk24"], z_oof):.3f} · '
      f'점진 고장 감지 {int(z_res_cv["event_by_type"].loc["점진 고장", "감지"])}/{int(z_res_cv["event_by_type"].loc["점진 고장", "고장"])} · '
      f'리드타임 중앙값 {z_res_cv["event"]["리드타임 중앙값(h)"]:.1f}h')


# 4. 검증 세트 평가 (한 번)

section('4. 검증 세트 평가')
cols = FEATURES[best_feat]
final = MODELS[best_model]().fit(train[cols].values.astype(float), y_tr)
p_val = final.predict_proba(valid[cols].values.astype(float))[:, 1]
th = oofs[best][1]
res = evaluate(valid, p_val, th, ev_valid)
print_report(res, f'우리 모델: {best_model} · {best_feat} (임계값 {th:.2f}, 학습 세트 교차검증으로 결정)')

b_final = make_baseline().fit(Btr, rec_tr['risk24'].values)
b_val = b_final.predict_proba(np.stack(rec_va['temporal']))[:, 1]
b_res = evaluate(rec_va, b_val, b_th, ev_valid)
print_report(b_res, f'기준 예제 (임계값 {b_th:.2f}, 같은 규칙)')

z_val = np.nan_to_num(valid['SCORE_F'].values, nan=LOW)
z_res = evaluate(valid, z_val, z_th, ev_valid)
print_report(z_res, f'Z-score 기준 모델 (점수 임계값 {z_th:.2f}, 같은 규칙)')

# 정답을 알 수 없어 평가에서 빠진 시점의 경보 (운영 비용에는 들어간다)
valid_ids = valid['SCENARIO_ID'].unique().tolist()
all_va = make_features(hr[hr['SCENARIO_ID'].isin(valid_ids)], make_prediction_points(valid_ids))
ua = unlabeled_alarms(all_va, final.predict_proba(all_va[cols].values.astype(float))[:, 1], th, valid)
rec_all = bl.make_records(obs, bl.load_targets(scen, keep_unlabeled=True))
rec_all = rec_all[rec_all['SPLIT'] == 'validation'].reset_index(drop=True)
ub = unlabeled_alarms(rec_all, b_final.predict_proba(np.stack(rec_all['temporal']))[:, 1], b_th, rec_va)
uz = unlabeled_alarms(all_va, np.nan_to_num(all_va['SCORE_F'].values, nan=LOW), z_th, valid)
for name, u, r in [('우리 모델', ua, res), ('기준 예제', ub, b_res), ('Z-score 기준 모델', uz, z_res)]:
    lo, hi = u['비용 범위'][PART_RULE]
    print(f'\n■ {name}: 정답을 알 수 없는 시점 {u["시점"]}개 중 경보 {u["경보 시점"]}개, 평가된 경보와 이어지지 않는 경보 묶음 {u["경보 묶음"]}개'
          f'\n  운영 비용({PART_RULE}) = 평가 가능 구간 {r["cost"][PART_RULE]:.0f} + 미확정 경보 {lo:.0f}~{hi:.0f}'
          f' = {r["cost"][PART_RULE] + lo:.0f}~{r["cost"][PART_RULE] + hi:.0f}')
print('  주의: 미확정 경보가 관측 뒤의 고장을 막았는지, "아무것도 안 함"에서 관측 뒤에 고장이 났는지는 알 수 없다')

# 참고: 다른 임계값 기준 (모두 학습 세트 교차검증에서 고름)
for crit, label in [('cost', f'{PART_RULE} 비용 최소'), ('f1', 'F1 최대')]:
    th_ref = choose_threshold(train, oofs[best][0], criterion=crit, events=ev_train)
    print_report(evaluate(valid, p_val, th_ref, ev_valid), f'참고: 우리 모델, {label} 임계값 {th_ref:.2f}')

section('5. 요약 (검증 세트, 대표 숫자)')
def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


summary = pd.DataFrame({
    name: {'AP': r['point']['AP'], '정밀도': r['point']['정밀도'], '재현율 (돌발 포함)': r['point']['재현율'],
           '재현율 (점진 고장 기준)': r['point']['재현율 (점진 고장만)'],
           'F1 (돌발 포함)': r['point']['F1'],
           'F1 (점진 고장 기준)': f1(r['point']['정밀도'], r['point']['재현율 (점진 고장만)']),
           'TP': r['point']['TP'], 'FP': r['point']['FP'], 'FN': r['point']['FN'],
           '고장 감지': f'{r["event"]["감지"]}/{r["event"]["고장"]}',
           '리드타임 중앙값(h)': r['event']['리드타임 중앙값(h)'],
           f'비용 ({PART_RULE}, 평가 가능 구간)': r['cost'][PART_RULE],
           '미확정 경보 묶음': u['경보 묶음'] if u else '—',
           f'미확정 경보 비용 ({PART_RULE})': '{:.0f}~{:.0f}'.format(*u['비용 범위'][PART_RULE]) if u else '—',
           '비용 (반반, 최악)': r['cost']['반반'],
           '비용 (맞힘, 상한)': r['cost']['맞힘']}
    for name, (r, u) in {'Z-score 기준 모델': (z_res, uz), '기준 예제 (같은 규칙)': (b_res, ub),
                         '우리 모델': (res, ua)}.items()
})
summary['아무것도 안 함'] = pd.Series({f'비용 ({PART_RULE}, 평가 가능 구간)': res['cost_reference']['아무것도 안 함'],
                                  '비용 (반반, 최악)': res['cost_reference']['아무것도 안 함'],
                                  '비용 (맞힘, 상한)': res['cost_reference']['아무것도 안 함']})
print(summary.to_string())


# 6. 근거 그림

section('6. 근거 그림')

# 12. 교차검증 모델 비교: 막대 = 교차검증 AP, 가로선 = 기준 예제
fig, ax = plt.subplots(figsize=(8, 4))
c = cv.reset_index()
labels = [f'{m}\n{f}' for m, f in zip(c['모델'], c['특징'])]
colors = [OURS if (m, f) == best else '#a9bfd6' for m, f in zip(c['모델'], c['특징'])]
bars = ax.barh(labels, c['CV AP'], color=colors)
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
ax.axvline(b_summ['CV AP'], color=BASE, ls='--', lw=1)
ax.text(b_summ['CV AP'], len(c) - 0.4, f' 기준 예제 {b_summ["CV AP"]:.3f}', color='#616161', fontsize=9, va='bottom')
ax.invert_yaxis()
ax.set_xlim(0.5, 1.0)
ax.set_xlabel('교차검증 AP (학습 세트, 시나리오 단위 5겹)')
ax.set_title(f'모델·특징 묶음 비교 → 선택: {best_model} · {best_feat} (진한 막대)')
save_figure(fig, '12_model_교차검증_모델비교.png')

# 13. 검증 세트 정밀도-재현율 곡선과 실제로 쓰는 임계값의 위치
fig, ax = plt.subplots(figsize=(6, 5))
for name, y, p, t, r, color, offset in [
        (f'우리 모델 (AP {res["point"]["AP"]:.3f})', valid['risk24'], p_val, th, res, OURS, (8, 8)),
        (f'기준 예제 (AP {b_res["point"]["AP"]:.3f})', rec_va['risk24'], b_val, b_th, b_res, BASE, (-75, -20))]:
    pr, rc, _ = precision_recall_curve(y, p)
    ax.plot(rc, pr, color=color, label=name, lw=1.8)
    ax.scatter(r['point']['재현율'], r['point']['정밀도'], color=color, s=60, zorder=3, edgecolor='k')
    ax.annotate(f'임계값 {t:.2f}', (r['point']['재현율'], r['point']['정밀도']),
                textcoords='offset points', xytext=offset, fontsize=9, color=color)
ax.axhline(valid['risk24'].mean(), color='k', lw=0.5, ls=':')
ax.text(0.55, valid['risk24'].mean() + 0.015, f'무작위로 경보할 때 ({valid["risk24"].mean():.2f})', fontsize=8)
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.set_xlabel('재현율 (24시간 안 고장 시점 중 경보 비율)')
ax.set_ylabel('정밀도 (경보 중 실제 24시간 안 고장 비율)')
ax.set_title('검증 세트 정밀도-재현율 (점: 실제 쓰는 임계값)')
ax.legend(loc='lower left', bbox_to_anchor=(0, 0.15))
save_figure(fig, '13_model_검증_정밀도재현율곡선.png')

# 14. 임계값별 결과 (학습 세트 교차검증 예측): 임계값을 고른 곳과 같은 자료
oof_best = oofs[best][0]
d_tr = attach_failure_info(train, ev_train)
rows = []
for t in np.round(np.arange(0.50, 0.991, 0.01), 2):   # 0.5 아래는 오경보가 급격히 늘어 생략
    summ, by_type, cost, _, _ = event_and_cost(d_tr, oof_best, t, ev_train)
    pm = point_metrics(d_tr, oof_best, t)
    g = by_type.loc['점진 고장']
    rows.append({'t': t, '점진 고장 놓침': int(g['고장'] - g['감지']), '진짜 오경보': summ['고장과 무관한 경보 묶음: 오경보'],
                 '리드타임': summ['리드타임 중앙값(h)'], '비용': cost[PART_RULE], '정밀도': pm['정밀도'], '재현율': pm['재현율']})
sweep = pd.DataFrame(rows)
th_cost = choose_threshold(train, oof_best, criterion='cost', events=ev_train)
fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
axes[0].plot(sweep['t'], sweep['점진 고장 놓침'], color='#eb5757', label='놓친 점진 고장 (건)', marker='.')
axes[0].plot(sweep['t'], sweep['진짜 오경보'], color='#f2994a', label='진짜 오경보 (경보 묶음)', marker='.')
axes[0].set_ylabel('건수'); axes[0].legend(fontsize=9)
axes[1].plot(sweep['t'], sweep['리드타임'], color=OURS, marker='.')
axes[1].axhline(24, color='k', lw=0.5, ls=':')
axes[1].set_ylabel('리드타임 중앙값 (h)')
axes[2].plot(sweep['t'], sweep['비용'], color='#4a9b5e', marker='.')
axes[2].set_ylabel(f'비용 ({PART_RULE}, 점수)')
axes[2].set_xlabel('임계값')
for ax in axes:
    ax.axvline(th, color=OURS, lw=1.2)
    ax.axvline(th_cost, color=BASE, lw=1, ls='--')
axes[0].set_title(f'임계값별 결과 (학습 세트 교차검증)\n실선: 선택 {th:.2f} (점진 고장 모두 감지 + 진짜 오경보 0 중 가장 낮은 값)'
                  f' · 점선: 비용 최소 {th_cost:.2f}', fontsize=10)
save_figure(fig, '14_model_임계값별_경보결과.png')

# 15. 검증 세트 고장마다 예측 시점별 경보 (가로축: 고장까지 남은 시간)
d_va = attach_failure_info(valid, ev_valid).assign(PROB=p_val, ALARM=p_val >= th)
ev_plot = res['events'].copy()
ev_plot['ORDER'] = ev_plot['TYPE'].map({'점진 고장': 0, '돌발 고장': 1})
ev_plot = ev_plot.sort_values(['ORDER', '리드타임(h)'], ascending=[True, False]).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 0.42 * len(ev_plot) + 1.5))
for i, e in ev_plot.iterrows():
    g = d_va[d_va['FAIL_WO'] == e['WO_NO']]
    ax.plot([-g['HTF'].max(), 0], [i, i], color='#e0e0e0', lw=1, zorder=1)
    ax.scatter(-g.loc[~g['ALARM'], 'HTF'], [i] * (~g['ALARM']).sum(), facecolor='white', edgecolor='#757575', s=40, zorder=2)
    ax.scatter(-g.loc[g['ALARM'], 'HTF'], [i] * g['ALARM'].sum(), color='#eb5757', s=40, zorder=3)
    ax.scatter(0, i, marker='x', color='k', s=40, zorder=3)
lab = [f'{e.SCENARIO_ID} ({e.TYPE})' for e in ev_plot.itertuples()]
ax.set_yticks(range(len(ev_plot))); ax.set_yticklabels(lab, fontsize=9)
ax.invert_yaxis()
ax.axvspan(-24, 0, color='#fde9e9', zorder=0)
ax.axvline(-4, color='k', lw=0.6, ls=':')
ax.set_xlabel('고장까지 남은 시간 (h)  · 분홍 구간: 정답 1(24시간 안 고장) · 점선: 교체에 필요한 4시간 · ×: 고장')
ax.scatter([], [], color='#eb5757', label=f'경보 (확률 ≥ {th:.2f})')
ax.scatter([], [], facecolor='white', edgecolor='#757575', label='경보 없음')
ax.legend(loc='lower left', fontsize=9)
ax.set_title(f'검증 세트 고장 {len(ev_plot)}건의 예측 시점별 경보 (같은 운전 구간 안 예측 시점만)')
save_figure(fig, '15_model_검증_고장별_경보타임라인.png')
