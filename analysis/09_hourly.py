"""개선 1: 예측을 6시간마다 → 1시간마다 (학습 세트 교차검증으로 정하고, 검증 세트에서 평가한다).

    python analysis/09_hourly.py      # 약 5분

중간발표 버전(git 태그 midterm)은 6시간마다 예측했다. 그래서 신호가 예측 직후에 넘어서면 경보가 최대 5시간 늦다.
실제 서비스는 1시간 요약이 끝날 때마다 예측할 수 있으므로, 1시간마다 예측하면 경보가 얼마나 빨라지는지 본다.

비교 (같은 모델·특징: src/model.py의 FINAL. 폴드는 1시간 시점 표에서 시나리오 단위 5겹)
  A  6시간 시점으로 학습 · 6시간마다 예측 (중간발표 방식)
  B  6시간 시점으로 학습 · 1시간마다 예측 (모델은 그대로, 예측만 자주)
  C  1시간 시점으로 학습 · 1시간마다 예측
선택 규칙 (결과를 보기 전에 정함): 1시간마다 예측하는 B·C 중 교차검증 AP가 높은 쪽. 임계값은 팀 결정 D 규칙 그대로
(점진 고장을 모두 잡고 진짜 오경보 0인 값 중 가장 낮은 값). 검증 세트는 마지막 7절에서 평가에만 쓴다.

주의할 점
- 정답: 정답 파일(targets)은 6시간 간격이라 features.make_targets_hourly로 1시간 정답을 만든다. 6시간 시점에서는 정답 파일과 같다.
- 시점 단위 숫자(AP, FP 개수)는 시점 수가 6배라 중간발표 숫자와 바로 비교하지 않는다. 비교는 고장 단위(감지, 리드타임, 오경보 묶음)로 한다.
- A의 교차검증 숫자는 04_model.py와 폴드 나눔이 달라(1시간 표 기준) 조금 다르다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import (CAUTION_MIN_PRECISION, _alarm_episodes, attach_failure_info, caution_report, caution_sweep,
                          choose_caution_threshold, choose_threshold, degraded_now, evaluate, failure_events, print_report)
from src.features import make_features, make_targets, make_targets_hourly, normal_scenarios
from src.model import CAUTION_FINAL, FEATURES, FINAL, MODELS, fold_features, time_fold_features
from src.posko_data import load_scenarios
from src.preprocess import apply_params, build_hourly, fit_params


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def summary(d, prob, events, th=None):
    """임계값(없으면 규칙 D로 고름)에서의 고장 단위 결과 한 줄"""
    th = choose_threshold(d, prob, criterion='earliest', events=events) if th is None else th
    r = evaluate(d, prob, th, events)
    e, g = r['event'], r['event_by_type']
    ep = _alarm_episodes(attach_failure_info(d, events), prob >= th)
    per_fail = ep[ep['FAIL_WO'].notna()].groupby('FAIL_WO')['EP'].nunique()
    return r, {
        '시점': len(d), 'AP': round(average_precision_score(d['risk24'], prob), 3), '임계값': th,
        '점진 감지': f'{int(g.loc["점진 고장", "감지"])}/{int(g.loc["점진 고장", "고장"])}',
        '리드타임 중앙값(h)': round(e['리드타임 중앙값(h)'], 1),
        '리드타임 평균(h)': round(r['events'].loc[r['events']['TYPE'] == '점진 고장', '리드타임(h)'].mean(), 1),
        '진짜 오경보 묶음': e['고장과 무관한 경보 묶음: 오경보'],
        '필요한 교체 묶음': e['고장과 무관한 경보 묶음: 필요한 교체'],
        '경보 깜빡임(고장 수)': int((per_fail > 1).sum()),
        '비용(두 부품)': r['cost']['두 부품'],
    }


# 1. 데이터와 1시간 정답

section('1. 데이터와 1시간 정답')
scen = load_scenarios()
train_ids = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID'].tolist()
valid_ids = scen.loc[scen['SPLIT'] == 'validation', 'SCENARIO_ID'].tolist()
hr = build_hourly(train_ids + valid_ids)
tg = make_targets_hourly(train_ids + valid_ids, step_h=1)
ref = make_targets(train_ids + valid_ids)
m = ref.merge(tg, on=['SCENARIO_ID', 'OBSERVED_AT'], how='left', suffixes=('', '_h'))
same = m['risk24_h'].notna().all() and (m['risk24'] == m['risk24_h']).all()
print(f'1시간 정답 {len(tg)}개 시점 (양성 {int(tg["risk24"].sum())}) · 6시간 시점 {len(ref)}개는 정답 파일과 '
      f'{"모두 같음" if same else "다름"}')
if not same:
    sys.exit('1시간 정답이 6시간 정답과 다르다. features.make_targets_hourly를 확인할 것')
train = tg[tg['SPLIT'] == 'train'].reset_index(drop=True)
valid = tg[tg['SPLIT'] == 'validation'].reset_index(drop=True)
ev_train, ev_valid = failure_events(train_ids), failure_events(valid_ids)
six = (train['T_H'] % 6 == 0).values
cols, make = FEATURES[FINAL['features']], MODELS[FINAL['model']]
y = train['risk24'].values


# 2. 학습 세트 교차검증: A / B / C

section('2. 학습 세트 교차검증 (시나리오 단위 5겹, 폴드마다 보정값 다시 정함)')
folds = fold_features(hr, train)
oof6, oof1 = np.zeros(len(train)), np.zeros(len(train))
for tr_idx, te_idx, F in folds:
    X = F[cols].values.astype(float)
    i6 = tr_idx[six[tr_idx]]
    oof6[te_idx] = make().fit(X[i6], y[i6]).predict_proba(X[te_idx])[:, 1]
    oof1[te_idx] = make().fit(X[tr_idx], y[tr_idx]).predict_proba(X[te_idx])[:, 1]
rows, res = {}, {}
for name, sel, p in [('A 6h 학습·6h 예측', six, oof6), ('B 6h 학습·1h 예측', slice(None), oof6),
                     ('C 1h 학습·1h 예측', slice(None), oof1)]:
    d = train[sel].reset_index(drop=True)
    res[name], rows[name] = summary(d, p[sel], ev_train)
print(pd.DataFrame(rows).T.to_string())
choice = 'C 1h 학습·1h 예측' if rows['C 1h 학습·1h 예측']['AP'] >= rows['B 6h 학습·1h 예측']['AP'] else 'B 6h 학습·1h 예측'
train_on_hourly = choice.startswith('C')
TH = rows[choice]['임계값']
oof = oof1 if train_on_hourly else oof6
print(f'\n선택: {choice} (1시간 예측 중 교차검증 AP가 높은 쪽) · 경보 기준 {TH:.2f}')

lead = pd.DataFrame({k: r['events'].set_index('WO_NO')['리드타임(h)'] for k, r in res.items()})
lead = lead[res[choice]['events'].set_index('WO_NO')['TYPE'] == '점진 고장']
dl = lead[choice] - lead['A 6h 학습·6h 예측']
print(f'고장별 첫 경보 비교 (점진 고장 {len(dl)}건, {choice[:1]} − A): 평균 {dl.mean():+.1f}시간, '
      f'빨라짐 {int((dl > 0).sum())}건 · 같음 {int((dl == 0).sum())}건 · 늦어짐 {int((dl < 0).sum())}건')

print('\n임계값 민감도 (선택한 방식, 학습 세트 교차검증)')
sens = {}
for dt in [-0.10, -0.05, 0, 0.05, 0.10]:
    t = round(TH + dt, 2)
    sens[t] = summary(train, oof, ev_train, th=t)[1]
print(pd.DataFrame(sens).T[['점진 감지', '리드타임 중앙값(h)', '진짜 오경보 묶음', '경보 깜빡임(고장 수)']].to_string())


# 3. 확인: 시간 순서 교차검증

section('3. 확인: 시간 순서 교차검증 (앞 시나리오로 학습 → 다음 묶음 평가, 선택에는 쓰지 않음)')
t_folds, mask = time_fold_features(hr, train)
t6, t1 = np.zeros(len(train)), np.zeros(len(train))
for tr_idx, te_idx, F in t_folds:
    X = F[cols].values.astype(float)
    i6 = tr_idx[six[tr_idx]]
    t6[te_idx] = make().fit(X[i6], y[i6]).predict_proba(X[te_idx])[:, 1]
    t1[te_idx] = make().fit(X[tr_idx], y[tr_idx]).predict_proba(X[te_idx])[:, 1]
ev_mask = failure_events(train.loc[mask, 'SCENARIO_ID'].unique().tolist())
trows = {}
for name, sel, p in [('A 6h 학습·6h 예측', six & mask, t6), ('B 6h 학습·1h 예측', mask, t6), ('C 1h 학습·1h 예측', mask, t1)]:
    trows[name] = summary(train[sel].reset_index(drop=True), p[sel], ev_mask)[1]
print(pd.DataFrame(trows).T.to_string())


# 4. 주의 단계 (1시간마다, 06_caution.py와 같은 규칙)

section(f'4. 주의 단계: {CAUTION_FINAL["model"]} · {CAUTION_FINAL["features"]}, 주의 정밀도 ≥ {CAUTION_MIN_PRECISION} 중 가장 낮은 기준')
c_cols, make_c = FEATURES[CAUTION_FINAL['features']], MODELS[CAUTION_FINAL['model']]
deg = degraded_now(train).astype(int)
c_oof = np.zeros(len(train))
for tr_idx, te_idx, F in folds:
    X = F[c_cols].values.astype(float)
    i = tr_idx if train_on_hourly else tr_idx[six[tr_idx]]
    c_oof[te_idx] = make_c().fit(X[i], deg[i]).predict_proba(X[te_idx])[:, 1]
CTH = choose_caution_threshold(caution_sweep(train, c_oof, oof, TH, ev_train))
c_cv, _ = caution_report(train, c_oof, CTH, oof, TH, ev_train)
print(f'주의 기준 {CTH:.2f} · 교차검증 주의 정밀도 {c_cv["주의 정밀도"]:.3f}, 주의로 감지 {c_cv["주의로 감지"]}/{c_cv["점진 고장"]}, '
      f'경보보다 먼저 평균 {c_cv["추가 준비 시간 평균(h)"]:.1f}시간')


# 5. 최종 모델 학습 (학습 세트 전체)

section('5. 최종 모델 학습 (학습 세트 전체, 보정값은 학습 세트 정상 시나리오)')
params = fit_params(hr[hr['SCENARIO_ID'].isin(train_ids)], normal_scenarios(train_ids))
feat = make_features(apply_params(hr, params), tg)
ftr = feat[feat['SPLIT'] == 'train'].reset_index(drop=True)
fva = feat[feat['SPLIT'] == 'validation'].reset_index(drop=True)
fit_rows = slice(None) if train_on_hourly else (ftr['T_H'] % 6 == 0).values
model = make().fit(ftr.loc[fit_rows, cols].values.astype(float), ftr.loc[fit_rows, 'risk24'].values)
caution = make_c().fit(ftr.loc[fit_rows, c_cols].values.astype(float), degraded_now(ftr)[fit_rows].astype(int))
print(f'학습 시점 {int(np.sum(np.ones(len(ftr), bool)[fit_rows]))}개 · 경보 기준 {TH:.2f} · 주의 기준 {CTH:.2f}')


# 6. 중간발표 기준값 (검증 세트, 6시간마다)

section('6. 참고: 중간발표 버전의 검증 세트 결과 (git 태그 midterm, 결과보고서 부록 B-6)')
print('  6시간마다 예측 · 경보 0.67 · 점진 고장 14/14 · 리드타임 중앙값 24.4시간 · 진짜 오경보 1 · 비용(두 부품) 860')


# 7. 검증 세트 평가 (1시간마다, 위에서 정한 모델·기준 그대로)

section('7. 검증 세트 평가 (평가에만 사용)')
pv = model.predict_proba(fva[cols].values.astype(float))[:, 1]
rv, sv = summary(fva, pv, ev_valid, th=TH)
print_report(rv, f'{choice} · 검증 세트')
print(pd.Series(sv).to_string())
cv_v, _ = caution_report(fva, caution.predict_proba(fva[c_cols].values.astype(float))[:, 1], CTH, pv, TH, ev_valid)
print(f'\n주의 단계 · 검증 세트: 주의 정밀도 {cv_v["주의 정밀도"]:.3f} (헛걸음 {cv_v["열화 아닌 설비의 주의"]}/{cv_v["주의 묶음"]}), '
      f'주의로 감지 {cv_v["주의로 감지"]}/{cv_v["점진 고장"]}, 경보보다 먼저 평균 {cv_v["추가 준비 시간 평균(h)"]:.1f}시간')
