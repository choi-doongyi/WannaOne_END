"""DSS 정책 시뮬레이션: 같은 데이터·같은 모델 예측으로 세 정책의 비용을 비교한다 (src/dss.py, 결과보고서 7절).

    python analysis/10_dss_policy.py      # 약 5분

부품 조달 (5절): 데이터에 재고·조달 기간이 없어 가정값을 미리 정해 넣는다(강사 피드백 2026-10-15, 값은 정건우 현장 경험).
  설비마다 예비 밸브 2 · 진동장치 1, 정상 조달 72시간, 긴급 조달 12시간(부품값 +50%). 1~4절도 이 기본값으로 계산한다.

정책
  ① 정기 교체만     경보를 쓰지 않는다 (데이터에 있던 정기 교체만)
  ② 경보 즉시 교체  경보가 켜지면 바로 두 부품 교체 (보고서의 비용 규칙)
  ③ DSS            경보가 켜지면 '지금 교체'와 '다음 정기 교체 때 두 부품 함께'를 기대 비용으로 비교.
                   기다리는 동안 고장 날 확률이 안전 기준(기본 0.2)을 넘으면 비용과 상관없이 지금 교체
모델 (두 모델에 같은 정책을 적용해 모델과 정책의 몫을 나눠 본다)
  중간발표  6시간마다 예측, 경보 기준 0.67 (git 태그 midterm과 같은 모델)
  개선 1    1시간마다 예측, 경보 기준 0.56 (analysis/09_hourly.py)

순서 (지금까지의 원칙)
  - 정책의 값(안전 기준 0.2, 기다릴 수 있는 최대 48시간, 다른 부품이 원인일 확률 0.5)은 결과를 보기 전에 정한 기본값이다.
    팀이 정할 값이라(결과보고서 부록 D-2) 학습 세트에서 민감도를 함께 보인다.
  - '경보 뒤 고장까지 시간' 분포는 학습 세트 교차검증 예측의 경보로만 추정한다.
  - 검증 세트는 마지막에 평가에만 쓴다. 결과는 "시뮬레이션에 따른 예상 비용"이다(실제 돈이 아닌 데이터셋 비용 점수).
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.dss import Costs, DSSConfig, PartsPolicy, episodes, failure_curve, field_costs, simulate
from src.evaluate import evaluate, failure_events
from src.features import TARGET_COLS, make_features, make_targets_hourly, normal_scenarios, state_outcomes
from src.model import FEATURES, FINAL, MODELS, fold_features
from src.posko_data import load_scenarios
from src.preprocess import apply_params, build_hourly, fit_params

MODELS_USED = {'중간발표 (6시간)': {'step_h': 6, 'th': 0.67}, '개선 1 (1시간)': {'step_h': 1, 'th': 0.56}}
POLICIES = ['regular', 'now', 'dss']
CFG = DSSConfig()


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def with_labels(ids, step_h):
    """모든 예측 시점(정답을 알 수 없는 시점 포함)에 정답 열을 붙인다. 정답이 없는 시점은 risk24가 NaN"""
    pts = state_outcomes(ids, step_h=step_h)
    lab = make_targets_hourly(ids, step_h=step_h)[['SCENARIO_ID', 'OBSERVED_AT', 'risk24', 'RUL_EVENT', 'RUL_HOURS']]
    pts = pts.merge(lab, on=['SCENARIO_ID', 'OBSERVED_AT'], how='left')
    pts['T_H'] = pts['T_H'].astype(float)
    return pts


def run(pts, prob, th, curve, cfg=CFG):
    return {p: simulate(pts, prob, th, curve, cfg, policy=p) for p in POLICIES}


# 1. 데이터

section('1. 데이터')
scen = load_scenarios()
train_ids = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID'].tolist()
valid_ids = scen.loc[scen['SPLIT'] == 'validation', 'SCENARIO_ID'].tolist()
hr = build_hourly(train_ids + valid_ids)
params = fit_params(hr[hr['SCENARIO_ID'].isin(train_ids)], normal_scenarios(train_ids))
hr_fit = apply_params(hr, params)
cols, make = FEATURES[FINAL['features']], MODELS[FINAL['model']]
print(f'학습 {len(train_ids)}개 · 검증 {len(valid_ids)}개 시나리오 · DSS 기본값 {CFG}')


# 2. 학습 세트: 교차검증 예측으로 '경보 뒤 고장까지 시간' 분포와 정책 비용

section('2. 학습 세트 교차검증 (정책 기본값 그대로)')
curves, train_res, train_inputs = {}, {}, {}
for name, m in MODELS_USED.items():
    pts = with_labels(train_ids, m['step_h'])
    oof = np.full(len(pts), np.nan)
    lab = pts['risk24'].notna().values
    for tr_idx, te_idx, F in fold_features(hr, pts[TARGET_COLS].assign(risk24=pts['risk24'].fillna(-1))):
        X = F[cols].values.astype(float)
        fit = tr_idx[lab[tr_idx]]
        oof[te_idx] = make().fit(X[fit], pts['risk24'].values[fit].astype(int)).predict_proba(X[te_idx])[:, 1]
    curves[name] = failure_curve(episodes(pts, oof, m['th']))
    train_inputs[name] = (pts, oof)
    res = run(pts, oof, m['th'], curves[name])
    for p, r in res.items():
        train_res[(name, r['summary']['정책'])] = r
    if name.startswith('개선'):
        train_pts, train_oof = pts, oof
c = pd.DataFrame(curves).loc[[3, 6, 12, 24, 36, 48]]
print('경보가 처음 켜진 뒤 h시간 안에 고장 날 확률 (학습 세트, 정비가 없었다면)')
print(c.round(2).T.to_string())
print('\n정책별 비용 (학습 세트 교차검증, 점수)')
print(pd.DataFrame({k: {kk: vv for kk, vv in r['summary'].items() if kk != '정책'} for k, r in train_res.items()}).T.to_string())

print('\n민감도: 안전 기준 (개선 1 모델, 학습 세트)')
sens = {}
for s_max in [0.05, 0.1, 0.2, 0.3, 0.5]:
    r = simulate(train_pts, train_oof, MODELS_USED['개선 1 (1시간)']['th'], curves['개선 1 (1시간)'],
                 replace(CFG, safety_max=s_max), policy='dss')['summary']
    sens[s_max] = {k: r[k] for k in ['비용', '고장', '지금 교체', '정기 교체 때', '기다리다 고장']}
print(pd.DataFrame(sens).T.to_string())


# 3. 검증 세트 (학습 세트로 학습한 모델, 학습 세트에서 추정한 분포, 기본값 그대로)

print('\n민감도: 준비 시간 (학습 세트)')
prep_tr = {}
for prep_h in [4, 12, 24]:
    for name, m in MODELS_USED.items():
        pts, prob = train_inputs[name]
        for p in ['now', 'dss']:
            r = simulate(pts, prob, m['th'], curves[name], replace(CFG, prep_h=prep_h), policy=p)['summary']
            prep_tr[(prep_h, name, r['정책'])] = {k: r[k] for k in ['비용', '고장', '지금 교체', '정기 교체 때']}
print(pd.DataFrame(prep_tr).T.to_string())


print('\n민감도: 고장 벌점 (안전·품질·생산 영향의 대리값, 데이터셋 규칙은 20 → 고장 시 손실 66). 준비 시간 4시간 / 12시간, ③ DSS (학습 세트)')
pen = {}
for prep_h in [4, 12]:
    for penalty in [20, 50, 100, 200]:
        cfg = replace(CFG, prep_h=prep_h, costs=Costs(failure_penalty=penalty))
        row = {}
        for name, m in MODELS_USED.items():
            pts, prob = train_inputs[name]
            r = simulate(pts, prob, m['th'], curves[name], cfg, policy='dss')['summary']
            row[f'{name} 비용'], row[f'{name} 고장'] = r['비용'], r['고장']
        row['1시간 − 6시간'] = row['개선 1 (1시간) 비용'] - row['중간발표 (6시간) 비용']
        pen[(prep_h, penalty, cfg.costs.failure_loss)] = row
print(pd.DataFrame(pen).T.rename_axis(['준비 시간', '고장 벌점', '고장 시 손실']).to_string())


section('3. 검증 세트 (평가에만 사용)')
val_res, val_eps, val_inputs = {}, {}, {}
for name, m in MODELS_USED.items():
    tr = make_targets_hourly(train_ids, step_h=m['step_h'])
    ftr = make_features(hr_fit, tr)
    model = make().fit(ftr[cols].values.astype(float), ftr['risk24'].values)
    pts = with_labels(valid_ids, m['step_h'])
    fva = make_features(hr_fit, pts[TARGET_COLS])
    prob = model.predict_proba(fva[cols].values.astype(float))[:, 1]
    # 확인: 정답이 있는 시점의 '경보 즉시 교체' 비용이 평가 함수와 같은 틀인지 (깜빡임 중복 제외 차이만 있어야 함)
    labeled = pts['risk24'].notna().values
    ev_cost = evaluate(fva[labeled].reset_index(drop=True), prob[labeled], m['th'],
                       failure_events(valid_ids))['cost']['두 부품']
    val_inputs[name] = (pts, prob)
    res = run(pts, prob, m['th'], curves[name])
    for p, r in res.items():
        val_res[(name, r['summary']['정책'])] = r
    val_eps[name] = res['dss']['episodes']
    print(f'{name}: 평가 함수의 즉시 교체 비용 {ev_cost:.0f} · 시뮬레이션 ② {res["now"]["summary"]["비용"]:.0f} '
          f'(운전 구간마다 한 번만 교체하므로 깜빡임 중복이 빠짐)')
print('\n정책별 비용 (검증 세트, 점수)')
vt = pd.DataFrame({k: {kk: vv for kk, vv in r['summary'].items() if kk != '정책'} for k, r in val_res.items()}).T
print(vt.to_string())
for name in MODELS_USED:
    base, now, dss = (vt.loc[(name, p), '비용'] for p in ['① 정기 교체만', '② 경보 즉시 교체', '③ DSS'])
    print(f'  {name}: ①→② {now - base:+.0f}점, ②→③ {dss - now:+.0f}점, ①→③ {dss - base:+.0f}점 ({(dss - base) / base:+.0%})')

print('\n민감도: 경보 뒤 교체를 끝내기까지 필요한 준비 시간 (부품 조달 포함, 가정값. 데이터셋 규칙은 4시간)')
print('  리드타임의 가치는 비용 규칙(4시간)에서는 드러나지 않는다. 현장에서는 부품 조달 때문에 24시간도 빠듯하다(결과보고서 7-4절)')
prep = {}
for prep_h in [4, 12, 24]:
    cfg = replace(CFG, prep_h=prep_h)
    for name, m in MODELS_USED.items():
        pts, prob = val_inputs[name]
        for p in ['now', 'dss']:
            r = simulate(pts, prob, m['th'], curves[name], cfg, policy=p)['summary']
            prep[(prep_h, name, r['정책'])] = {k: r[k] for k in ['비용', '고장', '지금 교체', '정기 교체 때', '기다리다 고장']}
print(pd.DataFrame(prep).T.to_string())

print('\n민감도: 고장 벌점 (검증 세트, ③ DSS)')
pen = {}
for prep_h in [4, 12]:
    for penalty in [20, 50, 100, 200]:
        cfg = replace(CFG, prep_h=prep_h, costs=Costs(failure_penalty=penalty))
        row = {}
        for name, m in MODELS_USED.items():
            pts, prob = val_inputs[name]
            r = simulate(pts, prob, m['th'], curves[name], cfg, policy='dss')['summary']
            row[f'{name} 비용'], row[f'{name} 고장'] = r['비용'], r['고장']
        row['1시간 − 6시간'] = row['개선 1 (1시간) 비용'] - row['중간발표 (6시간) 비용']
        pen[(prep_h, penalty, cfg.costs.failure_loss)] = row
print(pd.DataFrame(pen).T.rename_axis(['준비 시간', '고장 벌점', '고장 시 손실']).to_string())

section('4. 현장 가정 (만 원, 정건우 현장 경험): 소모품 100 · 중요 부품 개당 200 · 공임 파트당 20 · 인력 3/시간·명(4인 1팀) · 정지 1시간 500 · 고장 추가 300')
FIELD = replace(CFG, costs=field_costs())
c = FIELD.costs
print(f'지금 두 부품 교체 {2 * c.part + c.replace_fixed + c.replace_h * c.downtime_h:,.0f} · 정기 교체 때 추가 {c.part:,.0f} · 고장 {c.failure_loss:,.0f} (만 원)')
for label, inputs in [('학습 세트 교차검증', train_inputs), ('검증 세트', val_inputs)]:
    rows = {}
    for name, m in MODELS_USED.items():
        pts, prob = inputs[name]
        res = {p: simulate(pts, prob, m['th'], curves[name], FIELD, policy=p)['summary'] for p in POLICIES}
        for p, r in res.items():
            rows[(name, r['정책'])] = {'비용 (만 원)': r['비용'], '예상 절감액 (① 대비)': res['regular']['비용'] - r['비용'],
                                      '고장': r['고장'], '정기 교체 때': r['정기 교체 때'], '계획 밖 정지(h)': r['계획 밖 정지(h)']}
    print(f'\n{label}')
    print(pd.DataFrame(rows).T.to_string())

print('\n민감도: 정지 1시간 생산 차질 비용 (만 원, 검증 세트, ③ DSS, 개선 1 모델)')
sens = {}
for dt in [0, 100, 500, 1000]:
    cfg = replace(CFG, costs=field_costs(downtime_h=dt))
    pts, prob = val_inputs['개선 1 (1시간)']
    res = {p: simulate(pts, prob, MODELS_USED['개선 1 (1시간)']['th'], curves['개선 1 (1시간)'], cfg, policy=p)['summary'] for p in POLICIES}
    sens[dt] = {'① 정기 교체만': res['regular']['비용'], '② 경보 즉시 교체': res['now']['비용'], '③ DSS': res['dss']['비용'],
                '③ 절감액 (① 대비)': res['regular']['비용'] - res['dss']['비용']}
print(pd.DataFrame(sens).T.rename_axis('생산 차질 (만 원/시간)').to_string())

print('\nDSS 판단 (검증 세트, 개선 1 모델)')
e = val_eps['개선 1 (1시간)']
print(e[['SCENARIO_ID', '경보 시각(h)', '정기 교체까지(h)', '판단', '결과', '비용', '근거']].to_string(index=False))


# 5. 부품 조달 가정 (데이터에 없음, 미리 정한 값과 민감도)

section('5. 부품 조달: 예비 재고 밸브 2 · 진동장치 1, 정상 조달 72시간, 긴급 조달 12시간(부품값 +50%)')
print('정상 조달 72시간은 72시간 시나리오 안에 도착하지 못해, 재고가 모자라면 긴급 조달로 계산한다(웃돈 + 부품이 올 때까지 대기).')
m1 = MODELS_USED['개선 1 (1시간)']
for label, inputs in [('학습 세트 교차검증', train_inputs), ('검증 세트', val_inputs)]:
    pts, prob = inputs['개선 1 (1시간)']
    for cname, costs in [('점수', Costs()), ('만 원', field_costs())]:
        rows = {}
        for sname, parts in [('재고 무제한 (이전 계산)', PartsPolicy(99, 99)), ('기본: 밸브 2 · 진동장치 1', PartsPolicy()),
                             ('밸브 1 · 진동장치 1', PartsPolicy(1, 1)), ('재고 없음', PartsPolicy(0, 0))]:
            cfg = replace(CFG, costs=costs, parts=parts)
            res = {p: simulate(pts, prob, m1['th'], curves['개선 1 (1시간)'], cfg, policy=p)['summary'] for p in POLICIES}
            for p, r in res.items():
                rows[(sname, r['정책'])] = {'비용': r['비용'], '고장': r['고장'], '긴급 조달': r['긴급 조달'],
                                            '긴급 조달 비용': r['긴급 조달 비용'], '계획 밖 정지(h)': r['계획 밖 정지(h)']}
        print(f'\n{label} · {cname} (개선 1 모델)')
        print(pd.DataFrame(rows).T.to_string())

print('\n민감도: 긴급 조달 기간 (검증 세트, 만 원, ③ DSS, 재고 기본값과 재고 없음)')
pts, prob = val_inputs['개선 1 (1시간)']
rush = {}
for rush_h in [6, 12, 24]:
    for sname, base in [('기본 재고', PartsPolicy()), ('재고 없음', PartsPolicy(0, 0))]:
        cfg = replace(CFG, costs=field_costs(), parts=replace(base, rush_h=rush_h))
        r = simulate(pts, prob, m1['th'], curves['개선 1 (1시간)'], cfg, policy='dss')['summary']
        rush[(rush_h, sname)] = {'비용 (만 원)': r['비용'], '고장': r['고장'], '긴급 조달': r['긴급 조달']}
print(pd.DataFrame(rush).T.rename_axis(['긴급 조달(h)', '재고']).to_string())

