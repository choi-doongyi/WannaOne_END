"""테스트 세트(정답 없음) 24시간 고장위험 예측.

04_model.py·09_hourly.py에서 검증 세트로 평가한 모델을 그대로 쓴다 (보수적 기준: 평가하지 않은 모델은 내보내지 않는다).
- 모델: src/model.py의 FINAL (랜덤포레스트 · 센서 + 정비 이력), 학습 세트로 학습
- 예측 간격: src/model.py의 PRED_STEP_H (개선 1부터 1시간. 중간발표 버전은 6시간, git 태그 midterm)
- 보정값: 학습 세트 정상 시나리오에서 정한 값 (preprocess.fit_params)
- 임계값: 학습 세트 교차검증에서 점진 고장을 모두 잡고 진짜 오경보가 0인 임계값 중 가장 낮은 것
- 2단계 경보의 주의 단계: src/model.py의 CAUTION_FINAL(열화 감지 모델)과 06_caution.py와 같은 규칙의 주의 기준

    python analysis/05_predict_test.py

결과 (둘 다 저장소에는 올리지 않음)
  predictions/test_risk24.csv    테스트 세트 예측
  models/risk24.joblib           학습한 모델·임계값·보정값·검증 세트 결과 (대시보드 app/dashboard.py가 불러온다)

predictions/test_risk24.csv 열
  SCENARIO_ID, OBSERVED_AT, T_H  예측 시점
  RISK24_PROB                    24시간 안 고장 확률
  ALARM                          경보 여부 (RISK24_PROB ≥ 임계값)
  DEGRADED_PROB                  지금 열화 중일 확률 (주의 모델)
  STAGE                          경보 / 주의 / 정상 (경보가 아니고 DEGRADED_PROB ≥ 주의 기준이면 주의)
  SCORE, SCORE_F                 네 신호 평균 점수 (재가동 기준, 처음 수준 기준) — 경보 근거 확인용

참고: 데이터셋 앱용 예측 JSON(posko.export_predictions)은 '관측 교환본'의 해시 값이 필요한데,
받은 데이터 폴더에는 관측 교환본이 없어 CSV로 저장한다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import (CAUTION_MIN_PRECISION, caution_report, caution_sweep, choose_caution_threshold, choose_threshold,
                          degraded_now, evaluate, failure_events, unlabeled_alarms)
from src.dss import DSSConfig, episodes, failure_curve, field_costs, simulate
from src.features import build_dataset, make_features, make_prediction_points, state_outcomes
from src.model import CAUTION_FINAL, FEATURES, FINAL, MODELS, PRED_STEP_H, fold_features, oof_from_folds
from src.posko_data import load_answer, load_scenarios
from src.preprocess import EQP, apply_params, build_hourly
from src.serving import save_bundle

OUT = ROOT / 'predictions' / 'test_risk24.csv'


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


# 1. 예측 시점 규칙 확인: 학습·검증 세트에서 정답 파일의 AT_RISK와 같은가

section('1. 예측 시점 규칙 확인 (학습·검증 세트)')
scen = load_scenarios()
known = scen.loc[scen['SPLIT'] != 'test', 'SCENARIO_ID'].tolist()
test_ids = scen.loc[scen['SPLIT'] == 'test', 'SCENARIO_ID'].tolist()

mine = make_prediction_points(known).set_index(['SCENARIO_ID', 'OBSERVED_AT']).index
tg = load_answer('targets').query('EQP_CD == @EQP and SCENARIO_ID in @known')
tg = tg[(tg['OBSERVED_AT'] - tg['SCENARIO_ID'].map(scen.set_index('SCENARIO_ID')['START_AT'])) < pd.Timedelta(hours=72)]
at_risk = tg[tg['AT_RISK'] == 1].set_index(['SCENARIO_ID', 'OBSERVED_AT']).index
same = mine.sort_values().equals(at_risk.sort_values())
print(f'만든 시점 {len(mine)}개 vs AT_RISK=1 {len(at_risk)}개 → {"일치" if same else "불일치"}')
if not same:
    sys.exit('예측 시점 규칙이 정답 파일과 다르다. make_prediction_points를 확인할 것')


# 2. 학습 세트로 최종 모델 학습, 교차검증으로 임계값

section(f'2. 최종 모델: {FINAL["model"]} · {FINAL["features"]} · {PRED_STEP_H}시간마다 예측')
X, params, hr = build_dataset(step_h=PRED_STEP_H)
train = X[X['SPLIT'] == 'train'].reset_index(drop=True)
cols = FEATURES[FINAL['features']]
make_model = MODELS[FINAL['model']]
Xtr, ytr = train[cols].values.astype(float), train['risk24'].values

folds = fold_features(hr, train)                                       # 폴드마다 보정값을 다시 정한 교차검증
ev_train = failure_events(train['SCENARIO_ID'].unique().tolist())
oof = oof_from_folds(make_model, cols, ytr, folds)
th = choose_threshold(train, oof, criterion='earliest', events=ev_train)
model = make_model().fit(Xtr, ytr)
print(f'학습 시점 {len(train)}개로 학습 · 임계값 {th:.2f} (학습 세트 교차검증)')

# 주의 모델: 정답 = 지금 열화 중 (06_caution.py와 같은 모델·같은 기준 규칙)
c_cols = FEATURES[CAUTION_FINAL['features']]
make_caution = MODELS[CAUTION_FINAL['model']]
deg_tr = degraded_now(train).astype(int)
c_oof = oof_from_folds(make_caution, c_cols, deg_tr, folds)
c_th = choose_caution_threshold(caution_sweep(train, c_oof, oof, th, ev_train))
caution_model = make_caution().fit(train[c_cols].values.astype(float), deg_tr)

# 의사결정 지원(DSS): 경보가 처음 켜진 뒤 고장까지 시간 분포 (학습 세트 교차검증 예측의 경보로 추정, src/dss.py)
so_tr = state_outcomes(train['SCENARIO_ID'].unique().tolist(), step_h=PRED_STEP_H)
p_tr = so_tr.merge(train[['SCENARIO_ID', 'OBSERVED_AT']].assign(P=oof), on=['SCENARIO_ID', 'OBSERVED_AT'], how='left')['P']
curve = failure_curve(episodes(so_tr, p_tr.fillna(0).values, th))   # 정답이 없는 관측 끝 시점은 경보 판단에서 뺀다
print('DSS: 경보 뒤 h시간 안 고장 확률 (학습 세트) ' + ', '.join(f'{h}h {curve[h]:.2f}' for h in [6, 12, 24, 36]))
print(f'주의 모델: {CAUTION_FINAL["model"]} · {CAUTION_FINAL["features"]} · 주의 기준 {c_th:.2f} '
      f'(학습 세트 교차검증, 주의 정밀도 ≥ {CAUTION_MIN_PRECISION} 중 가장 낮은 값)')


# 3. 테스트 세트 특징과 예측

section('3. 테스트 세트 예측')
hr_test = apply_params(build_hourly(test_ids), params)          # 보정값은 학습 세트에서 정한 그대로
pts = make_prediction_points(test_ids, step_h=PRED_STEP_H)
Xte = make_features(hr_test, pts)
prob = model.predict_proba(Xte[cols].values.astype(float))[:, 1]

pred = Xte[['SCENARIO_ID', 'OBSERVED_AT', 'T_H']].copy()
pred['RISK24_PROB'] = prob.round(4)
pred['ALARM'] = (prob >= th).astype(int)
c_prob = caution_model.predict_proba(Xte[c_cols].values.astype(float))[:, 1]
pred['DEGRADED_PROB'] = c_prob.round(4)
pred['STAGE'] = np.where(prob >= th, '경보', np.where(c_prob >= c_th, '주의', '정상'))
pred['SCORE'] = Xte['SCORE'].round(3)
pred['SCORE_F'] = Xte['SCORE_F'].round(3)
OUT.parent.mkdir(exist_ok=True)
pred.to_csv(OUT, index=False, encoding='utf-8-sig')

print(f'시나리오 {pred["SCENARIO_ID"].nunique()}개, 예측 시점 {len(pred)}개 → {OUT.relative_to(ROOT)}')
print(f'경보 {int(pred["ALARM"].sum())}개 시점 ({pred["ALARM"].mean():.1%}), '
      f'경보가 한 번이라도 난 시나리오 {pred.loc[pred["ALARM"] == 1, "SCENARIO_ID"].nunique()}개')
print(f'주의(경보 제외) {int((pred["STAGE"] == "주의").sum())}개 시점, '
      f'주의 이상이 한 번이라도 난 시나리오 {pred.loc[pred["STAGE"] != "정상", "SCENARIO_ID"].nunique()}개')
print('센서 특징이 비어 있는 시점(시작 시점·정비 직후):', int(Xte['SCORE'].isna().sum()))

# 검증 세트와 비교: 경보 비율이 크게 다르면 데이터가 달라졌을 가능성을 확인한다
valid = X[X['SPLIT'] == 'validation']
v_alarm = (model.predict_proba(valid[cols].values.astype(float))[:, 1] >= th).mean()
print(f'참고: 같은 모델의 검증 세트 경보 비율 {v_alarm:.1%} (정답 시점 기준이라 정확히 같은 조건은 아님)')
print('\n확률 분포 (테스트 세트)')
print(pd.Series(prob).describe(percentiles=[0.5, 0.9, 0.95]).round(3).to_string())


# 4. 모델 저장 (대시보드용)

section('4. 모델 저장')
res = evaluate(valid.reset_index(drop=True), model.predict_proba(valid[cols].values.astype(float))[:, 1], th)
ev = res['events'][['SCENARIO_ID', 'WO_NO', 'FAILURE_DT', 'TYPE', '감지', '리드타임(h)']]
v_ids = valid['SCENARIO_ID'].unique().tolist()
all_va = make_features(hr[hr['SCENARIO_ID'].isin(v_ids)], make_prediction_points(v_ids, step_h=PRED_STEP_H))
ua = unlabeled_alarms(all_va, model.predict_proba(all_va[cols].values.astype(float))[:, 1], th, valid)
va = valid.reset_index(drop=True)
c_sum, c_ev = caution_report(va, caution_model.predict_proba(va[c_cols].values.astype(float))[:, 1], c_th,
                             model.predict_proba(va[cols].values.astype(float))[:, 1], th)
so_va = state_outcomes(v_ids, step_h=PRED_STEP_H)
f_va = make_features(hr[hr['SCENARIO_ID'].isin(v_ids)], so_va)
p_va = model.predict_proba(f_va[cols].values.astype(float))[:, 1]
dss_val = {p: simulate(so_va, p_va, th, curve, DSSConfig(), policy=p)['summary'] for p in ['regular', 'now', 'dss']}
# 현장 가정(만 원)으로도 같은 시뮬레이션 (대시보드의 비용 기준 전환용)
dss_val_field = {p: simulate(so_va, p_va, th, curve, DSSConfig(costs=field_costs()), policy=p)['summary']
                 for p in ['regular', 'now', 'dss']}
print('DSS 정책 비용 (검증 세트): ' + ' · '.join(f'{r["정책"]} {r["비용"]:.0f}' for r in dss_val.values()))
print('  현장 가정(만 원): ' + ' · '.join(f'{r["정책"]} {r["비용"]:,.0f}' for r in dss_val_field.values()))

bundle = {
    'caution': {'model': caution_model, 'threshold': c_th, 'features': c_cols, 'final': CAUTION_FINAL,
                'min_precision': CAUTION_MIN_PRECISION, 'validation': c_sum, 'validation_events': c_ev},
    'model': model, 'threshold': th, 'features': cols, 'params': params, 'final': FINAL, 'step_h': PRED_STEP_H,
    # 입력 감시: 학습 세트 특징의 0.1~99.9% 범위 (결측은 무시). 서비스에서 이 범위 밖이면 현장 확인을 띄운다
    'feature_range': {'lo': train[cols].quantile(0.001).tolist(), 'hi': train[cols].quantile(0.999).tolist()},
    'dss': {'curve': curve, 'config': DSSConfig(), 'validation': dss_val, 'validation_field': dss_val_field},
    'created_at': pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M'),
    'n_train_points': len(train),
    'validation': {'point': res['point'], 'event': res['event'], 'cost': res['cost'],
                   'cost_reference': res['cost_reference'], 'events': ev,
                   'unlabeled': {k: ua[k] for k in ['시점', '경보 시점', '경보 묶음', '비용 범위']}},
}
path = save_bundle(bundle)
print(f'{path.relative_to(ROOT)} 저장 · 검증 세트 AP {res["point"]["AP"]:.3f}, '
      f'고장 감지 {res["event"]["감지"]}/{res["event"]["고장"]}, 비용(두 부품) {res["cost"]["두 부품"]:.0f}')
print(f'  주의 단계: 검증 세트 주의 정밀도 {c_sum["주의 정밀도"]:.3f}, 경보보다 먼저 평균 {c_sum["추가 준비 시간 평균(h)"]:.1f}시간')
