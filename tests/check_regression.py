"""회귀 점검: 코드를 고친 뒤에도 보고서·발표의 숫자와 서비스 동작이 그대로인지 확인한다.

    python analysis/05_predict_test.py      # 모델을 다시 만든 경우
    python tests/check_regression.py        # 약 1분

점검 항목 (하나라도 실패하면 종료 코드 1)
  1. 모델 파일: 해시·필수 값·만든 환경(meta)이 맞는가
  2. 대표 숫자: 예측 간격 1시간, 경보 기준 0.56, 주의 기준 0.57, 검증 세트 AP 0.866, 고장 감지 14/16, 비용(두 부품) 1020
  3. 서비스 예측: 대시보드 경로(serving.score_scenarios)의 테스트 세트 예측이 predictions/test_risk24.csv와 같은가
     (테스트 5,028개 시점 중 경보 554개, 경보 시나리오 18개)
  4. 경보 이유: 랜덤포레스트 기여도의 기준값 + 합이 위험 확률과 같은가
  5. 비용 곡선: 대시보드 비용 계산기가 평가 함수(src/evaluate.py)와 같은 비용을 내는가 (손익분기 0.60)
  6. 안전장치: 모르는 정비 유형, 빠진 특징, 바뀐 모델 파일을 조용히 넘기지 않고 멈추는가
  7. 1시간 정답: make_targets_hourly가 6시간 시점에서 정답 파일 기준(make_targets)과 같은가
  8. 의사결정 지원(DSS): 모델 파일의 정책 비용(검증 세트 ① 1056 ② 998 ③ 918, 부품 재고 가정 포함)과, 정기 교체 직전이면 기다리고 준비가 안 되면 지금 교체하는가
  9. 보안 (결과보고서 9절·부록 E): 클라우드용 데이터(app_data)가 서비스 예측과 같고 원자료·모델 객체가 없는가, 바뀐 파일을 막는가,
     비밀번호 확인이 맞는가, 입력 감시 범위가 모델 파일에 있는가

숫자를 일부러 바꾼 경우(모델 개선 등)에는 보고서를 고친 뒤 EXPECTED도 함께 고친다.
"""

import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.features import _maintenance_features, make_targets, make_targets_hourly
from src.posko_data import load_scenarios
from src.serving import MODEL_PATH, _forest_contributions, _matrix, cost_curve, explain, load_bundle, score_scenarios

# 대표 숫자. 중간발표(git 태그 midterm, 6시간 예측): 경보 0.67, 주의 0.38, AP 0.860, 비용 860, 손익분기 0.67, 테스트 833시점·경보 83
# 개선 1(2026-10-15, 1시간 예측, analysis/09_hourly.py)부터 아래 값
EXPECTED = {
    'threshold': 0.56, 'caution_threshold': 0.57, 'AP': 0.866, '감지': 14, '고장': 16, '비용 두 부품': 1020,
    'test_points': 5028, 'test_alarms': 554, 'test_alarm_scenarios': 18, 'breakeven_p': 0.60, 'step_h': 1,
    'dss_cost': {'regular': 1056, 'now': 998, 'dss': 918},     # 부품 재고 가정 포함 (재고 무제한이면 1056 / 980 / 900)
}

results = []


def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(f'  [{"통과" if ok else "실패"}] {name}' + (f' — {detail}' if detail else ''))


def raises(fn, error):
    try:
        fn()
    except error:
        return True
    return False


print('1. 모델 파일')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('ignore')                 # 라이브러리 내부 경고(joblib 불러오기 중 NumPy 사용 중단 안내)는 빼고
    warnings.simplefilter('always', UserWarning)    # load_bundle이 내는 환경 경고만 본다
    bundle = load_bundle()
    w = [x for x in w if issubclass(x.category, UserWarning) and not issubclass(x.category, DeprecationWarning)]
check('해시·필수 값 확인을 통과해 불러온다', True)
check('만든 환경(meta)이 있고 지금 환경과 같다 (경고 없음)', 'meta' in bundle and not w,
      '; '.join(str(x.message) for x in w) or str(bundle.get('meta')))

print('\n2. 대표 숫자')
v = bundle['validation']
check('예측 간격', bundle.get('step_h', 6) == EXPECTED['step_h'], f'{bundle.get("step_h", 6)}시간')
check('경보 기준', round(bundle['threshold'], 2) == EXPECTED['threshold'], f'{bundle["threshold"]:.2f}')
check('주의 기준', round(bundle['caution']['threshold'], 2) == EXPECTED['caution_threshold'],
      f'{bundle["caution"]["threshold"]:.2f}')
check('검증 세트 AP', round(v['point']['AP'], 3) == EXPECTED['AP'], f'{v["point"]["AP"]:.3f}')
check('고장 감지', (v['event']['감지'], v['event']['고장']) == (EXPECTED['감지'], EXPECTED['고장']),
      f'{v["event"]["감지"]}/{v["event"]["고장"]}')
check('비용 (두 부품)', round(v['cost']['두 부품']) == EXPECTED['비용 두 부품'], f'{v["cost"]["두 부품"]:.0f}')

print('\n3. 서비스 예측 (테스트 세트)')
test_ids = load_scenarios().query('SPLIT == "test"')['SCENARIO_ID'].tolist()
pts, _ = score_scenarios(test_ids, bundle)
saved = pd.read_csv(ROOT / 'predictions' / 'test_risk24.csv')
same_keys = (pts['SCENARIO_ID'].tolist() == saved['SCENARIO_ID'].tolist()) and np.array_equal(pts['T_H'], saved['T_H'])
check('예측 시점이 저장된 예측 CSV와 같다', same_keys, f'{len(pts)}개')
if same_keys:
    check('위험 확률이 같다 (소수 4자리)', np.array_equal(pts['PROB'].round(4).values, saved['RISK24_PROB'].values))
    check('열화 확률이 같다 (소수 4자리)', np.array_equal(pts['DEGRADED_PROB'].round(4).values, saved['DEGRADED_PROB'].values))
    check('단계(경보·주의·정상)가 같다', (pts['STAGE'].values == saved['STAGE'].values).all())
n_alarm, n_scen = int(pts['ALARM'].sum()), pts.loc[pts['ALARM'], 'SCENARIO_ID'].nunique()
check('테스트 예측 시점·경보 수', (len(pts), n_alarm, n_scen) == (EXPECTED['test_points'], EXPECTED['test_alarms'],
                                                        EXPECTED['test_alarm_scenarios']),
      f'시점 {len(pts)}, 경보 {n_alarm}, 경보 시나리오 {n_scen}')

print('\n4. 경보 이유 (기여도)')
contrib, unit = explain(bundle, pts)
check('최종 모델의 기여도를 계산한다', contrib is not None, unit or '')
if contrib is not None and unit.startswith('확률'):
    *prep, (_, forest) = bundle['model'].steps
    X = _matrix(pts, bundle['features'])
    for _, step in prep:
        X = step.transform(X)
    raw, base = _forest_contributions(forest, X)
    gap = np.abs(raw.sum(axis=1) + base - pts['PROB'].values).max()
    check('기준값 + 기여도 합 = 위험 확률', gap < 1e-9, f'최대 차이 {gap:.1e}, 기준값 {base:.3f}')
    check('신호 묶음으로 더해도 합이 같다', np.allclose(contrib.sum(axis=1).values, raw.sum(axis=1)))

print('\n5. 비용 곡선')
e = v['event']
n = dict(n_det=e['감지'], n_miss=e['놓침'], n_needed=e['고장과 무관한 경보 묶음: 필요한 교체'],
         n_false=e['고장과 무관한 경보 묶음: 오경보'])
cc = cost_curve(**n)
check('두 부품 교체 비용 = 평가 함수 비용', cc['두 부품 모두 교체'].iloc[0] == v['cost']['두 부품'],
      f'{cc["두 부품 모두 교체"].iloc[0]:.0f} vs {v["cost"]["두 부품"]:.0f}')
check('p=1 한 부품 비용 = 평가 함수 "맞힘"', np.isclose(cc['한 부품만 교체'].iloc[-1], v['cost']['맞힘']))
check('p=0.5 한 부품 비용 = 평가 함수 "반반"', np.isclose(cc['한 부품만 교체'].iloc[0], v['cost']['반반']))
even = cc.loc[cc['한 부품만 교체'] <= cc['두 부품 모두 교체'], 'p']
check('손익분기', len(even) and round(even.iloc[0], 2) == EXPECTED['breakeven_p'],
      f'{even.iloc[0]:.2f}' if len(even) else '없음')

print('\n6. 안전장치')
odd = pd.DataFrame({'END_DT': [pd.Timestamp('2025-01-01', tz='UTC')], 'MNT_TYPE': ['inspect']})
check('모르는 정비 유형이면 멈춘다', raises(lambda: _maintenance_features(odd, pd.Timestamp('2025-01-02', tz='UTC')),
                                   ValueError))
check('모델이 쓰는 특징이 빠지면 멈춘다', raises(lambda: _matrix(pts.drop(columns=bundle['features'][0]),
                                                     bundle['features']), KeyError))
bad = pts.copy()
bad.loc[bad.index[0], bundle['features'][0]] = np.inf
check('특징에 무한대 값이 있으면 멈춘다', raises(lambda: _matrix(bad, bundle['features']), ValueError))
with tempfile.TemporaryDirectory() as d:
    copy = Path(d) / MODEL_PATH.name
    shutil.copy(MODEL_PATH, copy)
    shutil.copy(MODEL_PATH.with_name(MODEL_PATH.name + '.sha256'), Path(d))
    with open(copy, 'ab') as f:
        f.write(b'0')
    check('저장 뒤 바뀐 모델 파일은 불러오지 않는다', raises(lambda: load_bundle(copy), ValueError))

print('\n7. 1시간 정답')
known = load_scenarios().query('SPLIT != "test"')['SCENARIO_ID'].tolist()
ref, hourly = make_targets(known), make_targets_hourly(known, step_h=1)
m = ref.merge(hourly, on=['SCENARIO_ID', 'OBSERVED_AT'], how='left', suffixes=('', '_h'))
check('6시간 시점이 모두 1시간 정답에 있다', m['risk24_h'].notna().all(), f'{len(ref)}개')
check('정답·잔여수명이 같다', (m['risk24'] == m['risk24_h']).all() and (m['RUL_EVENT'] == m['RUL_EVENT_h']).all()
      and np.allclose(m['RUL_HOURS'].fillna(-1), m['RUL_HOURS_h'].fillna(-1)), f'1시간 정답 {len(hourly)}개')
h6 = make_targets_hourly(known, step_h=6)
check('6시간 간격으로 만들면 정답 파일 기준과 똑같다 (잔여수명은 부동소수 오차 안)',
      h6.drop(columns='RUL_HOURS').equals(ref.drop(columns='RUL_HOURS'))
      and np.allclose(h6['RUL_HOURS'].fillna(-1), ref['RUL_HOURS'].fillna(-1)))

print('\n8. 의사결정 지원 (DSS)')
from src.dss import DSSConfig, decide
D = bundle.get('dss')
check('모델 파일에 DSS 정보가 있다', D is not None)
if D:
    got = {p: round(r['비용']) for p, r in D['validation'].items()}
    check('검증 세트 정책 비용', got == EXPECTED['dss_cost'], str(got))
    check('정기 교체 3시간 전 경보면 기다린다', decide({'WAIT_H': 3}, D['curve'], DSSConfig(), 'dss')[0] == 'wait')
    check('계획이 없으면 지금 교체', decide({'WAIT_H': np.nan}, D['curve'], DSSConfig(), 'dss')[0] == 'now')
    check('준비 시간 24시간이면 3시간 뒤 정기 교체를 기다리지 않는다',
          decide({'WAIT_H': 3}, D['curve'], DSSConfig(prep_h=24), 'dss')[0] == 'now')
    from src.dss import field_costs
    fc = field_costs()
    check('현장 가정(만 원): 지금 교체 2,588 · 정기 교체 때 추가 220 · 고장 4,716',
          (2 * fc.part + fc.unnecessary + fc.replace_fixed + fc.replace_h * fc.downtime_h, fc.part, fc.failure_loss) == (2588, 220, 4716))
    vf = D.get('validation_field') or {}
    check('현장 가정 검증 세트 정책 비용 (만 원) ① 75,456 ② 64,080 ③ 54,608 (부품 재고 가정 포함)',
          {k: round(v['비용']) for k, v in vf.items()} == {'regular': 75456, 'now': 64080, 'dss': 54608}, str({k: round(v['비용']) for k, v in vf.items()}))
    check('안전 기준을 넘으면 지금 교체 (30시간 기다림)',
          decide({'WAIT_H': 30}, D['curve'], DSSConfig(), 'dss')[0] == 'now')
    from src.dss import PartsPolicy, options
    _o_ok = options(np.nan, D['curve'], DSSConfig(), stock={'valve': 2, 'oscillator': 1})[0]
    _o_no = options(np.nan, D['curve'], DSSConfig(), stock={'valve': 2, 'oscillator': 0})[0]
    check('재고가 없으면 긴급 조달: 시작이 12시간 늦어지고 웃돈(부품값 50%)이 붙는다',
          _o_no['정비 완료까지(h)'] - _o_ok['정비 완료까지(h)'] == 12 and _o_no['정비 비용'] - _o_ok['정비 비용'] == 6)
    check('정기 교체 때 함께: 다른 부품 재고가 없고 정기 교체가 긴급 조달보다 이르면 고를 수 없다',
          len(options(3, D['curve'], DSSConfig(), stock={'valve': 0, 'oscillator': 0}, plan_part='oscillator')) == 1
          and len(options(3, D['curve'], DSSConfig(), stock={'valve': 1, 'oscillator': 0}, plan_part='oscillator')) == 2)
    check('부품 조달 기본값: 밸브 2 · 진동장치 1, 정상 72시간, 긴급 12시간 +50%',
          (PartsPolicy().stock(), PartsPolicy().lead_h, PartsPolicy().rush_h, PartsPolicy().rush_markup)
          == ({'valve': 2, 'oscillator': 1}, 72.0, 12.0, 0.5))

print('\n9. 보안')
import json
import tempfile as _tf
from src import app_data
from src.auth import hash_password, verify
check('입력 감시 범위가 모델 파일에 있다', 'feature_range' in bundle and len(bundle['feature_range']['lo']) == len(bundle['features']))
check('서비스 예측에 입력 감시 열이 있다', {'OOR_N', 'OOR_FEATS'} <= set(pts.columns))
salt, dig, it = hash_password('test-only-password', iterations=1000)
check('비밀번호 확인: 맞으면 통과, 틀리면 거부', verify('test-only-password', salt, dig, it) and not verify('wrong', salt, dig, it))
with _tf.TemporaryDirectory() as d:
    app_data.export(d)
    meta, data = app_data.load(d)
    p_app = data['test'][0]
    check('클라우드용 데이터의 테스트 예측이 서비스 예측과 같다',
          np.allclose(p_app['PROB'].values, pts['PROB'].values) and (p_app['STAGE'].values == pts['STAGE'].values).all())
    text = (Path(d) / 'meta.json').read_text(encoding='utf-8')
    names = sorted(f.name for f in Path(d).iterdir())
    check('클라우드용 데이터에 모델 객체·원자료 특징이 없다 (Parquet·JSON만, 1시간 점수까지만)',
          all(n.endswith(('.parquet', '.json')) for n in names) and 'RandomForest' not in text
          and not any(c in data['test'][1].columns for c in ['LOAD', 'WD-LOD', 'MLD-LVL']))
    check('클라우드용 정책 비용이 모델 파일과 같다', {k: round(v['비용']) for k, v in meta['dss']['validation'].items()} == EXPECTED['dss_cost'])
    with open(Path(d) / 'test_pts.parquet', 'ab') as f:
        f.write(b'0')
    check('바뀐 클라우드용 파일은 불러오지 않는다', raises(lambda: app_data.load(d), ValueError))
from src.auth import public_by_decision
with _tf.TemporaryDirectory() as d:
    f = Path(d) / 'deploy.toml'
    check('공개 결정은 deploy.toml에 명시했을 때만 (없거나 false면 비공개)', not public_by_decision(f)
          and (f.write_text('[access]\npublic = false\n') or True) and not public_by_decision(f)
          and (f.write_text('[access]\npublic = true\n') or True) and public_by_decision(f))
from src.posko_data import _check_manifest
check('manifest의 경로 조작 값(../)을 막는다',
      raises(lambda: _check_manifest({'files': [{'id': '../x', 'scenarioId': 'S0001', 'chunks': []}], 'scenarios': []}), ValueError)
      and raises(lambda: _check_manifest({'files': [{'id': 'casting', 'scenarioId': 'S0001',
                                                     'chunks': [{'sha256': '../../etc/passwd'}]}], 'scenarios': []}), ValueError))

print('\n10. 온톨로지 (OWL/RDF)')
from rdflib import Literal as _Lit
from src import ontology
from src.dss import DSSConfig, options, recommend
_m, _data = app_data.load()
_pv, _hv, _, _, _mv, _plv = _data['validation']
_now = 16
_last = _pv[_pv['T_H'] <= _now].sort_values('T_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
_st = pd.DataFrame({'STAGE': np.where(_last['ALARM'], '경보', _last['STAGE']), 'PROB': _last['PROB'],
                    'DEGRADED_PROB': _last['DEGRADED_PROB'], 'PRED_H': _last['T_H'],
                    'FIRST_ALARM_H': np.nan, 'FIRST_CAUTION_H': np.nan}, index=_last.index)
_sg = _hv[_hv['DONE_H'] <= _now].dropna().sort_values('DONE_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
_curve = _m['dss']['curve']
_dec = []
for _s in _st.index[_st['STAGE'] == '경보']:
    _nx = _plv[(_plv['SCENARIO_ID'] == _s) & (_plv['PLAN_H'] > _now)]['PLAN_H'].min()
    _opts = options(_nx - _now, _curve, DSSConfig())
    _dec.append({'sid': _s, 'rank': None, 'constraint': '', 'best': recommend(_opts)[0]['선택지'], 'options': _opts})
_g = ontology.build(_now, _st, _sg, _mv[_mv['START_H'] <= _now], _plv, _dec, cost_unit='점', safety_max=0.2, th=0.56, cth=0.57)
check('스키마(TBox)를 읽고 그래프 규칙 점검을 통과한다', len(ontology.tbox()) > 100 and ontology.check(_g) == [], ontology.check(_g))
_tr = dict(ontology.trace(_g, 'S0248'))
check('근거 사슬: S0248(검증, 16h)은 경보 → 예측 → 신호 → 정기 교체 때 함께 교체 추천',
      '경보 (WARNING)' in _tr.get('경보 단계', '') and '몰드 레벨 진폭' in _tr.get('근거 신호', '')
      and any(k == '정비 결정' and v.startswith('추천: 다음 정기 교체') for k, v in ontology.trace(_g, 'S0248')))
check('현장 작업용 근거 사슬에는 비용이 없다', not any('비용' in v for _, v in ontology.trace(_g, 'S0248', with_cost=False)))
check('경보 설비마다 두 부품이 필요 부품으로 잡힌다',
      ontology.parts_needed(_g) == {'밸브': len(_dec), '진동장치': len(_dec)}, ontology.parts_needed(_g))
_g.add((ontology.CC['read_S0248_MLD_AMP'], ontology.CC.replaces, ontology.CC['part_S0248_valve']))
check('신호 → 부품 링크를 넣으면 점검에서 잡는다', any('신호 → 부품' in b for b in ontology.check(_g)))
check('IRI에 쓸 수 없는 ID는 막는다', raises(lambda: ontology._sid('S0248> . <x'), ValueError))

print('\n11. 온톨로지 액션 (승인·보류·발주·점검 결과)')
from src import actions as _ac
with _tf.TemporaryDirectory() as d:
    _f = Path(d) / 'log.csv'
    _log = _ac.ActionLog(_f)
    _log.record('승인', 'validation', 'S0248', 16, '정비 책임자', option='다음 정기 교체 때 두 부품 함께', note='=HYPERLINK("x")')
    _log.record('부품 발주', 'validation', 'S0248', 16, '정비 책임자', part='valve')
    check('정해진 값이 아니면 기록하지 않는다 (종류·역할·설비 ID·시각·부품, 승인은 정비 책임자만)',
          all(raises(fn, ValueError) for fn in [
              lambda: _log.record('삭제', 'validation', 'S0248', 16, '정비 책임자'),
              lambda: _log.record('승인', 'validation', 'S0248', 16, '현장 작업'),
              lambda: _log.record('승인', 'validation', '../S0248', 16, '정비 책임자'),
              lambda: _log.record('승인', 'validation', 'S0248', 99, '정비 책임자'),
              lambda: _log.record('부품 발주', 'validation', 'S0248', 16, '정비 책임자', part='pump')]))
    check('메모의 표 계산기 수식 주입을 막는다', _log.rows[0]['메모'].startswith("'="))
    with open(_f, 'a', encoding='utf-8') as fh:
        fh.write('2026-01-01,validation,S9999 OR 1,1,승인,정비 책임자,,,,,\n')
    _log2 = _ac.ActionLog(_f)
    check('파일에서 다시 읽고, 형식이 맞지 않는 줄은 버린다', len(_log2.rows) == 2)
    check('승인은 그 뒤 정비가 시작되면 끝난다', 'S0248' in _log2.decision_state('validation', 20)
          and 'S0248' not in _log2.decision_state('validation', 30, {'S0248': 24.0})
          and 'S0248' not in _log2.decision_state('validation', 15))
_job = _ac.ActionLog(None)
for _k, _r in [('승인 요청', '현장 작업'), ('승인', '정비 책임자'), ('작업 시작', '현장 작업')]:
    _job.record(_k, 'test', 'S0294', 72, _r)
check('진행 단계: 승인 요청 → 승인 → 작업 시작이 차례로 바뀌고, 승인은 그대로 남는다',
      _job.job_state('test', 72) == {'S0294': '작업 시작'} and _job.decision_state('test', 72)['S0294']['종류'] == '승인')
_mem = _ac.ActionLog(None)
for _i in range(10):
    _mem.record('점검 결과', 'test', f'S{300 + _i:04d}', 10, '현장 작업', judged='valve', confirmed='valve' if _i < 7 else 'oscillator')
_n, _acc = _mem.inspection_accuracy()
check('점검 되먹임: 10건 중 7건 맞힘 → 0.70 ≥ 손익분기 0.60이면 점검 후 한 부품, 9건이면 아직',
      _n == 10 and abs(_acc - 0.7) < 1e-9 and _ac.one_part_ready(_n, _acc, 0.60) and not _ac.one_part_ready(9, _acc, 0.60))
_acts = pd.DataFrame([
    _ac.validate({'종류': '승인', '데이터 세트': 'validation', '설비': 'S0248', '시각(h)': 16, '누가': '정비 책임자',
                  '선택지': '다음 정기 교체 때 두 부품 함께'}),
    _ac.validate({'종류': '점검 결과', '데이터 세트': 'validation', '설비': 'S0248', '시각(h)': 16, '누가': '현장 작업',
                  '점검 판단': 'valve', '확인된 원인': '아직 모름'})])
_g2 = ontology.build(_now, _st, _sg, _mv[_mv['START_H'] <= _now], _plv, _dec, cost_unit='점', safety_max=0.2, th=0.56, cth=0.57,
                     actions=_acts, feedback={'n': 10, 'accuracy': 0.7, 'breakeven': 0.6, 'ready': True})
_tr2 = ontology.trace(_g2, 'S0248')
check('그래프: 승인이 정비 결정과 추천 선택지에 이어지고 규칙 점검 통과',
      ontology.check(_g2) == [] and any(k == '조치' and v.startswith('승인') and '다음 정기 교체' in v for k, v in _tr2)
      and any(k == '원인 부품' and '점검 후 한 부품' in v for k, v in _tr2), ontology.check(_g2))
_g2.add((ontology.CC['action_S0248_1'], ontology.CC.judgedPart, ontology.CC['part_S0222_valve']))
check('다른 설비의 부품을 가리키는 점검 결과는 점검에서 잡는다', any('그 설비의 부품이 아니다' in b for b in ontology.check(_g2)))

print('\n12. 한계와 대응 (강사 피드백)')
from src.limitations import LIMITS, markdown as _lim_md
_rep = (ROOT / 'reports' / '연주_조기경보_결과보고서.md').read_text(encoding='utf-8')
check('결과보고서 10절 표가 src/limitations.py와 같다 (대시보드와 같은 목록)', _lim_md() in _rep)
check('한계마다 지금 대응과 추가 대응이 있다', all(r['지금 대응'] and r['추가 대응'] and r['중요도'] in ('높음', '중간', '낮음') for r in LIMITS))

print('\n13. 작업 지시 (팀원 설계와 합침: 배정·시작·완료 보고)')
from src import tasks as _tk
for _name, _path in [('세션', None), ('SQLite', Path(_tf.mkdtemp()) / 't.sqlite3')]:
    _s = _tk.TaskStore(_path)
    _a = _s.add('test', 'S0294', 'S0294 지금 두 부품 교체', '정비 1팀', '긴급', 72, 72, 'g', 'r', '지금 두 부품 교체')
    _b = _s.add('test', 'S0294', '중복', '정비 2팀', '높음', 72, 72, 'g', 'r')
    _c = _s.add('test', 'S0294', '과거에 끼워 넣기', '정비 2팀', '보통', 10, 10, 'g', 'r')
    _tid = int(_s.frame('test', 72).iloc[0]['id'])
    _no_detail = raises(lambda: _s.complete(_tid, 72, '이상 발견', '  ', 'valve', 'valve'), ValueError)
    _s.start(_tid, 72)
    _done = _s.complete(_tid, 73, '이상 발견', '=HYPERLINK(1)', 'valve', 'valve')
    _again = _s.complete(_tid, 73, '이상 발견', 'x', 'valve', 'valve')
    check(f'작업 지시({_name}): 같은 설비 중복 배정 차단, 수행 내용 없이는 완료 불가, 두 번 완료 불가',
          _a and not _b and not _c and _no_detail and _done and not _again)
    check(f'작업 지시({_name}): 재생 시각 기준 상태 (72h 진행 전 대기, 73h 완료, 50h에는 아직 없음)',
          _s.frame('test', 72).iloc[0]['상태'] == '진행 중' and _s.frame('test', 73).iloc[0]['상태'] == '완료'
          and _s.frame('test', 50).empty)
    check(f'작업 지시({_name}): 내려받는 CSV의 수식 주입을 막는다', "'=HYPERLINK" in _tk.safe_csv(_s.frame('test', 73)).decode('utf-8-sig'))
    _fix = _s.edit_report(_tid, '추가 점검 필요', '보고 고침', 'oscillator', 'oscillator')
    _row = _s.frame('test', 73).iloc[0]
    _open = _s.add('test', 'S0300', 't', '정비 1팀', '보통', 73, 73, 'g', 'r')
    _open_id = int(_s.frame('test', 73).query('sid == "S0300"').iloc[0]['id'])
    check(f'작업 지시({_name}): 완료된 작업의 보고를 고칠 수 있고(완료 시각 유지), 완료 전 작업은 보고 수정 불가',
          _fix and _row['outcome'] == '추가 점검 필요' and _row['confirmed'] == 'oscillator' and _row['completed_h'] == 73
          and not _s.edit_report(_open_id, '이상 발견', 'x', 'valve', 'valve'))
_ss = _tk.TaskStore(None)
_ss.add('test', 'S0294', 't', '정비 1팀', '긴급', 60, 60, 'g', 'r')
_st = lambda: _ss.frame('test', 72).iloc[0]['상태']
_flow = []
for _to in ['진행 중', '대기']:
    _ss.set_status(1, _to, 65)
    _flow.append(_st())
_ss.set_status(1, '완료', 70, '이상 발견', '교체', 'valve', 'valve')
_flow.append(_st())
_ss.set_status(1, '진행 중', 71)
_flow.append(_st())
check('작업 상태를 설비관리자가 바꿀 수 있다 (대기·진행 중·완료, 완료를 되돌리면 완료 보고 지움, 빈 보고로는 완료 불가)',
      _flow == ['진행 중', '대기', '완료', '진행 중'] and _ss.frame('test', 72).iloc[0]['outcome'] == ''
      and raises(lambda: _ss.set_status(1, '완료', 71, '이상 발견', ' '), ValueError))
check('작업 지시: 정해진 팀·우선순위가 아니면 배정하지 않는다',
      raises(lambda: _tk.TaskStore(None).add('test', 'S0294', 't', '외부 업체', '긴급', 1, 1, '', ''), ValueError)
      and raises(lambda: _tk.TaskStore(None).add('test', '../x', 't', '정비 1팀', '긴급', 1, 1, '', ''), ValueError))
_lg = _ac.ActionLog(None)
_lg.record('점검 결과', 'test', 'S0294', 70, '현장 작업', judged='valve', confirmed='valve')
_lg.record('점검 결과', 'test', 'S0294', 70, '정비 책임자', judged='valve', confirmed='oscillator', note='완료 보고 수정')
check('완료 보고를 고치면 점검 되먹임은 마지막 보고만 센다', _lg.inspection_accuracy() == (1, 0.0))
_lg.record('점검 결과', 'test', 'S0294', 70, '정비 책임자', judged='구분 못함', confirmed='아직 모름', note='완료 되돌림')
check('완료를 되돌리면 그때의 점검 결과는 되먹임에서 빠진다', _lg.inspection_accuracy() == (0, None))
check('작업 지시 SQLite 파일은 커밋되지 않는다 (actions/ 무시)',
      __import__('subprocess').run(['git', 'check-ignore', '-q', 'actions/tasks.sqlite3'], cwd=ROOT).returncode == 0)

n_fail = sum(not ok for _, ok in results)
print(f'\n결과: {len(results) - n_fail}/{len(results)}개 통과')
sys.exit(1 if n_fail else 0)
