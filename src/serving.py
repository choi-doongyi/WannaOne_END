"""조기경보 서비스용 함수: 저장한 모델로 예측하고, 경보 이유를 계산한다.

analysis/05_predict_test.py가 학습한 모델을 models/risk24.joblib에 저장하고,
대시보드(app/dashboard.py)는 저장된 모델을 불러와 예측만 한다(다시 학습하지 않는다).

    from src.serving import load_bundle, score_scenarios, explain

    bundle = load_bundle()
    pts, hr = score_scenarios(['S0289', 'S0290'], bundle)   # 예측 시점별 확률·경보·주의, 1시간 요약
    contrib, unit = explain(bundle, pts)                     # 시점별·신호 묶음별 기여도와 단위 (랜덤포레스트·로지스틱 회귀)

실시간 데이터가 생기면 score_scenarios의 원자료 읽기(build_hourly)만 새로 들어온 데이터로 바꾸면 된다.
특징은 예측 시점까지 끝난 정보만으로 계산하므로(03_check_features.py 누설 검사) 같은 결과가 나온다.
"""

import hashlib
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.evaluate import COST_FAILURE, COST_REPLACE, COST_UNNECESSARY, PART_RULES
from src.features import make_features, make_prediction_points
from src.preprocess import apply_params, build_hourly

MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'risk24.joblib'
DATA_VERSION = 'pdm-v2.2.0'
REQUIRED_KEYS = ['model', 'threshold', 'features', 'params', 'final', 'validation']   # step_h는 없으면 6시간(예전 형식)

# 특징 이름 → 화면에 보여 줄 묶음 이름
GROUPS = [
    ('MLD-AMP', '몰드 레벨 진폭'),
    ('CONTROL_DELAY', '제어 지연'),
    ('WDL', '인발 부하'),
    ('OSC-STK', '진동 스트로크'),
    ('SCORE', '네 신호 결합 점수'),
    ('PERSIST', '네 신호 결합 점수'),
    ('N_MNT', '정비 이력'),
    ('H_SINCE_MNT', '정비 이력'),
    ('LAST_', '정비 이력'),
    ('SEG_HOURS', '운전 상태'),
    ('GAP_H', '운전 상태'),
]


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _hash_path(path):
    return path.with_name(path.name + '.sha256')


def save_bundle(bundle, path=MODEL_PATH):
    """모델과 함께 쓰는 값(임계값, 특징 목록, 보정값, 검증 결과)을 한 파일로 저장한다.

    만든 환경(라이브러리 버전, 데이터 버전)을 meta로 함께 저장하고, 파일 해시를 옆 파일(.sha256)에 적는다.
    불러올 때 둘을 확인해 다른 환경에서 만든 모델이나 바뀐 파일을 알아챈다.
    """
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f'모델 묶음에 빠진 값: {missing}')
    bundle = {**bundle, 'meta': {'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__,
                                 'python': platform.python_version(), 'data_version': DATA_VERSION}}
    path.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, path)
    _hash_path(path).write_text(_file_sha256(path) + '\n', encoding='utf-8')
    return path


def load_bundle(path=MODEL_PATH):
    """저장한 모델 묶음을 불러온다.

    joblib(pickle) 파일은 불러올 때 코드가 실행될 수 있으므로, 이 프로젝트가 직접 만든 파일만 불러온다.
    - 해시 파일이 있는데 값이 다르면 멈춘다 (저장 뒤 파일이 바뀜)
    - 만든 scikit-learn 버전과 지금 버전이 다르면 경고한다 (같은 입력에 다른 확률이 나올 수 있음)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} 가 없다. 먼저 python analysis/05_predict_test.py 를 실행할 것')
    hp = _hash_path(path)
    if hp.exists() and hp.read_text(encoding='utf-8').strip() != _file_sha256(path):
        raise ValueError(f'{path.name} 의 해시가 저장할 때와 다르다. python analysis/05_predict_test.py 로 다시 만들 것')
    bundle = joblib.load(path)
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ValueError(f'{path.name} 에 빠진 값: {missing}. python analysis/05_predict_test.py 로 다시 만들 것')
    meta = bundle.get('meta')
    if meta is None:
        warnings.warn(f'{path.name} 에 만든 환경 정보(meta)가 없다(예전 형식). 05_predict_test.py 로 다시 만들기를 권장', stacklevel=2)
    elif meta['sklearn'] != sklearn.__version__:
        warnings.warn(f'모델은 scikit-learn {meta["sklearn"]}에서 만들었고 지금은 {sklearn.__version__}이다. '
                      '결과가 달라질 수 있으니 05_predict_test.py 로 다시 만들 것', stacklevel=2)
    return bundle


def _matrix(pts, cols):
    """특징 표에서 모델 입력 행렬을 꺼낸다. 빠진 열이나 무한대 값이 있으면 멈춘다 (조용히 틀린 확률을 내지 않게)."""
    missing = [c for c in cols if c not in pts.columns]
    if missing:
        raise KeyError(f'모델이 쓰는 특징이 없다: {missing}')
    X = pts[cols].to_numpy(dtype=float)
    if np.isinf(X).any():
        bad = [c for c, b in zip(cols, np.isinf(X).any(axis=0)) if b]
        raise ValueError(f'특징에 무한대 값이 있다: {bad}')
    return X


def score_scenarios(scenario_ids, bundle):
    """시나리오들의 예측 시점(모델의 예측 간격 step_h마다)별 24시간 고장 확률과 경보.

    반환
      pts  예측 시점 표: SCENARIO_ID, OBSERVED_AT, T_H, 특징들, PROB(24시간 안 고장 확률), ALARM(경보 여부),
           OOR_N·OOR_FEATS(학습 범위 밖 특징 수·이름, 모델 파일에 범위가 있을 때)
           주의 모델이 있으면 DEGRADED_PROB(지금 열화 중일 확률)과 STAGE(경보 / 주의 / 정상)도 붙는다
      hr   1시간 요약 표: 신호별 표준화 점수(Z_), 평균 점수(SCORE) — 화면의 신호 흐름용
    """
    ids = list(scenario_ids)
    if not ids:
        raise ValueError('시나리오가 없다')
    hr = apply_params(build_hourly(ids), bundle['params'])   # 보정값은 학습 세트에서 정한 그대로
    pts = make_features(hr, make_prediction_points(ids, step_h=bundle.get('step_h', 6)))   # 예전 모델 파일은 6시간
    X = _matrix(pts, bundle['features'])
    pts['PROB'] = bundle['model'].predict_proba(X)[:, 1]
    rng = bundle.get('feature_range')
    if rng is not None:   # 입력 감시: 학습 세트에서 본 범위(0.1~99.9%) 밖의 특징 수. 센서 이상이거나 아주 심한 열화
        lo, hi = np.array(rng['lo']), np.array(rng['hi'])
        out = (X < lo) | (X > hi)
        pts['OOR_N'] = out.sum(axis=1)
        names = np.array(bundle['features'])
        pts['OOR_FEATS'] = [', '.join(names[r]) for r in out]
    pts['ALARM'] = pts['PROB'] >= bundle['threshold']
    c = bundle.get('caution')
    if c:   # 2단계 경보: 경보가 아니어도 열화 확률이 주의 기준 이상이면 주의
        pts['DEGRADED_PROB'] = c['model'].predict_proba(_matrix(pts, c['features']))[:, 1]
        pts['STAGE'] = np.where(pts['ALARM'], '경보', np.where(pts['DEGRADED_PROB'] >= c['threshold'], '주의', '정상'))
    else:
        pts['STAGE'] = np.where(pts['ALARM'], '경보', '정상')
    return pts, hr


def _group(name):
    base = name.removeprefix('missingindicator_')
    group = next((g for key, g in GROUPS if base.startswith(key)), '기타')
    return '결측 표시 (정비 직후 등)' if name.startswith('missingindicator_') else group


def _forest_contributions(forest, X):
    """랜덤포레스트의 시점별·특징별 기여도 (나무마다 결정 경로를 따라 확률이 바뀐 만큼을 그 분기 특징에 준다).

    나무 하나에서 뿌리 → 잎으로 내려갈 때, 분기마다 '양성 비율'이 바뀐 양을 그 분기에 쓰인 특징의 몫으로 센다.
    나무 전체를 평균하면 기준값(뿌리의 양성 비율 평균) + 기여도 합 = predict_proba 와 정확히 같다.
    """
    contrib = np.zeros(X.shape)
    base = 0.0
    X32 = X.astype(np.float32)   # 나무는 float32로 분기한다
    for est in forest.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        p = v[:, 1] / v.sum(axis=1)
        parent = np.full(t.node_count, -1)
        for side in (t.children_left, t.children_right):
            ok = side >= 0
            parent[side[ok]] = np.where(ok)[0]
        kids = np.where(parent >= 0)[0]
        step = np.zeros((t.node_count, X.shape[1]))
        step[kids, t.feature[parent[kids]]] = p[kids] - p[parent[kids]]
        contrib += est.decision_path(X32) @ step
        base += p[0]
    return contrib / len(forest.estimators_), base / len(forest.estimators_)


def explain(bundle, pts):
    """경보 모델의 특징별 기여도를 신호 묶음별로 더한다. 반환: (기여도 표, 단위 설명), 계산할 수 없으면 (None, None)

    로지스틱 회귀: 기여도 = 계수 × 표준화한 특징값 (로그 오즈). 0은 학습 세트 평균 시점과 같다는 뜻이다.
    랜덤포레스트: 나무의 결정 경로 기여도 (확률). 기준값(클래스 가중치를 맞춘 학습 세트에서 0.5 근처) + 기여도 합 = 위험 확률.
    양수면 위험을 올린 쪽, 음수면 내린 쪽이다.
    """
    model = bundle['model']
    *prep, (_, clf) = model.steps
    X = _matrix(pts, bundle['features'])
    for _, step in prep:
        X = step.transform(X)
    names = model.steps[0][1].get_feature_names_out(bundle['features'])
    if hasattr(clf, 'coef_'):
        values, unit = X * clf.coef_[0], '로그 오즈'
    elif hasattr(clf, 'estimators_') and hasattr(clf.estimators_[0], 'tree_'):
        values, base = _forest_contributions(clf, X)
        unit = f'확률 (기준값 {base:.2f}에서 더하고 뺀 양)'
    else:
        return None, None
    contrib = pd.DataFrame(values, columns=names, index=pts.index)
    return contrib.T.groupby([_group(n) for n in names]).sum().T, unit


def failure_hours(scenario_ids):
    """시나리오별 정비 기록(정비 시작·끝, 유형, 고장 여부)을 시작 후 경과 시간(h)으로 바꾼다."""
    from src.posko_data import load_scenarios, load_source
    from src.preprocess import EQP

    start = load_scenarios().set_index('SCENARIO_ID')['START_AT']
    m = load_source('maintenance').query('EQP_TAG == @EQP and SCENARIO_ID in @scenario_ids').copy()
    s0 = m['SCENARIO_ID'].map(start)
    m['START_H'] = (m['STRT_DT'] - s0).dt.total_seconds() / 3600
    m['END_H'] = ((m['END_DT'] - s0).dt.total_seconds() / 3600).fillna(np.inf)   # 관측 끝까지 진행 중인 정비
    m['FAIL_H'] = (m['FAILURE_DT'] - s0).dt.total_seconds() / 3600
    names = {'repair': '고장 후 복구', 'replace': '정기 교체', 'partial': '부분 정비', 'inspect': '점검'}
    m['종류'] = m['MNT_TYPE'].map(names).fillna(m['MNT_TYPE'])   # 모르는 유형은 원래 이름 그대로 보인다
    return m[['SCENARIO_ID', 'PART_CD', '종류', 'START_H', 'END_H', 'FAIL_H']].reset_index(drop=True)


def cost_curve(n_det, n_miss, n_needed, n_false, p_grid=np.linspace(0.5, 1.0, 51), costs=None):
    """부품을 맞힐 확률 p에 따른 비용 (평가 함수 src/evaluate.py와 같은 비용 규칙·상수).

    한 부품만 교체: 맞히면 교체 20, 틀리면 교체 20 + 불필요 정비 8 + 고장 66 = 94
    두 부품 모두 교체: 고장 전 40, 오경보 48 (p와 상관없음)
    costs (src.dss.Costs)를 주면 같은 식을 그 비용 가정으로 계산한다 (대시보드 현장 가정, 만 원).
    기본 Costs()로 계산하면 위 데이터셋 상수와 같다.
    """
    if costs is None:
        hit, both = PART_RULES['맞힘'], PART_RULES['두 부품']
        rep, unn, fail = COST_REPLACE, COST_UNNECESSARY, COST_FAILURE
    else:
        rep = costs.part + costs.replace_fixed + costs.replace_h * costs.downtime_h      # 한 부품 교체 (정지 포함)
        unn, fail = costs.unnecessary, costs.failure_loss
        two = 2 * costs.part + unn + costs.replace_fixed + costs.replace_h * costs.downtime_h
        hit = {'prevent': rep, 'needed': rep, 'false': rep + unn}
        both = {'prevent': two, 'needed': two, 'false': two + unn}
    wrong_prevent = rep + unn + fail        # 94: 틀린 부품 교체 후 고장
    wrong_needed = rep + unn                # 28: 필요한 교체인데 틀린 부품
    rows = []
    for p in p_grid:
        one = (n_det * (hit['prevent'] * p + wrong_prevent * (1 - p)) + n_miss * fail
               + n_needed * (hit['needed'] * p + wrong_needed * (1 - p)) + n_false * hit['false'])
        rows.append({'p': p, '한 부품만 교체': one})
    out = pd.DataFrame(rows)
    out['두 부품 모두 교체'] = (n_det * both['prevent'] + n_miss * fail
                           + n_needed * both['needed'] + n_false * both['false'])
    out['아무것도 안 함'] = (n_det + n_miss) * fail
    return out
