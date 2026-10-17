"""클라우드 배포용 대시보드 데이터: 화면에 필요한 결과만 내보내고(데이터 최소화), 모델 파일 없이 불러온다.

    python -m src.app_data          # models/risk24.joblib + data/posko/ → app_data/ (약 2분)

왜 (reports/연주_조기경보_결과보고서.md 9절·부록 E)
- 원자료(5초 센서 1,866만 행)와 특징을 클라우드에 올리지 않는다. 화면에 쓰는 예측 확률·단계·표준화 점수(1시간)·정비 기록·
  정기 교체 계획·검증 요약만 내보낸다.
- 모델 파일(joblib = pickle)은 불러올 때 코드가 실행될 수 있다. 클라우드에서는 모델을 불러오지 않고, 미리 계산한 결과를
  Parquet·JSON(코드가 실행되지 않는 형식)으로만 읽는다.
- 내보낸 파일마다 SHA-256을 MANIFEST.json에 적고, 불러올 때 맞는지 확인한다(파일이 깨지거나 바뀌면 멈춘다).

app_data/
  meta.json                     기준·모델 이름·검증 요약·DSS 가정값·고장까지 시간 분포 (모델 객체 없음)
  {test,validation}_pts.parquet 예측 시점별 확률·경보·단계·입력 감시·경보 이유(기여도)
  {test,validation}_hr.parquet  1시간 표준화 점수 (네 신호와 평균)
  {test,validation}_mnt.parquet 정비 기록 (시나리오 시작 후 시각)
  {test,validation}_plans.parquet 정기 교체 계획
  validation_events.parquet, caution_events.parquet  검증 세트 고장별 결과
  MANIFEST.json                 파일별 SHA-256
"""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / 'app_data'
SPLITS = ['test', 'validation']
SIGNALS = ['Z_MLD-AMP', 'Z_CONTROL_DELAY', 'Z_WDL', 'Z_OSC-STK']


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(x):
    """JSON으로 쓸 수 있는 값으로 바꾼다 (NumPy 수, NaN, 표)"""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.DataFrame, pd.Series)):
        raise TypeError('표는 Parquet으로 따로 쓴다')
    return x


def export(out_dir=APP_DIR):
    """저장한 모델로 테스트·검증 세트를 계산해 화면용 결과만 out_dir에 쓴다 (로컬에서만, 원자료가 있어야 한다)"""
    from src.dss import plans
    from src.posko_data import load_scenarios
    from src.serving import _file_sha256, MODEL_PATH, explain, failure_hours, load_bundle, score_scenarios

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    b = load_bundle()
    scen = load_scenarios()
    start = scen.set_index('SCENARIO_ID')['START_AT']
    units = {}
    for split in SPLITS:
        ids = scen.loc[scen['SPLIT'] == split, 'SCENARIO_ID'].tolist()
        pts, hr = score_scenarios(ids, b)
        contrib, units[split] = explain(b, pts)
        keep = ['SCENARIO_ID', 'T_H', 'PROB', 'ALARM', 'STAGE'] + [c for c in ['DEGRADED_PROB', 'OOR_N', 'OOR_FEATS'] if c in pts]
        p = pts[keep].reset_index(drop=True)
        if contrib is not None:
            p = pd.concat([p, contrib.reset_index(drop=True).add_prefix('C_')], axis=1)
        p.to_parquet(out_dir / f'{split}_pts.parquet', index=False)
        h = hr.assign(DONE_H=hr['HOUR'] + 1)[['SCENARIO_ID', 'DONE_H', 'SCORE', *SIGNALS]]
        h.to_parquet(out_dir / f'{split}_hr.parquet', index=False)
        failure_hours(ids).to_parquet(out_dir / f'{split}_mnt.parquet', index=False)
        pl = plans(ids)
        pl['PLAN_H'] = (pl['PLAN_START'] - pl['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600
        pl[['SCENARIO_ID', 'PLAN_H', 'PLAN_PART']].to_parquet(out_dir / f'{split}_plans.parquet', index=False)

    v = b['validation']
    v['events'].to_parquet(out_dir / 'validation_events.parquet', index=False)
    c = b.get('caution')
    if c:
        c['validation_events'].to_parquet(out_dir / 'caution_events.parquet', index=False)
    d = b.get('dss')
    meta = {
        'threshold': b['threshold'], 'step_h': b.get('step_h', 6), 'features': list(b['features']), 'final': b['final'],
        'created_at': b['created_at'], 'n_train_points': b['n_train_points'], 'meta': b.get('meta'),
        'model_sha256': _file_sha256(MODEL_PATH), 'contrib_unit': units,
        'validation': {'point': v['point'], 'event': v['event'], 'cost': v['cost'], 'cost_reference': v['cost_reference'],
                       'unlabeled': {k: v['unlabeled'][k] for k in ['시점', '경보 시점', '경보 묶음', '비용 범위']}},
        'caution': ({'threshold': c['threshold'], 'features': list(c['features']), 'final': c['final'],
                     'min_precision': c['min_precision'], 'validation': c['validation']} if c else None),
        'dss': ({'config': dataclasses.asdict(d['config']), 'curve': {'h': d['curve'].index.tolist(), 'F': d['curve'].tolist()},
                 'validation': d['validation'], 'validation_field': d.get('validation_field')} if d else None),
    }
    (out_dir / 'meta.json').write_text(json.dumps(_plain(meta), ensure_ascii=False, indent=1), encoding='utf-8')
    files = sorted(f.name for f in out_dir.iterdir() if f.name != 'MANIFEST.json')
    (out_dir / 'MANIFEST.json').write_text(json.dumps({f: _sha256(out_dir / f) for f in files}, indent=1), encoding='utf-8')
    return out_dir


def available(app_dir=APP_DIR):
    return (Path(app_dir) / 'meta.json').exists()


def load(app_dir=APP_DIR):
    """app_data를 불러온다: (화면용 묶음 dict, {split: (pts, hr, contrib, unit, mnt, plans)}). 모델·원자료는 쓰지 않는다.

    MANIFEST.json의 SHA-256과 다르거나 목록에 없는 파일이 있으면 멈춘다.
    """
    from src.dss import Costs, DSSConfig, PartsPolicy

    app_dir = Path(app_dir)
    manifest = json.loads((app_dir / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, digest in manifest.items():
        if _sha256(app_dir / name) != digest:
            raise ValueError(f'app_data/{name} 가 내보낼 때와 다르다. python -m src.app_data 로 다시 만들 것')
    extra = {f.name for f in app_dir.iterdir()} - set(manifest) - {'MANIFEST.json'}
    if extra:
        raise ValueError(f'app_data에 목록에 없는 파일이 있다: {sorted(extra)}')

    m = json.loads((app_dir / 'meta.json').read_text(encoding='utf-8'))
    m['validation']['events'] = pd.read_parquet(app_dir / 'validation_events.parquet')
    if m.get('caution'):
        m['caution']['validation_events'] = pd.read_parquet(app_dir / 'caution_events.parquet')
    if m.get('dss'):
        conf = dict(m['dss']['config'])
        conf['costs'] = Costs(**conf['costs'])
        conf['parts'] = PartsPolicy(**conf['parts']) if 'parts' in conf else PartsPolicy()
        m['dss']['config'] = DSSConfig(**conf)
        m['dss']['curve'] = pd.Series(m['dss']['curve']['F'], index=m['dss']['curve']['h'], name='F')
    data = {}
    for split in SPLITS:
        pts = pd.read_parquet(app_dir / f'{split}_pts.parquet')
        cc = [c for c in pts.columns if c.startswith('C_')]
        contrib = pts[cc].rename(columns=lambda c: c[2:]) if cc else None
        data[split] = (pts.drop(columns=cc), pd.read_parquet(app_dir / f'{split}_hr.parquet'), contrib,
                       m['contrib_unit'].get(split), pd.read_parquet(app_dir / f'{split}_mnt.parquet'),
                       pd.read_parquet(app_dir / f'{split}_plans.parquet'))
    return m, data


if __name__ == '__main__':
    out = export()
    size = sum(f.stat().st_size for f in out.iterdir())
    print(f'{out.relative_to(ROOT)}/ 에 {len(list(out.iterdir()))}개 파일, {size / 1e6:.1f}MB')
