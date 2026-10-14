"""Bounded CSV iteration and version-bound prediction exchange; student supplies ML."""
import csv, json
from bisect import bisect_right
from pathlib import Path

def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)

def source_rows(root, scenario_id, file_id):
    """Stream one logical source CSV with its original chunk/row references."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    source = next(f for f in manifest['files'] if f['scenarioId'] == scenario_id and f['id'] == file_id)
    for chunk in source['chunks']:
        for index, row in enumerate(read_rows(root / 'chunks' / chunk['sha256'])):
            yield {**row, '_SOURCE_CHUNK': chunk['sha256'], '_SOURCE_ROW': chunk['rowStart'] + index}

def supervision_rows(root, scenario_id, kind='labels'):
    """Read public training/validation targets. These fields must not be predictors."""
    if kind not in ['labels', 'targets', 'diagnostics']:
        raise ValueError('kind must be labels, targets or diagnostics')
    manifest = json.loads((Path(root) / 'manifest.json').read_text())
    scenario = next(s for s in manifest['scenarios'] if s['id'] == scenario_id)
    if scenario['split'] == 'test':
        raise ValueError('Test supervision is withheld for instructor evaluation.')
    return list(read_rows(Path(root) / kind / f'{scenario_id}.csv'))

def annotate_diagnosis(rows, intervals):
    """Attach GT_* labels for inspection/training; keep these out of feature matrices."""
    grouped = {}
    for interval in intervals:
        grouped.setdefault((interval['SCENARIO_ID'], interval['EQP_CD']), []).append(interval)
    lookup = {}
    for key, values in grouped.items():
        values.sort(key=lambda r: r['START_AT'])
        lookup[key] = ([r['START_AT'] for r in values], values)
    for row in rows:
        starts, values = lookup[(row['SCENARIO_ID'], row['EQP_CD'])]
        index = bisect_right(starts, row['MEAS_DT']) - 1
        if index < 0 or row['MEAS_DT'] >= values[index]['END_AT']:
            raise ValueError('Observation is outside diagnostic coverage')
        target = values[index]
        yield {**row, **{'GT_' + key: target[key] for key in ['HEALTH_STATE', 'CAUSE_PART', 'FAULT_MODE']}}

def export_predictions(observation, predictions, destination):
    required=['scenarioId','branchId','datasetVersion','observedAt','observationHash']
    result={'schemaVersion':2, **{k:observation[k] for k in required}, 'predictions':predictions}
    for r in predictions:
        assert 0 <= r['risk6h'] <= r['risk24h'] <= 1
    Path(destination).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
