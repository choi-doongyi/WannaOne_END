"""C equipment: current-only vs past-window risk and timing benchmarks.

Student example: thresholds are selected on validation. Test labels are withheld.
Timing regressions are conditional on observed events / beneficial replacement,
not a survival model and not a complete autonomous maintenance policy.
"""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.pipeline import make_pipeline

COLS = ['LOAD', 'WD-SPD', 'WD-LOD', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY']


def window_features(observations, h):
    recent = [o for o in observations if max(0, h-12) <= o['h'] <= h]
    if not recent or recent[-1]['RUNNING'] != 1:
        return None
    # Reset history after any stop. Do not treat shutdown/restart as degradation.
    last_stop = max((j for j, o in enumerate(recent) if o['RUNNING'] == 0), default=-1)
    recent = recent[last_stop+1:]
    current = np.array([float(recent[-1][k]) for k in COLS])
    history = np.array([[float(o[k]) for k in COLS] for o in recent])
    elapsed = np.array([o['h'] for o in recent])
    centered = elapsed - elapsed.mean()
    slope = ((history-history.mean(axis=0))*centered[:, None]).sum(axis=0) / max(float((centered**2).sum()), 1e-9)
    delta = history[-1]-history[0] if len(recent)>1 else np.full(len(COLS), np.nan)
    return current, np.concatenate([current, delta, slope, history.mean(axis=0), history.std(axis=0), [elapsed[-1]-elapsed[0]]])


def load_csv_scenarios(public, private, include_test):
    manifest = json.loads((public/'manifest.json').read_text())
    files = {(f['scenarioId'], f['id']): f for f in manifest['files']}
    for scenario in manifest['scenarios']:
        sid, split = scenario['id'], scenario['split']
        if split == 'test' and not include_test:
            continue
        f = files[sid, 'casting']
        cache = {}
        observations = []
        begin = datetime.fromisoformat(scenario['startAt'].replace('Z', '+00:00'))
        for h in np.arange(0, 72, .5):
            at = (begin+timedelta(hours=float(h))).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
            chunk = next(c for c in f['chunks'] if c['startAt'] <= at <= c['endAt'])
            digest = chunk['sha256']
            if digest not in cache:
                cache[digest] = (public/'chunks'/digest).read_bytes()
            data = cache[digest]
            position = data.find(b','+at.encode()+b',P1-CC1-INS01,')
            if position < 0:
                raise ValueError(f'Missing C row {sid} {at}')
            start, end = data.rfind(b'\n', 0, position)+1, data.find(b'\n', position)
            row = dict(zip(f['columns'], next(csv.reader([data[start:end].decode()]))))
            observations.append({'h': float(h), 'RUNNING': int(row['RUNNING']), **{k: float(row[k]) if row[k] else np.nan for k in COLS}})
        target_path = private/f'{sid}-targets.csv' if split == 'test' else public/'targets'/f'{sid}.csv'
        with target_path.open(newline='') as stream:
            targets = [{**t, 'h': (datetime.fromisoformat(t['OBSERVED_AT'].replace('Z', '+00:00'))-begin).total_seconds()/3600} for t in csv.DictReader(stream) if t['EQP_CD']=='P1-CC1-INS01']
        yield {'id':sid,'split':split,'observations':observations,'targets':targets}


def make_records(scenarios):
    records=[]
    for s in scenarios:
        for target in s['targets']:
            if int(target['AT_RISK']) != 1:
                continue
            h=target['h']
            features=window_features(s['observations'],h)
            if features is None:
                continue
            event=int(target['RUL_EVENT']); followup=float(target['FOLLOWUP_HOURS'])
            risk=int(followup<=24) if event else (0 if followup>=24 else None)
            # A future event after 24 h is a known negative for the 24 h target.
            record={'id':s['id'],'split':s['split'],'h':h,'current':features[0],'temporal':features[1],
                    'risk24':risk,'rul':float(target['RUL_HOURS']) if event else None,
                    'replacementDelay':float(target['POLICY_DELAY_H']) if target['POLICY_ACTION']=='replace' else None}
            records.append(record)
    return records


def evaluate(records, include_test=False):
    report={'featureMethod':'current six C channels; temporal adds past 12 h changes, slopes, means, standard deviations; half-hour samples; reset after stops; no identifiers, labels, future records or raw time as predictors',
            'risk24':{},'rul':{},'replacementDelay':{},
            'timingScope':'RUL evaluated on uncensored at-risk windows only. Replacement delay evaluated only where the oracle recommends replacement; correct component and replacement need are assumed. Not a complete learned policy.'}
    for kind in ['risk24','rul','replacementDelay']:
        subsets={split:[r for r in records if r['split']==split and r[kind] is not None] for split in ['train','validation','test']}
        for mode in ['current','temporal']:
            train=subsets['train'];valid=subsets['validation']
            if kind=='risk24':
                estimator=RandomForestClassifier(n_estimators=180,max_depth=7,min_samples_leaf=5,class_weight='balanced',random_state=31,n_jobs=2)
            else:
                estimator=RandomForestRegressor(n_estimators=180,max_depth=7,min_samples_leaf=4,random_state=31,n_jobs=2)
            model=make_pipeline(SimpleImputer(add_indicator=True),estimator)
            model.fit(np.array([r[mode] for r in train]),np.array([r[kind] for r in train]))
            threshold=None
            if kind=='risk24':
                v=model.predict_proba(np.array([r[mode] for r in valid]))[:,1]
                yv=np.array([r[kind] for r in valid])
                threshold=float(max(np.arange(.05,.96,.05),key=lambda t:f1_score(yv,v>=t,zero_division=0)))
            for split in ['validation']+(['test'] if include_test else []):
                rows=subsets[split];y=np.array([r[kind] for r in rows]);x=np.array([r[mode] for r in rows])
                if kind=='risk24':
                    p=model.predict_proba(x)[:,1]
                    metrics={'windows':len(rows),'positiveWindows':int(y.sum()),'positiveScenarios':len({r['id'] for r in rows if r[kind]}),
                             'AP':float(average_precision_score(y,p)),'prevalence':float(y.mean()),'threshold':threshold,
                             'precision':float(precision_score(y,p>=threshold,zero_division=0)),'recall':float(recall_score(y,p>=threshold,zero_division=0))}
                else:
                    p=model.predict(x);constant=float(np.median([r[kind] for r in train]))
                    metrics={'windows':len(rows),'scenarios':len({r['id'] for r in rows}),'MAE_hours':float(mean_absolute_error(y,p)),
                             'medianBaselineMAE_hours':float(mean_absolute_error(y,np.full(len(rows),constant)))}
                report[kind].setdefault(split,{})[mode]=metrics
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--out',type=Path,default=Path('casting-validation-results.json'))
    args=parser.parse_args()
    scenarios=load_csv_scenarios(args.data_root,None,False)
    report=evaluate(make_records(scenarios),False)
    manifest=(args.data_root/'manifest.json').read_bytes();report['datasetVersion']=json.loads(manifest)['datasetVersion'];report['manifestSha256']=hashlib.sha256(manifest).hexdigest()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
