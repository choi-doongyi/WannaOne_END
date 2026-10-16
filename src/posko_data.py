"""POSKO 예측정비 데이터(pdm-v2.2.0) 변환과 불러오기.

원본 폴더의 원자료는 `chunks/`에 해시 이름 CSV 6천여 개로 나뉘어 있다.
manifest.json의 순서대로 이어 붙여 원자료 종류별 Parquet 파일 하나로 만든다.
정답 폴더(labels, diagnostics, targets, flow)도 시나리오별 CSV를 하나로 합친다.

변환 (한 번만, 약 5분):

    python -m src.posko_data --src ~/Downloads/데이터셋/합성데이터_POSKO/POSKO-PdM-Data-pdm-v2.2.0

불러오기:

    from src.posko_data import load_source, load_answer, load_scenarios

    scen = load_scenarios()                   # SCENARIO_ID, START_AT, SPLIT
    casting = load_source('casting')          # 연주 원자료 1,866만 행
    s1 = load_source('casting', scenarios=['S0001'])
    labels = load_answer('labels')            # 6·24시간 고장위험 정답

정답 파일의 열은 예측 특징량에 넣지 않는다(toolkit/README.md).
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'posko'

# 시각 열: 모두 UTC ISO 문자열이다
TIME_COLUMNS = {'MEAS_DT', 'STRT_DT', 'END_DT', 'DETC_DT', 'FAILURE_DT', 'EVT_DT',
                'OBSERVED_AT', 'START_AT', 'END_AT'}

# 정답 폴더: 출력 파일 이름 → 원본 하위 폴더
ANSWER_DIRS = {
    'labels': 'labels',
    'diagnostics': 'diagnostics',
    'targets': 'targets',
    'flow_operations': 'flow/operations',
    'flow_inspections': 'flow/inspections',
    'flow_quality': 'flow/quality',
}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _to_time(s):
    # 시나리오마다 추론 단위(s/us)가 달라지지 않게 ms로 고정한다
    return pd.to_datetime(s, utc=True, format='ISO8601').astype('datetime64[ms, UTC]')


def _numeric_columns(entries):
    """manifest 청크 요약(summaries)에 있는 열이 수치 열이다."""
    cols = set()
    for e in entries:
        for chunk in e['chunks']:
            cols.update(chunk['summaries'])
    return cols


def _cast(df, numeric):
    """시각 열은 UTC datetime, 수치 열은 float64, 나머지는 문자열로 맞춘다.

    시나리오마다 같은 스키마로 써야 Parquet 하나로 이어 붙일 수 있다.
    """
    for col in df.columns:
        if col in TIME_COLUMNS:
            df[col] = _to_time(df[col])
        elif col in numeric:
            df[col] = pd.to_numeric(df[col]).astype('float64')
        else:
            df[col] = df[col].astype('string')
    return df


def _convert_source(src, file_id, entries, verify):
    """원자료 한 종류를 시나리오 순서대로 읽어 Parquet 하나로 쓴다."""
    numeric = _numeric_columns(entries)
    columns = entries[0]['columns']
    out = OUT_DIR / 'raw' / f'{file_id}.parquet'
    writer = None
    n_rows = 0
    try:
        for e in entries:
            if not e['chunks']:  # 행이 없는 시나리오 (예: 정기수리 없음)
                continue
            parts = []
            for chunk in e['chunks']:
                path = src / 'chunks' / chunk['sha256']
                if verify and _sha256(path) != chunk['sha256']:
                    raise ValueError(f'SHA-256 불일치: {path}')
                part = pd.read_csv(path, dtype=str, keep_default_na=False,
                                   na_values=[''], encoding='utf-8-sig')
                if len(part) != chunk['rows']:
                    raise ValueError(f'행 수 불일치: {file_id} {e["scenarioId"]}')
                parts.append(part)
            df = _cast(pd.concat(parts, ignore_index=True)[columns], numeric)
            table = pa.Table.from_pandas(df, preserve_index=False,
                                         schema=writer.schema if writer else None)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression='zstd')
            writer.write_table(table)
            n_rows += len(df)
    finally:
        if writer is not None:
            writer.close()
    expected = sum(e['rows'] for e in entries)
    if n_rows != expected:
        raise ValueError(f'{file_id}: {n_rows}행, manifest {expected}행')
    return n_rows


def _convert_answer(src, name, sub):
    """시나리오별 정답 CSV를 하나로 합친다. 수치 열은 pandas 추론에 맡긴다."""
    files = sorted((src / sub).glob('S*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for col in df.columns:
        if col in TIME_COLUMNS:
            df[col] = _to_time(df[col])
    df.to_parquet(OUT_DIR / f'{name}.parquet', index=False, compression='zstd')
    return len(files), len(df)


def prepare(src, verify=True):
    src = Path(src).expanduser()
    manifest = json.loads((src / 'manifest.json').read_text(encoding='utf-8'))
    (OUT_DIR / 'raw').mkdir(parents=True, exist_ok=True)

    scen = pd.DataFrame(manifest['scenarios']).rename(columns={
        'id': 'SCENARIO_ID', 'startAt': 'START_AT',
        'durationHours': 'DURATION_H', 'split': 'SPLIT'})
    scen['START_AT'] = _to_time(scen['START_AT'])
    scen.to_parquet(OUT_DIR / 'scenarios.parquet', index=False)
    print(f'scenarios: {len(scen)}개 {scen["SPLIT"].value_counts().to_dict()}', flush=True)

    by_id = {}
    for e in manifest['files']:
        by_id.setdefault(e['id'], []).append(e)
    for file_id, entries in by_id.items():
        n = _convert_source(src, file_id, entries, verify)
        print(f'raw/{file_id}: {n:,}행', flush=True)

    for name, sub in ANSWER_DIRS.items():
        n_files, n_rows = _convert_answer(src, name, sub)
        print(f'{name}: 시나리오 {n_files}개, {n_rows:,}행', flush=True)


def load_scenarios():
    return pd.read_parquet(OUT_DIR / 'scenarios.parquet')


def load_source(file_id, scenarios=None, columns=None):
    """원자료 한 종류를 불러온다. scenarios로 시나리오를 골라 메모리를 아낀다."""
    filters = [('SCENARIO_ID', 'in', list(scenarios))] if scenarios is not None else None
    df = pd.read_parquet(OUT_DIR / 'raw' / f'{file_id}.parquet',
                         columns=columns, filters=filters)
    # 원자료의 문자열 열(string)을 정답 파일과 같은 형식(str)으로 맞춘다.
    # 형식이 다르면 merge_asof 같은 결합이 실패한다
    for col in df.columns[df.dtypes == 'string']:
        df[col] = df[col].astype('str')
    return df


def load_answer(name):
    """정답 파일을 불러온다: labels, diagnostics, targets, flow_*.

    flow_operations와 flow_inspections는 정답이 아니라 관측 기록이다.
    """
    return pd.read_parquet(OUT_DIR / f'{name}.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='POSKO 원본 폴더를 data/posko/ Parquet으로 변환')
    parser.add_argument('--src', required=True, help='POSKO-PdM-Data-pdm-v2.2.0 폴더 경로')
    parser.add_argument('--no-verify', action='store_true', help='청크 SHA-256 검사 생략')
    args = parser.parse_args()
    prepare(args.src, verify=not args.no_verify)
