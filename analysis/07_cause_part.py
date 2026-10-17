"""후속 과제 2: 원인 부품(진동장치·밸브)을 가려낼 수 있는가.

    python analysis/07_cause_part.py

EDA E9는 1시간 요약 신호로는 원인 부품을 구분할 수 없다고 했다. 여기서는 두 가지를 더 확인한다.
1. 5초 원자료의 파형 (흔들림 크기, 빠른 변화, 주파수, 자기상관, 신호 간 상관 등 신호마다 19가지)
2. 정비 기록 (직전에 어떤 부품을 어떻게 정비했나). 예측 시점 전에 끝난 정비만 쓴다
   → "직전이 부분 정비면 그 부품" 규칙이 나오지만, 현장에서는 성립하지 않아(제대로 된 부분 정비 뒤 재고장은 드묾)
     합성 데이터의 성질로 보고 채택하지 않는다 (결과보고서 7-6절)

평가 원칙은 앞 단계와 같다: 학습 세트에서 시나리오 단위 5겹 교차검증으로 판단하고, 검증 세트는 평가에만 쓴다.
정답(진단 정답의 CAUSE_PART, 정비 기록의 PART_CD)은 평가에만 쓴다.

그림 (figures/, 저장소에는 올리지 않음)
  18_part_5초파형_부품구분.png   5초 파형으로 부품을 맞힌 결과와, 라벨을 섞었을 때와의 비교
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.evaluate import COST_FAILURE, COST_REPLACE, COST_UNNECESSARY, PART_RULES, MIN_LEAD_H, _alarm_episodes, \
    attach_failure_info, failure_events
from src.features import build_dataset
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_answer, load_scenarios, load_source
from src.preprocess import EQP
from src.serving import load_bundle

setup_korean_font()
SEED = 31
SIGNALS = ['WD-LOD', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY', 'LOAD']
PAIRS = [('MLD-LVL', 'CONTROL_DELAY'), ('MLD-LVL', 'OSC-STK'), ('WD-LOD', 'OSC-STK'), ('WD-LOD', 'LOAD'),
         ('CONTROL_DELAY', 'OSC-STK')]
WINDOW_H = 6


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def wave_features(w):
    """5초 원자료 구간 하나의 파형 특징. 운전 중인 행만 넣는다"""
    out = {}
    for c in SIGNALS:
        x = w[c].interpolate().bfill().ffill().values
        d = np.diff(x)
        xc = x - pd.Series(x).rolling(121, center=True, min_periods=1).mean().values   # 10분 추세를 뺀 빠른 흔들림
        f, P = signal.welch(x - x.mean(), fs=1 / 5, nperseg=min(1024, len(x)))
        f, P = f[1:], P[1:]
        out |= {f'{c}_평균': x.mean(), f'{c}_표준편차': x.std(), f'{c}_왜도': stats.skew(x), f'{c}_첨도': stats.kurtosis(x),
                f'{c}_변화량_표준편차': d.std(), f'{c}_변화량_왜도': stats.skew(d), f'{c}_빠른흔들림': xc.std(),
                f'{c}_주파수_최대': f[P.argmax()]}
        out |= {f'{c}_자기상관{k}': np.corrcoef(xc[:-k], xc[k:])[0, 1] for k in (1, 2, 6, 12, 60)}
        out |= {f'{c}_대역{lo}': P[(f > lo) & (f <= hi)].sum() / P.sum()
                for lo, hi in [(0, 0.002), (0.002, 0.01), (0.01, 0.03), (0.03, 0.1)]}
    out |= {f'상관_{a}_{b}': np.corrcoef(w[a].fillna(w[a].mean()), w[b].fillna(w[b].mean()))[0, 1] for a, b in PAIRS}
    out['결측_비율'] = (w['QUALITY_FLAG'] != 'ok').mean()
    return out


def cv_auc(df, cols, y):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
    X = df[cols].fillna(df[cols].median()).values
    p = cross_val_predict(m, X, y, groups=df['SCENARIO_ID'].values, cv=GroupKFold(5), method='predict_proba')[:, 1]
    return roc_auc_score(y, p), accuracy_score(y, p >= 0.5), max(y.mean(), 1 - y.mean())


# 1. 열화 구간마다 5초 파형 특징

section('1. 열화 구간마다 5초 파형 특징 (구간 끝 6시간, 시나리오 처음 정상 6시간 대비)')
split = load_scenarios().set_index('SCENARIO_ID')['SPLIT']
diag = load_answer('diagnostics').query('EQP_CD == @EQP').reset_index(drop=True)
dg = diag[diag['HEALTH_STATE'] == 'degraded']
rows = []
for sid, g in dg.groupby('SCENARIO_ID'):
    cast = load_source('casting', scenarios=[sid]).sort_values('MEAS_DT')
    cast = cast[cast['RUNNING'] == 1]
    n0 = diag[(diag['SCENARIO_ID'] == sid) & (diag['HEALTH_STATE'] == 'normal')].iloc[0]
    base = cast[(cast['MEAS_DT'] >= n0['START_AT']) & (cast['MEAS_DT'] < min(n0['END_AT'], n0['START_AT'] + pd.Timedelta(hours=WINDOW_H)))]
    fb = wave_features(base) if len(base) > 100 else {}
    for _, r in g.iterrows():
        w = cast[(cast['MEAS_DT'] >= max(r['START_AT'], r['END_AT'] - pd.Timedelta(hours=WINDOW_H))) & (cast['MEAS_DT'] < r['END_AT'])]
        if len(w) <= 100:
            continue
        f = wave_features(w)
        rows.append({'SCENARIO_ID': sid, 'SPLIT': split[sid], 'PART': r['CAUSE_PART'], 'MODE': r['FAULT_MODE'],
                     **f, **{f'기준대비_{k}': v - fb[k] for k, v in f.items() if k in fb}})
wf = pd.DataFrame(rows)
FEAT = [c for c in wf.columns if c not in ('SCENARIO_ID', 'SPLIT', 'PART', 'MODE')]
print(f'열화 구간 {len(wf)}개 (학습 {int((wf["SPLIT"] == "train").sum())}, 검증 {int((wf["SPLIT"] == "validation").sum())}) · 특징 {len(FEAT)}개')
print(pd.crosstab([wf['SPLIT'], wf['MODE']], wf['PART']).to_string())


# 2. 학습 세트 교차검증: 부품을 맞힐 수 있나

section('2. 5초 파형으로 부품 맞히기 (학습 세트, 시나리오 단위 5겹 교차검증, 랜덤포레스트)')
tr = wf[wf['SPLIT'] == 'train'].reset_index(drop=True)
severe = tr[tr['MODE'] != 'background-wear'].reset_index(drop=True)   # 고장으로 이어지는 모드
bg = tr[tr['MODE'] == 'background-wear'].reset_index(drop=True)
res = {}
for name, d, y in [('전체 열화 구간', tr, (tr['PART'] == 'valve').astype(int).values),
                   ('고장으로 이어지는 모드만\n(마찰 vs 고착·제어 지연)', severe, (severe['PART'] == 'valve').astype(int).values),
                   ('배경 마모 안에서만', bg, (bg['PART'] == 'valve').astype(int).values),
                   ('참고: 배경 마모인지 아닌지', tr, (tr['MODE'] == 'background-wear').astype(int).values)]:
    auc, acc, base = cv_auc(d, FEAT, y)
    res[name] = auc
    print(f'  {name.replace(chr(10), " ")}: 구간 {len(d)}개 · AUC {auc:.3f} · 정확도 {acc:.3f} (가장 흔한 쪽으로 찍기 {base:.3f})')
print('\n  → 전체에서 AUC가 0.5보다 높은 것은 부품이 아니라 "심각도"를 맞힌 것이다.')
print(f'    배경 마모(가벼운 열화, 고장으로 이어지지 않음)는 진동장치 쪽에 많고 '
      f'({int(((tr["MODE"] == "background-wear") & (tr["PART"] == "oscillator")).sum())} vs '
      f'{int(((tr["MODE"] == "background-wear") & (tr["PART"] == "valve")).sum())}), 파형으로 쉽게 구분된다.')
print('    심각도를 맞춰 놓고(같은 모드끼리) 보면 찍기 수준이다.')

# 라벨 섞기 검사: 특징 하나만 볼 때 가장 좋은 AUC가 우연으로도 그만큼 나오나 (고장으로 이어지는 모드)
ys = (severe['PART'] == 'valve').astype(int).values
Xs = severe[FEAT].fillna(severe[FEAT].median())
best1 = lambda yy: max(abs(roc_auc_score(yy, Xs[c]) - 0.5) for c in FEAT) + 0.5
real = best1(ys)
rng = np.random.default_rng(SEED)
null = np.array([best1(rng.permutation(ys)) for _ in range(200)])
print(f'\n  라벨 섞기 검사 (고장으로 이어지는 모드, 특징 {len(FEAT)}개 중 가장 좋은 하나): 실제 AUC {real:.3f} · '
      f'부품 라벨을 무작위로 섞었을 때 {np.median(null):.3f} (95%: {np.quantile(null, 0.95):.3f})')
print(f'  → 실제 값이 섞은 라벨의 95% 경계와 비슷하다(섞은 라벨에서 이 이상이 나온 비율 {np.mean(null >= real):.0%}). '
      '특징이 많으면 우연히 좋아 보이는 것이 하나쯤 생긴다. 여러 특징을 함께 쓴 교차검증 AUC가 0.5 아래인 것과 같은 결론이다.')

# 검증 세트 한 번: 고장으로 이어지는 모드에서 학습 → 검증
va = wf[(wf['SPLIT'] == 'validation') & (wf['MODE'] != 'background-wear')]
m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
m.fit(severe[FEAT].fillna(severe[FEAT].median()).values, ys)
pv = m.predict_proba(va[FEAT].fillna(severe[FEAT].median()).values)[:, 1]
yv = (va['PART'] == 'valve').astype(int).values
print(f'\n  검증 세트 (고장으로 이어지는 모드, 구간 {len(va)}개): AUC {roc_auc_score(yv, pv):.3f}')


# 3. 정비 기록: 직전 정비와 다음 고장 부품

section('3. 정비 기록으로 보기: 직전 정비와 다음 고장 부품 (점진 고장)')
mnt = load_source('maintenance').query('EQP_TAG == @EQP').sort_values(['SCENARIO_ID', 'STRT_DT'])


def prior_maintenance(sid, t):
    """시각 t 전에 끝난 마지막 정비 (부품, 유형). 없으면 (None, None)"""
    p = mnt[(mnt['SCENARIO_ID'] == sid) & (mnt['END_DT'] <= t)]
    return (p['PART_CD'].iloc[-1], p['MNT_TYPE'].iloc[-1]) if len(p) else (None, None)


for sp in ['train', 'validation']:
    ev = failure_events(split[split == sp].index.tolist())
    ev = ev[ev['TYPE'] == '점진 고장'].merge(mnt[['WO_NO', 'PART_CD']], on='WO_NO')
    pm = [prior_maintenance(s, t) for s, t in zip(ev['SCENARIO_ID'], ev['FAILURE_DT'])]
    ev['직전 정비'] = [f'{p} {k}' if p else '없음' for p, k in pm]
    print(f'\n{"학습" if sp == "train" else "검증"} 세트 점진 고장 {len(ev)}건 (밸브 {int((ev["PART_CD"] == "valve").sum())}, '
          f'진동장치 {int((ev["PART_CD"] == "oscillator").sum())})')
    print(pd.crosstab(ev['직전 정비'], ev['PART_CD']).to_string())
    part = ev[ev['직전 정비'].str.endswith('partial')]
    hit = (part['직전 정비'].str.split().str[0] == part['PART_CD']).sum()
    print(f'  직전 정비가 부분 정비였던 고장 {len(part)}건 중 {hit}건이 그 부품 고장')
part_all = mnt[(mnt['MNT_TYPE'] == 'partial') & mnt['SCENARIO_ID'].map(split).isin(['train', 'validation'])]
nxt = [mnt[(mnt['SCENARIO_ID'] == p['SCENARIO_ID']) & (mnt['STRT_DT'] >= p['END_DT']) & mnt['FAILURE_DT'].notna()].head(1)
       for _, p in part_all.iterrows()]
same = [len(n) and n['PART_CD'].iloc[0] == p for n, p in zip(nxt, part_all['PART_CD'])]
gap = [(n['FAILURE_DT'].iloc[0] - e).total_seconds() / 3600 for n, e in zip(nxt, part_all['END_DT']) if len(n)]
print(f'\n  거꾸로 보기: 학습·검증 세트의 부분 정비 {len(part_all)}번 중 {sum(same)}번이 같은 부품 고장으로 이어짐 '
      f'({min(gap):.1f}~{max(gap):.1f}시간 뒤)')
print('  → 이 데이터에서 부분 정비는 고장을 하루쯤 미룰 뿐이다. 현장에서는 제대로 된 부분 정비 뒤 같은 부품 재고장은 드물고,')
print('     정비 불량이나 부품 불량일 때만 생긴다(팀원 현장 경험). 그래서 이 규칙은 합성 데이터의 성질로 보고 결정 C는 그대로 둔다.')


# 4. 규칙의 비용 효과 (검증 세트, 경보 모델은 그대로)

section('4. 참고: 규칙의 비용 효과 (검증 세트, 데이터에만 맞는 규칙이라 채택하지 않음)')
bundle = load_bundle()
X, _, _ = build_dataset(step_h=bundle.get('step_h', 6))   # 모델과 같은 예측 간격
valid = X[X['SPLIT'] == 'validation'].reset_index(drop=True)
ev_va = failure_events(valid['SCENARIO_ID'].unique().tolist())
prob = bundle['model'].predict_proba(valid[bundle['features']].values.astype(float))[:, 1]
d = attach_failure_info(valid, ev_va)
alarm = prob >= bundle['threshold']
both, one_fail = PART_RULES['두 부품'], COST_REPLACE + COST_UNNECESSARY + COST_FAILURE   # 틀린 한 부품: 20 + 8 + 66
cost_both = cost_rule = 0
applied = []
fw = ev_va.merge(mnt[['WO_NO', 'PART_CD']], on='WO_NO')
first = d[alarm & (d['HTF'] >= MIN_LEAD_H)].groupby('FAIL_WO')['OBSERVED_AT'].min()
for _, e in fw.iterrows():
    if e['WO_NO'] not in first.index:              # 놓친 고장: 두 규칙 모두 고장 비용
        cost_both += COST_FAILURE
        cost_rule += COST_FAILURE
        continue
    p, k = prior_maintenance(e['SCENARIO_ID'], first[e['WO_NO']])
    cost_both += both['prevent']
    if k == 'partial':
        c = COST_REPLACE if p == e['PART_CD'] else one_fail
        applied.append((e['SCENARIO_ID'], '고장 전 경보', p, e['PART_CD'], c - both['prevent']))
        cost_rule += c
    else:
        cost_rule += both['prevent']
ep = _alarm_episodes(d, alarm)
other = ep[ep['FAIL_WO'].isna()].groupby('EP').agg(SCENARIO_ID=('SCENARIO_ID', 'first'), T=('OBSERVED_AT', 'min'),
                                                    RUL_EVENT=('RUL_EVENT', 'first'))
for _, o in other.iterrows():
    kind = 'needed' if o['RUL_EVENT'] == 1 else 'false'
    cost_both += both[kind]
    p, k = prior_maintenance(o['SCENARIO_ID'], o['T'])
    if k == 'partial':
        # 고장 예정이 없는 경보(오경보)면 한 부품만 바꿔 불필요 정비가 하나 준다. 필요한 교체였다면 맞은 부품인지 알 수 없어 최악(틀림)으로 센다
        c = COST_REPLACE + COST_UNNECESSARY
        applied.append((o['SCENARIO_ID'], '고장과 무관한 경보' if kind == 'false' else '필요한 교체(부품 미상, 최악으로 셈)', p, '—',
                        c - both[kind]))
        cost_rule += c
    else:
        cost_rule += both[kind]
print(f'두 부품 모두 교체: {cost_both:.0f}점 (04_model.py의 평가 가능 구간 비용과 같아야 함: '
      f'{bundle["validation"]["cost"]["두 부품"]:.0f})')
print(f'규칙 적용:        {cost_rule:.0f}점 ({cost_rule - cost_both:+.0f})')
print(pd.DataFrame(applied, columns=['시나리오', '경보', '직전 부분 정비 부품', '실제 고장 부품', '비용 차이']).to_string(index=False))


# 5. 근거 그림

section('5. 근거 그림')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1.25, 1]})
names = list(res)
colors = ['#9e9e9e', '#2f5d8a', '#2f5d8a', '#c9b8e8']
bars = axes[0].barh(names, [res[n] for n in names], color=colors)
axes[0].bar_label(bars, fmt='%.2f', padding=3)
axes[0].axvline(0.5, color='k', ls=':', lw=1)
axes[0].text(0.505, -0.42, '찍기 (0.5)', fontsize=9, va='top')
axes[0].set_xlim(0.3, 1.0)
axes[0].invert_yaxis()
axes[0].set_xlabel('교차검증 AUC (학습 세트, 시나리오 단위 5겹)')
axes[0].set_title('5초 파형으로 맞히기: 위 3개는 부품, 맨 아래는 심각도', fontsize=10)
axes[1].hist(null, bins=20, color='#c9c9c9', edgecolor='white', label='부품 라벨을 섞었을 때 (200번)')
axes[1].axvline(real, color='#eb5757', lw=2, label=f'실제 라벨 ({real:.2f})')
axes[1].set_xlabel(f'특징 {len(FEAT)}개 중 가장 좋은 하나의 AUC')
axes[1].set_ylabel('횟수')
axes[1].set_title('고장으로 이어지는 모드: 실제도 우연과 같은 수준', fontsize=10)
axes[1].legend(fontsize=8)
fig.suptitle('5초 원자료 파형으로도 원인 부품은 구분되지 않는다', fontsize=11)
fig.tight_layout()
save_figure(fig, '18_part_5초파형_부품구분.png')
