"""연주 설비(P1-CC1-INS01) 조기경보 EDA.

개인 탐색 노트북(eda.ipynb)에서 확정한 내용을 처음부터 끝까지 다시 계산한다.
절 번호는 결과보고서 부록 A(reports/연주_조기경보_결과보고서.md)의 근거 번호(E1~E12)와 함께 적혀 있다.

    python analysis/01_eda.py

- 데이터: src/posko_data.py로 변환한 data/posko/ Parquet (변환 방법은 src/posko_data.py 참고)
- 범위: 학습 세트 216개 시나리오만 쓴다. 검증·테스트 세트는 보지 않는다(분할 원칙).
- 결과: 표는 화면에 출력하고, 그림은 figures/에 저장한다.
- 실행 시간: 약 20초 (학습 세트 원자료 약 1,120만 행을 한 번에 불러온다)

절 구성
  1. 데이터 기본 정보        2. 시나리오와 원인          3·5. 대표 시나리오 비교 그래프
  6. 센서별 열화 변화량       7. 조기경보의 한계          8. 운전조건 이벤트
  9. 센서 간 관계            10. 원인 구분 가능성         11. 정비 효과
  12. 시계열 분석: 주기·추세·변동성·상태 변화 구간
  (노트북 4절 '대표 시나리오 기본 점검'은 1-3~1-5와 결과가 같아 생략)

진단 정답과 고장 시각은 비교 대상을 나누는 데만 쓰고, 특징량으로 쓰지 않는다.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # 화면 없이 그림 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.plotting import save_figure, setup_korean_font
from src.posko_data import load_answer, load_scenarios, load_source

setup_korean_font()   # macOS·Windows·Linux 한글 글꼴 (src/plotting.py)
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_colwidth', 120)

EQP = 'P1-CC1-INS01'                                   # 연주 설비 ID
SENSORS = ['LOAD', 'WD-SPD', 'WD-LOD', 'OSC-FRQ', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY']
MAIN_MODES = ['friction', 'sticking', 'control-delay']  # 주 고장 원인 (background-wear 제외)
MODE_COLOR = dict(zip(MAIN_MODES, ['#2f5d8a', '#e07b39', '#4a9b5e']))
STATE_COLOR = {'degraded': '#f2c94c', 'failed': '#eb5757', 'maintenance': '#bdbdbd'}


def section(title):
    print(f'\n{"=" * 80}\n{title}\n{"=" * 80}')


def show(df, title=None):
    if title:
        print(f'\n[{title}]')
    print(df.to_string() if isinstance(df, (pd.DataFrame, pd.Series)) else df)


def save(fig, name):
    save_figure(fig, name)


def runs(mask, keys):
    """같은 키(시나리오) 안에서 mask가 연속으로 True인 구간마다 번호를 매긴다 (False인 곳은 NaN)"""
    new = mask & ~(mask.shift(fill_value=False) & (keys == keys.shift()))
    return new.cumsum().where(mask)


# 1. 데이터 기본 정보 ──────────────────────────────────────────────────────────

section('1. 데이터 기본 정보 (학습 세트)')

scen = load_scenarios()
TRAIN = scen.loc[scen['SPLIT'] == 'train', 'SCENARIO_ID'].tolist()
START = scen.set_index('SCENARIO_ID')['START_AT']

diag = (load_answer('diagnostics').query('EQP_CD == @EQP')
        .merge(scen[['SCENARIO_ID', 'SPLIT']], on='SCENARIO_ID'))
diag_tr = diag[diag['SCENARIO_ID'].isin(TRAIN)]
mnt_all = load_source('maintenance').query('EQP_TAG == @EQP')
mnt = mnt_all[mnt_all['SCENARIO_ID'].isin(TRAIN)].copy()

# 1-1. 학습 세트 연주 원자료 (216 × 51,840 = 약 1,120만 행)
ct = load_source('casting', scenarios=TRAIN,
                 columns=['SCENARIO_ID', 'MEAS_DT', 'RUNNING', 'QUALITY_FLAG'] + SENSORS)
ct = ct.sort_values(['SCENARIO_ID', 'MEAS_DT'], ignore_index=True)
ct['H'] = (ct['MEAS_DT'] - ct['SCENARIO_ID'].map(START)).dt.total_seconds() / 3600   # 경과 시간(h)
print(f'1-1. 시나리오 {ct["SCENARIO_ID"].nunique()}개, {len(ct):,}행')

# 1-2. 열별 기본 통계 (운전 중/정지 중), 시나리오 간 기준 수준 차이
stats = (ct.groupby('RUNNING')[SENSORS].describe(percentiles=[0.01, 0.5, 0.99])
         .stack(level=0, future_stack=True).round(4))
show(stats, '1-2. 센서별 통계 (RUNNING 0 = 정지, 1 = 운전)')

run = ct[ct['RUNNING'] == 1]
between = run.groupby('SCENARIO_ID')[SENSORS].mean().std()      # 시나리오 평균들의 표준편차
within = run.groupby('SCENARIO_ID')[SENSORS].std().median()     # 시나리오 안 표준편차의 중앙값
show(pd.DataFrame({'시나리오 간': between, '시나리오 안(중앙값)': within, '비율(간/안)': between / within}).round(4),
     '1-2. 시나리오 간 vs 시나리오 안 표준편차 (비율이 크면 절대값 대신 변화량이 필요)')

# 1-3. 관측 간격
gap = ct.groupby('SCENARIO_ID')['MEAS_DT'].diff().dt.total_seconds()
print('\n1-3. 행 수 분포:', ct.groupby('SCENARIO_ID').size().value_counts().to_dict(),
      '/ 5초 간격이 아닌 곳:', int((gap.notna() & (gap != 5)).sum()),
      '/ 중복 시각:', int(ct.duplicated(['SCENARIO_ID', 'MEAS_DT']).sum()))

# 1-4. 결측
na = ct[SENSORS].isna()
flag_missing = ct['QUALITY_FLAG'] == 'missing'
print('1-4. 열별 결측:', na.sum()[na.sum() > 0].to_dict(),
      '/ QUALITY_FLAG=missing과 일치:', bool((flag_missing == na.any(axis=1)).all()),
      '/ 결측 시각(h):', sorted(ct.loc[flag_missing, 'H'].round(2).unique()))

# 1-5. 정지: 연속된 정지 행을 하나의 구간으로 묶고, 진단 상태와 맞춰 본다
stop = ct['RUNNING'] == 0
run_id = ((stop != stop.shift()) | (ct['SCENARIO_ID'] != ct['SCENARIO_ID'].shift())).astype(int).cumsum()
stops = (ct[stop].groupby(run_id[stop])
         .agg(SCENARIO_ID=('SCENARIO_ID', 'first'), START=('MEAS_DT', 'min'), ROWS=('MEAS_DT', 'size')))
stops['DUR_H'] = (stops['ROWS'] * 5 / 3600).round(1)
s = pd.merge_asof(stops.sort_values('START'),
                  diag_tr.sort_values('START_AT')[['SCENARIO_ID', 'START_AT', 'HEALTH_STATE']],
                  left_on='START', right_on='START_AT', by='SCENARIO_ID')
show(s.groupby(['HEALTH_STATE', 'DUR_H']).size().rename('정지 구간 수'), '1-5. 정지 구간의 진단 상태와 길이(h)')
del stop, run_id, gap, na, flag_missing, run

# 1-6. 정답 분포
lab = load_answer('labels').query('EQP_CD == @EQP and SCENARIO_ID in @TRAIN').copy()
lab['상태'] = np.where(lab['CENSORED_24H'] == 1, '검열', lab['RISK_24H'].map({1.0: '1', 0.0: '0'}))
lab['H'] = ((lab['OBSERVED_AT'] - lab['SCENARIO_ID'].map(START)).dt.total_seconds() / 3600).astype(int)
show(pd.crosstab(lab['H'], lab['상태'], margins=True), '1-6. RISK_24H 경과 시간별 분포')
tg = load_answer('targets').query('EQP_CD == @EQP and SCENARIO_ID in @TRAIN')

# 1-7. 정비 이력: 유형·부품, 계획 여부, 진단 failed와 정비 기록 고장 비교
cal = load_source('calendar').query('EQP_TAG == @EQP and SCENARIO_ID in @TRAIN')
mnt['계획'] = np.where(mnt['WO_NO'].isin(cal['PLAN_ID']), '계획됨', '계획 밖')
show(pd.crosstab([mnt['MNT_TYPE'], mnt['계획']], mnt['PART_CD'], margins=True), '1-7. 정비 유형 × 계획 여부 × 부품')
failed_diag = set(diag_tr.loc[diag_tr['HEALTH_STATE'] == 'failed', 'SCENARIO_ID'])
failed_scen = set(mnt_all.loc[mnt_all['FAILURE_DT'].notna(), 'SCENARIO_ID'])
print(f'1-7. 진단 failed 구간 시나리오 {len(failed_diag)}개 vs 정비 기록 고장 시나리오 '
      f'{len(failed_scen & set(TRAIN))}개 → 고장 판정은 정비 기록(FAILURE_DT)으로 한다')


# 2. 시나리오와 원인 ────────────────────────────────────────────────────────────

section('2. 시나리오 원인별 분류 (학습 세트)')

mode_by_scen = diag[diag['FAULT_MODE'].isin(MAIN_MODES)].groupby('SCENARIO_ID')['FAULT_MODE'].first()
scen_type = (scen[scen['SPLIT'] == 'train'][['SCENARIO_ID']]
             .assign(MODE=lambda d: d['SCENARIO_ID'].map(mode_by_scen).fillna('normal'),
                     FAILED=lambda d: d['SCENARIO_ID'].isin(failed_scen)))
show(pd.crosstab(scen_type['MODE'], scen_type['FAILED'], margins=True), '원인 × 72시간 안 고장 여부')


# 3·5. 대표 시나리오 비교 그래프 ────────────────────────────────────────────────

section('3·5. 대표 시나리오 4개 비교 (정상 1 + 원인별 1, 원인별은 열화 구간이 가장 긴 학습 시나리오)')

REP = {'normal': 'S0001', 'friction': 'S0052', 'sticking': 'S0034', 'control-delay': 'S0024'}
fig, axes = plt.subplots(len(SENSORS), len(REP), figsize=(18, 2.0 * len(SENSORS)), sharex=True, sharey='row')
for j, (mode, sid) in enumerate(REP.items()):
    g = ct[ct['SCENARIO_ID'] == sid]
    d = diag_tr[diag_tr['SCENARIO_ID'] == sid]
    t0 = START[sid]
    for i, col in enumerate(SENSORS):
        ax = axes[i, j]
        for r in d.itertuples():
            if r.HEALTH_STATE in STATE_COLOR:
                ax.axvspan((r.START_AT - t0).total_seconds() / 3600, (r.END_AT - t0).total_seconds() / 3600,
                           color=STATE_COLOR[r.HEALTH_STATE], alpha=0.35, lw=0)
        ax.plot(g['H'], g[col], lw=0.4, color='#2f5d8a')
        if j == 0:
            ax.set_ylabel(col)
        if i == 0:
            ax.set_title(f'{mode} ({sid})')
for ax in axes[-1]:
    ax.set_xlabel('경과 시간(h)')
    ax.set_xticks(range(0, 73, 12))
fig.suptitle('대표 시나리오 센서 비교 (노랑: 열화, 빨강: 고장, 회색: 정비)', y=1.0)
fig.tight_layout()
save(fig, '01_eda_대표시나리오_센서비교.png')


# 6. 센서별 열화 변화량 ─────────────────────────────────────────────────────────

section('6. 센서별 열화 변화량 비교 (학습 세트 전체)')

# 6-1. 1시간 요약: 운전 중 행만, 운전 구간(SEG)은 정비 후 재가동할 때마다 새로 시작
ct['HOUR'] = ct['H'].astype(int)
restart = (ct['RUNNING'] == 1) & (ct.groupby('SCENARIO_ID')['RUNNING'].shift() == 0)
ct['SEG'] = restart.astype(int).groupby(ct['SCENARIO_ID']).cumsum()
run = ct[ct['RUNNING'] == 1]
FEATS = ['LOAD', 'WD-LOD', 'OSC-STK', 'MLD-LVL', 'CONTROL_DELAY']
hr = (run.groupby(['SCENARIO_ID', 'HOUR'])
         .agg(**{c: (c, 'mean') for c in FEATS},
              **{'MLD-AMP': ('MLD-LVL', 'std')},          # 몰드 레벨 진폭 (22분 주기 진동의 폭)
              SEG=('SEG', 'max'), N_RUN=('RUNNING', 'size'))
         .reset_index())
hr = hr[hr['N_RUN'] >= 360].copy()                      # 절반 이상 운전한 시간만
hr['HOUR_START'] = (hr['SCENARIO_ID'].map(START) + pd.to_timedelta(hr['HOUR'], unit='h')).astype('datetime64[ms, UTC]')
hr['MODE'] = hr['SCENARIO_ID'].map(mode_by_scen).fillna('normal')
del run, restart
print(f'6-1. 1시간 구간 {len(hr):,}개')

# 비교용 이름표: 같은 운전 구간 안에서 고장까지 남은 시간(HTF)
fails = mnt[mnt['FAILURE_DT'].notna()][['SCENARIO_ID', 'WO_NO', 'FAILURE_DT', 'PART_CD']]
hr = pd.merge_asof(hr.sort_values('HOUR_START'), fails.sort_values('FAILURE_DT'),
                   left_on='HOUR_START', right_on='FAILURE_DT', by='SCENARIO_ID', direction='forward')
hr['HTF'] = (hr['HOUR_START'] + pd.Timedelta(hours=1) - hr['FAILURE_DT']).dt.total_seconds() / 3600
hr = hr.sort_values(['SCENARIO_ID', 'HOUR'], ignore_index=True)
fail_seg = hr[hr['HTF'] <= 0].groupby(['SCENARIO_ID', 'FAILURE_DT'])['SEG'].max().rename('FAIL_SEG')
hr = hr.join(fail_seg, on=['SCENARIO_ID', 'FAILURE_DT'])
hr.loc[hr['SEG'] != hr['FAIL_SEG'], 'HTF'] = np.nan

# 6-2. 부하 영향 제거: 정상 시나리오 안에서 구한 부하 기울기만큼 뺀다
SIG = ['WD-LOD', 'OSC-STK', 'MLD-LVL', 'MLD-AMP', 'CONTROL_DELAY']
LOAD_REF = hr['LOAD'].mean()
nrm = hr[hr['MODE'] == 'normal']
dm = nrm[SIG + ['LOAD']] - nrm.groupby('SCENARIO_ID')[SIG + ['LOAD']].transform('mean')
slope = {c: (dm[c] * dm['LOAD']).sum() / (dm['LOAD'] ** 2).sum() for c in SIG}
for c in SIG:
    hr[c + '_ADJ'] = hr[c] - slope[c] * (hr['LOAD'] - LOAD_REF)

# 6-6. WD-LOD는 시나리오마다 부하 기울기가 달라 절편 보정 비율을 쓴다
fit = nrm.groupby('SCENARIO_ID')[['LOAD', 'WD-LOD']].apply(
    lambda g: pd.Series(np.polyfit(g['LOAD'], g['WD-LOD'], 1), index=['기울기', '절편']))
INTERCEPT = fit['절편'].median()
show(fit.describe().round(2), '6-6. 정상 시나리오별 WD-LOD = 절편 + 기울기 × LOAD')
WDL_OPTS = {'A. 공통 기울기 차감': 'WDL_A', 'B. 비율': 'WDL_B', 'C. 절편 보정 비율': 'WDL_C'}
hr['WDL_A'] = hr['WD-LOD_ADJ']
hr['WDL_B'] = hr['WD-LOD'] / hr['LOAD']
hr['WDL_C'] = (hr['WD-LOD'] - INTERCEPT) / hr['LOAD']

# 6-3. 기준 대비 변화량(_D): 운전 구간마다 첫 2시간 중앙값을 뺀다
hr['SEG_HOUR'] = hr.groupby(['SCENARIO_ID', 'SEG']).cumcount()
LEVEL = [c + '_ADJ' for c in SIG] + list(WDL_OPTS.values())
base = hr[hr['SEG_HOUR'] < 2].groupby(['SCENARIO_ID', 'SEG'])[LEVEL].median()
hr = hr.join(base, on=['SCENARIO_ID', 'SEG'], rsuffix='_BASE')
for c in SIG:
    hr[c + '_D'] = hr[c + '_ADJ'] - hr[c + '_ADJ_BASE']
for c in WDL_OPTS.values():
    hr[c + '_D'] = hr[c] - hr[c + '_BASE']

normal_mask = (hr['MODE'] == 'normal') & (hr['SEG_HOUR'] >= 2)    # 정상 비교군


def auc_table(cols, windows, groups=MAIN_MODES):
    """원인별·고장 전 시간대별로, 변화량의 AUC(고장 전 구간 vs 정상 구간)"""
    neg = hr[normal_mask]
    rows = []
    for mode in groups:
        for wname, (a, b) in windows.items():
            pos = hr[(hr['MODE'].isin([mode] if mode != '전체' else MAIN_MODES))
                     & (hr['HTF'] > a) & (hr['HTF'] <= b) & (hr['SEG_HOUR'] >= 2)]
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            row = {'원인': mode, '시간대': wname}
            row.update({c: roc_auc_score(y, np.r_[pos[c].values, neg[c].values]) for c in cols})
            rows.append(row)
    return pd.DataFrame(rows).set_index(['원인', '시간대'])


# 6-4. 고장 시각에 맞춘 변화량 그래프
DSIG = ['WDL_C_D', 'OSC-STK_D', 'MLD-LVL_D', 'MLD-AMP_D', 'CONTROL_DELAY_D']
band = hr.loc[normal_mask, DSIG].quantile([0.05, 0.95]).T
pre = hr[(hr['HTF'] <= 0) & (hr['HTF'] > -48) & (hr['MODE'] != 'normal')].copy()
pre['HTF_BIN'] = np.floor(pre['HTF']).astype(int)
fig, axes = plt.subplots(len(DSIG), len(MAIN_MODES), figsize=(16, 2.3 * len(DSIG)), sharex=True, sharey='row')
for j, mode in enumerate(MAIN_MODES):
    p = pre[pre['MODE'] == mode]
    for i, c in enumerate(DSIG):
        ax = axes[i, j]
        ax.axhspan(band.loc[c, 0.05], band.loc[c, 0.95], color='#bdbdbd', alpha=0.4, lw=0)
        q = p.groupby('HTF_BIN')[c].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color='#2f5d8a', alpha=0.25, lw=0)
        ax.plot(q.index, q[0.5], color='#2f5d8a', lw=1.5)
        ax.axhline(0, color='k', lw=0.5)
        if j == 0:
            ax.set_ylabel(c)
        if i == 0:
            ax.set_title(f'{mode} (고장 시나리오 {p["SCENARIO_ID"].nunique()}개)')
for ax in axes[-1]:
    ax.set_xlabel('고장까지 남은 시간(h)')
    ax.set_xticks(range(-48, 1, 12))
fig.suptitle('고장 전 변화량 (선: 중앙값, 띠: 25~75%, 회색: 정상 5~95%)', y=1.0)
fig.tight_layout()
save(fig, '02_eda_고장전_센서변화량.png')

# 6-5. 정상과 얼마나 갈리나 (AUC)
WINDOWS = {'0~6h': (-6, 0), '6~12h': (-12, -6), '12~24h': (-24, -12), '24~36h': (-36, -24)}
show(auc_table(DSIG, WINDOWS).round(2), '6-5. 고장 전 구간 vs 정상 구간 AUC (0.5 = 구분 못 함, OSC-STK는 작아지는 방향이라 0에 가까울수록 좋음)')

# 6-6. WD-LOD 보정 방식 비교
neg = hr[normal_mask]
rows = []
for name, col in WDL_OPTS.items():
    corr = nrm.assign(v=hr.loc[nrm.index, col]).groupby('SCENARIO_ID')[['v', 'LOAD']].corr().xs('LOAD', level=1)['v'].median()
    width = neg[col + '_D'].quantile(0.95) - neg[col + '_D'].quantile(0.05)
    row = {'방식': name, '부하 상관': corr}
    for wname, (a, b) in {'0~6h': (-6, 0), '24~36h': (-36, -24)}.items():
        pos = hr[(hr['MODE'] != 'normal') & (hr['HTF'] > a) & (hr['HTF'] <= b) & (hr['SEG_HOUR'] >= 2)]
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        row[f'신호/잡음 {wname}'] = pos[col + '_D'].median() / width
        row[f'AUC {wname}'] = roc_auc_score(y, np.r_[pos[col + '_D'].values, neg[col + '_D'].values])
    rows.append(row)
show(pd.DataFrame(rows).set_index('방식').round(3), '6-6. WD-LOD 부하 보정 방식 비교 → C 채택')
print(f'  C 방식 공통 절편: {INTERCEPT:.2f}')

sid = (fit['기울기'] - fit['기울기'].median()).abs().idxmax()     # 기울기가 공통값과 가장 다른 정상 시나리오
g = hr[hr['SCENARIO_ID'] == sid]
fig, axes = plt.subplots(len(WDL_OPTS) + 1, 1, figsize=(12, 8), sharex=True)
axes[0].plot(g['HOUR'], g['LOAD'], color='#888888')
axes[0].set_ylabel('LOAD')
axes[0].set_title(f'정상 시나리오 {sid}의 WD-LOD 보정 비교 (부하 기울기 {fit.loc[sid, "기울기"]:.0f}, 공통 {fit["기울기"].median():.0f})')
for ax, (name, col) in zip(axes[1:], WDL_OPTS.items()):
    ax.plot(g['HOUR'], g[col + '_D'], color='#2f5d8a')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel(name, rotation=0, ha='right')
axes[-1].set_xlabel('경과 시간(h)')
fig.tight_layout()
save(fig, '03_eda_WD-LOD_부하보정비교.png')


# 7. 조기경보의 한계 ───────────────────────────────────────────────────────────

section('7. 조기경보가 실제로 얼마나 가능한가')

# 신호 4개와 정상 범위 경계(정상 95%, OSC-STK는 5%)
SIG4 = {'MLD-AMP_D': 'up', 'CONTROL_DELAY_D': 'up', 'WDL_C_D': 'up', 'OSC-STK_D': 'down'}
S4 = list(SIG4)
sgn = {c: (1 if d == 'up' else -1) for c, d in SIG4.items()}
TH = {c: neg[c].quantile(0.95 if d == 'up' else 0.05) for c, d in SIG4.items()}
OUTS = ['OUT_' + c for c in S4]
for c in S4:
    hr['OUT_' + c] = (hr[c] - TH[c]) * sgn[c] > 0
hr['N_OUT'] = hr[OUTS].sum(axis=1).where(hr['SEG_HOUR'] >= 2)

# 7-1. 리드타임: 고장 직전 열화 구간의 시작 → 고장 (없으면 돌발 고장)
dg = diag_tr[diag_tr['HEALTH_STATE'] == 'degraded']
fl = fails.copy()
fl['MODE'] = fl['SCENARIO_ID'].map(mode_by_scen)
x = fl.merge(dg[['SCENARIO_ID', 'START_AT', 'END_AT']], on='SCENARIO_ID')
x = x[(x['END_AT'] - x['FAILURE_DT']).abs() <= pd.Timedelta(seconds=1)]
fl = fl.merge(x[['WO_NO', 'START_AT']].rename(columns={'START_AT': 'DEG_START'}), on='WO_NO', how='left')
fl['LEAD_H'] = (fl['FAILURE_DT'] - fl['DEG_START']).dt.total_seconds() / 3600
fl['TYPE'] = np.where(fl['DEG_START'].notna(), '점진 고장', '돌발 고장')
print('7-1. 고장 종류:', fl['TYPE'].value_counts().to_dict())
show(fl.groupby('MODE')['LEAD_H'].describe()[['count', 'min', '50%', 'max']].round(1), '7-1. 점진 고장 리드타임(h)')


def detect_lead(g, col):
    """고장 직전부터 거꾸로, col이 계속 True인 시간 수 (= 센서 감지 리드타임)"""
    v = g.sort_values('HTF', ascending=False)[col].values
    return int(np.argmin(v)) if not v.all() else len(v)


pre = hr[(hr['HTF'] <= 0) & (hr['SEG_HOUR'] >= 2)].assign(ANY2=lambda d: d['N_OUT'] >= 2)
det = pd.DataFrame([{'SCENARIO_ID': sid, 'FAILURE_DT': fdt, **{c: detect_lead(g, c) for c in OUTS + ['ANY2']}}
                    for (sid, fdt), g in pre.groupby(['SCENARIO_ID', 'FAILURE_DT'])])
det = det.merge(fl[['SCENARIO_ID', 'FAILURE_DT', 'TYPE', 'MODE', 'LEAD_H']], on=['SCENARIO_ID', 'FAILURE_DT'])
summary = det.groupby('TYPE')[OUTS + ['ANY2']].agg(lambda s: f'{(s >= 24).sum()}/{len(s)} (중앙값 {s.median():.0f}h)')
show(summary, '7-1. 센서 감지 리드타임 24시간 이상인 고장 수')

g = det[det['TYPE'] == '점진 고장']
fig, ax = plt.subplots(figsize=(6, 5))
for mode, color in MODE_COLOR.items():
    s = g[g['MODE'] == mode]
    ax.scatter(s['LEAD_H'], s['ANY2'], label=mode, color=color, alpha=0.8)
ax.plot([0, 60], [0, 60], color='k', lw=0.5, ls='--')
ax.axhline(24, color='#eb5757', lw=0.8)
ax.set_xlim(0, 60); ax.set_ylim(0, 60)
ax.set_xlabel('진단 리드타임: 열화 시작 → 고장 (h)')
ax.set_ylabel('센서 감지 리드타임: 2개 이상 벗어남 → 고장 (h)')
ax.set_title('점진 고장의 리드타임 (빨간 선: 24시간)')
ax.legend()
save(fig, '04_eda_리드타임_진단vs센서.png')

# 7-2. 고장 없이 끝난 원인 시나리오
nofail = scen_type.query('MODE != "normal" and not FAILED')['SCENARIO_ID'].tolist()
lab_t = lab.merge(tg[['SCENARIO_ID', 'OBSERVED_AT', 'RUL_EVENT']], on=['SCENARIO_ID', 'OBSERVED_AT'])
lab_t = lab_t[lab_t['SCENARIO_ID'].isin(nofail)]
print(f'\n7-2. 고장 없이 끝난 원인 시나리오 {len(nofail)}개, 정비 유형:',
      mnt[mnt['SCENARIO_ID'].isin(nofail)]['MNT_TYPE'].value_counts().to_dict(),
      '/ RISK_24H 상태:', lab_t['상태'].value_counts().to_dict(),
      '/ RISK_24H=1인 시나리오:', lab_t.loc[lab_t['RISK_24H'] == 1, 'SCENARIO_ID'].nunique())

# 7-3. 정상 시나리오의 일시적 혹 vs 고장 전 연속 이상
nh = hr[hr['MODE'] == 'normal'].sort_values(['SCENARIO_ID', 'HOUR'])
m = (nh['N_OUT'] >= 2).fillna(False).astype(bool)
normal_len = nh.groupby(runs(m, nh['SCENARIO_ID']))['HOUR'].size()
fail_len = det.loc[det['TYPE'] == '점진 고장', 'ANY2']
print(f'\n7-3. 정상 시나리오 혹(신호 2개 이상) {len(normal_len)}개, '
      f'{nh.loc[runs(m, nh["SCENARIO_ID"]).notna(), "SCENARIO_ID"].nunique()}개 시나리오, 최장 {normal_len.max()}h')
for k in [3, 6, 12]:
    print(f'  {k}시간 이상 지속: 정상 혹 {int((normal_len >= k).sum())}/{len(normal_len)}, '
          f'점진 고장 {int((fail_len >= k).sum())}/{len(fail_len)}')
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.arange(0, 56, 2)
ax.hist(normal_len, bins=bins, alpha=0.6, label=f'정상 시나리오의 혹 ({len(normal_len)}개)', color='#888888')
ax.hist(fail_len, bins=bins, alpha=0.6, label=f'점진 고장 전 연속 이상 ({len(fail_len)}건)', color='#2f5d8a')
ax.set_xlabel('신호 2개 이상이 연속으로 정상 범위를 벗어난 길이(h)')
ax.set_ylabel('개수')
ax.legend()
save(fig, '05_eda_정상혹_vs_고장전_지속시간.png')


# 8. 운전조건 이벤트 ───────────────────────────────────────────────────────────

section('8. 운전조건 이벤트')

evt = load_source('events').query('EQP_TAG == @EQP and SCENARIO_ID in @TRAIN').copy()
evt['H'] = (evt['EVT_DT'] - evt['SCENARIO_ID'].map(START)).dt.total_seconds() / 3600
sched = evt.sort_values('H').groupby('SCENARIO_ID').apply(
    lambda g: ' → '.join(f'{h:.0f}h:{int(v)}' for h, v in zip(g['H'], g['AFT_VAL'])), include_groups=False)
show(sched.value_counts(), '8-1. 운전조건 일정 (시나리오 수)')

COND_START = evt.groupby('H')['AFT_VAL'].first()
hr['COND'] = pd.cut(hr['HOUR'], bins=list(COND_START.index) + [72], right=False,
                    labels=COND_START.values, ordered=False).astype(int)
COLS8 = ['WDL_C', 'OSC-STK_ADJ', 'MLD-AMP', 'CONTROL_DELAY_ADJ']
nrm = hr[hr['MODE'] == 'normal']
dm = nrm[COLS8] - nrm.groupby('SCENARIO_ID')[COLS8].transform('mean')
by_cond = dm.groupby(nrm['COND'])[COLS8].mean().T
by_cond['효과 크기'] = (by_cond.max(axis=1) - by_cond.min(axis=1)) / dm.std()
show(by_cond.round(4), '8-2. 운전조건별 센서 수준 (정상, 시나리오 평균 제거). 효과 크기 < 0.2면 무시할 만함')

COND_COLOR = {0: '#ffffff', 1: '#e8f0f8', 2: '#d0e0f0'}
nn = hr[normal_mask]
SIGS8 = ['WDL_C_D', 'OSC-STK_D', 'MLD-AMP_D', 'CONTROL_DELAY_D']
fig, axes = plt.subplots(len(SIGS8), 1, figsize=(12, 2.2 * len(SIGS8)), sharex=True)
for ax, c in zip(axes, SIGS8):
    for h0, v in COND_START.items():
        ax.axvspan(h0, h0 + 12, color=COND_COLOR[int(v)], lw=0)
    q = nn.groupby('HOUR')[c].quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(q.index, q[0.25], q[0.75], color='#2f5d8a', alpha=0.25, lw=0)
    ax.plot(q.index, q[0.5], color='#2f5d8a')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_ylabel(c)
axes[0].set_title('정상 시나리오 변화량 (중앙값, 25~75%) — 배경: 운전조건 0(흰색)·1·2(진한 색)')
axes[-1].set_xlabel('경과 시간(h)')
axes[-1].set_xticks(range(0, 73, 12))
fig.tight_layout()
save(fig, '06_eda_운전조건별_정상변화량.png')

deg_start_h = ((dg['START_AT'] - dg['SCENARIO_ID'].map(START)).dt.total_seconds() / 3600)[dg['FAULT_MODE'].isin(MAIN_MODES)]
fail_h = (fl['FAILURE_DT'] - fl['SCENARIO_ID'].map(START)).dt.total_seconds() / 3600
fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
for ax, (name, v) in zip(axes, {'열화 시작 (주 원인)': deg_start_h, '고장': fail_h}.items()):
    for h0, c in COND_START.items():
        ax.axvspan(h0, h0 + 12, color=COND_COLOR[int(c)], lw=0)
    ax.hist(v, bins=np.arange(0, 73, 2), color='#2f5d8a', alpha=0.8)
    ax.set_title(f'{name} 시각 ({len(v)}건)')
    ax.set_xlabel('경과 시간(h)')
    ax.set_xticks(range(0, 73, 12))
save(fig, '07_eda_열화시작_고장시각_분포.png')


# 9. 센서 간 관계 ──────────────────────────────────────────────────────────────

section('9. 센서 간 관계')

LBL = {c: ('-' if sgn[c] < 0 else '') + c for c in S4}


def oriented(df):
    """신호 4개를 '클수록 이상' 방향으로 맞춘다"""
    return pd.DataFrame({LBL[c]: df[c] * sgn[c] for c in S4})


normal_h = hr[normal_mask]
pre36_h = hr[(hr['HTF'] > -36) & (hr['HTF'] <= 0) & (hr['SEG_HOUR'] >= 2) & (hr['MODE'] != 'normal')]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (name, d) in zip(axes, {'정상 구간': normal_h, '고장 전 36시간': pre36_h}.items()):
    cm = oriented(d).corr()
    im = ax.imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(cm.columns, rotation=30, ha='right')
    ax.set_yticks(range(4)); ax.set_yticklabels(cm.columns)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm.iloc[i, j]:.2f}', ha='center', va='center')
    ax.set_title(f'{name} ({len(d):,}개 시간 구간)')
fig.colorbar(im, ax=axes, shrink=0.8)
save(fig, '08_eda_신호간_상관.png')

p = oriented(pre36_h).assign(SCENARIO_ID=pre36_h['SCENARIO_ID'].values, HTF=pre36_h['HTF'].values)
cols = list(LBL.values())
show(p.groupby('SCENARIO_ID')[cols].corr().groupby(level=1).median().loc[cols, cols].round(2),
     '9-1. 고장 전 시나리오 안 상관 (중앙값)')
show(p[p['HTF'] > -1].groupby('SCENARIO_ID')[cols].mean().corr().round(2),
     '9-1. 고장 직전 1시간 값의 시나리오 간 상관')

leads = det[det['TYPE'] == '점진 고장'].set_index('SCENARIO_ID')[OUTS].rename(columns=lambda c: c[4:])
show(leads.describe().loc[['min', '50%', 'max']].round(0), '9-2. 신호별 감지 리드타임(h)')
print('  가장 먼저 벗어난 신호(동률 포함):', leads.eq(leads.max(axis=1), axis=0).sum().to_dict())

# 9-3. 정상 분포로 표준화한 점수(Z)와 결합 점수
med = {c: normal_h[c].median() for c in S4}
p95 = {c: (normal_h[c] * sgn[c]).quantile(0.95) - med[c] * sgn[c] for c in S4}
Z = pd.DataFrame({c: (hr[c] - med[c]) * sgn[c] / p95[c] for c in S4})
Zs = np.sort(Z.values, axis=1)
scores = Z.rename(columns=LBL).assign(**{'평균': Z.mean(axis=1), '최댓값': Zs[:, -1], '2번째로 큰 값': Zs[:, -2]})
pos_mask = (hr['MODE'] != 'normal') & (hr['HTF'] > -36) & (hr['HTF'] <= -24) & (hr['SEG_HOUR'] >= 2)
show(pd.DataFrame({'정상 구간 경계 초과': (scores[normal_mask] > 1).mean(),
                   '고장 전 24~36h 경계 초과': (scores[pos_mask] > 1).mean()}).round(3),
     '9-3. 표준화 점수 > 1(정상 범위 경계)인 비율 → 평균 점수가 오경보를 크게 줄인다')


# 10. 원인 구분 가능성 ─────────────────────────────────────────────────────────

section('10. 원인 구분 가능성')

# 10-1. 주 원인 열화 구간마다 끝나기 직전 6시간의 Z 평균 = 반응 지문
ZC = ['Z_' + c for c in S4]
h = pd.concat([hr[['SCENARIO_ID', 'HOUR_START', 'SEG_HOUR']], Z.add_prefix('Z_')], axis=1).sort_values('HOUR_START')
ep = dg[dg['FAULT_MODE'].isin(MAIN_MODES)][['SCENARIO_ID', 'START_AT', 'END_AT', 'FAULT_MODE', 'CAUSE_PART']].reset_index(drop=True)
ep['EP'] = ep.index
h = pd.merge_asof(h, ep.sort_values('START_AT'), left_on='HOUR_START', right_on='START_AT', by='SCENARIO_ID')
h = h[(h['HOUR_START'] + pd.Timedelta(hours=1) <= h['END_AT']) & (h['SEG_HOUR'] >= 2)]
fp = h.sort_values('HOUR_START').groupby('EP').tail(6).groupby('EP').agg(
    **{c: (c, 'mean') for c in ZC}, N_H=('HOUR_START', 'size'))
fp = fp.join(ep.set_index('EP')[['SCENARIO_ID', 'FAULT_MODE', 'CAUSE_PART']])
fp = fp[fp['N_H'] >= 3]
PC = [c.replace('Z_', 'P_') for c in ZC]
clipped = fp[ZC].clip(lower=0)
fp[PC] = clipped.div(clipped.sum(axis=1), axis=0).values
print(f'10-1. 열화 구간 {len(fp)}개 (시나리오 {fp["SCENARIO_ID"].nunique()}개)')
show(pd.crosstab(fp['FAULT_MODE'], fp['CAUSE_PART']), '10-1. 원인 × 부품')

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for row, (cols, name) in enumerate([(ZC, '세기 (표준화 점수)'), (PC, '비율 (합 = 1)')]):
    for ax, c in zip(axes[row], cols):
        bp = ax.boxplot([fp.loc[fp['FAULT_MODE'] == m, c] for m in MAIN_MODES], patch_artist=True, widths=0.6)
        for patch, m in zip(bp['boxes'], MAIN_MODES):
            patch.set_facecolor(MODE_COLOR[m]); patch.set_alpha(0.6)
        ax.set_xticks(range(1, 4)); ax.set_xticklabels(MAIN_MODES, rotation=15)
        ax.set_title(c.split('_', 1)[1])
    axes[row, 0].set_ylabel(name)
fig.suptitle('원인별 반응 지문 (열화 구간 끝 6시간)', y=1.0)
fig.tight_layout()
save(fig, '09_eda_원인별_반응지문.png')

# 10-3. 시나리오 단위 leave-one-out으로 원인·부품 분류
rows = []
for tname, target in [('원인 (3가지)', 'FAULT_MODE'), ('부품 (2가지)', 'CAUSE_PART')]:
    for fname, cols in [('세기', ZC), ('비율', PC), ('세기 + 비율', ZC + PC)]:
        pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                                 fp[cols], fp[target], groups=fp['SCENARIO_ID'], cv=LeaveOneGroupOut())
        rows.append({'대상': tname, '특징': fname, '정확도': (pred == fp[target]).mean(),
                     '기준선(최다 클래스)': fp[target].value_counts(normalize=True).max()})
show(pd.DataFrame(rows).set_index(['대상', '특징']).round(3), '10-3. 원인 분류 정확도 → 기준선과 같아 구분 불가')
pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                         fp[ZC + PC], fp['FAULT_MODE'], groups=fp['SCENARIO_ID'], cv=LeaveOneGroupOut())
show(pd.DataFrame(confusion_matrix(fp['FAULT_MODE'], pred, labels=MAIN_MODES),
                  index=[f'실제 {m}' for m in MAIN_MODES], columns=[f'예측 {m}' for m in MAIN_MODES]),
     '10-3. 혼동행렬 (세기 + 비율)')


# 11. 정비 효과 ───────────────────────────────────────────────────────────────

section('11. 정비 효과')

mt = mnt[mnt['END_DT'].notna()].reset_index(drop=True)    # 관측 끝까지 진행 중인 정비는 뺀다
LVL = {'MLD-AMP_D': 'MLD-AMP_ADJ', 'CONTROL_DELAY_D': 'CONTROL_DELAY_ADJ', 'WDL_C_D': 'WDL_C', 'OSC-STK_D': 'OSC-STK_ADJ'}
first_base = hr[hr['SEG'] == 0].groupby('SCENARIO_ID').apply(
    lambda g: g.nsmallest(2, 'HOUR')[list(LVL.values())].median(), include_groups=False)   # 시나리오 처음 수준

# 11-2. 회복률 = (정비 직전 − 정비 직후) / (정비 직전 − 처음)
rows = []
for r in mt.itertuples():
    g = hr[hr['SCENARIO_ID'] == r.SCENARIO_ID]
    before = g[g['HOUR_START'] + pd.Timedelta(hours=1) <= r.STRT_DT].nlargest(2, 'HOUR_START')
    after = g[g['HOUR_START'] >= r.END_DT].nsmallest(2, 'HOUR_START')
    if len(before) < 2 or len(after) < 2:
        continue
    row = {'MNT_TYPE': r.MNT_TYPE}
    for sig, col in LVL.items():
        f0, b, a = first_base.loc[r.SCENARIO_ID, col], before[col].mean(), after[col].mean()
        row[sig] = (b - a) / (b - f0) if (b - f0) * sgn[sig] > abs(TH[sig]) else np.nan
    rows.append(row)
show(pd.DataFrame(rows).groupby('MNT_TYPE').median().round(2), '11-2. 정비 유형별 회복률 중앙값 (1 = 처음 수준으로 완전 회복)')

# 부분 정비 전후 흐름: 처음 수준 대비 변화를 정상 범위 폭으로 나눈 값
part = mt[mt['MNT_TYPE'] == 'partial']
fig, axes = plt.subplots(len(part), 1, figsize=(12, 2.4 * len(part)), sharex=True)
for ax, r in zip(np.atleast_1d(axes), part.itertuples()):
    g = hr[hr['SCENARIO_ID'] == r.SCENARIO_ID].sort_values('HOUR_START')
    rel = (g['HOUR_START'] - r.STRT_DT).dt.total_seconds() / 3600
    keep = (rel > -24) & (rel < 26)
    for sig, color in zip(S4, ['#2f5d8a', '#e07b39', '#4a9b5e', '#9b4a8a']):
        from_first = (g[LVL[sig]] - first_base.loc[r.SCENARIO_ID, LVL[sig]]) * sgn[sig] / p95[sig]
        ax.plot(rel[keep], from_first[keep], label=sig[:-2], color=color, marker='.', ms=3)
    ax.axvspan(0, (r.END_DT - r.STRT_DT).total_seconds() / 3600, color='#bdbdbd', alpha=0.5, lw=0)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(1, color='#eb5757', lw=0.6)
    ax.set_ylabel(f'{r.SCENARIO_ID}\n{r.PART_CD}', rotation=0, ha='right')
np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8)
np.atleast_1d(axes)[0].set_title('부분 정비 전후 신호 (처음 수준 대비, 1 = 정상 범위 폭만큼 이상 방향, 회색: 정비 중)')
np.atleast_1d(axes)[-1].set_xlabel('부분 정비 시작부터 경과 시간(h)')
fig.tight_layout()
save(fig, '10_eda_부분정비_전후신호.png')

# 11-3. 부분 정비 뒤 다음 열화·고장까지
rows = []
for r in part.itertuples():
    d = diag_tr[(diag_tr['SCENARIO_ID'] == r.SCENARIO_ID) & (diag_tr['START_AT'] >= r.END_DT)]
    nxt = d.loc[d['HEALTH_STATE'] == 'degraded', 'START_AT'].min()
    f = fails[(fails['SCENARIO_ID'] == r.SCENARIO_ID) & (fails['FAILURE_DT'] > r.END_DT)]['FAILURE_DT'].min()
    rows.append({'SCENARIO_ID': r.SCENARIO_ID, '부품': r.PART_CD,
                 '정비 끝 → 다음 열화(h)': (nxt - r.END_DT).total_seconds() / 3600 if pd.notna(nxt) else np.nan,
                 '정비 끝 → 고장(h)': (f - r.END_DT).total_seconds() / 3600 if pd.notna(f) else np.nan})
show(pd.DataFrame(rows).round(1), '11-3. 부분 정비 뒤 다음 열화·고장까지 걸린 시간')

# 12. 시계열 분석: 주기·추세·변동성·상태 변화 구간 ─────────────────────────────────

section('12. 시계열 분석: 주기·추세·변동성·상태 변화 구간')

pos = ct.groupby('SCENARIO_ID').indices          # 시나리오별 원자료 행 위치 (ct는 시나리오·시각 순서)


def raw_window(sid, t0, t1):
    w = ct.iloc[pos[sid]]
    return w[(w['RUNNING'] == 1) & (w['MEAS_DT'] >= t0) & (w['MEAS_DT'] < t1)]


def acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x - np.nanmean(x)
    x = np.nan_to_num(x)
    d = (x * x).sum()
    return np.array([(x[:len(x) - k] * x[k:]).sum() / d for k in range(max_lag + 1)])


def mld_period(w):
    """5초 몰드 레벨의 주된 주기(분)와 흔들림 폭(표준편차)"""
    x = w['MLD-LVL'].interpolate().bfill().values
    a = acf(x, 720)
    return (120 + a[120:481].argmax()) * 5 / 60, x.std(), a      # 자기상관이 가장 큰 시차(10~40분 사이) = 주기


# 12-1. 주기 (5초): 정상 시나리오는 첫 운전 구간 2~8시간, 열화는 주 원인 열화 구간의 끝 6시간
rows, ac_n, ac_d = [], [], []
for sid in scen_type.query('MODE == "normal"')['SCENARIO_ID']:
    w = raw_window(sid, START[sid] + pd.Timedelta(hours=2), START[sid] + pd.Timedelta(hours=8))
    if len(w) > 2000:
        per, amp, a = mld_period(w)
        rows.append({'구간': '정상', '주기(분)': per, '흔들림 폭(mm)': amp})
        ac_n.append(a)
for r in dg[dg['FAULT_MODE'].isin(MAIN_MODES)].itertuples():
    w = raw_window(r.SCENARIO_ID, max(r.START_AT, r.END_AT - pd.Timedelta(hours=6)), r.END_AT)
    if len(w) > 2000:
        per, amp, a = mld_period(w)
        rows.append({'구간': '열화 (끝 6시간)', '주기(분)': per, '흔들림 폭(mm)': amp})
        ac_d.append(a)
per_t = pd.DataFrame(rows)
show(per_t.groupby('구간').agg(구간수=('주기(분)', 'size'), 주기_중앙값=('주기(분)', 'median'),
                             주기_최소=('주기(분)', 'min'), 주기_최대=('주기(분)', 'max'),
                             흔들림폭_중앙값=('흔들림 폭(mm)', 'median')).round(2),
     '12-1. 몰드 레벨(5초)의 주기와 흔들림 폭 → 주기는 그대로, 폭만 커진다')

# 12-1. 주기 (1시간): 정상 시나리오의 부하, 인발 부하 원값, 보정한 변화량
nh1 = hr[(hr['MODE'] == 'normal') & (hr['SEG'] == 0)].sort_values(['SCENARIO_ID', 'HOUR'])
ac_h = {c: np.median([acf(g[c].values, 36) for _, g in nh1.groupby('SCENARIO_ID') if g[c].notna().sum() > 48], axis=0)
        for c in ['LOAD', 'WD-LOD', 'WDL_C_D']}
lag_load = int(np.argmax(ac_h['LOAD'][6:]) + 6)
show(pd.DataFrame({'부하 주기의 자기상관': {c: ac_h[c][lag_load] for c in ac_h},
                   '부하와의 상관 (정상 1시간)': {c: nh1[c].corr(nh1['LOAD']) for c in ac_h}}).round(2),
     f'12-1. 부하의 주된 주기 {lag_load}시간 → 인발 부하에는 남고, 보정한 변화량에서는 부하와의 상관이 사라진다')
show(pd.DataFrame(ac_h).iloc[::3].round(2).T, '12-1. 1시간 자기상관 (정상 시나리오 중앙값, 3시간 간격)')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
lag_min = np.arange(721) * 5 / 60
amp = per_t.groupby('구간')['흔들림 폭(mm)'].median()
for a, color, ls, name in [(ac_n, '#9e9e9e', '-', f'정상 ({len(ac_n)}개 시나리오, 폭 {amp["정상"]:.2f}mm)'),
                           (ac_d, '#eb5757', '--', f'열화 끝 6시간 ({len(ac_d)}개 구간, 폭 {amp["열화 (끝 6시간)"]:.2f}mm)')]:
    m = np.median(a, axis=0)
    axes[0].plot(lag_min, m, color=color, ls=ls, lw=2, label=name)
    axes[0].fill_between(lag_min, *np.percentile(a, [25, 75], axis=0), color=color, alpha=0.2)
axes[0].axvline(per_t['주기(분)'].median(), color='k', ls=':', lw=0.8)
axes[0].set_xlabel('시차 (분)'); axes[0].set_ylabel('자기상관')
axes[0].set_title(f'몰드 레벨(5초): 주기 약 {per_t["주기(분)"].median():.0f}분으로 정상·열화가 같고, 폭만 커진다', fontsize=10)
axes[0].legend(fontsize=9)
for c, color, ls, name in [('LOAD', '#9e9e9e', '-', '부하 (운전 조건)'), ('WD-LOD', '#e07b39', '--', '인발 부하 원값 (부하와 겹침)'),
                           ('WDL_C_D', '#2f5d8a', '-', '인발 부하 보정 변화량 (모델에 씀)')]:
    axes[1].plot(range(37), ac_h[c], color=color, ls=ls, lw=2 if c == 'LOAD' else 1.5, marker='.', label=name)
axes[1].axvline(lag_load, color='k', ls=':', lw=0.8)
axes[1].axhline(0, color='k', lw=0.5)
axes[1].set_xlabel('시차 (시간)'); axes[1].set_ylabel('자기상관 (정상 시나리오 중앙값)')
axes[1].set_title(f'1시간 요약: 부하 주기 {lag_load}시간. 보정 뒤 부하와의 상관은 0이지만 주기 성분이 약하게 남는다', fontsize=10)
axes[1].legend(fontsize=9)
fig.tight_layout()
save(fig, '20_eda_주기분석.png')

# 12-2. 추세: 네 신호 평균 점수의 6시간 기울기 (같은 운전 구간 안, 기준 구간 제외)
ts = hr[['SCENARIO_ID', 'SEG', 'SEG_HOUR', 'HOUR_START', 'HTF', 'FAILURE_DT', 'MODE']].copy()
ts['SCORE'] = Z.mean(axis=1).where(hr['SEG_HOUR'] >= 2)
ts[list(LBL.values())] = Z.rename(columns=LBL).where(hr['SEG_HOUR'] >= 2, np.nan).values
grp = ts.groupby(['SCENARIO_ID', 'SEG'])
ts['SLOPE6'] = (ts['SCORE'] - grp['SCORE'].shift(6)) / 6
for c in LBL.values():
    ts['VOL6_' + c] = grp[c].transform(lambda s: s.rolling(6, min_periods=6).std())
gradual = fl.loc[fl['TYPE'] == '점진 고장', ['SCENARIO_ID', 'FAILURE_DT', 'DEG_START']]
ts = ts.merge(gradual, on=['SCENARIO_ID', 'FAILURE_DT'], how='left')
BINS = {'정상': None, '고장 전 36~24h': (-36, -24), '24~12h': (-24, -12), '12~0h': (-12, 0)}


def bin_mask(b):
    if b == '정상':
        return ((ts['MODE'] == 'normal') & (ts['SEG_HOUR'] >= 2)).values
    lo, hi = BINS[b]
    return (ts['DEG_START'].notna() & (ts['HTF'] > lo) & (ts['HTF'] <= hi)).values


trend = pd.DataFrame({b: {'기울기 중앙값 (점수/시간)': ts.loc[bin_mask(b), 'SLOPE6'].median(),
                          '기울기 > 0 비율': (ts.loc[bin_mask(b), 'SLOPE6'].dropna() > 0).mean(),
                          '평균 점수 중앙값': ts.loc[bin_mask(b), 'SCORE'].median()} for b in BINS}).T
show(trend.round(3), '12-2. 추세: 고장에 가까울수록 점수가 빨리 오른다 (점진 고장)')

# 12-3. 변동성: 신호별 6시간 이동표준편차
VOLC = ['VOL6_' + c for c in LBL.values()]
vol = pd.DataFrame({b: ts.loc[bin_mask(b), VOLC].median() for b in BINS}).T
vol.columns = list(LBL.values())
show(vol.round(3), '12-3. 변동성: 6시간 이동표준편차 중앙값 (표준화 점수 단위) → 열화가 진행될수록 흔들림도 커진다')

# 12-4. 상태 변화 구간: 평균 점수의 누적합(CUSUM)으로 변화 시점을 찾고 진단상 열화 시작과 비교
K = 0.5                                           # 찾으려는 변화(정상 0 → 경계 1)의 절반. 누적합의 표준 설정


def cusum_first(g, h):
    s, t_det = 0.0, None
    for v, t in zip(g['SCORE'].values, g['HOUR_START'].values):
        if np.isnan(v):
            continue
        s = max(0.0, s + v - K)
        if s > h:
            t_det = t
            break
    return t_det


seg_n = [g for _, g in ts[(ts['MODE'] == 'normal')].groupby(['SCENARIO_ID', 'SEG'])]
H_GRID = np.round(np.arange(0.1, 30.01, 0.1), 1)
fa = {h: np.mean([any(cusum_first(g, h) is not None for g in seg_n if g['SCENARIO_ID'].iloc[0] == sid)
                  for sid in {g['SCENARIO_ID'].iloc[0] for g in seg_n}]) for h in H_GRID}
H_CUSUM = float(min(h for h in H_GRID if fa[h] <= 0.05))
print(f'12-4. 누적합 기준: 점수가 {K}를 넘는 만큼 쌓아 {H_CUSUM}을 넘으면 변화 시작 '
      f'(정상 시나리오 중 변화로 잡히는 비율 {fa[H_CUSUM]:.1%} ≤ 5%가 되는 가장 작은 값)')
cp, cp_times = [], []
for (sid, fdt), g in ts[ts['DEG_START'].notna() & ts['HTF'].notna()].groupby(['SCENARIO_ID', 'FAILURE_DT']):
    t = cusum_first(g.sort_values('HOUR_START'), H_CUSUM)
    d0 = g['DEG_START'].iloc[0]
    if t is not None:
        t = pd.Timestamp(t, tz='UTC') + pd.Timedelta(hours=1)      # 1시간 구간이 끝나야 알 수 있다
    cp_times.append(t)
    cp.append({'SCENARIO_ID': sid, '변화 감지': t is not None,
               '열화 시작 → 변화 감지(h)': (t - d0).total_seconds() / 3600 if t is not None else np.nan,
               '변화 감지 → 고장(h)': (fdt - t).total_seconds() / 3600 if t is not None else np.nan,
               '열화 시작 → 고장(h)': (fdt - d0).total_seconds() / 3600})
cp = pd.DataFrame(cp)
# 변화 감지 시각에 진단상 열화(배경 마모 포함) 중이었나
deg_any = diag_tr[diag_tr['HEALTH_STATE'] == 'degraded']
def in_degraded(sid, t):
    d = deg_any[deg_any['SCENARIO_ID'] == sid]
    return bool(((d['START_AT'] <= t) & (d['END_AT'] > t)).any())
cp['감지 때 열화 중'] = [t is not None and in_degraded(sid, t) for sid, t in zip(cp['SCENARIO_ID'], cp_times)]
show(cp[['열화 시작 → 변화 감지(h)', '변화 감지 → 고장(h)', '열화 시작 → 고장(h)']].describe().loc[['count', 'min', '25%', '50%', '75%', 'max']].round(1),
     f'12-4. 점진 고장 {len(cp)}건: 변화 감지 {int(cp["변화 감지"].sum())}건')
print(f'  변화 감지가 고장 24시간 이상 전: {int((cp["변화 감지 → 고장(h)"] >= 24).sum())}건, '
      f'열화 시작보다 먼저 감지: {int((cp["열화 시작 → 변화 감지(h)"] < 0).sum())}건 '
      f'(그중 감지 때 다른 열화 구간 중: {int((cp["감지 때 열화 중"] & (cp["열화 시작 → 변화 감지(h)"] < 0)).sum())}건)')

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
order = list(BINS)
axes[0, 0].boxplot([ts.loc[bin_mask(b), 'SLOPE6'].dropna() for b in order], showfliers=False)
axes[0, 0].set_xticks(range(1, 5), order)
axes[0, 0].axhline(0, color='k', lw=0.5)
axes[0, 0].set_ylabel('평균 점수의 6시간 기울기 (점수/시간)')
axes[0, 0].set_title('추세: 고장에 가까울수록 빨리 나빠진다')
for c, color in zip(vol.columns, ['#2f5d8a', '#e07b39', '#4a9b5e', '#9b4a8a']):
    axes[0, 1].plot(order, vol[c], marker='o', color=color, label=c.removesuffix('_D'))
axes[0, 1].set_ylabel('6시간 이동표준편차 중앙값 (표준화 점수)')
axes[0, 1].set_title('변동성: 열화가 진행될수록 흔들림도 커진다')
axes[0, 1].legend(fontsize=8)
axes[1, 0].hist(cp['열화 시작 → 변화 감지(h)'].dropna(), bins=np.arange(-16, 9, 2), color='#2f5d8a', edgecolor='white')
axes[1, 0].axvline(0, color='k', lw=0.8)
axes[1, 0].set_xlabel('진단상 열화 시작 → 누적합 변화 감지 (h, 음수 = 먼저 감지)')
axes[1, 0].set_ylabel('점진 고장 수')
md = cp['열화 시작 → 변화 감지(h)'].median()
axes[1, 0].set_title(f'상태 변화 구간: 누적합 변화 감지는 진단상 열화 시작과 거의 같다 (중앙값 {abs(md):.1f}시간 {"먼저" if md < 0 else "뒤"})', fontsize=10)
ex = cp.dropna().sort_values('열화 시작 → 고장(h)').iloc[len(cp.dropna()) // 2]['SCENARIO_ID']
g = ts[(ts['SCENARIO_ID'] == ex) & ts['DEG_START'].notna() & ts['HTF'].notna()].sort_values('HOUR_START')
s_acc, cs = 0.0, []
for v in g['SCORE'].values:
    s_acc = max(0.0, s_acc + (0 if np.isnan(v) else v - K))
    cs.append(s_acc)
ax = axes[1, 1]
ax.plot(g['HTF'], g['SCORE'], color='k', label='네 신호 평균 점수')
ax2 = ax.twinx()
ax2.plot(g['HTF'], cs, color='#e07b39', label='누적합')
ax2.axhline(H_CUSUM, color='#e07b39', ls='--', lw=0.8)
d0 = (g['DEG_START'].iloc[0] - g['FAILURE_DT'].iloc[0]).total_seconds() / 3600
ax.axvspan(d0, 0, color='#c9b8e8', alpha=0.4, lw=0)
r = cp[cp['SCENARIO_ID'] == ex].iloc[0]
ax.axvline(-r['변화 감지 → 고장(h)'], color='#e07b39', lw=1.5)
ax.axhline(1, color='k', ls=':', lw=0.8)
ax.set_xlabel('고장까지 남은 시간 (h) · 보라 띠: 진단상 열화 · 주황 세로선: 변화 감지')
ax.set_ylabel('평균 점수'); ax2.set_ylabel('누적합')
ax.set_title(f'예: {ex} (열화 시작 {r["열화 시작 → 변화 감지(h)"]:+.0f}h에 감지, 고장 {r["변화 감지 → 고장(h)"]:.0f}h 전)')
fig.tight_layout()
save(fig, '21_eda_추세_변동성_변화점.png')

section('완료: 결론과 결정 사항은 reports/연주_조기경보_결과보고서.md 부록 A 참고')
