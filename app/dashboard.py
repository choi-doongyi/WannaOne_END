"""CAST-PdM: 정비 의사결정 지원 솔루션 (시연용). 팀원 설계 화면(원래 이름 CAST GUARD AI)과 합침 2026-10-15, 이름 변경 2026-10-16.

    python analysis/05_predict_test.py      # 처음 한 번: 모델 학습·저장 (models/risk24.joblib)
    streamlit run app/dashboard.py          # 브라우저에서 http://localhost:8501

- 저장된 모델로 예측만 한다(다시 학습하지 않는다). 원자료가 없으면 app_data/의 미리 계산한 결과만 읽는다(클라우드).
- 오른쪽 위 시연 설정에서 데이터 세트와 '지금 시각'을 고른다. 그 시각까지의 데이터만으로 본 상태를 보여 준다(재생 방식).
- 2단계 경보(결과보고서 5절): 주의 = 열화 감지 → 점검·부품 준비, 경보 = 24시간 안 고장 위험 → 교체

화면 (화면마다 맨 위에 '지금 할 일' 하나와 주 버튼 하나: 콜투액션)
  작업자      내 작업    내 팀의 지금 할 일 → 작업 시작 → 완료 보고(결과·수행 내용·어느 부품이었나). 즉시 조치 설비는 배정 전에도 시작
                         아래 탭: 지금 · 지연 · 예정 · 완료·이력 (재생 시각 기준). 비용은 보여 주지 않는다
              설비 상세  위험 확률·열화 확률과 네 신호의 흐름, 정비 기록, 경보 이유, 근거 따라가기(온톨로지)
  설비관리자  배정할 경보 하나(DSS 추천·기대 총비용·재고 → 배정 / 보류 / 부품 발주)
              ① 대상과 비용(상태 필터·검색, 선택지별 비용 막대와 근거, 바로 아래 배정·보류·부품 발주)
              ② 진행 현황(작업 수정, 이력 CSV) ③ 하위 페이지(노션처럼 눌러 들어감): 설비 상세 · 정책·비용 · 부품 손익 · 모델 정보·한계

기록: 작업 지시는 src/tasks.py, 판단·진행 기록(승인·보류·발주·시작·완료·점검 결과)은 src/actions.py.
  로컬은 actions/ 아래 파일(커밋하지 않음), 공개 클라우드는 보는 사람의 세션에만 남긴다.
온톨로지 (src/ontology.py, OWL/RDF): 지금 시각의 설비·예측·신호·경보·정비 결정·조치를 한 그래프로 잇는다.
"""

import html
import re
import sys
from dataclasses import replace
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.posko_data import load_scenarios
from src.dss import PARTS, PartsPolicy, fail_prob, field_costs, missing_parts, options, plans, recommend
from src import actions, app_data, ontology, tasks
from src.limitations import LIMITS
from src.auth import require_login
from src.serving import cost_curve, explain, failure_hours, load_bundle, score_scenarios

st.set_page_config(page_title='CAST-PdM', page_icon=':material/shield:', layout='wide',
                   initial_sidebar_state='collapsed')

# 실행 방식 (결과보고서 9절·부록 E)
#   로컬: data/posko/ 원자료와 models/risk24.joblib로 직접 계산한다 (내 컴퓨터 안에서만 열림, 비밀번호 없음)
#   클라우드: 원자료가 없고 app_data/만 있다. 미리 계산한 결과만 읽고(모델 파일을 불러오지 않음), 비밀번호를 요구한다
CLOUD = not (ROOT / 'data' / 'posko').exists() and app_data.available()
require_login(st, CLOUD)

SIGNAL_NAMES = {'Z_MLD-AMP': '몰드 레벨 진폭', 'Z_CONTROL_DELAY': '제어 지연',
                'Z_WDL': '인발 부하', 'Z_OSC-STK': '진동 스트로크'}
SPLITS = {'테스트 (정답 없음)': 'test', '검증 (정답 있음)': 'validation'}   # 화면 위 선택지
# 라이트·다크 테마 모두에서 구분되는 색: 경보 빨강 · 주의 노랑 · 정상 회색 · 정비 띠 · 선(파랑·초록) · 기준선(밝은 글자색)
ALARM_RED, CAUTION_YELLOW, OK_GRAY, MNT_GRAY = '#FF667A', '#F7B955', '#7D8FA9', '#3A4D66'
LINE_BLUE, LINE_GREEN, INK = '#60A5FA', '#2DD4A0', '#8392AB'   # INK: 라이트·다크 모두에서 보이는 회청색 (평균선·기준선)
# 화면에 보여 줄 단계 이름: 정상(NORMAL) → 주의(CAUTION, 열화 감지) → 경보(WARNING, 24시간 고장 위험)
# 모델 출력(STAGE)은 '경보'/'주의'/'정상' 그대로 두고 화면에서만 이름을 바꾼다 (결과보고서 8-1절)
LABEL = {'경보': '경보 (WARNING)', '주의': '주의 (CAUTION)', '정상': '정상 (NORMAL)'}
WARN, CAU, NORM = LABEL['경보'], LABEL['주의'], LABEL['정상']
STAGE_COLORS = alt.Scale(domain=[WARN, CAU, NORM], range=[ALARM_RED, CAUTION_YELLOW, OK_GRAY])


# 데이터 불러오기 (처음 한 번만 계산하고 기억해 둔다)

@st.cache_resource
def get_app_data():
    return app_data.load()


@st.cache_resource
def get_bundle():
    return get_app_data()[0] if CLOUD else load_bundle()


@st.cache_data(show_spinner='센서 데이터를 요약하고 예측하는 중...')
def get_data(split):
    if CLOUD:
        pts, hr, contrib, unit, mnt, _ = get_app_data()[1][split]
        return pts, hr, contrib, unit, mnt
    bundle = get_bundle()
    ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
    pts, hr = score_scenarios(ids, bundle)
    contrib, unit = explain(bundle, pts)
    hr = hr.assign(DONE_H=hr['HOUR'] + 1)          # 1시간 구간이 끝난 시각 = 그 값을 알 수 있게 되는 시각
    return pts, hr[['SCENARIO_ID', 'DONE_H', 'SCORE', *SIGNAL_NAMES]], contrib, unit, failure_hours(ids)


try:
    bundle = get_bundle()
except (FileNotFoundError, ValueError) as e:   # 없는 파일, 저장 뒤 바뀐 파일, 빠진 값
    print(f'[dashboard] 모델·데이터를 불러오지 못함: {e!r}', file=sys.stderr)   # 자세한 내용(파일 경로 등)은 서버 로그에만
    st.error('모델·데이터 파일을 쓸 수 없습니다. 관리자에게 문의하세요.' if CLOUD else
             '모델 파일을 쓸 수 없습니다. 터미널에서 `python analysis/05_predict_test.py`를 실행해 다시 만드세요. '
             '(자세한 내용은 터미널 로그)')
    st.stop()
TH = bundle['threshold']
STEP = bundle.get('step_h', 6)           # 예측 간격(시간). 예전 모델 파일은 6시간
CAUTION = bundle.get('caution')          # 주의 모델이 없는 예전 모델 파일이면 경보 1단계만 보여 준다
CTH = CAUTION['threshold'] if CAUTION else None


# 화면 위: CAST-PdM 머리말 · 시연 설정 · 보는 사람 (팀원 설계 화면과 합침 2026-10-15)
#   작업자      현장: 내 팀의 지금 할 일 하나(시작 → 완료 보고), 날짜 대신 재생 시각 기준 지금·지연·예정·완료. 비용 없음
#   설비관리자  의사결정: 배정할 경보 하나(→ ①로 이동), ① 대상·비용·배정 ② 진행 현황 ③ 하위 페이지(근거·정책·모델)
WORKER, MANAGER = '현장 작업', '정비 책임자'              # 기록(src/actions.py)에 쓰는 역할 이름
ROLE_LABEL = {WORKER: '작업자', MANAGER: '설비관리자'}
MANAGER_TABS = {'설비 상세': '설비 상세', '정책·비용': '의사결정', '부품 손익': '비용 계산기', '모델 정보·한계': '모델 정보'}
ICONS = {WORKER: ':material/engineering:', MANAGER: ':material/manage_accounts:',
         '내 작업': ':material/checklist:', '설비 상세': ':material/monitoring:', '정책·비용': ':material/balance:',
         '부품 손익': ':material/calculate:', '모델 정보·한계': ':material/model_training:'}

st.session_state.setdefault('role', WORKER)       # 기본값은 세션 상태로만 정한다 (버튼으로 화면을 바꾸기 때문)
st.session_state.setdefault('split_label', list(SPLITS)[0])
st.session_state.setdefault('now', 72)
with st.container(horizontal=True, vertical_alignment='center', gap='medium'):
    st.markdown('### :material/shield: CAST-:blue[PdM]')
    st.caption('정비 의사결정 지원 솔루션 · 시연')
    st.space('stretch')
    # 시연 설정: 실제 운영에는 없는 재생용 값이라 한 곳에 접어 둔다 (지금 값은 버튼 이름에 보인다)
    with st.popover(f'{st.session_state["split_label"].split(" ")[0]} · {st.session_state["now"]}시간',
                    icon=':material/tune:', help='시연 설정: 데이터 세트와 지금 시각'):
        split_label = st.selectbox('데이터 세트', list(SPLITS), key='split_label',
                                   help='테스트: 정답이 없는 실제 운영과 같은 상황 · 검증: 정답이 있어 실제 고장과 비교할 수 있음')
        split = SPLITS[split_label]
        now = st.slider('지금 시각 (시나리오 시작 후 h)', 0, 72, step=1, key='now',
                        help='이 시각까지의 데이터만으로 본 상태를 보여 줍니다. 왼쪽으로 옮기면 과거로 돌아가 재생할 수 있습니다.')
        show_truth = st.toggle('정답 보기 (검증 세트)', key='truth', disabled=split != 'validation',
                               help='앞으로 일어날 실제 고장 시각(정답)을 겹쳐 봅니다. 모델은 이 정보를 보지 않습니다.') and split == 'validation'
        st.caption('독립된 합성 시나리오를 재생합니다 · 실시간 설비 연결 아님')
# 주소(URL)와 화면 위치를 맞춘다 → 브라우저·마우스의 뒤로 가기/앞으로 가기가 된다
#   ?view=m (설비관리자) &page=설비 상세|정책·비용|... (관리자 하위 페이지) 또는 equip (작업자 설비 상세) &sid=S0294
#   Streamlit은 st.query_params를 바꾸면 기록을 하나 쌓고(pushState), 뒤로 가기로 주소가 바뀌면 그 주소로 다시 실행한다.
#   그래서 주소가 '우리가 마지막으로 쓴 주소'와 다를 때만(뒤로·앞으로 가기, 주소 직접 입력) 주소 → 화면으로 옮기고,
#   나머지 경우는 맨 아래 sync_url()이 화면 → 주소로 쓴다. 기본 화면(작업자·내 작업)은 주소에 아무것도 넣지 않는다.
def url_state():
    return {k: v for k, v in st.query_params.to_dict().items() if k in ('view', 'page', 'sid')}


def apply_url():
    q = url_state()
    if q == st.session_state.get('_nav_last', {}):
        return
    st.session_state['_nav_last'] = q
    role_ = MANAGER if q.get('view') == 'm' else WORKER
    page_ = q.get('page')
    sid_ = q.get('sid') if re.fullmatch(r'S\d{4}', q.get('sid', '')) else None
    before = (st.session_state.get('role'), st.session_state.get('msub'), st.session_state.get('wview'))
    st.session_state['role'] = role_
    if role_ == MANAGER:
        if page_ in MANAGER_TABS:
            st.session_state['msub'] = page_
        else:
            st.session_state.pop('msub', None)
    else:
        st.session_state.pop('msub', None)
        st.session_state['wview'] = '설비 상세' if page_ == 'equip' else '내 작업'
    if sid_:
        st.session_state['sid'] = sid_
        if role_ == MANAGER and page_ not in MANAGER_TABS:
            st.session_state['decision_sid'] = sid_
            st.session_state['nav_pick'] = True
    elif role_ == MANAGER and page_ not in MANAGER_TABS:
        st.session_state.pop('nav_pick', None)
        st.session_state.pop('decision_sid', None)
    if before != (st.session_state.get('role'), st.session_state.get('msub'), st.session_state.get('wview')):
        st.session_state['scroll_top'] = True


def sync_url():
    """지금 화면 위치를 주소에 쓴다 (바뀌었을 때만 → 브라우저 기록 한 칸)"""
    q = {}
    if st.session_state.get('role') == MANAGER:
        q['view'] = 'm'
        if st.session_state.get('msub') in MANAGER_TABS:
            q['page'] = st.session_state['msub']
            if st.session_state['msub'] == '설비 상세' and st.session_state.get('sid'):
                q['sid'] = st.session_state['sid']
        elif st.session_state.get('nav_pick') and st.session_state.get('decision_sid'):
            q['sid'] = st.session_state['decision_sid']
    elif st.session_state.get('wview') == '설비 상세':
        q['page'] = 'equip'
        if st.session_state.get('sid'):
            q['sid'] = st.session_state['sid']
    if q != url_state():
        st.query_params.from_dict(q)
    st.session_state['_nav_last'] = q


def pick_equipment(sid):
    """설비를 고른다 (설비관리자 목록·맨 위 카드): 비용 비교·배정·설비 페이지가 이 설비로, 주소에도 남긴다"""
    st.session_state.update({'decision_sid': sid, 'sid': sid, 'nav_pick': True})


apply_url()


def close_sub():
    """설비관리자 하위 페이지에서 나오기 (역할을 바꿀 때도)"""
    if st.session_state.pop('msub', None):
        st.session_state['scroll_top'] = True


role = st.segmented_control('보는 사람', [WORKER, MANAGER], key='role', required=True, label_visibility='collapsed', on_change=close_sub,
                            format_func=lambda r: f'{ICONS[r]} {ROLE_LABEL[r]}')

pts, hr, contrib, contrib_unit, mnt = get_data(split)


@st.cache_data
def get_plans(split):
    """정기 교체 계획 (calendar 표, 미리 안다고 가정): 시나리오 시작 후 시각(h)과 계획된 부품"""
    if CLOUD:
        return get_app_data()[1][split][5]
    ids = load_scenarios().query('SPLIT == @split')['SCENARIO_ID'].tolist()
    start = load_scenarios().set_index('SCENARIO_ID')['START_AT']
    pl = plans(ids)
    pl['PLAN_H'] = (pl['PLAN_START'] - pl['SCENARIO_ID'].map(start)).dt.total_seconds() / 3600
    return pl


PART_NAMES = {'valve': '밸브', 'oscillator': '진동장치'}
pts_now = pts[pts['T_H'] <= now]
hr_now = hr[hr['DONE_H'] <= now]
mnt_now = mnt[mnt['START_H'] <= now].assign(
    END_SHOWN=lambda d: d['END_H'].clip(upper=now),
    시작=lambda d: d['START_H'].map(lambda x: f'{x:.1f}h'),
    종료=lambda d: d['END_H'].map(lambda x: f'{x:.1f}h' if x <= now else '진행 중 (종료 미확정)'))   # 미래의 종료 시각은 보여 주지 않는다
last_mnt_start = mnt_now.groupby('SCENARIO_ID')['START_H'].max()     # 지금까지 마지막으로 시작된 정비


# 액션 기록 (src/actions.py): 로컬은 CSV 파일(actions/, 커밋하지 않음), 클라우드는 보는 사람의 세션에만
#   클라우드 대시보드는 공개(비밀번호 없음)라 서버 파일에 쓰게 하면 누구나 기록을 남기고 남의 기록을 볼 수 있다
@st.cache_resource
def get_local_log():
    return actions.ActionLog(ROOT / 'actions' / 'action_log.csv')


if CLOUD:
    if 'action_log' not in st.session_state:
        st.session_state['action_log'] = actions.ActionLog(None)
    LOG = st.session_state['action_log']
else:
    LOG = get_local_log()
MNT_START = last_mnt_start.to_dict()


def part_breakeven():
    """원인 부품을 맞힐 확률이 이 값 이상이면 한 부품만 교체하는 쪽이 싸다 (비용 계산기와 같은 계산, 같은 만 원 가정값)"""
    v = bundle['validation']['event']
    cc = cost_curve(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
                    n_false=v['고장과 무관한 경보 묶음: 오경보'], costs=make_cfg(assumptions())[0])
    even = cc.loc[cc['한 부품만 교체'] <= cc['두 부품 모두 교체'].iloc[0], 'p']
    return float(even.iloc[0]) if len(even) else None


def feedback():
    """점검 되먹임: 점검으로 원인 부품을 맞힌 비율이 손익분기를 넘으면 '점검 후 한 부품'"""
    n, acc = LOG.inspection_accuracy()
    be = part_breakeven()
    return {'n': n, 'accuracy': acc, 'breakeven': be, 'ready': be is not None and actions.one_part_ready(n, acc, be)}


def after_last_maintenance(df, time_col):
    """마지막 정비가 시작된 뒤의 행만 남긴다. 정비 전 값은 지금 상태가 아니기 때문이다."""
    start = df['SCENARIO_ID'].map(last_mnt_start).fillna(-np.inf)
    return df[df[time_col] >= start]


def status_table():
    """시나리오마다 지금 시각 기준 상태 한 줄"""
    last = pts_now.sort_values('T_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
    first_alarm = pts_now[pts_now['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
    first_caution = pts_now[pts_now['STAGE'] != '정상'].groupby('SCENARIO_ID')['T_H'].min()
    cur = after_last_maintenance(hr_now.assign(HOUR_START=hr_now['DONE_H'] - 1), 'HOUR_START')
    last_hr = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1)
    worst = last_hr.set_index('SCENARIO_ID')[list(SIGNAL_NAMES)].idxmax(axis=1).map(SIGNAL_NAMES)
    in_mnt = set(mnt[(mnt['START_H'] <= now) & (mnt['END_H'] > now)]['SCENARIO_ID'])
    # 마지막 예측 뒤에 정비가 시작됐으면 그 예측은 정비 전 상태라 쓰지 않는다
    stale = set(last_mnt_start[last_mnt_start > last['T_H'].reindex(last_mnt_start.index).fillna(-1)].index) - in_mnt
    t = pd.DataFrame({'설비 (시나리오)': sorted(pts['SCENARIO_ID'].unique())}).set_index('설비 (시나리오)', drop=False)
    t['위험 확률'] = last['PROB']
    if CAUTION:
        t['열화 확률'] = last['DEGRADED_PROB']
    t['마지막 예측 (h)'] = last['T_H']
    # 경보 유지: 마지막 정비 시작 이후 한 번이라도 경보였으면, 지금 확률이 내려가도 정비 전까지 경보로 둔다
    since = after_last_maintenance(pts_now, 'T_H')
    latched = set(since.loc[since['ALARM'], 'SCENARIO_ID']) - set(last.index[last['ALARM']])
    stage = last['STAGE'].reindex(t.index).fillna('정상').where(~t.index.isin(latched), '경보')
    t['상태'] = np.select([t.index.isin(in_mnt), t.index.isin(stale)], ['정비 중', '정비 후 예측 대기'], default=stage)
    t['경보 유지'] = np.where(t.index.isin(latched) & (t['상태'] == '경보'), '유지 (확률 내려감)', '')
    t.loc[t['상태'].isin(['정비 중', '정비 후 예측 대기']), ['위험 확률', '열화 확률'] if CAUTION else ['위험 확률']] = np.nan
    if CAUTION:
        t['첫 주의 (h)'] = first_caution
    t['첫 경보 (h)'] = first_alarm
    t['가장 많이 벗어난 신호'] = worst
    if show_truth:
        fails = mnt.dropna(subset=['FAIL_H'])
        nxt = fails[fails['FAIL_H'] > now].groupby('SCENARIO_ID')['FAIL_H'].min()
        t['다음 실제 고장까지 (h, 정답)'] = (nxt - now).round(1)
    order = t['상태'].map({'경보': 0, '주의': 1, '정상': 2}).fillna(3)
    t['상태'] = t['상태'].replace(LABEL)
    return t.assign(_o=order).sort_values(['_o', '위험 확률'], ascending=[True, False]).drop(columns='_o')


def mnt_layer(sid, y_field_max):
    """정비 기간(회색 띠)과 고장 시각(빨간 선)"""
    m = mnt_now[mnt_now['SCENARIO_ID'] == sid]
    band = alt.Chart(m).mark_rect(color=MNT_GRAY, opacity=0.35).encode(
        x='START_H:Q', x2='END_SHOWN:Q', tooltip=['종류', 'PART_CD', '시작', '종료'])
    f = m.dropna(subset=['FAIL_H'])
    fail = alt.Chart(f).mark_rule(color=ALARM_RED, strokeWidth=2).encode(x='FAIL_H:Q', tooltip=['종류', 'FAIL_H'])
    layers = [band, fail]
    if show_truth:
        fut = mnt[(mnt['SCENARIO_ID'] == sid) & (mnt['FAIL_H'] > now)].dropna(subset=['FAIL_H'])
        layers.append(alt.Chart(fut).mark_rule(color=ALARM_RED, strokeDash=[4, 4]).encode(
            x='FAIL_H:Q', tooltip=[alt.Tooltip('FAIL_H', title='실제 고장 (정답)')]))
    return layers



# 의사결정 계산 (설비관리자 화면과 작업자 '내 작업'이 같이 쓴다)

def assumptions():
    """설비관리자가 정한 가정값 (없으면 기본값 = 모델 파일의 DSS 가정값 + 팀 결정: 정비팀 2팀, 팀당 하루 8공수)"""
    D = bundle.get('dss')
    base, bc = D['config'], D['config'].costs
    bp = getattr(base, 'parts', PartsPolicy())
    default = {'safety': float(base.safety_max), 'prep': int(base.prep_h), 'crew_wait': int(base.crew_wait_h),
               'n_crews': 2, 'crew_day_h': 8, 'down_h': float(bc.downtime_h), 'penalty': float(bc.failure_penalty),
               'p_other': float(base.p_other_needed), 'max_wait': int(base.max_wait_h), 'delay_h': 0,
               'delay_rate': float(bc.downtime_h),
               # 비용 기준: 서비스 화면은 현장 가정(만 원, 정건우 현장 경험 2026-10-15) 하나로 통일 (2026-10-15 사용자 요청).
               #   데이터셋 점수(과제 공통 비용 규칙)는 모델 평가용이라 결과보고서·발표에만 쓴다
               'mode': COST_MODES[1], 'f_cons': 100.0, 'f_part': 200.0, 'f_labor_part': 20.0, 'f_labor_ph': 3.0,
               'f_crew': 4, 'f_down': 500.0, 'f_fail': 300.0, 'f_delay_rate': 500.0,
               # 부품 조달 (데이터에 없어 미리 정한 가정값, 정건우 현장 경험 2026-10-15)
               'stock_v': int(bp.stock_valve), 'stock_o': int(bp.stock_oscillator), 'lead': float(bp.lead_h),
               'rush_h': float(bp.rush_h), 'rush_mk': float(bp.rush_markup)}
    return {**default, **st.session_state.get('assume', {}), 'mode': COST_MODES[1]}


COST_MODES = ['데이터셋 점수', '현장 가정 (만 원)']


def make_cfg(A):
    base = bundle['dss']['config']
    if A['mode'] == COST_MODES[1]:
        costs = field_costs(consumables=A['f_cons'], major_part=A['f_part'], labor_per_part=A['f_labor_part'],
                            labor_per_person_h=A['f_labor_ph'], crew_size=A['f_crew'], downtime_h=A['f_down'],
                            failure_extra=A['f_fail'])
    else:
        costs = replace(base.costs, downtime_h=A['down_h'], failure_penalty=A['penalty'])
    parts = PartsPolicy(stock_valve=A['stock_v'], stock_oscillator=A['stock_o'], lead_h=A['lead'], rush_h=A['rush_h'],
                        rush_markup=A['rush_mk'])
    cfg = replace(base, safety_max=A['safety'], prep_h=A['prep'], crew_wait_h=A['crew_wait'], p_other_needed=A['p_other'],
                  max_wait_h=A['max_wait'], costs=costs, parts=parts)
    return costs, cfg


def stock_now(cfg):
    """설비마다 지금 남은 예비 재고 {sid: {'valve': n, 'oscillator': n}}.
    처음 재고 − 고장 복구에 쓴 부품 − 작업 완료(두 부품 교체)에 쓴 부품 + 도착한 발주. 정기 교체 부품은 계획과 함께 들어온다고 본다"""
    out = {sid: cfg.parts.stock() for sid in pts['SCENARIO_ID'].unique()}
    for r in mnt_now[mnt_now['종류'] == '고장 후 복구'].itertuples():
        if r.PART_CD in PARTS:
            out[r.SCENARIO_ID][r.PART_CD] = max(0, out[r.SCENARIO_ID][r.PART_CD] - 1)
    lg = LOG.frame(split, now)
    # 작업 완료(두 부품 교체)마다 부품을 쓴다. 설비관리자가 완료를 되돌린 기록('… 완료 → …')은 그만큼 돌려놓는다
    used = (lg[lg['종류'] == '작업 완료'].groupby('설비').size()
            .sub(lg[lg['메모'].astype(str).str.contains('완료 → ', regex=False)].groupby('설비').size(), fill_value=0))
    for sid_, n_ in used.items():
        for q in PARTS:
            out[sid_][q] = max(0, out[sid_][q] - int(max(0, n_)))
    for _, r in lg[lg['종류'] == '부품 발주'].iterrows():
        arrive = r['시각(h)'] + (cfg.parts.rush_h if r['선택지'] == '긴급' else cfg.parts.lead_h)
        if arrive <= now and r['부품'] in PARTS:
            out[r['설비']][r['부품']] += 1
    return out


def in_transit(cfg):
    """아직 도착하지 않은 발주 {sid: [(부품, 도착 시각, 긴급 여부)]}"""
    lg = LOG.frame(split, now)
    out = {}
    for _, r in lg[lg['종류'] == '부품 발주'].iterrows():
        arrive = r['시각(h)'] + (cfg.parts.rush_h if r['선택지'] == '긴급' else cfg.parts.lead_h)
        if arrive > now:
            out.setdefault(r['설비'], []).append((r['부품'], arrive, r['선택지'] == '긴급'))
    return out


def procurement(cfg, stock, ids):
    """주의 설비의 부품 준비: 두 부품 교체에 모자라는 부품과 발주 권고 (발주 시점 = 주의 단계에서 재고 부족)"""
    trans = in_transit(cfg)
    rows = []
    for sid in ids:
        st_ = stock[sid]
        coming = {q for q, _, _ in trans.get(sid, [])}
        short = [q for q in missing_parts(st_, PARTS) if q not in coming]
        rows.append({'설비': sid, '재고': stock_text(st_),
                     '발주': ('발주 필요: ' + ', '.join(PART_NAMES[q] for q in short) + f' (정상 {cfg.parts.lead_h:.0f}시간)') if short else
                            ('도착 예정: ' + ', '.join(f'{PART_NAMES[q]} {a:.0f}h' for q, a, _ in trans.get(sid, []))) if coming else '준비됨',
                     '_short': short})
    return pd.DataFrame(rows, columns=['설비', '재고', '발주', '_short'])


def stock_text(st_):
    return f'밸브 {st_["valve"]} · 진동장치 {st_["oscillator"]}'


def work_plan(A, t):
    """경보 설비마다 추천 선택지와 정비팀 일정 (우선순위, 하루 작업 한도 포함)"""
    costs, cfg = make_cfg(A)
    STOCK = stock_now(cfg)
    n_crews, crew_day_h = A['n_crews'], A['crew_day_h']
    delay_h = A['delay_h']
    delay_rate = A['f_delay_rate'] if A['mode'] == COST_MODES[1] else A['delay_rate']
    curve = bundle['dss']['curve']
    pl = get_plans(split)
    since = after_last_maintenance(pts_now, 'T_H')
    first_alarm_now = since[since['ALARM']].groupby('SCENARIO_ID')['T_H'].min()
    dly = delay_h if delay_h > 0 else None

    def evaluate_options(sid, crew_extra=0.0, p_override=None):
        t_a = first_alarm_now.get(sid, np.nan)
        nxt = pl[(pl['SCENARIO_ID'] == sid) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').head(1)
        wait_h = float(nxt['PLAN_H'].iloc[0] - now) if len(nxt) else np.nan
        part = PART_NAMES.get(nxt['PLAN_PART'].iloc[0], nxt['PLAN_PART'].iloc[0]) if len(nxt) else None
        plan_part = nxt['PLAN_PART'].iloc[0] if len(nxt) else None
        elapsed = max(0.0, now - t_a) if pd.notna(t_a) else 0.0
        c = replace(cfg, crew_wait_h=cfg.crew_wait_h + crew_extra)
        opts = options(wait_h, curve, c, elapsed, delay_h=dly, delay_downtime_h=delay_rate, p_override=p_override,
                       stock=STOCK.get(sid), plan_part=plan_part)
        best, urgent = recommend(opts)
        return {'t_a': t_a, 'wait_h': wait_h, 'part': part, 'plan_part': plan_part, 'elapsed': elapsed, 'opts': opts, 'best': best,
                'urgent': urgent}

    alarm_ids = t.index[t['상태'] == WARN].tolist()
    first = {sid: evaluate_options(sid) for sid in alarm_ids}
    # 우선순위: 지금 정비가 필요한 설비를 위험한 순서로(한 차례 미루면 고장 확률이 큰 순서) 정비팀에 배정한다
    need_crew = sorted([s for s in alarm_ids if not first[s]['best']['선택지'].startswith('다음 정기')],
                       key=lambda s: -fail_prob(curve, first[s]['elapsed'], first[s]['opts'][0]['정비 완료까지(h)'] + costs.replace_h))
    per_day = max(1, int(crew_day_h // costs.replace_h))     # 팀 하나가 하루에 할 수 있는 교체 수 (8공수 ÷ 4시간 = 2)

    def crew_delay(k):
        """k번째(0부터) 정비 작업이 시작될 때까지 기다리는 시간: 팀 수만큼 동시에, 팀당 하루 per_day번, 넘으면 다음 날"""
        rnd = k // n_crews                 # 몇 번째 차례인가 (모든 팀이 동시에 한 대씩)
        day, within = divmod(rnd, per_day)
        return day * 24.0 + within * costs.replace_h

    rows = []
    for k, sid in enumerate(need_crew + [s for s in alarm_ids if s not in need_crew]):
        wait_crew = crew_delay(k) if sid in need_crew else 0.0
        slot = wait_crew
        e = evaluate_options(sid, crew_extra=wait_crew) if wait_crew else first[sid]
        b = e['best']
        rows.append({'순위': k + 1 if sid in need_crew else None, '설비 (시나리오)': sid, '추천': b['선택지'],
                     '정비 완료까지 (h)': b['정비 완료까지(h)'], '기대 총비용': b['기대 총비용'], '완료 전 고장 확률': b['고장 확률'],
                     '제약': ('즉시 조치: 지금 해도 완료 전 고장 확률이 안전 기준 초과' if first[sid]['urgent'] else
                            ('안전 기준 초과: 정비팀 하루 작업 한도 초과 → 다음 날 (팀·공수 추가 필요)' if wait_crew >= 24 else
                             '안전 기준 초과: 정비팀 부족 (팀 추가 필요)') if e['urgent'] else
                            (f'충족 (정비팀 차례 대기 {wait_crew:.0f}시간' + (', 다음 날 작업' if wait_crew >= 24 else '') + ')'
                             if wait_crew else '충족')),
                     '첫 경보 (h)': e['t_a'],
                     '다음 정기 교체': f'{e["wait_h"]:.0f}시간 뒤 ({e["part"]})' if pd.notna(e['wait_h']) else '계획 없음',
                     '_e': e, '_slot': slot,
                     '_start': (e['wait_h'] if b['선택지'].startswith('다음 정기') else b['정비 완료까지(h)'] - costs.replace_h)})
    dec = pd.DataFrame(rows)
    if len(dec):
        state = LOG.job_state(split, now, MNT_START)
        dec.insert(3, '방법', '점검 후 한 부품' if feedback()['ready'] else '두 부품')
        dec.insert(1, '조치', dec['설비 (시나리오)'].map(lambda x: state.get(x, '대기')))
    if len(dec):
        dec.insert(4, '재고', dec['설비 (시나리오)'].map(lambda x: stock_text(STOCK[x])))
    return {'stock': STOCK, 'dec': dec, 'need_crew': need_crew, 'per_day': per_day, 'evaluate_options': evaluate_options,
            'cfg': cfg, 'costs': costs, 'curve': curve, 'pl': pl}


def risk_word(p, safety):
    """고장 확률을 현장 말로 (비용 없이 위험만)"""
    if pd.isna(p):
        return '—'
    return '높음' if p > safety else ('보통' if p >= 0.05 else '낮음')


def ontology_graph(A, t, with_decisions=True):
    """지금 시각의 상태를 온톨로지 그래프로 (src/ontology.py). 화면이 이미 계산한 값만 쓴다"""
    status = pd.DataFrame({
        'STAGE': t['상태'].str.split(' (', regex=False).str[0], 'PROB': t['위험 확률'],
        'DEGRADED_PROB': t['열화 확률'] if CAUTION else np.nan, 'PRED_H': t['마지막 예측 (h)'],
        'FIRST_ALARM_H': t['첫 경보 (h)'], 'FIRST_CAUTION_H': t['첫 주의 (h)'] if CAUTION else np.nan}, index=t.index)
    cur = after_last_maintenance(hr_now.assign(HOUR_START=hr_now['DONE_H'] - 1), 'HOUR_START')
    signals = cur.dropna(subset=list(SIGNAL_NAMES)).sort_values('DONE_H').groupby('SCENARIO_ID').tail(1).set_index('SCENARIO_ID')
    decisions, unit, safety = [], None, None
    if with_decisions and bundle.get('dss'):
        W = work_plan(A, t)
        unit, safety = W['costs'].unit, W['cfg'].safety_max
        decisions = [{'sid': r['설비 (시나리오)'], 'rank': r['순위'], 'constraint': r['제약'], 'best': r['추천'],
                      'options': r['_e']['opts']} for _, r in W['dec'].iterrows()]
    return ontology.build(now, status, signals, mnt_now, get_plans(split), decisions, cost_unit=unit, safety_max=safety,
                          th=TH, cth=CTH, actions=LOG.frame(split, now), feedback=feedback())


def open_detail(table_key, ids):
    """표에서 고른 설비로 '설비 상세' 화면을 연다 (st.dataframe 선택 콜백)"""
    rows = st.session_state[table_key].selection.rows
    if rows:
        go_detail(ids[rows[0]])


def open_sub(key, sid=None):
    """설비관리자 하위 페이지 열기 (노션의 하위 페이지처럼)"""
    st.session_state['msub'] = key
    st.session_state['scroll_top'] = True
    if sid:
        st.session_state['sid'] = sid


def go_detail(sid):
    """버튼에서 '설비 상세' 화면으로 (콜백): 작업자는 '설비 상세' 보기, 설비관리자는 아래 탭"""
    st.session_state['sid'] = sid
    if st.session_state.get('role') == WORKER:
        st.session_state['wview'] = '설비 상세'
    else:
        st.session_state['msub'] = '설비 상세'


SHOW_SUBPAGE_LIST = False   # 설비관리자 ③ 하위 페이지 목록 (2026-10-15 사용자 요청으로 숨김. 설비 페이지는 비용 비교 제목 옆 버튼으로 들어감)
PLAN_GUESS_H = 12   # 정기 교체 계획이 없을 때 비용 비교에만 쓰는 '계획 정비' 시점 (가정값, 추천에는 안 씀)
STAGE_DOT = {'경보': 'red', '주의': 'orange', '정상': 'green', '정비 중': 'gray', '정비 후 예측 대기': 'gray'}   # 설비 목록 상태 점 색


def act(kind, sid, role_, note_key=None, **kw):
    """조치 기록 버튼 콜백: 기록하고 알림 (note_key: 메모 입력칸의 key)"""
    LOG.record(kind, split, sid, now, role_, note=st.session_state.get(note_key, '') if note_key else '', **kw)
    st.toast(f'{sid}: {kind}', icon=':material/check_circle:')


RISK_BADGE = {'높음': ':red-badge[:material/warning: 끝나기 전 고장 위험 높음]', '보통': ':orange-badge[끝나기 전 고장 위험 보통]',
              '낮음': ':green-badge[끝나기 전 고장 위험 낮음]', '—': ''}
# 작업 지시 저장 (src/tasks.py): 로컬은 SQLite(actions/tasks.sqlite3, 커밋하지 않음), 공개 클라우드는 보는 사람의 세션에만

@st.cache_resource
def get_local_tasks():
    return tasks.TaskStore(ROOT / 'actions' / 'tasks.sqlite3')


if CLOUD:
    if 'task_store' not in st.session_state:
        st.session_state['task_store'] = tasks.TaskStore(None)
    TASKS = st.session_state['task_store']
else:
    TASKS = get_local_tasks()

PRIORITY_BADGE = {'긴급': ':red-badge[긴급]', '높음': ':orange-badge[높음]', '보통': ':gray-badge[보통]'}
TASK_BADGE = {'대기': ':gray-badge[대기]', '진행 중': ':blue-badge[:material/build: 진행 중]', '완료': ':green-badge[:material/task_alt: 완료]'}


def fmt_cost(x, costs):
    """비용 표시: 만 원은 1억 이상이면 억 원으로"""
    if costs.unit == '만 원' and abs(x) >= 10000:
        return f'{x / 10000:,.2f}억 원'
    return f'{x:,.0f}{"만 원" if costs.unit == "만 원" else "점"}'


def cost_bars(items, costs):
    """선택지별 기대 총비용 막대 (팀원 설계의 비용 막대). items: [(이름, 값, 색, 추천 여부)]. 글은 모두 escape"""
    top = max([v for _, v, _, _ in items] + [1e-9])
    rows = []
    for name, v, color, rec in items:
        w = max(2.0, min(100.0, v / top * 100))
        rows.append(
            f'<div style="background:rgba(127,140,160,0.08);border:1px solid {LINE_BLUE if rec else "rgba(127,140,160,0.35)"};border-radius:11px;'
            f'padding:9px 13px;margin:6px 0">'
            f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:center">'
            f'<span style="opacity:0.85;font-weight:650">{html.escape(name)}{" · 추천" if rec else ""}</span>'
            f'<span style="font-weight:800;font-size:1.1rem;font-variant-numeric:tabular-nums">{html.escape(fmt_cost(v, costs))}</span></div>'
            f'<div style="height:8px;border-radius:99px;background:rgba(127,140,160,0.25);margin-top:8px;overflow:hidden">'
            f'<div style="height:100%;width:{w:.1f}%;border-radius:99px;background:{color}"></div></div></div>')
    st.markdown(''.join(rows), unsafe_allow_html=True)


def start_task(tid, sid):
    if TASKS.start(tid, now):
        LOG.record('작업 시작', split, sid, now, WORKER)
        st.toast(f'{sid}: 작업 시작', icon=':material/play_arrow:')


def urgent_start(sid, team, option):
    """즉시 조치 설비: 배정 전이라도 현장이 바로 시작한다 (안전이 먼저). 작업을 만들고 시작까지 기록"""
    ok = TASKS.add(split, sid, f'{sid} 즉시 조치: 두 부품 교체', team, '긴급', now, now, tasks.DEFAULT_GUIDE,
                   '지금 교체해도 완료 전 고장 확률이 안전 기준을 넘음 → 배정 전 긴급 시작', option)
    if ok:
        start_task(TASKS.open_task(split, sid, now)['id'], sid)
    else:
        st.toast(f'{sid}: 이미 열린 작업이 있습니다', icon=':material/info:')


def assign_task(sid, title, team, priority, due_h, guide, reason, option):
    """작업 배정 = 정비 결정 승인 + 작업 지시 (같은 설비에 열린 작업이 있으면 배정하지 않는다)"""
    try:
        ok = TASKS.add(split, sid, title, team, priority, due_h, now, guide, reason, option)
    except ValueError as e:
        st.toast(str(e), icon=':material/error:')
        return
    if ok:
        LOG.record('승인', split, sid, now, MANAGER, option=option)
        st.toast(f'{sid}: {team}에 배정', icon=':material/assignment_ind:')
    else:
        st.toast(f'{sid}: 이미 열린 작업이 있어 배정하지 않았습니다', icon=':material/block:')


def assign_box(sid, srow, sstage, job_, dec_i, W, cfg, load):
    """비용 비교 바로 아래의 실행 칸: 팀 고르고 배정(주 버튼 하나) · 보류 · 부품 발주. 세부 내용은 접어 둔다"""
    existing = TASKS.blocking(split, sid, now)
    if existing:
        when = '' if existing['created_h'] <= now else f' (재생 시각 {existing["created_h"]:.0f}시간에 만든 작업)'
        st.warning(f'이미 작업 #{int(existing["id"])} · {existing["title"]} ({existing["team"]}){when}이 있어 다시 배정하지 않습니다. '
                   '아래 ② 진행 현황에서 확인하세요.', icon=':material/block:')
        return
    if job_ == '작업 완료':
        return
    alarm = sid in dec_i.index
    urgent = alarm and dec_i.loc[sid, '제약'].startswith('즉시')
    rec = dec_i.loc[sid, '_e']['best']['선택지'] if alarm else ('점검·부품 준비' if sstage == '주의' else '상태 점검')
    with st.form(f'assign_{sid}', border=True):
        st.markdown(f'**배정: {sid} {rec}**')
        c1, c2 = st.columns([1.3, 1], vertical_alignment='bottom')
        team_f = c1.selectbox('담당 팀', list(tasks.TEAMS), index=list(tasks.TEAMS).index(min(load, key=load.get)),
                              format_func=lambda k: f'{k} · 열린 작업 {load[k]}건')
        go = c2.form_submit_button('작업자 화면으로 배정', icon=':material/send:', type='primary', width='stretch')
        with st.expander('작업 내용 바꾸기 (작업명 · 우선순위 · 시작 시각 · 점검 가이드)'):
            title_f = st.text_input('작업명', value=f'{sid} {rec}', max_chars=tasks.LIMITS['title'])
            d1, d2 = st.columns(2)
            prio_f = d1.selectbox('우선순위', tasks.PRIORITIES, index=0 if urgent else ({'경보': 1, '주의': 1}.get(sstage, 2)))
            due_f = d2.number_input('시작 예정 (h)', 0.0, 240.0,
                                    min(240.0, float(now + max(0.0, dec_i.loc[sid, '_start']))) if alarm else float(now), 1.0, format='%.0f')
            guide_f = st.text_area('점검 가이드', value=tasks.DEFAULT_GUIDE, height=140, max_chars=tasks.LIMITS['guide'])
            reason_f = st.text_area('추천 이유', max_chars=tasks.LIMITS['reason'],
                                    value=f'{sid} · {srow["상태"]} · 추천 {rec}' + (f' · 24시간 안 고장 확률 {srow["위험 확률"]:.2f}'
                                                                                 if pd.notna(srow['위험 확률']) else ''))
        st.caption('배정하면 그 팀 작업자 화면의 "지금 할 일"에 바로 나타납니다.')
        if go:
            assign_task(sid, title_f, team_f, prio_f, due_f, guide_f, reason_f, rec)
            st.rerun()
    if sstage in ('경보', '주의'):
        coming_ = {x for x, _, _ in in_transit(cfg).get(sid, [])}
        with st.container(horizontal=True):
            if alarm:
                st.button('보류', icon=':material/pause_circle:', key='hold_btn', on_click=act,
                          args=('보류', sid, MANAGER), kwargs={'option': rec}, help='지금은 배정하지 않고 지켜봅니다. 배정할 경보 수에서 빠집니다.')
            for part in PARTS:
                empty = W['stock'][sid][part] < 1
                st.button(f'{PART_NAMES[part]} ' + ('조달 중' if part in coming_ else
                                                    f'긴급 발주 ({cfg.parts.rush_h:.0f}h)' if empty else f'보충 발주 ({cfg.parts.lead_h:.0f}h)'),
                          icon=':material/local_shipping:', key=f'order_{part}', disabled=part in coming_, on_click=act,
                          args=('부품 발주', sid, MANAGER), kwargs={'part': part, 'option': '긴급' if empty else '정상'})


def cancel_task(tid, sid, title):
    """배정 취소 (콜백): 완료 전 작업을 지우고 진행 기록에 '배정 취소'를 남긴다"""
    if TASKS.cancel(tid):
        LOG.record('배정 취소', split, sid, now, MANAGER, note=f'#{tid} {title}')
        st.session_state.pop('edit_tid', None)
        st.toast(f'{sid}: 배정을 취소했습니다. 다시 배정할 수 있습니다', icon=':material/undo:')
    else:
        st.toast('완료된 작업이거나 이미 없는 작업입니다', icon=':material/error:')


def team_load(tf):
    """팀마다 열린 작업 수 (배정 기본값: 가장 한가한 팀)"""
    o = tf[tf['상태'] != '완료']
    return {k: int((o['team'] == k).sum()) for k in tasks.TEAMS}


def task_line(r):
    """작업 한 줄 (카드 목록용)"""
    with st.container(border=True):
        st.markdown(f'**#{int(r["id"])} {r["title"] if str(r["title"]).startswith(r["sid"]) else r["sid"] + " — " + r["title"]}**  \n'
                    + ' '.join([PRIORITY_BADGE.get(r['priority'], ''), TASK_BADGE.get(r['상태'], ''),
                                ':red-badge[지연]' if r['구분'] == '지연' else '', f':gray-badge[{r["team"]}]']))
        st.caption(f'예정 {r["due_h"]:.0f}시간 · 배정 {r["created_h"]:.0f}시간'
                   + (f' · 완료 {r["completed_h"]:.0f}시간' if pd.notna(r['completed_h']) and r['상태'] == '완료' else '')
                   + (f' · {r["reason"]}' if r['reason'] else ''))
        if r['상태'] == '완료':
            st.markdown(f'완료 보고: **{r["outcome"]}** · 점검 판단 {PART_NAMES.get(r["judged"], r["judged"])} · '
                        f'확인된 원인 {PART_NAMES.get(r["confirmed"], r["confirmed"])}  \n{r["details"]}')
        else:
            st.button('이 작업 먼저', key=f'focus_{int(r["id"])}', icon=':material/push_pin:',
                      on_click=lambda i=int(r['id']): st.session_state.update({'focus_tid': i}))


# 0. 작업자: 내 작업 (지금 할 일 하나 → 시작 → 완료 보고)

page = None
if role == WORKER:
    st.session_state.setdefault('wview', '내 작업')
    wview = st.segmented_control('작업자 화면', ['내 작업', '설비 상세'], key='wview', required=True,
                                 label_visibility='collapsed', format_func=lambda x: f'{ICONS[x]} {x}')
    page = '설비 상세' if wview == '설비 상세' else None
    if wview == '내 작업':
        if not bundle.get('dss'):
            st.warning('이 모델 파일에는 정비 계획 정보가 없습니다. python analysis/05_predict_test.py 로 다시 만드세요.')
            st.stop()
        A = assumptions()
        t = status_table()
        W = work_plan(A, t)
        dec, cfg, costs = W['dec'], W['cfg'], W['costs']
        tf = TASKS.frame(split, now)
        st.session_state.setdefault('my_team', list(tasks.TEAMS)[0])
        with st.container(horizontal=True, vertical_alignment='center'):
            st.subheader('내 작업', icon=':material/checklist:')
            st.space('stretch')
            team = st.segmented_control('내 팀', list(tasks.TEAMS), key='my_team', required=True, label_visibility='collapsed',
                                        help='팀원: ' + ' / '.join(f'{k} ' + '·'.join(v) for k, v in tasks.TEAMS.items()))
        mine = tf[tf['team'] == team]
        open_ = mine[mine['상태'] != '완료']
        open_ = open_.assign(_r=(open_['상태'] != '진행 중').astype(int)).sort_values('_r', kind='stable')
        dec_i = dec.set_index('설비 (시나리오)') if len(dec) else pd.DataFrame()
        busy = TASKS.blocked_sids(split, now)
        js = LOG.job_state(split, now, MNT_START)
        done_ = {x for x, v in js.items() if v == '작업 완료'}     # 이번 운전 구간에서 이미 교체를 마친 설비
        urgent_un = [x for x in dec_i.index if dec_i.loc[x, '제약'].startswith('즉시') and x not in busy | done_] if len(dec_i) else []
        ft = st.session_state.get('focus_tid')
        focus = open_[open_['id'] == ft].iloc[0] if ft in set(open_['id']) else (open_.iloc[0] if len(open_) else None)

        with st.container(border=True):
            if focus is not None:
                sid = focus['sid']
                info = dec_i.loc[sid] if sid in dec_i.index else None
                plan_part = info['_e'].get('plan_part') if info is not None else None
                need = PARTS if not str(focus['option']).startswith('다음 정기') else [q for q in PARTS if q != plan_part]
                miss_ = missing_parts(W['stock'][sid], need)
                st.caption(f'지금 할 일 · {team} · 작업 #{int(focus["id"])}')
                st.markdown(f'### {focus["title"] if str(focus["title"]).startswith(sid) else sid + " — " + focus["title"]}')
                st.markdown(' '.join(x for x in [
                    PRIORITY_BADGE.get(focus['priority'], ''), TASK_BADGE.get(focus['상태'], ''),
                    ':red-badge[:material/schedule: 예정 시각 지남]' if focus['구분'] == '지연' else '',
                    (':red-badge[:material/emergency: 즉시 조치]' if info is not None and info['제약'].startswith('즉시') else ''),
                    (RISK_BADGE.get(risk_word(info['완료 전 고장 확률'], cfg.safety_max), '') if info is not None else ''),
                    (f':orange-badge[:material/inventory_2: {", ".join(PART_NAMES[q] for q in miss_)} 재고 없음 · 긴급 조달 '
                     f'{cfg.parts.rush_h:.0f}시간]' if miss_ else ':green-badge[:material/inventory_2: 부품 준비됨]')] if x))
                st.markdown(f'**예정 {focus["due_h"]:.0f}시간 → {focus["due_h"] + costs.replace_h:.0f}시간** · {focus["reason"]}')
                with st.expander('점검 가이드', icon=':material/fact_check:', expanded=focus['상태'] == '진행 중'):
                    st.markdown(focus['guide'].replace('\n', '  \n'))
                    st.caption('현장 안전 규정과 작업 허가가 우선입니다.')
                if focus['상태'] == '대기':
                    with st.container(horizontal=True):
                        st.button('작업 시작', icon=':material/play_arrow:', type='primary', on_click=start_task,
                                  args=(int(focus['id']), sid))
                        st.button('설비 보기', icon=':material/monitoring:', on_click=go_detail, args=(sid,))
                else:
                    st.markdown('**완료 보고** · 결과와 수행 내용을 남겨야 완료됩니다')
                    with st.form(f'complete_{int(focus["id"])}', border=False):
                        c1, c2, c3 = st.columns([1.2, 1, 1])
                        outcome = c1.segmented_control('작업 결과', tasks.OUTCOMES, default='이상 발견', required=True)
                        judged = c2.segmented_control('점검으로 판단한 부품', tasks.JUDGED, default='구분 못함', required=True,
                                                      format_func=lambda x: PART_NAMES.get(x, x))
                        confirmed = c3.segmented_control('교체 뒤 확인된 원인 부품', tasks.CONFIRMED, default='아직 모름', required=True,
                                                         format_func=lambda x: PART_NAMES.get(x, x))
                        details = st.text_area('수행 내용 (필수)', max_chars=tasks.LIMITS['details'],
                                               placeholder='예: 진동장치 베어링 마모 확인, 두 부품 교체 후 신호 정상')
                        if st.form_submit_button('완료 보고', icon=':material/task_alt:', type='primary'):
                            try:
                                done = TASKS.complete(int(focus['id']), now, outcome, details, judged, confirmed)
                            except ValueError as e:
                                st.error(str(e))
                            else:
                                if done:
                                    LOG.record('작업 완료', split, sid, now, WORKER, note=details)
                                    LOG.record('점검 결과', split, sid, now, WORKER, judged=judged, confirmed=confirmed, note=outcome)
                                    st.session_state.pop('focus_tid', None)
                                    st.toast(f'{sid}: 완료 보고', icon=':material/task_alt:')
                                    st.rerun()
                                else:
                                    st.warning('이미 완료된 작업입니다.')
                    st.button('설비 보기', icon=':material/monitoring:', on_click=go_detail, args=(sid,))
            elif urgent_un:
                sid = urgent_un[0]
                st.caption('지금 할 일 · 배정 전 즉시 조치')
                st.markdown(f'### {sid} — 즉시 조치: 두 부품 교체')
                st.markdown(':red-badge[:material/emergency: 지금 교체해도 완료 전 고장 확률이 안전 기준 초과] :gray-badge[아직 배정 전]')
                st.markdown('안전이 먼저입니다. 설비관리자의 배정을 기다리지 않고 바로 시작할 수 있습니다.')
                with st.container(horizontal=True):
                    st.button(f'{team}이 바로 시작', icon=':material/play_arrow:', type='primary', on_click=urgent_start,
                              args=(sid, team, dec_i.loc[sid, '추천']))
                    st.button('설비 보기', icon=':material/monitoring:', on_click=go_detail, args=(sid,))
            else:
                st.markdown('##### :material/check_circle: 지금 할 작업이 없습니다')
                st.caption('설비관리자가 배정하면 여기에 나타납니다.')
        if focus is not None and urgent_un:
            st.markdown(f':red-badge[:material/emergency: 배정 전 즉시 조치 {len(urgent_un)}대] {", ".join(urgent_un)} — '
                        '설비관리자에게 알리세요')

        cnt = mine['구분'].value_counts()
        t1, t2, t3, t4 = st.tabs([f'지금 {cnt.get("지금", 0)}', f'지연 {cnt.get("지연", 0)}', f'예정 {cnt.get("예정", 0)}',
                                  f'완료·이력 {cnt.get("완료", 0)}'])
        for tab, key in [(t1, '지금'), (t2, '지연'), (t3, '예정'), (t4, '완료')]:
            with tab:
                rows = mine[mine['구분'] == key]
                if rows.empty:
                    st.caption('없습니다.')
                for _, r in rows.iterrows():
                    task_line(r)
        st.caption('재생 시각 기준: 지금 = 예정 시각이 됨 · 지연 = 예정 시각이 지났는데 시작 전 · 예정 = 아직 · '
                   + ('기록은 이 브라우저 세션에만 남습니다(공개 화면).' if CLOUD else '기록은 이 컴퓨터의 actions/tasks.sqlite3에 남습니다.'))


# 1. 설비관리자: 배정할 경보 하나 → ① 대상·비용·배정 ② 진행 현황, ③ 아래 탭

else:
    if not bundle.get('dss'):
        st.warning('이 모델 파일에는 의사결정 정보가 없습니다. 터미널에서 `python analysis/05_predict_test.py`로 다시 만드세요.')
        st.stop()
    # 하위 페이지 (노션처럼): 설비 상세·정책·비용·부품 손익·모델 정보는 목록에서 눌러야 들어가고, 위의 경로로 돌아온다
    sub = st.session_state.get('msub')
    if st.session_state.pop('scroll_top', False):    # 페이지를 바꾸면 맨 위(경로)부터 보이게
        st.iframe('<script>const d = window.parent.document;'
                  '["[data-testid=stMain]", "[data-testid=stAppViewContainer]", "section.main"]'
                  '.forEach(q => { const e = d.querySelector(q); if (e) e.scrollTo(0, 0); });'
                  'window.parent.scrollTo(0, 0);</script>', height=1)   # 고정된 글(사용자 입력 없음)만 넣는다
    if sub in MANAGER_TABS:
        with st.container(horizontal=True, vertical_alignment='center', gap='small'):
            st.button('정비 의사결정', icon=':material/arrow_back:', type='tertiary', key='sub_back', on_click=close_sub)
            st.markdown(f'/ &nbsp;{ICONS[sub]} **{st.session_state.get("sid", "") if sub == "설비 상세" else sub}**')
        page = MANAGER_TABS[sub]
    else:
        A = assumptions()
        t = status_table()
        W = work_plan(A, t)
        dec, cfg, costs = W['dec'], W['cfg'], W['costs']
        dec_i = dec.set_index('설비 (시나리오)') if len(dec) else pd.DataFrame()
        tf = TASKS.frame(split, now)
        busy = TASKS.blocked_sids(split, now)      # 배정을 막는 작업이 있는 설비 (재생에서 미래에 만든 작업 포함, 배정 규칙과 같음)
        js = LOG.job_state(split, now, MNT_START)
        load = team_load(tf)
        st.subheader('정비 의사결정', icon=':material/balance:')

        k1, k2, k3, k4, k5 = st.columns(5, border=True)
        # 배정할 경보: 열린 작업이 없고, 보류하지 않았고, 이번 운전 구간에서 작업을 마치지 않은 경보 설비
        #   (재생 데이터에는 우리가 한 교체가 반영되지 않아 경보가 남으므로, 완료한 설비는 정비가 시작될 때까지 다시 올리지 않는다)
        queue = dec[~dec['설비 (시나리오)'].isin(busy) & ~dec['설비 (시나리오)'].map(lambda x: js.get(x)).isin(['보류', '작업 완료'])] if len(dec) else dec
        k1.metric('경보', int((t['상태'] == WARN).sum()), help='24시간 안 고장 위험. 정비 중인 설비는 빼고 셉니다')
        k2.metric('주의', int((t['상태'] == CAU).sum()) if CAUTION else '—', help='열화 감지: 점검·부품 준비')
        k3.metric('배정할 경보', len(queue), help='경보인데 아직 작업이 없는 설비 (보류 제외)')
        k4.metric('진행 중 작업', int((tf['상태'] == '진행 중').sum()))
        k5.metric('지연', int((tf['구분'] == '지연').sum()), help='예정 시각이 지났는데 시작 전인 작업')

        # 배정할 경보 (CTA): 즉시 조치 → 정비팀 배정 순서
        with st.container(border=True):
            if len(queue):
                q = queue.assign(_u=(~queue['제약'].str.startswith('즉시')).astype(int)).sort_values(['_u', '순위'], na_position='last')
                r = q.iloc[0]
                fsid, fe = r['설비 (시나리오)'], r['_e']
                fb_ = fe['best']
                urgent = r['제약'].startswith('즉시')
                held = sum(v == '보류' for v in js.values())
                st.caption(f'배정할 경보 {len(queue)}건' + (f' · 보류 {held}건' if held else ''))
                st.markdown(f'### {fsid} — {fb_["선택지"]}')
                miss_ = missing_parts(W['stock'][fsid], PARTS)
                st.markdown(' '.join(x for x in [
                    ':red-badge[:material/emergency: 즉시 조치: 지금 해도 안전 기준 초과]' if urgent else '',
                    f':gray-badge[정비팀 순서 {int(r["순위"])}번]' if pd.notna(r['순위']) else ':gray-badge[정기 교체 때]',
                    f':violet-badge[방법: {r["방법"]}]',
                    (f':orange-badge[:material/inventory_2: {", ".join(PART_NAMES[x] for x in miss_)} 재고 없음]' if miss_
                     else ':green-badge[:material/inventory_2: 부품 준비됨]')] if x))
                st.markdown(f'기대 총비용 **{fmt_cost(fb_["기대 총비용"], costs)}** · 완료 전 고장 확률 **{fb_["고장 확률"]:.2f}** '
                            f'(안전 기준 {cfg.safety_max:.2f}) · {r["다음 정기 교체"]} · 예비 재고 {stock_text(W["stock"][fsid])}')
                st.caption('아래 ① 목록에서 이 설비를 누르면 비용 비교와 배정이 열립니다.')
            else:
                st.markdown('##### :material/check_circle: 지금 배정할 경보가 없습니다')
                st.caption(f'열린 작업 {len(busy)}건 · 팀별 ' + ', '.join(f'{k} {v}건' for k, v in load.items()))

        # ① 대상·비용
        st.markdown('#### ① 대상과 비용')
        left, right = st.columns([1, 1.15], gap='medium')
        with left:
            filt = st.segmented_control('대상 필터', ['경보·주의', '경보', '주의', '정상', '전체'], default='경보·주의', key='stage_filter',
                                        required=True)
            kw = st.text_input('설비 검색', placeholder='예: S0294', key='scenario_search')
            short = t['상태'].str.split(' (', regex=False).str[0]
            keep = {'경보·주의': ['경보', '주의'], '경보': ['경보'], '주의': ['주의'], '정상': ['정상']}.get(filt)
            vis = t[short.isin(keep)] if keep else t
            if kw.strip():
                vis = vis[vis.index.str.contains(kw.strip(), case=False, regex=False)]
            ids_ = vis.index.tolist()
            if st.session_state.get('decision_sid') not in t.index:
                st.session_state['decision_sid'] = ids_[0] if ids_ else t.index[0]
            st.caption(f'{len(ids_)}대 · 위험한 순서 · 누르면 오른쪽에 그 설비의 비용 비교와 배정이 열립니다')
            # 설비 목록: 버튼 한 줄 = 설비 하나 (상태 색 점 · 상태 · 위험 확률 · 진행 중인 작업). 고른 설비는 주 버튼 색
            with st.container(height=360, border=True):
                if not ids_:
                    st.caption('조건에 맞는 설비가 없습니다.')
                for x in ids_:
                    stg = short[x]
                    p_ = vis.loc[x, '위험 확률']
                    # 목록에는 지금 의미 있는 진행만 (배정 취소·승인 같은 지난 단계는 숨김: 취소한 설비는 그냥 배정 대기로 보인다)
                    job = '작업 있음' if x in busy else (js.get(x, '') if js.get(x) in ('보류', '작업 시작', '작업 완료') else '')
                    label = ' · '.join(v for v in [f':{STAGE_DOT.get(stg, "gray")}[●] {x}', stg,
                                                   f'{p_:.0%}' if pd.notna(p_) else '', job] if v)
                    st.button(label, key=f'pick_{x}', width='stretch',
                              type='primary' if x == st.session_state['decision_sid'] else 'secondary',
                              on_click=pick_equipment, args=(x,))
        sid = st.session_state['decision_sid']
        srow = t.loc[sid]
        sstage = srow['상태'].split(' (')[0]
        with left:
            st.caption(f'선택: {sid} · {sstage} · {now}h')
        with right:
            with st.container(horizontal=True, vertical_alignment='center'):
                st.markdown(f'##### {sid} · 정비 비용 비교')
                st.space('stretch')
                st.button(f'{sid} 페이지', icon=':material/description:', type='tertiary', key='open_sid_page',
                          on_click=open_sub, args=('설비 상세', sid), help='위험 확률·네 신호·정비 이력을 하위 페이지로 엽니다')
            st.caption(f'상태: {sstage}' + (f' · 24시간 고장 위험 {srow["위험 확률"]:.1%}' if pd.notna(srow['위험 확률']) else ''))
            # 이미 손댄 설비는 먼저 그 사실을 보여 준다 (재생 데이터에는 우리 교체가 반영되지 않아 경보가 그대로 남기 때문)
            job_ = '작업 있음' if sid in busy else js.get(sid)
            JOB_NOTE = {'작업 완료': '작업 완료 · 재가동 뒤 새 예측이 나올 때까지 경보 표시가 남습니다',
                        '작업 있음': '작업이 이미 배정돼 있습니다 · 아래 ② 진행 현황에서 확인',
                        '작업 시작': '작업 진행 중', '보류': '보류한 설비 · 다시 배정하려면 아래에서 배정', '승인': '승인됨 · 작업 배정 전'}
            if job_ in JOB_NOTE:
                st.markdown(f':green-badge[:material/task_alt: {JOB_NOTE[job_]}]' if job_ == '작업 완료'
                            else f':blue-badge[:material/info: {JOB_NOTE[job_]}]')
            if sid in dec_i.index:
                r_ = dec_i.loc[sid]
                e = r_['_e']
                b = e['best']
                p24 = fail_prob(W['curve'], e['elapsed'], 24)
                OPT_COLOR = lambda name: LINE_BLUE if name.startswith('지금') else CAUTION_YELLOW
                items = [(o['선택지'], o['기대 총비용'], OPT_COLOR(o['선택지']), o is b) for o in e['opts']]
                # 계획 정비 막대는 늘 보여 준다: 엔진 조건(최대 대기·부품·정비팀) 밖이거나 정기 교체 계획이 없으면 비교용으로만 계산
                plan_note = ''
                if all(o['선택지'].startswith('지금') for o in e['opts']):
                    loose = replace(cfg, max_wait_h=1e9, prep_h=0.0, crew_wait_h=0.0)
                    has_plan = pd.notna(e['wait_h'])
                    # 정기 교체가 있으면 그것으로, 없거나 (부품이 모자라) 계산이 안 되면 N시간 뒤 정지를 가정한다
                    for d_ in ((None,) if has_plan else ()) + (PLAN_GUESS_H, 2 * PLAN_GUESS_H):
                        po_ = [o for o in options(e['wait_h'] if d_ is None else np.nan, W['curve'], loose, e['elapsed'], delay_h=d_,
                                                  stock=W['stock'][sid], plan_part=e['plan_part']) if not o['선택지'].startswith('지금')]
                        if po_:
                            items.insert(1, (f'계획 정비 · {po_[0]["선택지"]} (비교만)', po_[0]['기대 총비용'], CAUTION_YELLOW, False))
                            plan_note = (f'계획 정비는 다음 정기 교체({e["wait_h"]:.0f}시간 뒤)가 최대 대기·부품·정비팀 조건 밖이라 추천 대상에서 뺐습니다. '
                                         if d_ is None else
                                         (f'다음 정기 교체({e["wait_h"]:.0f}시간 뒤)에는 부품을 맞출 수 없어' if has_plan else '정기 교체 계획이 없어')
                                         + f' 계획 정비는 {d_}시간 뒤 정지를 가정해 비교만 합니다. ')
                            if po_[0]['고장 확률'] > cfg.safety_max:
                                plan_note += f'그때까지 고장 확률 {po_[0]["고장 확률"]:.2f}로 안전 기준을 넘습니다. '
                            break
                items.append(('24시간 조치 안 함', p24 * costs.failure_loss, ALARM_RED, False))
                cost_bars(items, costs)
                start_h = now + max(0.0, r_['_start'])
                todo = ('재가동 뒤 새 예측 확인' if job_ == '작업 완료' else
                        f'{r_["다음 정기 교체"]} 두 부품 함께 교체 · 그때까지 추이 관찰' if b['선택지'].startswith('다음 정기') else
                        f'아래에서 배정 · {start_h:.0f}시간 교체 시작 예정'
                        + (f' (정비팀 차례 대기 {r_["_slot"]:.0f}시간)' if r_['_slot'] else ''))
                if e['urgent'] and job_ != '작업 완료':
                    st.markdown(':red-badge[:material/emergency: 즉시 조치: 지금 해도 안전 기준 초과]')
                st.info(f'**추천:** {b["선택지"]} | **다음 할 일:** {todo}')
                st.caption(f'추천 = 완료 전 고장 확률이 안전 기준({cfg.safety_max:.2f}) 이하인 선택지 중 기대 총비용이 가장 낮은 것. '
                           + ('어떤 선택지도 안전 기준을 못 지켜 가장 빨리 할 수 있는 조치를 권합니다. ' if e['urgent'] else '')
                           + plan_note
                           + (f'"24시간 조치 안 함"이 더 싸 보여도 그동안 고장 확률이 {p24:.2f}로 안전 기준을 넘어 고르지 않습니다. '
                              if p24 * costs.failure_loss < b['기대 총비용'] and p24 > cfg.safety_max else '')
                           + '최소 비용과 실제 권장 조치는 다를 수 있습니다. 안전·현장 절차 우선.')
                with st.expander('이 설비 비용 가정', icon=':material/tune:'):
                    st.markdown(f'- 지금 두 부품 교체 비용: **{fmt_cost(e["opts"][0]["정비 비용"], costs)}** (교체 {costs.replace_h:.0f}시간)\n'
                                f'- 계획 밖 정지 1시간 생산 차질: **{fmt_cost(costs.downtime_h, costs)}**\n'
                                f'- 고장 시 손실: **{fmt_cost(costs.failure_loss, costs)}**\n'
                                f'- 안전 기준(완료 전 고장 확률): **{cfg.safety_max:.2f}** · 다음 정기 교체: **{r_["다음 정기 교체"]}**\n'
                                f'- 예비 재고: **{stock_text(W["stock"][sid])}** · 첫 경보 뒤 **{e["elapsed"]:.0f}시간** 지남')
                    st.caption(('가정값은 아래 ③ 하위 페이지 "정책·비용"에서 바꿉니다. 바꾸면 모든 설비에 같이 적용됩니다.' if SHOW_SUBPAGE_LIST
                                else '지금 가정값(기본값)으로 계산했습니다.'))
                with st.expander('비용 계산 근거', icon=':material/calculate:'):
                    st.markdown('  \n'.join(f'- **{o["선택지"]}**: 고장 확률 P {o["고장 확률"]:.2f} · {o["근거"]}' for o in e['opts'])
                                + f'  \n- **24시간 조치 안 함**: 앞으로 24시간 안 고장 확률 {p24:.2f} × 고장 시 손실'
                                + f'  \n- P는 학습 세트에서 경보 뒤 실제 고장까지 걸린 시간 분포(경보 뒤 {e["elapsed"]:.0f}시간 동안 고장 없음 조건).')
                st.caption('기대 총비용 = (1 − P) × (정비 비용 + 생산 차질) + P × 고장 시 손실 · P = 정비를 마치기 전에 고장 날 확률')
            elif sstage == '주의':
                pr = procurement(cfg, W['stock'], [sid]).iloc[0]
                st.info(f'**주의: 점검·부품 준비** · 열화가 시작됐지만 24시간 안 고장 위험은 낮습니다. 장비를 점검하고, 설비가 돌면 다음 정비 때 함께 '
                        f'정비합니다. 부품: {pr["발주"]} (재고 {pr["재고"]})', icon=':material/search_check:')
            elif sstage in ('정비 중', '정비 후 예측 대기'):
                st.info(f'{sstage}: 정비 전 예측은 쓰지 않습니다. 재가동 뒤 다음 예측을 기다립니다.', icon=':material/build:')
            else:
                st.success('정상 운전 · 추이 관찰. 정비를 추천하지 않습니다(지금 고장 위험이 낮아 정비 비용이 더 큼).',
                           icon=':material/check_circle:')
            assign_box(sid, srow, sstage, job_, dec_i, W, cfg, load)

        # ② 진행 현황
        st.markdown('#### ② 진행 현황')
        if tf.empty:
            st.caption('아직 작업이 없습니다.')
        else:
            f3 = st.segmented_control('작업 구분', ['전체', '지금', '지연', '예정', '완료'], default='전체', key='task_filter',
                                      label_visibility='collapsed')
            shown = tf if f3 in (None, '전체') else tf[tf['구분'] == f3]
            st.dataframe(shown.assign(confirmed=shown['confirmed'].map(lambda x: PART_NAMES.get(x, x)))[['id', 'sid', 'title', 'team', 'priority', '상태', '구분', 'due_h', 'outcome', 'confirmed']].rename(columns={
                'id': '번호', 'sid': '설비', 'title': '작업', 'team': '팀', 'priority': '우선순위', 'due_h': '예정 (h)', 'outcome': '완료 결과',
                'confirmed': '확인된 원인'}), hide_index=True,
                column_config={'예정 (h)': st.column_config.NumberColumn(format='%d')})
            with st.container(horizontal=True):
                with st.popover('작업 수정', icon=':material/edit:'):
                    # 완료된 작업도 고칠 수 있다 (완료 보고: 결과·부품·수행 내용). 고친 내용은 점검 결과로 다시 기록해 되먹임에 반영
                    tfi = tf.set_index('id')
                    tid = st.selectbox('작업', tf['id'].tolist(), key='edit_tid',
                                       format_func=lambda i: f'#{i} · {tfi.loc[i, "title"]} · {tfi.loc[i, "상태"]}')
                    cur = tfi.loc[tid]
                    done_ = cur['상태'] == '완료'
                    # 상태는 폼 밖에서 고른다: '완료'를 고르면 아래에 완료 보고 칸이 바로 나타나게
                    e_state = st.segmented_control('작업 상태', tasks.STATUSES, default=cur['상태'], required=True,
                                                   key=f'edit_state_{tid}',
                                                   help='대기·진행 중으로 되돌리면 완료 보고가 지워집니다. 완료로 바꾸려면 완료 보고가 필요합니다.')
                    with st.form(f'edit_task_{tid}', border=False):
                        e_team = st.selectbox('담당 팀', list(tasks.TEAMS), index=list(tasks.TEAMS).index(cur['team']))
                        e_prio = st.selectbox('우선순위', tasks.PRIORITIES, index=tasks.PRIORITIES.index(cur['priority']))
                        e_due = st.number_input('시작 예정 (h)', 0.0, 240.0, min(240.0, float(cur['due_h'])), 1.0, format='%.0f')
                        e_guide = st.text_area('점검 가이드', value=cur['guide'], max_chars=tasks.LIMITS['guide'])
                        e_reason = st.text_area('추천 이유', value=cur['reason'], max_chars=tasks.LIMITS['reason'])
                        if e_state == '완료':
                            st.markdown('**완료 보고**' + (f' (완료 {cur["completed_h"]:.0f}시간)' if done_ else f' (지금 {now}시간에 완료로 기록)'))
                            pick = lambda opts, v, d: opts.index(v) if v in opts else opts.index(d)
                            e_out = st.selectbox('작업 결과', tasks.OUTCOMES, index=pick(tasks.OUTCOMES, cur['outcome'], '이상 발견'))
                            e_jud = st.selectbox('점검으로 판단한 부품', tasks.JUDGED, index=pick(tasks.JUDGED, cur['judged'], '구분 못함'),
                                                 format_func=lambda x: PART_NAMES.get(x, x))
                            e_con = st.selectbox('교체 뒤 확인된 원인 부품', tasks.CONFIRMED,
                                                 index=pick(tasks.CONFIRMED, cur['confirmed'], '아직 모름'), format_func=lambda x: PART_NAMES.get(x, x))
                            e_det = st.text_area('수행 내용', value=cur['details'] if done_ else '', max_chars=tasks.LIMITS['details'])
                        if st.form_submit_button('수정 저장', type='primary'):
                            try:
                                TASKS.edit(tid, e_team, e_prio, e_due, e_guide, e_reason)
                                if e_state != cur['상태']:
                                    # 상태 변경: 작업 기록을 바꾸고, 진행 단계 기록(src/actions.py)에도 남겨 화면 전체가 같은 상태를 본다
                                    if e_state == '완료':
                                        TASKS.set_status(tid, '완료', now, e_out, e_det, e_jud, e_con)
                                        LOG.record('작업 완료', split, cur['sid'], now, MANAGER, note=f'상태 변경 #{tid}: {e_det}')
                                        LOG.record('점검 결과', split, cur['sid'], now, MANAGER, judged=e_jud, confirmed=e_con, note=e_out)
                                    else:
                                        TASKS.set_status(tid, e_state, now)
                                        if done_:   # 되돌린 완료의 점검 결과는 같은 시각에 '아직 모름'으로 덮는다 (되먹임은 마지막 기록만 센다)
                                            LOG.record('점검 결과', split, cur['sid'], float(cur['completed_h']), MANAGER, judged='구분 못함',
                                                       confirmed='아직 모름', note=f'완료 되돌림 #{tid}')
                                        LOG.record('작업 시작' if e_state == '진행 중' else '승인', split, cur['sid'], now, MANAGER,
                                                   note=f'상태 변경 #{tid}: {cur["상태"]} → {e_state}')
                                elif done_:
                                    TASKS.edit_report(tid, e_out, e_det, e_jud, e_con)
                                    if (e_out, e_jud, e_con) != (cur['outcome'], cur['judged'], cur['confirmed']):
                                        LOG.record('점검 결과', split, cur['sid'], float(cur['completed_h']), MANAGER, judged=e_jud,
                                                   confirmed=e_con, note=f'완료 보고 수정 #{tid}: {e_out}')
                                st.toast(f'#{tid} 수정했습니다' + (f' ({cur["상태"]} → {e_state})' if e_state != cur['상태'] else ''),
                                         icon=':material/check_circle:')
                                st.rerun()
                            except ValueError as err:
                                st.error(str(err))
                    # 배정 취소: 실수로 배정했거나 계획이 바뀐 경우. 그 설비는 다시 '배정할 경보'로 돌아간다
                    st.divider()
                    if done_:
                        st.caption('완료된 작업은 배정을 취소할 수 없습니다. 위에서 상태를 대기·진행 중으로 되돌린 뒤 취소하세요.')
                    else:
                        sure = st.checkbox(f'#{tid} 배정을 취소합니다 (작업 목록에서 빠지고, 설비는 다시 배정 대기로)', key=f'cancel_ok_{tid}')
                        st.button('배정 취소', icon=':material/undo:', key=f'cancel_{tid}', disabled=not sure, on_click=cancel_task,
                                  args=(tid, cur['sid'], cur['title']))
                st.download_button('작업 이력 CSV', tasks.safe_csv(tf.drop(columns=['split'])), file_name=f'cast_pdm_tasks_{split}.csv',
                                   mime='text/csv', icon=':material/download:')

        st.session_state.setdefault('sid', sid)
    if sub not in MANAGER_TABS and SHOW_SUBPAGE_LIST:
        st.markdown('#### ③ 하위 페이지')
        with st.container(border=True, gap='small'):
            for key_, label_ in [('설비 상세', f'{sid} 설비 상세 — 위험 확률·네 신호·정비 이력'),
                                 ('정책·비용', '정책·비용 — 가정값, 전체 정비 결정과 근거'),
                                 ('부품 손익', '부품 손익 — 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나'),
                                 ('모델 정보·한계', '모델 정보·한계 — 검증 성능, 한계와 대응, 온톨로지')]:
                st.button(label_, icon=':material/description:', type='tertiary', key=f'sub_{key_}', on_click=open_sub,
                          args=(key_, sid if key_ == '설비 상세' else None))
        st.caption('누르면 그 페이지로 들어갑니다. 위의 "← 정비 의사결정"으로 돌아옵니다.')


# 2. 설비 상세 (작업자 '설비 상세' 보기, 설비관리자 아래 탭)

if page == '설비 상세':
    t = status_table()
    if st.session_state.get('sid') not in t.index:
        st.session_state.pop('sid', None)
    sid = st.selectbox('설비 (지금 위험한 순서)', t.index.tolist(), key='sid',
                       format_func=lambda s: f'{s} · {t.loc[s, "상태"]} · 확률 {t.loc[s, "위험 확률"]:.2f}'
                       if pd.notna(t.loc[s, '위험 확률']) else f'{s} · {t.loc[s, "상태"]}')
    row = t.loc[sid]
    st.header(f'{sid}', icon=ICONS['설비 상세'])
    st.caption(f'{split_label} · 지금 {now}시간 기준')

    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('상태', row['상태'].split(' (')[0], help=row['상태'])   # 칸이 좁아 영문 이름은 도움말로
    c2.metric('24시간 안 고장 확률', '—' if pd.isna(row['위험 확률']) else f'{row["위험 확률"]:.2f}',
              help=f'마지막 예측: {row["마지막 예측 (h)"]:.0f}시간' if pd.notna(row['마지막 예측 (h)']) else None)
    if CAUTION:
        c3.metric('지금 열화 중일 확률', '—' if pd.isna(row['열화 확률']) else f'{row["열화 확률"]:.2f}')
        c4.metric('첫 주의 / 첫 경보', ' / '.join('—' if pd.isna(row[k]) else f'{row[k]:.0f}h' for k in ['첫 주의 (h)', '첫 경보 (h)']))
    else:
        c3.metric('첫 경보', '—' if pd.isna(row['첫 경보 (h)']) else f'{row["첫 경보 (h)"]:.0f}시간')

    if row['상태'] == WARN:
        st.error(
            '24시간 안에 고장 날 위험이 높습니다.\n'
            '- 교체에 4시간이 걸리므로, 늦어도 고장 예상 4시간 전에는 교체를 시작해야 합니다.\n'
            '- 센서로는 원인 부품(진동장치·밸브)을 구분할 수 없습니다. 두 부품을 모두 점검하세요.\n'
            '- 고장 원인과 상관없이 같은 신호가 오르므로, 아래 신호 흐름으로 열화가 계속되는지 확인하세요.',
            title='조치 안내: 교체', icon=':material/build:')
        if row['경보 유지']:
            st.caption('지금 확률은 경보 기준보다 낮지만, 이번 운전 구간에서 이미 경보가 켜져 정비 전까지 경보로 유지합니다.')
    elif row['상태'] == CAU:
        st.warning(
            '열화가 시작된 것으로 보입니다. 아직 24시간 안 고장 위험은 낮습니다.\n'
            '- 장비 상태를 확인하세요. 설비가 정상적으로 돌아가면 **다음 정비 때 이 부분까지 함께 정비**합니다.\n'
            '- 두 부품(진동장치·밸브)의 재고를 확인하고, 없으면 지금 발주를 시작하세요. 경보까지 보통 반나절 정도 남습니다.\n'
            f'- 주의 중 일부는 열화가 아닌 헛걸음입니다(검증 세트 {CAUTION["validation"]["주의 묶음"]}번 중 '
            f'{CAUTION["validation"]["열화 아닌 설비의 주의"]}번). 점검에서 이상이 없으면 그대로 운전합니다.',
            title='조치 안내: 점검', icon=':material/search_check:')
    elif row['상태'] in ('정비 중', '정비 후 예측 대기'):
        st.warning(f'정비가 시작돼 정비 전 예측은 더 이상 쓰지 않습니다. 재가동 뒤 다음 예측 시점({STEP}시간마다)에 다시 예측합니다.')
    else:
        st.success('지금은 열화 신호가 없고, 24시간 안 고장 위험이 낮습니다.' if CAUTION else '지금은 24시간 안 고장 위험이 낮습니다.')
    last_p = after_last_maintenance(pts_now[pts_now['SCENARIO_ID'] == sid], 'T_H').tail(1)
    if 'OOR_N' in last_p and len(last_p) and last_p['OOR_N'].iloc[0] > 0:
        st.warning(f'마지막 예측({last_p["T_H"].iloc[0]:.0f}시간)의 특징 {int(last_p["OOR_N"].iloc[0])}개가 '
                   f'학습 때 본 범위(0.1~99.9%) 밖입니다: {last_p["OOR_FEATS"].iloc[0]}. '
                   '아주 심한 열화이거나 센서 이상일 수 있어, 모델 확률만 믿지 말고 현장에서 센서와 설비를 확인하세요.',
                   title='입력 감시: 센서 확인', icon=':material/sensors:')

    # 위험 확률 흐름
    p = pts_now[pts_now['SCENARIO_ID'] == sid].rename(columns={'STAGE': '단계'})
    p['단계'] = p['단계'].replace(LABEL)
    x = alt.X('T_H:Q', title='시나리오 시작 후 경과 시간 (h)', scale=alt.Scale(domain=[0, 72], nice=False))
    tip = [alt.Tooltip('T_H', title='시각(h)'), alt.Tooltip('PROB', title='24시간 안 고장 확률', format='.2f'), '단계']
    if CAUTION:
        tip.insert(2, alt.Tooltip('DEGRADED_PROB', title='열화 확률', format='.2f'))
    line = alt.Chart(p).mark_line(color=LINE_BLUE).encode(x=x, y=alt.Y('PROB:Q', title='확률', scale=alt.Scale(domain=[0, 1])))
    dots = alt.Chart(p).mark_circle(size=90).encode(x=x, y='PROB:Q', color=alt.Color('단계:N', scale=STAGE_COLORS), tooltip=tip)
    layers = [*mnt_layer(sid, 1), line, dots,
              alt.Chart(pd.DataFrame({'y': [TH]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
    if CAUTION:
        layers += [alt.Chart(p).mark_line(color=CAUTION_YELLOW, strokeDash=[2, 2]).encode(x=x, y='DEGRADED_PROB:Q', tooltip=tip),
                   alt.Chart(pd.DataFrame({'y': [CTH]})).mark_rule(color=CAUTION_YELLOW, strokeDash=[6, 3]).encode(y='y:Q')]
    st.subheader('위험 확률과 열화 확률' if CAUTION else '위험 확률', icon=':material/trending_up:')
    st.altair_chart(alt.layer(*layers).properties(height=280), width='stretch')
    st.caption(('파란 선과 점: 24시간 안 고장 확률(점 색 = 단계) · 노란 점선: 지금 열화 중일 확률 · '
                f'빨간 가로 점선: 경보 기준 {TH:.2f} · 노란 가로 점선: 주의 기준 {CTH:.2f}' if CAUTION
                else f'점선: 경보 기준 {TH:.2f}')
               + ' · 회색 띠: 정비 · 빨간 실선: 고장'
               + (' · 빨간 점선: 앞으로 일어날 실제 고장 (정답)' if show_truth else ''))

    # 네 신호 흐름
    h = hr_now[hr_now['SCENARIO_ID'] == sid].rename(columns=SIGNAL_NAMES)
    long = h.melt(id_vars=['DONE_H'], value_vars=list(SIGNAL_NAMES.values()), var_name='신호', value_name='점수')
    sig = alt.Chart(long).mark_line(opacity=0.8).encode(
        x=alt.X('DONE_H:Q', title='시나리오 시작 후 경과 시간 (h)', scale=alt.Scale(domain=[0, 72], nice=False)),
        y=alt.Y('점수:Q', title='표준화 점수 (1 = 정상 경계)'), color='신호:N',
        tooltip=['신호', alt.Tooltip('DONE_H', title='시각(h)'), alt.Tooltip('점수', format='.2f')])
    avg = alt.Chart(h).mark_line(color=INK, strokeWidth=2.5).encode(
        x='DONE_H:Q', y='SCORE:Q', tooltip=[alt.Tooltip('SCORE', title='네 신호 평균', format='.2f')])
    one = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(color=INK, strokeDash=[2, 2]).encode(y='y:Q')
    st.subheader('네 신호의 흐름', icon=':material/ssid_chart:')
    st.altair_chart(alt.layer(*mnt_layer(sid, None), sig, avg, one).properties(height=300), width='stretch')
    st.caption('회색 굵은 선: 네 신호 평균 점수 · 점선: 정상 범위 경계(1). 정비 후 재가동하면 처음 2시간은 기준을 잡는 시간이라 비어 있습니다.')

    # 지금 신호 상태와 모델 기여도
    left, right = st.columns(2)
    with left:
        st.subheader('지금 신호 상태', icon=':material/sensors:')
        cur = after_last_maintenance(h.assign(HOUR_START=h['DONE_H'] - 1), 'HOUR_START')
        last = cur.dropna(subset=list(SIGNAL_NAMES.values())).tail(1)
        if row['상태'] == '정비 중':
            st.write('정비 중이라 신호가 없습니다.')
        elif last.empty:
            st.write('재가동 뒤 아직 계산된 신호가 없습니다. 재가동 후 처음 2시간은 기준을 잡는 시간입니다. '
                     '(정비 전 신호는 지금 상태가 아니라서 보여 주지 않습니다.)')
        else:
            bar = last.melt(value_vars=list(SIGNAL_NAMES.values()), var_name='신호', value_name='점수')
            bar['정상 범위'] = np.where(bar['점수'] > 1, '벗어남', '안')
            st.altair_chart(
                alt.layer(
                    alt.Chart(bar).mark_bar().encode(
                        y=alt.Y('신호:N', title=None, sort='-x'), x=alt.X('점수:Q', title='표준화 점수'),
                        color=alt.Color('정상 범위:N', scale=alt.Scale(domain=['벗어남', '안'], range=[ALARM_RED, OK_GRAY]))),
                    alt.Chart(pd.DataFrame({'x': [1]})).mark_rule(strokeDash=[2, 2]).encode(x='x:Q'),
                ).properties(height=180), width='stretch')
            st.caption(f'{last["DONE_H"].iloc[0]:.0f}시간까지의 1시간 값. 1을 넘으면 정상 시나리오의 95%보다 많이 벗어난 것입니다.')
    with right:
        with st.expander('자세히: 모델이 본 이유 (기여도)'):
            cp = after_last_maintenance(pts_now[pts_now['SCENARIO_ID'] == sid], 'T_H')
            if contrib is None:
                st.write(f'최종 모델({bundle["final"]["model"]})은 특징별 기여도를 계산할 수 없는 종류입니다. '
                         '왼쪽의 신호 상태와 위의 신호 흐름으로 경보 이유를 확인하세요.')
            elif cp.empty:
                st.write(f'아직 예측이 없습니다. 정비가 시작되면 정비 전 예측은 쓰지 않고, 재가동 뒤 다음 예측 시점({STEP}시간마다)을 기다립니다.')
            else:
                cr = contrib.loc[cp.index[-1]].rename('기여도').reset_index().rename(columns={'index': '묶음'})
                st.altair_chart(alt.Chart(cr).mark_bar().encode(
                    y=alt.Y('묶음:N', title=None, sort='-x'), x=alt.X('기여도:Q', title=f'위험을 올린 정도 ({contrib_unit})'),
                    color=alt.condition('datum.기여도 > 0', alt.value(ALARM_RED), alt.value(LINE_BLUE))
                ).properties(height=220), width='stretch')
                st.caption(f'{cp["T_H"].iloc[-1]:.0f}시간 예측 기준. 양수는 위험을 올린 쪽, 음수는 내린 쪽입니다. '
                           + ('나무마다 결정 경로를 따라 확률이 바뀐 양을 그 분기에 쓰인 특징 묶음에 나눠 준 값으로, '
                              '기준값에 모두 더하면 위의 위험 확률이 됩니다. ' if contrib_unit.startswith('확률')
                              else '0은 학습 세트 평균 시점입니다. ')
                           + '네 신호와 결합 점수는 서로 비슷한 정보라 모델 안에서 서로 상쇄될 수 있습니다. '
                           '묶음 하나의 크기보다 왼쪽의 신호 상태를 먼저 보세요.')

    with st.expander('근거 따라가기: 경보 → 예측 → 신호 → 정비 결정 (온톨로지)', icon=':material/account_tree:'):
        tr = ontology.trace(ontology_graph(assumptions(), t), sid, with_cost=(role == '정비 책임자'))
        st.dataframe(pd.DataFrame(tr, columns=['단계', '내용']), hide_index=True, width='stretch',
                     column_config={'내용': st.column_config.TextColumn(width='large')})
        st.caption('설비·예측·신호·경보·정비 결정·정기 교체 계획을 하나의 그래프(OWL/RDF)로 잇고, 그 링크를 따라 읽은 결과입니다. '
                   '판단은 그래프가 아니라 경보 기준과 의사결정 규칙이 합니다. 센서로는 원인 부품을 구분할 수 없어 '
                   '"신호 → 부품" 링크는 두지 않았습니다.')



# 3. 의사결정 (DSS)

elif page == '의사결정':
    D = bundle.get('dss')
    st.subheader('정책·비용: 가정값, 전체 정비 결정과 근거', icon=ICONS['정책·비용'])
    if not D:
        st.warning('이 모델 파일에는 의사결정 정보가 없습니다. 터미널에서 `python analysis/05_predict_test.py`로 다시 만드세요.')
        st.stop()
    A = assumptions()
    st.caption('비용 단위: 만 원 (현장 가정값, 아래 "가정값"에서 바꿀 수 있음). 과제 공통 비용 규칙(점수)으로 낸 모델 평가는 결과보고서에 있습니다.')
    FIELD = A['mode'] == COST_MODES[1]
    U = '만 원' if FIELD else '점'

    def won(x):
        return f'{x:,.0f}{U}' if not FIELD or abs(x) < 10000 else f'{x / 10000:,.2f}억 원'

    # ② 가정값 (여기서 바꾸면 위쪽 배정할 경보와 작업자 화면도 같은 계획으로 바뀐다)
    with st.expander('가정값 (바꾸면 추천·비용·배정할 경보가 다시 계산됩니다)', expanded=False, icon=':material/tune:'):
        assume_note = st.container()          # 지금 가정 요약 (아래에서 계산 뒤 채운다)
        st.markdown('**안전·실행 가능성 (비용보다 먼저 지키는 제약)**')
        a1, a2, a3, a4 = st.columns(4)
        A['safety'] = a1.slider('안전 기준: 정비 완료 전 고장 확률 상한', 0.0, 0.5, A['safety'], 0.01)
        A['prep'] = a2.slider('부품 준비 시간 (조달 + 교체, h)', 4, 36, A['prep'], 1,
                              help='데이터셋 비용 규칙은 교체 4시간. 부품을 조달해야 하면 더 길어집니다.')
        A['crew_wait'] = a3.slider('정비팀이 시작할 수 있을 때까지 (h)', 0, 24, A['crew_wait'], 1)
        A['n_crews'] = a4.number_input('동시에 교체할 수 있는 정비팀 수', 1, 10, A['n_crews'], 1,
                                       help='경보 설비가 이보다 많으면 위험한 순서대로 차례를 기다립니다.')
        A['crew_day_h'] = a4.number_input('정비팀 1팀의 하루 작업 한도 (공수, 시간)', 4, 24, A['crew_day_h'], 1,
                                          help='팀 결정(2026-10-15): 1팀당 하루 최대 8공수. 교체 1번이 4시간이라 팀당 하루 2대까지이고, 넘으면 다음 날로 넘어갑니다.')
        if FIELD:
            st.markdown('**비용 (만 원, 현장 가정)**')
            f1, f2, f3, f4 = st.columns(4)
            A['f_cons'] = f1.number_input('소모품 (교체 1번: 볼트·너트·O링·윤활유)', 0.0, 10000.0, A['f_cons'], 10.0)
            A['f_part'] = f2.number_input('중요 부품 (개당: 밸브·진동장치)', 0.0, 10000.0, A['f_part'], 10.0)
            A['f_labor_part'] = f3.number_input('공임 (파트당)', 0.0, 1000.0, A['f_labor_part'], 1.0)
            A['f_labor_ph'] = f4.number_input('정비 인력 (1명·1시간)', 0.0, 100.0, A['f_labor_ph'], 0.5)
            g1, g2, g3, g4 = st.columns(4)
            A['f_crew'] = g1.number_input('정비팀 인원 (명)', 1, 10, A['f_crew'], 1, help='현장: 보통 3~4인 1팀')
            A['f_down'] = g2.number_input('생산 차질 (정지 1시간)', 0.0, 100000.0, A['f_down'], 50.0)
            A['f_fail'] = g3.number_input('고장 1건 추가 손실 (불량·품질·안전 대응)', 0.0, 100000.0, A['f_fail'], 50.0)
        st.markdown('**부품 조달 (데이터에 없어 미리 정한 가정값)**')
        h1, h2, h3, h4, h5 = st.columns(5)
        A['stock_v'] = h1.number_input('예비 밸브 (설비당)', 0, 10, A['stock_v'], 1, help='정비 기록상 밸브 교체가 진동장치의 약 2배')
        A['stock_o'] = h2.number_input('예비 진동장치 (설비당)', 0, 10, A['stock_o'], 1)
        A['lead'] = h3.number_input('정상 조달 (h)', 1.0, 720.0, A['lead'], 12.0)
        A['rush_h'] = h4.number_input('긴급 조달 (h)', 1.0, 168.0, A['rush_h'], 1.0, help='웃돈을 내고 직접 수령')
        A['rush_mk'] = h5.number_input('긴급 조달 웃돈 (부품값 대비)', 0.0, 3.0, A['rush_mk'], 0.1)
        st.markdown('**비용 (점, 기본값 = 데이터셋 비용 규칙)**' if not FIELD else '**기타**')
        b1, b2, b3, b4 = st.columns(4)
        if not FIELD:
            A['down_h'] = b1.number_input('생산 차질 비용 (정지 1시간)', 0.0, 50.0, A['down_h'], 0.5)
            A['penalty'] = b2.number_input('고장 벌점 (안전·품질 영향)', 0.0, 200.0, A['penalty'], 5.0)
        A['p_other'] = b3.slider('함께 바꾸는 부품이 실제 원인일 확률', 0.0, 1.0, A['p_other'], 0.05,
                                 help='센서로 원인 부품을 구분할 수 없어 기본값은 0.5입니다.')
        A['max_wait'] = b4.slider('기다릴 수 있는 최대 시간 (h)', 0, 72, A['max_wait'], 1)
        st.markdown('**N시간 뒤 정비 선택지 (예: 생산 영향이 적은 시간대)**')
        d1, d2 = st.columns(2)
        A['delay_h'] = d1.slider('N시간 뒤 (0 = 쓰지 않음)', 0, 48, A['delay_h'], 1)
        if FIELD:
            A['f_delay_rate'] = d2.number_input('그 시간대 생산 차질 (정지 1시간, 만 원)', 0.0, 100000.0, A['f_delay_rate'], 50.0)
        else:
            A['delay_rate'] = d2.number_input('그 시간대 생산 차질 비용 (정지 1시간)', 0.0, 50.0, A['delay_rate'], 0.5)
        if st.button('기본값으로 되돌리기'):
            st.session_state.pop('assume', None)
            st.rerun()
    if st.session_state.get('assume') != dict(A):     # 가정값이 바뀌면 위쪽(배정할 경보·비용 막대)도 같은 값으로 다시 그린다
        st.session_state['assume'] = dict(A)
        st.rerun()
    costs, cfg = make_cfg(A)
    now_cost = 2 * costs.part + costs.unnecessary + costs.replace_fixed + costs.replace_h * costs.downtime_h
    assume_note.caption(f'지금 가정: 지금 두 부품 교체 **{won(now_cost)}** · 정기 교체 때 다른 부품 추가 **{won(costs.part)}** · '
               f'고장 시 손실 = 복구 {costs.repair:,.0f} + 복구 {costs.repair_h:.0f}시간 × (생산 차질 {costs.downtime_h:,.1f} + 인력 {costs.labor_h:,.1f}) '
               f'+ 고장 추가 손실 {costs.failure_penalty:,.0f} = **{won(costs.failure_loss)}** · 안전 기준 {cfg.safety_max:.2f} · '
               f'부품 준비 {cfg.prep_h:.0f}시간 · 정비팀 {A["n_crews"]}팀(팀당 하루 {A["crew_day_h"]}공수). 고장 확률은 학습 세트에서 경보 뒤 고장까지 걸린 시간 분포로 계산합니다.')

    t = status_table()
    W = work_plan(A, t)
    dec, need_crew, per_day, evaluate_options, cfg, costs, curve, pl = (W[k] for k in ['dec', 'need_crew', 'per_day', 'evaluate_options', 'cfg', 'costs', 'curve', 'pl'])
    n_crews, crew_day_h = A['n_crews'], A['crew_day_h']

    jobs_state = LOG.job_state(split, now, MNT_START)

    st.subheader(f'전체 정비 결정 · {len(dec)}건', icon=':material/event_note:')
    if dec.empty:
        st.success('지금 경보가 켜진 설비가 없습니다.', icon=':material/check_circle:')
    else:
        st.dataframe(dec.drop(columns=['_e', '_slot', '_start']).round({'기대 총비용': 0}), hide_index=True, column_config={
            '순위': st.column_config.NumberColumn(format='%d', help='지금 정비할 설비의 정비팀 배정 순서 (위험한 순서)'),
            '추천': st.column_config.TextColumn(width='medium'),
            '정비 완료까지 (h)': st.column_config.NumberColumn(format='%.0f'),
            '기대 총비용': st.column_config.NumberColumn(f'기대 총비용 ({U})', format='localized'),
            '완료 전 고장 확률': st.column_config.NumberColumn(format='%.2f'),
            '첫 경보 (h)': st.column_config.NumberColumn(format='%d')})
        st.caption(f'정비팀 {n_crews}팀이 한 번에 {n_crews}대씩 교체({costs.replace_h:.0f}시간)하고, 팀당 하루 {crew_day_h:.0f}공수(교체 {per_day}번)까지 합니다. '
                   f'하루에 {n_crews * per_day}대를 넘는 설비는 다음 날로 넘어갑니다. 차례를 기다리는 설비는 그만큼 늦게 끝나는 것으로 다시 계산합니다.')

        # ④ 설비별 근거: 선택지 비교와 산정 근거
        st.subheader('설비별 근거', icon=':material/rule:')
        sid = st.selectbox('설비를 골라 근거 보기', dec['설비 (시나리오)'].tolist())
        e = dec.set_index('설비 (시나리오)').loc[sid, '_e']
        slot = dec.set_index('설비 (시나리오)').loc[sid, '_slot']
        override = st.checkbox('고장 확률을 직접 입력 (현장 판단으로 덮어쓰기)', help='점검 결과 등 현장 판단이 있을 때 모델 확률 대신 씁니다.')
        if override:
            cols_o = st.columns(len(e['opts']))
            keys = {'지금 두 부품 교체': '지금 두 부품 교체', '다음 정기 교체 때 두 부품 함께': '정기 교체 때'}
            po = {}
            for col, o in zip(cols_o, e['opts']):
                key = keys.get(o['선택지'], 'N시간 뒤 교체')
                po[key] = col.number_input(f'{o["선택지"]}: 완료 전 고장 확률', 0.0, 1.0, float(round(o['고장 확률'], 2)), 0.01, key=f'po_{sid}_{key}')
            e = evaluate_options(sid, crew_extra=slot, p_override=po)
        b = e['best']
        msg = f'**{sid} · 추천: {b["선택지"]}**' + (f' ({e["urgent"]})' if e['urgent'] else '')
        (st.info if b['선택지'].startswith('다음 정기') else st.warning)(
            msg + f'\n\n기대 총비용 {won(b["기대 총비용"])} = (1 − 고장 확률 {b["고장 확률"]:.2f}) × (정비 {b["정비 비용"]:,.0f} + 생산 차질 {b["생산 차질 비용"]:,.0f})'
            f' + {b["고장 확률"]:.2f} × 고장 시 손실 {b["고장 시 손실"]:,.0f} ({U})')
        st.caption(f'이 설비 진행: {jobs_state.get(sid, "작업 없음")} · 배정·보류·부품 발주는 위 ① 비용 비교 아래에서 합니다.')
        left, right = st.columns([1.3, 1])
        with left:
            ot = pd.DataFrame(e['opts'])
            ot['추천'] = np.where(ot['선택지'] == b['선택지'], '✔', '')
            st.dataframe(ot[['추천', '선택지', '정비 완료까지(h)', '정비 비용', '생산 차질 비용', '고장 확률', '고장 시 손실', '기대 총비용', '안전 기준 충족']]
                         .round({'정비 비용': 0, '생산 차질 비용': 0, '고장 시 손실': 0, '기대 총비용': 0}),
                         hide_index=True, column_config={
                             '정비 완료까지(h)': st.column_config.NumberColumn(format='%.0f'),
                             '정비 비용': st.column_config.NumberColumn(format='localized'),
                             '생산 차질 비용': st.column_config.NumberColumn(format='localized'),
                             '고장 확률': st.column_config.NumberColumn('완료 전 고장 확률', format='%.2f'),
                             '고장 시 손실': st.column_config.NumberColumn(format='localized'),
                             '기대 총비용': st.column_config.NumberColumn(format='localized')})
            st.markdown('**산정 근거**\n' + '\n'.join(f'- {o["선택지"]}: {o["근거"]}' for o in e['opts'])
                        + f'\n- 고장 확률: 학습 세트에서 경보가 처음 켜진 뒤 실제 고장까지 걸린 시간 분포(카플란-마이어). '
                          f'이 설비는 경보 뒤 {e["elapsed"]:.0f}시간 동안 고장이 없었다는 조건으로 계산'
                        + ('\n- 고장 확률은 현장 판단으로 덮어쓴 값' if override else ''))
            wait = next((o for o in e['opts'] if o['선택지'].startswith('다음 정기')), None)
            if wait is not None:
                c_wait = wait['정비 비용'] + wait['생산 차질 비용']
                even = (e['opts'][0]['기대 총비용'] - c_wait) / (costs.failure_loss - c_wait)
                st.caption((f'추천이 바뀌는 경계: 안전 기준을 {wait["고장 확률"]:.2f}보다 낮추면 "지금 교체". ' if wait['고장 확률'] >= 0.005 else
                            '정기 교체 전까지 고장 확률이 거의 0이라 안전 기준과 상관없이 기다릴 수 있습니다. ')
                           + f'비용만 보면 정기 교체 전 고장 확률이 {even:.2f}를 넘을 때 지금 교체가 더 쌉니다. '
                           '부품 준비 시간이 정기 교체보다 길어지면 "지금 교체"로 바뀝니다.')
            elif pd.isna(e['wait_h']):
                st.caption('다음 정기 교체 계획이 없어 지금(또는 N시간 뒤) 교체만 가능합니다.')
        with right:
            cv = pd.DataFrame({'h': np.arange(0, 49)})
            cv['p'] = [fail_prob(curve, e['elapsed'], h) for h in cv['h']]
            marks = [alt.Chart(cv).mark_line(color=LINE_BLUE).encode(
                x=alt.X('h:Q', title='지금부터 시간 (h)'), y=alt.Y('p:Q', title='그 안에 고장 날 확률', scale=alt.Scale(domain=[0, 1])),
                tooltip=[alt.Tooltip('h', title='시간(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]),
                alt.Chart(pd.DataFrame({'y': [cfg.safety_max]})).mark_rule(color=ALARM_RED, strokeDash=[6, 3]).encode(y='y:Q')]
            pts_o = pd.DataFrame([{'h': o['정비 완료까지(h)'], 'p': fail_prob(curve, e['elapsed'], o['정비 완료까지(h)']), '선택지': o['선택지']}
                                  for o in e['opts']])
            marks.append(alt.Chart(pts_o).mark_point(size=90, filled=True, color=LINE_GREEN).encode(
                x='h:Q', y='p:Q', tooltip=['선택지', alt.Tooltip('h', title='정비 완료(h)'), alt.Tooltip('p', title='고장 확률', format='.2f')]))
            st.altair_chart(alt.layer(*marks).properties(height=240), width='stretch')
            st.caption('파란 선: 지금부터 h시간 안에 고장 날 확률 · 초록 점: 선택지별 정비 완료 시점 · 빨간 점선: 안전 기준')

    # ⑤ 조치 기록과 점검 되먹임
    with st.expander('조치 기록과 점검 되먹임', icon=':material/history:'):
        fb = feedback()
        f1, f2, f3 = st.columns(3, border=True)
        f1.metric('점검 결과 (원인 확인됨)', f'{fb["n"]}건', help=f'되먹임은 {actions.MIN_CONFIRMED}건 이상부터 씁니다 (가정값)')
        f2.metric('점검으로 원인 부품을 맞힌 비율', '—' if fb['accuracy'] is None else f'{fb["accuracy"]:.2f}')
        f3.metric('교체 방법 추천', '점검 후 한 부품' if fb['ready'] else '두 부품',
                  help=f'맞힌 비율이 손익분기 {fb["breakeven"]:.2f} 이상이고 기록이 {actions.MIN_CONFIRMED}건 이상이면 "점검 후 한 부품"' if fb['breakeven'] else None)
        st.caption('센서로는 원인 부품을 구분할 수 없어(EDA E9) 지금은 두 부품을 함께 교체합니다. 현장이 점검 결과와 교체 뒤 확인된 원인을 입력하면, '
                   '점검으로 부품을 맞히는 비율이 쌓이고 이 값이 비용 계산기의 손익분기를 넘으면 추천이 바뀝니다.')
        lg = LOG.frame(split, now)
        if lg.empty:
            st.write('이 데이터 세트에서 지금 시각까지 기록된 조치가 없습니다.')
        else:
            st.dataframe(lg.drop(columns=['데이터 세트']).iloc[::-1], hide_index=True,
                         column_config={'시각(h)': st.column_config.NumberColumn(format='%d'),
                                        '부품': st.column_config.TextColumn(), '메모': st.column_config.TextColumn(width='medium')})

    # ⑥ 주의 설비
    cau = t.index[t['상태'] == CAU].tolist()
    if cau:
        st.subheader('주의 설비 준비', icon=':material/search_check:')
        nxt = pl[pl['SCENARIO_ID'].isin(cau) & (pl['PLAN_H'] > now)].sort_values('PLAN_H').groupby('SCENARIO_ID').head(1)
        cd = pd.DataFrame({'설비 (시나리오)': cau})
        cd['다음 정기 교체'] = cd['설비 (시나리오)'].map(
            nxt.set_index('SCENARIO_ID').apply(lambda x: f'{x["PLAN_H"] - now:.0f}시간 뒤 ({PART_NAMES.get(x["PLAN_PART"], x["PLAN_PART"])})', axis=1)
            if len(nxt) else {}).fillna('계획 없음')
        cd['할 일'] = np.where(cd['다음 정기 교체'] != '계획 없음', '장비 점검, 정기 교체 때 두 부품 함께 교체 준비',
                              '장비 점검 (경보가 켜지면 지금 교체 대상)')
        pr = procurement(cfg, W['stock'], cau).set_index('설비')
        cd['재고'] = cd['설비 (시나리오)'].map(pr['재고'])
        cd['부품'] = cd['설비 (시나리오)'].map(pr['발주'])
        need_order = [(sid, q) for sid, qs in pr['_short'].items() for q in qs]
        if need_order:
            with st.container(border=True, horizontal=True, vertical_alignment='center'):
                st.markdown(f'**주의 설비 부품 발주 {len(need_order)}건** — 경보 전에 정상 조달({cfg.parts.lead_h:.0f}시간)로 재고를 채웁니다')
                st.space('stretch')

                def order_all():
                    for sid_, q_ in need_order:
                        LOG.record('부품 발주', split, sid_, now, '정비 책임자', part=q_, option='정상', note='주의 단계 재고 부족')
                    st.toast(f'부품 {len(need_order)}건 발주', icon=':material/local_shipping:')
                st.button('한 번에 발주', icon=':material/local_shipping:', type='primary', on_click=order_all, key='order_all')
        st.dataframe(cd, hide_index=True)
        st.caption(f'발주 시점: 주의 단계에서 두 부품 교체에 모자라는 재고가 있을 때(정상 조달 {cfg.parts.lead_h:.0f}시간). '
                   f'경보 때 재고가 없으면 긴급 조달({cfg.parts.rush_h:.0f}시간, 웃돈 {cfg.parts.rush_markup:.0%})로 교체가 늦어집니다.')

    # ⑦ 정책 시뮬레이션 상세 (요약 카드 '없었다면 → 있으면'은 2026-10-15 사용자 요청으로 뺐다)
    sim = D.get('validation_field') if FIELD else D.get('validation')
    if sim:
        with st.expander('정책 시뮬레이션 상세 (검증 세트, 기본 가정값)'):
            sv = pd.DataFrame(list(sim.values())).set_index('정책')
            sv['예상 절감액 (① 대비)'] = sim['regular']['비용'] - sv['비용']
            st.dataframe(sv[['비용', '예상 절감액 (① 대비)', '고장', '고장 손실', '정비 횟수', '지금 교체', '정기 교체 때',
                             '기다리다 고장', '고장 전 조치 완료율', '계획 밖 정지(h)', '미확정 경보']],
                         column_config={'비용': st.column_config.NumberColumn(f'기대 총비용 ({U})', format='localized'),
                                        '예상 절감액 (① 대비)': st.column_config.NumberColumn(format='localized'),
                                        '고장 손실': st.column_config.NumberColumn(format='localized'),
                                        '고장 전 조치 완료율': st.column_config.NumberColumn(format='%.2f')})
            st.caption('같은 데이터·같은 모델 예측으로 세 정책을 돌려 실제 고장 시각으로 채점했습니다. '
                       '준비 시간·안전 기준을 바꾼 민감도는 analysis/10_dss_policy.py에 있습니다.')

    with st.expander('이 화면 보는 법', icon=':material/help:'):
        st.markdown('- 경보 설비마다 정비 선택지(지금 / N시간 뒤 / 다음 정기 교체 때)의 **기대 총비용 = 정비 비용 + 생산 차질 비용 + '
                    '고장 확률 × 고장 시 손실**을 비교해 추천합니다.\n'
                    '- 안전 기준과 실행 가능성(정비팀·부품 준비)은 비용보다 먼저 지킵니다.\n'
                    '- 판단은 검증 가능한 예측값과 계산·규칙으로만 하며, 최종 결정은 정비 책임자가 합니다.')

# 4. 모델 정보

elif page == '모델 정보':
    v = bundle['validation']
    st.subheader('모델 정보와 검증 세트 성능', icon=ICONS['모델 정보·한계'])
    st.caption(f'모델 저장: {bundle["created_at"]}' + (' · 클라우드: 미리 계산한 결과만 사용 (원자료·모델 파일 없음)' if CLOUD else ''))
    st.markdown(
        f'- **모델**: {bundle["final"]["model"]} · 특징 {len(bundle["features"])}개 ({bundle["final"]["features"]}) · {STEP}시간마다 예측\n'
        f'- **학습**: 학습 세트 216개 시나리오의 예측 시점 {bundle["n_train_points"]:,}개\n'
        f'- **경보 기준**: 24시간 안 고장 확률 ≥ {TH:.2f}. 학습 세트 교차검증에서 '
        '점진 고장을 모두 잡고 진짜 오경보가 0인 값 중 가장 낮은 값(오경보 없이 가장 일찍 경보)\n'
        '- **검증 세트**: 72개 시나리오, 학습에 쓰지 않고 평가에만 사용(모델·기준 선택은 학습 세트 교차검증. 수정 이력은 결과보고서 부록 B-7)')
    c1, c2, c3, c4 = st.columns(4, border=True)
    c1.metric('AP (순위 능력)', f'{v["point"]["AP"]:.3f}', help='기준 예제 0.806')
    c2.metric('정밀도 / 재현율', f'{v["point"]["정밀도"]:.2f} / {v["point"]["재현율 (점진 고장만)"]:.2f}',
              help=f'재현율은 점진 고장 기준. 신호가 없는 돌발 고장을 포함하면 {v["point"]["재현율"]:.2f}')
    c3.metric('고장 감지', f'{v["event"]["감지"]}/{v["event"]["고장"]}', help='돌발 고장 2건은 신호가 없어 못 잡음')
    c4.metric('리드타임 중앙값', f'{v["event"]["리드타임 중앙값(h)"]:.0f}시간')
    st.markdown(
        f'- 오경보 {v["point"]["FP"]}개 중 {v["point"]["FP 중 이른 경보"]}개는 고장 24시간보다 일찍 울린 **이른 경보**이고, '
        f'고장 예정이 없는데 울린 진짜 오경보는 {v["point"]["FP 중 진짜 오경보"]}개입니다.\n'
        f'- 놓친 시점 {v["point"]["FN"]}개 중 {v["point"]["FN 중 돌발 고장"]}개는 신호 없이 갑자기 고장 난 돌발 고장입니다.\n'
        '- 위 숫자는 정답을 알 수 있는 시점만으로 셌습니다. 관측 끝부분의 미확정 경보는 "부품 손익" 탭에 있습니다.')
    if CAUTION:
        cv = CAUTION['validation']
        st.subheader('2단계 경보: 주의 단계', icon=':material/notifications:')
        st.markdown(
            f'- **주의 모델**: {CAUTION["final"]["model"]} · {CAUTION["final"]["features"]} 특징 {len(CAUTION["features"])}개. '
            '"지금 열화 중인가"를 맞힙니다(정답: 진단 정답의 열화 구간, 학습 때만 사용)\n'
            f'- **주의 기준**: 지금 열화 중일 확률 ≥ {CTH:.2f}. 학습 세트 교차검증에서 주의 중 {CAUTION["min_precision"]:.0%} 이상이 '
            '실제 열화인 값 중 가장 낮은 값\n'
            '- **헛걸음을 조금 허용하는 이유**: 현장에서는 주의를 받으면 장비 상태를 확인하고, 설비가 돌면 다음 정비 때 함께 정비합니다. '
            '설비를 세우거나 바로 발주하지 않으므로 헛걸음 비용은 점검 한 번입니다.')
        d1, d2, d3, d4 = st.columns(4, border=True)
        d1.metric('주의 정밀도', f'{cv["주의 정밀도"]:.2f}', help=f'주의 중 실제로 열화 중이던 비율. 목표 {CAUTION["min_precision"]}')
        d2.metric('주의가 켜진 점진 고장', f'{cv["주의로 감지"]}/{cv["점진 고장"]}')
        d3.metric('경보보다 먼저 (평균)', f'{cv["추가 준비 시간 평균(h)"]:.1f}시간')
        d4.metric('첫 주의 (중앙값)', f'고장 {cv["주의 리드타임 중앙값(h)"]:.0f}시간 전')
        st.caption(f'검증 세트 주의 {cv["주의 묶음"]}번 중 헛걸음 {cv["열화 아닌 설비의 주의"]}번. 대부분은 열화가 시작되기 몇 시간 전이었고, '
                   '일부는 끝까지 정상인 설비였습니다. 자세한 근거는 reports/연주_조기경보_결과보고서.md 5절')

    st.subheader('검증 세트 고장별 결과', icon=':material/fact_check:')
    ev = v['events'].rename(columns={'SCENARIO_ID': '시나리오', 'TYPE': '고장 종류'})
    st.dataframe(ev[['시나리오', '고장 종류', '감지', '리드타임(h)']].sort_values(['고장 종류', '리드타임(h)'], ascending=[True, False]),
                 hide_index=True, column_config={'리드타임(h)': st.column_config.NumberColumn(format='%.1f')})
    st.caption('리드타임: 고장 4시간 전까지 난 경보 중 가장 이른 경보부터 고장까지의 시간. '
               '자세한 근거는 reports/연주_조기경보_결과보고서.md 4절과 부록 B')


    st.subheader('한계와 대응 방안', icon=':material/report:')
    lim = pd.DataFrame(LIMITS)
    pick = st.pills('중요도', ['높음', '중간', '낮음'], selection_mode='multi', default=['높음', '중간', '낮음'], key='lim_pick')
    lim = lim[lim['중요도'].isin(pick or [])]
    st.dataframe(lim[['중요도', '구분', '한계', '영향', '지금 대응', '추가 대응', '근거']], hide_index=True,
                 column_config={c: st.column_config.TextColumn(width='large') for c in ['영향', '지금 대응', '추가 대응']})
    st.caption('돌발 고장·원인 부품·리드타임처럼 모델만으로 풀 수 없는 한계는 운영(정기 교체 유지, 점검 결과 되먹임, 예비 재고와 주의 단계 발주)으로 메웁니다. '
               '자세한 내용은 결과보고서 10절.')

    st.subheader('온톨로지: 지금 상태를 하나의 그래프로', icon=':material/account_tree:')
    g = ontology_graph(assumptions(), status_table())
    cnt = ontology.counts(g)
    bad = ontology.check(g)
    e1, e2, e3, e4 = st.columns(4, border=True)
    e1.metric('트리플 수', f'{len(g):,}')
    e2.metric('설비 / 예측', f'{cnt.get("설비", 0)} / {cnt.get("예측", 0)}')
    e3.metric('경보 / 주의', f'{cnt.get("경보 (WARNING)", 0)} / {cnt.get("주의 (CAUTION)", 0)}')
    e4.metric('규칙 점검', '통과' if not bad else f'{len(bad)}건 어김', help='; '.join(bad) or '신호→부품 링크 없음, 주의·경보 겹침 없음, 정비 결정마다 추천 있음')
    need = ontology.parts_needed(g)
    st.markdown(
        '- **스키마**: `ontology/casting_dss.ttl` (OWL). 객체 = 설비·부품·신호·예측·경보 단계·정비 결정·선택지·정비 작업·정기 교체 계획·고장, '
        '링크 = 가진다·관측한다·근거·요청한다·추천한다·교체한다 등\n'
        '- **인스턴스**: 이 화면의 지금 시각·데이터 세트·가정값으로 만든다(예측·신호·정비 기록·DSS 결정)\n'
        f'- **질의 예 (SPARQL)**: 경보 설비가 필요로 하는 부품 = '
        + (' · '.join(f'{k} {v}개' for k, v in need.items()) if need else '없음'))
    st.download_button('지금 그래프 내려받기 (Turtle)', g.serialize(format='turtle'), file_name=f'casting_{split}_{now}h.ttl',
                       mime='text/turtle', icon=':material/download:')



# 5. 비용 계산기

elif page == '비용 계산기':
    v = bundle['validation']['event']
    n = dict(n_det=v['감지'], n_miss=v['놓침'], n_needed=v['고장과 무관한 경보 묶음: 필요한 교체'],
             n_false=v['고장과 무관한 경보 묶음: 오경보'])
    st.subheader('부품 손익: 원인 부품을 얼마나 맞히면 한 부품만 바꿔도 되나', icon=ICONS['부품 손익'])
    st.caption('원인 부품을 얼마나 맞힐 수 있으면 이득인가')
    st.markdown(
        f'검증 세트 결과(고장 {v["고장"]}건 중 {n["n_det"]}건 감지, 고장과 무관한 교체 {n["n_needed"]}번, '
        f'진짜 오경보 {n["n_false"]}번)에 현장 비용 가정(만 원, "정책·비용" 탭의 가정값)을 적용합니다.\n'
        '- 경보가 나면 바로 교체합니다(정책: 경보 즉시 교체). 한 부품만 바꾸는데 틀리면, 불필요 정비가 되고 고장도 못 막습니다.\n'
        '- 정기 교체 직전에 울린 맞는 경보도 교체 한 번으로 셉니다. 예정 교체와 묶어 처리하면 이 비용은 줄어듭니다(의사결정 지원, 다음 단계).\n'
        '- 센서로는 원인 부품을 구분할 수 없어(EDA E9), 부품을 맞힐 확률은 현장 점검 능력에 달려 있습니다.')
    p = st.slider('원인 부품을 맞힐 확률', 0.5, 1.0, 0.5, step=0.01,
                  help='0.5 = 둘 중 하나를 찍음, 1.0 = 항상 맞힘')
    costs_f, _ = make_cfg(assumptions())
    cc = cost_curve(**n, costs=costs_f)
    one = float(np.interp(p, cc['p'], cc['한 부품만 교체']))
    both, none = cc['두 부품 모두 교체'].iloc[0], cc['아무것도 안 함'].iloc[0]
    c1, c2, c3 = st.columns(3, border=True)
    c1.metric('한 부품만 교체', fmt_cost(one, costs_f), f'{fmt_cost(one - none, costs_f)} (아무것도 안 함 대비)', delta_color='inverse')
    c2.metric('두 부품 모두 교체', fmt_cost(both, costs_f), fmt_cost(both - none, costs_f), delta_color='inverse')
    c3.metric('아무것도 안 함', fmt_cost(none, costs_f))
    u = bundle['validation'].get('unlabeled')
    if u:
        two_ = 2 * costs_f.part + costs_f.unnecessary + costs_f.replace_fixed + costs_f.replace_h * costs_f.downtime_h
        st.warning(
            f'**위 비용은 정답을 알 수 있는 구간만 센 값입니다.** 관측이 끝나 결과를 알 수 없는 시점 {u["시점"]}개에서도 '
            f'경보가 {u["경보 시점"]}번(경보 묶음 {u["경보 묶음"]}개) 울렸고, 실제 운영이라면 교체가 일어납니다. '
            f'두 부품 교체 기준으로 경보 묶음마다 약 {fmt_cost(two_, costs_f)}, 모두 {fmt_cost(u["경보 묶음"] * two_, costs_f)}까지 더 들 수 있습니다. 이 경보가 관측 뒤의 고장을 막았는지, '
            '"아무것도 안 함"에서 관측 뒤에 고장이 났는지는 알 수 없어 운영 비용은 확정할 수 없습니다.')
    even = cc.loc[cc['한 부품만 교체'] <= both, 'p']
    if len(even):
        st.info(f'부품을 맞힐 확률이 **{even.iloc[0]:.2f} 이상**이면 한 부품만 바꾸는 쪽이 두 부품 모두 바꾸는 것보다 쌉니다.')
    long = cc.melt(id_vars='p', var_name='방식', value_name='비용')
    chart = alt.Chart(long).mark_line(strokeWidth=2.5).encode(
        x=alt.X('p:Q', title='원인 부품을 맞힐 확률'), y=alt.Y('비용:Q', title='비용 (만 원, 낮을수록 좋음)'),
        color=alt.Color('방식:N', scale=alt.Scale(domain=['한 부품만 교체', '두 부품 모두 교체', '아무것도 안 함'],
                                                range=[LINE_BLUE, LINE_GREEN, OK_GRAY])))
    now_rule = alt.Chart(pd.DataFrame({'p': [p]})).mark_rule(strokeDash=[4, 4]).encode(x='p:Q')
    st.altair_chart((chart + now_rule).properties(height=360), width='stretch')


# 화면 위치 → 주소 (뒤로 가기·앞으로 가기용, 맨 위 apply_url과 짝)
sync_url()
