"""정비 작업 지시: 관리자가 배정하고 작업자가 시작·완료 보고한다 (팀원 설계 화면의 작업 관리 + 재생 시각).

    from src.tasks import TaskStore
    store = TaskStore(path)                  # 로컬: SQLite 파일 / TaskStore(None): 세션 안에서만 (공개 클라우드)
    store.add(split, sid, title, team, priority, due_h, created_h, guide, reason, option)   # 같은 설비에 열린 작업이 있으면 False
    store.start(task_id, hour); store.complete(task_id, hour, outcome, details, judged, confirmed)
    store.edit(...)  /  store.edit_report(task_id, outcome, details, judged, confirmed)   # 완료된 작업의 보고 고치기
    store.frame(split, now)                  # 지금 시각에 보이는 작업과 상태(대기·진행 중·완료)

재생 시각: 데이터는 시나리오 시작 후 0~72시간을 재생한다. 그래서 작업의 배정·시작·완료와 예정 시각을 시나리오 시각(h)으로
적고, 상태는 '지금 시각'으로 계산한다(슬라이더를 과거로 옮기면 그때의 작업만 보인다). 실제 기록 시각은 이력에만 남긴다.

저장 (팀 결정 2026-10-15)
  로컬      SQLite 파일 actions/tasks.sqlite3 (.gitignore). 연결은 쓸 때마다 열고 닫는다
  클라우드  공개(비밀번호 없음)라 서버 파일에 쓰지 않고 보는 사람의 세션에만 둔다

검사 (결과보고서 부록 E-1 13·14번과 같은 원칙)
  - 팀·우선순위·결과·부품은 정해진 값만, 설비 ID는 S0000 형식, 시각은 0~240시간
  - 글은 길이 제한과 제어 문자 제거(줄바꿈은 유지). 내려받는 CSV는 수식으로 읽히는 첫 글자(= + - @)를 막는다
  - 같은 설비에 열린(완료 전) 작업이 있으면 배정하지 않는다 (SQLite는 한 트랜잭션 안에서 확인하고 넣는다)
"""

import contextlib
import datetime as dt
import math
import re
import sqlite3
from pathlib import Path

import pandas as pd

TEAMS = {'정비 1팀': ['김 반장', '이 기사', '박 기사', '최 기사'], '정비 2팀': ['정 반장', '강 기사', '조 기사', '윤 기사']}
PRIORITIES = ['긴급', '높음', '보통']
OUTCOMES = ['이상 없음', '이상 발견', '추가 점검 필요']
STATUSES = ['대기', '진행 중', '완료']
JUDGED = ['valve', 'oscillator', '구분 못함']
CONFIRMED = ['valve', 'oscillator', '아직 모름']
LIMITS = {'title': 80, 'guide': 600, 'reason': 600, 'details': 500, 'option': 60}
COLUMNS = ['id', 'split', 'sid', 'title', 'team', 'priority', 'due_h', 'created_h', 'started_h', 'completed_h', 'guide',
           'reason', 'option', 'outcome', 'details', 'judged', 'confirmed', 'created_at', 'updated_at', 'completed_at']
_SID = re.compile(r'^S\d{4}$')
_CTRL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')     # 줄바꿈(\n, \x0a)은 남긴다

SCHEMA = """CREATE TABLE IF NOT EXISTS tasks(
    id INTEGER PRIMARY KEY AUTOINCREMENT, split TEXT NOT NULL, sid TEXT NOT NULL, title TEXT NOT NULL,
    team TEXT NOT NULL, priority TEXT NOT NULL, due_h REAL NOT NULL, created_h REAL NOT NULL,
    started_h REAL, completed_h REAL, guide TEXT NOT NULL, reason TEXT NOT NULL, option TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT '', details TEXT NOT NULL DEFAULT '', judged TEXT NOT NULL DEFAULT '',
    confirmed TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    completed_at TEXT NOT NULL DEFAULT '')"""

DEFAULT_GUIDE = ('① 안전 절차·작업 허가 확인\n'
                 '② 설비 정지와 잠금(LOTO) 확인\n'
                 '③ 진동장치·밸브 상태 점검 (센서로는 원인 부품을 구분할 수 없어 두 부품 모두 확인)\n'
                 '④ 교체 뒤 몰드 레벨·제어 지연·인발 부하·진동 스트로크 신호 확인\n'
                 '⑤ 어느 부품이 문제였는지 완료 보고에 기록')


def _now():
    return dt.datetime.now().isoformat(timespec='seconds')


def clean(text, limit):
    return _CTRL.sub(' ', str(text or '')).strip()[:limit]


def _hour(x):
    h = float(x)
    if not (0 <= h <= 240) or math.isnan(h):
        raise ValueError(f'시각 범위 밖: {x}')
    return h


def _conflict(created_h, completed_h, new_h):
    """같은 설비의 기존 작업이 새 배정과 겹치는가: 아직 완료 전이거나(재생에서 미래 작업 포함), 새 배정 시각에 열려 있다"""
    return completed_h is None or (isinstance(completed_h, float) and math.isnan(completed_h)) or created_h <= new_h < completed_h


def safe_csv(df):
    """내려받기용 CSV (표 계산기 수식 주입 방지, 한글 엑셀용 BOM)"""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_string_dtype(out[c]) or out[c].dtype == object:
            out[c] = out[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v)
    return out.to_csv(index=False).encode('utf-8-sig')


class TaskStore:
    """path가 있으면 SQLite(로컬), 없으면 메모리(세션)"""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.rows = []
        if self.path:
            with self._conn():
                pass                     # 파일과 표를 미리 만든다

    @contextlib.contextmanager
    def _conn(self):
        """연결을 열고, 끝나면 커밋(오류면 되돌림)하고 닫는다 (sqlite3의 with는 닫지 않아 직접 닫는다)"""
        self.path.parent.mkdir(parents=True, exist_ok=True)      # 폴더·파일이 지워져도 다시 만든다
        conn = sqlite3.connect(self.path, timeout=20, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('PRAGMA busy_timeout=20000')
            conn.execute(SCHEMA)
            yield conn
        finally:
            conn.close()

    # 읽기

    def all(self, split=None):
        if self.path:
            with self._conn() as c:
                df = pd.read_sql_query('SELECT * FROM tasks', c)
        else:
            df = pd.DataFrame(self.rows, columns=COLUMNS)
        df = df.reindex(columns=COLUMNS)
        return df[df['split'] == split].reset_index(drop=True) if split else df

    def frame(self, split, now):
        """지금 시각에 보이는 작업(배정 시각 ≤ 지금)과 그때의 상태"""
        df = self.all(split)
        df = df[df['created_h'] <= now].copy()
        done = df['completed_h'].notna() & (df['completed_h'] <= now)
        started = df['started_h'].notna() & (df['started_h'] <= now)
        df['상태'] = '대기'
        df.loc[started, '상태'] = '진행 중'
        df.loc[done, '상태'] = '완료'
        late = ~done & (df['due_h'] < now)
        df['구분'] = '예정'
        df.loc[~done & (df['due_h'] <= now), '구분'] = '지금'
        df.loc[late & ~started, '구분'] = '지연'
        df.loc[done, '구분'] = '완료'
        return df.sort_values(['구분', 'priority', 'due_h', 'id'],
                              key=lambda s: s.map({'지연': 0, '지금': 1, '예정': 2, '완료': 3, '긴급': 0, '높음': 1, '보통': 2})
                              if s.name in ('구분', 'priority') else s).reset_index(drop=True)

    def open_task(self, split, sid, now):
        """지금 시각에 열린(완료 전) 작업 한 줄 (없으면 None)"""
        df = self.frame(split, now)
        df = df[(df['sid'] == sid) & (df['상태'] != '완료')]
        return None if df.empty else df.iloc[0].to_dict()

    def blocking(self, split, sid, now):
        """now에 이 설비로 배정하면 막는 작업 (add와 같은 규칙: 완료 전이거나 now에 열려 있던 작업). 없으면 None"""
        df = self.all(split)
        df = df[df['sid'] == sid]
        hit = df[[_conflict(r.created_h, r.completed_h, now) for r in df.itertuples()]]
        return None if hit.empty else hit.iloc[-1].to_dict()

    def blocked_sids(self, split, now):
        df = self.all(split)
        return {r.sid for r in df.itertuples() if _conflict(r.created_h, r.completed_h, now)}

    # 쓰기

    def add(self, split, sid, title, team, priority, due_h, created_h, guide, reason, option=''):
        """작업 배정. 같은 설비에 지금 열린 작업이 있으면 넣지 않고 False"""
        if split not in ('test', 'validation') or not _SID.match(str(sid)):
            raise ValueError('데이터 세트·설비 ID 형식이 아님')
        if team not in TEAMS or priority not in PRIORITIES:
            raise ValueError('팀·우선순위는 정해진 값만')
        title = clean(title, LIMITS['title'])
        if not title:
            raise ValueError('작업명을 입력하세요')
        stamp = _now()
        row = {'split': split, 'sid': sid, 'title': title, 'team': team, 'priority': priority, 'due_h': _hour(due_h),
               'created_h': _hour(created_h), 'started_h': None, 'completed_h': None,
               'guide': clean(guide, LIMITS['guide']), 'reason': clean(reason, LIMITS['reason']),
               'option': clean(option, LIMITS['option']), 'outcome': '', 'details': '', 'judged': '', 'confirmed': '',
               'created_at': stamp, 'updated_at': stamp, 'completed_at': ''}
        if self.path:
            with self._conn() as c:
                c.execute('BEGIN IMMEDIATE')
                try:
                    rows = c.execute('SELECT created_h, completed_h FROM tasks WHERE split=? AND sid=?', (split, sid)).fetchall()
                    if any(_conflict(r['created_h'], r['completed_h'], row['created_h']) for r in rows):
                        c.execute('ROLLBACK')
                        return False
                    keys = [k for k in COLUMNS if k != 'id']
                    c.execute(f'INSERT INTO tasks ({",".join(keys)}) VALUES ({",".join("?" * len(keys))})', [row[k] for k in keys])
                    c.execute('COMMIT')
                except Exception:
                    c.execute('ROLLBACK')
                    raise
            return True
        if any(_conflict(r['created_h'], r['completed_h'], row['created_h']) for r in self.rows
               if r['split'] == split and r['sid'] == sid):
            return False
        row['id'] = max([r['id'] for r in self.rows], default=0) + 1
        self.rows.append(row)
        return True

    def _update(self, task_id, fields, where_open=False, where_done=False):
        fields = {**fields, 'updated_at': _now()}
        if self.path:
            with self._conn() as c:
                sets = ','.join(f'{k}=?' for k in fields)
                sql = (f'UPDATE tasks SET {sets} WHERE id=?' + (' AND completed_h IS NULL' if where_open else '')
                       + (' AND completed_h IS NOT NULL' if where_done else ''))
                return c.execute(sql, [*fields.values(), int(task_id)]).rowcount > 0
        for r in self.rows:
            if (r['id'] == int(task_id) and not (where_open and r['completed_h'] is not None)
                    and not (where_done and r['completed_h'] is None)):
                r.update(fields)
                return True
        return False

    def start(self, task_id, hour):
        return self._update(task_id, {'started_h': _hour(hour)}, where_open=True)

    def complete(self, task_id, hour, outcome, details, judged='구분 못함', confirmed='아직 모름'):
        """완료 보고: 결과와 수행 내용이 있어야 완료. 이미 완료된 작업이면 False"""
        details = clean(details, LIMITS['details'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED:
            raise ValueError('결과·부품은 정해진 값만')
        if not details:
            raise ValueError('수행 내용을 입력해야 완료할 수 있습니다')
        h = _hour(hour)
        return self._update(task_id, {'completed_h': h, 'outcome': outcome, 'details': details, 'judged': judged,
                                      'confirmed': confirmed, 'completed_at': _now()}, where_open=True)

    def set_status(self, task_id, status, hour, outcome='', details='', judged='구분 못함', confirmed='아직 모름'):
        """설비관리자가 작업 상태를 직접 바꾼다 (잘못 누른 시작·완료 되돌리기 포함). 상태는 시작·완료 시각으로 정해진다.
        대기       시작·완료 시각과 완료 보고를 지운다
        진행 중    완료 시각과 완료 보고를 지운다 (시작 시각이 없으면 hour로)
        완료       완료 보고가 있어야 한다 (complete와 같은 검사). 시작 시각이 없으면 hour로
        반환: 바뀌었으면 True"""
        h = _hour(hour)
        df = self.all()
        df = df[df['id'] == int(task_id)]
        if df.empty or status not in STATUSES:
            raise ValueError('없는 작업이거나 정해진 상태가 아님')
        started = float(df.iloc[0]['started_h']) if pd.notna(df.iloc[0]['started_h']) else None
        clear = {'completed_h': None, 'outcome': '', 'details': '', 'judged': '', 'confirmed': '', 'completed_at': ''}
        if status == '대기':
            return self._update(task_id, {'started_h': None, **clear})
        if status == '진행 중':
            return self._update(task_id, {'started_h': started if started is not None else h, **clear})
        details = clean(details, LIMITS['details'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED:
            raise ValueError('결과·부품은 정해진 값만')
        if not details:
            raise ValueError('수행 내용을 입력해야 완료할 수 있습니다')
        return self._update(task_id, {'started_h': started if started is not None else h, 'completed_h': max(h, started or h),
                                      'outcome': outcome, 'details': details, 'judged': judged, 'confirmed': confirmed,
                                      'completed_at': _now()})

    def cancel(self, task_id):
        """배정 취소 (설비관리자): 완료 전 작업만 지운다. 그 설비는 다시 '배정할 경보'로 돌아간다.
        취소 이력은 진행 기록(src/actions.py의 '배정 취소')에 남긴다. 완료된 작업은 먼저 상태를 되돌려야 한다"""
        if self.path:
            with self._conn() as c:
                return c.execute('DELETE FROM tasks WHERE id=? AND completed_h IS NULL', [int(task_id)]).rowcount > 0
        n = len(self.rows)
        self.rows = [r for r in self.rows if not (r['id'] == int(task_id) and r['completed_h'] is None)]
        return len(self.rows) < n

    def edit_report(self, task_id, outcome, details, judged, confirmed):
        """완료 보고 고치기 (설비관리자): 이미 완료된 작업만. 완료 시각은 그대로 둔다"""
        details = clean(details, LIMITS['details'])
        if outcome not in OUTCOMES or judged not in JUDGED or confirmed not in CONFIRMED:
            raise ValueError('결과·부품은 정해진 값만')
        if not details:
            raise ValueError('수행 내용은 비울 수 없습니다')
        return self._update(task_id, {'outcome': outcome, 'details': details, 'judged': judged, 'confirmed': confirmed},
                            where_done=True)

    def edit(self, task_id, team, priority, due_h, guide, reason):
        if team not in TEAMS or priority not in PRIORITIES:
            raise ValueError('팀·우선순위는 정해진 값만')
        return self._update(task_id, {'team': team, 'priority': priority, 'due_h': _hour(due_h),
                                      'guide': clean(guide, LIMITS['guide']), 'reason': clean(reason, LIMITS['reason'])})
