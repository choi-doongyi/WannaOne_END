"""온톨로지(OWL/RDF): 지금 시각의 설비·예측·경보·정비 결정을 하나의 그래프로 잇고, SPARQL로 근거를 따라간다.

    from src.ontology import build, trace, parts_needed, check
    g = build(now, status, signals, works, plans, decisions, cost_unit='점', safety_max=0.2, th=0.56, cth=0.57)
    trace(g, 'S0248')          # 설비 → 경보 → 예측 → 신호 → 정비 결정 → 선택지 → 정기 교체 계획
    g.serialize(format='turtle')

스키마(TBox): ontology/casting_dss.ttl. 설계: 결과보고서 8-2절 (설계 기록 부록 D-1).
인스턴스(ABox)는 화면이 이미 가진 값(예측·신호·정비 기록·정기 교체 계획·DSS 결정)으로만 만든다. 그래서 원자료·모델 파일이 없는
클라우드에서도 같은 그래프가 나온다. 판단(경보 기준, DSS 추천)은 그래프가 아니라 기존 계산(src/dss.py)이 하고, 그래프는 결과를
잇고 설명한다. "신호 → 부품" 링크는 만들지 않는다(센서로 원인 부품을 구분하지 못함, 결과보고서 부록 A E9). check()가 확인한다.

입력 (모두 지금 시각까지 알 수 있는 값)
  status     설비(시나리오)마다 한 줄. index = SCENARIO_ID, 열: STAGE('경보'/'주의'/'정상'/'정비 중'/'정비 후 예측 대기'),
             PROB, DEGRADED_PROB, PRED_H(마지막 예측 시각), FIRST_ALARM_H, FIRST_CAUTION_H
  signals    설비마다 마지막 1시간 신호. index = SCENARIO_ID, 열: DONE_H, SCORE, Z_MLD-AMP, Z_CONTROL_DELAY, Z_WDL, Z_OSC-STK
  works      정비 기록: SCENARIO_ID, PART_CD, 종류, START_H, END_H, FAIL_H (시작이 지금 이전인 것)
  plans      앞으로의 정기 교체 계획: SCENARIO_ID, PLAN_H, PLAN_PART
  decisions  경보 설비마다 {'sid', 'rank', 'constraint', 'best', 'options': [src.dss.options의 선택지 dict]}
  actions    사람이 한 일 (src.actions.ActionLog.frame(split, now)): 승인·보류·부품 발주·점검 결과
  feedback   점검 되먹임 {'n', 'accuracy', 'breakeven', 'ready'} (src.actions.inspection_accuracy, one_part_ready)
"""

import math
import re
from pathlib import Path

from rdflib import Graph, Literal, Namespace, RDF, RDFS, XSD

ROOT = Path(__file__).resolve().parent.parent
TBOX = ROOT / 'ontology' / 'casting_dss.ttl'
CC = Namespace('https://github.com/choi-doongyi/WannaOne_END/ontology#')
EQP_ID = 'P1-CC1-INS01'
SIGNALS = {'Z_MLD-AMP': CC.MLD_AMP, 'Z_CONTROL_DELAY': CC.CONTROL_DELAY, 'Z_WDL': CC.WDL, 'Z_OSC-STK': CC.OSC_STK,
           'SCORE': CC.SignalScore}
PART_CLASS = {'valve': CC.Valve, 'oscillator': CC.Oscillator}
_ID = re.compile(r'^[A-Za-z0-9_-]{1,32}$')      # IRI에 넣는 값은 이 모양만 (그래프 구조를 바꾸는 값이 들어오지 않게)


def _sid(s):
    s = str(s)
    if not _ID.match(s):
        raise ValueError(f'IRI에 쓸 수 없는 ID: {s!r}')
    return s


def _num(x):
    """NaN·None은 값을 넣지 않는다 (모르는 값과 0을 구분)"""
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else Literal(round(x, 4), datatype=XSD.decimal)


def _add(g, s, p, o):
    if o is not None:
        g.add((s, p, o))


def tbox():
    g = Graph()
    g.parse(TBOX, format='turtle')
    g.bind('cc', CC)
    return g


def build(now, status, signals, works, plans, decisions=(), cost_unit=None, safety_max=None, th=None, cth=None,
          actions=None, feedback=None):
    """지금 시각 now의 상태로 TBox + ABox 그래프를 만든다"""
    g = tbox()
    eq = {}
    for sid in status.index:
        s = _sid(sid)
        e = CC[f'eq_{s}']
        eq[s] = e
        g.add((e, RDF.type, CC.Equipment))
        g.add((e, RDFS.label, Literal(f'{EQP_ID} ({s})', lang='ko')))
        g.add((e, CC.equipmentId, Literal(EQP_ID)))
        g.add((e, CC.scenarioId, Literal(s)))
        for code, cls in PART_CLASS.items():
            p = CC[f'part_{s}_{code}']
            g.add((p, RDF.type, cls))
            g.add((e, CC.hasPart, p))

    # 예측 → 신호 (마지막 예측 하나와 그때까지의 마지막 1시간 신호)
    for sid, r in status.iterrows():
        s = _sid(sid)
        if r['STAGE'] in ('정비 중', '정비 후 예측 대기') or _num(r.get('PROB')) is None:
            continue
        pr = CC[f'pred_{s}']
        g.add((pr, RDF.type, CC.Prediction))
        g.add((pr, CC.predictsFor, eq[s]))
        _add(g, pr, CC.atHour, _num(r.get('PRED_H')))
        _add(g, pr, CC.failureProb24h, _num(r.get('PROB')))
        _add(g, pr, CC.degradedProb, _num(r.get('DEGRADED_PROB')))
        if s in signals.index:
            sg = signals.loc[s]
            for col, sig in SIGNALS.items():
                v = _num(sg.get(col))
                if v is None:
                    continue
                rd = CC[f'read_{s}_{sig.split("#")[-1]}']
                g.add((rd, RDF.type, CC.SignalReading))
                g.add((rd, CC.ofSignal, sig))
                g.add((rd, CC.observes, eq[s]))
                g.add((rd, CC.score, v))
                _add(g, rd, CC.atHour, _num(sg.get('DONE_H')))
                g.add((pr, CC.basedOn, rd))
        # 경보 단계 (경보는 정비 전까지 유지하는 화면 규칙을 그대로 따른다: status의 STAGE)
        if r['STAGE'] in ('경보', '주의'):
            warn = r['STAGE'] == '경보'
            al = CC[f'alarm_{s}']
            g.add((al, RDF.type, CC.WarningAlarm if warn else CC.CautionAlarm))
            g.add((al, CC.alarmOn, eq[s]))
            g.add((al, CC.triggeredBy, pr))
            _add(g, al, CC.firstOnHour, _num(r.get('FIRST_ALARM_H' if warn else 'FIRST_CAUTION_H')))
            _add(g, al, CC.threshold, _num(th if warn else cth))

    # 정비 기록과 고장 (지금까지 일어난 것)
    for i, w in works.reset_index(drop=True).iterrows():
        s = _sid(w['SCENARIO_ID'])
        if s not in eq:
            continue
        wk = CC[f'work_{s}_{i}']
        g.add((wk, RDF.type, CC.MaintenanceWork))
        g.add((wk, CC.performedOn, eq[s]))
        g.add((wk, CC.workType, Literal(str(w['종류']))))
        _add(g, wk, CC.atHour, _num(w['START_H']))
        _add(g, wk, CC.endHour, _num(w['END_H']) if w['END_H'] <= now else None)   # 미래의 종료 시각은 넣지 않는다
        if w['PART_CD'] in PART_CLASS:
            g.add((wk, CC.replaces, CC[f'part_{s}_{w["PART_CD"]}']))
        if _num(w.get('FAIL_H')) is not None and w['FAIL_H'] <= now:
            f = CC[f'fail_{s}_{i}']
            g.add((f, RDF.type, CC.Failure))
            g.add((f, CC.occurredOn, eq[s]))
            g.add((f, CC.atHour, _num(w['FAIL_H'])))
            if w['PART_CD'] in PART_CLASS:
                g.add((f, CC.causedBy, CC[f'part_{s}_{w["PART_CD"]}']))

    # 정기 교체 계획 (앞으로 남은 것)
    plan_of = {}
    for i, p in plans[plans['PLAN_H'] > now].sort_values('PLAN_H').reset_index(drop=True).iterrows():
        s = _sid(p['SCENARIO_ID'])
        if s not in eq:
            continue
        pl = CC[f'plan_{s}_{i}']
        g.add((pl, RDF.type, CC.MaintenancePlan))
        g.add((pl, CC.plannedFor, eq[s]))
        g.add((pl, CC.atHour, _num(p['PLAN_H'])))
        if p['PLAN_PART'] in PART_CLASS:
            g.add((pl, CC.replaces, CC[f'part_{s}_{p["PLAN_PART"]}']))
        plan_of.setdefault(s, pl)

    # 정비 결정 (DSS 결과)
    if decisions:
        rule = CC.safety_rule
        g.add((rule, RDF.type, CC.SafetyRule))
        _add(g, rule, CC.threshold, _num(safety_max))
        g.add((rule, CC.rationale, Literal('정비가 끝나기 전 고장 확률이 기준을 넘는 선택지는 고르지 않는다. 모두 넘으면 지금 교체(즉시 조치)')))
        cost = CC.cost_assumption
        g.add((cost, RDF.type, CC.CostAssumption))
        if cost_unit:
            g.add((cost, CC.costUnit, Literal(cost_unit)))
    for d in decisions:
        s = _sid(d['sid'])
        if s not in eq:
            continue
        dc = CC[f'decision_{s}']
        g.add((dc, RDF.type, CC.MaintenanceDecision))
        g.add((CC[f'alarm_{s}'], CC.requests, dc))
        g.add((dc, CC.checkedBy, CC.safety_rule))
        g.add((dc, CC.pricedBy, CC.cost_assumption))
        if d.get('rank') is not None and not (isinstance(d['rank'], float) and math.isnan(d['rank'])):
            g.add((dc, CC.priority, Literal(int(d['rank']), datatype=XSD.integer)))
        if d.get('constraint'):
            g.add((dc, CC.constraintNote, Literal(str(d['constraint']))))
        for j, o in enumerate(d['options']):
            op = CC[f'option_{s}_{j}']
            g.add((op, RDF.type, CC.MaintenanceOption))
            g.add((op, RDFS.label, Literal(o['선택지'], lang='ko')))
            g.add((dc, CC.considers, op))
            _add(g, op, CC.hoursUntilDone, _num(o['정비 완료까지(h)']))
            _add(g, op, CC.failProbBeforeDone, _num(o['고장 확률']))
            _add(g, op, CC.expectedCost, _num(o['기대 총비용']))
            g.add((op, CC.meetsSafety, Literal(bool(o['안전 기준 충족']))))
            if o.get('근거'):
                g.add((op, CC.rationale, Literal(o['근거'])))
            if o['선택지'].startswith('다음 정기') and s in plan_of:
                g.add((op, CC.joinsPlan, plan_of[s]))
            if o['선택지'] == d['best']:
                g.add((dc, CC.recommends, op))

    # 액션 (사람이 한 일). 승인·보류는 지금 결정이 있을 때 그 결정과 선택지에 잇는다
    kinds = {'승인 요청': CC.ApprovalRequest, '승인': CC.Approval, '보류': CC.Hold, '작업 시작': CC.WorkStart, '작업 완료': CC.WorkDone,
             '부품 발주': CC.PartOrder, '점검 결과': CC.InspectionResult}
    if actions is not None:
        for i, a in actions.reset_index(drop=True).iterrows():
            s = _sid(a['설비'])
            if s not in eq or a['종류'] not in kinds:
                continue
            ac = CC[f'action_{s}_{i}']
            g.add((ac, RDF.type, kinds[a['종류']]))
            g.add((ac, CC.actionOn, eq[s]))
            g.add((ac, CC.atHour, _num(a['시각(h)'])))
            g.add((ac, CC.byRole, Literal(str(a['누가']))))
            if a.get('메모'):
                g.add((ac, CC.note, Literal(str(a['메모']))))
            dc = CC[f'decision_{s}']
            if a['종류'] in ('승인 요청', '승인', '보류', '작업 시작', '작업 완료') and (dc, RDF.type, CC.MaintenanceDecision) in g:
                g.add((ac, CC.actsOn, dc))
                if a['종류'] == '승인':
                    for op in g.objects(dc, CC.considers):
                        if str(g.value(op, RDFS.label)) == a.get('선택지'):
                            g.add((ac, CC.approvedOption, op))
            if a['종류'] == '부품 발주' and a.get('부품') in PART_CLASS:
                g.add((ac, CC.orders, CC[f'part_{s}_{a["부품"]}']))
            if a['종류'] == '점검 결과':
                if a.get('점검 판단') in PART_CLASS:
                    g.add((ac, CC.judgedPart, CC[f'part_{s}_{a["점검 판단"]}']))
                if a.get('확인된 원인') in PART_CLASS:
                    g.add((ac, CC.confirmedPart, CC[f'part_{s}_{a["확인된 원인"]}']))
    if feedback is not None:
        fb = CC.inspection_feedback
        g.add((fb, RDF.type, CC.InspectionFeedback))
        g.add((fb, CC.sampleSize, Literal(int(feedback['n']), datatype=XSD.integer)))
        _add(g, fb, CC.accuracy, _num(feedback.get('accuracy')))
        _add(g, fb, CC.threshold, _num(feedback.get('breakeven')))
        g.add((fb, CC.onePartReady, Literal(bool(feedback.get('ready')))))
    return g


# 질의 (SPARQL)

PREFIX = 'PREFIX cc: <https://github.com/choi-doongyi/WannaOne_END/ontology#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n'

Q_ALARM = PREFIX + """
SELECT ?kind ?first ?th ?ph ?prob ?deg WHERE {
  ?al cc:alarmOn ?e ; a ?k ; cc:triggeredBy ?pr .
  ?k rdfs:subClassOf cc:Alarm ; rdfs:label ?kind .
  OPTIONAL { ?al cc:firstOnHour ?first }  OPTIONAL { ?al cc:threshold ?th }
  OPTIONAL { ?pr cc:atHour ?ph }  OPTIONAL { ?pr cc:failureProb24h ?prob }  OPTIONAL { ?pr cc:degradedProb ?deg }
}"""

Q_PRED = PREFIX + """
SELECT ?ph ?prob ?deg WHERE {
  ?pr cc:predictsFor ?e .
  OPTIONAL { ?pr cc:atHour ?ph }  OPTIONAL { ?pr cc:failureProb24h ?prob }  OPTIONAL { ?pr cc:degradedProb ?deg }
}"""

Q_SIGNALS = PREFIX + """
SELECT ?name ?score ?h WHERE {
  ?pr cc:predictsFor ?e ; cc:basedOn ?rd .
  ?rd cc:ofSignal ?sig ; cc:score ?score .
  ?sig rdfs:label ?name .
  OPTIONAL { ?rd cc:atHour ?h }
} ORDER BY DESC(?score)"""

Q_DECISION = PREFIX + """
SELECT ?name ?done ?p ?cost ?safe ?why ?rec ?rank ?note ?plan_h ?plan_part WHERE {
  ?al cc:alarmOn ?e ; cc:requests ?dc .
  ?dc cc:considers ?op .
  ?op rdfs:label ?name ; cc:meetsSafety ?safe .
  OPTIONAL { ?op cc:hoursUntilDone ?done }  OPTIONAL { ?op cc:failProbBeforeDone ?p }
  OPTIONAL { ?op cc:expectedCost ?cost }  OPTIONAL { ?op cc:rationale ?why }
  OPTIONAL { ?dc cc:priority ?rank }  OPTIONAL { ?dc cc:constraintNote ?note }
  OPTIONAL { ?op cc:joinsPlan ?pl . ?pl cc:atHour ?plan_h . OPTIONAL { ?pl cc:replaces ?pp . ?pp a ?pc . ?pc rdfs:subClassOf cc:Part ; rdfs:label ?plan_part } }
  BIND(EXISTS { ?dc cc:recommends ?op } AS ?rec)
}"""

Q_HISTORY = PREFIX + """
SELECT ?kind ?start ?end ?part WHERE {
  ?w cc:performedOn ?e ; cc:workType ?kind ; cc:atHour ?start .
  OPTIONAL { ?w cc:endHour ?end }
  OPTIONAL { ?w cc:replaces ?p . ?p a ?pc . ?pc rdfs:subClassOf cc:Part ; rdfs:label ?part }
} ORDER BY ?start"""

Q_ACTIONS = PREFIX + """
SELECT ?kind ?h ?role ?note ?opt ?ord ?judged ?conf WHERE {
  ?a cc:actionOn ?e ; a ?k ; cc:atHour ?h ; cc:byRole ?role .
  ?k rdfs:subClassOf cc:Action ; rdfs:label ?kind .
  OPTIONAL { ?a cc:note ?note }
  OPTIONAL { ?a cc:approvedOption ?o . ?o rdfs:label ?opt }
  OPTIONAL { ?a cc:orders ?p1 . ?p1 a ?c1 . ?c1 rdfs:subClassOf cc:Part ; rdfs:label ?ord }
  OPTIONAL { ?a cc:judgedPart ?p2 . ?p2 a ?c2 . ?c2 rdfs:subClassOf cc:Part ; rdfs:label ?judged }
  OPTIONAL { ?a cc:confirmedPart ?p3 . ?p3 a ?c3 . ?c3 rdfs:subClassOf cc:Part ; rdfs:label ?conf }
} ORDER BY ?h"""

Q_FEEDBACK = PREFIX + """
SELECT ?n ?acc ?be ?ready WHERE {
  ?f a cc:InspectionFeedback ; cc:sampleSize ?n ; cc:onePartReady ?ready .
  OPTIONAL { ?f cc:accuracy ?acc }  OPTIONAL { ?f cc:threshold ?be }
}"""

Q_PARTS_NEEDED = PREFIX + """
SELECT ?part (COUNT(DISTINCT ?e) AS ?n) WHERE {
  ?al a cc:WarningAlarm ; cc:alarmOn ?e .
  ?e cc:hasPart ?p . ?p a ?pc . ?pc rdfs:subClassOf cc:Part ; rdfs:label ?part .
} GROUP BY ?part ORDER BY ?part"""


def _py(v):
    return None if v is None else v.toPython()


def _rows(g, q, sid):
    return [{k: _py(r[k]) for k in r.labels} for r in g.query(q, initBindings={'e': CC[f'eq_{_sid(sid)}']})]


def trace(g, sid, with_cost=True):
    """설비 하나의 근거 사슬: [(단계, 내용)]. with_cost=False면 비용 숫자를 뺀다 (현장 작업 화면)"""
    out = []
    al = _rows(g, Q_ALARM, sid)
    pred = _rows(g, Q_PRED, sid)
    if al:
        a = al[0]
        first = f', 처음 켜진 시각 {a["first"]:.0f}h' if a['first'] is not None else ''
        th = f' (기준 {a["th"]:.2f})' if a['th'] is not None else ''
        out.append(('경보 단계', f'{a["kind"]}{first}{th}'))
    if pred:
        p = pred[0]
        txt = f'{p["ph"]:.0f}h 예측 · 24시간 안 고장 확률 {p["prob"]:.2f}' if p['prob'] is not None else '예측 없음'
        if p['deg'] is not None:
            txt += f' · 열화 확률 {p["deg"]:.2f}'
        out.append(('근거 예측', txt))
        sig = _rows(g, Q_SIGNALS, sid)
        main = [s for s in sig if s['name'] != '네 신호 평균']
        avg = [s for s in sig if s['name'] == '네 신호 평균']
        if main:
            out.append(('근거 신호', ' · '.join(f'{s["name"]} {s["score"]:.2f}' + (' (벗어남)' if s['score'] > 1 else '') for s in main)
                        + (f' · 평균 {avg[0]["score"]:.2f}' if avg else '') + f'  ({main[0]["h"]:.0f}h까지 1시간 값, 1 = 정상 경계)'))
    else:
        out.append(('근거 예측', '지금 쓸 수 있는 예측이 없습니다 (정비 중이거나 재가동 뒤 예측 대기)'))
    fb = [{k: _py(r[k]) for k in r.labels} for r in g.query(Q_FEEDBACK)]
    if fb and fb[0]['ready']:
        out.append(('원인 부품', f'센서로는 구분할 수 없지만, 점검으로 원인 부품을 맞힌 비율 {fb[0]["acc"]:.2f}({fb[0]["n"]}건)가 '
                                f'손익분기 {fb[0]["be"]:.2f}를 넘어 점검 후 한 부품 교체를 추천 (점검 되먹임)'))
    else:
        out.append(('원인 부품', '밸브 · 진동장치 중 어느 쪽인지 센서로는 구분할 수 없어 두 부품 모두 대상 (부록 A E9)'))
    dec = _rows(g, Q_DECISION, sid)
    if dec:
        d0 = dec[0]
        if d0['rank'] is not None:
            out.append(('정비팀 배정 순위', f'{d0["rank"]}번째'))
        for d in sorted(dec, key=lambda d: (not d['rec'], d['done'] or 0)):
            head = ('추천' if d['rec'] else '비교') + f': {d["name"]}'
            parts = [f'완료까지 {d["done"]:.0f}h' if d['done'] is not None else None,
                     f'완료 전 고장 확률 {d["p"]:.2f}' if d['p'] is not None else None,
                     '안전 기준 충족' if d['safe'] else '안전 기준 초과']
            if with_cost and d['cost'] is not None:
                parts.append(f'기대 총비용 {d["cost"]:,.0f}')
            if d['plan_h'] is not None:
                parts.append(f'정기 교체 계획 {d["plan_h"]:.0f}h' + (f' ({d["plan_part"]})' if d['plan_part'] else ''))
            out.append(('정비 결정', head + ' · ' + ' · '.join(x for x in parts if x)))
        if d0['note']:
            out.append(('제약', d0['note']))
    for a in _rows(g, Q_ACTIONS, sid):
        txt = f'{a["kind"]} · {a["h"]:.0f}h · {a["role"]}'
        if a['opt']:
            txt += f' · {a["opt"]}'
        if a['ord']:
            txt += f' · {a["ord"]}'
        if a['kind'] == '점검 결과':
            txt += f' · 점검 판단 {a["judged"] or "구분 못함"} · 확인된 원인 {a["conf"] or "아직 모름"}'
        if a['note']:
            txt += f' · 메모: {a["note"]}'
        out.append(('조치', txt))
    hist = _rows(g, Q_HISTORY, sid)
    if hist:
        out.append(('정비 이력', ' · '.join(f'{h["kind"]} {h["start"]:.0f}h' + (f'({h["part"]})' if h['part'] else '') for h in hist)))
    return out


def parts_needed(g):
    """경보 설비가 필요로 하는 부품 수 {부품 이름: 설비 수} (두 부품 모두 교체 대상)"""
    return {str(r['part']): int(r['n']) for r in g.query(Q_PARTS_NEEDED)}


def check(g):
    """그래프 규칙 점검: 어긴 내용 목록 (비면 통과)"""
    bad = []
    if g.query(PREFIX + 'ASK { ?s ?p ?o . ?s a/rdfs:subClassOf* cc:Signal . ?o a/rdfs:subClassOf* cc:Part }').askAnswer \
            or g.query(PREFIX + 'ASK { ?s ?p ?o . ?s a cc:SignalReading . ?o a/rdfs:subClassOf* cc:Part }').askAnswer:
        bad.append('신호 → 부품 링크가 있다 (센서로 원인 부품을 구분할 수 없으므로 두지 않는다)')
    if g.query(PREFIX + 'ASK { ?a a cc:CautionAlarm , cc:WarningAlarm }').askAnswer:
        bad.append('주의와 경보가 한 개체에 같이 붙었다')
    if g.query(PREFIX + 'ASK { ?d a cc:MaintenanceDecision . FILTER NOT EXISTS { ?d cc:recommends ?o } }').askAnswer:
        bad.append('추천 선택지가 없는 정비 결정이 있다')
    if g.query(PREFIX + 'ASK { ?a cc:requests ?d . FILTER NOT EXISTS { ?a a cc:WarningAlarm } }').askAnswer:
        bad.append('경보가 아닌 단계가 정비 결정을 요청했다')
    if g.query(PREFIX + 'ASK { ?a cc:actsOn ?d ; cc:actionOn ?e . ?al cc:requests ?d ; cc:alarmOn ?e2 . FILTER(?e != ?e2) }').askAnswer:
        bad.append('다른 설비의 정비 결정에 조치가 붙었다')
    if g.query(PREFIX + 'ASK { ?a a cc:Approval , cc:Hold }').askAnswer:
        bad.append('승인과 보류가 한 조치에 같이 붙었다')
    for prop in (CC.orders, CC.judgedPart, CC.confirmedPart):
        for a, p in g.subject_objects(prop):
            if (g.value(a, CC.actionOn), CC.hasPart, p) not in g:
                bad.append(f'{a} 의 부품이 그 설비의 부품이 아니다')
    for prop, dom in [(CC.predictsFor, CC.Prediction), (CC.performedOn, CC.MaintenanceWork), (CC.alarmOn, CC.Alarm), (CC.actionOn, CC.Action)]:
        for s in g.subjects(prop, None):
            if not any((s, RDF.type, c) in g for c in [dom, *g.subjects(RDFS.subClassOf, dom)]):
                bad.append(f'{s} 의 종류가 {dom} 가 아니다')
    return bad


def counts(g):
    """클래스별 개체 수 (화면 요약용)"""
    q = PREFIX + """SELECT ?label (COUNT(?x) AS ?n) WHERE { ?x a ?c . ?c rdfs:label ?label .
                   FILTER(STRSTARTS(STR(?c), STR(cc:))) } GROUP BY ?label ORDER BY DESC(?n)"""
    return {str(r['label']): int(r['n']) for r in g.query(q)}
