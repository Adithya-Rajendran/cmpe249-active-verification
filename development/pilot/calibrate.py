"""Enumerate all fixed development calibration candidates."""
from itertools import product
from metrics import summarize, veto

GRID = [(q,p,a,o) for q,p,a,o in product((.3,.5),(.7,.8,.9),(.4,.6,.7),(.2,.4,.6)) if a < p]
CONFIDENCE = (1/3,.5,.7,.9)

def candidate(rows, kind, setting):
    output=[]
    for row in rows:
        r=dict(row)
        if kind == 'image': r['failure'] = bool(r.get('evidence_failure', False))
        if not r['failure']:
            if kind == 'confidence':
                r['answer'] = r['base'] if r['confidence'] >= setting else 'UNKNOWN'
            else:
                q,p,a,o = setting
                present = r['valid'] and r['quality'] >= q and r['match'] >= p
                removal = r['valid'] and r['quality'] >= q and r['obstruction'] <= o and r['match'] <= a
                if kind == 'full': r['answer'] = veto(r['base'],present,removal)
                elif kind == 'image': r['answer'] = 'PRESENT' if present else 'REMOVED' if removal else 'UNKNOWN'
                else: raise ValueError(kind)
        else: r['answer'] = 'UNKNOWN'
        output.append(r)
    return output

def select(rows,kind):
    if not rows or any(r.get('partition') != 'calibration' for r in rows):
        raise ValueError('calibration accepts calibration rows only')
    base = summarize([dict(r,answer=r['base']) for r in rows])
    baseline = base['useful']['estimate']
    grid = CONFIDENCE if kind == 'confidence' else GRID
    results = []
    for setting in grid:
        m=summarize(candidate(rows,kind,setting))
        u,y,p,r = (m[k]['estimate'] for k in ('unsupported','useful','present_resolution','removal_resolution'))
        eligible = all(v is not None for v in (baseline,u,y,p,r)) and y >= baseline-.10-1e-12 and p>=.50 and r>=.50
        results.append({'setting':setting,'metrics':m,'eligible':eligible})
    feasible=[(i,x) for i,x in enumerate(results) if x['eligible']]
    selected=min(feasible,key=lambda ix:(ix[1]['metrics']['unsupported']['estimate'], -ix[1]['metrics']['useful']['estimate'],ix[0]))[0] if feasible else None
    return {'baseline':base,'candidates':results,'selected_index':selected}
