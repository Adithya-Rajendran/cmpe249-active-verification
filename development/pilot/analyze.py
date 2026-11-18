"""Score declared development arms and retain calibration candidates."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from calibrate import candidate, select
from metrics import summarize, veto, validate_split

ARMS=('base','place_first','confidence','full','removal_only','image','oracle','always_unknown','always_present','always_removed')

def report(rows):
    out={'overall':summarize(rows),'disputed_as_unknown':summarize(rows,True),'cells':{},'pairs':{}}
    cells=defaultdict(list);pairs=defaultdict(dict)
    for r in rows:
        cells[(r['physical'],r['support'])].append(r)
        if r['phase'] in ('occluded','removed'): pairs[(r['arrangement'],r['stop'])][r['phase']]=r
    for cell,group in cells.items(): out['cells']['/'.join(cell)]=summarize(group)
    complete=[p for p in pairs.values() if set(p)=={'occluded','removed'}]
    valid=[p for p in complete if not p['occluded']['failure'] and not p['removed']['failure']]
    for label, selected in [('all',complete),('unresolved_twins',[p for p in complete if p['occluded']['support']==p['removed']['support']=='UNKNOWN']),('clear_twins',[p for p in complete if p['occluded']['support']=='PRESENT' and p['removed']['support']=='REMOVED'])]:
        good=[p for p in selected if not p['occluded']['failure'] and not p['removed']['failure']]
        out['pairs'][label]={'scheduled':len(selected),'valid':len(good),
            'both_unknown_numerator':sum(p['occluded']['answer']==p['removed']['answer']=='UNKNOWN' for p in good),
            'both_unknown_denominator':len(selected)}
    return out

def build(rows, predictions):
    cache={(x['id'],x['prompt_arm']):x for x in predictions}
    if len(cache)!=len(predictions): raise ValueError('duplicate model outputs')
    output=[]
    for row in rows:
        for arm in ('base','place_first'):
            x=cache[(row['id'],arm)]
            row=dict(row);row[arm+'_failure']=x['failure']
            row[arm+'_answer']='UNKNOWN' if x['failure'] else x['result']['answer']
            if arm=='base': row['confidence']=None if x['failure'] else x['result']['candidates'][x['result']['code']]['normalized_score']
        row['base']=row['base_answer'];row['failure']=row['base_failure'];output.append(row)
    return output

def arm_rows(rows,arm,setting=None):
    if arm in ('confidence','full','image'): return candidate(rows,arm,setting)
    out=[]
    for x in rows:
        r=dict(x);r['answer']=r['base']
        if arm=='place_first': r['answer']=r['place_first_answer'];r['failure']=r['place_first_failure']
        elif arm=='removal_only':
            q,p,a,o=setting
            rem=r['valid'] and r['quality']>=q and r['match']<=a and r['obstruction']<=o
            r['answer']=veto(r['base'],False,rem,True)
        elif arm=='oracle': r['answer']=r['base'] if r['support'] in ('PRESENT','REMOVED') and r['base']==r['support'] else 'UNKNOWN'
        elif arm.startswith('always_'): r['answer']=arm.split('_')[1].upper();r['failure']=False
        if r['failure']: r['answer']='UNKNOWN'
        out.append(r)
    return out

def bootstrap(arms, reference, seed=20261116, draws=2000):
    rng=random.Random(seed); groups=defaultdict(list)
    for row in reference: groups[row['environment']].append(row['arrangement'])
    groups={k:sorted(set(v)) for k,v in groups.items()}
    lookup={name:{r['id']:r for r in rows} for name,rows in arms.items()}
    diffs=defaultdict(list)
    for _ in range(draws):
        chosen=[(env,rng.choice(ids)) for env,ids in groups.items() for _ in ids]
        samples={name:[] for name in arms}
        for index,(_,arr) in enumerate(chosen):
            for original in reference:
                if original['arrangement']==arr:
                    for name in arms: samples[name].append(dict(lookup[name][original['id']],arrangement=str(index)))
        estimates={name:summarize(sample) for name,sample in samples.items()}
        for left,right in [('full','base'),('full','place_first'),('full','confidence')]:
            if left not in estimates or right not in estimates: continue
            for metric in ('unsupported','useful'):
                a,b=estimates[left][metric]['estimate'],estimates[right][metric]['estimate']
                if a is not None and b is not None: diffs[f'{left}_minus_{right}/{metric}'].append(a-b)
    out={}
    for key,values in diffs.items():
        values.sort(); n=len(values)
        out[key]={'draws':n,'lower_2_5':values[round((n-1)*.025)],'upper_97_5':values[round((n-1)*.975)],'degenerate':values[0]==values[-1]}
    return {'seed':seed,'requested_draws':draws,'intervals':out,'warning':'Exploratory only; six evaluation arrangements. Zero events and degenerate intervals do not establish zero risk.'}

def main():
    manifest=json.loads(Path(sys.argv[1]).read_text());rows=manifest['rows']
    preds=[json.loads(line) for line in Path(sys.argv[2]).read_text().splitlines()]
    validate_split(rows); rows=build(rows,preds)
    cal=[r for r in rows if r['partition']=='calibration'];evaluation=[r for r in rows if r['partition']=='pilot_evaluation']
    chosen={k:select(cal,k) for k in ('full','confidence','image')}
    settings={k:(v['candidates'][v['selected_index']]['setting'] if v['selected_index'] is not None else None) for k,v in chosen.items()}
    result={'calibration':chosen,'settings':settings,'evaluation':{}}
    samples={}
    for arm in ARMS:
        setting=settings.get('full' if arm=='removal_only' else arm)
        if arm in ('full','removal_only','confidence','image') and setting is None:
            result['evaluation'][arm]={'status':'calibration_failed'};continue
        samples[arm]=arm_rows(evaluation,arm,setting)
        result['evaluation'][arm]=report(samples[arm])
    result['bootstrap']=bootstrap({k:v for k,v in samples.items() if k in {'base','place_first','full','confidence'}},evaluation)
    Path(sys.argv[3]).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
