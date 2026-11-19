"""Descriptive paired arrangement bootstrap for the completed cross-simulator pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np

CONTROLS=('primary','relaxed_guard','native_normalized_guard')
ENGINES=('isaac','mujoco')
ENVIRONMENTS=('shelf','tabletop')
LEVELS=('D0','D4','D6')
METRICS=('coverage','correct_yield','clear_present_resolution','clear_removed_resolution','placed_both_unknown','physical_error_rate')
DRAWS=2000
SEED=20261117


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rate(record,metric):
    if metric in ('coverage','correct_yield','placed_both_unknown'):
        item=record[metric]; numerator=item['numerator']; denominator=item['denominator']
        expected=20 if metric!='placed_both_unknown' else 5
    elif metric.startswith('clear_'):
        state=metric.removesuffix('_resolution'); item=record['clear_resolution'][state]
        numerator=item['numerator']; denominator=item['denominator']; expected=5
    elif metric=='physical_error_rate':
        numerator=record['physical_errors']; denominator=record['coverage']['denominator']; expected=20
    else: raise ValueError('unknown metric')
    if type(numerator) is not int or type(denominator) is not int or denominator!=expected or not 0<=numerator<=denominator: raise ValueError('invalid count or denominator')
    if metric!='physical_error_rate' and not np.isclose(item['rate'],numerator/denominator,atol=1e-12): raise ValueError('inconsistent rate')
    return numerator/denominator


def index_analysis(data):
    if data.get('study_id')!='cross-sim-dev-v1' or set(data.get('metrics',{}))!=set(CONTROLS): raise ValueError('unexpected analysis')
    index={}; bases={}
    for control in CONTROLS:
        records=data['metrics'][control]
        for r in records:
            if r['arrangement']=='ALL': continue
            key=(control,r['engine'],r['environment'],r['level'],r['arrangement'])
            if key in index or key[1] not in ENGINES or key[2] not in ENVIRONMENTS or key[3] not in LEVELS: raise ValueError('duplicate or invalid arrangement')
            index[key]={metric:rate(r,metric) for metric in METRICS}
            bases.setdefault(key[2],set()).add(key[4])
    if any(len(bases.get(e,()))!=5 for e in ENVIRONMENTS) or set(bases['shelf'])&set(bases['tabletop']): raise ValueError('expected five distinct base arrangements per environment')
    bases={e:sorted(bases[e]) for e in ENVIRONMENTS}
    expected={(c,engine,e,level,b) for c in CONTROLS for engine in ENGINES for e in ENVIRONMENTS for level in LEVELS for b in bases[e]}
    if set(index)!=expected: raise ValueError('incomplete paired arrangement summaries')
    return index,bases


def bootstrap_indices(bases,draws=DRAWS,seed=SEED):
    rng=np.random.Generator(np.random.PCG64(seed))
    return {environment:rng.integers(0,len(bases[environment]),size=(draws,len(bases[environment]))) for environment in ENVIRONMENTS}


def interval(values, samples):
    point=float(np.mean(values)); boot=np.mean(np.asarray(values)[samples],axis=1)
    lo,hi=np.percentile(boot,[2.5,97.5],method='linear')
    return {'estimate':point,'interval_95_percentile':[float(lo),float(hi)]}


def compute(data,draws=DRAWS,seed=SEED):
    if draws!=DRAWS or seed!=SEED: raise ValueError('bootstrap configuration is frozen')
    index,bases=index_analysis(data); samples=bootstrap_indices(bases,draws,seed)
    paired=[]
    for control in CONTROLS:
        for environment in ENVIRONMENTS:
            for level in LEVELS:
                for metric in METRICS:
                    differences=np.array([index[(control,'mujoco',environment,level,b)][metric]-index[(control,'isaac',environment,level,b)][metric] for b in bases[environment]])
                    paired.append({'control':control,'environment':environment,'level':level,'metric':metric,'arrangements':5,**interval(differences,samples[environment])})
    guards=[]
    # Each environment is a separate stratum. The combined-engine contrast averages paired engine effects per arrangement.
    for level in LEVELS:
        for engine in (*ENGINES,'combined'):
            per_env={}
            for environment in ENVIRONMENTS:
                values=[]
                for base in bases[environment]:
                    es=ENGINES if engine=='combined' else (engine,)
                    values.append(np.mean([index[('relaxed_guard',e,environment,level,base)]['correct_yield']-index[('primary',e,environment,level,base)]['correct_yield'] for e in es]))
                per_env[environment]=np.asarray(values)
                guards.append({'engine':engine,'environment':environment,'level':level,'metric':'correct_yield','arrangements':5,**interval(per_env[environment],samples[environment])})
            values=np.concatenate([per_env[e] for e in ENVIRONMENTS]); boot=sum(np.mean(per_env[e][samples[e]],axis=1) for e in ENVIRONMENTS)/len(ENVIRONMENTS)
            lo,hi=np.percentile(boot,[2.5,97.5],method='linear')
            guards.append({'engine':engine,'environment':'balanced_combined','level':level,'metric':'correct_yield','arrangements':10,
                           'estimate':float(np.mean(values)),'interval_95_percentile':[float(lo),float(hi)]})
    overall=[]
    # Pool the three fixed levels within each base arrangement before resampling.
    for engine in (*ENGINES,'combined'):
        per_env={}
        for environment in ENVIRONMENTS:
            es=ENGINES if engine=='combined' else (engine,)
            per_env[environment]=np.array([np.mean([index[('relaxed_guard',e,environment,level,base)]['correct_yield']-index[('primary',e,environment,level,base)]['correct_yield']
                for level in LEVELS for e in es]) for base in bases[environment]])
        boot=sum(np.mean(per_env[e][samples[e]],axis=1) for e in ENVIRONMENTS)/len(ENVIRONMENTS)
        lo,hi=np.percentile(boot,[2.5,97.5],method='linear')
        overall.append({'engine':engine,'environment':'balanced_combined','level':'all_fixed_levels','metric':'correct_yield','arrangements':10,
                        'estimate':float(np.mean([np.mean(per_env[e]) for e in ENVIRONMENTS])),'interval_95_percentile':[float(lo),float(hi)]})
    draw_sha=hashlib.sha256(b''.join(samples[e].astype('<i8').tobytes() for e in ENVIRONMENTS)).hexdigest()
    return {'study_id':'cross-sim-dev-v1','scope':'exploratory descriptive development intervals; not confirmatory population inference',
            'bootstrap':{'unit':'base arrangement','clusters_per_environment':5,'arrangements':bases,'draws':draws,'seed':seed,'generator':'NumPy PCG64',
                         'draw_order':list(ENVIRONMENTS),'draws_sha256':draw_sha,'pairing':'same draws across engines, controls, states and levels; environments sampled independently',
                         'interval':'two-sided 95% percentile, linear interpolation (NumPy percentile method=linear / Hyndman-Fan type 7)',
                         'balanced_combined':'equal mean of the two environment strata; combined engines average paired engine effects within each arrangement'},
            'engine_difference_direction':'mujoco_minus_isaac','guard_difference_direction':'relaxed_guard_minus_primary',
            'engine_differences':paired,'guard_correct_yield_differences':guards,'guard_correct_yield_all_levels':overall}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    path=Path(args.analysis).resolve(); output=Path(args.out).resolve()
    if output.exists(): raise FileExistsError(output)
    protocol=Path(__file__).with_name('scoring_protocol.json')
    specification=json.loads(protocol.read_text())['bootstrap']
    if specification != {'unit':'base_arrangement','draws':DRAWS,'seed':SEED}: raise ValueError('bootstrap protocol mismatch')
    data=json.loads(path.read_text()); result=compute(data)
    result['protocol_sha256']=sha(protocol)
    result['analysis_sha256']=sha(path); result['postprocessor_sha256']=sha(__file__); result['runtime']={'python':platform.python_version(),'numpy':np.__version__}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'engine_intervals':len(result['engine_differences']),'guard_intervals':len(result['guard_correct_yield_differences'])}))


if __name__=='__main__': main()
