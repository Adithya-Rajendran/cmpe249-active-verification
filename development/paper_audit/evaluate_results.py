"""Independently bind predictions to native captures and aggregate paired outcomes."""
import argparse
import csv
import hashlib
import json
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from development.paper_audit.inputs_build import sha,verify_entries,box_from_mask,check_release
from development.paper_audit.evaluate_witness import mask,primary_rgb,exposure
from development.paper_audit.evaluate_calibration import check_geometry


@lru_cache(maxsize=4096)
def pixel_hash(path):
    with Image.open(path) as im:
        arr=np.asarray(im.convert('RGB'))
    return hashlib.sha256(json.dumps(list(arr.shape),separators=(',',':')).encode()+b'\0'+arr.tobytes()).hexdigest()


def records(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    result={r['id']:r for r in rows}
    if len(result)!=len(rows): raise ValueError('duplicate prediction')
    return result


def failure_class(record):
    failure=record.get('failure')
    if failure in (None,False,''): return 'none'
    if failure=='registration_gate': return 'declared_registration_abstention'
    if failure=='invalid_model_output': return 'invalid_model_output'
    return 'technical_failure'


def ratio(n,d): return n/d if d else None


def measure(rows):
    n=len(rows); definite=sum(r['answer']!='UNKNOWN' for r in rows); correct=sum(r['correct'] for r in rows)
    wrong=sum(r['answer']!='UNKNOWN' and not r['correct'] for r in rows)
    present=[r for r in rows if r['truth']=='PRESENT']; removed=[r for r in rows if r['truth']=='REMOVED']
    return {'n':n,'correct':correct,'definite':definite,'wrong_definite':wrong,
            'accuracy_all':ratio(correct,n),'coverage':ratio(definite,n),'wrong_definite_all':ratio(wrong,n),
            'retained_n':len(present),'false_removal_n':sum(r['answer']=='REMOVED' for r in present),
            'false_removal_rate':ratio(sum(r['answer']=='REMOVED' for r in present),len(present)),
            'technical_failures':sum(r['failure_class']=='technical_failure' for r in rows),
            'invalid_model_outputs':sum(r['failure_class']=='invalid_model_output' for r in rows),
            'registration_abstentions':sum(r['failure_class']=='declared_registration_abstention' for r in rows),
            'unknown_without_failure':sum(r['answer']=='UNKNOWN' and r['failure_class']=='none' for r in rows),
            'removed_n':len(removed),'false_presence_n':sum(r['answer']=='PRESENT' for r in removed),
            'false_presence_rate':ratio(sum(r['answer']=='PRESENT' for r in removed),len(removed))}


def bootstrap(rows, weights, keys):
    counts=np.zeros((len(keys),6),dtype=np.int64); index={key:i for i,key in enumerate(keys)}
    for r in rows:
        a=counts[index[(r['environment'],r['arrangement'])]]
        a[0]+=1; a[1]+=r['correct']; a[2]+=r['answer']!='UNKNOWN'
        a[3]+=r['answer']!='UNKNOWN' and not r['correct']
        a[4]+=r['truth']=='PRESENT'; a[5]+=r['answer']=='REMOVED' and r['truth']=='PRESENT'
    totals=weights @ counts
    removed=np.array([sum(r['truth']=='REMOVED' and (r['environment'],r['arrangement'])==key for r in rows) for key in keys])
    false_present=np.array([sum(r['truth']=='REMOVED' and r['answer']=='PRESENT' and (r['environment'],r['arrangement'])==key for r in rows) for key in keys])
    def interval(n,d):
        if np.any(d==0): return None
        return np.quantile(n/d,[.025,.975]).tolist()
    return {'accuracy_all':interval(totals[:,1],totals[:,0]), 'coverage':interval(totals[:,2],totals[:,0]),
            'wrong_definite_all':interval(totals[:,3],totals[:,0]), 'false_removal_rate':interval(totals[:,5],totals[:,4]),
            'false_presence_rate':interval(weights@false_present,weights@removed)}


def evaluate(packet,ledger_path,methods,out,geometry_receipts,seed=249,repetitions=10000):
    geometry=check_geometry(geometry_receipts)
    packet=Path(packet); ledger=json.loads(Path(ledger_path).read_text()); cases=json.loads((packet/'cases.json').read_text())
    if check_release()!=ledger['design_release_sha256']: raise ValueError('design release changed')
    if sha(packet/'packet-metadata.json')!=ledger['packet_metadata_sha256']: raise ValueError('packet metadata changed')
    if sha(packet/'cases.json')!=ledger['cases_sha256']: raise ValueError('packet binding changed')
    for p,d in ledger['manifests'].items():
        if sha(p)!=d: raise ValueError('manifest changed')
    private={r['id']:r for r in ledger['rows']}; public={r['id']:r for r in cases}
    if len(private)!=len(ledger['rows']) or len(public)!=len(cases) or set(private)!=set(public): raise ValueError('case inventory mismatch')
    preds={name:records(path) for name,path in methods.items()}
    bindings={}
    for name,values in preds.items():
        binding=json.loads((Path(methods[name]).parent/'binding.json').read_text())
        bindings[name]=binding
        if binding['cases_sha256']!=ledger['cases_sha256'] or set(values)!=set(public): raise ValueError('scorer binding/inventory mismatch')
    declaration=json.loads(Path('development/paper_audit/evaluation_declaration.json').read_text())
    target_by_id={x['base_id']:x['target'] for x in declaration['arrangements']}
    for name,answer in (('always_UNKNOWN','UNKNOWN'),('always_REMOVED','REMOVED')):
        if name in preds: raise ValueError('reserved analytic method')
        preds[name]={uid:{'id':uid,'answer':answer,'reference_sha256':c['reference_sha256'],'query_sha256':c['query_sha256'],'box_xyxy':c['box_xyxy'],'failure':None} for uid,c in public.items()}
    result=[]; pair_data={}
    for uid,c in public.items():
        p=private[uid]; cap=p['capture']; state=p['state']; verify_entries(cap)
        if geometry.get(str(Path(cap['observed']['path']).resolve()))!=cap['observed']['sha256']: raise ValueError('missing geometry verification for capture')
        for key in ('reference','query'):
            if sha(packet/c[key])!=c[key+'_sha256']: raise ValueError('packet image changed')
        if box_from_mask(cap['reference']['target_mask']['path'])!=c['box_xyxy']: raise ValueError('box mismatch')
        for role,source in (('reference',cap['reference']['rgb']['path']),('query',cap['images'][state]['rgb']['path'])):
            image=primary_rgb(source); expected=hashlib.sha256(json.dumps(list(image.shape),separators=(',',':')).encode()+b'\0'+image.tobytes()).hexdigest()
            if expected!=pixel_hash(str(packet/c[role])): raise ValueError('delivered pixels differ from native export')
        if state.endswith('_removed') and mask(cap['images'][state]['target_mask']['path']).any(): raise ValueError('removed target mask nonempty')
        present=state.replace('_removed','_present')
        visible=mask(cap['images'][present]['target_mask']['path']); alone=mask(cap['isolated_target_mask']['path'])
        vis=exposure(visible,alone)
        truth='PRESENT' if state.endswith('_present') else 'REMOVED'
        pair=(cap['engine'],cap['arrangement'],cap['level'],cap['stop'],state.split('_')[0])
        pair_data.setdefault(pair,{})[truth]={'query_native':pixel_hash(cap['images'][state]['rgb']['path']),
            'query_delivered':pixel_hash(packet/c['query']), 'reference_native':pixel_hash(cap['reference']['rgb']['path']),
            'reference_delivered':pixel_hash(packet/c['reference']), 'box':c['box_xyxy'],'id':uid}
        for method,votes in preds.items():
            r=votes[uid]
            if any(r[k]!=c[k] for k in ('reference_sha256','query_sha256','box_xyxy')): raise ValueError('prediction input binding mismatch')
            answer=r['answer']
            if answer not in ('PRESENT','REMOVED','UNKNOWN'): raise ValueError('invalid answer')
            result.append({'id':uid,'method':method,'engine':cap['engine'],'environment':cap['environment'],
                'arrangement':cap['arrangement'],'target':target_by_id.get(cap['arrangement']),'level':cap['level'],'stop':cap['stop'],'state':state,
                'truth':truth,'answer':answer,'correct':int(answer==truth),'failure':r.get('failure'),
                'failure_class':failure_class(r),'refusal':r.get('refusal'),
                'runtime_seconds':r.get('runtime_seconds'),'prediction_record':json.dumps(r,sort_keys=True),'visible_pixels':vis['visible_pixels'],
                'alone_pixels':vis['alone_pixels'],'visible_fraction':vis['visible_fraction_of_alone'],
                'visible_outside_alone_pixels':vis['visible_outside_alone_pixels'],
                'reference_sha256':c['reference_sha256'],'query_sha256':c['query_sha256'],
                'native_reference_pixel_sha256_shape_bound':pixel_hash(cap['reference']['rgb']['path']),
                'native_query_pixel_sha256_shape_bound':pixel_hash(cap['images'][state]['rgb']['path']),
                'delivered_reference_pixel_sha256_shape_bound':pixel_hash(str(packet/c['reference'])),
                'delivered_query_pixel_sha256_shape_bound':pixel_hash(str(packet/c['query'])),
                'native_reference_file_sha256':cap['reference']['rgb']['sha256'],
                'native_query_file_sha256':cap['images'][state]['rgb']['sha256'],
                'native_reference_mask_sha256':cap['reference']['target_mask']['sha256'],
                'native_exposed_mask_sha256':cap['images'][present]['target_mask']['sha256'],
                'native_isolated_mask_sha256':cap['isolated_target_mask']['sha256'],
                'native_blocker_mask_sha256':cap['images'][state]['blocker_mask']['sha256'],
                'task_binding':json.dumps(bindings.get(method,{'analytic':method}),sort_keys=True),
                'box_xyxy':json.dumps(c['box_xyxy'])})
    pairs=[]
    for key,v in sorted(pair_data.items()):
        if set(v)!={'PRESENT','REMOVED'}: raise ValueError('missing pair')
        a,b=v['PRESENT'],v['REMOVED']; same_ref=a['reference_delivered']==b['reference_delivered']; same_box=a['box']==b['box']
        pairs.append(dict(zip(('engine','arrangement','level','stop','placement'),key),present_id=a['id'],removed_id=b['id'],
            native_query_equal=a['query_native']==b['query_native'],delivered_query_equal=a['query_delivered']==b['query_delivered'],
            native_reference_equal=a['reference_native']==b['reference_native'],delivered_reference_equal=same_ref,
            box_equal=same_box,complete_input_equal=same_ref and same_box and a['query_delivered']==b['query_delivered']))
    inventory=[]
    for engine in sorted({r['engine'] for r in result}):
        for env in sorted({a['environment'] for a in declaration['arrangements']}):
            planned={a['base_id'] for a in declaration['arrangements'] if a['environment']==env}
            actual={r['arrangement'] for r in result if r['engine']==engine and r['environment']==env}
            inventory.append({'engine':engine,'environment':env,'planned_layouts':sorted(planned),'observed_layouts':sorted(actual),'missing_layouts':sorted(planned-actual),
                'expected_d6_clear_removed':len(planned)*5,'observed_d6_clear_removed':sum(r['engine']==engine and r['environment']==env and r['level']=='D6' and r['state']=='clear_removed' and r['method']==next(iter(methods)) for r in result)})
    strata={env:sorted({r['arrangement'] for r in result if r['environment']==env}) for env in sorted({r['environment'] for r in result})}
    rng=np.random.default_rng(seed)
    keys=[(env,k) for env,ks in strata.items() for k in ks]; index={key:i for i,key in enumerate(keys)}
    weights=np.zeros((repetitions,len(keys)),dtype=np.int16)
    for env,ks in strata.items():
        drawn=rng.integers(0,len(ks),size=(repetitions,len(ks)))
        for j,k in enumerate(ks): weights[:,index[(env,k)]]=(drawn==j).sum(axis=1)
    groups=defaultdict(list)
    for r in result:
        for env in ('all',r['environment']):
            for level in ('all',r['level']):
                for state in ('all',r['state']): groups[(r['method'],r['engine'],env,level,state)].append(r)
    summaries=[]
    for key,group in sorted(groups.items()):
        d=dict(zip(('method','engine','environment','level','state'),key));d.update(measure(group));d['bootstrap_95']=bootstrap(group,weights,keys);summaries.append(d)
    by_id_method={(r['id'],r['method']):r for r in result}
    certificates=[]
    for pair in pairs:
        if not pair['complete_input_equal']: continue
        for method in preds:
            rp=by_id_method[(pair['present_id'],method)]; rr=by_id_method[(pair['removed_id'],method)]
            certificates.append({**pair,'method':method,'environment':rp['environment'],
                'present_answer':rp['answer'],'removed_answer':rr['answer'],
                'both_definite':rp['answer']!='UNKNOWN' and rr['answer']!='UNKNOWN',
                'different_answers':rp['answer']!=rr['answer'],
                'correct_count':rp['correct']+rr['correct'], 'physical_accuracy_ceiling_identical_deterministic':0.5})
    comparisons=[]
    names=sorted(methods)
    for i,first in enumerate(names):
        for second in names[i+1:]:
            subsets=defaultdict(list)
            for uid in public:
                a=by_id_method[(uid,first)];b=by_id_method[(uid,second)]
                for env in ('all',a['environment']):
                    for level in ('all',a['level']):
                        for state in ('all',a['state']): subsets[(a['engine'],env,level,state)].append((a,b))
            for key,items in sorted(subsets.items()):
                numerator=np.zeros(len(keys)); denominator=np.zeros(len(keys))
                for a,b in items:
                    ix=index[(a['environment'],a['arrangement'])]; numerator[ix]+=b['correct']-a['correct']; denominator[ix]+=1
                den=weights@denominator
                diff=weights@numerator
                ci=np.quantile(diff/den,[.025,.975]).tolist() if np.all(den>0) else None
                comparisons.append(dict(zip(('engine','environment','level','state'),key),first=first,second=second,n=len(items),
                    second_gained=sum(not a['correct'] and b['correct'] for a,b in items),
                    second_lost=sum(a['correct'] and not b['correct'] for a,b in items),
                    accuracy_difference_second_minus_first=sum(b['correct']-a['correct'] for a,b in items)/len(items),
                    bootstrap_95=ci, newly_wrong_definite=sum(b['answer']!='UNKNOWN' and not b['correct'] and not (a['answer']!='UNKNOWN' and not a['correct']) for a,b in items),
                    gained_layouts=sorted({a['arrangement'] for a,b in items if not a['correct'] and b['correct']})))
    detail=defaultdict(list)
    for r in result:
        for dimension in ('target','arrangement'):
            detail[(r['method'],r['engine'],r['environment'],r['level'],r['state'],dimension,str(r[dimension]))].append(r)
    breakdown=[]
    for key,group in sorted(detail.items()):
        x=dict(zip(('method','engine','environment','level','state','dimension','value'),key)); x.update(measure(group)); breakdown.append(x)
    certificate_summary=[]
    cert_groups=defaultdict(list)
    for c in certificates:
        for env in ('all',c['environment']): cert_groups[(c['method'],c['engine'],env,c['level'],c['placement'])].append(c)
    all_cert_keys=set()
    for pair in pairs:
        env=by_id_method[(pair['present_id'],next(iter(preds)))]['environment']
        for method in preds:
            for group_env in ('all',env): all_cert_keys.add((method,pair['engine'],group_env,pair['level'],pair['placement']))
    for key in sorted(all_cert_keys):
        group=cert_groups[key]
        certificate_summary.append(dict(zip(('method','engine','environment','level','placement'),key),
            opposite_truth_pairs=len(group),arrangements=len({r['arrangement'] for r in group}),
            both_definite=sum(r['both_definite'] for r in group),both_unknown=sum(r['present_answer']=='UNKNOWN' and r['removed_answer']=='UNKNOWN' for r in group),
            commitment_pairs=sum(r['present_answer']!='UNKNOWN' or r['removed_answer']!='UNKNOWN' for r in group),different_answers=sum(r['different_answers'] for r in group),
            correct=sum(r['correct_count'] for r in group),total_cases=2*len(group),
            commitment_rate=ratio(sum(r['present_answer']!='UNKNOWN' or r['removed_answer']!='UNKNOWN' for r in group),len(group)),
            deterministic_accuracy_upper_bound=.5 if group else None))
    criteria=[]
    if {'baseline_200','geometry'} <= set(methods):
        for engine in sorted({r['engine'] for r in result}):
            paired=[(by_id_method[(uid,'baseline_200')],by_id_method[(uid,'geometry')])
                    for uid in public if by_id_method[(uid,'baseline_200')]['engine']==engine]
            primary=[(a,b) for a,b in paired if a['level']=='D6' and a['state']=='clear_removed']
            tabletop=[(a,b) for a,b in primary if a['environment']=='tabletop']
            delta=sum(b['correct']-a['correct'] for a,b in primary)
            layout_delta=defaultdict(int)
            for a,b in tabletop: layout_delta[a['arrangement']]+=b['correct']-a['correct']
            gains=sorted(k for k,v in layout_delta.items() if v>0)
            newly_wrong=sum(b['answer']!='UNKNOWN' and not b['correct'] and not (a['answer']!='UNKNOWN' and not a['correct']) for a,b in paired)
            certified=[c for c in certificates if c['engine']==engine and c['method']=='geometry']
            commitments=sum(c['present_answer']!='UNKNOWN' or c['removed_answer']!='UNKNOWN' for c in certified)
            criteria.append({'engine':engine,'primary_n':len(primary),'primary_gain_correct_cases':delta,
                'strict_primary_gain':bool(primary) and delta>0,'tabletop_gain_layouts':gains,
                'at_least_two_tabletop_gain_layouts':len(gains)>=2,'newly_wrong_definite_all_cases':newly_wrong,
                'zero_newly_wrong_definite':newly_wrong==0,'certified_pairs':len(certified),
                'geometry_committing_certified_pairs':commitments,
                'zero_certified_commitments':commitments==0 if certified else None})
    claim_supported=bool(criteria) and all(c['strict_primary_gain'] and c['at_least_two_tabletop_gain_layouts'] and c['zero_newly_wrong_definite'] and c['zero_certified_commitments'] is True for c in criteria)
    output=Path(out);output.mkdir(parents=True,exist_ok=False)
    for name,rows in (('cases.csv',result),('pairs.csv',pairs)):
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'certificates.json').write_text(json.dumps({'summary':certificate_summary,'pairs':certificates},indent=2,allow_nan=False)+'\n')
    (output/'criteria.json').write_text(json.dumps({'tradeoff_free_improvement_supported':claim_supported,'engines':criteria},indent=2,allow_nan=False)+'\n')
    (output/'breakdown.json').write_text(json.dumps(breakdown,indent=2,allow_nan=False)+'\n')
    (output/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2,allow_nan=False)+'\n')
    (output/'summary.json').write_text(json.dumps({'seed':seed,'repetitions':repetitions,'design_release_sha256':check_release(),'evaluator_sha256':sha(__file__),'ledger_sha256':sha(ledger_path),'geometry_receipt_sha256':{str(p):sha(p) for p in geometry_receipts},'scorer_jsonl_sha256':{name:sha(path) for name,path in methods.items()},'inventory':inventory,'strata':strata,'weighting':'case weighted; arrangements sampled within environment, same draws across engines and methods', 'groups':summaries},indent=2,allow_nan=False)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--ledger',required=True);p.add_argument('--method',action='append',required=True,help='name=records.jsonl');p.add_argument('--geometry-receipt',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();evaluate(a.packet,a.ledger,dict(s.split('=',1) for s in a.method),a.out,a.geometry_receipt)
