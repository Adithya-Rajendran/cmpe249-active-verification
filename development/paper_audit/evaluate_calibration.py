"""Measure all locked D0 positive and independent negative witness contrasts."""
import argparse
import csv
import json
import numpy as np
import PIL
import platform
from collections import defaultdict
from pathlib import Path
from development.paper_audit.inputs_build import verify_entries,box_from_mask,check_release,sha
from development.paper_audit.verify_controls import verify as verify_states
from development.paper_audit.evaluate_witness import primary_rgb,primary_mask,mask,exposure,choose_patch,contrast,witness_pass


def check_geometry(receipts):
    verified={}
    for path in receipts:
        rec=json.loads(Path(path).read_text())
        verify_entries(rec['manifest'])
        for section in ('sources','inputs_sha256'):
            for filename,digest in rec[section].items():
                if sha(filename)!=digest: raise ValueError('geometry receipt input changed')
        for row in rec['rows']:
            if row['failures']: raise ValueError('geometry verification failed')
        successful={(r['base_id'],r['level']) for r in rec['rows']}
        for filename,digest in rec['inputs_sha256'].items():
            f=Path(filename)
            if f.name=='observed.json':
                if (f.parent.parent.name,f.parent.name) not in successful: raise ValueError('geometry row missing')
                verified[str(f.resolve())]=digest
    if not verified: raise ValueError('no verified observed geometry')
    return verified


def evaluate(manifests,out,geometry_receipts):
    check_release()
    geometry=check_geometry(geometry_receipts)
    rows=[]; seen=set(); verified={}
    for path in manifests:
        data=json.loads(Path(path).read_text());verify_entries(data)
        for c in data['rows']:
            if c['level']!='D0' or c['camera']!=c['nominal_camera']: raise ValueError('calibration requires fixed D0 camera')
            key=(c['engine'],c['arrangement'],c['level'],c['stop'])
            if key in seen: raise ValueError('duplicate calibration view')
            seen.add(key)
            observed=str(Path(c['observed']['path']).resolve())
            if geometry.get(observed)!=c['observed']['sha256']: raise ValueError('missing geometry verification for capture')
            folder=str(Path(observed).parent)
            if folder not in verified:
                checks=verify_states(Path(folder),c['engine'])
                if not checks or not all(item['state_geometry_matches'] for item in checks): raise ValueError('physical control mismatch')
                verified[folder]=True
            imgs=c['images']; refmask=primary_mask(c['reference']['target_mask']['path'])
            box=box_from_mask(c['reference']['target_mask']['path'])
            for place in ('clear','placed'):
                ps,rs=place+'_present',place+'_removed'
                for label in (rs,'repeat1_'+rs,'repeat2_'+rs,'distractor_hidden_'+rs):
                    if mask(imgs[label]['target_mask']['path']).any(): raise ValueError('removed target mask nonempty')
                exposed=primary_mask(imgs[ps]['target_mask']['path'])
                vis=exposure(mask(imgs[ps]['target_mask']['path']),mask(c['isolated_target_mask']['path']))
                patch=choose_patch(refmask,exposed,box)
                row={'engine':c['engine'],'environment':c['environment'],'arrangement':c['arrangement'],'stop':c['stop'],
                     'placement':place,'eligible':bool(patch),'patch':patch,'visibility':vis,
                     'positive':None,'repeat_bound':None,'positive_pass':None,'negatives':[]}
                a,b=primary_rgb(imgs[ps]['rgb']['path']),primary_rgb(imgs[rs]['rgb']['path'])
                def global_mae(x,y):return float(np.abs(x.astype(np.int16)-y.astype(np.int16)).mean())
                repeat_values={}; global_repeats={}
                for state,base in ((ps,a),(rs,b)):
                    repeat1=primary_rgb(imgs['repeat1_'+state]['rgb']['path'])
                    repeat2=primary_rgb(imgs['repeat2_'+state]['rgb']['path'])
                    global_repeats[state]=[global_mae(base,repeat1),global_mae(base,repeat2)]
                    if patch: repeat_values[state]=contrast(base,repeat1,patch,refmask,exposed)
                if patch:
                    repeat_p,repeat_r=repeat_values[ps],repeat_values[rs]
                    positive=contrast(a,b,patch,refmask,exposed)
                    row.update(positive=positive,repeat_bound={'present':repeat_p,'removed':repeat_r},positive_pass=witness_pass(positive,repeat_p,repeat_r))
                for state,base in ((ps,a),(rs,b)):
                    base_masks={role:mask(imgs[state][role]['path']) for role in ('target_mask','blocker_mask')}
                    for kind in ('repeat1_','repeat2_','distractor_hidden_'):
                        label=kind+state; control=primary_rgb(imgs[label]['rgb']['path'])
                        global_delta=np.abs(base.astype(np.int16)-control.astype(np.int16))
                        mask_changes={}
                        for role in ('target_mask','blocker_mask'):
                            cm=mask(imgs[label][role]['path']);bm=base_masks[role]
                            pm=primary_mask(imgs[label][role]['path']); pb=primary_mask(imgs[state][role]['path'])
                            changed=pm!=pb; overlap=None
                            if patch:
                                x,y,w,h=patch['xywh'];overlap=int(changed[y:y+h,x:x+w].sum())
                            mask_changes[role]={'native_changed':int(np.count_nonzero(cm!=bm)),
                                'native_added':int(np.count_nonzero(cm & ~bm)),
                                'native_removed':int(np.count_nonzero(bm & ~cm)),
                                'primary_patch_changed':overlap}
                        g=float(global_delta.mean()); value=contrast(base,control,patch,refmask,exposed) if patch else None
                        item={'state':state,'source':label,'meaning':('repeat bound' if kind=='repeat1_' else 'independent same-state repeat' if kind=='repeat2_' else 'distractor_0 hidden with target unchanged'),
                            'contrast':value,'whole_image_mean_abs_rgb':g,
                            'whole_image_changed_pixel_fraction':float(np.any(global_delta!=0,axis=2).mean()),
                            'whole_image_mae_above_both_repeats':g>max(global_repeats[state]) if kind=='distractor_hidden_' else None,
                            'mask_changes':mask_changes,
                            'pass':witness_pass(value,repeat_p,repeat_r) if patch and kind!='repeat1_' else None}
                        row['negatives'].append(item)
                rows.append(row)
    groups=defaultdict(list)
    for r in rows: groups[(r['engine'],r['environment'])].append(r)
    summary=[]
    for (engine,env),group in sorted(groups.items()):
        parked=[r for r in group if r['placement']=='clear']; placed=[r for r in group if r['placement']=='placed']
        arrangements=sorted({r['arrangement'] for r in group})
        partial={a:sum(bool(r['positive_pass']) and r['visibility']['visible_fraction_of_alone'] is not None and 0<r['visibility']['visible_fraction_of_alone']<1 for r in placed if r['arrangement']==a) for a in arrangements}
        negative=[n for r in group for n in r['negatives'] if n['meaning']!='repeat bound']
        distractors=[n for n in negative if n['meaning'].startswith('distractor_0')]
        summary.append({'engine':engine,'environment':env,'parked_total':len(parked),'parked_eligible':sum(r['eligible'] for r in parked),
            'parked_positive':sum(bool(r['positive_pass']) for r in parked),'parked_yield_pass':len(parked)==10 and sum(bool(r['positive_pass']) for r in parked)>=8,
            'placed_partial_positive_by_arrangement':partial,'placed_yield_pass':bool(partial) and all(v>=1 for v in partial.values()),
            'negative_total':len(negative),'negative_eligible':sum(n['pass'] is not None for n in negative),
            'negative_triggers':sum(n['pass'] is True for n in negative),
            'distractor_controls':len(distractors),'distractor_whole_image_mae_above_both_repeats':sum(n['whole_image_mae_above_both_repeats'] for n in distractors)})
    valid=len(rows)==80 and len(summary)==4 and all(s['parked_yield_pass'] and s['placed_yield_pass'] and s['negative_triggers']==0 for s in summary)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'calibration.json').write_text(json.dumps({'witness_main_endpoint_permitted':valid,'runtime':{'python':platform.python_version(),'pillow':PIL.__version__,'numpy':np.__version__},'design_release_sha256':check_release(),'evaluator_sha256':sha(__file__),'manifest_sha256':{str(p):sha(p) for p in manifests},'geometry_receipt_sha256':{str(p):sha(p) for p in geometry_receipts},'groups':summary,'rows':rows},indent=2,allow_nan=False)+'\n')
    flat=[]
    for r in rows:
        x={k:v for k,v in r.items() if k not in ('patch','visibility','repeat_bound','negatives')}
        x.update({k:json.dumps(r[k],sort_keys=True) for k in ('patch','visibility','repeat_bound','negatives')});flat.append(x)
    with (out/'calibration.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',required=True);p.add_argument('--geometry-receipt',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();evaluate(a.manifest,a.out,a.geometry_receipt)
