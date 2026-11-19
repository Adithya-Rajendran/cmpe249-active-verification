"""Generate fixed geometry-only support-label qualification arrangements."""
from __future__ import annotations
import argparse
import copy
import concurrent.futures
import json
import platform
from pathlib import Path
import numpy as np
from sim import arrange as A
from development.stress import build_cohort as B
from development.cross_sim import cohort as C

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=Path(__file__).with_name('support_label_cohort_protocol.json')


def prior_inventory(protocol):
    inventory_path=ROOT/protocol['historical_inventory']; inventory=json.loads(inventory_path.read_text())
    files={Path(p).resolve() for p in inventory['historical_arrangement_files']}
    manifest_hashes={}
    for name in protocol['historical_additional_manifests']:
        path=(ROOT/name).resolve(); manifest_hashes[str(path)]=A.sha256_file(path); manifest=json.loads(path.read_text())
        if 'arrangements' in manifest:
            for rec in manifest['arrangements']:
                for entry in rec['levels'].values():
                    file=(path.parent/entry['path']).resolve()
                    if file.parent!=path.parent or A.sha256_file(file)!=entry['sha256']:raise ValueError('prior manifest hash mismatch')
                    files.add(file)
        else:
            for rec in manifest['records']:
                file=(path.parent/rec['arrangement_file']).resolve()
                if file.parent!=path.parent:raise ValueError('prior path escapes manifest')
                files.add(file)
    seeds,layouts=set(),set(); file_hashes={}
    for file in sorted(files):
        if not file.is_file():raise FileNotFoundError(file)
        arr=json.loads(file.read_text());file_hashes[str(file)]=A.sha256_file(file)
        if isinstance(arr.get('seed'),int):seeds.add(arr['seed'])
        if isinstance(arr.get('objects'),dict):layouts.add(C.layout_hash(arr))
    return seeds,layouts,{'inventory_path':str(inventory_path),'inventory_sha256':A.sha256_file(inventory_path),'additional_manifest_sha256':manifest_hashes,'arrangement_file_sha256':file_hashes,'historical_seeds_count':len(seeds),'historical_layouts_count':len(layouts)}


def validate(protocol,old_seeds):
    if protocol['study_id']!='support-label-qualification-v1' or protocol['status']!='development':raise ValueError('invalid study')
    if [(x['base_id'],x['environment'],x['class'],x['seed_start']) for x in protocol['arrangements']] != [('LQS01','shelf','M16-L',9500),('LQS02','shelf','M16-R',9532),('LQT01','tabletop','M16-L',9700),('LQT02','tabletop','M16-R',9732)]:raise ValueError('reservation changed')
    declared=[]
    for row in protocol['arrangements']:
        for seed in range(row['seed_start'],row['seed_start']+protocol['seeds_per_arrangement']):
            A.check_dev_seed(seed)
            if seed in old_seeds:raise ValueError('historical seed reused')
            declared.append(seed)
    if len(set(declared))!=len(declared):raise ValueError('overlapping reservations')
    reserved=set()
    stress=json.loads(B.PROTOCOL.read_text())
    for start in stress['seed_starts'].values():reserved.update(range(start,start+12*stress['seeds_per_slot']))
    cross=json.loads(C.PROTOCOL.read_text())
    for split in cross['splits'].values():
        for env in cross['environments']:
            start=split['seed_starts'][env];reserved.update(range(start,start+split['count_per_environment']*cross['seeds_per_arrangement']))
    for config in B.CONFIGS.values():
        ctx=A.load_context(config);dev=ctx.config['slots']['dev']
        reserved.update(range(dev['primary_seed_start'],dev['primary_seed_start']+len(A.slot_list(ctx))))
        for values in dev['reserve'].values():reserved.update(values)
    if set(declared)&reserved:raise ValueError('earlier declared seed reservation reused')


def _evaluate(task):
    p,spec,slot,seed,attempt,signs,old_layouts=task
    ctx=A.load_context(B.CONFIGS[spec['environment']])
    try:
        arr=A.generate(seed,slot['family'],slot['m16_side'],attempt,ctx=ctx,target=slot['target'])
        target=A.body_of(arr['objects']['target'],ctx=ctx);placed=A.body_of(arr['objects']['occluder'],'placed',ctx)
        poses=B.motions(ctx,target,signs,p['levels'])
        if any(rec['margin_min_unrounded_px']<float(ctx.config['revisit']['min_margin_px']) for stops,_ in poses.values() for rec in stops.values()):raise ValueError('frame margin')
        digest=C.layout_hash(arr)
        if digest in old_layouts:raise ValueError('historical layout duplicate')
        coverage={level:{cam.name:float(A.coverage(A.perturb_camera(cam,stops[cam.name]['delta']),target,placed)) for cam in ctx.cams}
                  for level,(stops,_) in poses.items()}
        counts={level:sum(.10<v<.90 for v in values.values()) for level,values in coverage.items()};total=sum(counts.values())
        score=(int(total>=2),int(all(counts[l]>0 for l in ('D0','D6'))),total)
        log={'seed':seed,'attempt':attempt,'status':'candidate','layout_sha256':digest,'coverage':coverage,'middle_counts':counts,'rank':list(score)}
        return log,(score,seed,attempt,arr,poses,digest)
    except (A.PlacementFailure,RuntimeError,ValueError) as exc:
        return {'seed':seed,'attempt':attempt,'status':'rejected','reason':str(exc)},None


def parallel_candidates(p,slots,old_layouts):
    # Workers may evaluate ahead, but only the first sixteen valid cases in the
    # predeclared seed/attempt order enter the ranking and selection log.
    specs=p['arrangements']; pairs=[(1,1),(1,-1),(-1,1),(-1,-1)]
    sign_tables=[{stop:pairs[(5*i+j)%4] for j,stop in enumerate(p['stops'])} for i in range(len(specs))]
    tasks=[[(seed,attempt) for seed in range(spec['seed_start'],spec['seed_start']+p['seeds_per_arrangement']) for attempt in range(p['attempts_per_seed'])] for spec in specs]
    submitted=[0]*len(specs);committed=[0]*len(specs);buffers=[{} for _ in specs];logs=[[] for _ in specs];candidates=[[] for _ in specs];finished=[False]*len(specs)
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        pending={}
        while not all(finished):
            for i,spec in enumerate(specs):
                while not finished[i] and submitted[i]<len(tasks[i]) and submitted[i]-committed[i]<3:
                    seq=submitted[i];seed,attempt=tasks[i][seq];submitted[i]+=1
                    future=pool.submit(_evaluate,(p,spec,slots[i],seed,attempt,sign_tables[i],old_layouts));pending[future]=(i,seq)
            if not pending:raise RuntimeError('parallel candidate scheduler stalled')
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                i,seq=pending.pop(future); result=future.result()
                if not finished[i]:buffers[i][seq]=result
            for i in range(len(specs)):
                while not finished[i] and committed[i] in buffers[i]:
                    log,candidate=buffers[i].pop(committed[i]);committed[i]+=1;logs[i].append(log)
                    if candidate is not None:candidates[i].append(candidate)
                    if len(candidates[i])==p['valid_candidates_to_rank'] or committed[i]==len(tasks[i]):finished[i]=True
            for future,(i,_) in list(pending.items()):
                if finished[i] and future.cancel():pending.pop(future)
    return sign_tables,logs,candidates


def generate(out,p):
    if platform.python_version()!='3.13.5' or np.__version__!='2.5.3':raise RuntimeError('requires Python 3.13.5 and NumPy 2.5.3')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    seeds,layouts,inventory=prior_inventory(p);validate(p,seeds)
    source_files=[__file__,PROTOCOL,B.__file__,C.__file__,ROOT/'sim/arrange.py',ROOT/'sim/rig.py',ROOT/'sim/tabletop.py',B.PROTOCOL,C.PROTOCOL,*[ROOT/x for x in B.CONFIGS.values()]]
    provenance={'study_id':p['study_id'],'protocol_sha256':A.sha256_file(PROTOCOL),'source_sha256':{str(Path(x).resolve()):A.sha256_file(x) for x in source_files},
                'python':platform.python_version(),'numpy':np.__version__,'historical_check':inventory,
                'execution':'parallel speculative geometry evaluations; first 16 valid cases in declared order only'}
    out.mkdir(parents=True);(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    manifest={'study_id':p['study_id'],'split':p['capture_split'],'stops':p['stops'],'states':p['states'],'levels':[l['id'] for l in p['levels']],
              'lighting':p['lighting'],'reference':'nominal_clear_present','arrangements':[]}
    nominal={};used_targets={'shelf':set(),'tabletop':set()};slots=[]
    for spec in p['arrangements']:
        ctx=A.load_context(B.CONFIGS[spec['environment']]);slot=next(s for s in A.slot_list(ctx) if A.class_key(s['family'],s['m16_side'])==spec['class'] and s['target'] not in used_targets[spec['environment']])
        used_targets[spec['environment']].add(slot['target']);slots.append(slot)
    sign_tables,attempt_logs,all_candidates=parallel_candidates(p,slots,layouts)
    logs=[]
    for i,spec in enumerate(p['arrangements']):
        candidates=all_candidates[i]
        if not candidates:raise RuntimeError(f"no valid candidate for {spec['base_id']}")
        score,seed,attempt,arr,poses,digest=sorted(candidates,key=lambda x:(-x[0][0],-x[0][1],-x[0][2],x[1],x[2]))[0]
        if digest in layouts:raise ValueError('cross-arrangement layout duplicate')
        layouts.add(digest);log={'base_id':spec['base_id'],'slot':slots[i],'signs':sign_tables[i],'attempts':attempt_logs[i],
            'selected':{'seed':seed,'attempt':attempt,'rank':list(score),'candidate_count':len(candidates),'layout_sha256':digest,'enrichment_pass':bool(score[0])}};logs.append(log)
        item={'base_id':spec['base_id'],'environment':spec['environment'],'seed':seed,'levels':{}}
        nominal[spec['base_id']]={'environment':spec['environment'],'seed':seed,'cameras':copy.deepcopy(arr['scene']['cameras'])}
        for level in p['levels']:
            rec=copy.deepcopy(arr);rec['base_id']=spec['base_id'];rec['level']=level['id'];rec['arrangement_id']=spec['base_id']+level['id'];rec['study_id']=p['study_id']
            rec['stress_provenance']={'protocol_sha256':A.sha256_file(PROTOCOL),'builder_sha256':A.sha256_file(__file__)}
            stops,cameras=poses[level['id']];rec['revisit']={'schema':'cmpe249.stress-revisit.v1','stops':stops,'level':level};rec['scene']['revisit_cameras']=cameras
            rec['arrangement_sha256']=A.arrangement_hash(rec);path=out/(rec['arrangement_id']+'.json');path.write_text(json.dumps(rec,indent=2)+'\n')
            item['levels'][level['id']]={'path':path.name,'sha256':A.sha256_file(path)}
        manifest['arrangements'].append(item)
        (out/'generation_log.json').write_text(json.dumps(logs,indent=2)+'\n')
    (out/'nominal_cameras.json').write_text(json.dumps(nominal,indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args();generate(args.out,json.loads(PROTOCOL.read_text()))
