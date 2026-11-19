"""Generate the locked study cohort from fresh seeds and source geometry only."""
import argparse
import copy
import json
from pathlib import Path
from sim import arrange as A
from development.stress import build_cohort as B
from development.cross_sim import cohort as C

ROOT=Path(__file__).resolve().parents[2]
DECL=Path(__file__).with_name('capture_declaration.json')

def generate(output, declaration_path, prior=None):
    declaration_path=Path(declaration_path)
    declaration=json.loads(declaration_path.read_text())
    if not declaration.get('design_locked',False):
        raise ValueError('Capture design is not locked')
    out=Path(output)
    if out.exists(): raise FileExistsError(out)
    old_seeds,old_layouts,sources=C.historical(json.loads(C.PROTOCOL.read_text()))
    # Protect every previously declared development reservation, not only used seeds.
    reserved=set(declaration['fixture_seeds'])
    stress=json.loads(B.PROTOCOL.read_text())
    for start in stress['seed_starts'].values():
        reserved.update(range(start,start+12*stress['seeds_per_slot']))
    cross=json.loads(C.PROTOCOL.read_text())
    for split in cross['splits'].values():
        for start in split['seed_starts'].values():
            reserved.update(range(start,start+split['count_per_environment']*cross['seeds_per_arrangement']))
    for protocol in ('support_label_cohort_protocol.json','localization/cohort_protocol.json'):
        prior_protocol=json.loads((ROOT/'development/cross_sim'/protocol).read_text())
        for item in prior_protocol['arrangements']:
            reserved.update(range(item['seed_start'],item['seed_start']+prior_protocol['seeds_per_arrangement']))
    for config in B.CONFIGS.values():
        ctx=A.load_context(ROOT/config); dev=ctx.config['slots']['dev']
        reserved.update(range(dev['primary_seed_start'],dev['primary_seed_start']+len(A.slot_list(ctx))))
        for values in dev['reserve'].values(): reserved.update(values)
    mount=Path('/mnt/bench-data/datasets/cmpe249-active-verification')
    if not mount.is_dir(): raise RuntimeError('historical dataset mount unavailable')
    for name in ('cross-sim-dev-20261117','stress-dev-v1-20261117','support-labels-dev-20261118','label-localization-dev-20261118'):
        for path in (mount/name).rglob('*.json'):
            try: item=json.loads(path.read_bytes())
            except (ValueError,UnicodeError,OSError): continue
            if isinstance(item,dict) and isinstance(item.get('objects'),dict) and type(item.get('seed')) is int:
                old_seeds.add(item['seed']);old_layouts.add(C.layout_hash(item))

    rows=declaration['arrangements']; allocations=[]
    for spec in rows:
        alloc=set(range(spec['seed_start'],spec['seed_start']+declaration['seeds_per_arrangement']))
        if alloc & (old_seeds|reserved): raise ValueError('Reused seed or fixture reservation')
        reserved |= alloc;allocations.append(alloc)
    if prior is not None:
        prior=Path(prior)
        prior_manifest=json.loads(prior.read_text())
        for item in prior_manifest['arrangements']:
            if item['seed'] in old_seeds: raise ValueError('prior cohort reused historical seed')
            for rec in item['levels'].values():
                path=prior.parent/rec['path']
                if A.sha256_file(path)!=rec['sha256']: raise ValueError('prior cohort hash mismatch')
                old_layouts.add(C.layout_hash(json.loads(path.read_text())))
    out.mkdir(parents=True)
    manifest={'study_id':declaration['study_id'],'split':declaration['split'],'stops':declaration['stops'],
              'states':declaration['states'],'levels':declaration[declaration['split']+'_levels'],
              'lighting':'L1','reference':'nominal_clear_present','arrangements':[]}
    logs=[]
    for i,spec in enumerate(rows):
        cfg=json.loads((ROOT/B.CONFIGS[spec['environment']]).read_text())
        target=spec['target'];family=spec['family'];side=spec.get('side')
        class_key=A.class_key(family,side)
        cfg['target']['objects']=list(dict.fromkeys(cfg['target']['objects']+[target]))
        cls=cfg['slots']['target_assignment']['classes'][class_key]
        cfg['slots']['target_assignment']['classes'][class_key]=list(dict.fromkeys(cls+[target]))
        ctx=A.load_context(cfg)
        signs={s:[(1,1),(1,-1),(-1,1),(-1,-1)][(i*5+j)%4] for j,s in enumerate(manifest['stops'])}
        levels=[x for x in declaration['levels'] if x['id'] in manifest['levels']]
        chosen=None
        for seed in sorted(allocations[i]):
            for attempt in range(declaration['attempts_per_seed']):
                try:
                    arr=A.generate(seed,family,side,attempt,ctx=ctx,target=target)
                    body=A.body_of(arr['objects']['target'],ctx=ctx)
                    poses=B.motions(ctx,body,signs,levels)
                    if any(r['margin_min_unrounded_px']<float(ctx.config['revisit']['min_margin_px']) for stops,_ in poses.values() for r in stops.values()):raise ValueError('frame margin')
                    digest=C.layout_hash(arr)
                    if digest in old_layouts:raise ValueError('duplicate layout')
                    chosen=(seed,attempt,arr,poses,digest);break
                except (A.PlacementFailure,ValueError,RuntimeError) as e:logs.append({'base_id':spec['base_id'],'seed':seed,'attempt':attempt,'rejected':str(e)})
            if chosen:break
        if not chosen:
            logs.append({'base_id':spec['base_id'],'status':'missing','reason':'seed reserve exhausted'})
            continue
        seed,attempt,arr,poses,digest=chosen;old_layouts.add(digest)
        item={'base_id':spec['base_id'],'environment':spec['environment'],'seed':seed,'levels':{}}
        for level in levels:
            rec=copy.deepcopy(arr);rec.update(base_id=spec['base_id'],level=level['id'],arrangement_id=spec['base_id']+level['id'],study_id=declaration['study_id'])
            stops,cameras=poses[level['id']];rec['revisit']={'schema':'cmpe249.stress-revisit.v1','stops':stops,'level':level};rec['scene']['revisit_cameras']=cameras
            rec['arrangement_sha256']=A.arrangement_hash(rec);path=out/(rec['arrangement_id']+'.json');path.write_text(json.dumps(rec,indent=2)+'\n')
            item['levels'][level['id']]={'path':path.name,'sha256':A.sha256_file(path)}
        manifest['arrangements'].append(item);logs.append({'base_id':spec['base_id'],'selected_seed':seed,'attempt':attempt,'layout_sha256':digest})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'generation_log.json').write_text(json.dumps(logs,indent=2)+'\n')
    (out/'historical_inventory.json').write_text(json.dumps({'files':sources,'seed_count':len(old_seeds),'declaration_sha256':A.sha256_file(declaration_path)},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--declaration',required=True);p.add_argument('--prior');args=p.parse_args();generate(args.out,args.declaration,args.prior)
