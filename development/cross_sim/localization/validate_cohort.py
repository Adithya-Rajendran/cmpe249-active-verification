"""Independently verify cohort rankings, layouts, cameras, and reservations."""
import copy, json, platform
from pathlib import Path
import numpy as np
from sim import arrange as A
from development.stress import build_cohort as B
from development.cross_sim import cohort as C
from development.cross_sim.localization import cohort as G

root=Path(__file__).resolve().parents[3]
out=root/'runs/2026-11-18-label-localization-v1/cohort'
p=json.loads(G.PROTOCOL.read_text()); manifest=json.loads((out/'manifest.json').read_text()); logs=json.loads((out/'generation_log.json').read_text())
assert platform.python_version()=='3.13.5' and np.__version__=='2.5.3'
old_seeds,old_layouts,inventory=G.prior_inventory(p); G.validate(p,old_seeds)
seen=set(); results=[]; targets={'shelf':set(),'tabletop':set()}; signs_cycle=[(1,1),(1,-1),(-1,1),(-1,-1)]
for i,(spec,item,log) in enumerate(zip(p['arrangements'],manifest['arrangements'],logs,strict=True)):
    ctx=A.load_context(B.CONFIGS[spec['environment']]); slot=next(s for s in A.slot_list(ctx) if A.class_key(s['family'],s['m16_side'])==spec['class'] and s['target'] not in targets[spec['environment']]); targets[spec['environment']].add(slot['target']); assert slot==log['slot']
    signs={name:signs_cycle[(5*i+j)%4] for j,name in enumerate(p['stops'])}
    assert {k:list(v) for k,v in signs.items()}==log['signs']
    candidates=[]
    expected=[(seed,attempt) for seed in range(spec['seed_start'],spec['seed_start']+32) for attempt in range(21)]
    assert [(x['seed'],x['attempt']) for x in log['attempts']]==expected[:len(log['attempts'])]
    for row in log['attempts']:
        if row['status']!='candidate': continue
        counts=[sum(.1<v<.9 for v in row['coverage'][level].values()) for level in ('D0','D6')]
        screened=[sum(v>=.9 for v in row['coverage'][level].values()) for level in ('D0','D6')]
        rank=(int(all(counts)),sum(a>0 and b>0 for a,b in zip(counts,screened)),sum(counts))
        assert list(rank)==row['rank']; candidates.append((tuple(-v for v in rank)+(row['seed'],row['attempt']),row))
    assert len(candidates)==16 or len(log['attempts'])==len(expected)
    selected=min(candidates)[1]; assert selected['seed']==item['seed']==log['selected']['seed'] and selected['attempt']==log['selected']['attempt']
    arr=A.generate(selected['seed'],slot['family'],slot['m16_side'],selected['attempt'],ctx=ctx,target=slot['target']); digest=C.layout_hash(arr)
    assert digest==selected['layout_sha256'] and digest not in old_layouts and digest not in seen; seen.add(digest)
    target=A.body_of(arr['objects']['target'],ctx=ctx); placed=A.body_of(arr['objects']['occluder'],'placed',ctx); poses=B.motions(ctx,target,signs,p['levels'])
    for level in p['levels']:
        name=level['id']; entry=item['levels'][name]; path=out/entry['path']; assert A.sha256_file(path)==entry['sha256']; saved=json.loads(path.read_text()); stops,cameras=poses[name]
        assert saved['scene']['revisit_cameras']==cameras and saved['revisit']['stops']==stops and saved['objects']==arr['objects'] and saved['scene']['cameras']==arr['scene']['cameras']
        assert saved['arrangement_sha256']==A.arrangement_hash(saved)
        assert all(r['margin_min_unrounded_px']>=float(ctx.config['revisit']['min_margin_px']) for r in stops.values())
        actual={cam.name:float(A.coverage(A.perturb_camera(cam,stops[cam.name]['delta']),target,placed)) for cam in ctx.cams}; assert actual==selected['coverage'][name]
    results.append({'base_id':spec['base_id'],'seed':item['seed'],'attempt':selected['attempt'],'rank':selected['rank'],'layout_sha256':digest,'candidate_count':len(candidates)})
validation={'status':'PASS','manifest_sha256':A.sha256_file(out/'manifest.json'),'config_sha256':A.sha256_file(G.PROTOCOL),'generator_sha256':A.sha256_file(G.__file__),'validator_sha256':A.sha256_file(__file__),'generation_log_sha256':A.sha256_file(out/'generation_log.json'),'checks':['all previous declared reservations and historical manifests','candidate order and independently recomputed ranking from logged coverage','selected layout regeneration and historical uniqueness','selected coverage, frame margins, nominal and signed revisit cameras','arrangement and file hashes'],'limitation':'The validator checks logged candidate coverage for ranking and independently recomputes selected coverage. It does not regenerate every rejected or unselected candidate. Screened coverage is a geometry proxy, not a zero-visible-pixel certification.','results':results}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
