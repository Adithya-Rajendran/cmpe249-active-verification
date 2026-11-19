"""Geometry-only throwaway fixtures. Never use these seeds for study capture."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sim import arrange as A
from development.stress import build_cohort as B

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/2026-11-18-paper-audit-v1/operator/geometry-feasibility.json'

def task(row):
    env, obj, family, seed = row
    config=json.loads((ROOT/B.CONFIGS[env]).read_text())
    config['target']['objects']=list(dict.fromkeys(config['target']['objects']+[7,8,18]))
    config['slots']['target_assignment']['classes'][family]=list(dict.fromkeys(config['slots']['target_assignment']['classes'][family]+[7,8,18]))
    ctx=A.load_context(config)
    try:
        arr=A.generate(seed,family,None,0,ctx=ctx,target=obj)
        target=A.body_of(arr['objects']['target'],ctx=ctx)
        blocker=A.body_of(arr['objects']['occluder'],'placed',ctx=ctx)
        cov=[round(float(A.coverage(c,target,blocker)),4) for c in ctx.cams]
        return {'env':env,'target':obj,'family':family,'seed':seed,'ok':True,'coverage':cov}
    except (A.PlacementFailure,ValueError,RuntimeError) as exc:
        return {'env':env,'target':obj,'family':family,'seed':seed,'ok':False,'reason':str(exc)}

if __name__=='__main__':
    rows=[(env,obj,fam,7400+i*100+j*20+k*5+n) for i,env in enumerate(['shelf','tabletop']) for j,obj in enumerate([8,18,7]) for k,fam in enumerate(['IND28','white_box']) for n in range(3)]
    with ProcessPoolExecutor(max_workers=4) as pool: result=list(pool.map(task,rows))
    OUT.write_text(json.dumps({'purpose':'throwaway geometry only; not study inputs','reserved_fixture_seeds':'7400-7549','results':result},indent=2)+'\n')
    for env in ['shelf','tabletop']:
        for obj in [8,18,7]:
            for fam in ['IND28','white_box']:
                subset=[r for r in result if r['env']==env and r['target']==obj and r['family']==fam]
                print(env,obj,fam,sum(x['ok'] for x in subset),'/',len(subset),flush=True)
