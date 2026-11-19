"""Check physical receipts and distractor-only image changes without model scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

STATES=('clear_present','clear_removed','placed_present','placed_removed')

def verify(folder,engine):
    observed=json.loads((folder/'observed.json').read_text());rows=[]
    for state in STATES:
        for prefix in ('repeat1_','repeat2_','distractor_hidden_'):
            name=prefix+state
            if engine=='mujoco':
                baseline=observed['states'][state];control=observed['states'][name]
                target_blocker=all(baseline[k]==control[k] for k in ('target_enabled','blocker','cameras','static_bounds'))
                for obj in baseline['objects']:
                    if prefix=='distractor_hidden_' and obj=='distractor_0':
                        b=dict(baseline['objects'][obj]);c=dict(control['objects'][obj]); bg=b.pop('enabled_geom_groups');cg=c.pop('enabled_geom_groups')
                        target_blocker &= b==c and all(x==5 for x in cg) and all(x!=5 for x in bg)
                    else:target_blocker &= baseline['objects'][obj]==control['objects'][obj]
            else:
                baseline=observed['usd_per_state'][state];control=observed['usd_per_state'][name]
                target_blocker=True
                for path,b in baseline.items():
                    c=control[path]
                    if prefix=='distractor_hidden_' and (path=='/World/Objects/distractor_0' or path.startswith('/World/Objects/distractor_0/')):
                        b=dict(b);c=dict(c); b.pop('visibility'); v=c.pop('visibility');target_blocker &= b==c and v=='invisible'
                    else:target_blocker &= b==c
            for stop in ('P1','P2','P3','P4','P5'):
                a=np.asarray(Image.open(folder/'rgb'/f'{state}_{stop}.png'),dtype=np.int16)
                b=np.asarray(Image.open(folder/'rgb'/f'{name}_{stop}.png'),dtype=np.int16)
                masks={}
                for role in ('target','blocker'):
                    x=np.asarray(Image.open(folder/'masks'/f'{role}_{state}_{stop}.png'))
                    y=np.asarray(Image.open(folder/'masks'/f'{role}_{name}_{stop}.png'))
                    masks[role+'_mask_equal']=bool(np.array_equal(x,y))
                diff=np.abs(a-b)
                rows.append({'state':state,'control':name,'stop':stop,'state_geometry_matches':bool(target_blocker),
                    'changed_channel_values':int(np.count_nonzero(diff)),'rgb_mae':float(diff.mean()),**masks})
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--engine',choices=['mujoco','isaac'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=[]
    for folder in sorted(a.root.glob('*/D0')):
        for row in verify(folder,a.engine):rows.append({'arrangement':folder.parent.name,**row})
    a.output.write_text(json.dumps({'engine':a.engine,'rows':rows},indent=2)+'\n')
    if not rows or not all(r['state_geometry_matches'] for r in rows):raise SystemExit('state mismatch or no captures')
