"""Index completed captures for private evaluation without reading predictions."""
import argparse
import hashlib
import json
from pathlib import Path

STATES=('clear_present','clear_removed','placed_present','placed_removed')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def entry(path):
    if not path.is_file():raise FileNotFoundError(path)
    return {'path':str(path.resolve()),'sha256':sha(path)}

def make(cohort,root,engine,calibration,output):
    manifest=json.loads(cohort.read_text());rows=[]
    for arrangement in manifest['arrangements']:
        for level in manifest['levels']:
            folder=root/arrangement['base_id']/level
            marker=folder/('completion.json' if engine=='isaac' else 'complete.json')
            completed=json.loads(marker.read_text());inventory=completed['outputs']
            for name,digest in inventory.items():
                if sha(folder/name)!=digest:raise ValueError(f'completion mismatch: {folder/name}')
            observed=json.loads((folder/'observed.json').read_text())
            for stop in manifest['stops']:
                labels=list(STATES)
                if level=='D0':labels.insert(0,'reference')
                if calibration:labels += [f'repeat{n}_'+s for n in (1,2) for s in STATES]+['distractor_hidden_'+s for s in STATES]
                images={}
                reference_folder=root/arrangement['base_id']/'D0'
                reference={'rgb':entry(reference_folder/'rgb'/f'reference_{stop}.png'),
                           'target_mask':entry(reference_folder/'masks'/f'target_reference_{stop}.png')}
                for label in labels:
                    images[label]={'rgb':entry(folder/'rgb'/f'{label}_{stop}.png'),
                        'target_mask':entry(folder/'masks'/f'target_{label}_{stop}.png'),
                        'blocker_mask':entry(folder/'masks'/f'blocker_{label}_{stop}.png')}
                cams=observed['cameras'];camera=cams['revisit'][stop]
                rows.append({'engine':engine,'arrangement':arrangement['base_id'],'environment':arrangement['environment'],
                    'level':level,'stop':stop,'native_width':camera['width'],'native_height':camera['height'],
                    'camera':camera,'nominal_camera':cams['nominal'][stop],
                    'images':images,'reference':reference,'isolated_target_mask':entry(folder/'masks'/f'alone_{stop}.png'),
                    'observed':entry(folder/'observed.json'),'completion':entry(marker),
                    'distractor_control':'distractor_0 hidden; target and blocker state unchanged' if calibration else None})
    Path(output).write_text(json.dumps({'cohort':entry(cohort),'engine':engine,'rows':rows},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--engine',choices=['isaac','mujoco'],required=True);p.add_argument('--calibration',action='store_true');p.add_argument('--output',required=True);a=p.parse_args();make(a.cohort,a.root,a.engine,a.calibration,a.output)
