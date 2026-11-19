"""Check native mask encoding and target presence in every intervention."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

STATES=('clear_present','clear_removed','placed_present','placed_removed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--select',nargs='+',required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());rows=[];errors=[]
    if not set(a.select) <= {x['base_id'] for x in manifest['arrangements']}:raise ValueError('unknown selection')
    for item in manifest['arrangements']:
        if item['base_id'] not in a.select:continue
        for level in manifest['levels']:
            folder=a.root/item['base_id']/level
            for stop in manifest['stops']:
                counts={}
                labels=list(STATES)+(['reference'] if level=='D0' else [])
                if manifest['split']=='qualification':labels += [f'repeat{n}_'+s for n in (1,2) for s in STATES]+['distractor_hidden_'+s for s in STATES]
                for label in labels:
                    for role in ('target','blocker'):
                        path=folder/'masks'/f'{role}_{label}_{stop}.png'
                        im=Image.open(path);x=np.asarray(im)
                        if im.size!=(2304,1296) or im.mode not in ('L','1') or not np.isin(x,[0,1,255]).all():errors.append(str(path)+': invalid binary encoding')
                        count=int(np.count_nonzero(x));counts[role+'_'+label]=count
                        if role=='target' and label.endswith('_removed') and count:errors.append(str(path)+': removed target mask is nonempty')
                    with Image.open(folder/'rgb'/f'{label}_{stop}.png') as im:
                        if im.size!=(2304,1296) or im.mode!='RGB':errors.append(str(im.filename)+': invalid RGB encoding')
                alone=Image.open(folder/'masks'/f'alone_{stop}.png'); x=np.asarray(alone)
                if alone.size!=(2304,1296) or not np.isin(x,[0,1,255]).all():errors.append('invalid alone mask')
                counts['isolated_target']=int(np.count_nonzero(x));rows.append({'base_id':item['base_id'],'level':level,'stop':stop,'pixels':counts})
    a.output.write_text(json.dumps({'rows':rows,'errors':errors},indent=2)+'\n')
    if errors:raise SystemExit('mask or image checks failed')
