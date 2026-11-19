"""Prepare the prespecified archived calibration controls without predictions."""
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path('runs/2026-11-18-paper-audit-v1/scoring/controls')
ROOT = Path('/mnt/bench-data/datasets/cmpe249-active-verification/cross-sim-dev-20261117/qualification')
LEDGER = Path('runs/2026-11-16-support-audit-v1/ledger.json')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    items = []
    rows = json.loads(LEDGER.read_text())
    seen = set()
    for row in rows:
        key = (row['arrangement'], row['stop'])
        if key in seen: continue
        seen.add(key)
        path = Path(row['source_paths']['reference'])
        if sha(path) != row['source_sha256']['reference']: raise ValueError('ledger source mismatch')
        items.append((f'same-{key[0]}-{key[1]}', path, path, row['reference_box'], 'PRESENT', 'same'))
    if len(seen) != 36: raise ValueError(f'expected 36 unique references, got {len(seen)}')
    for engine in ('isaac-v1','mujoco-v1'):
        for layout in ('CSQS01','CSQT01'):
            for stop in range(1,6):
                refroot = ROOT / engine / layout / 'D0'
                mask = np.asarray(Image.open(refroot/'masks'/f'target_reference_P{stop}.png').convert('L')) != 0
                ys,xs = np.where(mask)
                if not len(xs): raise ValueError('empty target mask')
                box = [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1]
                ref = refroot/'rgb'/f'reference_P{stop}.png'
                for level in ('D0','D6'):
                    for label in ('present','removed'):
                        ident = f'{engine}-{layout}-{level}-P{stop}-{label}'
                        query = ROOT/engine/layout/level/'rgb'/f'clear_{label}_P{stop}.png'
                        items.append((ident, ref, query, box, label.upper(), engine))
    if len(items) != 116: raise ValueError('wrong inventory size')
    public=[]; private=[]; sources=[]
    for n,(name,ref,query,box,label,engine) in enumerate(items):
        ident = f'C{n:04d}'
        paths=[]
        for role,src in (('reference',ref),('query',query)):
            with Image.open(src) as im:
                w,h=im.size
                if (w,h)!=(2304,1296): raise ValueError('unexpected original dimensions')
                dst = OUT/f'{ident}-{role}.png'
                im.convert('RGB').resize((1200,675),Image.LANCZOS).save(dst)
            paths.append(dst)
            sources.append({'id':ident,'role':role,'path':str(src),'sha256':sha(src)})
        scaled = [math.floor(box[0]*1200/2304),math.floor(box[1]*675/1296),math.ceil(box[2]*1200/2304),math.ceil(box[3]*675/1296)]
        public.append({'id':ident,'reference':paths[0].name,'query':paths[1].name,'box_xyxy':scaled,
                       'reference_sha256':sha(paths[0]),'query_sha256':sha(paths[1])})
        private.append({'id':ident,'name':name,'expected':label,'engine':engine})
    random.Random(20261118).shuffle(public)
    (OUT/'cases.json').write_text(json.dumps(public,indent=2)+'\n')
    (OUT/'private.json').write_text(json.dumps(private,indent=2)+'\n')
    (OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('cases',len(public),'cases_sha256',sha(OUT/'cases.json'),'private_sha256',sha(OUT/'private.json'),'sources_sha256',sha(OUT/'sources.json'))
if __name__=='__main__': main()
