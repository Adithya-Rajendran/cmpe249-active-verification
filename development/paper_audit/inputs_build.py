"""Build neutral scorer inputs from private capture manifests after release."""
import argparse
import hashlib
import json
import math
import secrets
from shutil import copyfile
from pathlib import Path
import numpy as np
import PIL
import platform
from PIL import Image


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def check_release():
    release=Path('runs/2026-11-18-paper-audit-v1/design-release.json')
    data=json.loads(release.read_text())
    for path,digest in data['files'].items():
        if sha(path)!=digest: raise ValueError('design release changed')
    return sha(release)

def verify_entries(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            if sha(value['path']) != value['sha256']: raise ValueError('private input hash mismatch')
        else:
            for v in value.values(): verify_entries(v)
    elif isinstance(value,list):
        for v in value: verify_entries(v)


def box_from_mask(path):
    with Image.open(path) as im:
        if im.size != (2304,1296): raise ValueError('native mask dimensions')
        a=np.asarray(im)
    a=np.any(a[...,:3]!=0,axis=2) if a.ndim==3 else a!=0
    y,x=np.nonzero(a)
    if not len(x): raise ValueError('empty reference mask')
    return [math.floor(x.min()*1200/2304),math.floor(y.min()*675/1296),
            math.ceil((x.max()+1)*1200/2304),math.ceil((y.max()+1)*675/1296)]


def clean(source,dest):
    with Image.open(source) as im:
        if im.size != (2304,1296): raise ValueError('native RGB dimensions')
        im.convert('RGB').resize((1200,675),getattr(Image, "Resampling", Image).LANCZOS).save(dest)


def build(manifests,packet,ledger,caches=()):
    packet=Path(packet); ledger=Path(ledger)
    if packet.resolve() in ledger.resolve().parents: raise ValueError('private ledger inside packet')
    if packet.exists() or ledger.exists(): raise FileExistsError('output exists')
    release_sha=check_release()
    docs=[json.loads(Path(p).read_text()) for p in manifests]
    for doc in docs: verify_entries(doc)
    exports={}
    for cache in caches:
        folder=Path(cache); doc=json.loads((folder/'cache.json').read_text())
        if doc['pillow']!=PIL.__version__ or doc['numpy']!=np.__version__ or doc['python']!=platform.python_version(): raise ValueError('export cache runtime mismatch')
        for digest,entry in doc['sources'].items():
            source=folder/entry['path']
            if source.parent.resolve()!=folder.resolve() or sha(source)!=entry['sha256']: raise ValueError('export cache hash mismatch')
            exports[digest]=source
    def export(record,destination):
        source=exports.get(record['sha256'])
        if source is None: clean(record['path'],destination)
        else: copyfile(source,destination)
    packet.mkdir(parents=True)
    public=[]; private=[]; seen=set()
    for doc in docs:
        for row in doc['rows']:
            for state in ('clear_present','clear_removed','placed_present','placed_removed'):
                key=(row['engine'],row['arrangement'],row['level'],row['stop'],state)
                if key in seen: raise ValueError('duplicate input')
                seen.add(key); uid=secrets.token_hex(12)
                ref=packet/(uid+'-reference.png'); query=packet/(uid+'-query.png')
                export(row['reference']['rgb'],ref); export(row['images'][state]['rgb'],query)
                box=box_from_mask(row['reference']['target_mask']['path'])
                public.append({'id':uid,'reference':ref.name,'query':query.name,'box_xyxy':box,
                               'reference_sha256':sha(ref),'query_sha256':sha(query)})
                private.append({'id':uid,'state':state,'capture':row})
    secrets.SystemRandom().shuffle(public)
    (packet/'cases.json').write_text(json.dumps(public,indent=2)+'\n')
    (packet/'packet-metadata.json').write_text(json.dumps({'python':platform.python_version(),'pillow':PIL.__version__,'numpy':np.__version__,'cases_sha256':sha(packet/'cases.json'),'conversion':'RGB 2304x1296 to 1200x675 Pillow LANCZOS','box':'native floor minimum, ceil exclusive maximum'},indent=2)+'\n')
    ledger.parent.mkdir(parents=True,exist_ok=True)
    ledger.write_text(json.dumps({'design_release_sha256':release_sha,'packet_metadata_sha256':sha(packet/'packet-metadata.json'),'builder_sha256':sha(__file__),'cases_sha256':sha(packet/'cases.json'),'cache_manifests':{str((Path(p)/'cache.json').resolve()):sha(Path(p)/'cache.json') for p in caches},'manifests':{str(Path(p).resolve()):sha(p) for p in manifests},'rows':private},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',action='append',required=True);p.add_argument('--packet',required=True);p.add_argument('--ledger',required=True);p.add_argument('--cache',action='append',default=[]);a=p.parse_args();build(a.manifest,a.packet,a.ledger,a.cache)
