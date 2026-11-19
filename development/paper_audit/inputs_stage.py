"""Prepare hash-addressed RGB exports in a private cache before packet assembly."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import PIL
from development.paper_audit.inputs_build import sha,verify_entries,clean,check_release


def stage(manifests,destination):
    check_release();dest=Path(destination);dest.mkdir(parents=True,exist_ok=False)
    docs=[json.loads(Path(p).read_text()) for p in manifests]
    for doc in docs:verify_entries(doc)
    sources={}
    for doc in docs:
        for row in doc['rows']:
            records=[row['reference']['rgb']]+[row['images'][s]['rgb'] for s in ('clear_present','clear_removed','placed_present','placed_removed')]
            for rec in records: sources[rec['sha256']]=rec['path']
    entries={}
    for digest,path in sources.items():
        target=dest/(digest+'.png');clean(path,target);entries[digest]={'path':target.name,'sha256':sha(target)}
    (dest/'cache.json').write_text(json.dumps({'python':platform.python_version(),'pillow':PIL.__version__,'numpy':np.__version__,'sources':entries},indent=2)+'\n')
    print('prepared exports:',len(entries))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();stage(a.manifest,a.out)
