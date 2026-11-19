"""Check a future registered-place run against the archived qualification binding."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--script',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--geometry-checkpoint',type=Path,required=True);p.add_argument('--geometry-source',type=Path,required=True);p.add_argument('--dense-source',type=Path,required=True);a=p.parse_args()
    b=json.loads(a.binding.read_text())
    if sha(a.script)!=b['script_sha256'] or sha(a.spec)!=b['spec_sha256']: raise ValueError('qualified code or prompt changed')
    if a.model.name!=b['model_revision']: raise ValueError('model revision changed')
    if sha(a.geometry_checkpoint)!=b['geometry_checkpoint_sha256']: raise ValueError('geometry checkpoint changed')
    if sha(a.geometry_source)!='e3df38aa229b17fc0b410c07ca4c8a476b590077502acae3cecba3dc220c25b3' or sha(a.dense_source)!='2f07125c77ac2e8369a98e83897d1b5d305d4e48c380afadaa089c58f766aa59': raise ValueError('geometry adapter changed')
    for name,digest in b['model_shards'].items():
        if sha(a.model/name)!=digest: raise ValueError('model shard changed: '+name)
    import torch,transformers
    if torch.__version__!=b['torch'] or transformers.__version__!=b['transformers'] or torch.cuda.get_device_name()!=b['gpu']: raise ValueError('runtime changed')
    print('Qualified script, specification, model shards, geometry checkpoint, and runtime match.')
if __name__=='__main__': main()
