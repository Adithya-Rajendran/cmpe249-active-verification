"""Transfer and verify completed Isaac layouts for score-free staging.

Set CMPE249_CAPTURE_HOST to the configured SSH host for the capture machine.
Set CMPE249_CAPTURE_PYTHON to its Python executable if it differs from python3.
Run this module with the local Python environment that supplies NumPy and Pillow.
"""
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OP=ROOT/'runs/2026-11-18-paper-audit-v1/operator'
REMOTE='/mnt/bench-data/datasets/cmpe249-active-verification/'
BASE=REMOTE+'paper-audit-v1-20261118/'
SSH=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
HOST=os.environ.get('CMPE249_CAPTURE_HOST','gb10')
REMOTE_PYTHON=os.environ.get('CMPE249_CAPTURE_PYTHON','python3')
MANIFEST=OP/'evaluation-cohort/manifest.json'
DIR=OP/'evaluation/batches'
PYTHON=sys.executable

def run(args,**kw):subprocess.run([str(x) for x in args],check=True,cwd=ROOT,**kw)
def mapped(x):
    if x.startswith(BASE+'scratch-source/'):return str(ROOT/x[len(BASE+'scratch-source/'):])
    if x.startswith(BASE):return str(OP/x[len(BASE):])
    if x.startswith(REMOTE+'cross-sim-dev-20261117/assets/models/'):return str(OP/'assets/models'/x.split('/models/',1)[1])
    raise ValueError(x)
def stage(name):
    if name not in {x['base_id'] for x in json.loads(MANIFEST.read_text())['arrangements']}:raise ValueError('unknown layout')
    local=OP/'isaac-evaluation'/name
    if local.exists():raise FileExistsError(local)
    remote=BASE+'isaac-evaluation/'+name
    run(['scp',*SSH,'-r',HOST+':'+remote,OP/'isaac-evaluation'])
    receipt='isaac-evaluation-geometry-'+name+'.json'
    command=('cd '+BASE+'scratch-source && '+
      shlex.quote(REMOTE_PYTHON)+' -m development.paper_audit.verify_geometry_batch '+
      '--manifest ../evaluation-cohort/manifest.json --root ../isaac-evaluation --assets '+REMOTE+'cross-sim-dev-20261117/assets/models --select '+name+' --output ../'+receipt)
    run(['ssh',*SSH,HOST,command]);run(['scp',*SSH,HOST+':'+BASE+receipt,DIR])
    doc=json.loads((DIR/receipt).read_text());doc['manifest']['path']=mapped(doc['manifest']['path'])
    for key in ('sources','inputs_sha256'):doc[key]={mapped(k):v for k,v in doc[key].items()}
    files={doc['manifest']['path']:doc['manifest']['sha256'],**doc['sources'],**doc['inputs_sha256']}
    for path,digest in files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('geometry input hash mismatch: '+path)
    (DIR/('isaac-geometry-local-'+name+'.json')).write_text(json.dumps(doc,indent=2)+'\n')
    run([PYTHON,'-m','development.paper_audit.private_manifest_batch','--cohort',MANIFEST,'--root',OP/'isaac-evaluation','--engine','isaac','--select',name,'--output',DIR/('isaac-private-'+name+'.json')])
    run([PYTHON,'-m','development.paper_audit.verify_masks_batch','--manifest',MANIFEST,'--root',OP/'isaac-evaluation','--select',name,'--output',DIR/('isaac-masks-'+name+'.json')])
    print(name,'verified and staged',flush=True)

if __name__=='__main__':
    for name in sys.argv[1:]:stage(name)
