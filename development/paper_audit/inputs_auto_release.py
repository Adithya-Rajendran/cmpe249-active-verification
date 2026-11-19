"""Wait for full audited captures, then assemble and independently verify one packet."""
import json
import time
from pathlib import Path
from development.paper_audit.inputs_build import build
from development.paper_audit.inputs_validate import validate
from development.paper_audit.evaluate_calibration import check_geometry

ROOT=Path('runs/2026-11-18-paper-audit-v1')
OP=ROOT/'operator'/'evaluation'


def main():
    manifests=[OP/'mujoco-private-manifest.json',OP/'isaac-private-manifest.json']
    geometries=[OP/'mujoco-evaluation-geometry-local.json',OP/'isaac-evaluation-geometry-local.json']
    audits=[OP/'mujoco-native-mask-audit.json',OP/'isaac-native-mask-audit.json']
    caches=[OP/'mujoco-export-cache']+[OP/f'isaac-export-PAM{i:02}' for i in range(1,17)]
    ready=ROOT/'evaluation'/'scorer-ready.json'
    required=manifests+geometries+audits+[p/'cache.json' for p in caches]
    while True:
        if all(p.is_file() for p in required):
            try:
                for p in required: json.loads(p.read_text())
                break
            except json.JSONDecodeError:
                pass
        time.sleep(15)
    if ready.exists():raise FileExistsError('scorer-ready receipt exists')
    for audit in audits:
        record=json.loads(audit.read_text())
        if record['errors'] or len(record['rows'])!=160: raise ValueError('full native integrity audit failed or incomplete')
    for geometry in geometries:
        if len(check_geometry([geometry]))!=32: raise ValueError('full geometry audit incomplete')
    packet=ROOT/'scoring'/'fresh-packet'; ledger=OP/'private'/'fresh-ledger.json'
    print('all receipts present; building full packet',flush=True)
    build(manifests,packet,ledger,caches)
    print('packet built; independently verifying 1280 cases',flush=True)
    validate(packet,ledger,geometries,ready)
    print('scorer-ready receipt:',ready.resolve(),flush=True)

if __name__=='__main__':main()
