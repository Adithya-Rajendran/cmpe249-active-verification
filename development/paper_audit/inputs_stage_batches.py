"""Stage each fully verified Isaac evaluation layout as its receipts arrive."""
import json
import time
from pathlib import Path
from development.paper_audit.evaluate_calibration import check_geometry
from development.paper_audit.inputs_build import verify_entries
from development.paper_audit.inputs_stage import stage

ROOT=Path('runs/2026-11-18-paper-audit-v1/operator/evaluation')
BATCHES=ROOT/'batches'


def main():
    pending={f'PAM{i:02}' for i in range(1,17)}
    while pending:
        for name in sorted(pending.copy()):
            manifest=BATCHES/f'isaac-private-{name}.json'; geometry=BATCHES/f'isaac-geometry-local-{name}.json'; masks=BATCHES/f'isaac-masks-{name}.json'
            if not all(p.is_file() for p in (manifest,geometry,masks)): continue
            try:
                capture=json.loads(manifest.read_text());audit=json.loads(masks.read_text());json.loads(geometry.read_text())
            except json.JSONDecodeError:
                continue
            rows=capture['rows']; wanted={(level,f'P{i}') for level in ('D0','D6') for i in range(1,6)}
            if len(rows)!=10 or {(r['level'],r['stop']) for r in rows}!=wanted or any(r['arrangement']!=name or r['engine']!='isaac' for r in rows): raise ValueError('incomplete or incorrect capture '+name)
            if audit['errors'] or len(audit['rows'])!=10 or {(r['level'],r['stop']) for r in audit['rows']}!=wanted or any(r['base_id']!=name for r in audit['rows']):raise ValueError('incomplete or failed native audit '+name)
            verify_entries(capture);observed=check_geometry([geometry])
            if any(observed.get(str(Path(r['observed']['path']).resolve()))!=r['observed']['sha256'] for r in rows): raise ValueError('geometry binding mismatch '+name)
            dest=ROOT/f'isaac-export-{name}'
            if not dest.exists():stage([manifest],dest)
            cache=json.loads((dest/'cache.json').read_text())
            if not cache['sources']:raise ValueError('empty export cache '+name)
            pending.remove(name)
            print('staged',name,'remaining',len(pending),flush=True)
        if pending: time.sleep(15)
    print('all Isaac layouts staged',flush=True)

if __name__=='__main__': main()
