"""Run the four approved smoke evidence analyses sequentially."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
source = root / 'evidence-source-v1'
a = json.loads((root / 'parity-primary-v1.json').read_text())
b = json.loads((root / 'parity-primary-v2.json').read_text())
if a['records'] != b['records'] or len(b['records']) != 2:
    raise ValueError('guarded parity does not match prior parity')
for rec in b['records']:
    for key in ('evidence_comparison', 'alignment_info_comparison'):
        if rec[key]['max_abs_difference'] != 0 or rec[key]['nonnumeric_mismatches']:
            raise ValueError('parity comparison failed')
    if not rec['alignment_ok_equal']:
        raise ValueError('parity alignment decision differs')
out = root / 'evidence-smoke-v2'
out.mkdir(exist_ok=False)
for analysis in ('primary', 'native', 'native_normalized_guard', 'relaxed_guard'):
    guard = out / (analysis + '-guard.json')
    dest = out / analysis
    cmd = [sys.executable, str(root / 'offline_exec.py'), '--guard-record', str(guard),
           str(source / 'development/stress/run_evidence.py'), '--prepared', str(root / 'smoke-prepared-v1'),
           '--out', str(dest), '--analysis', analysis, '--device', 'cuda']
    start = datetime.now(timezone.utc).isoformat()
    t = time.monotonic()
    with (out / (analysis + '.log')).open('x') as log:
        env = dict(os.environ, PYTHONPATH=str(source))
        proc = subprocess.run(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    entry = {'analysis': analysis, 'command': cmd, 'cwd': str(source), 'PYTHONPATH': str(source),
             'start_utc': start, 'exit_code': proc.returncode,
             'elapsed_s': time.monotonic() - t}
    (out / (analysis + '-execution.json')).write_text(json.dumps(entry, indent=2) + '\n')
    print(json.dumps(entry), flush=True)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    files = list((dest / 'records').glob('*.json'))
    if len(files) != 60:
        raise ValueError(f'{analysis}: expected 60 records, got {len(files)}')
