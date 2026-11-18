"""Preflight and run the four approved full-cohort image-evidence analyses."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'full-scoring-source-v1'
PREPARED = ROOT / 'full-prepared-v1'
OUTPUT = ROOT / 'full-evidence-v1'
EXPECTED = {'runner': 'dfe8a35fa514aaed80efd84f0db8a51ead4b822481746c6d4294b64cae1234d9',
            'guard': 'd55e63527c675b852b2115ca7fb2b09678e5d78653857d73e72cfea3ba023e1d'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preflight():
    runtime = json.loads((ROOT / 'full-scoring-runtime-preflight.json').read_text())
    preparation = json.loads((PREPARED / 'preparation.json').read_text())
    review = json.loads((ROOT / 'full-review/cohort-validation.json').read_text())
    prepared_audit_path = ROOT / 'full-main-review-v1/prepared_audit.json'
    prepared_audit = json.loads(prepared_audit_path.read_text())
    if prepared_audit['status'] != 'PASS' or prepared_audit['input_count'] != 1800:
        raise ValueError('independent prepared-input audit incomplete')
    if prepared_audit['preparation_sha256'] != sha(PREPARED / 'preparation.json'):
        raise ValueError('prepared-input audit binding mismatch')
    if runtime['runner_sha256'] != EXPECTED['runner'] or sha(SOURCE / 'development/stress/run_evidence.py') != EXPECTED['runner']:
        raise ValueError('runner mismatch')
    if runtime['guard_sha256'] != EXPECTED['guard'] or sha(ROOT / 'offline_exec.py') != EXPECTED['guard']:
        raise ValueError('offline guard mismatch')
    if review['status'] != 'PASS' or review['arrangements'] != 120 or review['rgb_count'] != 2400 or review['mask_count'] != 5400:
        raise ValueError('full render validation incomplete')
    if preparation['study_id'] != 'support-stress-dev-v1' or preparation['scope'] != 'full' or preparation['count'] != 1800:
        raise ValueError('full preparation incomplete')
    if preparation['protocol_sha256'] != runtime['protocol_sha256']:
        raise ValueError('protocol binding mismatch')
    for size in ('primary', 'native'):
        jobs = PREPARED / 'verifier' / size / 'jobs.json'
        if sha(jobs) != preparation['jobs_sha256'][size] or len(json.loads(jobs.read_text())) != 1800:
            raise ValueError(f'prepared jobs mismatch: {size}')
    return {'study_id': preparation['study_id'], 'source': str(SOURCE), 'prepared': str(PREPARED),
            'runtime_preflight_sha256': sha(ROOT / 'full-scoring-runtime-preflight.json'),
            'preparation_sha256': sha(PREPARED / 'preparation.json'),
            'full_review_sha256': sha(ROOT / 'full-review/cohort-validation.json'),
            'prepared_audit_sha256': sha(prepared_audit_path),
            'driver_sha256': sha(__file__), 'runner_sha256': EXPECTED['runner'], 'guard_sha256': EXPECTED['guard'],
            'analyses': ['primary', 'native', 'native_normalized_guard', 'relaxed_guard'],
            'records_per_analysis': 1800, 'cwd': str(SOURCE), 'PYTHONPATH': str(SOURCE),
            'python': sys.executable, 'device': 'cuda'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--capsule-out')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    binding = preflight()
    if a.capsule_out:
        with Path(a.capsule_out).open('x') as f:
            json.dump(binding, f, indent=2)
            f.write('\n')
    if not a.execute:
        print(json.dumps({'valid': True, 'would_start_gpu': False, 'count': 1800}))
        return
    OUTPUT.mkdir(exist_ok=False)
    with (OUTPUT / 'driver-capsule.json').open('x') as f:
        json.dump(binding, f, indent=2)
        f.write('\n')
    for analysis in binding['analyses']:
        cmd = [sys.executable, str(ROOT / 'offline_exec.py'), '--guard-record', str(OUTPUT / (analysis + '-guard.json')),
               str(SOURCE / 'development/stress/run_evidence.py'), '--prepared', str(PREPARED),
               '--out', str(OUTPUT / analysis), '--analysis', analysis, '--device', 'cuda']
        env = dict(os.environ, PYTHONPATH=str(SOURCE))
        start, tick = datetime.now(timezone.utc).isoformat(), time.monotonic()
        with (OUTPUT / (analysis + '.log')).open('x') as log:
            proc = subprocess.run(cmd, cwd=SOURCE, env=env, stdout=log, stderr=subprocess.STDOUT)
        entry = {'analysis': analysis, 'command': cmd, 'cwd': str(SOURCE), 'PYTHONPATH': str(SOURCE),
                 'start_utc': start, 'elapsed_s': time.monotonic() - tick, 'exit_code': proc.returncode}
        with (OUTPUT / (analysis + '-execution.json')).open('x') as f:
            json.dump(entry, f, indent=2)
            f.write('\n')
        print(json.dumps(entry), flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)
        if len(list((OUTPUT / analysis / 'records').glob('*.json'))) != 1800:
            raise ValueError(f'{analysis} produced an incomplete record set')


if __name__ == '__main__':
    main()
