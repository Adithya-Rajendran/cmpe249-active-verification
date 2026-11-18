"""Export simulator-mask proxy labels for evaluation, never for verifier inputs."""
from __future__ import annotations
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.alignment_diagnostics import check_frozen
from development.stress.prepare_inputs import PROTOCOL, STUDY, json_hash, load, sha

FIELDS = ['id', 'label', 'cue', 'reviewer', 'source', 'input_size', 'reference_sha256', 'query_sha256', 'box_sha256']


def classify(physical, visible, obstruction):
    for value in (visible, obstruction):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('invalid mask measurement')
    if physical not in {'PRESENT', 'REMOVED'}:
        raise ValueError('invalid physical state')
    if physical == 'PRESENT' and visible is not None and visible >= .90:
        return 'PRESENT', f'Mask visible fraction {visible} >= 0.90; physical intervention has target present.'
    if physical == 'REMOVED' and obstruction is not None and obstruction < .10:
        return 'REMOVED', f'Mask region obstruction {obstruction} < 0.10; physical intervention has target removed.'
    return 'UNKNOWN', (f'Mask visible fraction {visible}, region obstruction {obstruction}; proxy cutoff not met. '
                       'UNKNOWN does not establish visual unanswerability.')


def export(prepared, cohort_path, output, size):
    prepared, cohort_path, output = map(Path, (prepared, cohort_path, output))
    if output.exists():
        raise FileExistsError(output)
    if size not in {'primary', 'native'}:
        raise ValueError('unknown input size')
    pre, cohort, proto = load(prepared / 'preparation.json'), load(cohort_path), load(PROTOCOL)
    if pre['study_id'] != STUDY or cohort['study_id'] != STUDY or pre['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('study or protocol mismatch')
    if pre['cohort_sha256'] != sha(cohort_path) or pre['frozen_hashes'] != check_frozen():
        raise ValueError('cohort or frozen source mismatch')
    ledger_path, jobs_path = prepared / 'ledger.json', prepared / 'verifier' / size / 'jobs.json'
    if sha(ledger_path) != pre['ledger_sha256'] or sha(jobs_path) != pre['jobs_sha256'][size]:
        raise ValueError('preparation inputs mismatch')
    ledger, jobs = load(ledger_path), load(jobs_path)
    by_id = {j['id']: j for j in jobs}
    if len(by_id) != len(jobs) or len(ledger) != pre['count'] or len({r['id'] for r in ledger}) != len(ledger) or set(by_id) != {r['id'] for r in ledger}:
        raise ValueError('incomplete or duplicate prepared inputs')
    cohort_by_id = {r['arrangement_id']: r for r in cohort['records']}
    if len(cohort_by_id) != len(cohort['records']):
        raise ValueError('duplicate cohort arrangement')
    dimensions = proto['primary_size'] if size == 'primary' else proto['render_size']
    source_cache = {}
    rows = []
    for item in ledger:
        rec = cohort_by_id[item['arrangement_id']]
        if any(rec[k] != item[v] for k, v in [('base_id', 'base_id'), ('environment', 'environment'), ('level_id', 'level')]):
            raise ValueError('cohort ledger mismatch')
        label_path = cohort_path.parent / rec['render_directory'] / 'labels.json'
        key = str(label_path)
        if key not in source_cache:
            source_cache[key] = (sha(label_path), load(label_path))
        digest, labels = source_cache[key]
        if digest != item['labels_sha256'] or labels['arrangement_id'] != item['arrangement_id']:
            raise ValueError('source labels hash or identity mismatch')
        phase, stop = item['phase'], item['stop']
        if phase not in proto['query_phases'] or stop not in proto['stops']:
            raise ValueError('unknown phase or stop')
        physical = 'REMOVED' if phase == 'removed' else 'PRESENT'
        visible = labels['visible_fraction'].get(phase, {}).get(stop)
        obstruction = labels['region_obstruction'].get(phase, {}).get(stop)
        if physical != item['physical_state'] or visible != item['visible_fraction'] or obstruction != item['region_obstruction']:
            raise ValueError('source mask or intervention differs from ledger')
        job = by_id[item['id']]
        for name in ('reference', 'query'):
            image = jobs_path.parent / job[name]
            if image.parent != jobs_path.parent or sha(image) != job[name + '_sha256']:
                raise ValueError('prepared image hash mismatch')
        label, cue = classify(physical, visible, obstruction)
        rows.append({'id': item['id'], 'label': label, 'cue': cue, 'reviewer': 'sim_mask_proxy_v1',
                     'source': 'geometry_proxy', 'input_size': f'{dimensions[0]}x{dimensions[1]}',
                     'reference_sha256': job['reference_sha256'], 'query_sha256': job['query_sha256'],
                     'box_sha256': json_hash(job['box'])})
    with output.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True); parser.add_argument('--cohort', required=True)
    parser.add_argument('--out', required=True); parser.add_argument('--size', required=True, choices=['primary', 'native'])
    args = parser.parse_args()
    export(args.prepared, args.cohort, args.out, args.size)
