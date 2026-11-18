"""Build a blinded fixed-sample review pack, with separate manager provenance."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.alignment_diagnostics import check_frozen
from development.stress.prepare_inputs import PROTOCOL, STUDY, case_id, clean_image, json_hash, load, sha, validate_render

PREPARER_SHA = 'c3a91fe2eeb8e143a49e9cac1d1c05d8a8f1c26cdad94bbb34b0276b6145ab2a'
PLAN = ROOT / 'development/stress/label_plan_ai_v1.json'
FIELDS = ['id', 'label', 'cue', 'reviewer', 'source', 'input_size', 'reference_sha256', 'query_sha256', 'box_sha256']
INSTRUCTIONS = ('Identify the remembered object in the red reference rectangle using only this image pair. '
                'PRESENT requires identifying evidence at the remembered region. REMOVED requires a sufficiently exposed '
                'region and distinguishing absence evidence. UNKNOWN means insufficient visual evidence. DISPUTED means '
                'judgment remains unresolved. Fragments, shadows, and reflections count only if they identify the target '
                'or absence. Do not infer from filenames, another query, or state balance.')


def build(cohort_path, sample_path, starter_path, starter_meta_path, reviewer_root, manager_root):
    cohort_path, sample_path, starter_path, starter_meta_path, reviewer_root, manager_root = map(
        lambda p: Path(p).resolve(), (cohort_path, sample_path, starter_path, starter_meta_path, reviewer_root, manager_root))
    if reviewer_root == manager_root or reviewer_root in manager_root.parents or manager_root in reviewer_root.parents:
        raise ValueError('reviewer and manager directories must be separate')
    if reviewer_root.exists() or manager_root.exists():
        raise FileExistsError('destination already exists')
    if sha(ROOT / 'development/stress/prepare_inputs.py') != PREPARER_SHA or pillow_version != '12.3.0':
        raise ValueError('preparer or Pillow differs from full preparation')
    protocol, cohort, sample, meta = load(PROTOCOL), load(cohort_path), load(sample_path), load(starter_meta_path)
    if cohort['study_id'] != STUDY or sample['study_id'] != STUDY or protocol['study_id'] != STUDY:
        raise ValueError('study mismatch')
    if sample['cohort_manifest_sha256'] != sha(cohort_path) or sample['label_plan_sha256'] != sha(PLAN):
        raise ValueError('sample provenance mismatch')
    frozen = check_frozen()
    selected = sample['selected_ids']
    if len(selected) != 180 or len(set(selected)) != 180:
        raise ValueError('sample must contain 180 unique IDs')
    with starter_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    starter = {r['id']: r for r in rows}
    if len(starter) != len(rows) or any(r['source'] != 'ai' or r['label'] not in {'PRESENT','REMOVED','UNKNOWN','DISPUTED'}
                                         or not r['cue'].strip() or not r['reviewer'].strip() for r in rows):
        raise ValueError('invalid starter labels')
    reused = set(selected) & set(starter)
    if len(reused) != 4 or reused != set(sample['overlap_with_smoke_ids']):
        raise ValueError('unexpected starter overlap')
    shown = {(x['id'], x['kind']): x['sha256'] for x in meta['presented_images']}
    if len(shown) != len(meta['presented_images']) or meta['model'] != 'gpt-6.1-terra':
        raise ValueError('invalid starter image provenance')
    records = cohort['records']
    by_base = {(r['base_id'], r['level_id']): r for r in records}
    cases = {}
    for rec in records:
        for stop in protocol['stops']:
            for phase in protocol['query_phases']:
                ident = case_id(rec, stop, phase)
                if ident in cases:
                    raise ValueError('case ID collision')
                cases[ident] = (rec, stop, phase)
    if not set(selected) <= set(cases):
        raise ValueError('sample contains unknown cases')
    selected_new = sorted(set(selected) - reused)
    if len(selected_new) != 176:
        raise ValueError('expected 176 new cases')
    shard = {ident: ('shard-1' if i < 88 else 'shard-2') for i, ident in enumerate(selected_new)}
    reviewer_root.mkdir(parents=True); manager_root.mkdir(parents=True)
    for name in ('shard-1', 'shard-2'):
        (reviewer_root / name / 'images').mkdir(parents=True)
    checked, private, public = {}, [], {'shard-1': [], 'shard-2': []}
    with tempfile.TemporaryDirectory(prefix='stress-ai-audit-') as tmp:
        tmp = Path(tmp)
        for ident in sorted(selected):
            rec, stop, phase = cases[ident]
            ref_rec = by_base[(rec['base_id'], 'D0')]
            for r in (rec, ref_rec):
                if r['arrangement_id'] not in checked:
                    checked[r['arrangement_id']] = validate_render(r, cohort_path.parent, protocol, frozen)
            _, _, folder = checked[rec['arrangement_id']]
            _, ref_lab, ref_folder = checked[ref_rec['arrangement_id']]
            box = ref_lab['reference_box']['L1'][stop]
            if not box or len(box) != 4:
                raise ValueError('missing reference box')
            scaled = [round(x * protocol['primary_size'][j % 2] / protocol['render_size'][j % 2]) for j,x in enumerate(box)]
            ref_source = ref_folder / 'rgb' / f'reference_{stop}_L1.png'
            query_source = folder / 'rgb' / f'{phase}_{stop}_L1.png'
            ref_clean, query_clean, ref_marked = (tmp / (ident + suffix) for suffix in ('_ref.png', '_query.png', '_marked.png'))
            clean_image(ref_source, ref_clean, protocol['primary_size'])
            clean_image(query_source, query_clean, protocol['primary_size'])
            with Image.open(ref_clean) as im:
                overlay = im.copy()
            ImageDraw.Draw(overlay).rectangle(scaled, outline='red', width=4)
            overlay.save(ref_marked)
            item = {'id': ident, 'reference_sha256': sha(ref_clean), 'query_sha256': sha(query_clean),
                    'box': scaled, 'box_sha256': json_hash(scaled), 'input_size': '1200x675',
                    'presented_reference_sha256': sha(ref_marked), 'presented_query_sha256': sha(query_clean),
                    'source_reference_sha256': sha(ref_source), 'source_query_sha256': sha(query_source),
                    'source_labels_sha256': sha(folder / 'labels.json'), 'source_reference_labels_sha256': sha(ref_folder / 'labels.json'),
                    'arrangement_file_sha256': sha(cohort_path.parent / rec['arrangement_file']),
                    'reference_arrangement_file_sha256': sha(cohort_path.parent / ref_rec['arrangement_file'])}
            private.append(item)
            if ident in reused:
                row = starter[ident]
                for key in ('reference_sha256', 'query_sha256', 'box_sha256', 'input_size'):
                    if row[key] != item[key]: raise ValueError('starter verifier input binding mismatch')
                if row['reviewer'] != meta['model']: raise ValueError('starter model mismatch')
                if shown.get((ident, 'reference')) != item['presented_reference_sha256'] or shown.get((ident, 'query')) != item['presented_query_sha256']:
                    raise ValueError('starter presented image binding mismatch')
            else:
                dest = reviewer_root / shard[ident] / 'images'
                ref_name, query_name = ident+'_r.png', ident+'_q.png'
                (dest/ref_name).write_bytes(ref_marked.read_bytes()); (dest/query_name).write_bytes(query_clean.read_bytes())
                public[shard[ident]].append({'id': ident, 'reference': ref_name, 'query': query_name,
                    'presented_reference_sha256': item['presented_reference_sha256'], 'presented_query_sha256': item['presented_query_sha256']})
    for name, items in public.items():
        (reviewer_root/name/'pairs.json').write_text(json.dumps({'instructions': INSTRUCTIONS, 'pairs': items}, indent=2)+'\n')
    with (manager_root/'reused_labels.csv').open('x', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(starter[x] for x in sorted(reused))
    manifest = {'study_id': STUDY, 'cohort_sha256': sha(cohort_path), 'sample_sha256': sha(sample_path),
                'protocol_sha256': sha(PROTOCOL), 'label_plan_sha256': sha(PLAN), 'preparer_sha256': PREPARER_SHA,
                'builder_sha256': sha(__file__), 'starter_labels_sha256': sha(starter_path),
                'starter_meta_sha256': sha(starter_meta_path), 'python': platform.python_version(), 'pillow': pillow_version,
                'case_count': len(private), 'reused_ids': sorted(reused), 'records': private,
                'shard_manifest_sha256': {name: sha(reviewer_root/name/'pairs.json') for name in public},
                'reviewer_model': None, 'reviewer_model_note': 'Record actual model and instructions with each completed review.',
                'acceptance': 'Do not accept audit labels until verify-full succeeds against completed full preparation.'}
    (manager_root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def verify_full(manager_root, full_prepared, output):
    manager_root, full_prepared, output = map(Path, (manager_root, full_prepared, output))
    if output.exists(): raise FileExistsError(output)
    manifest = load(manager_root/'manifest.json'); pre = load(full_prepared/'preparation.json')
    jobs_path = full_prepared/'verifier/primary/jobs.json'
    if pre['study_id'] != STUDY or pre['scope'] != 'full' or pre['protocol_sha256'] != manifest['protocol_sha256'] or pre['cohort_sha256'] != manifest['cohort_sha256']:
        raise ValueError('full preparation provenance mismatch')
    if pre['jobs_sha256']['primary'] != sha(jobs_path) or pre['frozen_hashes'] != check_frozen():
        raise ValueError('full preparation binding mismatch')
    jobs = load(jobs_path); by_id = {j['id']: j for j in jobs}
    if pre['count'] != 1800 or len(jobs) != pre['count'] or len(by_id) != len(jobs): raise ValueError('full jobs are incomplete')
    if manifest['case_count'] != 180 or len(manifest['records']) != 180 or len({x['id'] for x in manifest['records']}) != 180:
        raise ValueError('audit sample is incomplete')
    for item in manifest['records']:
        job = by_id[item['id']]
        if any(job[k] != item[k] for k in ('reference_sha256', 'query_sha256', 'box')):
            raise ValueError(f"full input mismatch: {item['id']}")
        for field in ('reference', 'query'):
            if sha(jobs_path.parent/job[field]) != job[field+'_sha256']:
                raise ValueError('full prepared image hash mismatch')
    output.write_text(json.dumps({'manager_manifest_sha256': sha(manager_root/'manifest.json'),
                                  'full_preparation_sha256': sha(full_prepared/'preparation.json'),
                                  'full_jobs_sha256': sha(jobs_path), 'matched_inputs': len(manifest['records'])}, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command', required=True)
    b=sub.add_parser('build'); b.add_argument('--cohort',required=True); b.add_argument('--sample',required=True)
    b.add_argument('--starter-labels',required=True); b.add_argument('--starter-meta',required=True)
    b.add_argument('--reviewer-root',required=True); b.add_argument('--manager-root',required=True)
    v=sub.add_parser('verify-full'); v.add_argument('--manager-root',required=True); v.add_argument('--full-prepared',required=True); v.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command == 'build': build(a.cohort,a.sample,a.starter_labels,a.starter_meta,a.reviewer_root,a.manager_root)
    else: verify_full(a.manager_root,a.full_prepared,a.out)
