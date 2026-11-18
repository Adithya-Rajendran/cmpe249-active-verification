"""Select private calibration controls and build a blinded qualification pack.

No verifier results or AI judgments are inputs. A visual approval is required before release.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.alignment_diagnostics import check_frozen
from development.stress.prepare_inputs import STUDY, PROTOCOL, json_hash, load, sha

SEED = 'label-qualification-v1|20261117|'
LEVEL_QUOTAS = {'D0': 3, 'D4': 3, 'D6': 2}
UNKNOWN_QUOTAS = {'D0': {'PRESENT': 2, 'REMOVED': 1}, 'D4': {'PRESENT': 1, 'REMOVED': 2},
                  'D6': {'PRESENT': 1, 'REMOVED': 1}}
INSTRUCTIONS = ('Identify the remembered object in the red reference rectangle using only this image pair. '
                'PRESENT requires identifying evidence at the remembered region. REMOVED requires a sufficiently exposed '
                'region and distinguishing absence evidence. UNKNOWN means insufficient visual evidence. DISPUTED means '
                'judgment remains unresolved. Fragments, shadows, and reflections count only if they identify the target '
                'or absence. Do not infer from filenames, another query, or state balance.')


def ranked_diverse(rows):
    """Greedy least-used base within the cell, then SHA-256 rank, then neutral ID."""
    remaining = list(rows)
    counts, result = Counter(), []
    while remaining:
        item = min(remaining, key=lambda r: (counts[r['base_id']], hashlib.sha256((SEED+r['id']).encode()).hexdigest(), r['id']))
        remaining.remove(item); result.append(item['id']); counts[item['base_id']] += 1
    return result


def cohort_inputs(prepared):
    prepared = Path(prepared).resolve(); pre = load(prepared/'preparation.json')
    if pre['study_id'] != STUDY or pre['scope'] != 'full' or pre['count'] != 1800 or pre['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('full preparation study, scope, count or protocol mismatch')
    if pre['frozen_hashes'] != check_frozen(): raise ValueError('frozen sources mismatch')
    ledger_path, jobs_path = prepared/'ledger.json', prepared/'verifier/primary/jobs.json'
    if sha(ledger_path) != pre['ledger_sha256'] or sha(jobs_path) != pre['jobs_sha256']['primary']:
        raise ValueError('ledger or jobs binding mismatch')
    ledger, jobs = load(ledger_path), load(jobs_path)
    by_id = {j['id']: j for j in jobs}
    if len(jobs) != 1800 or len(by_id) != len(jobs) or len(ledger) != 1800 or len({x['id'] for x in ledger}) != 1800 or set(by_id) != {x['id'] for x in ledger}:
        raise ValueError('incomplete or duplicate preparation')
    return prepared, pre, ledger, by_id


def select(prepared, audit_path, smoke_path, plan_path, out):
    out = Path(out)
    if out.exists(): raise FileExistsError(out)
    prepared, pre, ledger, jobs = cohort_inputs(prepared)
    plan, audit = load(plan_path), load(audit_path)
    if plan.get('id') != 'label-qualification-v1' or plan.get('seed') != 20261117 or plan.get('rank_text') != 'label-qualification-v1|20261117|ID' or plan.get('source_prepared') != prepared.name or audit['study_id'] != STUDY or audit['cohort_manifest_sha256'] != pre['cohort_sha256']:
        raise ValueError('plan or audit study/cohort mismatch')
    if plan.get('reference_review', {}).get('maximum_replacements') != 12 or plan.get('input_size') != [1200, 675]:
        raise ValueError('qualification plan mismatch')
    expected_cells = []
    for env in ('shelf', 'tabletop'):
        for target in ('PRESENT', 'REMOVED', 'UNKNOWN'):
            for level in ('D0', 'D4', 'D6'):
                for physical in (('PRESENT', 'REMOVED') if target == 'UNKNOWN' else (target,)):
                    quota = UNKNOWN_QUOTAS[level][physical] if target == 'UNKNOWN' else LEVEL_QUOTAS[level]
                    expected_cells.append({'environment': env, 'label': target, 'level': level, 'physical_state': physical, 'count': quota})
    if plan['cells'] != expected_cells: raise ValueError('qualification cell specification mismatch')
    selected = audit['selected_ids']
    with Path(smoke_path).open(newline='') as f:
        smoke = [r['id'] for r in csv.DictReader(f)]
    if len(selected) != 180 or len(set(selected)) != 180 or len(smoke) != 60 or len(set(smoke)) != 60:
        raise ValueError('audit or smoke exclusion list incomplete')
    excluded = set(selected) | set(smoke)
    if not excluded <= set(jobs): raise ValueError('exclusions include unknown inputs')
    cells = []
    for env in ('shelf', 'tabletop'):
        for target in ('PRESENT', 'REMOVED', 'UNKNOWN'):
            for level in ('D0', 'D4', 'D6'):
                states = ('PRESENT', 'REMOVED') if target == 'UNKNOWN' else (None,)
                for physical in states:
                    quota = UNKNOWN_QUOTAS[level][physical] if physical else LEVEL_QUOTAS[level]
                    eligible=[]
                    for r in ledger:
                        if r['id'] in excluded or r['environment'] != env or r['level'] != level: continue
                        if target == 'PRESENT':
                            match = r['phase'] == 'visible' and r['physical_state'] == 'PRESENT' and r['visible_fraction'] is not None and r['visible_fraction'] >= .99
                        elif target == 'REMOVED':
                            match = r['phase'] == 'removed' and r['physical_state'] == 'REMOVED' and r['region_obstruction'] == 0
                        elif physical == 'PRESENT':
                            match = r['phase'] == 'occluded' and r['physical_state'] == 'PRESENT' and r['visible_fraction'] == 0 and r['region_obstruction'] is not None and r['region_obstruction'] >= .99
                        else:
                            match = r['phase'] == 'removed' and r['physical_state'] == 'REMOVED' and r['region_obstruction'] is not None and r['region_obstruction'] >= .99
                        if match: eligible.append(r)
                    ordered = ranked_diverse(eligible)
                    if len(ordered) < quota: raise ValueError(f'insufficient candidates for {env}/{target}/{level}/{physical}: {len(ordered)} < {quota}')
                    cells.append({'environment': env, 'provisional_class': target, 'level': level,
                                  'physical_subcell': physical, 'quota': quota, 'candidate_ids': ordered,
                                  'proposed_ids': ordered[:quota]})
    proposed = [ident for cell in cells for ident in cell['proposed_ids']]
    if len(proposed) != 48 or len(set(proposed)) != 48: raise ValueError('control quotas not met')
    items = {r['id']: {'source_record': r, 'job': jobs[r['id']]} for r in ledger if r['id'] in {i for c in cells for i in c['candidate_ids']}}
    result = {'study_id': STUDY, 'status': 'provisional; visual confirmation required before exposure',
              'ranking': 'Within each environment/class/level/physical subcell, repeatedly take the remaining candidate from the least-used base, then ascending SHA256 UTF-8 label-qualification-v1|20261117|ID, then ID.',
              'protocol_sha256': sha(PROTOCOL), 'qualification_plan_sha256': sha(plan_path),
              'preparation_sha256': sha(prepared/'preparation.json'), 'jobs_sha256': sha(prepared/'verifier/primary/jobs.json'),
              'ledger_sha256': sha(prepared/'ledger.json'), 'audit_selection_sha256': sha(audit_path), 'smoke_ids_source_sha256': sha(smoke_path),
              'selector_sha256': sha(__file__), 'excluded_unique_ids': len(excluded), 'cells': cells, 'candidates': items}
    out.write_text(json.dumps(result, indent=2)+'\n')


def preview(prepared, selection_path, out, extra_ids=()):
    prepared, _, _, jobs = cohort_inputs(prepared); selection_path, out = Path(selection_path), Path(out)
    if out.exists(): raise FileExistsError(out)
    if pillow_version != '12.3.0': raise ValueError('Pillow 12.3.0 required')
    selection = load(selection_path)
    if selection['preparation_sha256'] != sha(prepared/'preparation.json') or selection['jobs_sha256'] != sha(prepared/'verifier/primary/jobs.json'):
        raise ValueError('selection preparation mismatch')
    ids = {i for c in selection['cells'] for i in c['proposed_ids']} | set(extra_ids)
    if not ids <= set(selection['candidates']): raise ValueError('extra ID is not a ranked candidate')
    out.mkdir(parents=True); (out/'images').mkdir()
    records=[]
    for ident in sorted(ids):
        job=jobs[ident]
        if job != selection['candidates'][ident]['job']: raise ValueError('job changed')
        paths={side: prepared/'verifier/primary'/job[side] for side in ('reference','query')}
        for side,path in paths.items():
            if path.parent != prepared/'verifier/primary' or sha(path) != job[side+'_sha256']:
                raise ValueError('prepared input hash mismatch')
        ref, query=out/'images'/(ident+'_r.png'), out/'images'/(ident+'_q.png')
        with Image.open(paths['reference']) as image:
            if image.size != (1200,675) or image.mode != 'RGB' or image.info: raise ValueError('bad reference input')
            marked=image.copy()
        ImageDraw.Draw(marked).rectangle(job['box'], outline='red', width=4); marked.save(ref)
        with Image.open(paths['query']) as image:
            if image.size != (1200,675) or image.mode != 'RGB' or image.info: raise ValueError('bad query input')
        query.write_bytes(paths['query'].read_bytes())
        records.append({'id':ident, 'reference':ref.name, 'query':query.name, 'presented_reference_sha256':sha(ref),
                        'presented_query_sha256':sha(query), 'reference_sha256':job['reference_sha256'],
                        'query_sha256':job['query_sha256'], 'box_sha256':json_hash(job['box'])})
    (out/'preview.json').write_text(json.dumps({'selection_sha256':sha(selection_path), 'records':records,
        'warning':'PRIVATE: main reviewer inspection only. No reviewer exposure until approval.'}, indent=2)+'\n')


def release(selection_path, preview_dir, approval_path, reviewer_dir, plan_path, arm):
    selection_path, preview_dir, approval_path, reviewer_dir, plan_path = map(Path, (selection_path, preview_dir, approval_path, reviewer_dir, plan_path))
    if reviewer_dir.exists(): raise FileExistsError(reviewer_dir)
    selection, preview, approval, plan = load(selection_path), load(preview_dir/'preview.json'), load(approval_path), load(plan_path)
    if sha(plan_path) != selection['qualification_plan_sha256'] or arm not in plan['arms']:
        raise ValueError('qualification plan or arm mismatch')
    if preview['selection_sha256'] != sha(selection_path) or approval['selection_sha256'] != sha(selection_path) or approval['preview_sha256'] != sha(preview_dir/'preview.json'):
        raise ValueError('selection, preview or visual approval mismatch')
    if not approval.get('reviewer', '').strip() or approval.get('locked') is not True:
        raise ValueError('locked visual review required')
    rejected = approval.get('excluded', [])
    if any(not x.get('id') or not x.get('reason','').strip() for x in rejected) or len({x['id'] for x in rejected}) != len(rejected):
        raise ValueError('each visual exclusion needs a unique ID and reason')
    if len(rejected) > 12: raise ValueError('qualification permits at most 12 visual exclusions')
    excluded = {x['id'] for x in rejected}
    candidates = {i for c in selection['cells'] for i in c['candidate_ids']}
    if not excluded <= candidates: raise ValueError('unknown exclusion')
    expected=[]
    expected_labels={}
    for cell in selection['cells']:
        available=[i for i in cell['candidate_ids'] if i not in excluded]
        if len(available) < cell['quota']: raise ValueError('insufficient candidates after exclusions')
        expected.extend(available[:cell['quota']])
        expected_labels.update({i: cell['provisional_class'] for i in available[:cell['quota']]})
    accepted=approval.get('accepted_ids', [])
    if len(accepted) != 48 or len(set(accepted)) != 48 or set(accepted) != set(expected):
        raise ValueError('approval must cover the ranked replacement set')
    if approval.get('reference_labels') != expected_labels:
        raise ValueError('locked reference labels must match visually accepted provisional labels')
    visual_cues = approval.get('visual_cues', {})
    if set(visual_cues) != set(expected) or any(not isinstance(v,str) or not v.strip() for v in visual_cues.values()):
        raise ValueError('every accepted case needs a visual cue')
    by_id={r['id']:r for r in preview['records']}
    if len(by_id) != len(preview['records']) or not (set(accepted) | excluded) <= set(by_id):
        raise ValueError('accepted and excluded cases must have been previewed')
    # Check all copies before creating a reviewer-visible directory.
    for ident in accepted:
        rec=by_id[ident]
        for side in ('reference','query'):
            path=preview_dir/'images'/rec[side]
            if path.parent != preview_dir/'images' or sha(path) != rec['presented_'+side+'_sha256']:
                raise ValueError('preview image changed')
    reviewer_dir.mkdir(parents=True); (reviewer_dir/'images').mkdir()
    public=[]
    for ident in sorted(accepted):
        rec=by_id[ident]
        for side in ('reference','query'):
            (reviewer_dir/'images'/rec[side]).write_bytes((preview_dir/'images'/rec[side]).read_bytes())
        public.append({k:rec[k] for k in ('id','reference','query','presented_reference_sha256','presented_query_sha256')})
    (reviewer_dir/'pairs.json').write_text(json.dumps({'instructions':plan['arms'][arm],'pairs':public}, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command', required=True)
    s=sub.add_parser('select'); s.add_argument('--prepared',required=True); s.add_argument('--audit',required=True); s.add_argument('--smoke',required=True); s.add_argument('--plan',required=True); s.add_argument('--out',required=True)
    v=sub.add_parser('preview'); v.add_argument('--prepared',required=True); v.add_argument('--selection',required=True); v.add_argument('--out',required=True); v.add_argument('--extra-id',action='append',default=[])
    r=sub.add_parser('release'); r.add_argument('--selection',required=True); r.add_argument('--preview',required=True); r.add_argument('--approval',required=True); r.add_argument('--reviewer-dir',required=True); r.add_argument('--plan',required=True); r.add_argument('--arm', choices=['original', 'localize_then_label_v2'], required=True)
    a=p.parse_args()
    if a.command=='select': select(a.prepared,a.audit,a.smoke,a.plan,a.out)
    elif a.command=='preview': preview(a.prepared,a.selection,a.out,a.extra_id)
    else: release(a.selection,a.preview,a.approval,a.reviewer_dir,a.plan,a.arm)
