"""Analyze complete image-evidence records with source-typed reference labels."""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.pilot.metrics import summarize
from development.stress.prepare_inputs import PROTOCOL, load, sha, STUDY, json_hash
from development.stress.alignment_diagnostics import check_frozen, plane_diagnostics


def human_labels(path, jobs, input_size, label_source="human"):
    if label_source not in {"human", "ai", "geometry_proxy"}:
        raise ValueError("unknown label source")
    by_id = {job['id']: job for job in jobs}
    ids = set(by_id)
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ids) or {r['id'] for r in rows} != ids or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('human labels must contain exactly one row per input')
    for r in rows:
        if r['source'] != label_source or r['label'] not in {'PRESENT', 'REMOVED', 'UNKNOWN', 'DISPUTED'}:
            raise ValueError('every label must be complete and match the selected source')
        if not r['reviewer'].strip() or not r['cue'].strip():
            raise ValueError('human label needs reviewer and cue/uncertainty')
        job = by_id[r['id']]
        expected = {'input_size': input_size, 'reference_sha256': job['reference_sha256'],
                    'query_sha256': job['query_sha256'], 'box_sha256': json_hash(job['box'])}
        if any(r.get(k) != v for k, v in expected.items()):
            raise ValueError('human label input image, size or box binding mismatch')
    return {r['id']: r for r in rows}


def recoverable_abstentions(rows):
    """Count abstentions that could become useful answers under the human label."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['arrangement']].append(row)
    def recoverable(row):
        return (row['answer'] == 'UNKNOWN' and row['support'] in {'PRESENT', 'REMOVED'}
                and row['support'] == row['physical'])
    rates = {key: sum(recoverable(row) for row in group) / len(group) for key, group in grouped.items()}
    return {'numerator': sum(recoverable(row) for row in rows), 'denominator': len(rows),
            'arrangement_weighted': sum(rates.values()) / len(rates) if rates else None,
            'arrangement_rates': rates}


def human_physical_conflicts(rows):
    """Count definite human judgments that disagree with physical intervention."""
    definite = [r for r in rows if r['support'] in {'PRESENT', 'REMOVED'}]
    return {'definite_human_labels': len(definite),
            'human_present_physical_removed': sum(r['support'] == 'PRESENT' and r['physical'] == 'REMOVED' for r in definite),
            'human_removed_physical_present': sum(r['support'] == 'REMOVED' and r['physical'] == 'PRESENT' for r in definite),
            'total': sum(r['support'] != r['physical'] for r in definite)}


def development_decisions(results, scope, has_labels, analysis, label_source="human"):
    if label_source == "geometry_proxy":
        return {"status": "not eligible", "reason": "Geometry proxy is not a support-label basis for triage."}
    if label_source == "ai":
        # Basic qualification does not validate a complete support-label version.
        # Keep metrics available while the reviewed label and analysis gates remain open.
        return {"status": "not eligible", "reason": "AI reviewer qualification alone does not establish a decision-ready support-label reference.",
                "next_step": "Use the repeated audit descriptively. Validate the support reference and complete the declared data before a reviewed analysis revision."}
    primary_key = "human_undisputed" if label_source == "human" else "ai_reference"
    sensitivity_key = "disputed_as_unknown" if label_source == "human" else "ai_reference_disputed_as_unknown"
    if analysis != 'primary':
        return {'status': 'not eligible', 'reason': 'Triage applies only to the primary analysis.'}
    if scope != 'full' or not has_labels:
        return {'status': 'not eligible', 'reason': 'Full cohort and complete labels from the selected source required.'}
    output = {}
    for env, levels in results.items():
        names = ('D0', 'D1', 'D2', 'D4', 'D6')
        if any(name not in levels for name in names):
            output[env] = {'outcome': 'INDETERMINATE', 'reason': 'Missing required level; audit labels and cohort.'}
            continue
        if any(levels[name][primary_key]['unsupported']['estimate'] is None for name in names):
            output[env] = {'outcome': 'INDETERMINATE', 'reason': 'Undefined primary support denominator; label audit required.'}
            continue
        if any(levels[name][sensitivity_key]['unsupported']['estimate'] is None for name in names):
            output[env] = {'outcome': 'INDETERMINATE', 'reason': 'Undefined sensitivity support denominator; label audit required.'}
            continue
        triggering = []
        for name in names[1:]:
            primary = levels[name][primary_key]
            sensitivity = levels[name][sensitivity_key]
            if all(x['unsupported']['estimate'] > .01 and x['unsupported_error_arrangements'] >= 3
                   for x in (primary, sensitivity)):
                triggering.append(name)
        d0 = levels['D0'][primary_key]['useful']['estimate']
        d6 = levels['D6'][primary_key]['useful']['estimate']
        if triggering:
            outcome, affected = 'A', triggering
        elif d0 is None or d6 is None:
            output[env] = {'outcome': 'INDETERMINATE', 'reason': 'Undefined useful-yield denominator; label audit required.'}
            continue
        elif d0 - d6 >= .10:
            outcome, affected = 'B', ['D6']
        else:
            outcome, affected = 'C', []
        headroom = {}
        for name in affected:
            value = levels[name]
            primary = value['recoverable_abstentions']['arrangement_weighted']
            sensitivity = value['recoverable_abstentions_disputed_as_unknown']['arrangement_weighted']
            if primary is None or sensitivity is None:
                status = 'INDETERMINATE'
            elif primary >= .10 and sensitivity >= .10:
                status = 'threshold met'
            else:
                status = 'threshold not met'
            headroom[name] = {'status': status, 'primary': value['recoverable_abstentions'],
                              'disputed_as_unknown': value['recoverable_abstentions_disputed_as_unknown'],
                              'next_step': 'Model execution remains pending a credible guard comparison and qualification.'}
        output[env] = {'outcome': outcome, 'scope': ('PROVISIONAL AI-only, not human validated' if label_source == 'ai' else 'development triage, not confirmatory'),
                       'cascade_headroom': headroom}
    return output


def analyze(prepared, run_dir, out, labels_path=None, label_source="human", label_plan=None):
    if label_source not in {"human", "ai", "geometry_proxy"}:
        raise ValueError("unknown label source")
    primary_key = "human_undisputed" if label_source == "human" else label_source + "_reference"
    sensitivity_key = "disputed_as_unknown" if label_source == "human" else label_source + "_reference_disputed_as_unknown"
    prepared, run_dir, out = map(Path, (prepared, run_dir, out))
    pre, manifest = load(prepared / 'preparation.json'), load(run_dir / 'manifest.json')
    frozen = check_frozen()
    if pre.get('frozen_hashes') != frozen or manifest.get('frozen_hashes') != frozen:
        raise ValueError('frozen source binding mismatch')
    if pre['study_id'] != STUDY or manifest['study_id'] != STUDY or manifest['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('study or protocol mismatch')
    if manifest['preparation_sha256'] != sha(prepared / 'preparation.json'):
        raise ValueError('preparation binding mismatch')
    proto = load(PROTOCOL)
    spec = next(x for x in proto['analyses'] if x['id'] == manifest['analysis'])
    size = 'primary' if spec['size'] == proto['primary_size'] else 'native'
    jobs_path = prepared / 'verifier' / size / 'jobs.json'
    if pre['jobs_sha256'][size] != sha(jobs_path) or manifest['jobs_sha256'] != sha(jobs_path):
        raise ValueError('jobs binding mismatch')
    if pre['ledger_sha256'] != sha(prepared / 'ledger.json'):
        raise ValueError('ledger binding mismatch')
    jobs, ledger = load(jobs_path), load(prepared / 'ledger.json')
    for job in jobs:
        for key in ('reference', 'query'):
            if sha(jobs_path.parent / job[key]) != job[key + '_sha256']:
                raise ValueError('prepared image hash mismatch')
    audit_path = ROOT / 'runs/2026-11-17-stress-dev-v1/reviewer/geometry_audit.json'
    sign_audit = None
    if audit_path.is_file():
        audit = load(audit_path)
        if (audit.get('protocol_sha256') == sha(PROTOCOL) and
                all(audit.get('arrangement_file_sha256', {}).get(r.get('arrangement_id')) == r['arrangement_file_sha256'] for r in ledger)):
            sign_audit = {'path': str(audit_path), 'sha256': sha(audit_path),
                          'interpretation': 'For nonzero levels only, same signs add and opposite signs cancel in the audited target-point projection.'}
    ids = {r['id'] for r in ledger}
    if len(ledger) != pre['count'] or len(ids) != len(ledger) or {j['id'] for j in jobs} != ids:
        raise ValueError('input/ledger mismatch')
    files = list((run_dir / 'records').glob('*.json'))
    if {f.stem for f in files} != ids:
        raise ValueError('missing or extra evidence records')
    evidence = {f.stem: load(f) for f in files}
    if any(r.get('id') != k or r.get('binding_sha256') != sha(run_dir / 'manifest.json') for k, r in evidence.items()):
        raise ValueError('evidence binding mismatch')
    labels = human_labels(labels_path, jobs, f"{spec['size'][0]}x{spec['size'][1]}", label_source) if labels_path else None
    grouped = defaultdict(list)
    diagnostics = []
    for item in ledger:
        record = evidence[item['id']]
        if record.get('failure') is not False:
            raise ValueError('failed record cannot be silently analyzed')
        e, diag, t = record['evidence'], record['diagnostics'], proto['thresholds']
        if record['refusal'] not in {'accepted', 'correlation', 'shift', 'scale', 'optimizer_or_input'}:
            raise ValueError('invalid refusal category')
        if any(not isinstance(e[k], (float, int)) or not math.isfinite(e[k]) or not 0 <= e[k] <= 1 for k in ('quality', 'match_score', 'obstruction')):
            raise ValueError('invalid evidence score')
        if type(diag['align_ok']) is not bool:
            raise ValueError('invalid alignment flag')
        valid = bool(diag['align_ok'])
        present = valid and e['quality'] >= t['quality'] and e['match_score'] >= t['present']
        removed = valid and e['quality'] >= t['quality'] and e['match_score'] <= t['absent'] and e['obstruction'] <= t['obstruction']
        if present and removed:
            raise ValueError('contradictory gates')
        answer = 'PRESENT' if present else 'REMOVED' if removed else 'UNKNOWN'
        row = {'arrangement': item['base_id'], 'physical': item['physical_state'], 'answer': answer,
               'failure': False, 'support': labels[item['id']]['label'] if labels else None,
               'sign_translation': item['sign_translation'], 'sign_yaw': item['sign_yaw'],
               'region_obstruction': item['region_obstruction'], 'refusal': record['refusal']}
        grouped[(item['environment'], item['level'])].append(row)
        if sha(item['arrangement_file']) != item['arrangement_file_sha256']:
            raise ValueError('analysis arrangement file hash mismatch')
        arr = load(item['arrangement_file'])
        from sim.arrange import arrangement_hash
        if arr.get('arrangement_sha256') != arrangement_hash(arr):
            raise ValueError('analysis arrangement content hash mismatch')
        d = plane_diagnostics(record['raw_H_query_to_reference'], arr, item['stop'], item['reference_box_native'], spec['size'][0])
        diagnostics.append({'id': item['id'], 'registration_refusal': record['refusal'], 'planes': d})
    results = {}
    for (env, level), rows in grouped.items():
        metrics = summarize(rows)
        metrics = {k: metrics[k] for k in ('coverage', 'unknown', 'failure', 'false_removal', 'false_presence', 'answered_error')}
        result = {'engineering': metrics, 'sign_groups': {}, 'motion_groups': {}, 'refusal_counts': {}, 'obstruction_bands': {'low': 0, 'middle': 0, 'high': 0, 'unavailable': 0}}
        for row in rows:
            result['refusal_counts'][row['refusal']] = result['refusal_counts'].get(row['refusal'], 0) + 1
            band = row['region_obstruction']
            name = 'unavailable' if band is None else ('low' if band <= .3 else ('middle' if band < .9 else 'high'))
            result['obstruction_bands'][name] += 1
        result['obstruction_strata'] = {}
        for label, pred in [('low', lambda v: v is not None and v <= .3), ('middle', lambda v: v is not None and .3 < v < .9), ('high', lambda v: v is not None and v >= .9), ('unavailable', lambda v: v is None)]:
            subset = [r for r in rows if pred(r['region_obstruction'])]
            m = summarize(subset)
            result['obstruction_strata'][label] = {k: m[k] for k in ('coverage', 'unknown', 'false_removal', 'false_presence')}
        for label, pred in [('same_sign', lambda r: r['sign_translation'] == r['sign_yaw']), ('opposite_sign', lambda r: r['sign_translation'] != r['sign_yaw'])]:
            subset = [r for r in rows if pred(r)]
            m = summarize(subset)
            result['motion_groups'][label] = {k: m[k] for k in ('coverage', 'unknown', 'false_removal', 'false_presence')}
        for sign in sorted({(r['sign_translation'], r['sign_yaw']) for r in rows}):
            group = [r for r in rows if (r['sign_translation'], r['sign_yaw']) == sign]
            m = summarize(group)
            result['sign_groups'][str(sign)] = {k: m[k] for k in ('coverage', 'unknown', 'false_removal', 'false_presence')}
        if labels:
            result[primary_key] = summarize(rows)
            result[sensitivity_key] = summarize(rows, disputed_as_unknown=True)
            eligible = [r for r in rows if r['support'] in {'PRESENT', 'REMOVED', 'UNKNOWN'}]
            result['recoverable_abstentions'] = recoverable_abstentions(eligible)
            conflicts = human_physical_conflicts(rows)
            result['human_physical_conflicts' if label_source == 'human' else label_source + '_physical_conflicts'] = (conflicts if label_source == 'human' else {k.replace('human_', 'reference_'): v for k, v in conflicts.items()})
            sens = [dict(r, support='UNKNOWN') if r['support'] == 'DISPUTED' else r for r in rows]
            result['recoverable_abstentions_disputed_as_unknown'] = recoverable_abstentions(sens)
            for category, adjusted in [(primary_key, rows), (sensitivity_key, sens)]:
                result[category]['unsupported_error_arrangements'] = len({r['arrangement'] for r in adjusted if r['support'] in {'PRESENT', 'REMOVED', 'UNKNOWN'} and r['answer'] in {'PRESENT', 'REMOVED'} and r['answer'] != r['support']})
        results.setdefault(env, {})[level] = result
    # Paired arrangement bootstrap on arrangement-level rates; never resample individual views.
    rng = random.Random(proto['bootstrap']['seed'])
    comparisons = {}
    for env, levels in results.items():
        comparisons[env] = {}
        if 'D0' not in levels:
            raise ValueError('missing D0 comparison')
        for level, result in levels.items():
            if level == 'D0':
                continue
            contrasts = {}
            for category in (['engineering', primary_key, sensitivity_key] if labels else ['engineering']):
              for metric in result[category]:
                if metric == 'unsupported_error_arrangements': continue
                a = levels['D0'][category][metric]['arrangement_rates']
                b = result[category][metric]['arrangement_rates']
                if set(a) != set(b):
                    raise ValueError('unpaired arrangements')
                differences = [b[k] - a[k] for k in sorted(a) if a[k] is not None and b[k] is not None]
                if differences:
                    draws = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                                   for _ in range(proto['bootstrap']['draws']))
                    contrasts[category + ':' + metric] = {'difference': sum(differences)/len(differences),
                                         'interval_95_percentile': [draws[int(.025*len(draws))], draws[int(.975*len(draws))-1]],
                                         'paired_arrangements': len(differences)}
            if labels:
                for metric in ('recoverable_abstentions', 'recoverable_abstentions_disputed_as_unknown'):
                    a, b = levels['D0'][metric]['arrangement_rates'], result[metric]['arrangement_rates']
                    keys = sorted(set(a) & set(b))
                    if keys:
                        differences = [b[k]-a[k] for k in keys]
                        draws = sorted(sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(proto['bootstrap']['draws']))
                        contrasts[metric] = {'difference': sum(differences)/len(differences), 'interval_95_percentile': [draws[int(.025*len(draws))], draws[int(.975*len(draws))-1]], 'paired_arrangements': len(keys)}
            comparisons[env][level] = contrasts
    report = {'study_id': STUDY, 'analysis': manifest['analysis'], 'scope': pre['scope'],
              'input_count': len(ids), 'human_labels': {'sha256': sha(labels_path), 'type': 'human'} if labels and label_source == 'human' else None,
              'support_labels': None if labels is None else {'source': label_source, 'sha256': sha(labels_path), 'reviewer_ids': sorted({r['reviewer'] for r in labels.values()}), 'label_plan_sha256': None if label_plan is None else sha(label_plan)},
              'interpretation': 'Engineering factual scores compare with physical intervention. They do not establish visual support.',
              'provenance': {'analysis_sha256': sha(__file__),
                             'metrics_sha256': sha(ROOT / 'development/pilot/metrics.py'),
                             'alignment_diagnostics_sha256': sha(ROOT / 'development/stress/alignment_diagnostics.py'),
                             'protocol_sha256': sha(PROTOCOL),
                             'preparation_sha256': sha(prepared / 'preparation.json'),
                             'ledger_sha256': sha(prepared / 'ledger.json'),
                             'jobs_sha256': sha(jobs_path),
                             'run_manifest_sha256': sha(run_dir / 'manifest.json'),
                             'labels_sha256': None if labels_path is None else sha(labels_path)},
              'results': results, 'paired_bootstrap': comparisons, 'alignment': diagnostics, 'sign_interpretation': sign_audit,
              'development_decisions': development_decisions(results, pre['scope'], bool(labels), manifest['analysis'], label_source)}
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', required=True); p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    p.add_argument('--labels')
    p.add_argument('--label-source', choices=['human', 'ai', 'geometry_proxy'], default='human')
    p.add_argument('--label-plan')
    a = p.parse_args()
    analyze(a.prepared, a.run, a.out, a.labels, a.label_source, a.label_plan)
