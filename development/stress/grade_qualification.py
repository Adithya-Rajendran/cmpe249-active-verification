"""Grade a completed AI control review against a previously locked reference."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def grade_counts(rows, criteria):
    """Apply every declared class and environment gate, including definite errors."""
    classes = ('PRESENT', 'REMOVED', 'UNKNOWN')
    total = Counter(r['reference_label'] for r in rows)
    correct = Counter(r['reference_label'] for r in rows if r['reference_label'] == r['label'])
    env_total = Counter((r['environment'], r['reference_label']) for r in rows)
    env_correct = Counter((r['environment'], r['reference_label']) for r in rows if r['reference_label'] == r['label'])
    if total != Counter({c: criteria['denominator_each_label'] for c in classes}):
        raise ValueError('incorrect reference class counts')
    if env_total != Counter({(e, c): criteria['denominator_each_environment_label'] for e in ('shelf', 'tabletop') for c in classes}):
        raise ValueError('incorrect reference environment counts')
    opposite = [r['id'] for r in rows if {r['reference_label'], r['label']} == {'PRESENT', 'REMOVED'}]
    gates = {c: correct[c] >= criteria['minimum_correct_by_label'][c] for c in classes}
    env_gates = {e: {c: env_correct[e, c] >= criteria['minimum_correct_by_environment_and_label'][c] for c in classes} for e in ('shelf', 'tabletop')}
    passed = all(gates.values()) and all(all(x.values()) for x in env_gates.values()) and len(opposite) <= criteria['maximum_opposite_definite_errors_on_clear_controls']
    confusion = {c: dict(Counter(r['label'] for r in rows if r['reference_label'] == c)) for c in classes}
    return {'passed': passed, 'count': len(rows), 'correct': sum(correct.values()),
            'correct_by_label': {c: {'correct': correct[c], 'total': total[c], 'passed': gates[c]} for c in classes},
            'correct_by_environment_and_label': {e: {c: {'correct': env_correct[e, c], 'total': env_total[e, c], 'passed': env_gates[e][c]} for c in classes} for e in ('shelf', 'tabletop')},
            'confusion': confusion, 'opposite_definite_error_ids': opposite,
            'errors': [r for r in rows if r['reference_label'] != r['label']]}


def grade(selection_path, preview_dir, approval_path, plan_path, pairs_path, labels_path, arm):
    selection, preview, approval, plan, public, result = map(read, (selection_path, Path(preview_dir)/'preview.json', approval_path, plan_path, pairs_path, labels_path))
    if selection['qualification_plan_sha256'] != sha(plan_path) or approval.get('locked') is not True:
        raise ValueError('plan or locked approval mismatch')
    if preview['selection_sha256'] != sha(selection_path) or approval['selection_sha256'] != sha(selection_path) or approval['preview_sha256'] != sha(Path(preview_dir)/'preview.json'):
        raise ValueError('private reference binding mismatch')
    if public['instructions'] != plan['arms'][arm] or result.get('instructions') != public['instructions']:
        raise ValueError('review instructions mismatch')
    if result.get('pairs_sha256') != sha(pairs_path) or result.get('model') != 'gpt-6.1-terra' or result.get('complete') is not True:
        raise ValueError('incomplete review or model/input binding mismatch')
    pairs = {r['id']: r for r in public['pairs']}
    labels = {r['id']: r for r in result['labels']}
    accepted = set(approval['accepted_ids'])
    if len(accepted) != 48 or len(pairs) != len(public['pairs']) or len(labels) != len(result['labels']) or set(pairs) != accepted or set(labels) != accepted:
        raise ValueError('missing, duplicate or unexpected control labels')
    previews = {r['id']: r for r in preview['records']}
    expected_presentations = set()
    for ident, pair in pairs.items():
        for side in ('reference', 'query'):
            expected = previews[ident]['presented_'+side+'_sha256']
            path = Path(pairs_path).parent/'images'/pair[side]
            if path.parent != Path(pairs_path).parent/'images' or sha(path) != expected or pair['presented_'+side+'_sha256'] != expected:
                raise ValueError('presented image binding mismatch')
            expected_presentations.add((ident, side, expected))
    actual = [(r['id'], r['side'], r['sha256']) for r in result['presented_images']]
    if len(actual) != 96 or set(actual) != expected_presentations:
        raise ValueError('actual presentation record incomplete or mismatched')
    rows = []
    for ident in approval['accepted_ids']:
        rec = labels[ident]
        if rec['label'] not in ('PRESENT', 'REMOVED', 'UNKNOWN', 'DISPUTED') or not isinstance(rec.get('cue'), str) or not rec['cue'].strip() or rec.get('source') != 'ai':
            raise ValueError('invalid AI label or missing observation')
        if arm == 'localize_then_label_v2' and any(not isinstance(rec.get(k), str) or not rec[k].strip() for k in ('target_identity', 'query_region', 'evidence')):
            raise ValueError('revised instructions require identity, region and evidence observations')
        rows.append({'id': ident, 'environment': selection['candidates'][ident]['source_record']['environment'],
                     'reference_label': approval['reference_labels'][ident], 'label': rec['label'], 'cue': rec['cue']})
    score = grade_counts(rows, plan['qualification'])
    score.update({'arm': arm, 'scope': plan['scope'], 'human_validation': False,
                  'supports_one_percent_error_bound': False, 'releases_task_model': False,
                  'reviewed_image_count': 96,
                  'bindings': {name: sha(path) for name, path in [('selection', selection_path), ('preview', Path(preview_dir)/'preview.json'), ('approval', approval_path), ('plan', plan_path), ('pairs', pairs_path), ('labels', labels_path), ('grader', __file__)]},
                  'rows': rows})
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('selection', 'preview', 'approval', 'plan', 'pairs', 'labels', 'out'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--arm', choices=('original', 'localize_then_label_v2'), required=True)
    args = parser.parse_args()
    score = grade(args.selection, args.preview, args.approval, args.plan, args.pairs, args.labels, args.arm)
    with Path(args.out).open('x') as stream:
        stream.write(json.dumps(score, indent=2)+'\n')
    print(json.dumps({k: score[k] for k in ('arm', 'passed', 'count', 'correct', 'correct_by_label')}))
