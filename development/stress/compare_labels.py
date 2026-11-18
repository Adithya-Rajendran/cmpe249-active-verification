"""Compare complete AI and mask-proxy labels after independent AI review is complete."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from development.stress.alignment_diagnostics import check_frozen
from development.stress.analysis import human_labels
from development.stress.prepare_inputs import PROTOCOL, STUDY, load, sha


def compare(prepared, ai_path, proxy_path, out, size):
    prepared, out = Path(prepared), Path(out)
    if out.exists(): raise FileExistsError(out)
    pre, proto = load(prepared/'preparation.json'), load(PROTOCOL)
    if pre['study_id'] != STUDY or pre['protocol_sha256'] != sha(PROTOCOL) or pre['frozen_hashes'] != check_frozen():
        raise ValueError('preparation source mismatch')
    if size not in {'primary', 'native'}: raise ValueError('invalid input size')
    jobs_path = prepared/'verifier'/size/'jobs.json'
    if sha(jobs_path) != pre['jobs_sha256'][size]: raise ValueError('jobs binding mismatch')
    jobs = load(jobs_path)
    if len(jobs) != pre['count'] or len({j['id'] for j in jobs}) != len(jobs): raise ValueError('jobs incomplete')
    for job in jobs:
        for name in ('reference', 'query'):
            path = jobs_path.parent/job[name]
            if path.parent != jobs_path.parent or sha(path) != job[name+'_sha256']:
                raise ValueError('prepared image hash mismatch')
    dims = proto['primary_size'] if size == 'primary' else proto['render_size']
    input_size = f'{dims[0]}x{dims[1]}'
    ai = human_labels(ai_path, jobs, input_size, 'ai')
    proxy = human_labels(proxy_path, jobs, input_size, 'geometry_proxy')
    table = Counter((ai[k]['label'], proxy[k]['label']) for k in ai)
    labels = ['PRESENT', 'REMOVED', 'UNKNOWN', 'DISPUTED']
    result = {'study_id': STUDY, 'scope': pre['scope'], 'size': input_size,
              'limitation': 'Mask proxy is not a human support label and does not establish visual answerability.',
              'count': len(ai), 'agreement_count': sum(ai[k]['label'] == proxy[k]['label'] for k in ai),
              'agreement_rate': sum(ai[k]['label'] == proxy[k]['label'] for k in ai) / len(ai) if ai else None,
              'cross_tab_ai_rows_proxy_columns': {a: {b: table[a,b] for b in labels} for a in labels},
              'input_hashes': {'preparation': sha(prepared/'preparation.json'), 'jobs': sha(jobs_path),
                               'ai_labels': sha(ai_path), 'proxy_labels': sha(proxy_path),
                               'comparison_source': sha(__file__)}}
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--prepared', required=True); p.add_argument('--ai', required=True)
    p.add_argument('--proxy', required=True); p.add_argument('--out', required=True)
    p.add_argument('--size', choices=['primary', 'native'], required=True)
    a = p.parse_args(); compare(a.prepared, a.ai, a.proxy, a.out, a.size)
