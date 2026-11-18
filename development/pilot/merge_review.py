"""Bind the separate AI support review to the fixed development manifest."""
import hashlib
import json
from pathlib import Path
P=Path('runs/2026-11-16-support-audit-v1')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    review=json.loads((P/'support_review.json').read_text())
    if review['study_id']!='support-dev-pilot-v1' or review['reviewer_type']!='AI' or review['task_predictions_seen']:
        raise ValueError('invalid review provenance')
    if digest(P/'ledger.json')!=review['ledger_sha256_before_merge'] or digest(P/'prepared_inputs.json')!=review['prepared_inputs_sha256']:
        raise ValueError('reviewed inputs changed')
    path=P/'pilot_manifest.json';manifest=json.loads(path.read_text());rows=manifest['rows']
    by_id={r['record_index']:r for r in review['rows']}
    if len(by_id)!=len(rows) or set(by_id)!=set(range(len(rows))): raise ValueError('review record mismatch')
    for r in rows:
        x=by_id[r['id']]
        for k in ('arrangement','stop','phase','partition'):
            if r[k]!=x[k]: raise ValueError(f'review mismatch {k}')
        if r['physical'].lower()!=x['physical_state'] or r['reference']['sha256']!=x['prepared_reference_sha256'] or r['query']['sha256']!=x['prepared_query_sha256']:
            raise ValueError('reviewed image mismatch')
        if x['reviewer_type']!='AI' or x['support'] not in {'PRESENT','REMOVED','UNKNOWN','DISPUTED'}: raise ValueError('invalid review')
        r.update(support=x['support'],reviewer_type='AI')
    manifest['support_review_sha256']=digest(P/'support_review.json')
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('bound',len(rows),'review records')
if __name__=='__main__':main()
