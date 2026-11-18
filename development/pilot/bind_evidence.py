"""Bind sanitized-image evidence to an existing reviewed manifest."""
import hashlib
import json
from pathlib import Path
P=Path('runs/2026-11-16-support-audit-v1')
def main():
    path=P/'pilot_manifest.json'; manifest=json.loads(path.read_text()); rows=manifest['rows']
    score={x['episode_id']:x for x in map(json.loads,(P/'evidence_unmarked.jsonl').read_text().splitlines())}
    diag={x['episode_id']:x for x in map(json.loads,(P/'evidence_unmarked.diag.jsonl').read_text().splitlines())}
    expected={f'dev-{i:03d}' for i in range(len(rows))}
    if set(score)!=expected or set(diag)!=expected: raise ValueError('missing or extra evidence')
    for row in rows:
        key=f"dev-{row['id']:03d}";s=score[key];d=diag[key]
        row.update(valid=bool(d['align_ok']),quality=s['quality'],match=s['match_score'],
                   obstruction=s['obstruction'],evidence_failure=False)
    manifest['evidence_sources_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
       for p in (P/'evidence_unmarked.jsonl',P/'evidence_unmarked.diag.jsonl',P/'evidence_jobs/jobs.json')}
    gate_files=sorted((P/'gate_references').glob('*.png'))
    annotation_files=sorted((P/'evidence_jobs').glob('ann_*.json'))
    if len(gate_files)!=36 or len(annotation_files)!=108: raise ValueError('gate inputs incomplete')
    manifest['gate_input_sha256']={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in gate_files+annotation_files}
    manifest['evidence_preprocessing']='sanitized 1200x675 unmarked reference and query; scaled reference box'
    manifest['evidence_status']='SANITIZED_UNMARKED_V1'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(len(rows),'rows; valid alignment',sum(r['valid'] for r in rows))
if __name__=='__main__':main()
