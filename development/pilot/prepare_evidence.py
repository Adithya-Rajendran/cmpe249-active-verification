"""Write image-only evidence jobs using fixed, sanitized pilot inputs."""
import json
from pathlib import Path

OUT=Path('runs/2026-11-16-support-audit-v1')
DEST=OUT/'evidence_jobs'
REMOTE='/tmp/cmpe249-support-dev-v1'

def main():
    DEST.mkdir(exist_ok=True)
    rows=json.loads((OUT/'ledger.json').read_text())
    imgs=json.loads((OUT/'prepared_inputs.json').read_text())
    jobs=[]
    for i,(row,img) in enumerate(zip(rows,imgs,strict=True)):
        box=row['reference_box']; scaled=[round(box[0]*1200/2304),round(box[1]*675/1296),round(box[2]*1200/2304),round(box[3]*675/1296)]
        ann={'schema':'cmpe249.evidence-annotation.v0','reference':f"{REMOTE}/gate_references/{img['reference']['name']}",
             'stop':0,'targets':{'target':{'box':scaled}}}
        path=DEST/f'ann_{i:03d}.json';path.write_text(json.dumps(ann)+'\n')
        jobs.append({'episode_id':f'dev-{i:03d}','stop':0,'target':'target','annotation':f'{REMOTE}/evidence_jobs/{path.name}',
                     'query':f"{REMOTE}/review_inputs/{img['query']['name']}"})
    (DEST/'jobs.json').write_text(json.dumps({'schema':'cmpe249.evidence-jobs.v0','jobs':jobs},indent=2)+'\n')
    print(len(jobs),'jobs')
if __name__=='__main__':main()
