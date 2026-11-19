"""Independently recount the frozen archived zero-shot qualification."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from development.paper_audit.inputs_build import sha


def run(packet,private,record,out):
    packet=Path(packet);cases=json.loads((packet/'cases.json').read_text()); truth=json.loads(Path(private).read_text())
    predictions=[json.loads(x) for x in Path(record).read_text().splitlines() if x.strip()]
    binding=json.loads((Path(record).parent/'binding.json').read_text())
    if binding['cases_sha256']!=sha(packet/'cases.json'): raise ValueError('score binding mismatch')
    c={x['id']:x for x in cases}; t={x['id']:x for x in truth}; p={x['id']:x for x in predictions}
    if len(c)!=len(cases) or len(t)!=len(truth) or len(p)!=len(predictions) or set(c)!=set(t) or set(c)!=set(p): raise ValueError('inventory mismatch')
    rows=[]
    for uid in sorted(c):
        for role in ('reference','query'):
            if sha(packet/c[uid][role])!=c[uid][role+'_sha256'] or p[uid][role+'_sha256']!=c[uid][role+'_sha256']: raise ValueError('image binding mismatch')
        if p[uid]['box_xyxy']!=c[uid]['box_xyxy']: raise ValueError('box binding mismatch')
        rows.append({**t[uid],'answer':p[uid]['answer'],'correct':p[uid]['answer']==t[uid]['expected'],'failure':p[uid].get('failure')})
    groups={}
    for name,subset,minimum,expected_n in (
        ('same',[r for r in rows if r['engine']=='same'],34,36),
        ('clear_present',[r for r in rows if r['engine']!='same' and r['expected']=='PRESENT'],36,40),
        ('clear_removed',[r for r in rows if r['engine']!='same' and r['expected']=='REMOVED'],36,40)):
        correct=sum(r['correct'] for r in subset)
        groups[name]={'n':len(subset),'correct':correct,'minimum':minimum,'passed':len(subset)==expected_n and correct>=minimum,'answers':dict(Counter(r['answer'] for r in subset))}
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'qualification.json').write_text(json.dumps({'passed':all(v['passed'] for v in groups.values()),'groups':groups,
        'packet_sha256':sha(packet/'cases.json'),'private_sha256':sha(private),'records_sha256':sha(record),'evaluator_sha256':sha(__file__)},indent=2)+'\n')
    with (out/'qualification.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--private',required=True);p.add_argument('--record',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.packet,a.private,a.record,a.out)
