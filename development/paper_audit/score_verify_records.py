"""Check a scoring arm against its public case list without reading labels."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected',type=int,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    cases=json.loads(a.cases.read_text())
    records_path=a.output/'records.jsonl'; binding_path=a.output/'binding.json'
    binding=json.loads(binding_path.read_text())
    if binding['cases_sha256']!=sha(a.cases): raise ValueError('case binding mismatch')
    inputs={x['id']:x for x in cases}
    records=[json.loads(line) for line in records_path.read_text().splitlines()]
    found={x['id']:x for x in records}
    if len(cases)!=a.expected or len(inputs)!=a.expected or len(records)!=a.expected or len(found)!=a.expected or set(inputs)!=set(found): raise ValueError('case inventory mismatch')
    answers=collections.Counter(); failures=collections.Counter()
    for ident,r in found.items():
        if r.get('answer') not in ('PRESENT','REMOVED','UNKNOWN'): raise ValueError('invalid answer')
        if any(r.get(k)!=inputs[ident][k] for k in ('reference_sha256','query_sha256','box_xyxy')): raise ValueError('record input mismatch')
        answers[r['answer']]+=1
        if r.get('failure'): failures[str(r['failure'])]+=1
    receipt={'cases_sha256':sha(a.cases),'binding_sha256':sha(binding_path),'records_sha256':sha(records_path),
             'cases':len(cases),'answers':dict(answers),'failures':dict(failures)}
    with a.receipt.open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    print('Validated',len(cases),'unique records. Failures:',sum(failures.values()))
if __name__=='__main__': main()
