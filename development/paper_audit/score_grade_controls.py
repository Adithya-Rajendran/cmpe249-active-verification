"""Grade frozen archived qualification outcomes by class and engine."""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--private',type=Path,required=True); p.add_argument('--cases',type=Path,required=True); p.add_argument('--records',type=Path,required=True); p.add_argument('--binding',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    private=json.loads(a.private.read_text()); cases=json.loads(a.cases.read_text()); binding=json.loads(a.binding.read_text())
    if binding['cases_sha256']!=sha(a.cases): raise ValueError('input binding mismatch')
    rows=[json.loads(line) for line in a.records.read_text().splitlines()]
    byid={r['id']:r for r in rows}; labels={r['id']:r for r in private}; inputs={r['id']:r for r in cases}
    if len(rows)!=116 or len(byid)!=116 or set(byid)!=set(labels) or set(labels)!=set(inputs): raise ValueError('incomplete or duplicate records')
    groups=collections.defaultdict(lambda: {'correct':0,'total':0,'answers':collections.Counter(),'failures':collections.Counter()})
    for ident in labels:
        row=byid[ident]; label=labels[ident]; inp=inputs[ident]
        if any(row[k]!=inp[k] for k in ('reference_sha256','query_sha256','box_xyxy')): raise ValueError('record input mismatch')
        name='same' if label['engine']=='same' else label['expected'].lower()
        keys=[name] if name=='same' else [name, label['engine']+'_'+name]
        for key in keys:
            g=groups[key]; g['total']+=1; g['correct']+=row['answer']==label['expected']; g['answers'][row['answer']]+=1
            if row.get('failure'): g['failures'][row['failure']]+=1
    counts={k:dict(v) for k,v in groups.items()}
    for group in counts.values():
        group['answers']=dict(group['answers']);group['failures']=dict(group['failures'])
    if any(counts[k]['total']!=v for k,v in {'same':36,'present':40,'removed':40}.items()): raise ValueError('wrong group counts')
    eligible=counts['same']['correct']>=34 and counts['present']['correct']>=36 and counts['removed']['correct']>=36
    result={'qualified':eligible,'criteria':'same >=34/36; clear present >=36/40 and clear removed >=36/40',
            'counts':counts,'median_seconds':statistics.median(r['runtime_seconds'] for r in rows),
            'mean_seconds':statistics.mean(r['runtime_seconds'] for r in rows),'records_sha256':sha(a.records),
            'binding_sha256':sha(a.binding),'cases_sha256':sha(a.cases),'private_sha256':sha(a.private)}
    with a.out.open('x') as f: json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
