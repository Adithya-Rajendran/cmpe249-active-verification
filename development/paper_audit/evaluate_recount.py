"""Recount definite, incorrect and primary outcomes from the exported case table."""
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

root=Path('runs/2026-11-18-paper-audit-v1/evaluation/final')
source=root/'cases.csv';rows=list(csv.DictReader(source.open())); counts=defaultdict(Counter)
for r in rows:
    c=counts[(r['engine'],r['method'])]
    c['rows']+=1;c['definite']+=r['answer']!='UNKNOWN';c['wrong_definite']+=r['answer'] not in ('UNKNOWN',r['truth'])
    if r['level']=='D6' and r['state']=='clear_removed':
        c['primary_n']+=1;c['primary_correct']+=r['answer']==r['truth']
result={'case_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'recount_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'groups':[{'engine':engine,'method':method,**values} for (engine,method),values in sorted(counts.items())]}
with (root/'recount.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
