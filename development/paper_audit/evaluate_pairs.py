"""Aggregate locked case outcomes by engine and resample whole arrangements."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
import numpy as np


def evaluate(rows, repetitions=10000, seed=249):
    """Rows contain engine, arrangement, retained and removed categorical answers.

    Answers are present, absent or unknown. Returns per-pair outcomes and paired
    arrangement bootstrap intervals. Missing/duplicate pairs fail closed.
    """
    if not rows or repetitions < 1:
        raise ValueError('empty rows or invalid repetitions')
    grouped = defaultdict(list)
    seen = set()
    outcomes = []
    for r in rows:
        key = (r['engine'], r['arrangement'], r['pair'])
        if key in seen:
            raise ValueError('duplicate pair')
        seen.add(key)
        if r['retained'] not in ('present', 'absent', 'unknown') or r['removed'] not in ('present', 'absent', 'unknown'):
            raise ValueError('invalid answer')
        correct = (r['retained'] == 'present', r['removed'] == 'absent')
        definite = (r['retained'] != 'unknown', r['removed'] != 'unknown')
        result = {'engine':r['engine'], 'arrangement':r['arrangement'], 'pair':r['pair'],
                  'both_correct':int(all(correct)), 'correct':sum(correct),
                  'definite':sum(definite), 'wrong_definite':sum(d and not c for d,c in zip(definite,correct))}
        outcomes.append(result)
        grouped[r['engine']].append(result)
    rng = np.random.default_rng(seed)
    summary = {}
    for engine, cases in sorted(grouped.items()):
        clusters = defaultdict(list)
        for row in cases:
            clusters[row['arrangement']].append(row)
        keys = sorted(clusters)
        def stats(selected):
            data = [r for k in selected for r in clusters[k]]
            return [sum(r['both_correct'] for r in data)/len(data),
                    sum(r['correct'] for r in data)/(2*len(data)),
                    sum(r['definite'] for r in data)/(2*len(data)),
                    sum(r['wrong_definite'] for r in data)/(2*len(data))]
        names = ('both_correct', 'correct', 'coverage', 'wrong_definite')
        estimate = stats(keys)
        boot = np.array([stats(rng.choice(keys, size=len(keys), replace=True)) for _ in range(repetitions)])
        summary[engine] = {'arrangements':len(keys), 'pairs':len(cases),
            'metrics':{name:{'estimate':value, 'bootstrap_95_percentile':np.quantile(boot[:,i],[.025,.975]).tolist() if len(keys)>1 else None}
                       for i,(name,value) in enumerate(zip(names,estimate))}}
    return {'seed':seed, 'repetitions':repetitions, 'outcomes':outcomes, 'engines':summary}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rows'); p.add_argument('output'); p.add_argument('--seed',type=int,default=249)
    p.add_argument('--repetitions',type=int,default=10000)
    args = p.parse_args()
    with open(args.rows) as f: rows = json.load(f)
    with open(args.output, 'x') as f: json.dump(evaluate(rows,args.repetitions,args.seed),f,indent=2,allow_nan=False); f.write('\n')
