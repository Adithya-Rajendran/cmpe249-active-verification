"""Development-only vetoes and arrangement-weighted metrics."""
from collections import defaultdict

DEFINITE = {'PRESENT', 'REMOVED'}
SUPPORT = DEFINITE | {'UNKNOWN'}

def veto(base, present_support, removal_support, removal_only=False):
    if base == 'PRESENT':
        return base if removal_only or present_support else 'UNKNOWN'
    if base == 'REMOVED':
        return base if removal_support else 'UNKNOWN'
    return 'UNKNOWN'

def summarize(rows, disputed_as_unknown=False):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['arrangement']].append(row)
    results = {}
    definitions = {
        'unsupported': (lambda r: r['support'] in SUPPORT,
                        lambda r: not r['failure'] and r['answer'] in DEFINITE and r['answer'] != r['support']),
        'unsupported_present': (lambda r: r['support'] in SUPPORT,
                                lambda r: not r['failure'] and r['answer'] == 'PRESENT' and r['answer'] != r['support']),
        'unsupported_removed': (lambda r: r['support'] in SUPPORT,
                                lambda r: not r['failure'] and r['answer'] == 'REMOVED' and r['answer'] != r['support']),
        'useful': (lambda r: r['support'] in SUPPORT,
                   lambda r: not r['failure'] and r['answer'] in DEFINITE and r['answer'] == r['support'] == r['physical']),
        'present_resolution': (lambda r: r['support'] == 'PRESENT',
                               lambda r: not r['failure'] and r['answer'] == r['physical'] == 'PRESENT'),
        'removal_resolution': (lambda r: r['support'] == 'REMOVED',
                               lambda r: not r['failure'] and r['answer'] == r['physical'] == 'REMOVED'),
        'false_removal': (lambda r: r['physical'] == 'PRESENT',
                          lambda r: not r['failure'] and r['answer'] == 'REMOVED'),
        'false_presence': (lambda r: r['physical'] == 'REMOVED',
                           lambda r: not r['failure'] and r['answer'] == 'PRESENT'),
        'coverage': (lambda r: True, lambda r: not r['failure'] and r['answer'] in DEFINITE),
        'unknown': (lambda r: True, lambda r: r['failure'] or r['answer'] == 'UNKNOWN'),
        'failure': (lambda r: True, lambda r: r['failure']),
        'answered_error': (lambda r: not r['failure'] and r['answer'] in DEFINITE,
                           lambda r: r['answer'] != r['physical']),
    }
    for name, (eligible, event) in definitions.items():
        group_rates, numerator, denominator = {}, 0, 0
        for group, values in grouped.items():
            values = [dict(r, support='UNKNOWN') if disputed_as_unknown and r['support'] == 'DISPUTED' else r for r in values]
            selected = [r for r in values if eligible(r)]
            n, d = sum(bool(event(r)) for r in selected), len(selected)
            group_rates[group] = n / d if d else None
            numerator += n; denominator += d
        rates = [v for v in group_rates.values() if v is not None]
        results[name] = {'estimate':sum(rates) / len(rates) if rates else None,
                         'numerator':numerator,'denominator':denominator,
                         'eligible_arrangements':len(rates),'arrangement_rates':group_rates}
    return results

def validate_split(rows):
    partitions = defaultdict(set)
    for row in rows: partitions[row['arrangement']].add(row['partition'])
    overlap = {key:value for key,value in partitions.items() if len(value) != 1}
    if overlap: raise ValueError(f'arrangements cross partitions: {overlap}')
