"""Analyze complete paired evidence using physical interventions only."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image
from development.cross_sim.scoring_prepare import PROTOCOL, STUDY, check_fixed_protocol
from development.stress.prepare_inputs import sha, load

CONTROLS = ('primary', 'relaxed_guard', 'native_normalized_guard')


def answer(record, thresholds):
    if record.get('failure') is not False: raise ValueError('failed evidence')
    e = record['evidence']; diag = record['diagnostics']
    if type(diag['align_ok']) is not bool: raise ValueError('invalid alignment')
    if record.get('refusal') not in {'accepted','correlation','shift','scale','optimizer_or_input'}: raise ValueError('invalid refusal')
    if any(type(e.get(k)) not in (int,float) or not math.isfinite(e[k]) or not 0 <= e[k] <= 1 for k in ('quality','match_score','obstruction')): raise ValueError('invalid score')
    if not diag['align_ok'] or e['quality'] < thresholds['quality']: return 'UNKNOWN'
    if e['match_score'] >= thresholds['present']: return 'PRESENT'
    if e['match_score'] <= thresholds['absent'] and e['obstruction'] <= thresholds['obstruction']: return 'REMOVED'
    return 'UNKNOWN'


def fraction(n, d): return {'numerator':n,'denominator':d,'rate':n/d if d else None}


def summarize(rows):
    total=len(rows); definite=[r for r in rows if r['answer'] != 'UNKNOWN']
    clear={}
    for state in ('clear_present','clear_removed'):
        sub=[r for r in rows if r['state']==state]
        clear[state]=fraction(sum(r['answer']==r['physical_state'] for r in sub),len(sub))
    placed=defaultdict(dict)
    for r in rows:
        if r['state'].startswith('placed_'): placed[r['pair_id']][r['state']]=r['answer']
    if any(set(pair) != {'placed_present','placed_removed'} for pair in placed.values()): raise ValueError('incomplete placed pair')
    placed_unknown = {}
    for state in ('placed_present','placed_removed'):
        subset = [r for r in rows if r['state'] == state]
        placed_unknown[state] = fraction(sum(r['answer'] == 'UNKNOWN' for r in subset), len(subset))
    return {'coverage':fraction(len(definite),total), 'correct_yield':fraction(sum(r['answer']==r['physical_state'] for r in rows),total),
            'physical_errors':sum(r['answer'] != r['physical_state'] for r in definite), 'clear_resolution':clear,
            'placed_unknown':placed_unknown,
            'placed_both_unknown':fraction(sum(all(v=='UNKNOWN' for v in pair.values()) for pair in placed.values()),len(placed))}


def clear_unknown_reason(row, thresholds):
    """Classify one clear abstention in the order of the frozen gates."""
    if not row['alignment_valid']:
        return 'clear_unresolved:alignment:' + row['refusal']
    if row['quality'] < thresholds['quality']:
        return 'clear_unresolved:quality_below_threshold'
    if thresholds['absent'] < row['target_match'] < thresholds['present']:
        return 'clear_unresolved:match_intermediate'
    if row['target_match'] <= thresholds['absent'] and row['obstruction'] > thresholds['obstruction']:
        return 'clear_unresolved:low_match_high_obstruction'
    raise ValueError('unexplained clear UNKNOWN')


def decide(rows_by_control):
    if set(rows_by_control) != set(CONTROLS): raise ValueError('all controls required')
    stops=[]
    for control,rows in rows_by_control.items():
        groups=defaultdict(list)
        for row in rows: groups[(row['engine'],row['environment'],row['level'])].append(row)
        expected={(e,v,l) for e in ('isaac','mujoco') for v in ('shelf','tabletop') for l in ('D0','D4','D6')}
        if set(groups) != expected or any(len(g) != 100 for g in groups.values()): raise ValueError('incomplete decision groups')
        for group in groups.values():
            bases = {r['base_id'] for r in group}
            if len(bases) != 5: raise ValueError('invalid arrangement count')
            expected_cases = {(b, p, state) for b in bases for p in ('P1','P2','P3','P4','P5')
                              for state in ('clear_present','clear_removed','placed_present','placed_removed')}
            actual_cases = [(r['base_id'],r['stop'],r['state']) for r in group]
            if len(set(actual_cases)) != 100 or set(actual_cases) != expected_cases: raise ValueError('incomplete arrangement cases')
        meets=all(all(v['rate'] is not None and v['rate'] >= .9 for v in summarize(group)['clear_resolution'].values()) for group in groups.values())
        if meets and not any(summarize(group)['physical_errors'] for group in groups.values()): stops.append(control)
    if stops: return {'outcome':'stop_added_veto_cascade_claim','controls':stops,'continuing_mechanisms':[]}
    # These are observable failure patterns, not established causes.
    # Correct abstention on a placed pair is retained in metrics but is not
    # evidence of a continuing mechanism.
    thresholds = load(PROTOCOL)['thresholds']
    indexed={}
    for control,rows in rows_by_control.items():
        patterns=defaultdict(set)
        for r in rows:
            if r['answer'] == 'UNKNOWN':
                if r['state'] not in ('clear_present', 'clear_removed'):
                    continue
                reason = clear_unknown_reason(r, thresholds)
            elif r['answer'] != r['physical_state']:
                reason='physical_error:'+r['answer']
            else:
                continue
            key=(r['engine'],r['environment'],r['level'],r['state'],reason)
            patterns[key].add(r['base_id'])
        indexed[control]=patterns
    found=[]
    for key in set.intersection(*(set(indexed[c]) for c in CONTROLS)):
        common=set.intersection(*(indexed[c][key] for c in CONTROLS))
        if len(common)>=3:
            found.append({'engine':key[0],'environment':key[1],'level':key[2],'state':key[3],
                          'observable_pattern':key[4],'arrangements':sorted(common)})
    return {'outcome':'continuing_mechanism' if found else 'inconclusive','controls':[], 'continuing_mechanisms':sorted(found,key=str)}


def analyze(prepared, run_dirs, out):
    check_fixed_protocol()
    prepared=Path(prepared).resolve(); pre=load(prepared/'preparation.json'); protocol=load(PROTOCOL)
    if set(run_dirs)!=set(CONTROLS) or pre.get('study_id')!=STUDY or pre.get('protocol_sha256')!=sha(PROTOCOL) or pre.get('count')!=1200 or pre.get('ledger_sha256')!=sha(prepared/'ledger.json') or pre.get('preparer_sha256')!=sha(Path(__file__).with_name('scoring_prepare.py')): raise ValueError('preparation mismatch')
    ledger=load(prepared/'ledger.json'); ids=[x['id'] for x in ledger]
    if len(ids)!=1200 or len(set(ids))!=1200: raise ValueError('ledger count or duplicates')
    keys=[(r['engine'],r['base_id'],r['level'],r['stop'],r['state']) for r in ledger]
    if len(set(keys)) != 1200: raise ValueError('duplicate cases')
    if any(r['state'] not in ('clear_present','clear_removed','placed_present','placed_removed') or r['stop'] not in ('P1','P2','P3','P4','P5') or r['physical_state'] != ('REMOVED' if r['state'].endswith('_removed') else 'PRESENT') for r in ledger): raise ValueError('intervention mismatch')
    # Reconstruct resized inputs from native images and check box scaling before analysis.
    native=prepared/'verifier/native'; primary=prepared/'verifier/primary'
    native_jobs=load(native/'jobs.json'); primary_jobs=load(primary/'jobs.json')
    nj={j['id']:j for j in native_jobs}; pj={j['id']:j for j in primary_jobs}
    if len(nj)!=1200 or len(pj)!=1200 or set(nj)!=set(ids) or set(pj)!=set(ids): raise ValueError('size job mismatch')
    for ident in ids:
        n=nj[ident]; p=pj[ident]
        if p['box'] != [round(x*[1200,675][i%2]/[2304,1296][i%2]) for i,x in enumerate(n['box'])]: raise ValueError('box scaling mismatch')
        for key in ('reference','query'):
            with Image.open(native/n[key]) as im:
                if im.size != (2304,1296) or im.mode != 'RGB' or im.info: raise ValueError('native input format mismatch')
                scaled=np.asarray(im.resize((1200,675),Image.Resampling.LANCZOS))
            with Image.open(primary/p[key]) as im:
                if im.size != (1200,675) or im.mode != 'RGB' or im.info or not np.array_equal(scaled,np.asarray(im)): raise ValueError('resized input mismatch')
    rows_by_control={}; certificates={}; metrics={}
    for control in CONTROLS:
        run_root=Path(run_dirs[control]).resolve(); cross=load(run_root/'cross_binding.json'); guard_files=list(run_root.glob('guard-*.json'))
        if cross.get('synthetic_self_test') is not False or cross.get('control') != control or cross.get('protocol_sha256') != sha(PROTOCOL) or cross.get('runner_sha256') != sha(Path(__file__).with_name('scoring_run.py')) or cross.get('guard_sha256') != sha(Path(__file__).resolve().parents[1]/'stress/offline_exec.py') or cross.get('clip_sha256') != '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af' or not guard_files: raise ValueError('cross-simulator guard binding mismatch')
        for gf in guard_files:
            guard=load(gf)
            if gf.name != 'guard-' + sha(gf) + '.json' or guard.get('guard_sha256') != cross.get('guard_sha256') or guard.get('clip_sha256') != cross.get('clip_sha256') or Path(guard.get('script','')).name != 'scoring_run.py': raise ValueError('invalid guard record')
        run=run_root/'evidence'; binding=load(run/'manifest.json'); size='native' if control=='native_normalized_guard' else 'primary'; folder=prepared/'verifier'/size; jobs_path=folder/'jobs.json'
        if binding.get('runner_sha256')!=sha(Path(__file__).resolve().parents[1]/'stress/run_evidence.py') or binding.get('analysis')!=control or binding.get('study_id')!=STUDY or binding.get('protocol_sha256')!=sha(PROTOCOL) or binding.get('preparation_sha256')!=sha(prepared/'preparation.json') or binding.get('jobs_sha256')!=sha(jobs_path) or pre['jobs_sha256'][size]!=sha(jobs_path): raise ValueError('evidence binding mismatch')
        jobs=load(jobs_path); byid={j['id']:j for j in jobs}
        if len(jobs)!=1200 or len(byid)!=1200 or set(byid)!=set(ids): raise ValueError('job mismatch')
        for job in jobs:
            for key in ('reference','query'):
                path=(folder/job[key]).resolve()
                if path.parent!=folder or sha(path)!=job[key+'_sha256']: raise ValueError('input changed')
        files=list((run/'records').glob('*.json'))
        if len(files)!=1200 or {f.stem for f in files}!=set(ids): raise ValueError('missing or extra records')
        records={f.stem:load(f) for f in files}; rows=[]
        for item in ledger:
            rec=records[item['id']]
            if rec.get('id')!=item['id'] or rec.get('binding_sha256')!=sha(run/'manifest.json'): raise ValueError('record binding mismatch')
            rows.append({**item,'answer':answer(rec,protocol['thresholds']),'refusal':rec['refusal'],
                         'alignment_valid':rec['diagnostics']['align_ok'], 'quality':rec['evidence']['quality'],
                         'target_match':rec['evidence']['match_score'], 'obstruction':rec['evidence']['obstruction']})
        rows_by_control[control]=rows
        grouped=defaultdict(list)
        for row in rows:
            grouped[(row['engine'],row['environment'],row['level'],row['base_id'])].append(row)
            grouped[(row['engine'],row['environment'],row['level'],'ALL')].append(row)
        metrics[control]=[{'engine':k[0],'environment':k[1],'level':k[2],'arrangement':k[3],**summarize(v)} for k,v in sorted(grouped.items())]
        pairs=defaultdict(dict)
        for item in ledger: pairs[item['pair_id']][item['state']]=item
        cert=[]
        for pair in pairs.values():
            if len(pair)!=2: raise ValueError('pair incomplete')
            a,b=pair.values(); ja,jb=byid[a['id']],byid[b['id']]
            def pixels(name):
                with Image.open(folder/name) as im: return np.asarray(im.convert('RGB')).copy()
            equal=(ja['box']==jb['box'] and np.array_equal(pixels(ja['reference']),pixels(jb['reference'])) and np.array_equal(pixels(ja['query']),pixels(jb['query'])))
            cert.append({'pair_id':a['pair_id'],'ids':[a['id'],b['id']],'equal_all_supplied_inputs':bool(equal)})
        certificates[control]=cert
    differences={}
    for control,records in metrics.items():
        lookup={(r['engine'],r['environment'],r['level'],r['arrangement']):r for r in records if r['arrangement'] != 'ALL'}
        deltas=[]
        for key,left in sorted(lookup.items()):
            if key[0] != 'isaac': continue
            right=lookup.get(('mujoco',*key[1:]))
            if right is None: raise ValueError('unpaired engine arrangement')
            delta={'environment':key[1],'level':key[2],'arrangement':key[3], 'direction':'mujoco_minus_isaac'}
            for metric in ('coverage','correct_yield','placed_both_unknown'):
                delta[metric]=right[metric]['rate'] - left[metric]['rate']
            delta['physical_errors']=right['physical_errors']-left['physical_errors']
            delta['clear_resolution']={state:right['clear_resolution'][state]['rate']-left['clear_resolution'][state]['rate'] for state in ('clear_present','clear_removed')}
            deltas.append(delta)
        differences[control]=deltas
    result={'study_id':STUDY,'preparation_sha256':sha(prepared/'preparation.json'),'control_manifest_sha256':{c:sha(Path(run_dirs[c])/'evidence'/'manifest.json') for c in CONTROLS},'metrics':metrics,'paired_engine_differences':differences,'certificates':certificates,'decision':decide(rows_by_control),'rows':rows_by_control}
    with Path(out).open('x') as f: json.dump(result,f,indent=2,allow_nan=False); f.write('\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepared',required=True); ap.add_argument('--primary',required=True); ap.add_argument('--relaxed-guard',required=True); ap.add_argument('--native-normalized-guard',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); analyze(a.prepared,{'primary':a.primary,'relaxed_guard':a.relaxed_guard,'native_normalized_guard':a.native_normalized_guard},a.out)
