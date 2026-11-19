"""Export a completed paired analysis to a CSV and an aggregate-count figure."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CONTROLS = ('primary', 'relaxed_guard', 'native_normalized_guard')
CONTROL_NAMES = {'primary': 'Primary 100 px', 'relaxed_guard': 'Primary 200 px', 'native_normalized_guard': 'Native 192 px'}
ENGINES = ('isaac', 'mujoco')
ENVIRONMENTS = ('shelf', 'tabletop')
LEVELS = ('D0', 'D4', 'D6')
STATES = ('clear_present', 'clear_removed', 'placed_present', 'placed_removed')
STOPS = ('P1', 'P2', 'P3', 'P4', 'P5')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(data):
    if data.get('study_id') != 'cross-sim-dev-v1' or set(data.get('rows', {})) != set(CONTROLS):
        raise ValueError('expected complete cross-simulator analysis')
    reference = None
    for control in CONTROLS:
        rows = data['rows'][control]
        if len(rows) != 1200 or len({r['id'] for r in rows}) != 1200:
            raise ValueError('analysis must contain 1200 unique rows per control')
        cells = defaultdict(list)
        cases = {}
        for r in rows:
            key = (r['engine'], r['environment'], r['level'], r['base_id'], r['stop'], r['state'])
            if (r['engine'] not in ENGINES or r['environment'] not in ENVIRONMENTS or r['level'] not in LEVELS
                or r['stop'] not in STOPS or r['state'] not in STATES or r['answer'] not in ('PRESENT','REMOVED','UNKNOWN')
                or r['physical_state'] != ('PRESENT' if r['state'].endswith('_present') else 'REMOVED') or key in cases):
                raise ValueError('invalid or duplicate case')
            cases[key] = (r['id'], r['pair_id'])
            cells[(r['engine'],r['environment'],r['level'],r['base_id'])].append(r)
        combos = {(e,v,l) for e in ENGINES for v in ENVIRONMENTS for l in LEVELS}
        if {(e,v,l) for e,v,l,_ in cells} != combos or any(len(group) != 20 for group in cells.values()):
            raise ValueError('incomplete arrangement')
        for e,v,l in combos:
            group = [k[3] for k in cells if k[:3] == (e,v,l)]
            if len(group) != 5: raise ValueError('five arrangements required per environment and level')
        for group in cells.values():
            if {(r['stop'],r['state']) for r in group} != {(s,t) for s in STOPS for t in STATES}:
                raise ValueError('arrangement lacks a stop or state')
        pair_cases = defaultdict(list)
        for r in rows: pair_cases[r['pair_id']].append(r)
        for group in pair_cases.values():
            if (len(group) != 2 or len({(r['engine'],r['environment'],r['level'],r['base_id'],r['stop'],r['state'].split('_')[0]) for r in group}) != 1
                or {r['physical_state'] for r in group} != {'PRESENT','REMOVED'}):
                raise ValueError('invalid matched pair')
        if reference is not None and cases != reference: raise ValueError('control cases differ')
        reference = cases
    # Each engine must use the same base IDs in every level.
    for engine in ENGINES:
        for environment in ENVIRONMENTS:
            groups = [{r['base_id'] for r in data['rows']['primary'] if r['engine']==engine and r['environment']==environment and r['level']==level} for level in LEVELS]
            if any(group != groups[0] for group in groups): raise ValueError('unpaired levels')
            other = 'mujoco' if engine == 'isaac' else 'isaac'
            if groups[0] != {r['base_id'] for r in data['rows']['primary'] if r['engine']==other and r['environment']==environment and r['level']=='D0'}: raise ValueError('unpaired engines')


def summary(rows):
    total=len(rows); definite=sum(r['answer']!='UNKNOWN' for r in rows)
    result={'denominator':total,'definite':definite,'unknown':total-definite,
            'correct':sum(r['answer']==r['physical_state'] for r in rows),
            'physical_errors':sum(r['answer'] not in (r['physical_state'],'UNKNOWN') for r in rows)}
    for state in STATES:
        part=[r for r in rows if r['state']==state]
        result[state+'_denominator']=len(part)
        if state.startswith('clear_'):
            result[state+'_correct']=sum(r['answer']==r['physical_state'] for r in part)
            result[state+'_resolution']=result[state+'_correct']/len(part) if part else None
        else:
            result[state+'_unknown']=sum(r['answer']=='UNKNOWN' for r in part)
            result[state+'_unknown_rate']=result[state+'_unknown']/len(part) if part else None
    pairs=defaultdict(dict)
    for r in rows:
        if r['state'].startswith('placed_'):
            if r['state'] in pairs[r['pair_id']]: raise ValueError('duplicate placed pair')
            pairs[r['pair_id']][r['state']]=r['answer']
    if any(set(v)!={'placed_present','placed_removed'} for v in pairs.values()): raise ValueError('incomplete placed pair')
    result['placed_pair_denominator']=len(pairs)
    result['placed_both_unknown']=sum(all(answer=='UNKNOWN' for answer in values.values()) for values in pairs.values())
    result['placed_both_unknown_rate']=result['placed_both_unknown']/len(pairs) if pairs else None
    return result


def build_rows(data):
    validate(data); output=[]
    for control in CONTROLS:
        groups=defaultdict(list)
        for r in data['rows'][control]:
            prefix=(r['engine'],r['environment'],r['level'])
            groups[(*prefix,r['base_id'])].append(r)
            groups[(*prefix,'ALL')].append(r)
        for key,values in sorted(groups.items()):
            output.append({'control':control,'engine':key[0],'environment':key[1],'level':key[2],
                           'scope':'aggregate' if key[3]=='ALL' else 'arrangement','arrangement':key[3],**summary(values)})
    return output


def plot(rows, output, source_hash, synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    lookup={(r['control'],r['engine'],r['environment'],r['level']):r for r in rows if r['scope']=='aggregate'}
    colors=('#27648d','#d18921','#7a5793')
    fig,axes=plt.subplots(4,2,figsize=(13.6,13.0),sharey=True)
    for row,(environment,state) in enumerate((('shelf','clear_present'),('shelf','clear_removed'),('tabletop','clear_present'),('tabletop','clear_removed'))):
        for col,engine in enumerate(ENGINES):
            ax=axes[row,col]
            for k,control in enumerate(CONTROLS):
                values=[]
                for level in LEVELS:
                    cell=lookup[(control,engine,environment,level)]
                    if cell[state+'_denominator']!=25: raise ValueError('clear aggregate denominator is not 25')
                    values.append(cell[state+'_correct'])
                positions=[x+(k-1)*.24 for x in range(3)]
                bars=ax.bar(positions,values,width=.22,color=colors[k],edgecolor='white',linewidth=.5)
                for bar,count in zip(bars,values): ax.text(bar.get_x()+bar.get_width()/2,count+.42,str(count),ha='center',va='bottom',fontsize=8)
            ax.set_ylim(0,28); ax.set_yticks([0,5,10,15,20,25]); ax.grid(axis='y',alpha=.22); ax.set_axisbelow(True)
            ax.set_xticks(range(3)); ax.set_xticklabels(LEVELS)
            ax.set_title(f'{environment.title()} · {state.replace("_", " ").title()} · {engine.title()}',fontsize=10,loc='left')
            if col==0: ax.set_ylabel('Correct decisions / 25',fontsize=9)
    fig.subplots_adjust(left=.075,right=.985,top=.93,bottom=.13,hspace=.52,wspace=.10)
    fig.suptitle(('SYNTHETIC TEST DATA — ' if synthetic else '')+'Clear-state correct decisions by control',fontsize=15,fontweight='bold')
    fig.legend(handles=[Patch(color=colors[i],label=CONTROL_NAMES[c]) for i,c in enumerate(CONTROLS)],loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.055))
    fig.text(.01,.02,'Each bar aggregates 25 views from five arrangements. Views are not independent.  Source SHA-256: '+source_hash,fontsize=7)
    fig.savefig(output,dpi=180,bbox_inches='tight',metadata={'Source SHA256':source_hash, 'Synthetic':str(synthetic)})
    plt.close(fig)
    return matplotlib.__version__


def export(analysis, destination, synthetic=False):
    analysis=Path(analysis).resolve(); destination=Path(destination).resolve(); data=json.loads(analysis.read_text())
    if bool(data.get('synthetic',False)) != synthetic: raise ValueError('synthetic input requires explicit matching --synthetic flag')
    rows=build_rows(data); source=digest(analysis)
    if destination.exists(): raise FileExistsError('output directory exists')
    destination.mkdir(parents=True)
    with (destination/'arrangement_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    version=plot(rows,destination/'clear_correct_counts.png',source,synthetic)
    files={p.name:digest(p) for p in (destination/'arrangement_metrics.csv',destination/'clear_correct_counts.png')}
    record={'study_id':data['study_id'],'synthetic':synthetic,'analysis_path':str(analysis),'analysis_sha256':source,
            'exporter_sha256':digest(__file__),'matplotlib_version':version,'outputs_sha256':files,
            'figure':'Clear-state correct counts; 25 per aggregate cell, five arrangements per cell.','csv_rows':len(rows)}
    (destination/'export_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--analysis',required=True); parser.add_argument('--out-dir',required=True); parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args(); print(json.dumps(export(args.analysis,args.out_dir,args.synthetic),indent=2))
