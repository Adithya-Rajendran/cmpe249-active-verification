"""Evaluator-only target-landmark residuals for completed pilot evidence."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from development.cross_sim import qualification_geometry as geometry

CONTROLS=('primary','relaxed_guard','native_normalized_guard')
SIZES={'primary':(1200,675),'relaxed_guard':(1200,675),'native_normalized_guard':(2304,1296)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def sample_index(points, size, native=(2304,1296)):
    return np.asarray(points,float)*np.asarray(size,float)/np.asarray(native,float)-.5


def residuals(raw_h, query, reference, box):
    """Compare fitted query→reference points to true target reference points."""
    if raw_h is None: return {'status':'missing_matrix'}
    h=np.asarray(raw_h,float); query=np.asarray(query,float); reference=np.asarray(reference,float)
    if h.shape!=(3,3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h)!=3: return {'status':'invalid_matrix'}
    if query.ndim!=2 or query.shape[1]!=2 or query.shape!=reference.shape or not len(query): raise ValueError('invalid landmark points')
    diag=float(np.hypot(box[2]-box[0],box[3]-box[1]))
    if not np.isfinite(diag) or diag<=0: raise ValueError('invalid reference box diagonal')
    homogeneous=np.column_stack((query,np.ones(len(query))))@h.T
    if np.any(~np.isfinite(homogeneous)) or np.any(np.abs(homogeneous[:,2])<1e-12): return {'status':'invalid_projection'}
    mapped=homogeneous[:,:2]/homogeneous[:,2,None]; errors=np.linalg.norm(mapped-reference,axis=1)
    if not np.isfinite(errors).all(): return {'status':'invalid_projection'}
    median,p90=np.percentile(errors,[50,90])
    return {'status':'available','landmark_count':len(errors),'reference_box_diagonal_px':diag,
            'median_px':float(median),'p90_px':float(p90),'median_box_fraction':float(median/diag),'p90_box_fraction':float(p90/diag)}


def summarize(records):
    dimensions=('control','engine','environment','refusal','outcome','visibility_class')
    axes=[(),*( (name,) for name in dimensions),dimensions]
    results=[]
    for axis in axes:
        groups=defaultdict(list)
        for r in records: groups[tuple(r[name] for name in axis)].append(r)
        for key,rows in sorted(groups.items()):
            available=[r for r in rows if r['fit']['status']=='available']
            result={'group_by':list(axis),**dict(zip(axis,key))}
            result.update({'cases':len(rows),'available':len(available),'missing_matrix':sum(r['fit']['status']=='missing_matrix' for r in rows),
                           'invalid_matrix':sum(r['fit']['status']=='invalid_matrix' for r in rows),
                           'invalid_projection':sum(r['fit']['status']=='invalid_projection' for r in rows)})
            for metric in ('median_px','p90_px','median_box_fraction','p90_box_fraction'):
                values=[r['fit'][metric] for r in available]
                result[metric]={'case_median':float(np.median(values)) if values else None,'case_p90':float(np.percentile(values,90)) if values else None}
            results.append(result)
    return results


def run(analysis_path, cohort_path, prepared, controls, assets, out):
    analysis_path,cohort_path,prepared,controls,assets,out=map(lambda x:Path(x).resolve(),(analysis_path,cohort_path,prepared,controls,assets,out))
    analysis=load(analysis_path); cohort=load(cohort_path); pre=load(prepared/'preparation.json')
    if analysis.get('study_id')!='cross-sim-dev-v1' or cohort.get('split')!='pilot' or cohort.get('study_id')!=analysis['study_id'] or set(analysis.get('rows',{}))!=set(CONTROLS): raise ValueError('study or controls mismatch')
    if analysis.get('preparation_sha256')!=sha(prepared/'preparation.json') or pre.get('cohort_sha256')!=sha(cohort_path) or pre.get('ledger_sha256')!=sha(prepared/'ledger.json'): raise ValueError('analysis/input bindings differ')
    ledger=load(prepared/'ledger.json'); ledger_by_id={r['id']:r for r in ledger}
    if len(ledger)!=1200 or len(ledger_by_id)!=1200: raise ValueError('incomplete ledger')
    arrangements={x['base_id']:x for x in cohort['arrangements']}
    if len(arrangements)!=10: raise ValueError('incomplete cohort')
    # The full vertex list is fixed by the original source PLY or source scene mesh.
    target_by_base={}; cameras={}; asset_hashes={}; arrangement_hashes={}
    for base,item in arrangements.items():
        for level,entry in item['levels'].items():
            path=(cohort_path.parent/entry['path']).resolve()
            if path.parent!=cohort_path.parent or sha(path)!=entry['sha256']: raise ValueError('arrangement input hash mismatch')
            arr=load(path); arrangement_hashes[str(path)]=entry['sha256']
            if base not in target_by_base:
                target=arr['objects']['target']
                if target['kind']=='ycbv':
                    source=assets/target['ply']; asset_hashes[str(source)]=sha(source)
                    target_by_base[base]=geometry.object_mesh(target,assets)['vertices']
                else:
                    target_by_base[base]=geometry.source_geometry(arr,assets)['target']['vertices']
            for stop in cohort['stops']:
                ref,_=geometry.project(target_by_base[base],geometry.camera_record(arr,stop))
                qry,_=geometry.project(target_by_base[base],geometry.camera_record(arr,stop,revisit=True))
                cameras[(base,level,stop)]=(ref,qry)
    records=[]; bindings={}
    for control in CONTROLS:
        root=controls/control/'evidence'; binding_path=root/'manifest.json'; binding=load(binding_path)
        if analysis['control_manifest_sha256'][control]!=sha(binding_path) or binding.get('analysis')!=control or binding.get('preparation_sha256')!=sha(prepared/'preparation.json'): raise ValueError('evidence binding mismatch')
        size='native' if control=='native_normalized_guard' else 'primary'; jobs_path=prepared/'verifier'/size/'jobs.json'
        if pre['jobs_sha256'][size]!=sha(jobs_path) or binding['jobs_sha256']!=sha(jobs_path): raise ValueError('job binding mismatch')
        joblist=load(jobs_path); jobs={j['id']:j for j in joblist}; rows=analysis['rows'][control]
        if len(rows)!=1200 or len({r['id'] for r in rows})!=1200 or len(jobs)!=1200 or set(jobs)!=set(ledger_by_id) or {r['id'] for r in rows}!=set(jobs): raise ValueError('incomplete control')
        files=list((root/'records').glob('*.json'))
        if len(files)!=1200 or {p.stem for p in files}!=set(jobs): raise ValueError('evidence files incomplete')
        bindings[control]={'evidence_manifest_sha256':sha(binding_path),'jobs_sha256':sha(jobs_path)}
        for row in rows:
            ident=row['id']; original=ledger_by_id[ident]
            for key in ('engine','environment','base_id','level','stop','state','physical_state'):
                if row[key]!=original[key]: raise ValueError('analysis ledger mismatch')
            evidence_path=root/'records'/f'{ident}.json'; evidence=load(evidence_path)
            if evidence.get('id')!=ident or evidence.get('failure') is not False or evidence.get('binding_sha256')!=sha(binding_path) or evidence.get('refusal')!=row['refusal']: raise ValueError('evidence record mismatch')
            ref,qry=cameras[(row['base_id'],row['level'],row['stop'])]; size_xy=SIZES[control]
            fit=residuals(evidence.get('raw_H_query_to_reference'),sample_index(qry,size_xy),sample_index(ref,size_xy),jobs[ident]['box'])
            outcome='unknown' if row['answer']=='UNKNOWN' else ('correct' if row['answer']==row['physical_state'] else 'physical_error')
            records.append({'id':ident,'control':control,'engine':row['engine'],'environment':row['environment'],'base_id':row['base_id'],'level':row['level'],'stop':row['stop'],'state':row['state'],
                            'visibility_class':'clear' if row['state'].startswith('clear_') else 'placed','physical_state':row['physical_state'],'answer':row['answer'],'outcome':outcome,'refusal':row['refusal'],
                            'evidence_sha256':sha(evidence_path),'fit':fit})
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    result={'purpose':'evaluator-only post-hoc geometry diagnostic; no threshold or decision changes',
            'landmarks':'all original source target mesh vertices, including occluded or removed vertices; not uniformly area weighted',
            'direction':'raw query-to-reference homography applied to true query projections, compared to true reference projections',
            'coordinates':'native continuous pixels mapped to control OpenCV sample indices as u*scale_x-0.5, v*scale_y-0.5',
            'normalization':'diagonal of the actual control reference box',
            'analysis_sha256':sha(analysis_path),'cohort_sha256':sha(cohort_path),'preparation_sha256':sha(prepared/'preparation.json'),'ledger_sha256':sha(prepared/'ledger.json'),
            'helper_sha256':sha(__file__),'geometry_helper_sha256':sha(geometry.__file__),'source_assets_sha256':asset_hashes,'arrangement_sha256':arrangement_hashes,'bindings':bindings,
            'case_count':len(records),'summary':summarize(records),'records':records}
    (out/'landmark_residuals.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fields=['id','control','engine','environment','base_id','level','stop','state','visibility_class','physical_state','answer','outcome','refusal','evidence_sha256',
            'status','landmark_count','reference_box_diagonal_px','median_px','p90_px','median_box_fraction','p90_box_fraction']
    with (out/'landmark_cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for record in records:
            writer.writerow({**{k:v for k,v in record.items() if k!='fit'},**record['fit']})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--analysis',required=True); p.add_argument('--cohort',required=True); p.add_argument('--prepared',required=True); p.add_argument('--controls',required=True); p.add_argument('--assets',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); result=run(a.analysis,a.cohort,a.prepared,a.controls,a.assets,a.out); print(json.dumps({'case_count':result['case_count'],'output':str(Path(a.out)/'landmark_residuals.json')}))
