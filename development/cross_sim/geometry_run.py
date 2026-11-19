"""Development-only comparison: one downstream evidence bridge, two registration providers."""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import time
from pathlib import Path
import cv2
import numpy as np
from perception import evidence as ev, evidence_v03 as v03
from verify.schema import EvidenceRecord
from development.stress.alignment_diagnostics import check_frozen, observe_fit, refusal_reason
from development.stress.prepare_inputs import sha, load
from development.stress.run_evidence import digest_tree

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'development/cross_sim/scoring_protocol.json'


def validate_config(config):
    if config.get('method')=='baseline_200':
        if set(config)!={'method'}: raise ValueError('baseline config has unexpected keys')
    elif config.get('method')=='geometry':
        if set(config)!={'method','provider','provider_config','min_target_valid_fraction'}: raise ValueError('geometry config keys mismatch')
        threshold=config['min_target_valid_fraction']
        if type(threshold) not in (int,float) or threshold!=0.60 or not isinstance(config['provider'],str) or ':' not in config['provider'] or not isinstance(config['provider_config'],dict): raise ValueError('invalid geometry config')
    else: raise ValueError('unknown method')


def box_valid_fraction(valid, box):
    height,width=valid.shape; mask=ev.box_mask(box,(0,0,width,height))
    if not mask.any(): raise ValueError('empty target box')
    return float(valid[mask].mean())


def downstream(ref,box,warped,nearest,valid,align_ok,encoder,detector,params,vparams,registration):
    """Apply frozen v0.2.1 cues and v0.3 background cue in original order."""
    box=tuple(float(x) for x in box)
    if ref.dtype!=np.uint8 or warped.dtype!=np.uint8 or nearest.dtype!=np.uint8 or ref.ndim!=3 or ref.shape[2]!=3 or warped.shape!=ref.shape or nearest.shape!=ref.shape or valid.dtype!=bool or valid.shape!=ref.shape[:2] or type(align_ok) is not bool:
        raise ValueError('invalid registered images or validity mask')
    h,w=ref.shape[:2]
    match,novelty=ev.match_and_novelty(ref[:,:,::-1],warped[:,:,::-1],box,encoder,params)
    changed_win=ev.window_change_mask(ref,warped,box,params)
    quality,qparts=ev.quality_score(ref,warped,valid,box,params,sharp_bgr=nearest,changed_win=changed_win)
    if not align_ok: quality=0.0
    x0,y0,x1,y1=ev.grow(box,params.det_context,w,h)
    # Detector boxes are crop-relative, in registered reference coordinates.
    dets=[ev.Detection((d.box[0]+x0,d.box[1]+y0,d.box[2]+x0,d.box[3]+y0),d.cls,d.conf)
          for d in detector(np.ascontiguousarray(warped[y0:y1,x0:x1]))]
    u,uparts=ev.obstruction_score(ev.change_mask(ref,warped,box,params),dets,box,params,novelty=novelty)
    u_bg,parts=v03.background_obstruction(ref,warped,valid,box,params,vparams)
    rec=EvidenceRecord(0,match,quality,max(u,u_bg))
    diag={'align_ok':align_ok,'quality_parts':qparts,'obstruction_parts':uparts,
          'detections':[{'box':[round(v,1) for v in d.box],'cls':d.cls,'conf':round(d.conf,3),'iou_box':round(ev.iou(d.box,box),3)} for d in dets],
          'obstruction_v021':u,'v03_parts':parts,'registration':registration}
    return rec,diag


class BaselineProvider:
    def __init__(self,align_params): self.align_params=align_params
    def register(self,ref,query,box):
        h,info,raw=observe_fit(v03,ref,query,tuple(box),self.align_params)
        linear,valid,ok=ev.register(ref,query,lambda r,q:h)
        nearest,_,_=ev.register(ref,query,lambda r,q:h,cv2.INTER_NEAREST)
        return linear,valid,{'warped_nearest_bgr':nearest,'registration_ok':ok,'reason':refusal_reason(info),
                             'registration':info,'raw_H_query_to_reference':None if raw is None else raw.tolist()}


def process(ref,query,box,provider,config,encoder,detector,params,vparams,thresholds,return_valid=False):
    warped,valid,details=provider.register(ref,query,box)
    if not isinstance(details,dict) or 'warped_nearest_bgr' not in details or type(details.get('registration_ok')) is not bool or not isinstance(details.get('reason'),str) or not details['reason']:
        raise ValueError('invalid provider diagnostics')
    nearest=details['warped_nearest_bgr']; valid=np.asarray(valid)
    if valid.dtype!=bool or valid.shape!=ref.shape[:2]: raise ValueError('provider validity must be a boolean image')
    fraction=box_valid_fraction(valid,box); registration={k:v for k,v in details.items() if k!='warped_nearest_bgr'}
    registration['target_valid_fraction']=fraction
    if config['method']=='baseline_200':
        align_ok=details['registration_ok']; reason=details['reason']
    else:
        align_ok=details['registration_ok'] and fraction>=config['min_target_valid_fraction']
        reason=details['reason'] if not details['registration_ok'] else ('accepted' if align_ok else 'target_valid_fraction_below_threshold')
    rec,diag=downstream(ref,box,warped,nearest,valid,align_ok,encoder,detector,params,vparams,registration)
    e=dataclasses.asdict(rec)
    values=[e[k] for k in ('match_score','quality','obstruction')]
    if any(not np.isfinite(v) or not 0<=v<=1 for v in values): raise ValueError('invalid evidence values')
    present=align_ok and e['quality']>=thresholds['quality'] and e['match_score']>=thresholds['present']
    removed=align_ok and e['quality']>=thresholds['quality'] and e['match_score']<=thresholds['absent'] and e['obstruction']<=thresholds['obstruction']
    if present and removed: raise ValueError('contradictory gates')
    record={'evidence':e,'diagnostics':diag,'refusal':reason,'answer':'PRESENT' if present else 'REMOVED' if removed else 'UNKNOWN','failure':False}
    return (record,valid) if return_valid else record



def artifact_bounds(box,shape):
    height,width=shape
    if len(box)!=4 or any(type(x) is not int for x in box): raise ValueError('artifact requires integer half-open box')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=width and 0<=y0<y1<=height): raise ValueError('invalid target box')
    return [max(0,x0-1),max(0,y0-1),min(width,x1+1),min(height,y1+1)]


def save_artifact(provider,ident,box,valid,folder,binding_sha):
    if not callable(getattr(provider,'geometry_artifact',None)): raise ValueError('provider lacks geometry_artifact')
    artifact=provider.geometry_artifact()
    if not isinstance(artifact,dict) or set(artifact)!={'bounds_xyxy','query_xy','valid'}: raise ValueError('invalid artifact fields')
    bounds=artifact_bounds(box,valid.shape)
    if list(artifact['bounds_xyxy'])!=bounds: raise ValueError('artifact bounds mismatch')
    x0,y0,x1,y1=bounds; query=np.asarray(artifact['query_xy']); mask=np.asarray(artifact['valid'])
    if query.shape!=(y1-y0,x1-x0,2) or query.dtype not in (np.float32,np.float64) or mask.shape!=(y1-y0,x1-x0) or mask.dtype!=bool or not np.array_equal(mask,valid[y0:y1,x0:x1]) or not np.isfinite(query[mask]).all(): raise ValueError('artifact data mismatch')
    query=query.astype(np.float32)
    if not np.isfinite(query[mask]).all(): raise ValueError('artifact conversion overflow')
    folder.mkdir(exist_ok=True); data_path=folder/(ident+'.npz'); meta_path=folder/(ident+'.json')
    if data_path.exists() or meta_path.exists(): raise FileExistsError('artifact already exists')
    temp=folder/(ident+'.npz.tmp')
    with temp.open('xb') as stream: np.savez_compressed(stream,query_xy=query,valid=mask)
    temp.replace(data_path)
    meta={'id':ident,'binding_sha256':binding_sha,'bounds_xyxy':bounds,'query_xy_dtype':'float32','valid_dtype':'bool','npz_sha256':sha(data_path)}
    temp=folder/(ident+'.json.tmp');temp.write_text(json.dumps(meta,sort_keys=True)+'\n');temp.replace(meta_path)


def verify_artifact(folder,ident,box,binding_sha):
    data=folder/(ident+'.npz'); meta=load(folder/(ident+'.json'))
    if meta.get('id')!=ident or meta.get('binding_sha256')!=binding_sha or meta.get('npz_sha256')!=sha(data) or meta.get('query_xy_dtype')!='float32' or meta.get('valid_dtype')!='bool': raise ValueError('artifact binding mismatch')
    bounds=artifact_bounds(box,(675,1200))
    if meta.get('bounds_xyxy')!=bounds: raise ValueError('artifact bounds mismatch')
    x0,y0,x1,y1=bounds
    with np.load(data,allow_pickle=False) as content:
        if set(content.files)!={'query_xy','valid'}: raise ValueError('invalid artifact array names')
        query=content['query_xy']; mask=content['valid']
        if query.shape!=(y1-y0,x1-x0,2) or query.dtype!=np.float32 or mask.shape!=(y1-y0,x1-x0) or mask.dtype!=bool or not np.isfinite(query[mask]).all(): raise ValueError('artifact payload mismatch')
    return meta

def verify_guard(path):
    guard=load(path); guard_file=ROOT/'development/stress/offline_exec.py'
    if (guard.get('guard_sha256')!=sha(guard_file) or Path(guard.get('script','')).resolve()!=Path(__file__).resolve() or
        guard.get('clip_sha256')!='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af' or
        any(os.environ.get(k)!=v for k,v in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','YOLO_AUTOINSTALL':'false'}.items())): raise ValueError('offline guard binding mismatch')
    for family,address in ((socket.AF_INET,('127.0.0.1',9)),(socket.AF_INET6,('::1',9))):
        try:
            with socket.socket(family) as sock: sock.connect(address)
        except PermissionError: continue
        except OSError as exc: raise ValueError('socket guard not active') from exc
        raise ValueError('socket guard not active')
    try: socket.getaddrinfo('localhost',80)
    except PermissionError: return guard
    raise ValueError('DNS guard not active')


def provider_from_config(config,align_params):
    if config['method']=='baseline_200': return BaselineProvider(align_params),None
    module_name,cls_name=config['provider'].split(':',1)
    module=importlib.import_module(module_name); module_file=Path(module.__file__).resolve()
    provider=getattr(module,cls_name)(config['provider_config'])
    if not callable(getattr(provider,'provenance',None)): raise ValueError('provider must supply provenance()')
    provenance=provider.provenance()
    if not isinstance(provenance,dict): raise ValueError('provider provenance must be a dict')
    json.dumps(provenance,allow_nan=False)
    return provider,{'module':module_name,'class':cls_name,'source':str(module_file),'source_sha256':sha(module_file),'provenance':provenance}


def run(jobs_path,config_path,output,device,guard_record,baseline_manifest,baseline_records=None):
    guard=verify_guard(guard_record); config=load(config_path); validate_config(config)
    jobs_path=Path(jobs_path).resolve(); output=Path(output).resolve(); jobs=load(jobs_path)
    if len(jobs)!=1200 or len({j['id'] for j in jobs})!=1200: raise ValueError('expected 1200 unique blinded jobs')
    for job in jobs:
        if set(job)!={'id','reference','query','box','reference_sha256','query_sha256'} or not job['id'].isalnum(): raise ValueError('invalid blinded job')
        for key in ('reference','query'):
            path=(jobs_path.parent/job[key]).resolve()
            if path.parent!=jobs_path.parent or sha(path)!=job[key+'_sha256']: raise ValueError('input changed')
    frozen=check_frozen(); proto=load(PROTOCOL); thresholds=proto['thresholds']
    params,vparams=ev.Params(),v03.V03Params(); align_params=v03.AlignParams(**{**proto['align_params'],'max_shift_px':200.0})
    reference=load(baseline_manifest)
    if reference.get('analysis')!='relaxed_guard' or reference.get('jobs_sha256')!=sha(jobs_path) or reference.get('frozen_hashes')!=frozen or reference.get('params')!=dataclasses.asdict(params) or reference.get('v03_params')!=dataclasses.asdict(vparams) or reference.get('align_params')!=dataclasses.asdict(align_params) or reference.get('thresholds')!=thresholds: raise ValueError('baseline evidence binding mismatch')
    if config['method']=='baseline_200' and baseline_records is None: raise ValueError('baseline replay requires original records for parity')
    if config['method']=='geometry' and baseline_records is not None: raise ValueError('geometry process must not receive baseline case records')
    baseline_digest=None
    if baseline_records is not None:
        baseline_records=Path(baseline_records).resolve(); files=list(baseline_records.glob('*.json'))
        if len(files)!=1200 or {f.stem for f in files}!={j['id'] for j in jobs}: raise ValueError('baseline records incomplete')
        inventory={f.name:sha(f) for f in sorted(files)}
        baseline_digest=hashlib.sha256(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    from huggingface_hub import snapshot_download
    import torch
    from perception.common import CHECKPOINTS
    snapshot=snapshot_download(ev.DINOV2_REPO,revision=ev.DINOV2_REVISION,local_files_only=True)
    weights={'dinov2':digest_tree(snapshot),'yolo':digest_tree(CHECKPOINTS/ev.DETECTOR_WEIGHTS),'clip':digest_tree(CHECKPOINTS/'clip/ViT-B-32.pt')}
    if weights!=reference['checkpoints']: raise ValueError('checkpoint identity differs from original')
    versions={}
    for name in ('numpy','opencv-python','torch','transformers','ultralytics','huggingface-hub','Pillow'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    actual_gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') and torch.cuda.is_available() else None
    if (versions!=reference['versions'] or platform.python_version()!=reference['python'] or torch.version.cuda!=reference['torch_cuda']
        or device!=reference['device'] or actual_gpu!=reference['gpu']): raise ValueError('runtime differs from original baseline')
    provider,provider_binding=provider_from_config(config,align_params)
    binding={'study_id':'cross-sim-geometry-development-v1','config_sha256':sha(config_path),'config':config,'jobs_sha256':sha(jobs_path),
             'reference_manifest_sha256':sha(baseline_manifest),'baseline_record_inventory_sha256':baseline_digest,'source_sha256':sha(__file__),'protocol_sha256':sha(PROTOCOL),
             'guard_sha256':guard['guard_sha256'],'clip_sha256':guard['clip_sha256'],'frozen_hashes':frozen,'provider':provider_binding,
             'params':dataclasses.asdict(params),'v03_params':dataclasses.asdict(vparams),'align_params':dataclasses.asdict(align_params),'thresholds':thresholds,
             'checkpoints':weights,'device':device,'python':platform.python_version(),'versions':versions,'torch_cuda':torch.version.cuda,
             'gpu':actual_gpu}
    output.mkdir(parents=True,exist_ok=True); manifest=output/'manifest.json'
    if manifest.exists():
        if load(manifest)!=binding: raise ValueError('resume binding mismatch')
    else:
        if any(output.iterdir()): raise ValueError('output exists without binding')
        manifest.write_text(json.dumps(binding,indent=2,sort_keys=True,allow_nan=False)+'\n')
    guard_copy=output/('guard-'+sha(guard_record)+'.json')
    if not guard_copy.exists(): guard_copy.write_bytes(Path(guard_record).read_bytes())
    records=output/'records';records.mkdir(exist_ok=True)
    ids={j['id'] for j in jobs}; existing=list(records.glob('*.json'))
    if not {p.stem for p in existing}<=ids: raise ValueError('unexpected record')
    for f in existing:
        r=load(f)
        stored=r.pop('record_payload_sha256',None)
        payload=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if stored!=payload or r.get('id')!=f.stem or r.get('binding_sha256')!=sha(manifest) or r.get('failure') is not False: raise ValueError('existing record mismatch')
    if any(records.glob('*.tmp')): raise ValueError('unfinished record temporary file')
    artifacts=output/'artifacts'
    if config['method']=='geometry':
        artifacts.mkdir(exist_ok=True)
        stems={p.stem for p in existing}
        if {p.stem for p in artifacts.glob('*.npz')}!=stems or {p.stem for p in artifacts.glob('*.json')}!=stems or any(artifacts.glob('*.tmp')): raise ValueError('missing or orphan geometry artifact')
        byid={j['id']:j for j in jobs}
        for ident in stems: verify_artifact(artifacts,ident,byid[ident]['box'],sha(manifest))
    elif artifacts.exists(): raise ValueError('baseline output has geometry artifacts')
    pending=[j for j in jobs if not (records/(j['id']+'.json')).exists()]
    if not pending:return
    encoder,detector=ev.DinoV2Patches(device),ev.YoloWorld(device,p=params)
    for job in pending:
        ref=ev.read_bgr(jobs_path.parent/job['reference']); query=ev.read_bgr(jobs_path.parent/job['query'])
        if ref.shape!=(675,1200,3) or query.shape!=ref.shape: raise ValueError('input dimensions mismatch')
        start=time.monotonic(); record,registered_valid=process(ref,query,job['box'],provider,config,encoder,detector,params,vparams,thresholds,return_valid=True)
        record.update({'id':job['id'],'binding_sha256':sha(manifest),'reference_sha256':job['reference_sha256'],'query_sha256':job['query_sha256'],'runtime_seconds':time.monotonic()-start})
        if config['method']=='baseline_200':
            old=load(baseline_records/(job['id']+'.json'))
            if old.get('id')!=job['id'] or old.get('binding_sha256')!=sha(baseline_manifest) or old.get('failure') is not False: raise ValueError('baseline record binding mismatch')
            if any(abs(record['evidence'][k]-old['evidence'][k])>1e-6 for k in ('match_score','quality','obstruction')) or record['diagnostics']['align_ok']!=old['diagnostics']['align_ok'] or record['refusal']!=old['refusal']:
                raise ValueError(f'baseline replay differs: {job["id"]}')
            oe=old['evidence']; ok=old['diagnostics']['align_ok']; p=ok and oe['quality']>=thresholds['quality'] and oe['match_score']>=thresholds['present']; a=ok and oe['quality']>=thresholds['quality'] and oe['match_score']<=thresholds['absent'] and oe['obstruction']<=thresholds['obstruction']
            expected='PRESENT' if p else 'REMOVED' if a else 'UNKNOWN'
            if record['answer']!=expected: raise ValueError('baseline decision differs')
            record['baseline_parity']='PASS'
        if config['method']=='geometry': save_artifact(provider,job['id'],job['box'],registered_valid,artifacts,sha(manifest))
        record=json.loads(json.dumps(record,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))
        record['record_payload_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        temp=records/(job['id']+'.tmp'); temp.write_text(json.dumps(record,allow_nan=False)+'\n'); temp.replace(records/(job['id']+'.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--guard-record',required=True);p.add_argument('--baseline-manifest',required=True);p.add_argument('--baseline-records')
    a=p.parse_args();run(a.jobs,a.config,a.out,a.device,a.guard_record,a.baseline_manifest,a.baseline_records)
