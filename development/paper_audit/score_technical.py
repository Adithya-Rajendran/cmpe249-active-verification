"""Replay the frozen 200-pixel and VGGT evidence rules on blinded RGB inputs."""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import platform
import json
import os
import time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['YOLO_AUTOINSTALL']='false'
import cv2
import numpy as np
import torch
from perception import evidence as ev, evidence_v03 as v03
from development.cross_sim import geometry_run as g
from development.paper_audit.score_registered import load_cases, sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=('baseline_200','geometry'),required=True)
    p.add_argument('--geometry-config',type=Path,required=True)
    p.add_argument('--baseline-manifest',type=Path,required=True)
    a=p.parse_args()
    cases=load_cases(a.cases.resolve())
    config={'method':'baseline_200'} if a.method=='baseline_200' else json.loads(a.geometry_config.read_text())
    g.validate_config(config)
    proto=json.loads(g.PROTOCOL.read_text())
    params,vparams=ev.Params(),v03.V03Params()
    align=v03.AlignParams(**{**proto['align_params'],'max_shift_px':200.0})
    if a.out.exists(): raise FileExistsError('output exists')
    a.out.mkdir(parents=True)
    torch.set_num_threads(4); cv2.setNumThreads(4)
    reference=json.loads(a.baseline_manifest.read_text())
    versions={name:importlib.metadata.version(name) for name in ('numpy','opencv-python','torch','transformers','ultralytics','huggingface-hub','Pillow')}
    if versions!=reference['versions'] or platform.python_version()!=reference['python'] or torch.version.cuda!=reference['torch_cuda'] or torch.cuda.get_device_name()!=reference['gpu']: raise ValueError('historical runtime mismatch')
    from development.stress.alignment_diagnostics import check_frozen
    if check_frozen()!=reference['frozen_hashes']: raise ValueError('frozen source mismatch')
    if sha(Path(g.__file__))!='2033a30ad44c4416e689b642e7882f07ae3b7a23d01ee4bebd9a2cca2baec66b': raise ValueError('shared scoring source mismatch')
    from huggingface_hub import snapshot_download
    from perception.common import CHECKPOINTS
    from development.stress.run_evidence import digest_tree
    snapshot=snapshot_download(ev.DINOV2_REPO,revision=ev.DINOV2_REVISION,local_files_only=True)
    weights={'dinov2':digest_tree(snapshot),'yolo':digest_tree(CHECKPOINTS/ev.DETECTOR_WEIGHTS),'clip':digest_tree(CHECKPOINTS/'clip/ViT-B-32.pt')}
    if weights != reference['checkpoints']: raise ValueError('frozen baseline checkpoint mismatch')
    if (dataclasses.asdict(params)!=reference['params'] or dataclasses.asdict(vparams)!=reference['v03_params'] or dataclasses.asdict(align)!=reference['align_params'] or proto['thresholds']!=reference['thresholds']): raise ValueError('frozen baseline parameters mismatch')
    provider,source=g.provider_from_config(config,align)
    binding={'script_sha256':sha(__file__),'cross_sim_scorer_sha256':sha(Path(g.__file__)),
             'protocol_sha256':sha(g.PROTOCOL),'geometry_config_sha256':sha(a.geometry_config),
             'cases_sha256':sha(a.cases),'method':a.method,'provider':source,'checkpoint_digests':weights,'baseline_manifest_sha256':sha(a.baseline_manifest),
             'params':dataclasses.asdict(params),'v03_params':dataclasses.asdict(vparams),
             'align_params':dataclasses.asdict(align),'thresholds':proto['thresholds'],
             'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    (a.out/'binding.json').write_text(json.dumps(binding,indent=2,sort_keys=True)+'\n')
    encoder,detector=ev.DinoV2Patches('cuda'),ev.YoloWorld('cuda',p=params)
    with (a.out/'records.jsonl').open('x',buffering=1) as stream:
        for case in cases:
            started=time.monotonic()
            record={'id':case['id'],'reference_sha256':case['reference_sha256'],'query_sha256':case['query_sha256'],
                    'box_xyxy':case['box_xyxy'],'answer':'UNKNOWN','failure':None}
            try:
                ref=ev.read_bgr(a.cases.parent/case['reference']); query=ev.read_bgr(a.cases.parent/case['query'])
                if ref.shape!=(675,1200,3) or query.shape!=ref.shape: raise ValueError('expected 1200x675 inputs')
                result=g.process(ref,query,case['box_xyxy'],provider,config,encoder,detector,params,vparams,proto['thresholds'])
                record.update(result)
            except Exception as exc:
                record['failure']=type(exc).__name__+': '+str(exc)
            record['runtime_seconds']=time.monotonic()-started
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')

if __name__=='__main__': main()
