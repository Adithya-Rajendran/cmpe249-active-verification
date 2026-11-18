"""Resumable development model runner. Requires a reviewed development manifest."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from model_scoring import score_candidates
from boundary import load_image, make_messages
from cohort import validate

MODELS={'qwen':('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
        'smol':('HuggingFaceTB/SmolVLM2-2.2B-Instruct','be92141c8d0ceb98596a923c308ac2bbb4a7c466')}
PROMPT=('The first image is a reference. The red box marks the remembered target. '
'The second image is a later view of the same scene. '
'The target was either left in its original region or removed from the scene. '
'Choose A for PRESENT, B for REMOVED, or C for UNKNOWN. '
'Choose PRESENT only if the image supports that the target remains in its original region. '
'Choose REMOVED only if the image supports absence from that region. '
'Choose UNKNOWN if the view does not support either decision. '
'A missed detection alone does not establish removal. '
'Earlier presence alone does not establish current presence behind a blocker. ')
PLACE=('First inspect the remembered region and decide which parts are visible or blocked. '
'Then inspect for evidence of the target and choose the supported answer code. ')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--images',required=True)
    parser.add_argument('--model',choices=MODELS,required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args(); manifest=json.loads(Path(args.manifest).read_text())
    validate(manifest, args.images)
    rows=manifest['rows']
    if any(r['support'] not in {'PRESENT','REMOVED','UNKNOWN','DISPUTED'} or r['reviewer_type']!='AI' for r in rows):
        raise ValueError('incomplete AI support review')
    model_id,revision=MODELS[args.model]
    binding=hashlib.sha256(Path(args.manifest).read_bytes()+Path(__file__).read_bytes()+Path(__file__).with_name('model_scoring.py').read_bytes()+Path(__file__).with_name('boundary.py').read_bytes()+Path(__file__).with_name('cohort.py').read_bytes()+revision.encode()).hexdigest()
    path=Path(args.out);done=set()
    if path.exists():
        for line in path.read_text().splitlines():
            item=json.loads(line)
            if item.get('binding') != binding: raise ValueError('resume binding mismatch')
            if item.get('id') not in range(108) or item.get('prompt_arm') not in ('base','place_first') or item.get('revision')!=revision: raise ValueError('invalid resume key')
            key=(item['id'],item['prompt_arm'])
            if key in done: raise ValueError('duplicate cached prediction')
            done.add(key)
    torch.cuda.reset_peak_memory_stats()
    t0=time.monotonic()
    processor=AutoProcessor.from_pretrained(model_id,revision=revision,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(model_id,revision=revision,local_files_only=True,dtype=torch.bfloat16).to('cuda:0').eval()
    load_seconds=time.monotonic()-t0
    load_peak_bytes=torch.cuda.max_memory_allocated()
    for row in rows:
        for arm in ('base','place_first'):
            if (row['id'],arm) in done: continue
            started=time.monotonic(); record={'id':row['id'],'prompt_arm':arm,'model':model_id,'revision':revision,'model_load_s':load_seconds,'binding':binding}
            try:
                images=[load_image(Path(args.images),row[k]) for k in ('reference','query')]
                prompt=PROMPT+(PLACE if arm=='place_first' else '')
                messages=make_messages(row['target_description'],prompt)
                inference_start=time.monotonic()
                result=score_candidates(model,processor,messages,images)
                record.update({'failure':False,'result':result,'three_candidate_inference_s':time.monotonic()-inference_start})
            except Exception as e:
                record.update({'failure':True,'result':None,'error_type':type(e).__name__,'error':str(e)})
            record['elapsed_s']=time.monotonic()-started
            record['gpu_peak_bytes_since_load']=torch.cuda.max_memory_allocated()
            record['model_load_peak_bytes']=load_peak_bytes
            with path.open('a') as f:
                f.write(json.dumps(record,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
if __name__=='__main__': main()
