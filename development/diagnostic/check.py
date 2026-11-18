"""Post-hoc calibration-only diagnosis of model input and scoring."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, '/tmp/cmpe249-pilot-run/development/pilot')
from run_models import MODELS, PROMPT
from boundary import load_image, make_messages

ROOT=Path('/tmp/cmpe249-support-dev-v1')
MANIFEST=Path('/tmp/cmpe249-pilot-run/runs/2026-11-16-support-audit-v1/pilot_manifest.json')
OUT=Path('/tmp/cmpe249-support-diagnostic-v1')
LITERAL=('The first image is a reference. The red box marks a mug. The second image is a later view of the same scene. '
'Is the mug still in its original region, removed from that region, or is the image insufficient to decide? '
'Reply PRESENT, REMOVED, or UNKNOWN.')
DESCRIPTION='Describe the visible objects in this image briefly.'

def digest(data): return hashlib.sha256(data).hexdigest()
def message(text,n): return [{'role':'user','content':[{'type':'image'} for _ in range(n)]+[{'type':'text','text':text}]}]
def prepare(processor,messages,images):
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=processor(text=[text],images=images,return_tensors='pt').to('cuda:0')
    hashes={}
    for k,v in inputs.items():
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy()
            hashes[k]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':digest(a.tobytes())}
    return text,inputs,hashes

def append_record(name,record):
    with (OUT/f'{name}.jsonl').open('a') as f:
        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()

def generate(model,processor,name,row,kind,messages,images):
    rendered,inputs,hashes=prepare(processor,messages,images);start=time.monotonic()
    with torch.inference_mode():
        out=model.generate(**inputs,max_new_tokens=48,do_sample=False)
    ids=out[0,inputs['input_ids'].shape[1]:].tolist()
    append_record(name,{'row_id':row,'check':kind,'rendered_template':rendered,
        'input_ids':inputs['input_ids'][0].tolist(),'tensor_hashes':hashes,'generated_token_ids':ids,
        'raw_output':processor.tokenizer.decode(ids,skip_special_tokens=False),
        'decoded_output':processor.tokenizer.decode(ids,skip_special_tokens=True),'elapsed_s':time.monotonic()-start})

def scoring(model,processor,name,row,messages,images):
    text,inputs,hashes=prepare(processor,messages,images);tok=processor.tokenizer
    base=tok.encode(text,add_special_tokens=False);tail=base[-8:]
    if inputs['input_ids'][0,-len(tail):].tolist()!=tail: raise ValueError('prefix mismatch')
    n=inputs['input_ids'].shape[1]
    with torch.inference_mode(): prefix=model(**inputs).logits[0,n-1].float().log_softmax(dim=-1).cpu()
    np.save(OUT/f'{name}-{row}-prefix.npy',prefix.numpy())
    comparisons={}
    for code in 'ABC':
        full=processor.apply_chat_template(messages+[{'role':'assistant','content':[{'type':'text','text':code}]}],tokenize=False,enable_thinking=False)
        ids=tok.encode(full,add_special_tokens=False)
        if not full.startswith(text) or ids[:len(base)]!=base: raise ValueError('template prefix mismatch')
        suffix=ids[len(base):]; eos=suffix.index(tok.eos_token_id);suffix=suffix[:eos+1]
        v=torch.tensor([suffix],device='cuda:0'); extended=dict(inputs)
        extended['input_ids']=torch.cat([inputs['input_ids'],v],1)
        extended['attention_mask']=torch.cat([inputs['attention_mask'],torch.ones_like(v)],1)
        if 'mm_token_type_ids' in inputs:extended['mm_token_type_ids']=torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(v)],1)
        with torch.inference_mode(): teacher=model(**extended).logits[0,n-1].float().log_softmax(dim=-1).cpu()
        np.save(OUT/f'{name}-{row}-teacher-{code}.npy',teacher.numpy())
        comparisons[code]={'continuation_ids':suffix,'first_code_logprob_prefix':float(prefix[suffix[0]]),
           'first_code_logprob_teacher':float(teacher[suffix[0]]),'max_abs_logprob_difference':float((prefix-teacher).abs().max()),
           'prefix_distribution_sha256':digest(prefix.numpy().tobytes()),'teacher_distribution_sha256':digest(teacher.numpy().tobytes())}
    append_record(name,{'row_id':row,'check':'prefix_vs_teacher','rendered_template':text,
        'input_ids':inputs['input_ids'][0].tolist(),'tensor_hashes':hashes,'comparisons':comparisons})

def main():
    OUT.mkdir(exist_ok=True);name=sys.argv[1]
    if (OUT/f'{name}.jsonl').exists(): raise ValueError('diagnostic output already exists')
    manifest=json.loads(MANIFEST.read_text()); rows={r['id']:r for r in manifest['rows']}
    expected={72:('TD5013','P1','visible'),74:('TD5013','P1','removed'),77:('TD5013','P3','removed')}
    for i,e in expected.items():
        r=rows[i]
        if (r['arrangement'],r['stop'],r['phase'])!=e or r['partition']!='calibration':raise ValueError('unexpected cohort')
    ident,revision=MODELS[name]
    processor=AutoProcessor.from_pretrained(ident,revision=revision,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(ident,revision=revision,local_files_only=True,dtype=torch.bfloat16).to('cuda:0').eval()
    append_record(name,{'check':'setup','model':ident,'revision':revision,'manifest_sha256':digest(MANIFEST.read_bytes()),
                        'script_sha256':digest(Path(__file__).read_bytes()),'dtype':'bfloat16'})
    for i in (72,74,77):
        r=rows[i]; imgs=[load_image(ROOT/'review_inputs',r[k]) for k in ('reference','query')]
        messages=make_messages(r['target_description'],PROMPT)
        scoring(model,processor,name,i,messages,imgs)
        generate(model,processor,name,i,'exact_v1_base_generation',messages,imgs)
        generate(model,processor,name,i,'literal_full_labels',message(LITERAL,2),imgs)
    r=rows[72];marked=load_image(ROOT/'review_inputs',r['reference'])
    unmarked_path=ROOT/'gate_references'/r['reference']['name']
    expected_hash=manifest['gate_input_sha256']['gate_references/'+r['reference']['name']]
    if digest(unmarked_path.read_bytes())!=expected_hash:raise ValueError('gate reference hash mismatch')
    with Image.open(unmarked_path) as source:
        if source.info or source.size!=(1200,675):raise ValueError('gate reference invalid')
        unmarked=Image.new('RGB',source.size);unmarked.paste(source.convert('RGB'))
    generate(model,processor,name,72,'same_view_base',make_messages(r['target_description'],PROMPT),[marked,unmarked])
    query=load_image(ROOT/'review_inputs',r['query'])
    generate(model,processor,name,72,'single_image_description',message(DESCRIPTION,1),[query])
if __name__=='__main__':main()
