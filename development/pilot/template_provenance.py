"""Capture synthetic template boundaries, candidate tokens and processor settings."""
import json
import platform
import torch
import transformers
from transformers import AutoProcessor

MODELS=[('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),('HuggingFaceTB/SmolVLM2-2.2B-Instruct','be92141c8d0ceb98596a923c308ac2bbb4a7c466')]
result={'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'dtype':'bfloat16','models':[]}
for name,revision in MODELS:
    p=AutoProcessor.from_pretrained(name,revision=revision,local_files_only=True)
    m=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':'Synthetic prompt.'}]}]
    pre=p.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    before=p.tokenizer.encode(pre,add_special_tokens=False); candidates={}
    for c in 'ABC':
        full=p.apply_chat_template(m+[{'role':'assistant','content':[{'type':'text','text':c}]}],tokenize=False,enable_thinking=False)
        ids=p.tokenizer.encode(full,add_special_tokens=False)
        assert full.startswith(pre) and ids[:len(before)]==before
        suffix=ids[len(before):]; assert suffix.count(p.tokenizer.eos_token_id)==1
        end=suffix.index(p.tokenizer.eos_token_id)
        assert not p.tokenizer.decode(suffix[end+1:]).strip()
        candidates[c]={'template_suffix':full[len(pre):],'scored_token_ids':suffix[:end+1]}
    result['models'].append({'id':name,'revision':revision,'prefix':pre,'eos_token':p.tokenizer.eos_token,
      'eos_token_id':p.tokenizer.eos_token_id,'candidates':candidates,'image_processor':p.image_processor.to_dict()})
print(json.dumps(result,indent=2,default=str))
