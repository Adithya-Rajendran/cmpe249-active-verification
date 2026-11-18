"""Run on Ascent with both pinned processors and synthetic images."""
import json
from PIL import Image
from transformers import AutoProcessor
models=[('Qwen/Qwen3.5-4B','851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),('HuggingFaceTB/SmolVLM2-2.2B-Instruct','be92141c8d0ceb98596a923c308ac2bbb4a7c466')]
for name,rev in models:
 p=AutoProcessor.from_pretrained(name,revision=rev,local_files_only=True)
 m=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':'Synthetic prompt.'}]}]
 text=p.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 expected=p.tokenizer.encode(text,add_special_tokens=False)[-8:]
 inputs=p(text=[text],images=[Image.new('RGB',(128,128),'blue'),Image.new('RGB',(128,128),'green')],return_tensors='pt')
 actual=inputs['input_ids'][0,-len(expected):].tolist()
 assert actual==expected,(name,expected,actual)
 print(json.dumps({'model':name,'expected_tail':expected,'processed_tail':actual,'pass':True}))
