"""Non-task two-image forward-pass check for pinned development models."""
import json
import sys
import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODELS = {
 'qwen': ('Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
 'smol': ('HuggingFaceTB/SmolVLM2-2.2B-Instruct', 'be92141c8d0ceb98596a923c308ac2bbb4a7c466'),
}
name = sys.argv[1]
model_id, revision = MODELS[name]
start = time.time()
processor = AutoProcessor.from_pretrained(model_id, revision=revision, local_files_only=True)
model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, local_files_only=True,
    dtype=torch.bfloat16).to('cuda:0').eval()
loaded = time.time()
messages = [{'role':'user','content':[{'type':'image'}, {'type':'image'},
    {'type':'text','text':'Two synthetic color patches. Choose A, B, or C.'}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images = [Image.new('RGB',(128,128),'blue'), Image.new('RGB',(128,128),'green')]
inputs = processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
with torch.inference_mode():
    logits = model(**inputs).logits[0,-1].float()
tokenizer = processor.tokenizer
codes = {c:tokenizer.encode(c,add_special_tokens=False) for c in 'ABC'}
print(json.dumps({'model':model_id,'revision':revision,'load_s':loaded-start,
 'forward_s':time.time()-loaded,'input_ids_shape':list(inputs['input_ids'].shape),
 'input_keys':list(inputs.keys()),'codes':codes,'eos':tokenizer.eos_token_id,
 'last_logits_finite':bool(torch.isfinite(logits).all()),
 'gpu_peak_bytes':torch.cuda.max_memory_allocated()},indent=2))

from model_scoring import score_candidates
print(json.dumps({'complete_scoring':score_candidates(model, processor, messages, images)},indent=2))
