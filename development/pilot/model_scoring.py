"""Complete candidate continuation scores for a two-image model prompt."""
import math
import torch

CODES = {'A': 'PRESENT', 'B': 'REMOVED', 'C': 'UNKNOWN'}

def score_candidates(model, processor, messages, images):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer = processor.tokenizer
    base_tokens = tokenizer.encode(text, add_special_tokens=False)
    continuations = {}
    for code in CODES:
        full = processor.apply_chat_template(messages + [{'role':'assistant','content':[{'type':'text','text':code}]}],
                    tokenize=False, enable_thinking=False)
        if not full.startswith(text): raise ValueError('assistant template prefix mismatch')
        combined = tokenizer.encode(full, add_special_tokens=False)
        if combined[:len(base_tokens)] != base_tokens: raise ValueError('candidate tokenization changes prefix')
        suffix = combined[len(base_tokens):]
        if suffix.count(tokenizer.eos_token_id) != 1: raise ValueError('ambiguous assistant terminator')
        end = suffix.index(tokenizer.eos_token_id)
        if tokenizer.decode(suffix[end+1:]).strip(): raise ValueError('unexpected text after terminator')
        continuations[code] = suffix[:end+1]
    inputs = processor(text=[text], images=images, return_tensors='pt').to('cuda:0')
    prefix_tail=base_tokens[-8:]
    if inputs['input_ids'][0,-len(prefix_tail):].tolist()!=prefix_tail:
        raise ValueError('processed input does not end at assistant prefix')
    n = inputs['input_ids'].shape[1]
    records = {}
    for code in CODES:
        tokens = continuations[code]
        if len(tokens) < 2: raise ValueError('empty code or missing terminator')
        extended = dict(inputs)
        suffix = torch.tensor([tokens], device='cuda:0')
        extended['input_ids'] = torch.cat([inputs['input_ids'],suffix],dim=1)
        extended['attention_mask'] = torch.cat([inputs['attention_mask'],torch.ones_like(suffix)],dim=1)
        if 'mm_token_type_ids' in inputs:
            extended['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'],torch.zeros_like(suffix)],dim=1)
        with torch.inference_mode():
            logits = model(**extended).logits[0,n-1:n-1+len(tokens)].float()
            logprobs = torch.log_softmax(logits,dim=-1)
            token_scores = [float(logprobs[i,t]) for i,t in enumerate(tokens)]
        if not all(math.isfinite(x) for x in token_scores):
            raise ValueError('nonfinite candidate score')
        records[code] = {'token_ids':tokens,'token_logprobs':token_scores,'score':sum(token_scores)}
    scores = torch.tensor([records[c]['score'] for c in CODES],dtype=torch.float64)
    probabilities = torch.softmax(scores,dim=0).tolist()
    for c,p in zip(CODES,probabilities): records[c]['normalized_score'] = p
    selected = max(CODES, key=lambda c: records[c]['score'])
    transforms={k:(v.tolist() if k == 'image_grid_thw' else list(v.shape)) for k,v in inputs.items() if hasattr(v,'shape')}
    return {'answer':CODES[selected], 'code':selected, 'candidates':records, 'processor_outputs':transforms}
