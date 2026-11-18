"""Construct model messages without source file information."""
import hashlib
from pathlib import Path
from PIL import Image

ALLOWED_TARGETS={'tuna fish can','potted meat can','mug','tomato soup can','foam brick'}

def load_image(folder,item):
    p=Path(folder)/item['name']; raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=item['sha256']: raise ValueError('image hash mismatch')
    with Image.open(p) as im:
        if im.info or im.size != (1200,675): raise ValueError('unexpected metadata or dimensions')
        clean=Image.new('RGB',im.size);clean.paste(im.convert('RGB'))
    return clean

def make_messages(description,prompt):
    if description not in ALLOWED_TARGETS: raise ValueError('unrecognized target description')
    return [{'role':'user','content':[{'type':'image'},{'type':'image'}, {'type':'text','text':prompt+'Target: '+description+'. Return exactly one answer code.'}]}]
