"""Compare all scorer-visible fields of two cases, excluding opaque labels.

Callers must include every visible prompt field in `fields` and provide the
actual instruction bytes and rendered RGB images. Equality is meaningful only
for a fully enumerated interface; packet IDs must never encode private state.
"""
import hashlib
from pathlib import Path
from PIL import Image


def pixels(path):
    with Image.open(path) as im:
        im=im.convert('RGB')
        return {'size':list(im.size),'sha256':hashlib.sha256(im.tobytes()).hexdigest()}


def certificate(a,b):
    def components(c):
        return {'reference':pixels(c['reference']), 'query':pixels(c['query']),
                'instructions_sha256':hashlib.sha256(Path(c['instructions']).read_bytes()).hexdigest(),
                'fields':c['fields']}
    left,right=components(a),components(b)
    checks={k:left[k]==right[k] for k in left}
    return {'equal_complete_input':all(checks.values()), 'component_equal':checks,
            'left':left,'right':right}
