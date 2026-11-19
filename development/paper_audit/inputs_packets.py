"""Create opaque score-free packets and a private mapping from a declared inventory.

Inventory rows require private_id, reference, query and box_xyxy. The packet
contains no private identifiers, engine, state, masks or pair information.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import secrets
from pathlib import Path
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(inventory, instructions, output):
    out = Path(output)
    if not inventory:
        raise ValueError('empty inventory')
    out.mkdir(parents=True, exist_ok=False)
    private = []
    public = []
    seen = set()
    for row in inventory:
        if row['private_id'] in seen:
            raise ValueError('duplicate private ID')
        seen.add(row['private_id'])
        uid = secrets.token_hex(12)
        paths = {}
        for role in ('reference','query'):
            dst = out / f'{uid}-{role}.png'
            with Image.open(row[role]) as image:
                image.convert('RGB').save(dst)
            paths[role] = dst
        box = row['box_xyxy']
        with Image.open(paths['reference']) as ref:
            w,h = ref.size
        if len(box) != 4 or any(type(v) is not int for v in box) or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError('invalid reference box')
        public.append({'id':uid,'reference':paths['reference'].name,'query':paths['query'].name,
                       'box_xyxy':box,'reference_sha256':sha(paths['reference']), 'query_sha256':sha(paths['query'])})
        private.append({'id':uid,'private_id':row['private_id']})
    secrets.SystemRandom().shuffle(public)
    (out/'instructions.txt').write_bytes(Path(instructions).read_bytes())
    (out/'cases.json').write_text(json.dumps(public,indent=2)+'\n')
    return private


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inventory'); p.add_argument('instructions'); p.add_argument('packet'); p.add_argument('private_mapping')
    args=p.parse_args()
    with open(args.inventory) as f: rows=json.load(f)
    if Path(args.packet).resolve() in Path(args.private_mapping).resolve().parents:
        raise ValueError('private mapping must be outside packet')
    mapping=prepare(rows,args.instructions,args.packet)
    with open(args.private_mapping,'x') as f: json.dump(mapping,f,indent=2); f.write('\n')
