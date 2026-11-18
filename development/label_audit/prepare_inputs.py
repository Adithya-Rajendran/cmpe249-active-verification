"""Prepare neutral, metadata-free, reviewable development image inputs."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path('runs/2026-11-16-support-audit-v1')

def save_image(src, dst, box=None):
    with Image.open(src) as image:
        original_size = image.size
        clean = Image.new('RGB', image.size)
        clean.paste(image.convert('RGB'))
    clean = clean.resize((1200, 675), Image.LANCZOS)
    if box:
        x0, y0, x1, y1 = box
        scaled = (round(x0 * 1200 / original_size[0]), round(y0 * 675 / original_size[1]),
                  round(x1 * 1200 / original_size[0]), round(y1 * 675 / original_size[1]))
        ImageDraw.Draw(clean).rectangle(scaled, outline=(255, 0, 0), width=4)
    clean.save(dst, format='PNG')
    with Image.open(dst) as check:
        assert not check.info and check.size == (1200, 675)
    return hashlib.sha256(dst.read_bytes()).hexdigest()

def main():
    ledger = json.loads((OUT / 'ledger.json').read_text())
    folder = OUT / 'review_inputs'
    folder.mkdir(exist_ok=True)
    prepared = []
    refs = {}
    for i, row in enumerate(ledger):
        key = (row['arrangement'], row['stop'])
        if key not in refs:
            name = f'reference_{len(refs):03d}.png'
            h = save_image(row['source_paths']['reference'], folder / name, row['reference_box'])
            refs[key] = {'name': name, 'sha256': h}
        name = f'query_{i:03d}.png'
        h = save_image(row['source_paths'][row['phase']], folder / name)
        prepared.append({'record_index':i, 'reference':refs[key], 'query':{'name':name, 'sha256':h}})
    (OUT/'prepared_inputs.json').write_text(json.dumps(prepared, indent=2)+'\n')
    print(f'{len(refs)} references, {len(prepared)} queries')

if __name__ == '__main__': main()
