"""Check saved smoke artifacts and create a review contact sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim import arrange as A
from sim import isaac_batch as B

IDS = ('SXS01D0', 'SXS01D6', 'SXT01D0', 'SXT01D6')
PHASES = ('reference', 'visible', 'occluded', 'removed')


def check(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    records, arrs, thumbs = {}, {}, []
    for ident in IDS:
        folder = root / ident
        arr = json.loads((folder / 'arrangement.json').read_text())
        labels = json.loads((folder / 'labels.json').read_text())
        render = json.loads((folder / 'render.json').read_text())
        assert arr['arrangement_id'] == ident and arr['arrangement_sha256'] == A.arrangement_hash(arr)
        assert labels['hashes']['arrangement_file_sha256'] == A.sha256_file(folder / 'arrangement.json')
        assert labels['lightings'] == ['L1'] and labels['stops'] == arr['stops']
        assert render['settings']['resolution'] == [[2304, 1296]] * 5
        assert {x['phase']: (x['target_present'], x['occluder_pose'], x['camera']) for x in arr['phases']} == {
            'reference': (True, 'parked', 'nominal'), 'visible': (True, 'parked', 'revisit'),
            'occluded': (True, 'placed', 'revisit'), 'removed': (False, 'placed', 'revisit')}
        summary = {'timings': render['timings'], 'views': {}, 'mask_counts': {}}
        for stop in arr['stops']:
            for phase in PHASES:
                rgb = folder / 'rgb' / f'{phase}_{stop}_L1.png'
                with Image.open(rgb) as im:
                    assert im.size == (2304, 1296)
                    metadata = json.loads(im.info['cmpe249'])
                    camera = 'nominal' if phase == 'reference' else 'revisit'
                    assert metadata == B.png_meta(arr, phase, stop, 'L1', camera)
                    if stop == 'P3':
                        img = im.convert('RGB')
                        img.thumbnail((480, 270), Image.Resampling.LANCZOS)
                        thumbs.append((ident, phase, img.copy()))
                with Image.open(folder / 'masks' / f'classes_{phase}_{stop}.png') as im:
                    classes = np.array(im)
                assert classes.shape == (1296, 2304)
                counts = {str(v): int(n) for v, n in zip(*np.unique(classes, return_counts=True))}
                summary['mask_counts'][phase + '_' + stop] = counts
                if phase == 'removed':
                    assert counts.get('1', 0) == 0
                else:
                    with Image.open(folder / 'masks' / f'target_{phase}_{stop}.png') as im:
                        target = np.array(im)
                    assert np.array_equal(target > 0, classes == 1)
                summary['views'][phase + '_' + stop] = {'camera': camera, 'image_sha256': A.sha256_file(rgb)}
            for name in ('target_alone', 'target_alone_revisit'):
                with Image.open(folder / 'masks' / f'{name}_{stop}.png') as im:
                    mask = np.array(im)
                assert mask.shape == (1296, 2304)
                summary['mask_counts'][name + '_' + stop] = {'target': int(np.count_nonzero(mask))}
        records[ident] = summary
        arrs[ident] = arr
    for prefix in ('SXS', 'SXT'):
        x, y = arrs[prefix + '01D0'], arrs[prefix + '01D6']
        assert x['objects'] == y['objects'] and x['scene']['cameras'] == y['scene']['cameras']
    canvas = Image.new('RGB', (480 * 4, 295 * 4), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (ident, phase, img) in enumerate(thumbs):
        x, y = (i % 4) * 480, (i // 4) * 295
        canvas.paste(img, (x, y + 25))
        draw.text((x + 5, y + 5), f'{ident}  P3  {phase}', fill='black')
    canvas.save(out / 'contact-sheet.jpg', quality=90)
    (out / 'artifact-check.json').write_text(json.dumps({'status': 'passed', 'arrangements': records}, indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--renders', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    check(args.renders, args.out)
