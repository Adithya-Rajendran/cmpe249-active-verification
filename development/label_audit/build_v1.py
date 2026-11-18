"""Build a development-only cue audit. Source paths never enter model inputs."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/mnt/bench-data/datasets/cmpe249-active-verification/sim')
OUT = Path('runs/2026-11-16-support-audit-v1')
# Lexical first four complete random arrangements, independent of model results.
GROUPS = {'shelf_dev_v3': ['SD1000', 'SD1001', 'SD1002', 'SD1003', 'SD1004', 'SD1005'],
          'tabletop_dev49': ['TD5011', 'TD5012', 'TD5013', 'TD5014', 'TD5015', 'TD5016']}
STOPS = ['P1', 'P3', 'P5']
PHASES = ['reference', 'visible', 'occluded', 'removed']

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for group, ids in GROUPS.items():
      for index, name in enumerate(ids):
        folder = ROOT / group / name
        a = json.loads((folder / 'arrangement.json').read_text())
        labels = json.loads((folder / 'labels.json').read_text())
        for stop in STOPS:
            paths = {phase: folder / 'rgb' / f'{phase}_{stop}_L1.png' for phase in PHASES}
            box = labels['reference_box']['L1'][stop]
            sheet = Image.new('RGB', (1200, 940), 'white')
            draw = ImageDraw.Draw(sheet)
            for col, (phase, path) in enumerate(paths.items()):
                with Image.open(path) as src:
                    image = src.convert('RGB')
                    image.thumbnail((600, 400))
                x, y = (col % 2) * 600, (col // 2) * 470
                sheet.paste(image, (x, y + 25))
                draw.text((x + 5, y + 5), f'{name} {stop} {phase}', fill='black')
            sheet.save(OUT / f'{name}_{stop}.jpg', quality=92)
            for phase in PHASES[1:]:
                records.append({'arrangement':name, 'partition':'calibration' if index % 2 == 0 else 'pilot_evaluation', 'environment':group,
                    'stop':stop, 'lighting':'L1', 'phase':phase,
                    'physical_state':labels['truth'][phase], 'support':'PENDING_REVIEW',
                    'reference_box':box, 'target':a['target_id'], 'occluder_family':a['occluder_family'],
                    'visible_fraction':labels['visible_fraction'].get(phase, {}).get(stop),
                    'region_obstruction':labels['region_obstruction'].get(phase, {}).get(stop),
                    'source_paths':{k:str(v) for k,v in paths.items()},
                    'source_sha256':{k:digest(v) for k,v in paths.items()},
                    'arrangement_sha256':digest(folder/'arrangement.json'), 'labels_sha256':digest(folder/'labels.json'),
                    'audit':{'matching':'PENDING','target_fragments':'PENDING','shadows':'PENDING',
                             'reflections':'PENDING','alignment':'PENDING','blur':'PENDING','context':'PENDING',
                             'reviewer':'','reviewer_type':'PENDING','reason':''}})
    (OUT/'ledger.json').write_text(json.dumps(records, indent=2)+'\n')
    print(f'{len(records)} records, {sum(map(len, GROUPS.values()))*len(STOPS)} sheets')

if __name__ == '__main__': main()
