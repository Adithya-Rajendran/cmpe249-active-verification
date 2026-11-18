"""Exercise mismatch refusal with an isolated copy of smoke inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from development.stress.render_smoke import IDS

p = argparse.ArgumentParser()
p.add_argument('--run-root', required=True)
p.add_argument('--assets', required=True)
p.add_argument('--tags', required=True)
p.add_argument('--textures', required=True)
a = p.parse_args()
root = Path(a.run_root).resolve()
probe = root / 'resume-mismatch-probe'
probe.mkdir(exist_ok=False)
geometry, renders = probe / 'geometry', probe / 'renders'
geometry.mkdir()
renders.mkdir()
for name in ['summary.json', 'provenance.json', 'cohort_manifest.json', *[i + '.json' for i in IDS]]:
    shutil.copyfile(root / 'geometry-v3' / name, geometry / name)
capsule = json.loads((root / 'renders/smoke_inputs.json').read_text())
capsule['output_dir'] = str(renders)
capsule['source']['sim/isaac_batch.py'] = 'deliberately-changed-for-test'
(renders / 'smoke_inputs.json').write_text(json.dumps(capsule, indent=2) + '\n')
cmd = [sys.executable, '-m', 'development.stress.render_smoke', '--geometry', str(geometry), '--output', str(renders),
       '--assets', a.assets, '--tags', a.tags, '--textures', a.textures, '--execute']
r = subprocess.run(cmd, text=True, capture_output=True)
result = {'command': cmd, 'exit_code': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr,
          'no_launch_log': not list(renders.glob('launch-*'))}
(probe / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
assert r.returncode != 0 and 'resume input provenance mismatch' in r.stderr and result['no_launch_log']
print('PASS changed source provenance refused without simulator startup')
