#!/bin/bash
# The class matrix at the amended configuration (amendment_2026-11-04.json; scan_plan.json after_the_scan.class_matrix):
# base_config.json with every board at H_T = 0.18 m, IND28 0.36 m wide, the four kept ycb_box choices; 100 real generator
# tries per (slot class, target) on the pre-planned probe seeds 7200-7299. Run from the repo root (background).
set -euo pipefail
D=data/raw/2026-11-04_tabletop_scan
UV="$HOME/.local/bin/uv run --no-project --python 3.13.5 --with numpy==2.5.3 python"
date -u +%Y-%m-%dT%H:%M:%SZ > $D/matrix_started_utc.txt
git rev-parse HEAD >> $D/matrix_started_utc.txt
$UV -m sim.feasibility --config $D/base_config.json --workers 5 matrix --tries 100 --seed0 7200 \
  --set occluder.families.S8.height_m=0.18 --set occluder.families.M16.height_m=0.18 \
  --set occluder.families.IND28.height_m=0.18 --set occluder.families.IND28.width_m=0.36 \
  --set 'occluder.families.ycb_box.choices=[[2,"upright"],[3,"upright"],[16,"upright"],[7,"edge"]]' \
  --out $D/matrix.json
date -u +%Y-%m-%dT%H:%M:%SZ > $D/matrix_finished_utc.txt
