#!/bin/bash
# Render one folder of frozen TEST arrangements with Isaac Sim on the GB10, through render_test.py (the freeze gate
# around isaac_batch.py, which stays byte-identical), in the form tools/pipeline.py expects (spec render.script
# "render_test.py"): the log gets stdout and stderr, and the rc file is written when the render exits. The same as
# run_render.sh apart from the script and the copy of the committed FREEZE.md it passes. Only after the freeze: copy it
# to the GB10 run dir next to render_test.py, isaac_batch.py, isaac_rig.py and a copy of docs/FREEZE.md, then:
#   bash run_render_test.sh ARR_DIR OUT_SET LOG RC_FILE FREEZE_COPY [extra isaac_batch args, e.g. --skip-existing]
set -u
ARR_DIR=$1; OUT=$2; LOG=$3; RC=$4; FREEZE=$5; shift 5
cd "$(dirname "$0")"
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES
ISAAC_PY=${ISAAC_PY:-$HOME/cmpe249-sim/isaacsim-6.2/.venv/bin/python}
rm -f "$RC"
date +%s > "$RC.start"
~/.local/bin/uv run --no-project --python "$ISAAC_PY" render_test.py \
  --arrangements "$ARR_DIR"/*.json --assets-dir ../assets/ycbv/models --tags-dir tags --out "$OUT" \
  --freeze "$FREEZE" "$@" > "$LOG" 2>&1
echo $? > "$RC"
date +%s > "$RC.end"
