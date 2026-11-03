#!/bin/bash
# Render one arrangement folder with Isaac Sim on the GB10, in the background-friendly form tools/pipeline.py expects:
# the log gets stdout and stderr, and the rc file is written when isaac_batch exits.
# Copy it to the GB10 run dir next to isaac_batch.py and isaac_rig.py, then:
#   bash run_render.sh ARR_DIR OUT_SET LOG RC_FILE [extra isaac_batch args, e.g. --skip-existing]
set -u
ARR_DIR=$1; OUT=$2; LOG=$3; RC=$4; shift 4
cd "$(dirname "$0")"
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 OMNI_KIT_ACCEPT_EULA=YES
ISAAC_PY=${ISAAC_PY:-$HOME/cmpe249-sim/isaacsim-6.2/.venv/bin/python}
rm -f "$RC"
date +%s > "$RC.start"
~/.local/bin/uv run --no-project --python "$ISAAC_PY" isaac_batch.py \
  --arrangements "$ARR_DIR"/*.json --assets-dir ../assets/ycbv/models --tags-dir tags --out "$OUT" "$@" > "$LOG" 2>&1
echo $? > "$RC"
date +%s > "$RC.end"
