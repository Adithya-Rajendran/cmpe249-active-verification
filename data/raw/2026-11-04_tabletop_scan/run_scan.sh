#!/bin/bash
# The board-height scan of scan_plan.json, then the decision. Run from the repo root (background; ~35 min expected).
set -euo pipefail
D=data/raw/2026-11-04_tabletop_scan
UV="$HOME/.local/bin/uv run --no-project --python 3.13.5 --with numpy==2.5.3 python"
date -u +%Y-%m-%dT%H:%M:%SZ > $D/scan_started_utc.txt
git rev-parse HEAD >> $D/scan_started_utc.txt
$UV -m sim.feasibility --workers 5 heightscan --plan $D/scan_plan.json --log $D/cells.jsonl --out $D/results.json
$UV -m sim.feasibility decide --plan $D/scan_plan.json --results $D/results.json --out $D/decision.json
date -u +%Y-%m-%dT%H:%M:%SZ > $D/scan_finished_utc.txt
