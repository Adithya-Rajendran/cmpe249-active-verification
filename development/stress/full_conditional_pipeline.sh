#!/bin/bash
# Reviewed conditional operator pipeline. It never runs a task language model.
set -Eeuo pipefail
root=/mnt/bench-data/datasets/cmpe249-active-verification/stress-dev-v1-20261117
py=/home/ascent/Projects/SJSU/Fall26/CMPE249/cmpe249-active-verification/.venv/bin/python
scoring="$root/full-scoring-source-v1"
analysis="$root/full-analysis-source-v1"
job="$root/full-conditional-job-v1"
mkdir "$job"
trap 'code=$?; printf "%s\n" "$code" > "$job/exit-code.txt"; exit "$code"' EXIT
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$job/start-utc.txt"
check_hash() { printf '%s  %s\n' "$1" "$2" | sha256sum -c -; }
check_hash 79f308a2791220afbff2f7fd1c0b1b33a018bdfe2d439879674f871605e2e8fa "$root/run_full_evidence.py"
check_hash dfe8a35fa514aaed80efd84f0db8a51ead4b822481746c6d4294b64cae1234d9 "$scoring/development/stress/run_evidence.py"
check_hash d55e63527c675b852b2115ca7fb2b09678e5d78653857d73e72cfea3ba023e1d "$root/offline_exec.py"
check_hash a9a687bf77b56f1d527dddd88baa0c0aa5aa10b91b207694bdbcaa7427c67f55 "$root/full-main-review-v1/audit_full_prepared.py"
check_hash b16bc75e1556f900df3b41eb972a45d418a053831be9d35da5c70071eabbb6b5 "$analysis/development/stress/analysis.py"
check_hash 9cb9a220caaccea3108cf1203501ac288b09cd7359779ca29e498cb74aa84ddd "$analysis/development/stress/label_proxy.py"
check_hash 3043c608b024d98c69544126c2962730b740821e1f477a35be3d0167ee2c8ad5 "$analysis/development/stress/label_plan_ai_v1.json"
check_hash 4308617ff750bc899e42a1189db447b4cb4cf434275d7aa06730c36da090d3e2 "$analysis/development/pilot/metrics.py"
check_hash 5455c6a302387445058483046a9473db970fdd46426da927527a459668ada280 "$analysis/development/stress/protocol.json"
for path in "$root/full-evidence-v1" "$root/full-proxy-v1" "$root/full-reports-v1"; do
  if [ -e "$path" ]; then printf 'destination already exists: %s\n' "$path" >&2; exit 1; fi
done
while [ ! -f "$root/full-preparation.exit" ]; do
  if ! kill -0 1204714 2>/dev/null; then echo 'preparer stopped without exit record' >&2; exit 1; fi
  sleep 30
done
[ "$(cat "$root/full-preparation.exit")" = 0 ] || { echo 'preparation failed' >&2; exit 1; }
while [ ! -f "$root/full-main-review-v1/prepared_audit.exit" ]; do
  if ! kill -0 1210328 2>/dev/null; then echo 'prepared audit stopped without exit record' >&2; exit 1; fi
  sleep 30
done
[ "$(cat "$root/full-main-review-v1/prepared_audit.exit")" = 0 ] || { echo 'prepared audit failed' >&2; exit 1; }
"$py" - "$root" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]); p=r/'full-prepared-v1/preparation.json'
a=json.loads((r/'full-main-review-v1/prepared_audit.json').read_text())
assert a['status']=='PASS' and a['input_count']==1800
assert a['script_sha256']=='a9a687bf77b56f1d527dddd88baa0c0aa5aa10b91b207694bdbcaa7427c67f55'
assert a['render_audit_sha256']=='56e4089ed5b6f9be5e909518ca0335dadffd6afe91884d85524c9248ad54029c'
assert a['sanitized_image_checks']==7200 and a['source_pixel_comparisons']==96
assert a['blank_human_labels']==1800 and a['human_labels_created'] is False
assert a['preparation_sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
PY
while [ ! -f "$root/full-scoring-preflight-v1.exit" ]; do
  if ! kill -0 1211918 2>/dev/null; then echo 'scoring preflight stopped without exit record' >&2; exit 1; fi
  sleep 30
done
[ "$(cat "$root/full-scoring-preflight-v1.exit")" = 0 ] || { echo 'scoring preflight failed' >&2; exit 1; }
cd "$scoring"
export PYTHONPATH="$scoring"
"$py" -c 'from development.stress.alignment_diagnostics import check_frozen; assert len(check_frozen()) == 73'
"$py" "$root/run_full_evidence.py" --capsule-out "$job/capsule-recheck.json" > "$job/capsule-recheck.log" 2>&1
cmp "$root/full-scoring-capsule-v1.json" "$job/capsule-recheck.json"
sha256sum "$job/capsule-recheck.json" > "$job/capsule-recheck.sha256"
"$py" "$root/run_full_evidence.py" --execute > "$job/evidence-driver.log" 2>&1
for name in primary native native_normalized_guard relaxed_guard; do
  "$py" - "$root/full-evidence-v1/$name" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); assert len(list((p/'records').glob('*.json')))==1800
assert json.loads((p.parent/(p.name+'-execution.json')).read_text())['exit_code']==0
PY
done
cd "$analysis"
export PYTHONPATH="$analysis"
"$py" -c 'from development.stress.alignment_diagnostics import check_frozen; assert len(check_frozen()) == 73'
mkdir "$root/full-proxy-v1"
for size in primary native; do
  "$py" development/stress/label_proxy.py --prepared "$root/full-prepared-v1" --cohort "$root/geometry-v3/cohort_manifest.json" --out "$root/full-proxy-v1/$size.csv" --size "$size" > "$job/proxy-$size.log" 2>&1
  "$py" - "$root/full-proxy-v1/$size.csv" <<'PY'
import csv,sys
with open(sys.argv[1], newline='') as f: rows=list(csv.DictReader(f))
assert len(rows)==1800 and all(r['source']=='geometry_proxy' for r in rows)
PY
done
mkdir "$root/full-reports-v1"
for name in primary native native_normalized_guard relaxed_guard; do
  size=primary
  case "$name" in native|native_normalized_guard) size=native;; esac
  "$py" development/stress/analysis.py --prepared "$root/full-prepared-v1" --run "$root/full-evidence-v1/$name" --out "$root/full-reports-v1/$name-unlabeled.json" > "$job/report-$name-unlabeled.log" 2>&1
  "$py" development/stress/analysis.py --prepared "$root/full-prepared-v1" --run "$root/full-evidence-v1/$name" --out "$root/full-reports-v1/$name-proxy.json" --labels "$root/full-proxy-v1/$size.csv" --label-source geometry_proxy --label-plan development/stress/label_plan_ai_v1.json > "$job/report-$name-proxy.log" 2>&1
  "$py" - "$root/full-reports-v1/$name-unlabeled.json" "$root/full-reports-v1/$name-proxy.json" <<'PY'
import json,sys
unlabeled,proxy=(json.load(open(p)) for p in sys.argv[1:])
assert unlabeled['input_count']==proxy['input_count']==1800
assert unlabeled['support_labels'] is None
assert proxy['support_labels']['source']=='geometry_proxy'
assert proxy['development_decisions']['status']=='not eligible'
PY
done
printf '%s\n' 'all four analyses, both proxy exports and eight reports completed' > "$job/complete.txt"
