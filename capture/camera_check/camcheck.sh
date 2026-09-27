#!/bin/bash
# Camera characterization helper, run ON the Pi (bash + rpicam-still + stdlib python3 only).
# Usage: camcheck.sh af | afwin X,Y,W,H | sweep [DIR] | fine OUTDIR LO HI STEP | rep N | drift N INTERVAL_S
# Every rpicam-still invocation is appended verbatim to commands.log with its wall time.
set -euo pipefail
OUT=${OUT:-$HOME/cmpe249/camera-check/2026-09-26}
mkdir -p "$OUT"; cd "$OUT"
LOG=$OUT/commands.log
W=2304; H=1296; MODE=2304:1296:10:P

shot() {  # shot <name-without-ext> [extra rpicam-still args...]
  local name=$1; shift
  mkdir -p "$(dirname "$name")"
  if [[ -e "$name.jpg" ]]; then echo "refusing to overwrite $name.jpg" >&2; exit 3; fi
  local cmd=(rpicam-still -n --width $W --height $H --mode $MODE -q 93
             --metadata "$name.json" --metadata-format json -o "$name.jpg" "$@")
  local s e
  s=$(date +%s.%N)
  "${cmd[@]}" 2> "$name.log"
  e=$(date +%s.%N)
  echo "$(date -Is) wall_s=$(python3 -c "print(round($e-$s,3))") ${cmd[*]}" >> "$LOG"
}

# Print "shutter_us total_gain red_gain blue_gain lens_pos" from a metadata json.
lockvals() {
  python3 - "$1" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
g = m["AnalogueGain"] * m.get("DigitalGain", 1.0)
r, b = m["ColourGains"]
print(m["ExposureTime"], f"{g:.4f}", f"{r:.4f}", f"{b:.4f}", f"{m.get('LensPosition', 1.0):.4f}")
EOF
}

stage=${1:?stage}
case "$stage" in
  af)
    for i in 1 2 3; do shot af/cont_normal_$i -t 3000 --autofocus-mode continuous --autofocus-range normal; done
    for i in 1 2 3; do shot af/cont_macro_$i  -t 3000 --autofocus-mode continuous --autofocus-range macro; done
    for i in 1 2 3; do shot af/auto_normal_$i -t 2000 --autofocus-mode auto --autofocus-range normal --autofocus-on-capture; done
    for i in 1 2 3; do shot af/auto_macro_$i  -t 2000 --autofocus-mode auto --autofocus-range macro  --autofocus-on-capture; done
    for i in 1 2 3; do shot af/auto_full_$i   -t 2000 --autofocus-mode auto --autofocus-range full   --autofocus-on-capture; done
    ;;
  afwin)  # afwin x,y,w,h (normalised) : AF restricted to a metering window
    for i in 1 2 3; do shot af/auto_window_$i -t 2000 --autofocus-mode auto --autofocus-range full --autofocus-on-capture --autofocus-window "$2"; done
    ;;
  sweep)  # sweep [DIR] : auto calibration still, then 0..15 dioptres in 0.5 steps (ascending), exposure/AWB locked
    D=${2:-sweep}
    shot "$D/calib" -t 3000
    read -r S G R B L < <(lockvals "$D/calib.json")
    echo "$D lock: shutter=$S gain=$G awb=$R,$B (calib AF lens=$L)" | tee -a "$LOG"
    for lp in $(python3 -c "print(' '.join(f'{i*0.5:.1f}' for i in range(31)))"); do
      shot "$D/lp_$(printf '%04.1f' "$lp")" -t 1200 --shutter "$S" --gain "$G" --awbgains "$R,$B" --lens-position "$lp"
    done
    ;;
  fine)  # fine OUTDIR LO HI STEP : fresh auto calibration still, then a DESCENDING fine sweep with exposure/AWB locked
    D=$2
    shot "$D/calib" -t 3000
    read -r S G R B L < <(lockvals "$D/calib.json")
    echo "$D lock: shutter=$S gain=$G awb=$R,$B (calib AF lens=$L)" | tee -a "$LOG"
    for lp in $(python3 -c "
lo,hi,st=float('$3'),float('$4'),float('$5'); n=round((hi-lo)/st)
print(' '.join(f'{hi-i*st:.2f}' for i in range(n+1)))"); do
      shot "$D/lp_$(printf '%05.2f' "$lp")" -t 1200 --shutter "$S" --gain "$G" --awbgains "$R,$B" --lens-position "$lp"
    done
    ;;
  rep)  # rep N : N full-auto stills, then a calibration still, then N fully locked stills
    N=${2:-8}
    for i in $(seq -w 1 "$N"); do shot rep/auto_$i -t 3000; done
    shot rep/calib -t 3000
    read -r S G R B L < <(lockvals rep/calib.json)
    echo "rep lock: shutter=$S gain=$G awb=$R,$B lens=$L" | tee -a "$LOG"
    for i in $(seq -w 1 "$N"); do
      shot rep/locked_$i -t 1500 --shutter "$S" --gain "$G" --awbgains "$R,$B" --lens-position "$L"
    done
    ;;
  drift)  # drift N INTERVAL_S : locked stills (rep/calib lock) spaced in time
    N=${2:-12}; IV=${3:-60}
    read -r S G R B L < <(lockvals rep/calib.json)
    for i in $(seq -w 1 "$N"); do
      shot drift/locked_$i -t 1500 --shutter "$S" --gain "$G" --awbgains "$R,$B" --lens-position "$L"
      if (( 10#$i < N )); then sleep "$IV"; fi
    done
    ;;
  *) echo "unknown stage $stage" >&2; exit 2;;
esac
echo "stage $stage done"
