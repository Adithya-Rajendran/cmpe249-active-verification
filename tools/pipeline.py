"""Run driver for a simulated dev set: render on the GB10, then build episodes and check the set on the Jetson.

Runs unattended from a run spec (configs/pipeline/*.json) and writes everything a person or a watcher needs to a run
directory, runs/<run_id>/:

    events.jsonl   one JSON object per line: ts, run_id, stage, level (info | alert | done), msg, data
    status.json    the latest state (stage, progress, ETA, last event), rewritten on every event
    <stage>.out    stdout and stderr of the local stages

Levels: "info" is progress and heartbeats; "alert" means a person must look (failure, stall, flagged arrangement);
"done" is the terminal success line. A terminal failure is an alert with data.terminal = true, and the driver
exits non-zero. tools/watch.py prints only alerts and the terminal line, so it can wake an operator.

Stages:
    render    attach to the render already running on the GB10 for this spec, or launch it with --launch; follow
              its log until its rc file appears. Alerts: a Traceback or Python error line, an arrangement whose
              label sanity check failed, an arrangement slower than slow_factor x the running median, no new
              arrangement for stall_minutes, the render process gone without an rc file, a non-zero rc, or fewer
              arrangements than expected.
    episodes  python -m sim.episodes --root SET --contact-sheets --generation-log GENLOG
    check     python -m sim.check_batch --root SET

Stdlib only. Usage:
    python tools/pipeline.py configs/pipeline/shelf_dev_v3.json [--launch] [--from-stage episodes]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGES = ("render", "episodes", "check")
ERROR_LINE = re.compile(r"^(Traceback \(most recent call last\)|\w*(Error|Exception): )")
# Known harmless line from Isaac's Vulkan probe on the GB10.
IGNORED = ("TU: error:",)


class Run:
    def __init__(self, spec: dict, run_dir: Path):
        self.spec = spec
        self.run_id = spec["run_id"]
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.events = self.dir / "events.jsonl"
        self.status: dict = {"run_id": self.run_id, "pid": os.getpid(), "started": now()}

    def emit(self, stage: str, level: str, msg: str, **data) -> None:
        event = {"ts": now(), "run_id": self.run_id, "stage": stage, "level": level, "msg": msg, "data": data}
        with self.events.open("a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
        print(json.dumps(event, sort_keys=True), flush=True)
        self.status.update(stage=stage, last_event=event)
        tmp = self.dir / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=1, sort_keys=True))
        tmp.replace(self.dir / "status.json")

    def fail(self, stage: str, msg: str, **data) -> "NoReturn":
        self.emit(stage, "alert", msg, terminal=True, **data)
        sys.exit(1)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Remote:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def run(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        c = self.cfg
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"HostKeyAlias={c['host_key_alias']}",
                "-i", os.path.expanduser(c["key"]), f"{c['user']}@{c['host']}", command]
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def stage_render(run: Run, launch: bool) -> None:
    r = run.spec["render"]
    remote = Remote(run.spec["remote"])
    rd = run.spec["remote"]["run_dir"]
    log, rc_file = f"{rd}/{r['log']}", f"{rd}/{r['rc_file']}"
    pattern = f"isaac_batch.py --arrangements {r['arrangements_dir']}/"

    def probe() -> dict:
        cmd = (f"pgrep -f {shlex.quote(pattern)} >/dev/null && echo RUNNING=1 || echo RUNNING=0; "
               f"test -f {rc_file} && echo RC=$(cat {rc_file}) || echo RC=; "
               f"wc -l < {log} 2>/dev/null || echo 0")
        out = remote.run(cmd).stdout.split()
        return {"running": "RUNNING=1" in out, "rc": next((o[3:] for o in out if o.startswith("RC=")), ""),
                "lines": int(out[-1]) if out and out[-1].isdigit() else 0}

    state = probe()
    if not state["running"] and state["rc"] == "":
        if not launch:
            run.fail("render", "no render is running for this spec and none has finished; rerun with --launch")
        remote.run(f"cd {rd} && setsid nohup bash {r['launch_script']} >/dev/null 2>&1 < /dev/null &")
        run.emit("render", "info", "launched", script=r["launch_script"])
    else:
        run.emit("render", "info", "attached", running=state["running"], rc=state["rc"])

    seen, durations, offset = set(), [], 0
    last_progress, last_heartbeat, ssh_failures = time.time(), 0.0, 0
    expected = r["expected"]
    while True:
        try:
            # Read only the new part of the log.
            res = remote.run(f"tail -n +{offset + 1} {log} 2>/dev/null", timeout=120)
            state = probe()
            ssh_failures = 0
        except (subprocess.TimeoutExpired, OSError) as e:
            ssh_failures += 1
            if ssh_failures >= 5:
                run.fail("render", "GB10 unreachable over ssh 5 times in a row", error=str(e))
            time.sleep(run.spec["poll_seconds"])
            continue
        new_lines = res.stdout.splitlines()
        offset += len(new_lines)
        for line in new_lines:
            if line.startswith("ISAAC_BATCH "):
                rec = json.loads(line[len("ISAAC_BATCH "):])
                arr, secs = rec["arrangement_id"], rec.get("total_s")
                if arr in seen:
                    continue
                seen.add(arr)
                last_progress = time.time()
                if secs is not None:
                    if len(durations) >= 3 and secs > run.spec["slow_factor"] * statistics.median(durations):
                        run.emit("render", "alert", f"{arr} slow: {secs:.0f} s vs median "
                                 f"{statistics.median(durations):.0f} s", arrangement=arr, seconds=secs)
                    durations.append(secs)
                if rec.get("label_sanity_ok") is False:
                    run.emit("render", "alert", f"{arr} failed the label sanity check", arrangement=arr)
                eta = statistics.median(durations) * (expected - len(seen)) if durations else None
                run.status.update(done=len(seen), expected=expected, eta_s=eta)
                run.emit("render", "info", f"{arr} rendered ({len(seen)}/{expected})", arrangement=arr,
                         seconds=secs, patterns=rec.get("patterns"), eta_s=eta)
            elif ERROR_LINE.match(line.strip()) and not line.startswith(IGNORED):
                run.emit("render", "alert", "error line in the render log", line=line.strip()[:500])
        if state["rc"] != "":
            if state["rc"] != "0":
                run.fail("render", f"render exited with rc {state['rc']}", rendered=len(seen))
            if len(seen) < expected:
                run.fail("render", f"render exited 0 but only {len(seen)}/{expected} arrangements reported")
            run.emit("render", "info", "render finished", rendered=len(seen),
                     median_s=statistics.median(durations) if durations else None,
                     max_s=max(durations) if durations else None)
            return
        if not state["running"]:
            # The launcher writes rc right after the process exits; give it a moment before calling it lost.
            time.sleep(10)
            if probe()["rc"] == "":
                run.fail("render", "render process is gone and wrote no rc file", rendered=len(seen))
            continue
        stall = run.spec["stall_minutes"] * 60
        if time.time() - last_progress > stall:
            run.emit("render", "alert", f"no arrangement finished in {run.spec['stall_minutes']} min",
                     rendered=len(seen))
            last_progress = time.time()  # alert again after another full stall window, not every poll
        if time.time() - last_heartbeat > run.spec["heartbeat_seconds"]:
            run.emit("render", "info", "heartbeat", rendered=len(seen), expected=expected)
            last_heartbeat = time.time()
        time.sleep(run.spec["poll_seconds"])


def stage_local(run: Run, stage: str, module_args: list[str]) -> None:
    argv = run.spec["local_python"] + module_args
    out = run.dir / f"{stage}.out"
    run.emit(stage, "info", "started", argv=argv)
    t0 = time.time()
    with out.open("w") as f:
        rc = subprocess.run(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    tail = out.read_text().splitlines()[-15:]
    if rc != 0:
        run.fail(stage, f"{stage} exited with rc {rc}", rc=rc, tail=tail, log=str(out))
    run.emit(stage, "info", "finished", seconds=round(time.time() - t0, 1), tail=tail[-5:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--launch", action="store_true", help="start the render if it is not already running")
    ap.add_argument("--from-stage", choices=STAGES, default="render")
    ap.add_argument("--run-dir", type=Path, help="default: runs/<run_id>")
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text())
    if isinstance(spec["remote"], str):  # host details live in an untracked file (configs/pipeline/remote.example.json)
        spec["remote"] = json.loads((REPO / spec["remote"]).read_text())
    run = Run(spec, args.run_dir or REPO / "runs" / spec["run_id"])
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    run.status.update(git=git, dirty=dirty, spec=str(args.spec))
    run.emit("driver", "info", "run started", git=git, dirty=dirty, spec=spec, from_stage=args.from_stage)

    root = spec["set_root"]
    todo = STAGES[STAGES.index(args.from_stage):]
    if "render" in todo:
        stage_render(run, args.launch)
    if "episodes" in todo:
        stage_local(run, "episodes", ["-m", "sim.episodes", "--root", root, "--contact-sheets",
                                      "--generation-log", spec["generation_log"]])
    if "check" in todo:
        stage_local(run, "check", ["-m", "sim.check_batch", "--root", root])
    run.emit("driver", "done", "run finished", set_root=root)


if __name__ == "__main__":
    main()
