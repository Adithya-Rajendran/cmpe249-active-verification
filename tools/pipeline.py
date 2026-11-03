"""Run driver for a simulated dev set: render on the GB10, build episodes and check the set on the Jetson, then
compute per-view evidence on the GB10 and calibrate each evidence variant with select-v4 on the Jetson.

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
    evidence  ship `git archive HEAD` (a clean tree, frozen perception/evidence.py) to the GB10 and run the
              spec's evidence commands there, one after another, in the GB10 project uv env; the outputs land on
              the shared dataset volume. Alerts: an error line, a non-zero step, a step slower than
              step_alert_minutes, the job gone without an rc file, an output with the wrong number of records.
              Writes provenance.json (git sha, commands, sha256 and record count of every output).
    select    python -m verify.select on each variant's evidence (random dev arrangements only). Exit 0, 3
              (calibration infeasible) and 4 (candidate not eligible for freeze) are results, not failures.
    summary   python tools/evidence_summary.py: per-lighting alignment refusals, gate pass by view class,
              held-out policy FRR / recall / coverage for every variant, and the DES cases separately.

A stage whose spec section is missing is skipped. Stdlib only. Usage:
    python tools/pipeline.py configs/pipeline/shelf_dev_v3.json [--launch] [--from-stage episodes]
    python tools/pipeline.py configs/pipeline/shelf_dev_v3_evidence.json --from-stage evidence
"""
from __future__ import annotations

import argparse
import hashlib
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
STAGES = ("render", "episodes", "check", "evidence", "select", "summary")
# perception/evidence.py is frozen (v0.2.1); the evidence stage refuses to ship a tree where it differs.
FROZEN_EVIDENCE_SHA256 = "d0e0927ff9d36ee2a8027cc8050f1801e372faea1e12cd98c513345d258d46cf"
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

    def argv(self, command: str) -> list[str]:
        c = self.cfg
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"HostKeyAlias={c['host_key_alias']}",
                "-i", os.path.expanduser(c["key"]), f"{c['user']}@{c['host']}", command]

    def run(self, command: str, timeout: int = 60, stdin: bytes | str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(self.argv(command), input=stdin, capture_output=True, timeout=timeout,
                              text=not isinstance(stdin, bytes))


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


def stage_local(run: Run, stage: str, module_args: list[str], ok_rcs: tuple[int, ...] = (0,),
                out_name: str | None = None) -> int:
    argv = run.spec["local_python"] + module_args
    out = run.dir / f"{out_name or stage}.out"
    run.emit(stage, "info", "started", argv=argv)
    t0 = time.time()
    with out.open("w") as f:
        rc = subprocess.run(argv, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode
    tail = out.read_text().splitlines()[-15:]
    if rc not in ok_rcs:
        run.fail(stage, f"{out_name or stage} exited with rc {rc}", rc=rc, tail=tail, log=str(out))
    run.emit(stage, "info", "finished", rc=rc, seconds=round(time.time() - t0, 1), tail=tail[-5:])
    return rc


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def evidence_steps(spec: dict) -> list[dict]:
    """One step per (jobs file, variant): the command, and the outputs it must write with their record count."""
    ev, root = spec["evidence"], spec["set_root"]
    steps = []
    for jobs in ev["jobs"]:
        jobs_path = f"{root}/{jobs}"
        n = len(json.loads(Path(jobs_path).read_text())["jobs"])
        part = Path(jobs).stem.removeprefix("jobs_")
        for name, v in ev["variants"].items():
            fmt = {"jobs": jobs_path, "out": ev["out_dir"], "part": part}
            steps.append({"name": f"{name}_{part}", "args": [a.format(**fmt) for a in v["args"]], "records": n,
                          "outputs": [o.format(**fmt) for o in v["outputs"]]})
    return steps


def stage_evidence(run: Run) -> None:
    ev = run.spec["evidence"]
    remote = Remote(run.spec["remote"])
    head = git("rev-parse", "HEAD").strip()
    if git("status", "--porcelain", "--untracked-files=no").strip():
        run.fail("evidence", "tracked files differ from HEAD; commit first (the GB10 runs `git archive HEAD`)")
    frozen = hashlib.sha256(subprocess.run(["git", "show", "HEAD:perception/evidence.py"], cwd=REPO,
                                           capture_output=True, check=True).stdout).hexdigest()
    if frozen != FROZEN_EVIDENCE_SHA256:
        run.fail("evidence", "perception/evidence.py at HEAD is not the frozen v0.2.1", sha256=frozen)
    steps = evidence_steps(run.spec)
    rd = run.spec["remote"]["run_dir"]
    tag = f"evidence_{run.run_id}_{head[:12]}"
    log, rc_file, pid_file = f"{rd}/{tag}.log", f"{rd}/{tag}.rc", f"{rd}/{tag}.pid"
    code = f"{ev['code_dir']}/{head[:12]}"

    def probe() -> dict:
        cmd = (f"test -f {pid_file} && kill -0 $(cat {pid_file}) 2>/dev/null && echo RUNNING=1 || echo RUNNING=0; "
               f"test -f {rc_file} && echo RC=$(cat {rc_file}) || echo RC=")
        out = remote.run(cmd).stdout.split()
        return {"running": "RUNNING=1" in out, "rc": next((o[3:] for o in out if o.startswith("RC=")), "")}

    state = probe()
    if not state["running"] and state["rc"] == "":
        archive = subprocess.run(["git", "archive", "--format=tar", head], cwd=REPO, capture_output=True,
                                 check=True).stdout
        res = remote.run(f"rm -rf {code} && mkdir -p {code} {ev['out_dir']} && tar -x -C {code} && "
                         f"ln -sfn {ev['checkpoints']} {code}/checkpoints", timeout=300, stdin=archive)
        if res.returncode != 0:
            run.fail("evidence", "could not ship the code to the GB10", stderr=str(res.stderr)[-500:])
        py = f"env PYTHONPATH={code} {ev['uv']} run --no-sync --project {ev['project']} python"
        lines = ["#!/bin/bash", "set -u", f"cd {code}"]
        for i, st in enumerate(steps):
            mark = json.dumps({"i": i, "n": len(steps), "name": st["name"]})
            lines += [f"echo 'EVIDENCE_STEP start {mark}'", "t0=$(date +%s)",
                      f"{py} {' '.join(shlex.quote(a) for a in st['args'])}", "rc=$?",
                      f"printf 'EVIDENCE_STEP end {{\"i\": {i}, \"rc\": %d, \"seconds\": %d}}\\n' "
                      f"$rc $(( $(date +%s) - t0 ))",
                      f"if [ $rc -ne 0 ]; then echo $rc > {rc_file}; exit $rc; fi"]
        lines.append(f"echo 0 > {rc_file}")
        script = f"{rd}/{tag}.sh"
        remote.run(f"cat > {script}", stdin="\n".join(lines) + "\n")
        remote.run(f"cd {rd} && setsid nohup bash {script} > {log} 2>&1 < /dev/null & echo $! > {pid_file}")
        run.emit("evidence", "info", "launched", git=head, code=code, steps=[s["name"] for s in steps])
    else:
        run.emit("evidence", "info", "attached", running=state["running"], rc=state["rc"])

    offset, step_started, ssh_failures, last_heartbeat = 0, time.time(), 0, 0.0
    alert_after = ev.get("step_alert_minutes", 30) * 60
    alerted = False
    while True:
        try:
            res = remote.run(f"tail -n +{offset + 1} {log} 2>/dev/null", timeout=120)
            state = probe()
            ssh_failures = 0
        except (subprocess.TimeoutExpired, OSError) as e:
            ssh_failures += 1
            if ssh_failures >= 5:
                run.fail("evidence", "GB10 unreachable over ssh 5 times in a row", error=str(e))
            time.sleep(run.spec["poll_seconds"])
            continue
        new_lines = res.stdout.splitlines()
        offset += len(new_lines)
        for line in new_lines:
            if line.startswith("EVIDENCE_STEP start "):
                rec = json.loads(line.split(" ", 2)[2])
                step_started, alerted = time.time(), False
                run.emit("evidence", "info", f"step {rec['i'] + 1}/{rec['n']} {rec['name']} started", **rec)
            elif line.startswith("EVIDENCE_STEP end "):
                rec = json.loads(line.split(" ", 2)[2])
                run.emit("evidence", "info", f"step {rec['i'] + 1} {steps[rec['i']]['name']} finished", **rec)
            elif ERROR_LINE.match(line.strip()) and not line.startswith(IGNORED):
                run.emit("evidence", "alert", "error line in the evidence log", line=line.strip()[:500])
        if state["rc"] != "":
            if state["rc"] != "0":
                run.fail("evidence", f"evidence exited with rc {state['rc']}", log=log)
            break
        if not state["running"]:
            time.sleep(10)
            if probe()["rc"] == "":
                run.fail("evidence", "evidence job is gone and wrote no rc file", log=log)
            continue
        if not alerted and time.time() - step_started > alert_after:
            run.emit("evidence", "alert", f"an evidence step has run for over {alert_after / 60:.0f} min")
            alerted = True
        if time.time() - last_heartbeat > run.spec["heartbeat_seconds"]:
            run.emit("evidence", "info", "heartbeat")
            last_heartbeat = time.time()
        time.sleep(run.spec["poll_seconds"])

    # the outputs are on the shared volume: check every one and record its provenance
    outputs = {}
    for st in steps:
        for o in st["outputs"]:
            path = Path(o)
            if not path.exists():
                run.fail("evidence", f"{st['name']} did not write {o}")
            n = sum(1 for line in path.open() if line.strip())
            outputs[o] = {"sha256": sha256(path), "records": n}
            if n != st["records"]:
                run.fail("evidence", f"{o} has {n} records, expected {st['records']} (one per job)")
    prov = {"git": head, "code_dir": code, "frozen_evidence_sha256": frozen, "set_root": run.spec["set_root"],
            "steps": steps, "outputs": outputs, "remote_log": log, "finished": now()}
    text = json.dumps(prov, indent=1, sort_keys=True)
    (run.dir / "evidence_provenance.json").write_text(text)
    # written by the GB10 user, who owns out_dir on the shared volume
    remote.run(f"cat > {ev['out_dir']}/provenance_{head[:12]}.json", stdin=text)
    run.emit("evidence", "info", "evidence finished", outputs=len(outputs))


def stage_select(run: Run) -> None:
    sel, root = run.spec["select"], run.spec["set_root"]
    out_root = run.dir / "select"
    results = {}
    for name, evidence in sel["variants"].items():
        evidence = evidence.format(out=run.spec["evidence"]["out_dir"])
        rc = stage_local(run, "select", ["-m", "verify.select", "--manifest", f"{root}/manifest.json",
                                         "--evidence", evidence, "--view-labels", f"{root}/view_labels.json",
                                         "--positions", f"{root}/{sel['positions']}", "--model-id", name,
                                         "--tag", name, "--out", str(out_root / name)],
                         ok_rcs=(0, 3, 4), out_name=f"select_{name}")
        results[name] = {0: "eligible", 3: "calibration infeasible", 4: "candidate not eligible for freeze"}[rc]
        run.emit("select", "info", f"{name}: {results[name]}", variant=name, rc=rc)
    (out_root / "outcomes.json").write_text(json.dumps(results, indent=1, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec", type=Path)
    ap.add_argument("--launch", action="store_true", help="start the render if it is not already running")
    ap.add_argument("--from-stage", choices=STAGES, default="render")
    ap.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
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
    todo = STAGES[STAGES.index(args.from_stage):STAGES.index(args.to_stage) + 1]
    if "render" in todo:
        stage_render(run, args.launch)
    if "episodes" in todo:
        stage_local(run, "episodes", ["-m", "sim.episodes", "--root", root, "--contact-sheets",
                                      "--generation-log", spec["generation_log"]])
    if "check" in todo:
        stage_local(run, "check", ["-m", "sim.check_batch", "--root", root])
    for stage, fn in (("evidence", stage_evidence), ("select", stage_select)):
        if stage in todo:
            if stage in spec:
                fn(run)
            else:
                run.emit(stage, "info", "skipped: no section in the spec")
    if "summary" in todo and "summary" in spec:
        stage_local(run, "summary", ["tools/evidence_summary.py", "--spec", str(args.spec.resolve()),
                                     "--run-dir", str(run.dir)])
    run.emit("driver", "done", "run finished", set_root=root)


if __name__ == "__main__":
    main()
