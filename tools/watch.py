"""Watch a pipeline run and print one line per event that needs a person; silent otherwise.

Follows runs/<run_id>/events.jsonl (written by tools/pipeline.py) and prints:
    ALERT ...   every alert event
    DONE ...    the terminal success event, then exits 0
    FAILED ...  a terminal alert, then exits 1
    ALERT ...   also when the driver process has died, or events.jsonl has gone quiet for longer than
                --stale-minutes (a hung driver writes nothing, so silence alone must not look like success)

Each printed line is meant to wake an operator (e.g. as a Claude Code Monitor event). Usage:
    python tools/watch.py runs/shelf_dev_v3 [--stale-minutes 20] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--stale-minutes", type=float, default=20)
    ap.add_argument("--from-start", action="store_true", help="also report alerts already in the log")
    ap.add_argument("--resume", action="store_true",
                    help="continue after the last event a previous watch handled (RUN_DIR/watch.offset), so a "
                         "re-armed watcher neither repeats nor misses events")
    args = ap.parse_args()
    events = args.run_dir / "events.jsonl"
    while not events.exists():
        time.sleep(2)
    f = events.open()
    offset_file = args.run_dir / "watch.offset"
    if args.resume and offset_file.exists():
        f.seek(int(offset_file.read_text()))
    elif not args.from_start:
        f.seek(0, os.SEEK_END)
    last_event, stale_reported, buf = time.time(), False, ""

    def drain() -> bool:
        """Handle every complete line written so far; return True if any arrived."""
        nonlocal buf, last_event, stale_reported
        got = False
        while chunk := f.readline():
            buf += chunk
            if not buf.endswith("\n"):
                break  # partial line; the rest comes later
            line, buf = buf, ""
            got, last_event, stale_reported = True, time.time(), False
            offset_file.write_text(str(f.tell()))  # before handle(), which may exit
            handle(json.loads(line))
        return got

    while True:
        if drain():
            continue
        time.sleep(2)
        try:
            pid = json.loads((args.run_dir / "status.json").read_text())["pid"]
        except (OSError, ValueError, KeyError):
            pid = None
        if pid is not None and not alive(pid):
            time.sleep(3)
            if not drain():  # a terminal event would have exited inside drain()
                print(f"FAILED [watch] driver pid {pid} died without a terminal event", flush=True)
                sys.exit(1)
            continue
        if not stale_reported and time.time() - last_event > args.stale_minutes * 60:
            print(f"ALERT [watch] no event for {args.stale_minutes:g} min (driver hung or ssh stuck?)", flush=True)
            stale_reported = True


def handle(ev: dict) -> None:
    where = f"[{ev['stage']}] {ev['msg']}"
    detail = {k: v for k, v in ev["data"].items() if k in ("arrangement", "line", "rc", "tail", "log")}
    if ev["level"] == "done":
        print(f"DONE {where}", flush=True)
        sys.exit(0)
    if ev["level"] == "alert":
        tag = "FAILED" if ev["data"].get("terminal") else "ALERT"
        print(f"{tag} {where} {json.dumps(detail) if detail else ''}".rstrip(), flush=True)
        if tag == "FAILED":
            sys.exit(1)

if __name__ == "__main__":
    main()
