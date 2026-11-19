"""Replay the three paper commands on Ascent with a new anonymous packet and output directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def score(args):
    packet = args.cases.resolve()
    new = args.out.resolve()
    old = args.historical.resolve()
    candidate = args.candidate.resolve()
    project = args.project.resolve()
    if digest(packet) != args.sha256:
        raise ValueError("Anonymous packet checksum differs from its release")
    cases = load(packet)
    by_id = {x["id"]: x for x in cases}
    if len(cases) != len(by_id) or len(cases) != 320:
        raise ValueError("Incorrect new case inventory")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, (old, candidate, project))),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        YOLO_AUTOINSTALL="false",
        TOKENIZERS_PARALLELISM="false",
    )
    guard = candidate / "development/stress/offline_exec.py"
    new.mkdir(parents=True, exist_ok=True)
    for name, prior in [
        ("baseline_200", "baseline-guard.json"),
        ("geometry", "geometry-guard.json"),
        ("vggt_qwen", "vggt-qwen-guard.json"),
    ]:
        receipt = load(old / prior)
        script = Path(receipt["script"])
        before = load(old / name / "binding.json")
        arguments = list(receipt["argv"])
        oldhash = before.pop("cases_sha256")
        require_hash = digest(script)
        if (
            require_hash != before["script_sha256"]
            or digest(guard) != receipt["guard_sha256"]
        ):
            raise ValueError("Original program changed")
        arguments[arguments.index("--cases") + 1] = str(packet)
        arguments[arguments.index("--out") + 1] = str(new / name)
        output = new / name
        verified = new / (name + "-verified.json")
        if not output.exists():
            cmd = [
                sys.executable,
                str(guard),
                "--guard-record",
                str(new / (name + "-guard.json")),
                str(script),
                *arguments,
            ]
            with (new / (name + ".log")).open("x") as log:
                subprocess.run(
                    cmd,
                    check=True,
                    cwd=candidate,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        bind = load(output / "binding.json")
        packet_hash = bind.pop("cases_sha256")
        if bind != before or packet_hash != args.sha256:
            raise ValueError(f"{name} binding differs from prior")
        rows = [
            json.loads(line)
            for line in (output / "records.jsonl").read_text().splitlines()
        ]
        if len(rows) != 320 or {x["id"] for x in rows} != set(by_id):
            raise ValueError(f"Incomplete {name} output")
        for row in rows:
            c = by_id[row["id"]]
            if row["answer"] not in ("PRESENT", "REMOVED", "UNKNOWN") or any(
                row[k] != c[k] for k in ["box_xyxy", "query_sha256", "reference_sha256"]
            ):
                raise ValueError("Wrong score or case")
        receipt = {
            "arm": name,
            "cases": len(rows),
            "cases_sha256": packet_hash,
            "previous_cases_sha256": oldhash,
            "records_sha256": digest(output / "records.jsonl"),
            "binding_sha256": digest(output / "binding.json"),
            "technical_failures": sum(
                x.get("failure")
                not in [None, "", False, "registration_gate", "invalid_model_output"]
                for x in rows
            ),
            "invalid_answers": sum(
                x.get("failure") == "invalid_model_output" for x in rows
            ),
            "registration_stops": sum(
                x.get("failure") == "registration_gate" for x in rows
            ),
        }
        if verified.exists():
            if load(verified) != receipt:
                raise ValueError("Previously verified scores changed")
        else:
            verified.write_text(json.dumps(receipt, indent=2) + "\n")
        print(name, len(rows), "original output records verified", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("cases", "out", "historical", "candidate", "project"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--sha256", required=True)
    score(p.parse_args())
