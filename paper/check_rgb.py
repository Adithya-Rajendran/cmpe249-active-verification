"""Independently reconstruct complete-input certificates from the delivered image archive."""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def audit(archive):
    with zipfile.ZipFile(archive) as z:

        def rows(name):
            return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8"))))

        queries = {r["id"]: r for r in rows("queries.csv")}
        cases = json.loads(z.read("cases.json"))
        if len(cases) != len(queries) or len({c["id"] for c in cases}) != len(queries):
            raise ValueError("Duplicate or missing cases")
        memo = {}

        def decoded(file_hash):
            if file_hash not in memo:
                value = z.read("rgb/" + file_hash + ".png")
                if hashlib.sha256(value).hexdigest() != file_hash:
                    raise ValueError("PNG file hash mismatch")
                with Image.open(io.BytesIO(value)) as im:
                    if im.size != (1200, 675):
                        raise ValueError("Delivered resolution mismatch")
                    memo[file_hash] = hashlib.sha256(
                        im.convert("RGB").tobytes()
                    ).hexdigest()
            return memo[file_hash]

        inputs = {}
        for c in cases:
            q = queries[c["id"]]
            fields = [tuple(c["box_xyxy"]), q["task"]]
            if tuple(json.loads(q["box"])) != fields[0]:
                raise ValueError("Box mismatch")
            for role in ("reference", "query"):
                d = c[role + "_sha256"]
                if q[role + "_file_sha256"] != d or q[
                    role + "_pixels_sha256"
                ] != decoded(d):
                    raise ValueError("RGB binding mismatch")
                fields.append(memo[d])
            inputs[c["id"]] = tuple(fields)
        groups = defaultdict(Counter)
        observed = set()
        for p in rows("pairs.csv"):
            a, b = p["present_id"], p["removed_id"]
            if a == b or a in observed or b in observed:
                raise ValueError("Duplicated pair member")
            observed.update((a, b))
            if (queries[a]["truth"], queries[b]["truth"]) != ("PRESENT", "REMOVED"):
                raise ValueError("Unbalanced pair")
            equal = inputs[a] == inputs[b]
            if equal != (p["exact"] == "1"):
                raise ValueError("Certificate differs")
            result = groups[p["engine"]]
            result["pairs"] += 1
            result["identical"] += equal
            if equal:
                for m in ("homography", "dense", "registered"):
                    result[m + "_different"] += queries[a][m] != queries[b][m]
                    result[m + "_commit"] += (
                        queries[a][m] != "UNKNOWN" or queries[b][m] != "UNKNOWN"
                    )
        if observed != queries.keys():
            raise ValueError("Unpaired observations")
        return {
            "case_count": len(cases),
            "distinct_pngs": len(memo),
            "engines": dict(groups),
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    print(json.dumps(audit(a.archive), indent=2))
