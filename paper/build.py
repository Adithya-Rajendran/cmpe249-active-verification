"""Build the CoRL preprint and a standalone archive of its editable LaTeX source."""

import csv
import json
import os
import subprocess
import tempfile
import zipfile
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEM = "Rajendran-counterfactual-scene-memory"
TABLE_INPUTS = (
    "evidence/summary.json",
    "evidence/still-comparison-counts.csv",
    "evidence/still-exact-pairs.csv",
)
ENGINES = {"isaac": "Isaac Sim", "mujoco": "MuJoCo"}
ORIGINAL = {"homography": "Homography", "dense": "Dense geometry", "registered": "Zero-shot Qwen"}
DELETION = {
    "homography": "Homography",
    "dense": "Dense geometry",
    "registered_qwen": "Zero-shot Qwen",
    "still_place": "STILL, registered crop",
    "still_place_conservative": "STILL, conservative crop",
    "still_fused": "STILL, scene plus crop",
}


def signed_points(number):
    rounded = Decimal(str(number)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return "$" + f"{rounded:+.1f}" + "$"


def read_csv(name):
    with (ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table_source():
    engines = json.loads((ROOT / TABLE_INPUTS[0]).read_text())["engines"]
    rows = {"PrimaryRows": [], "CoverageRows": [], "ExactRows": [], "DeletionRows": []}
    for engine, label in ENGINES.items():
        all_queries = engines[engine]["environments"]["all"]
        primary = all_queries["primary"]
        low, high = map(signed_points, primary["delta_interval_pp"])
        rows["PrimaryRows"].append((
            label, f"{primary['homography_correct']}/{primary['n']}",
            f"{primary['dense_correct']}/{primary['n']}",
            f"{signed_points(primary['delta_pp'])} ({low} to {high})",
        ))
        overall = all_queries["overall"]
        for method, title in ORIGINAL.items():
            counts = [
                overall[f"{method}_{stat}"]
                for stat in ("correct", "definite", "false_removal", "false_presence")
            ]
            rows["CoverageRows"].append((label if method == "homography" else "", title, *counts))
        cert = engines[engine]["certificates"]
        exact = cert["exact"]
        commitments = [f"{cert[method]['commit']}/{exact}" if exact else "---" for method in ORIGINAL]
        rows["ExactRows"].append((label, f"{exact}/{cert['pairs']}", cert["layouts"], *commitments))

    full = {
        row["method"]: row for row in read_csv(TABLE_INPUTS[1])
        if (row["subset"], row["engine"]) == ("full", "both")
    }
    exact_pairs = {r["method"]: r for r in read_csv(TABLE_INPUTS[2])}
    for method, label in DELETION.items():
        r = full[method]
        rows["DeletionRows"].append((
            label, r["missed_removals"], r["false_deletions"], exact_pairs[method]["delete_both"],
        ))

    lines = ["% Generated from archived evidence. Run build.py to regenerate these rows."]
    for command, records in rows.items():
        lines.append(f"\\newcommand{{\\{command}}}{{%")
        for cells in records:
            lines.append("  " + " & ".join(map(str, cells)) + r" \\")
        lines.append("}")
    return "\n".join(lines) + "\n"


def package_source(output):
    files = [
        "manuscript.tex", "appendix.tex", "result-tables.tex", "references.bib",
        "corl_2026.sty", "corlabbrvnat.bst", "build.py", *TABLE_INPUTS,
    ]
    files.extend(p.relative_to(ROOT).as_posix() for p in (ROOT / "figures").glob("*.png"))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, (2026, 11, 21, 12, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / name).read_bytes(), compress_type=zipfile.ZIP_DEFLATED)


def main():
    tables = ROOT / "result-tables.tex"
    generated = table_source()
    if not tables.exists() or tables.read_text() != generated:
        tables.write_text(generated, encoding="utf-8")

    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "1795262400")
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        staging = Path(folder)
        subprocess.run(
            ["tectonic", "--outdir", str(staging), "manuscript.tex"],
            cwd=ROOT, env=env, check=True,
        )
        archive = staging / f"{STEM}-latex.zip"
        package_source(archive)
        (staging / "manuscript.pdf").replace(ROOT / f"{STEM}.pdf")
        archive.replace(ROOT / archive.name)
    print(ROOT / f"{STEM}.pdf")
    print(ROOT / f"{STEM}-latex.zip")


if __name__ == "__main__":
    main()
