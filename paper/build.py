"""Typeset the article and its figures. Run recount.py separately to audit the numbers."""

from pathlib import Path

import typst

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript.typ"
TARGET = ROOT / "Rajendran-counterfactual-scene-memory.pdf"

if __name__ == "__main__":
    typst.compile(
        str(SOURCE), output=str(TARGET), root=str(ROOT), pdf_standards=["1.7"]
    )
    print(TARGET)
