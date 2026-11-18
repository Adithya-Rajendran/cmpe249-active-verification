"""Render the dated feedback source with the existing plotting environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmpe249-feedback-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = HERE / "progress-feedback.md"
OUTPUT = HERE / "Rajendran-CMPE249-progress-feedback-2026-11-17.pdf"
CHECKS = REPO / "runs/2026-11-17-feedback-packet"
INK, MUTED, TEAL = "#19313D", "#526873", "#087D83"
WIDTH, HEIGHT, MARGIN = 612, 792, 48
TEXT_WIDTH = WIDTH - 2 * MARGIN
LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})


def blocks(source):
    """Yield headings, tables, images and paragraphs from this document's subset."""
    lines = source.strip().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("#"):
            yield "heading", (len(line) - len(line.lstrip("#")), line.lstrip("# "))
            index += 1
        elif line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].startswith("|"):
                row = [x.strip() for x in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", x) for x in row):
                    table.append(row)
                index += 1
            yield "table", table
        elif line.startswith("!["):
            yield "image", LINK.search(line).groups()
            index += 1
        elif re.match(r"(?:- |\d+\. )", line):
            yield "item", line
            index += 1
        else:
            paragraph = []
            while index < len(lines) and lines[index].strip():
                paragraph.append(lines[index].strip())
                index += 1
            yield "paragraph", " ".join(paragraph)


class Page:
    def __init__(self, number):
        self.figure = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=144, facecolor="white")
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.axes.set(xlim=(0, WIDTH), ylim=(0, HEIGHT))
        self.axes.axis("off")
        self.figure.canvas.draw()
        self.renderer = self.figure.canvas.get_renderer()
        self.y = HEIGHT - 44
        self.text_bounds = []
        self.axes.add_patch(Rectangle((MARGIN, HEIGHT - 31), 32, 3, color=TEAL))
        self.axes.plot([MARGIN, WIDTH - MARGIN], [32, 32], color="#D7E0E3", lw=.6)
        self.place(MARGIN, 24, "CMPE 249  |  Development progress  |  17 November 2026", 7.5, MUTED)
        self.place(WIDTH - MARGIN, 24, f"{number} / 2", 7.5, MUTED, align="right")

    def measure(self, text, size, bold=False):
        prop = FontProperties(family="DejaVu Sans", size=size, weight="bold" if bold else "normal")
        pixels = self.renderer.get_text_width_height_descent(text, prop, False)[0]
        return pixels * 72 / self.figure.dpi

    def wrap(self, text, size, width=TEXT_WIDTH, bold=False):
        lines, current = [], ""
        for word in text.split():
            candidate = f"{current} {word}".strip()
            if current and self.measure(candidate, size, bold) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def place(self, x, y, value, size, color=INK, bold=False, align="left", url=None):
        obj = self.axes.text(x, y, value, fontsize=size, va="top", ha=align,
                             color=color, weight="bold" if bold else "normal", url=url)
        box = obj.get_window_extent(self.renderer).transformed(self.axes.transData.inverted())
        self.text_bounds.append({"text": value, "bounds": list(box.extents)})
        if box.x0 < 20 or box.x1 > WIDTH - 20 or box.y0 < 9 or box.y1 > HEIGHT - 15:
            raise ValueError(f"Text outside page: {value}")
        return obj

    def paragraph(self, text, size=9.6, color=INK, gap=8, bold=False, indent=0):
        leading = size * 1.36
        for line in self.wrap(text, size, TEXT_WIDTH - indent, bold):
            self.place(MARGIN + indent, self.y, line, size, color, bold)
            self.y -= leading
        self.y -= gap

    def linked_line(self, text, size=8.2):
        x = MARGIN
        for label, url in LINK.findall(text):
            width = self.measure(label, size)
            if x + width > WIDTH - MARGIN:
                x, self.y = MARGIN, self.y - 13
            self.place(x, self.y, label, size, TEAL, url=url)
            x += width + 18
        self.y -= 18

    def table(self, rows):
        widths = [282, 150, 84]
        for index, row in enumerate(rows):
            height = 24 if index == 0 else 21
            if index == 0 or index % 2:
                color = "#E9F1F2" if index == 0 else "#F5F7F8"
                self.axes.add_patch(Rectangle((MARGIN, self.y - height), TEXT_WIDTH, height,
                                             color=color, lw=0))
            x = MARGIN
            for col, value in enumerate(row):
                position = x + 8 if col == 0 else x + widths[col] - 8
                self.place(position, self.y - 6, value, 8.4, bold=index == 0,
                           align="left" if col == 0 else "right")
                x += widths[col]
            self.y -= height
        self.y -= 9

    def image(self, caption, path):
        img = Image.open((HERE / path).resolve())
        height = min(176, TEXT_WIDTH * img.height / img.width)
        width = height * img.width / img.height
        left = (WIDTH - width) / 2
        self.axes.imshow(img, extent=(left, left + width, self.y - height, self.y), aspect="auto")
        self.y -= height + 7


def main():
    CHECKS.mkdir(parents=True, exist_ok=True)
    parts = SOURCE.read_text().split("<!-- pagebreak -->")
    assert len(parts) == 2
    layouts = []
    with PdfPages(OUTPUT, metadata={"Title": "CMPE 249 Project Progress Feedback",
                                   "Author": "Adithya Rajendran",
                                   "Subject": "Development results and questions for Professor Kaikai Liu"}) as pdf:
        for number, source in enumerate(parts, 1):
            page = Page(number)
            for kind, value in blocks(source):
                if kind == "heading":
                    level, text = value
                    page.paragraph(text, size=20 if level == 1 else 10.4,
                                   color=INK if level == 1 else TEAL, bold=True,
                                   gap=12 if level == 1 else 6)
                elif kind == "table":
                    page.table(value)
                elif kind == "image":
                    page.image(*value)
                elif LINK.search(value):
                    page.linked_line(value)
                elif value.startswith("Adithya Rajendran"):
                    page.paragraph(value, size=7.7, color=MUTED, gap=8)
                elif value == "Evidence-Aware Verification for Robot Scene Memory":
                    page.paragraph(value, size=12, color=MUTED, gap=8)
                else:
                    if value.startswith("- "):
                        value = "• " + value[2:]
                    page.paragraph(value, gap=5 if kind == "item" else 8)
            if page.y < 45:
                raise ValueError(f"Page {number} overlaps footer: y={page.y}")
            pdf.savefig(page.figure)
            page.figure.savefig(CHECKS / f"page-{number:02d}.png", dpi=144)
            layouts.append({"page": number, "content_bottom": page.y, "text": page.text_bounds})
            plt.close(page.figure)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (CHECKS / "layout.json").write_text(json.dumps({"pages": layouts,
        "source_sha256": sha(SOURCE), "builder_sha256": sha(Path(__file__)),
        "evidence_sha256": sha(HERE / "evidence.json"), "pdf_sha256": sha(OUTPUT)}, indent=2) + "\n")
    print(json.dumps({"pdf": str(OUTPUT), "pages": len(parts),
                      "content_bottom": [x["content_bottom"] for x in layouts]}))


if __name__ == "__main__":
    main()
