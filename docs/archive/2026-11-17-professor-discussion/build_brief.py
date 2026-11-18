"""Build the professor discussion PDF from a fixed evidence snapshot."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cmpe249-brief-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "evidence.json").read_text())
OUT = ROOT / "professor-discussion.pdf"
PREVIEW = ROOT / "previews"
PREVIEW.mkdir(exist_ok=True)

PAPER = "#F7F6F1"
INK = "#17313F"
MUTED = "#536973"
TEAL = "#007E87"
TEAL_LIGHT = "#E0EFEB"
BLUE = "#467E9D"
BLUE_LIGHT = "#E6EEF4"
GOLD = "#A76F16"
GOLD_LIGHT = "#F5EBD5"
RED = "#B24E3E"
RED_LIGHT = "#F5E6DF"
LINE = "#CBD5D5"
WHITE = "#FFFFFF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": PAPER,
})

pages = []
texts = []
content = []
layout_errors = []
text_overlaps = []
fig = ax = None


def line(x1, y1, x2, y2, color=LINE, lw=1.2, **kwargs):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, solid_capstyle="round", **kwargs)


def rect(x, y, w, h, fill=WHITE, stroke=None, radius=10, lw=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=stroke or fill, linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def t(x, y, text, size=13, color=INK, weight="normal", width=None,
      align="left", leading=1.35, record=True, url=None):
    """Place searchable text. Coordinates use a 1280 by 720 page."""
    s = str(text)
    if width:
        renderer = fig.canvas.get_renderer()
        prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
        wrapped = []
        for paragraph in s.split("\n"):
            words = paragraph.split()
            current = ""
            for word in words:
                trial = (current + " " + word).strip()
                measured = renderer.get_text_width_height_descent(trial, prop, False)[0]
                if measured > width and current:
                    wrapped.append(current)
                    current = word
                else:
                    current = trial
            wrapped.append(current)
        s = "\n".join(wrapped)
    obj = ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                  va="top", ha=align, linespacing=leading, url=url)
    texts.append(obj)
    if record:
        content.append(str(text).replace("\n", " "))
    return obj


def badge(x, y, text, color=TEAL, fill=TEAL_LIGHT, w=None):
    w = w or (len(text) * 7.2 + 24)
    rect(x, y, w, 27, fill, radius=5)
    t(x + 12, y + 7, text, 8.7, color, "bold")


def arrow(x1, y1, x2, y2, color=TEAL, lw=1.7, style="-|>", bend=None):
    p = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=13, linewidth=lw, color=color,
                        connectionstyle=bend or "arc3,rad=0")
    ax.add_patch(p)


def dot(x, y, n, color=TEAL, radius=16):
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=color))
    t(x, y - 9, n, 11, WHITE, "bold", align="center", record=False)


def image_asset(name, x, y, w, h):
    im = Image.open(ROOT / "assets" / name)
    iw, ih = im.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    xx, yy = x + (w - dw) / 2, y + (h - dh) / 2
    ax.imshow(im, extent=(xx, xx + dw, yy + dh, yy), zorder=2)


def card(x, y, w, h, label, title, body, accent=TEAL, fill=WHITE,
         title_size=17, body_size=12.3):
    rect(x, y, w, h, fill)
    line(x + 20, y + 24, x + 45, y + 24, accent, 3)
    t(x + 57, y + 16, label.upper(), 9, accent, "bold")
    t(x + 20, y + 52, title, title_size, INK, "bold", w - 40)
    t(x + 20, y + 105, body, body_size, MUTED, width=w - 40)


def start(num, section, title, subtitle=None, sources=""):
    global fig, ax, texts, content
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1280), ylim=(720, 0))
    ax.axis("off")
    texts = []
    content = [f"# {num:02d}. {title}"]
    t(54, 29, f"CMPE 249  /  {section.upper()}", 9.1, TEAL, "bold", record=False)
    t(1226, 29, "17 NOV 2026", 9.1, MUTED, align="right", record=False)
    t(54, 72, title, 27, INK, "bold", width=1160)
    if subtitle:
        t(55, 125, subtitle, 12.3, MUTED, width=1155)
    line(54, 673, 1226, 673, LINE, 0.8)
    t(54, 689, "ADITHYA RAJENDRAN  ·  SJSU  ·  DEVELOPMENT DISCUSSION", 7.8, MUTED, record=False)
    t(1118, 689, sources, 7.8, MUTED, align="right", record=False)
    t(1226, 685, f"{num:02d} / 10", 9.5, INK, "bold", align="right", record=False)


def finish(pdf, num):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for obj in texts:
        b = obj.get_window_extent(renderer)
        if b.x0 < 37 or b.x1 > 1243 or b.y0 < 10 or b.y1 > 704:
            layout_errors.append({"page": num, "text": obj.get_text(), "bounds": list(b.bounds)})
    for i, left in enumerate(texts):
        if not left.get_text():
            continue
        a = left.get_window_extent(renderer)
        for right in texts[i + 1:]:
            if not right.get_text():
                continue
            b = right.get_window_extent(renderer)
            overlap_w = min(a.x1, b.x1) - max(a.x0, b.x0)
            overlap_h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if overlap_w > 1 and overlap_h > 1:
                text_overlaps.append({"page": num, "first": left.get_text(), "second": right.get_text()})
    pdf.savefig(fig, dpi=150)
    fig.savefig(PREVIEW / f"page-{num:02d}.png", dpi=100)
    pages.append("\n\n".join(content))
    plt.close(fig)


def schematic(x, y, w, state, support, blocked=False):
    rect(x, y, w, 196, WHITE)
    line(x + 24, y + 121, x + w - 24, y + 121, LINE, 2)
    cx = x + w / 2
    if state == "present" and not blocked:
        rect(cx - 20, y + 56, 40, 64, TEAL, radius=5)
        ax.add_patch(Ellipse((cx, y + 57), 40, 9, facecolor="#54A5A8", edgecolor=TEAL))
        ax.add_patch(Ellipse((cx + 22, y + 87), 25, 31, facecolor="none", edgecolor=TEAL, lw=3))
    if blocked:
        rect(cx - 49, y + 46, 98, 76, "#AEC0C4", radius=2)
        for d in [-30, -10, 10, 30]:
            line(cx + d - 12, y + 116, cx + d + 12, y + 54, "#D6E0E0", 1)
        t(cx, y + 13, "NO DISTINGUISHING CUE", 7.6, MUTED, "bold", align="center")
    else:
        t(cx, y + 13, "IDENTIFYING CUE" if state == "present" else "EXPOSED REGION", 8, MUTED, "bold", align="center")
    t(cx, y + 137, support, 15.2, TEAL if support != "UNKNOWN" else GOLD, "bold", align="center")
    t(cx, y + 170, f"Physical state: {state}", 9.4, MUTED, align="center")


with PdfPages(OUT) as pdf:
    pdf.infodict().update({
        "Title": "Evidence-Aware Verification for Robot Scene Memory — Professor Discussion",
        "Author": "Adithya Rajendran",
        "Subject": "CMPE 249 methodology discussion, 17 November 2026",
        "Keywords": "robot scene memory, occlusion, verification, development plan",
    })

    # 1. Frame the meeting around the research decision.
    start(1, "Research direction", "When does a view justify an answer?",
          "Evidence-Aware Verification for Robot Scene Memory", "S1 · S6 · S7")
    t(55, 171, "Methodology discussion with Professor Kaikai Liu", 11.5, MUTED)
    rect(54, 213, 1172, 112, INK)
    t(78, 232, "PROPOSED NEXT STEP", 9, "#A5D7D3", "bold")
    t(78, 260, "Stress image-only verification before adding a model.", 22, WHITE, "bold", width=1100)
    card(54, 352, 371, 200, "Completed", "View selection", 
         "The first study left little room for a better selector. Preserve its results and design limits.",
         accent=BLUE)
    card(454, 352, 371, 200, "Current", "Supported answers",
         "Can the supplied view justify PRESENT, REMOVED or UNKNOWN for a remembered target?",
         accent=TEAL, fill=TEAL_LIGHT)
    card(854, 352, 372, 200, "Conditional", "Model contribution",
         "Add a model only if it improves useful answers or rejects unsupported answers beyond strong controls.",
         accent=GOLD)
    arrow(430, 451, 449, 451, MUTED)
    arrow(830, 451, 849, 451, MUTED)
    badge(54, 579, "DEVELOPMENT", w=132)
    t(202, 581, "Full render complete. Full-cohort decision pending. No new test freeze.", 12.1, MUTED, width=1024)
    t(55, 628, "Meeting aim: agree on the scientific question, the comparison, and the evidence needed to continue.", 11.7, INK)
    finish(pdf, 1)

    # 2. Distinguish the physical intervention from available evidence.
    start(2, "Task", "Physical truth and visual support are different",
          "A correct guess about a hidden object can still lack support.", "S1 · S2")
    for x, state, support, blocked in [
        (54, "present", "PRESENT", False),
        (352, "removed", "REMOVED", False),
        (650, "present", "UNKNOWN", True),
        (948, "removed", "UNKNOWN", True),
    ]:
        schematic(x, 181, 278, state, support, blocked)
    t(54, 391, "Conceptual examples. UNKNOWN describes the evidence, not a third physical state.", 11.1, MUTED)
    rect(54, 431, 733, 181, TEAL_LIGHT)
    t(76, 451, "FROM INPUTS TO ONE DECISION", 9, TEAL, "bold")
    for x, label, detail in [(76, "Reference image", "Target box + description"),
                             (322, "One revisit image", "No query box"),
                             (569, "One decision", "PRESENT / REMOVED / UNKNOWN")]:
        t(x, 486, label, 13.1, INK, "bold")
        t(x, 520, detail, 10.2, MUTED, width=195)
    arrow(288, 499, 310, 499)
    arrow(535, 499, 557, 499)
    t(76, 574, "Masks, physical state, true pose and paired queries stay with the evaluator.", 10.7, INK, width=687)
    rect(809, 431, 417, 181, WHITE)
    t(831, 451, "TASK BOUNDARY", 9, BLUE, "bold")
    t(831, 486, "REMOVED means absent from the remembered region.", 14.2, INK, "bold", width=371)
    t(831, 554, "Relocation and indistinguishable replacement stay outside the current task.", 11.3, MUTED, width=371)
    finish(pdf, 2)

    # 3. A visual map of the declared factorial experiment.
    start(3, "Stage 1", "A controlled stress test of the image verifier",
          "Vary camera displacement while keeping matched hidden and removed scenes comparable.", "S2 · S6")
    image_asset("shelf.png", 54, 182, 276, 155)
    image_asset("tabletop.png", 352, 182, 276, 155)
    t(54, 348, "SHELF", 9, BLUE, "bold")
    t(352, 348, "TABLETOP", 9, BLUE, "bold")
    t(54, 378, "12 fresh arrangements per environment", 17.2, INK, "bold", width=574)
    t(54, 420, "Actual reference renders from the smoke run.", 10.7, MUTED)
    rect(661, 182, 565, 257, WHITE)
    t(685, 203, "DECLARED MOTION LEVELS", 9, TEAL, "bold")
    line(706, 281, 1180, 281, LINE, 2)
    for x, name, cm, deg in [(706, "D0", "0 cm", "0°"), (823, "D1", "1 cm", "0.5°"),
                             (941, "D2", "2 cm", "1°"), (1059, "D4", "4 cm", "2°"),
                             (1180, "D6", "6 cm", "3°")]:
        ax.add_patch(Circle((x, 281), 6, color=TEAL))
        t(x, 246, name, 13, TEAL, "bold", align="center")
        t(x, 305, cm, 12, INK, "bold", align="center")
        t(x, 337, deg + " yaw", 10.5, MUTED, align="center")
    t(685, 389, "Camera-right translation + yaw. Four balanced sign combinations.", 10.5, MUTED, width=517)
    rect(54, 469, 1172, 78, INK)
    t(77, 490, "24 arrangements × 5 stops × 5 motion levels × 3 query phases", 17.6, WHITE, "bold")
    t(1189, 514, "1,800 planned query views", 12, "#A5D7D3", align="right")
    t(54, 570, "CONTROLLED", 9, TEAL, "bold")
    t(54, 595, "L1 lighting. Fixed D0 reference. Matched blockers and distractors. Primary size: 1200 × 675.", 11.5, MUTED, width=710)
    t(820, 570, "PARTIAL OCCLUSION", 9, GOLD, "bold")
    t(820, 595, "Geometry selection targets ≥20% middle-band stops at D0 in each environment.", 11.2, MUTED, width=405)
    finish(pdf, 3)

    # 4. Actual small-run counts, deliberately not styled as a full result.
    start(4, "Preliminary evidence", "A stronger image-only control already matters",
          "Smoke run only: one independent arrangement per environment. Each cell contains 15 query views.", "S3 · S4")
    t(54, 179, "DEFINITE ANSWERS / 15", 9, BLUE, "bold")
    t(54, 204, "These counts measure coverage, not supported accuracy.", 10.5, MUTED)
    table_x, table_y = 54, 250
    col_x = [358, 446, 545, 642]
    t(409, table_y, "Shelf", 11, BLUE, "bold", align="center")
    t(594, table_y, "Tabletop", 11, BLUE, "bold", align="center")
    for xx, lab in zip(col_x, ["D0", "D6", "D0", "D6"]):
        t(xx + 18, table_y + 29, lab, 10.2, MUTED, align="center")
    rows = [
        ("primary", "Primary", "Width 1200 · shift limit 100 px"),
        ("native", "Native resolution", "Width 2304 · shift limit 100 px"),
        ("native_normalized_guard", "Normalized limit", "Width 2304 · shift limit 192 px"),
        ("relaxed_guard", "Relaxed limit", "Width 1200 · shift limit 200 px"),
    ]
    for i, (arm, name, detail) in enumerate(rows):
        yy = 318 + i * 65
        rect(54, yy - 5, 658, 60, TEAL_LIGHT if arm == "relaxed_guard" else WHITE, radius=5)
        t(67, yy + 3, name, 11.7, INK, "bold")
        t(67, yy + 28, detail, 9.1, MUTED)
        for xx, cell in zip(col_x, ["shelf:D0", "shelf:D6", "tabletop:D0", "tabletop:D6"]):
            value = DATA["smoke_definite"][arm][cell]["n"]
            t(xx + 18, yy + 10, value, 17, TEAL if arm == "relaxed_guard" else INK,
              "bold", align="center")
    rect(751, 179, 475, 399, WHITE)
    t(775, 200, "TABLETOP · D6 · PROVISIONAL AI REFERENCE", 8.8, GOLD, "bold")
    t(775, 233, "Useful answers", 20, INK, "bold")
    for yy, arm, title, color in [(291, "primary", "Primary", BLUE), (382, "relaxed_guard", "Relaxed limit", TEAL)]:
        n = DATA["ai_tabletop_D6"][arm]["useful"]["numerator"]
        t(775, yy, title, 11.3, MUTED)
        rect(775, yy + 29, 338, 27, "#EDF0ED", radius=3)
        rect(775, yy + 29, 338 * n / 15, 27, color, radius=3)
        t(1196, yy + 27, f"{n}/15", 17, color, "bold", align="right")
    t(775, 478, "Answerable abstentions: 5 → 2", 14, INK, "bold")
    t(775, 517, "One disagreement with the AI reference remains in each control.", 11, MUTED, width=419)
    t(54, 591, "Every definite answer in the four smoke analyses matched physical state. Visual support remains a separate question.", 11.5, MUTED, width=1160)
    badge(54, 635, "DECISION", w=103)
    t(173, 637, "Test strong image-only controls before attributing recoverable answers to a model.", 11.6, INK)
    finish(pdf, 4)

    # 5. Label provenance and metrics.
    start(5, "Measurement", "Judge usefulness and unsupported answers together",
          "Keep physical state, geometric proxies and support judgments separate.", "S1 · S2 · S5")
    card(54, 179, 371, 192, "Physical state", "What happened?",
         "The scene intervention records whether the target stayed or left the remembered region.", accent=BLUE)
    card(454, 179, 371, 192, "Geometry proxy", "What is exposed?",
         "Masks describe obstruction. They do not establish recognition or evidence of absence.", accent=TEAL)
    card(854, 179, 372, 192, "Support judgment", "What can we infer?",
         "A blinded reviewer sees only the allowed image pair. AI judgments remain provisional.", accent=GOLD)
    disagreement = DATA["ai_proxy_agreement"]["count"] - DATA["ai_proxy_agreement"]["agreement_count"]
    rect(54, 399, 371, 227, GOLD_LIGHT)
    t(76, 417, f"{disagreement} / 60", 34, GOLD, "bold")
    t(76, 476, "AI–geometry disagreements", 14.2, INK, "bold", width=326)
    t(76, 519, "The main reviewer disputes at least one AI judgment. Preserve original labels and report sensitivity.", 11.6, MUTED, width=326)
    t(76, 599, "Smoke audit, not an AI accuracy score.", 9.5, MUTED)
    rect(454, 399, 772, 227, WHITE)
    t(476, 418, "Useful yield  Y", 16.6, TEAL, "bold")
    t(476, 452, "Supported, physically correct definite answers / support-labeled inputs", 12.1, INK, width=720)
    line(476, 495, 1204, 495)
    t(476, 512, "Unsupported definite rate  U", 16.6, RED, "bold")
    t(476, 547, "Definite answers that disagree with the support label / support-labeled inputs", 12.1, INK, width=720)
    t(476, 595, "UNKNOWN lowers coverage. More answers alone do not establish improvement.", 10.7, MUTED, width=720)
    t(54, 645, "The full AI audit preselects 180 queries. A support-based go decision needs complete labels under the current rule.", 10.5, MUTED)
    finish(pdf, 5)

    # 6. Exact decisions from the adopted protocol.
    start(6, "Decision rules", "Let the result choose the next experiment",
          "Apply the development rules separately in each environment. Do not pool motion levels to trigger a branch.", "S1 · S2")
    rect(54, 178, 1172, 63, INK)
    t(77, 196, "First: finish full review, run all four controls, and complete the required support labels.", 15.6, WHITE, "bold", width=1125)
    branches = [
        (54, "A", "Unsupported answers", RED, RED_LIGHT,
         "U > 0.01 at one nonzero level, with events in at least three arrangements.",
         "Permit a qualified agreement-veto comparison.",
         "The rule must hold with undisputed labels and with DISPUTED treated as UNKNOWN."),
        (454, "B", "Useful answers decline", GOLD, GOLD_LIGHT,
         "No A, and useful yield at D6 falls at least 0.10 below D0.",
         "Drop veto as the main claim. Assess answerable abstentions.",
         "A cascade needs recoverable answers on ≥0.10 of labeled inputs, beyond credible image-only controls."),
        (854, "C", "Neither rule holds", TEAL, TEAL_LIGHT,
         "No A and no B under the declared rules.",
         "Stop expanding this simulation axis. Assess real development images.",
         "Keep the boundary finding. Do not add a model without an evidence-based reason."),
    ]
    for x, letter, title, color, fill, condition, action, caveat in branches:
        rect(x, 279, 372, 326, fill)
        dot(x + 31, 307, letter, color)
        t(x + 59, 294, title, 15.1, INK, "bold", width=291)
        t(x + 22, 347, condition, 12.2, INK, width=328)
        line(x + 22, 432, x + 350, 432, color, 0.7)
        t(x + 22, 451, action, 12.3, color, "bold", width=328)
        t(x + 22, 527, caveat, 10.8, MUTED, width=328)
        arrow(x + 186, 245, x + 186, 273, color)
    t(54, 627, "AI-only decisions are provisional. Geometry alone cannot authorize the model stage.", 11.9, INK)
    t(54, 650, "These thresholds guide development. They are not statistical significance tests or deployment guarantees.", 10.2, MUTED)
    finish(pdf, 6)

    # 7. A conditional model architecture and its qualification requirements.
    start(7, "Conditional stage 2", "Use a model only where it can add evidence",
          "The v1 model interface failed retained-target controls. Qualify the next interface before a task comparison.", "S1 · S7")
    rect(54, 181, 1172, 95, WHITE)
    t(75, 198, "QUALIFY THE INTERFACE FIRST", 9, TEAL, "bold")
    t(75, 230, "Qwen first", 14.5, INK, "bold")
    t(322, 230, "≥34/36 same-view PRESENT", 13.2, INK, "bold")
    t(763, 230, "≥90% correct in each clear-state class", 13, INK, "bold")
    rect(54, 306, 235, 91, TEAL_LIGHT)
    t(171, 326, "Image verifier", 16, INK, "bold", align="center")
    t(171, 359, "Same permitted image pair", 10.4, MUTED, align="center")
    arrow(294, 351, 351, 351)
    t(321, 318, "UNKNOWN", 8.5, GOLD, "bold", align="center")
    rect(359, 306, 342, 91, BLUE_LIGHT)
    t(530, 325, "Qualified model + rejection", 14.5, INK, "bold", align="center")
    t(530, 360, "No simulator truth or extra query", 10.2, MUTED, align="center")
    arrow(706, 351, 764, 351)
    rect(773, 306, 453, 91, WHITE)
    t(999, 325, "PRESENT / REMOVED / UNKNOWN", 14, INK, "bold", align="center")
    t(999, 360, "Rejected or failed calls return UNKNOWN", 10, MUTED, align="center")
    arrow(171, 401, 171, 443)
    t(171, 452, "Keep a definite image answer", 11.4, TEAL, "bold", align="center")
    t(359, 418, "The cascade cannot correct a definite image-only error.", 11.1, MUTED, width=826)
    t(359, 452, "At matched thresholds, veto can discard errors. It cannot add useful answers.", 11.1, MUTED, width=826)
    rect(54, 500, 721, 132, INK)
    t(77, 517, "CASCADE CONTINUATION RULE", 9, "#A5D7D3", "bold")
    t(77, 552, "Δ useful yield ≥ +0.10", 19, WHITE, "bold")
    t(77, 594, "Δ unsupported rate ≤ +0.02", 16, WHITE, "bold")
    rect(797, 500, 429, 132, GOLD_LIGHT)
    t(819, 517, "BOUNDED MODEL WORK", 9, GOLD, "bold")
    t(819, 548, "Permit one larger replacement after qualification failure. Stop model claims if both fail.", 11.8, INK, width=385)
    t(54, 646, "Require both cascade thresholds per claimed environment, including disputed-label sensitivity, against credible image-only controls.", 10.2, MUTED)
    finish(pdf, 7)

    # 8. Deliverables, measurement unit, and paper-readiness boundaries.
    start(8, "Roadmap", "Progress through evidence milestones",
          "The schedule follows the results. The course has no fixed deliverable in the current record.", "S1 · S2 · S6 · S8")
    steps = [
        (54, "1", "Finish the stress test", "NOW", TEAL,
         "Review the full outputs. Score four controls. Resolve label-sensitive decisions.",
         "Output: Y, U, refusals and recoverable answers by environment and level."),
        (352, "2", "Test the model", "CONDITIONAL", BLUE,
         "Qualify the interface. Split calibration and evaluation by whole arrangement.",
         "Output: cascade or veto comparison with image-only and confidence controls."),
        (650, "3", "Check real images", "AFTER C OR USEFUL STAGE 2", GOLD,
         "Plan ten arrangements, three stops and three query phases. Measure ten camera returns.",
         "Output: transfer limits, support labels and failure cases from real scenes."),
        (948, "4", "Prepare a fresh test", "PAPER READINESS", RED,
         "Compare closest work. Obtain independent support-label validation. Choose a generalization axis.",
         "Output: fixed settings, justified precision, sample size and a new freeze."),
    ]
    for x, n, title, state, color, body, output in steps:
        rect(x, 189, 278, 332, WHITE)
        dot(x + 28, 215, n, color)
        t(x + 17, 249, state, 7.4, color, "bold")
        t(x + 17, 281, title, 16.1, INK, "bold", width=244)
        t(x + 17, 341, body, 11.2, MUTED, width=244)
        t(x + 17, 444, output, 10.3, color, width=244)
    for x in [334, 632, 930]:
        arrow(x, 211, x + 14, 211, MUTED)
    rect(54, 550, 1172, 87, TEAL_LIGHT)
    t(77, 568, "THE INDEPENDENT UNIT IS THE ARRANGEMENT", 9, TEAL, "bold")
    t(77, 598, "Keep all views, states and motion levels together in splits and bootstrap draws. UNKNOWN counts against coverage.", 12.1, INK, width=1118)
    t(54, 649, "Current snapshot: 120 render folders complete. Structural validation passed. Full scientific review and decision remain open.", 10, MUTED)
    finish(pdf, 8)

    # 9. A meeting worksheet with concrete choices, not a speculative promise.
    start(9, "Professor discussion", "Four decisions to leave the meeting with",
          "The paper is a research aim. Its contribution remains conditional on stronger evidence.", "S1 · S8 · P1 · P2")
    prompts = [
        ("01", "Research question", "Is target-level supported-answer evaluation worth pursuing beyond change detection?",
         "Working position: keep the claim narrow. Do not claim a new abstention principle."),
        ("02", "Baseline comparison", "Which two-image baseline would make the comparison convincing?",
         "Candidates: CYWS-3D and the two-image SceneDiff setting. Declare adapters before evaluation."),
        ("03", "Real evidence and labels", "What real-data scope and independent label review would support a workshop submission?",
         "Working position: start with the small real pilot, then justify a fresh evaluation."),
        ("04", "Stopping and contribution", "If strong image-only controls solve the task, is the boundary finding worth pursuing?",
         "Working position: retain negative evidence and stop a model direction that adds no value."),
    ]
    for i, (n, title, question, position) in enumerate(prompts):
        yy = 180 + 110 * i
        rect(54, yy, 1172, 96, WHITE)
        t(75, yy + 17, n, 20, TEAL, "bold")
        t(133, yy + 17, title, 13, INK, "bold", width=220)
        t(379, yy + 15, question, 12.4, INK, "bold", width=822)
        t(379, yy + 57, position, 10.5, MUTED, width=822)
    t(54, 641, "Meeting notes:  priority __________________     baseline __________________     next evidence __________________", 10.7, MUTED)
    finish(pdf, 9)

    # 10. Compact source register with exact evidence paths and definitions.
    start(10, "Source register", "Evidence behind this discussion",
          "Snapshot: 17 November 2026. Results shown here are development observations.", "")
    t(54, 180, "PROJECT DOCUMENTS", 9, TEAL, "bold")
    refs = [
        ("S1", "Method, evaluation and current plan", "docs/METHOD.md · docs/EVALUATION_PLAN.md\nprogress/PLAN.md · progress/meetings/2026-11-17.md"),
        ("S2", "Stress design and label amendment", "development/stress/PROTOCOL.md\ndevelopment/stress/LABEL_PLAN_AI_V1.md"),
        ("S3", "Smoke counts and interpretation", "notes/2026-11-17-stress-execution.md\n[R]/reviewer/smoke_evidence_audit.json"),
        ("S4", "AI-reference useful yield", "[R]/label-evaluation-v1/ai-primary.json\n[R]/label-evaluation-v1/ai-relaxed_guard.json"),
        ("S5", "Label disagreements and spot review", "[R]/label-evaluation-v1/ai-vs-proxy.json\n[R]/reviewer/ai_label_spot_review.json"),
        ("S6", "Render completion and validation", "[R]/reviewer/ipv4_and_render_completion.json\n[R]/full-review/cohort-validation.json"),
    ]
    for i, (key, title, path) in enumerate(refs):
        yy = 212 + 67 * i
        t(54, yy, key, 9.4, TEAL, "bold")
        t(89, yy, title, 10.5, INK, "bold")
        t(89, yy + 22, path, 8.7, MUTED, width=630, leading=1.22)
    t(765, 180, "HISTORY AND READINESS", 9, TEAL, "bold")
    t(765, 214, "S7  Pilot failure and study limits", 10.6, INK, "bold")
    t(765, 239, "notes/2026-11-16-development-pilot.md\nnotes/2026-11-12-posthoc-review.md", 8.7, MUTED)
    t(765, 288, "S8  Research context and freeze policy", 10.6, INK, "bold")
    t(765, 313, "notes/2026-11-17-research-and-stress-plan.md\ndocs/FREEZE_POLICY.md", 8.7, MUTED)
    t(765, 369, "CLOSEST-WORK CANDIDATES", 9, TEAL, "bold")
    t(765, 402, "P1  CYWS-3D · Sachdeva & Zisserman", 10.4, INK, "bold")
    t(765, 428, "Two RGB images under viewpoint changes.", 10.2, MUTED)
    t(765, 453, "arxiv.org/abs/2308.10417", 10.2, TEAL, url="https://arxiv.org/abs/2308.10417")
    t(765, 498, "P2  SceneDiff · Wu and colleagues", 10.4, INK, "bold")
    t(765, 524, "Verifiable changes under occlusion and viewpoint shifts.", 10.1, MUTED, width=450)
    t(765, 567, "arxiv.org/abs/2512.16908", 10.2, TEAL, url="https://arxiv.org/abs/2512.16908")
    t(54, 627, "[R] = runs/2026-11-17-stress-dev-v1. Paths are relative to the project repository.", 9.7, MUTED)
    t(54, 650, "Evidence snapshot and file hashes accompany this PDF. AI assisted document preparation. No new experiment ran for this brief.", 9.4, MUTED)
    finish(pdf, 10)


(ROOT / "slide-content.md").write_text("\n\n---\n\n".join(pages) + "\n")
(ROOT / "layout-check.json").write_text(json.dumps({"page_count": len(pages), "out_of_page_text": layout_errors, "text_overlaps": text_overlaps}, indent=2) + "\n")

# A contact sheet supports visual review without a PDF editor.
thumbs = []
for p in sorted(PREVIEW.glob("page-*.png")):
    im = Image.open(p).convert("RGB")
    im.thumbnail((640, 360))
    thumbs.append(im)
sheet = Image.new("RGB", (1280, 360 * 5), "#DDE3E1")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % 2) * 640, (i // 2) * 360))
sheet.save(PREVIEW / "contact-sheet.jpg", quality=90)
print(json.dumps({"pdf": str(OUT), "pages": len(pages), "layout_errors": layout_errors, "text_overlaps": text_overlaps, "bytes": OUT.stat().st_size}, indent=2))
