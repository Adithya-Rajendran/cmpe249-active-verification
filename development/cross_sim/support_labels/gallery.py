"""Show the declared images, hidden counterpart and locked points after both reviews."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from development.cross_sim.support_labels.witness import mask_native, mask_primary, rgb_primary


ORANGE = "#ffab32"
BLUE = "#29cbfb"
GREEN = (108, 255, 118)


def draw_case(entry, roots, directory):
    c, votes = entry["case"], entry["votes"]
    folder = roots[c["engine"]] / c["base_id"] / c["level"]
    setting, truth = c["state"].split("_")
    other = "removed" if truth == "present" else "present"
    tag = "_" + c["stop"] + ".png"
    alternate = rgb_primary(folder/"rgb"/(setting+"_"+other+tag))
    visible = mask_primary(mask_native(folder/"masks"/("target_"+setting+"_present"+tag)))
    silhouette = mask_primary(mask_native(folder/"masks"/("alone"+tag)))
    x0, y0, x1, y1 = c["box_xyxy"]
    ys, xs = np.nonzero(silhouette)
    if not len(xs):
        raise ValueError("Missing isolated target silhouette")
    xx = [x0, x1, xs.min(), xs.max()]
    yy = [y0, y1, ys.min(), ys.max()]
    for vote in votes.values():
        if vote["point"] is not None:
            xx.append(vote["point"][0]); yy.append(vote["point"][1])
    w = min(1200, max(220, max(xx)-min(xx)+110, int((max(yy)-min(yy)+84)*1.7)))
    h = min(675, int(w/1.7))
    l = int(np.clip((max(xx)+min(xx)-w)//2, 0, 1200-w))
    t = int(np.clip((max(yy)+min(yy)-h)//2, 0, 675-h))
    crop = (l,t,int(l+w),int(t+h))
    with Image.open(c["prepared_reference"]) as src:
        ref = src.convert("RGB")
    with Image.open(c["prepared_query"]) as src:
        query = src.convert("RGB")
    ImageDraw.Draw(ref).rectangle((x0,y0,x1-1,y1-1),outline="#fec832",width=2)
    # The surviving silhouette belongs to the retained intervention in this pair.
    # Its border also marks where the can, brick, or other target would be visible after removal.
    mask = Image.fromarray(visible.astype(np.uint8)*255)
    border = visible & (np.asarray(mask.filter(ImageFilter.MinFilter(5))) == 0)
    alternate[border] = GREEN
    panels = [ref,query,Image.fromarray(alternate)]
    labels = ["Earlier reference", "Judged revisit", "Other state · target " + other]
    fonts = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
    font = ImageFont.truetype(fonts+"-Bold.ttf", 17)
    small = ImageFont.truetype(fonts+".ttf", 15)
    sheet = Image.new("RGB", (1176,297), "white")
    draw = ImageDraw.Draw(sheet)
    for i, im in enumerate(panels):
        panel = im.crop(crop).resize((378,222),Image.Resampling.NEAREST)
        if i==1:
            d = ImageDraw.Draw(panel)
            for reviewer in ("terra","main"):
                pt = votes[reviewer]["point"]
                if pt is None:
                    continue
                x=(pt[0]-l+.5)*378/w-.5; y=(pt[1]-t+.5)*222/h-.5
                if reviewer=="terra":
                    d.ellipse((x-7,y-7,x+7,y+7),outline="#141a20",width=4)
                    d.ellipse((x-7,y-7,x+7,y+7),outline=ORANGE,width=2)
                else:
                    for color, width in (("#141a20",5),(BLUE,3)):
                        d.line((x-6,y,x+6,y),fill=color,width=width)
                        d.line((x,y-6,x,y+6),fill=color,width=width)
        x = 9 + i*390
        draw.text((x,3),labels[i],font=font,fill="#183044")
        sheet.paste(panel,(x,31))
    draw.text((9,267),"Gold box = remembered target     Orange circle = Terra point     Blue + = main point     Green = visible target area in the retained state",font=small,fill="#374e60")
    directory.mkdir(parents=True,exist_ok=True)
    sheet.save(directory/(c["id"]+".png"))


def casebook(run, results, directory):
    with (results/"case-review.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    esc=html.escape
    sections=[]
    for r in rows:
        uid = r["id"]
        encoded=base64.b64encode((directory/(uid+".png")).read_bytes()).decode("ascii")
        engine = {"isaac": "Isaac Sim", "mujoco": "MuJoCo"}[r["engine"]]
        title=f'{engine} · {r["base_id"]} · {r["level"]}/{r["stop"]} · {r["condition"]}'
        decision = "Unresolved"
        if r["corroborated_definite"] == "True":
            decision = "Corroborated " + r["terra_answer"]
        elif r["certificate_confirmed_unknown"] == "True":
            decision = "Certified UNKNOWN"
        parts=[f'<section id="{esc(uid)}"><h2>{esc(title)}</h2><p>Case {uid} · Physical state {r["physical_state"]}</p>',
               f'<img src="data:image/png;base64,{encoded}" alt="Reference, judged revisit with review points, and evaluator-only opposite state">']
        for prefix,name in (("terra","Terra"),("main","Independent main review")):
            def field(k):
                return esc(r[prefix+"_"+k])
            rationale = " ".join(field(k).rstrip(".") + "." for k in ("target","anchor","cue","uncertainty"))
            detail = "No point supplied."
            if field("point"):
                detail = f'Point {field("point")}. {field("mask_pixels") or "0"} of 49 pixels lie in the retained target mask.'
                if field("answer") in ("PRESENT", "REMOVED"):
                    status = "passed" if field("point_pass") == "True" else "failed"
                    detail += f' The local evidence check {status}.'
            parts.append(f'<h3>{name} · {field("answer")}</h3><p>{rationale}</p><p class="measure">{detail}</p>')
        parts.append(f'<h3>{decision}</h3><p>{esc(r["reviewer_note"])}</p></section>')
        sections.append("".join(parts))
    doc=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
         '<title>Visual support pilot evidence</title><style>body{max-width:1080px;margin:3rem auto;padding:0 1.1rem;color:#182d3c;'
         'font:17px/1.5 system-ui,sans-serif}h1,h2,h3{line-height:1.2}h3{font-size:1rem;margin-bottom:.2rem}p{margin:.3rem 0 .85rem}'
         'section{border-top:1px solid #abc1cb;padding:1rem 0 1.5rem}img{width:100%;height:auto}.measure{font-size:.9rem;color:#4b5a63}</style>'
         '<main><h1>Visual support pilot evidence</h1><p>Each case contains both locked readings and the subsequent evidence review. '
         'Only the evaluator saw the other intervention and simulator masks. The green outline marks the exposed part of the retained target in either state.</p>'
         +''.join(sections)+'</main></html>')
    (results/"casebook.html").write_text(doc)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=("cards","book"))
    ap.add_argument("run",type=Path)
    for name in ("isaac","mujoco","results"):
        ap.add_argument("--"+name,type=Path)
    opt=ap.parse_args()
    cards=opt.run/"evidence-cards"
    if opt.mode=="cards":
        entries=json.loads((opt.run/"measurements.json").read_bytes())["rows"]
        for entry in entries:
            draw_case(entry,{"isaac":opt.isaac,"mujoco":opt.mujoco},cards)
    else:
        casebook(opt.run,opt.results,cards)
