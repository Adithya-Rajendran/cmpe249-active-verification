"""Measure the two locked arms and prepare evidence with concealed arm identities."""
import argparse
import json
import secrets
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from development.cross_sim.support_labels.evaluate import patch_pass, store_new
from development.cross_sim.support_labels.prepare import load, sha
from development.cross_sim.support_labels.witness import case_evidence, mask_native, mask_primary, rgb_primary


ARMS = ("original","two_step")
SLOTS = ("J1","J2")


def bbox(mask):
    yy,xx = np.nonzero(mask)
    if not len(yy):
        raise ValueError("Missing isolated target")
    return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]


def overlap(a,b):
    if a is None:
        return None
    w = max(0,min(a[2],b[2])-max(a[0],b[0]))
    h = max(0,min(a[3],b[3])-max(a[1],b[1]))
    area_a=(a[2]-a[0])*(a[3]-a[1])
    area_b=(b[2]-b[0])*(b[3]-b[1])
    return w*h/(area_a+area_b-w*h)


def figure(c, votes, directory, iso, visible, alternate):
    with Image.open(c["prepared_reference"]) as ref:
        reference=ref.convert("RGB")
    with Image.open(c["prepared_query"]) as current:
        query=current.convert("RGB")
    b = c["box_xyxy"]
    ImageDraw.Draw(reference).rectangle([b[0],b[1],b[2]-1,b[3]-1],outline="#f5c62a",width=2)
    border=visible & (np.asarray(Image.fromarray(visible.astype(np.uint8)*255).filter(ImageFilter.MinFilter(5)))==0)
    alternate[border]=[100,250,125]
    other = "removed" if c["state"].endswith("_present") else "retained"
    extents=[b,iso]+[[v["point"][0],v["point"][1],v["point"][0]+1,v["point"][1]+1]
                         for v in votes.values() if v["point"] is not None]
    x0=min(x[0] for x in extents);y0=min(x[1] for x in extents)
    x1=max(x[2] for x in extents);y1=max(x[3] for x in extents)
    w=min(1200,max(220,x1-x0+100,int((y1-y0+72)*1.7)))
    h=min(675,int(w/1.7))
    left=max(0,min(1200-w,(x0+x1-w)//2))
    top=max(0,min(675-h,(y0+y1-h)//2))
    sheet=Image.new("RGB",(1176,289),"white")
    draw=ImageDraw.Draw(sheet)
    fonts="/usr/share/fonts/truetype/dejavu/DejaVuSans"
    font=ImageFont.truetype(fonts+"-Bold.ttf",17)
    small=ImageFont.truetype(fonts+".ttf",15)
    panels = [("Earlier reference",reference),("Judged revisit",query),
              ("Other state · target "+other,Image.fromarray(alternate))]
    for col,(title,pixels) in enumerate(panels):
        panel=pixels.crop((left,top,left+w,top+h)).resize((378,222),Image.Resampling.NEAREST)
        if col==1:
            d=ImageDraw.Draw(panel)
            for key,v in votes.items():
                if v["point"] is None: continue
                x=(v["point"][0]-left+.5)*378/w-.5
                y=(v["point"][1]-top+.5)*222/h-.5
                if key=="main":
                    d.line((x-6,y,x+6,y),fill="#21d7ff",width=3)
                    d.line((x,y-6,x,y+6),fill="#21d7ff",width=3)
                elif key=="J1":
                    d.ellipse((x-7,y-7,x+7,y+7),outline="#ffae22",width=3)
                else:
                    d.rectangle((x-9,y-9,x+9,y+9),outline="#e86be6",width=2)
        px=9+col*390
        draw.text((px,3),title,font=font,fill="#183044")
        sheet.paste(panel,(px,31))
    draw.text((9,261),"Gold box = earlier target     Orange circle = J1     Purple square = J2     Blue cross = main     Green outline = exposed retained target",
              font=small,fill="#374e60")
    sheet.save(directory/(c["id"]+".png"))


def measure(run,roots):
    prep = run/"prepared"
    release_path=prep/"release.json"
    release=load(release_path)
    if (sha(prep/"ledger.json")!=release["ledger_sha256"] or sha(run/"design-release.json")!=release["design_sha256"]):
        raise ValueError("Case inputs changed")
    records={k:load(run/"sealed"/(k+".json")) for k in (*ARMS,"main")}
    if any(r["release_sha256"]!=sha(release_path) for r in records.values()):
        raise ValueError("Reader lock mismatch")
    ledger=load(prep/"ledger.json")
    if any(set(r["votes"])!={c["id"] for c in ledger} for r in records.values()):
        raise ValueError("Reader inventories differ")
    private=[];audit=[]
    cards=run/"cue-review"/"images"
    cards.mkdir(parents=True,exist_ok=True)
    for c in ledger:
        uid=c["id"]
        order=list(SLOTS)
        secrets.SystemRandom().shuffle(order)
        slot_for_arm=dict(zip(ARMS,order))
        votes={slot_for_arm[a]:records[a]["votes"][uid] for a in ARMS}
        votes["main"]=records["main"]["votes"][uid]
        evidence={}
        for arm,slot in slot_for_arm.items():
            evidence[arm]=case_evidence(c,{"terra":votes[slot],"main":votes["main"]},roots)
            if evidence[arm]["mask_counts_native"]["removed_target"]:
                raise ValueError("Target removal mask must be empty")
        first,second=(evidence[a] for a in ARMS)
        if (first["primary_pixel_sha256"]!=second["primary_pixel_sha256"] or
                first["reviewer_points"]["main"]!=second["reviewer_points"]["main"]):
            raise ValueError("Shared case or independent point differs between arms")
        root=roots[c["engine"]]/c["base_id"]/c["level"]
        suffix="_"+c["stop"]+".png"
        placement,truth=c["state"].split("_")
        other="removed" if truth=="present" else "present"
        isolated=bbox(mask_primary(mask_native(root/"masks"/("alone"+suffix))))
        visible=mask_primary(mask_native(root/"masks"/("target_"+placement+"_present"+suffix)))
        alternate=rgb_primary(root/"rgb"/(placement+"_"+other+suffix))
        location=records["two_step"]["locations"][uid]
        metrics={"main":first["reviewer_points"]["main"]}
        for arm,slot in slot_for_arm.items():
            metrics[slot]=evidence[arm]["reviewer_points"]["terra"]
        figure(c,votes,cards,isolated,visible,alternate)
        audit.append({"id":uid,"case":{k:c[k] for k in ("engine","base_id","environment","level","stop","state","condition")},
                      "visible": c["measurements"]["fraction_visible"] if c["condition"]!="clear" else 1.,
                      "exact":first["exact_opposite_input_certificate"],"votes":votes,"metrics":metrics})
        private.append({"case":c,"evidence":evidence,"slot_for_arm":slot_for_arm,
                        "isolated_box":isolated,"estimated_box":location["box"],"localization_iou":overlap(location["box"],isolated)})
    store_new(run/"measurements.json",{"rows":private,"reader_sha256":{k:sha(run/"sealed"/(k+".json")) for k in records},
                                       "release_sha256":sha(release_path)})
    store_new(run/"cue-review"/"cases.json",sorted(audit,key=lambda r:(r["case"]["base_id"],r["case"]["condition"],r["case"]["stop"],r["case"]["engine"])))
    print(json.dumps({"review_cases":len(audit),"blinded_explanations":2*len(audit)}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("run","isaac","mujoco"):p.add_argument("--"+arg,type=Path,required=True)
    o=p.parse_args()
    measure(o.run,{"isaac":o.isaac,"mujoco":o.mujoco})
