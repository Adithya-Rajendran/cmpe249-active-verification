"""Unmask adjudicated evidence and compare the instruction arms on paired cases."""
import argparse
import base64
import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import median

from development.cross_sim.localization.evidence import ARMS, SLOTS
from development.cross_sim.support_labels.evaluate import case_result, counts, screen, store_new
from development.cross_sim.support_labels.prepare import ENGINES, load, sha


def stage_summary(rows):
    values = [r["localization_iou"] for r in rows if r["localization_iou"] is not None]
    return {"n":len(rows),"estimated":len(values),"missing":len(rows)-len(values),
            "median_iou":median(values) if values else None}


def compare(rows):
    paired = {}
    for row in rows:
        paired.setdefault(row["id"],{})[row["arm"]]=row
    if any(set(p)!=set(ARMS) for p in paired.values()):
        raise ValueError("Unmatched cases in arm contrast")
    gains,losses,new_wrong,lost_wrong,new_exact = [],[],[],[],[]
    balance=Counter()
    for uid, p in paired.items():
        a,b=p["original"],p["two_step"]
        if a["condition"]=="partial":
            balance[a["environment"]+"/"+a["level"]+"/"+a["physical_state"]]+=1
            if a["physical_state"]=="REMOVED":
                if b["corroborated_definite"] and not a["corroborated_definite"]: gains.append(uid)
                if a["corroborated_definite"] and not b["corroborated_definite"]: losses.append(uid)
        if b["terra_wrong_definite"] and not a["terra_wrong_definite"]: new_wrong.append(uid)
        if a["terra_wrong_definite"] and not b["terra_wrong_definite"]: lost_wrong.append(uid)
        if b["terra_definite_on_exact"] and not a["terra_definite_on_exact"]: new_exact.append(uid)
    return {"new_corroborated_partial_removals":sorted(gains),"lost_corroborated_partial_removals":sorted(losses),
            "net_corroborated_partial_removals":len(gains)-len(losses),"new_wrong_definite":sorted(new_wrong),
            "lost_wrong_definite":sorted(lost_wrong),"new_definite_on_exact":sorted(new_exact),
            "partial_balance":dict(sorted(balance.items())),"complete_balance":len(balance)==8 and all(v==1 for v in balance.values())}


def report(run, dest):
    metrics = load(run/"measurements.json")
    reviews = load(run/"adjudication.json")
    locked = {r:load(run/"sealed"/(r+".json")) for r in (*ARMS,"main")}
    if any(sha(run/"sealed"/(r+".json"))!=h for r,h in metrics["reader_sha256"].items()):
        raise ValueError("Reader records changed after measurement")
    if set(reviews)!={e["case"]["id"] for e in metrics["rows"]} or any(set(v)!=set(SLOTS) for v in reviews.values()):
        raise ValueError("Missing or extra case adjudication")
    rows = []
    for entry in metrics["rows"]:
        c=entry["case"];uid=c["id"]
        for arm in ARMS:
            slot=entry["slot_for_arm"][arm]
            data={"case":c,"votes":{"terra":locked[arm]["votes"][uid],"main":locked["main"]["votes"][uid]},
                  "evidence":entry["evidence"][arm]}
            row=case_result(data,reviews[uid][slot])
            loc=locked["two_step"]["locations"][uid] if arm=="two_step" else {}
            row.update(arm=arm,figure_marker=slot,estimated_box=loc.get("box"),revisit_isolated_box=entry["isolated_box"],
                       localization_iou=entry["localization_iou"] if arm=="two_step" else None,
                       localization_landmarks=loc.get("landmarks"),localization_uncertainty=loc.get("uncertainty"))
            rows.append(row)
    rows.sort(key=lambda r:(r["base_id"],r["condition"],r["stop"],r["engine"],r["arm"]))
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"cases.csv").open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    arms={}
    for arm in ARMS:
        arms[arm]={}
        for engine in ENGINES:
            group=[r for r in rows if r["arm"]==arm and r["engine"]==engine]
            partial=[r for r in group if r["condition"]=="partial"]
            item={"qualification":screen(group),"overall":counts(group),
                  "conditions":{c:counts([r for r in group if r["condition"]==c]) for c in ("partial","clear","hidden")},
                  "partial_by_physical_state":{s:counts([r for r in partial if r["physical_state"]==s]) for s in ("PRESENT","REMOVED")}}
            if arm=="two_step":
                item["localization"]={"all":stage_summary(group),"partial":stage_summary(partial),
                      **{lv:stage_summary([r for r in partial if r["level"]==lv]) for lv in ("D0","D6")}}
            arms[arm][engine]=item
    contrasts={e:compare([r for r in rows if r["engine"]==e]) for e in ENGINES}
    advance=all(arms["two_step"][e]["qualification"]["outcome"]=="pass" and contrasts[e]["complete_balance"]
                and contrasts[e]["net_corroborated_partial_removals"]>=1 for e in ENGINES)
    output={"decision":"advance" if advance else "stop","unit":"base arrangement paired across simulators, cases paired across arms",
            "base_arrangements":len({r["base_id"] for r in rows}),"unique_cases":len(rows)//2,
            "terra_final_answers":len(rows),"arms":arms,"comparison":contrasts,
            "reader_sha256":metrics["reader_sha256"],"release_sha256":metrics["release_sha256"],
            "measurements_sha256":sha(run/"measurements.json"),"adjudication_sha256":sha(run/"adjudication.json"),
            "cases_sha256":sha(dest/"cases.csv"),"design_sha256":sha(run/"design-release.json")}
    store_new(dest/"results.json",output)
    return rows,output


def casebook(run,dest,rows):
    escape = html.escape
    title_for_arm={"original":"Original instructions","two_step":"Locate first"}
    cases={}
    for r in rows: cases.setdefault(r["id"],[]).append(r)
    links=[];sections=[]
    for uid,case in cases.items():
        first=case[0]
        engine={"isaac":"Isaac Sim","mujoco":"MuJoCo"}[first["engine"]]
        title=f'{engine} · {first["base_id"]} · {first["level"]}/{first["stop"]} · {first["condition"]}'
        links.append(f'<li><a href="#{uid}">{escape(title)}</a></li>')
        encoded=base64.b64encode((run/"cue-review"/"images"/(uid+".png")).read_bytes()).decode()
        block=[f'<section id="{uid}"><h2>{escape(title)}</h2><p>Case {uid}. Physical state {first["physical_state"]}.</p>',
               f'<img src="data:image/png;base64,{encoded}" alt="Earlier reference, marked revisit, and private opposite intervention">',
               f'<p><b>Independent main reading · {first["main_answer"]}</b> {escape(first["main_anchor"])} {escape(first["main_cue"])} '
               f'{escape(first["main_uncertainty"])} Point {first["main_point"]}.</p>']
        for row in case:
            arm=row["arm"]; heading=title_for_arm[arm];marker=row["figure_marker"]
            heading += " · "+("orange circle" if marker=="J1" else "purple square")
            block.append(f'<h3>{heading} · {row["terra_answer"]}</h3>')
            if arm=="two_step":
                iou=row["localization_iou"]
                iou_string="unavailable" if iou is None else f'{iou:.2f}'
                block.append(f'<p>Committed box {row["estimated_box"]}. Overlap {iou_string}. {escape(row["localization_landmarks"])} '
                             f'{escape(row["localization_uncertainty"])}</p>')
            block.append(f'<p>{escape(row["terra_anchor"])} {escape(row["terra_cue"])} {escape(row["terra_uncertainty"])}</p>')
            if row["terra_point"] is not None:
                n=row["terra_mask_pixels"]
                block.append(f'<p>Point {row["terra_point"]}. Retained target pixels at this point: {n} of 49.</p>')
            disposition=("Corroborated "+row["terra_answer"] if row["corroborated_definite"] else
                        "Certified UNKNOWN" if row["certificate_confirmed_unknown"] else "Unresolved")
            block.append(f'<p><b>{disposition}.</b> {escape(row["reviewer_note"])}</p>')
        sections.append("".join(block)+"</section>")
    doc=('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
         '<title>Original and two-step image reviews</title><style>body{max-width:1080px;margin:3rem auto;padding:0 1.1rem;color:#182d3c;'
         'font:17px/1.5 system-ui,sans-serif}h1,h2,h3{line-height:1.2}h3{font-size:1.1rem}p{margin:.4rem 0}section{border-top:1px solid #abc1cb;'
         'padding:1rem 0 1.5rem}img{width:100%;height:auto}summary{cursor:pointer}</style><main><h1>Original and two-step image reviews</h1>'
         '<p>Both groups received the same reference and revisit. The third panel shows the private opposite state. '
         'The independent main reviewer assessed both explanations under randomly assigned J1 and J2 labels before the instruction groups were revealed.</p>'
         '<details><summary>Jump to a case</summary><ul>'+''.join(links)+'</ul></details>'+''.join(sections)+'</main></html>')
    (dest/"casebook.html").write_text(doc)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("dest",type=Path)
    args=parser.parse_args()
    rows,data=report(args.run,args.dest)
    casebook(args.run,args.dest,rows)
    print(json.dumps({"decision":data["decision"],"unique_cases":data["unique_cases"],"comparison":data["comparison"]}))
