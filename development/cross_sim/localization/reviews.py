"""Bind each location estimate and blind review to the originally issued pixels."""
import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

from development.cross_sim.localization.prepare import AUDIT_FIELDS, HERE, ORIGINAL
from development.cross_sim.support_labels.evaluate import ANSWERS, DEFINITE, store_new
from development.cross_sim.support_labels.prepare import ROLES, load, sha


def now():
    return datetime.now(timezone.utc).isoformat()


def text_fields(row, fields):
    return all(isinstance(row.get(k),str) and row[k].strip() for k in fields)


def inventory(path, cases):
    data = load(path)
    if not isinstance(data,list) or len(data)!=len(cases) or {r["id"] for r in data}!=set(cases):
        raise ValueError("Reader inventory differs from its packet: "+str(path))
    return data


def check_packet(packet):
    folder = Path(packet["path"])
    if sha(folder/"cases.json")!=packet["sha256"] or sha(folder/"Instructions.md")!=packet["instructions_sha256"]:
        raise ValueError("Packet changed")
    rows = load(folder/"cases.json")
    if len(rows)!=packet["n"] or len({r["id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate or missing packet cases")
    for r in rows:
        if set(r)!=set(AUDIT_FIELDS):
            raise ValueError("Unexpected fields in blind packet")
        for k in ROLES:
            if sha(folder/(r["id"]+"-"+k+".png")) != r[k+"_sha256"]:
                raise ValueError("Presented pixels changed")
    return [r["id"] for r in rows]


def check_release(run):
    prep = run/"prepared"
    release = load(prep/"release.json")
    ledger = load(prep/"ledger.json")
    lookup = {r["id"]:r for r in ledger}
    if (len(lookup)!=len(ledger) or sha(prep/"ledger.json")!=release["ledger_sha256"] or
            sha(prep/"selection.json")!=release["selection_sha256"]):
        raise ValueError("Private case ledger changed")
    packets = release["packets"]
    by_pair = {}
    arms = {"original":[],"two_step":[]}
    for packet in [*packets, release["main"]]:
        folder = Path(packet["path"])
        ids = check_packet(packet)
        expected = {"cases.json","Instructions.md"}|{uid+"-"+role+".png" for uid in ids for role in ROLES}
        if {f.name for f in folder.iterdir()} != expected:
            raise ValueError("Extra files in a blind packet")
        if any(r != {k:lookup[r["id"]][k] for k in AUDIT_FIELDS} for r in load(folder/"cases.json")):
            raise ValueError("Packet mismatch with delivered images")
        if "arm" not in packet:
            if set(ids)!=set(lookup): raise ValueError("Main review is incomplete")
            continue
        arm = packet["arm"]
        if len({lookup[uid]["base_id"] for uid in ids})!=len(ids):
            raise ValueError("Same arrangement twice in one packet")
        if arm in by_pair.setdefault(packet["pair"],{}):
            raise ValueError("Repeated arm in matched packet pair")
        by_pair[packet["pair"]][arm] = (packet["sha256"],ids)
        arms[arm].extend(ids)
    if any(len(items)!=len(lookup) or set(items)!=set(lookup) for items in arms.values()):
        raise ValueError("Incomplete arm")
    if any(pair.get("original")!=pair.get("two_step") for pair in by_pair.values()):
        raise ValueError("Packet inputs or order differ by arm")
    if len({(r["engine"],r["base_id"],r["level"],r["stop"]) for r in ledger})!=len(ledger):
        raise ValueError("A camera intervention repeats")
    group = Counter(r["condition"] for r in ledger)
    receipt = {"status":"pass","n":len(lookup),"counts":group,"matched_packet_pairs":len(by_pair),
               "one_arrangement_per_session":True,"identical_pixels_boxes_order_between_arms":True,
               "one_intervention_per_engine_camera":True,"utc":now(),"release_sha256":sha(prep/"release.json")}
    store_new(prep/"blind-audit.json",receipt)
    return receipt


def validate_locations(rows):
    for row in rows:
        box = row["box"]
        if not text_fields(row,("target","landmarks","uncertainty")):
            raise ValueError("Missing localization text")
        if box is not None and not (isinstance(box,list) and len(box)==4 and all(type(x) is int for x in box)
                                  and 0<=box[0]<box[2]<=1200 and 0<=box[1]<box[3]<=675):
            raise ValueError("Invalid localization estimate")


def validate_votes(rows):
    for row in rows:
        pt = row["point"]
        if row["answer"] not in ANSWERS or not text_fields(row,("target","anchor","cue","uncertainty")):
            raise ValueError("Invalid answer or missing cue")
        if pt is None:
            if row["answer"] in DEFINITE: raise ValueError("Missing definite point")
        elif not (isinstance(pt,list) and len(pt)==2 and all(type(v) is int for v in pt)
                and 0<=pt[0]<1200 and 0<=pt[1]<675):
            raise ValueError("Invalid point")


def decision_instructions():
    return (HERE/"DECIDE.md").read_text()+"\n"+ORIGINAL.read_text()


def broker(run):
    prep = run/"prepared"
    if load(prep/"blind-audit.json")["release_sha256"] != sha(prep/"release.json"):
        raise ValueError("Unaudited release")
    registry = run/"sealed"/"locations"
    registry.mkdir(parents=True,exist_ok=True)
    waiting = {Path(p["path"]):p for p in load(prep/"release.json")["packets"] if p["arm"]=="two_step"}
    while waiting:
        for folder,packet in list(waiting.items()):
            source = folder/"localization.json"
            if not source.exists():
                continue
            try:
                rows = inventory(source,[r["id"] for r in load(folder/"cases.json")])
                validate_locations(rows)
            except (ValueError,KeyError,TypeError):
                continue
            sealed = registry/(folder.name+".json")
            if not sealed.exists():
                copyfile(source,sealed)
            if sha(sealed)!=sha(source):
                raise ValueError("The first-stage submission changed")
            record = registry/(folder.name+"-receipt.json")
            if not record.exists():
                store_new(record,{"sha256":sha(sealed),"utc":now(),"packet_sha256":packet["sha256"]})
            # The second stage appears only after the submitted estimates are safely copied.
            dest = folder/"Decision.md"
            if not dest.exists():
                temp = folder/".decision-writing"
                temp.write_text(decision_instructions())
                temp.replace(dest)
            if dest.read_text()!=decision_instructions():
                raise ValueError("Changed second-stage instructions")
            del waiting[folder]
        if waiting:
            time.sleep(1)


def seal(run,role):
    prep = run/"prepared"
    release = load(prep/"release.json")
    audit = load(prep/"blind-audit.json")
    if audit["release_sha256"]!=sha(prep/"release.json"):
        raise ValueError("Release changed after checks")
    packets = [release["main"]] if role=="main" else [p for p in release["packets"] if p["arm"]==role]
    sealed = run/"sealed"
    sealed.mkdir(exist_ok=True)
    votes,locations,meta = {},{},[]
    for pkt in packets:
        folder = Path(pkt["path"])
        ids = check_packet(pkt)
        if votes.keys() & set(ids): raise ValueError("An arm repeats a case")
        rows = inventory(folder/"votes.json",ids)
        validate_votes(rows)
        votes.update({r["id"]:r for r in rows})
        info = {"packet":folder.name,"inventory_sha256":pkt["sha256"],"votes_sha256":sha(folder/"votes.json")}
        if role=="two_step":
            saved = sealed/"locations"/(folder.name+".json")
            receipt = load(sealed/"locations"/(folder.name+"-receipt.json"))
            if not (receipt["sha256"]==sha(saved)==sha(folder/"localization.json")
                    and receipt["packet_sha256"]==pkt["sha256"] and (folder/"Decision.md").read_text()==decision_instructions()):
                raise ValueError("Localization lock mismatch")
            frozen = inventory(saved,ids)
            validate_locations(frozen)
            locations.update({r["id"]:r for r in frozen})
            info["localization_sha256"] = sha(saved)
            info["location_lock_utc"] = receipt["utc"]
        meta.append(info)
    if len(votes)!=audit["n"]: raise ValueError("Incomplete arm")
    store_new(sealed/(role+".json"),{"role":role,"utc":now(),"release_sha256":sha(prep/"release.json"),
              "votes":votes,"locations":locations,"packets":meta})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("audit","broker","original","two_step","main"))
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    if args.mode=="audit": print(json.dumps(check_release(args.run)))
    elif args.mode=="broker": broker(args.run)
    else: seal(args.run,args.mode)
