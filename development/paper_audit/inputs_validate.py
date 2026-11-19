"""Independently verify the full neutral packet and release the scorer-ready receipt."""
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from development.paper_audit.inputs_build import sha,verify_entries,box_from_mask,check_release
from development.paper_audit.evaluate_calibration import check_geometry
from development.paper_audit.evaluate_witness import primary_rgb,mask



@lru_cache(maxsize=8192)
def verify_mask(path):
    with Image.open(path) as im:
        if im.size!=(2304,1296): raise ValueError('mask dimensions')
        pixels=np.asarray(im)
        if not set(np.unique(pixels).tolist()).issubset({0,1,255}): raise ValueError('nonbinary mask')


def validate(packet,ledger_file,geometry_receipts,output):
    packet=Path(packet); ledger_file=Path(ledger_file)
    if packet.resolve() in ledger_file.resolve().parents: raise ValueError('ledger exposed in packet')
    ledger=json.loads(ledger_file.read_text()); cases=json.loads((packet/'cases.json').read_text())
    if sha(packet/'cases.json')!=ledger['cases_sha256'] or sha(packet/'packet-metadata.json')!=ledger['packet_metadata_sha256'] or check_release()!=ledger['design_release_sha256']:raise ValueError('packet or release binding mismatch')
    for path,digest in ledger['manifests'].items():
        if sha(path)!=digest: raise ValueError('capture manifest changed')
    geometry=check_geometry(geometry_receipts);private={r['id']:r for r in ledger['rows']};public={r['id']:r for r in cases}
    if len(private)!=len(ledger['rows']) or len(public)!=len(cases) or set(private)!=set(public):raise ValueError('packet id inventory mismatch')
    decl=json.loads(Path('development/paper_audit/evaluation_declaration.json').read_text())
    if [a['base_id'] for a in decl['arrangements']] != [f'PAM{i:02}' for i in range(1,17)] or [a['environment'] for a in decl['arrangements']] != ['shelf']*8+['tabletop']*8:
        raise ValueError('declared layout inventory mismatch')
    if decl['stops']!=[f'P{i}' for i in range(1,6)] or decl['states']!=['clear_present','clear_removed','placed_present','placed_removed']: raise ValueError('declared condition mismatch')
    expected={(engine,a['base_id'],lv,stop,state) for engine in ('isaac','mujoco') for a in decl['arrangements'] for lv in ('D0','D6') for stop in decl['stops'] for state in decl['states']}
    observed=set()
    for uid,c in public.items():
        if set(c)!={'id','reference','query','box_xyxy','reference_sha256','query_sha256'}: raise ValueError('unexpected public field')
        p=private[uid];r=p['capture'];state=p['state'];key=(r['engine'],r['arrangement'],r['level'],r['stop'],state)
        if key in observed: raise ValueError('duplicate design case')
        observed.add(key);verify_entries(r)
        if geometry.get(str(Path(r['observed']['path']).resolve()))!=r['observed']['sha256']:raise ValueError('geometry receipt mismatch')
        for source_mask in (r['reference']['target_mask'],r['isolated_target_mask'],r['images'][state]['target_mask'],r['images'][state]['blocker_mask']): verify_mask(source_mask['path'])
        if state.endswith('_removed') and mask(r['images'][state]['target_mask']['path']).any():raise ValueError('removed target mask nonempty')
        if c['box_xyxy']!=box_from_mask(r['reference']['target_mask']['path']):raise ValueError('box mismatch')
        for role,source in (('reference',r['reference']['rgb']['path']),('query',r['images'][state]['rgb']['path'])):
            path=(packet/c[role]).resolve()
            if path.parent!=packet.resolve() or sha(path)!=c[role+'_sha256']:raise ValueError('public file mismatch')
            with Image.open(path) as im:
                if im.mode!='RGB' or im.size!=(1200,675) or im.info:raise ValueError('unexpected image format or metadata')
                actual=np.asarray(im).copy()
            if not np.array_equal(actual,primary_rgb(source)):raise ValueError('native export mismatch')
    if observed!=expected or len(cases)!=1280:raise ValueError('incomplete declared cohort')
    out=Path(output)
    if out.exists(): raise FileExistsError('receipt exists')
    temporary=out.with_suffix(out.suffix+'.tmp')
    if temporary.exists(): raise FileExistsError('temporary receipt exists')
    temporary.write_text(json.dumps({'cases_path':str((packet/'cases.json').resolve()),'private_ledger_path':str(ledger_file.resolve()),
        'inventory_sha256':sha(packet/'cases.json'),'packet_metadata_sha256':sha(packet/'packet-metadata.json'),
        'ledger_sha256':sha(ledger_file),'geometry_receipt_sha256':{str(Path(p).resolve()):sha(p) for p in geometry_receipts},
        'validator_sha256':sha(__file__),'total_cases':len(cases),'status':'pass'},indent=2)+'\n')
    temporary.replace(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--ledger',required=True);p.add_argument('--geometry-receipt',action='append',required=True);p.add_argument('--out',required=True);a=p.parse_args();validate(a.packet,a.ledger,a.geometry_receipt,a.out)
