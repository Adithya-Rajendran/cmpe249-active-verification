"""Plot two saved MuJoCo false removals with evaluator-only projected vertices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from development.cross_sim import geometry_residuals as residuals
from development.cross_sim import landmark_residuals as landmarks
from development.cross_sim import qualification_geometry as geometry

ROOT = Path('/mnt/bench-data/datasets/cmpe249-active-verification/cross-sim-dev-20261117')
OUT = Path('notes/figures/2026-11-18-geometry-comparison')
CASES = [('Clear query', 'Ia48298bb838a8ecb2e05724fb53a01f6', 'clear_present'),
         ('Placed query', 'I0a4384b27d274900f01ff39596efb118', 'placed_present')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    prep = ROOT/'pilot/scoring-v1/prepared'
    cohort = ROOT/'pilot/cohort'
    candidate = ROOT/'geoscd-rgb-development-v1/candidate-v1'
    manifest_path = candidate/'manifest.json'
    manifest_sha = sha(manifest_path)
    preparation = read(prep/'preparation.json')
    ledger_path, jobs_path = prep/'ledger.json', prep/'verifier/primary/jobs.json'
    if preparation['ledger_sha256'] != sha(ledger_path) or preparation['jobs_sha256']['primary'] != sha(jobs_path):
        raise ValueError('prepared input binding changed')
    ledger = {r['id']: r for r in read(ledger_path)}
    jobs = {r['id']: r for r in read(jobs_path)}
    arrangement = cohort/'CSPS05D4.json'
    cohort_manifest = cohort/'manifest.json'
    if sha(cohort_manifest) != preparation['cohort_sha256']:
        raise ValueError('cohort binding changed')
    arr = read(arrangement)
    target = arr['objects']['target']
    if target['ycb'] != '061_foam_brick':
        raise ValueError('unexpected target')
    ply = ROOT/'assets/models'/target['ply']
    vertices = geometry.object_mesh(target, ROOT/'assets/models')['vertices']
    ref, _ = geometry.project(vertices, geometry.camera_record(arr, 'P4'))
    query, _ = geometry.project(vertices, geometry.camera_record(arr, 'P4', revisit=True))
    ref, query = landmarks.sample_index(ref, (1200, 675)), landmarks.sample_index(query, (1200, 675))
    maps, mask, images, details = [], np.ones(len(ref), bool), [], []
    for label, ident, state in CASES:
        item, job = ledger[ident], jobs[ident]
        if (item['base_id'], item['level'], item['stop'], item['engine'], item['state'], item['physical_state']) != ('CSPS05','D4','P4','mujoco',state,'PRESENT'):
            raise ValueError('unexpected case identity')
        if item['arrangement_sha256'] != sha(arrangement): raise ValueError('arrangement changed')
        rec_path = candidate/'records'/f'{ident}.json'
        rec = read(rec_path)
        if (rec['id'] != ident or rec['binding_sha256'] != manifest_sha or rec['answer'] != 'REMOVED'
                or rec['failure'] is not False or rec['record_payload_sha256'] != residuals._payload_hash(rec)):
            raise ValueError('prediction binding changed')
        for name in ('reference','query'):
            image = prep/'verifier/primary'/job[name]
            if sha(image) != job[name+'_sha256'] or rec[name+'_sha256'] != job[name+'_sha256']:
                raise ValueError('image binding changed')
        data = candidate/'artifacts'/f'{ident}.npz'
        meta_path = candidate/'artifacts'/f'{ident}.json'
        meta = read(meta_path)
        bounds = meta['bounds_xyxy']
        x0,y0,x1,y1 = job['box']
        expected_bounds = [max(0,x0-1),max(0,y0-1),min(1200,x1+1),min(675,y1+1)]
        if meta['id'] != ident or meta['binding_sha256'] != manifest_sha or meta['npz_sha256'] != sha(data) or bounds != expected_bounds:
            raise ValueError('geometry binding changed')
        with np.load(data, allow_pickle=False) as content:
            prediction, valid, _ = residuals.sample_map(content['query_xy'], content['valid'], bounds, ref)
        mask &= valid
        maps.append(prediction)
        images.append(np.array(Image.open(prep/'verifier/primary'/job['query']).convert('RGB')))
        details.append({'label':label,'id':ident,'state':state,'record_sha256':sha(rec_path),
                        'geometry_npz_sha256':sha(data),'geometry_metadata_sha256':sha(meta_path),
                        'query_path':str(prep/'verifier/primary'/job['query']), 'query_sha256':job['query_sha256'],
                        'answer':rec['answer'],'physical_state':item['physical_state'],
                        'target_valid_fraction':rec['diagnostics']['registration']['target_valid_fraction'],
                        'match_score':rec['evidence']['match_score']})
    if jobs[CASES[0][1]]['box'] != jobs[CASES[1][1]]['box'] or jobs[CASES[0][1]]['reference_sha256'] != jobs[CASES[1][1]]['reference_sha256']:
        raise ValueError('reference inputs differ')
    box = jobs[CASES[0][1]]['box']
    mask &= np.isfinite(ref).all(axis=1) & np.isfinite(query).all(axis=1)
    mask &= (ref[:,0]>=box[0]) & (ref[:,0]<box[2]) & (ref[:,1]>=box[1]) & (ref[:,1]<box[3])
    eligible = np.flatnonzero(mask)
    if len(eligible) < 3: raise ValueError('insufficient common target vertices')
    # Fixed reference-box anchors select points without inspecting their errors.
    anchors = [(0.25,0.20),(0.75,0.50),(0.35,0.80)]
    chosen = []
    for fx,fy in anchors:
        anchor = [box[0]+fx*(box[2]-box[0]), box[1]+fy*(box[3]-box[1])]
        candidates = np.array([idx for idx in eligible if idx not in chosen])
        chosen.append(int(candidates[np.argmin(np.sum((ref[candidates]-anchor)**2, axis=1))]))
    reference_path = prep/'verifier/primary'/jobs[CASES[0][1]]['reference']
    reference_image = np.array(Image.open(reference_path).convert('RGB'))
    fig, axes = plt.subplots(1,3,figsize=(12.2,4.5),layout='constrained')
    for ax, image in zip(axes, [reference_image,*images]):
        ax.imshow(image, interpolation='nearest')
        ax.set_xlim(550,700); ax.set_ylim(530,435)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_color('#cbd5e1')
    axes[0].set_title('Reference  •  delivered box', fontsize=11, weight='bold')
    axes[0].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1], fill=False, edgecolor='#ffcf40', linewidth=2))
    for number, idx in enumerate(chosen, 1):
        axes[0].scatter(*ref[idx], s=34, facecolor='#63e6be', edgecolor='black', linewidth=.6, zorder=4)
        axes[0].annotate(str(number), ref[idx]+[1.6,-2], color='white', fontsize=8, weight='bold',
                         bbox={'facecolor':'#0f172a','edgecolor':'none','pad':.5,'alpha':.8})
    for ax, (label,_,_), pred, item in zip(axes[1:], CASES, maps, details):
        ax.set_title(label + '  •  target present', fontsize=11, weight='bold')
        for number, idx in enumerate(chosen, 1):
            ax.plot([pred[idx,0],query[idx,0]],[pred[idx,1],query[idx,1]], color='white', linewidth=1.5, zorder=3)
            ax.scatter(*query[idx], s=38, facecolor='#63e6be', edgecolor='black', linewidth=.6, zorder=4)
            ax.scatter(*pred[idx], s=58, marker='x', c='#f15b8a', linewidths=2, zorder=5)
            ax.annotate(str(number), query[idx]+[1.6,-2], color='white', fontsize=8, weight='bold',
                        bbox={'facecolor':'#0f172a','edgecolor':'none','pad':.5,'alpha':.8})
        ax.set_xlabel(f"Model answered REMOVED   ·   match {item['match_score']:.3f}\n"
                      f"{item['target_valid_fraction']:.1%} of target-box pixels usable", fontsize=9, labelpad=8)
    fig.suptitle('The verifier declared this foam brick removed twice', fontsize=14, weight='bold')
    handles = [Line2D([],[],marker='o',color='none',markerfacecolor='#63e6be',markeredgecolor='black',markersize=7,label='Projected target vertex (evaluator only)'),
               Line2D([],[],marker='x',color='#f15b8a',linestyle='none',markeredgewidth=2,markersize=7,label='Geometry adapter sample')]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=9,bbox_to_anchor=(.5,-.035))
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT/'false-removals.png'
    fig.savefig(png,dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    caption = {'caption':'The foam brick remains in both queries. The geometry verifier answered REMOVED twice. The green points show target vertices projected for evaluation, including potentially hidden surfaces. The pink crosses show where the verifier mapped those same reference points. The blocker partly covers the brick in the third panel. Match and usable-box values come from the saved decisions.',
               'selection':'Three target vertices nearest fixed reference-box anchors (25%,20%), (75%,50%), (35%,80%), restricted to samples valid in both candidate maps. Anchors were fixed without selecting by residual.',
               'target_id':arr['target_id'],'target_ycb':target['ycb'],'case_scope':'Both blocker states belong to the same MuJoCo shelf arrangement, stop P4, D4.',
               'box_xyxy':box,'crop_xyxy':[550,435,700,530], 'chosen_vertex_indices':chosen,
               'points':[{'vertex_index':idx,'reference_xy':ref[idx].tolist(),'projected_query_xy':query[idx].tolist(),
                          'clear_adapter_xy':maps[0][idx].tolist(),'placed_adapter_xy':maps[1][idx].tolist()} for idx in chosen],
               'reference_path':str(reference_path),'reference_sha256':jobs[CASES[0][1]]['reference_sha256'],
               'cohort_manifest_sha256':sha(cohort_manifest),'arrangement_sha256':sha(arrangement),
               'target_ply_sha256':sha(ply),'ledger_sha256':sha(ledger_path),'jobs_sha256':sha(jobs_path),
               'candidate_manifest_sha256':manifest_sha,'cases':details,'plot_source_sha256':sha(__file__),'figure_sha256':sha(png)}
    (OUT/'false-removals.json').write_text(json.dumps(caption,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'figure':str(png.resolve()),'figure_sha256':sha(png),'vertices':chosen},indent=2))


if __name__ == '__main__': main()
