"""Validate completed stress renders and prepare blinded, metadata-free inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.arrange import arrangement_hash, check_dev_seed
from development.stress.alignment_diagnostics import check_frozen

PROTOCOL = ROOT / 'development/stress/protocol.json'
STUDY = 'support-stress-dev-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def case_id(record, stop, phase):
    """Stable opaque identifier for the predeclared case, independent of cohort subset."""
    case = [STUDY, record['environment'], record['base_id'], record['level_id'], stop, phase]
    return 'I' + hashlib.sha256(json.dumps(case, separators=(',', ':')).encode()).hexdigest()[:24]


def clean_image(source, destination, size):
    with Image.open(source) as im:
        if im.size != (2304, 1296):
            raise ValueError(f'unexpected source dimensions: {source}')
        clean = Image.new('RGB', im.size)
        clean.paste(im.convert('RGB'))
    if clean.size != tuple(size):
        clean = clean.resize(tuple(size), Image.LANCZOS)
    clean.save(destination, format='PNG')


def validate_render(record, root, protocol, frozen):
    folder = root / record['render_directory']
    if folder.with_name(folder.name + '.partial').exists():
        raise ValueError(f'partial render exists: {folder}')
    source = root / record['arrangement_file']
    arr = load(source)
    copied = folder / 'arrangement.json'
    lab, render = load(folder / 'labels.json'), load(folder / 'render.json')
    if sha(source) != sha(copied):
        raise ValueError(f'arrangement file differs: {folder}')
    if arr.get('study_id') != STUDY or arr.get('stress_provenance', {}).get('protocol_sha256') != sha(PROTOCOL):
        raise ValueError('study or protocol mismatch')
    if arr.get('arrangement_sha256') != arrangement_hash(arr):
        raise ValueError('arrangement content hash mismatch')
    for key, expected in [('arrangement_id', record['arrangement_id']), ('base_id', record['base_id']),
                          ('level', record['level_id']), ('seed', record['seed'])]:
        if arr.get(key) != expected:
            raise ValueError(f'arrangement {key} mismatch')
    check_dev_seed(record['seed'])
    env = record['environment']
    start = protocol['seed_starts'].get(env)
    if start is None or not start <= record['seed'] < start + 12 * protocol['seeds_per_slot']:
        raise ValueError('seed outside declared environment range')
    hashes = lab['hashes']
    if lab['arrangement_id'] != record['arrangement_id'] or render['arrangement_id'] != record['arrangement_id']:
        raise ValueError('render identity mismatch')
    if hashes['arrangement_file_sha256'] != sha(source) or hashes['arrangement_sha256'] != arr['arrangement_sha256']:
        raise ValueError('render arrangement provenance mismatch')
    for name in ('isaac_batch', 'isaac_rig'):
        key = name + '_sha256'
        if hashes[key] != frozen['sim/' + name + '.py'] or render[key] != hashes[key]:
            raise ValueError('renderer source hash mismatch')
    if render['settings_sha256'] != json_hash(render['settings']) or hashes['render_settings_sha256'] != render['settings_sha256']:
        raise ValueError('render settings hash mismatch')
    settings = render['settings']
    defaults = {'renderer': 'RaytracedLighting', 'rt_subframes': 16, 'settle_updates': 200,
                'capture_updates': 100, 'alone_updates': 10}
    if any(settings.get(k) != v for k, v in defaults.items()):
        raise ValueError('renderer defaults mismatch')
    if settings['lightings'] != ['L1'] or len(settings['resolution']) != len(protocol['stops']) or any(x != protocol['render_size'] for x in settings['resolution']):
        raise ValueError('render settings do not match protocol')
    if lab['stops'] != protocol['stops'] or lab['lightings'] != ['L1'] or not lab['label_sanity']['ok']:
        raise ValueError('incomplete or invalid render labels')
    expected = {f'{phase}_{stop}_L1.png' for phase in ['reference'] + protocol['query_phases'] for stop in protocol['stops']}
    actual = {p.name for p in (folder / 'rgb').glob('*.png')}
    if actual != expected:
        raise ValueError(f'RGB source count or names mismatch: {folder}')
    for name in expected:
        with Image.open(folder / 'rgb' / name) as im:
            if list(im.size) != protocol['render_size']:
                raise ValueError(f'bad RGB dimensions: {name}')
            phase, stop, lighting = name[:-4].split('_')
            meta = json.loads(im.info['cmpe249'])
            camera = 'nominal' if phase == 'reference' else 'revisit'
            delta = None if camera == 'nominal' else arr['revisit']['stops'][stop]['delta']
            expected_meta = {'arrangement_id': arr['arrangement_id'], 'seed': arr['seed'], 'phase': phase,
                             'stop': stop, 'lighting': lighting, 'camera': camera, 'revisit_delta': delta,
                             'arrangement_sha256': arr['arrangement_sha256']}
            if any(meta.get(k) != v for k, v in expected_meta.items()):
                raise ValueError(f'RGB metadata mismatch: {name}')
    return arr, lab, folder


def prepare(manifest_path, output, scope="full"):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    protocol, manifest = load(PROTOCOL), load(manifest_path)
    if protocol['study_id'] != STUDY or manifest['study_id'] != STUDY:
        raise ValueError('unknown study')
    frozen = check_frozen()
    records = manifest['records']
    if not records or len({(r['base_id'], r['level_id']) for r in records}) != len(records):
        raise ValueError('empty or duplicate cohort')
    levels = {x['id'] for x in protocol['levels']}
    if scope == 'smoke':
        first = {}
        for record in records:
            first.setdefault(record['environment'], record['base_id'])
        if set(first) != set(protocol['environments']):
            raise ValueError('smoke requires both environments')
        records = [r for r in records if r['base_id'] == first[r['environment']] and r['level_id'] in {'D0', 'D6'}]
        expected_bases, expected_levels = 2, {'D0', 'D6'}
    elif scope == 'full':
        expected_bases, expected_levels = 24, levels
    else:
        raise ValueError('unknown cohort scope')
    groups = {}
    for r in records:
        groups.setdefault(r['base_id'], []).append(r)
    if len(groups) != expected_bases or any({r['level_id'] for r in g} != expected_levels for g in groups.values()):
        raise ValueError('cohort scope is incomplete')
    if any(sum(g[0]['environment'] == e for g in groups.values()) != expected_bases // 2 for e in protocol['environments']):
        raise ValueError('cohort environment counts mismatch')
    by_base = {}
    checked = {}
    for r in records:
        if r['level_id'] not in levels:
            raise ValueError('unknown level')
        checked[(r['base_id'], r['level_id'])] = validate_render(r, manifest_path.parent, protocol, frozen)
        by_base.setdefault(r['base_id'], []).append(r)
    for base, group in by_base.items():
        if 'D0' not in {r['level_id'] for r in group} or len({(r['seed'], r['environment']) for r in group}) != 1:
            raise ValueError(f'missing D0 or inconsistent base: {base}')
    if output.exists():
        raise FileExistsError(f'output already exists: {output}')
    output.mkdir(parents=True)
    for size in ('primary', 'native'):
        (output / 'verifier' / size).mkdir(parents=True)
    rows = [(r, stop, phase) for r in records for stop in protocol['stops'] for phase in protocol['query_phases']]
    ids = [case_id(r, stop, phase) for r, stop, phase in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('case ID collision')
    ledger, jobs = [], {'primary': [], 'native': []}
    for i, (r, stop, phase) in enumerate(rows):
        ident = ids[i]
        arr, lab, folder = checked[(r['base_id'], r['level_id'])]
        _, ref_lab, ref_folder = checked[(r['base_id'], 'D0')]
        box = ref_lab['reference_box']['L1'][stop]
        if not box or len(box) != 4:
            raise ValueError('missing reference box')
        ref = ref_folder / 'rgb' / f'reference_{stop}_L1.png'
        query = folder / 'rgb' / f'{phase}_{stop}_L1.png'
        entry = {'id': ident, 'base_id': r['base_id'], 'environment': r['environment'], 'level': r['level_id'],
                 'arrangement_id': r['arrangement_id'], 'stop': stop, 'phase': phase,
                 'physical_state': 'REMOVED' if phase == 'removed' else 'PRESENT',
                 'arrangement_file': str((manifest_path.parent / r['arrangement_file']).resolve()),
                 'arrangement_file_sha256': sha(manifest_path.parent / r['arrangement_file']),
                 'source_reference_sha256': sha(ref), 'source_query_sha256': sha(query), 'reference_box_native': box,
                 'rendered_level_reference_sha256': sha(folder / 'rgb' / f'reference_{stop}_L1.png'),
                 'labels_sha256': sha(folder / 'labels.json'), 'render_sha256': sha(folder / 'render.json'),
                 'reference_labels_sha256': sha(ref_folder / 'labels.json'),
                 'visible_fraction': lab['visible_fraction'].get(phase, {}).get(stop),
                 'region_obstruction': lab['region_obstruction'].get(phase, {}).get(stop),
                 'sign_translation': arr['revisit']['stops'][stop]['sign_translation'],
                 'sign_yaw': arr['revisit']['stops'][stop]['sign_yaw']}
        ledger.append(entry)
        for size_name, dimensions in [('primary', protocol['primary_size']), ('native', protocol['render_size'])]:
            dest = output / 'verifier' / size_name
            ref_name, query_name = ident + '_r.png', ident + '_q.png'
            clean_image(ref, dest / ref_name, dimensions)
            clean_image(query, dest / query_name, dimensions)
            scaled = [round(x * dimensions[j % 2] / protocol['render_size'][j % 2]) for j, x in enumerate(box)]
            jobs[size_name].append({'id': ident, 'reference': ref_name, 'query': query_name, 'box': scaled,
                                    'reference_sha256': sha(dest / ref_name), 'query_sha256': sha(dest / query_name)})
    (output / 'ledger.json').write_text(json.dumps(ledger, indent=2) + '\n')
    for size, values in jobs.items():
        (output / 'verifier' / size / 'jobs.json').write_text(json.dumps(values, indent=2) + '\n')
    (output / 'preparation.json').write_text(json.dumps({'study_id': STUDY, 'protocol_sha256': sha(PROTOCOL),
        'cohort_sha256': sha(manifest_path), 'ledger_sha256': sha(output / 'ledger.json'),
        'jobs_sha256': {size: sha(output / 'verifier' / size / 'jobs.json') for size in jobs}, 'scope': scope, 'count': len(rows), 'frozen_hashes': frozen}, indent=2) + '\n')
    annotation_pack(output, jobs['primary'])


def annotation_pack(output, jobs):
    review = output / 'review'
    review.mkdir()
    (review / 'images').mkdir()
    items = []
    for job in sorted(jobs, key=lambda x: x['id']):
        with Image.open(output / 'verifier/primary' / job['reference']) as im:
            canvas = im.copy()
        ImageDraw.Draw(canvas).rectangle(job['box'], outline='red', width=4)
        canvas.save(review / 'images' / job['reference'])
        with Image.open(output / 'verifier/primary' / job['query']) as im:
            im.save(review / 'images' / job['query'])
        items.append({'id': job['id'], 'reference': job['reference'], 'query': job['query'],
                      'reference_sha256': job['reference_sha256'], 'query_sha256': job['query_sha256'],
                      'box_sha256': json_hash(job['box']), 'input_size': '1200x675'})
    with (review / 'blank_labels.csv').open('w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['id', 'label', 'cue', 'reviewer', 'source', 'input_size', 'reference_sha256', 'query_sha256', 'box_sha256'])
        writer.writerows((x['id'], '', '', '', 'human', x['input_size'], x['reference_sha256'], x['query_sha256'], x['box_sha256']) for x in items)
    data = json.dumps(items).replace('</', '<\\/')
    html = '''<!doctype html><meta charset="utf-8"><title>Image review</title>
<style>img{width:48%;height:auto}textarea{width:90%}</style><h1>Image review</h1>
<p>Identify the remembered object in the red reference rectangle using only this image pair. The query is unmarked. PRESENT requires identifying evidence. REMOVED requires a sufficiently exposed remembered region and distinguishing absence evidence. Use UNKNOWN for insufficient evidence and DISPUTED when the judgment is unresolved. Fragments, shadows and reflections count only if they support identification. Record the cue and uncertainty.</p>
<div id="counter"></div><div><img id="ref"><img id="query"></div><p>Reference (left), query (right)</p>
<label>Label <select id="label"><option value=""></option><option>PRESENT</option><option>REMOVED</option><option>UNKNOWN</option><option>DISPUTED</option></select></label>
<p><label>Cue and uncertainty<br><textarea id="cue"></textarea></label></p><p><label>Reviewer <input id="reviewer"></label></p>
<p>Export regularly. To carry completed labels into a later review, import the downloaded CSV.</p><label>Import saved CSV <input type="file" accept=".csv,text/csv" onchange="importCSV(this)"></label><p id="message"></p><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><button onclick="download()">Download CSV</button>
<script>const items=DATA; const storageKey='stress-review-'+KEY; let i=0; let saved=JSON.parse(localStorage.getItem(storageKey)||'{}'); let reviewerName=localStorage.getItem(storageKey+'-reviewer')||''; function save(){reviewerName=reviewer.value;localStorage.setItem(storageKey+'-reviewer',reviewerName);saved[items[i].id]={label:label.value,cue:cue.value,reviewer:reviewerName};localStorage.setItem(storageKey,JSON.stringify(saved));}
function show(){let x=items[i],s=saved[x.id]||{};counter.textContent=`${i+1} / ${items.length}: ${x.id}`;ref.src='images/'+x.reference;query.src='images/'+x.query;label.value=s.label||'';cue.value=s.cue||'';reviewer.value=s.reviewer||reviewerName;}
function move(d){save();i=Math.max(0,Math.min(items.length-1,i+d));show()}function download(){save();let q=x=>'"'+String(x||'').replaceAll('"','""')+'"';let lines=[['id','label','cue','reviewer','source','input_size','reference_sha256','query_sha256','box_sha256'].map(q).join(',')];for(let x of items){let s=saved[x.id]||{};lines.push([x.id,s.label,s.label?s.cue:'',s.label?s.reviewer:'','human',x.input_size,x.reference_sha256,x.query_sha256,x.box_sha256].map(q).join(','))}let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\r\\n')],{type:'text/csv'}));a.download='human_labels.csv';a.click()}
function parseCSV(text){let rows=[],row=[],field='',quoted=false,afterQuote=false;for(let i=0;i<text.length;i++){let c=text[i];if(quoted){if(c==='"'){if(text[i+1]==='"'){field+='"';i++}else{quoted=false;afterQuote=true}}else field+=c;}else if(c==='"'&&field===''&&!afterQuote){quoted=true}else if(c===','){row.push(field);field='';afterQuote=false}else if(c==='\\r'||c==='\\n'){row.push(field);rows.push(row);row=[];field='';afterQuote=false;if(c==='\\r'&&text[i+1]==='\\n')i++}else if(afterQuote){throw Error('Invalid CSV quoting')}else field+=c;}if(quoted)throw Error('Unclosed CSV quote');if(field!==''||row.length){row.push(field);rows.push(row)}return rows;}
async function importCSV(input){try{save();let rows=parseCSV(await input.files[0].text()), header=['id','label','cue','reviewer','source','input_size','reference_sha256','query_sha256','box_sha256'];if(!rows.length||JSON.stringify(rows.shift())!==JSON.stringify(header))throw Error('Unexpected CSV columns');let indexed=Object.fromEntries(items.map(x=>[x.id,x])),pending={},seen=new Set();for(let row of rows){if(row.length!==header.length)throw Error('Invalid row');let r=Object.fromEntries(header.map((k,i)=>[k,row[i]])),x=indexed[r.id];if(!x||seen.has(r.id))throw Error('Unknown or duplicate input ID');seen.add(r.id);for(let key of ['input_size','reference_sha256','query_sha256','box_sha256'])if(r[key]!==x[key])throw Error('Input image or box mismatch');if(r.source!=='human')throw Error('Source must be human');if(!r.label){if(r.cue||r.reviewer)throw Error('Incomplete row');continue}if(!['PRESENT','REMOVED','UNKNOWN','DISPUTED'].includes(r.label)||!r.cue.trim()||!r.reviewer.trim())throw Error('Incomplete human label');let prior=saved[r.id];if(prior&&prior.label&&(prior.label!==r.label||prior.cue!==r.cue||prior.reviewer!==r.reviewer))throw Error('Existing label conflicts with imported label');pending[r.id]={label:r.label,cue:r.cue,reviewer:r.reviewer}}Object.assign(saved,pending);localStorage.setItem(storageKey,JSON.stringify(saved));message.textContent='Imported '+Object.keys(pending).length+' labels.';show()}catch(e){message.textContent='Import refused: '+e.message}input.value=''}show();</script>'''
    (review / 'index.html').write_text(html.replace('DATA', data).replace('KEY', json.dumps(sha(output / 'preparation.json'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scope', choices=['full', 'smoke'], default='full')
    args = parser.parse_args()
    prepare(args.cohort, args.out, args.scope)
