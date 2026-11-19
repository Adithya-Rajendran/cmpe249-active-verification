"""Evaluator-only, matched target-region reprojection diagnostics.

This script consumes completed prediction records only after scoring. It never
passes geometry, truth, or outcome labels to a predictor. Removed-object
projections are virtual expected locations, not observed correspondences.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from development.cross_sim import landmark_residuals as prior
from development.cross_sim import qualification_geometry as geometry


def project_homography(raw_query_to_ref, reference):
    """Invert a raw query-to-reference homography and project reference points."""
    points = np.asarray(reference, dtype=np.float64)
    result = np.full_like(points, np.nan)
    available = np.zeros(len(points), dtype=bool)
    if raw_query_to_ref is None:
        return result, available, 'missing_matrix'
    h = np.asarray(raw_query_to_ref, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        return result, available, 'invalid_matrix'
    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return result, available, 'invalid_matrix'
    projected = np.column_stack((points, np.ones(len(points)))) @ inverse.T
    available = np.isfinite(projected).all(axis=1) & (np.abs(projected[:, 2]) > 1e-12)
    result[available] = projected[available, :2] / projected[available, 2, None]
    available &= np.isfinite(result).all(axis=1)
    return result, available, 'available' if available.any() else 'invalid_projection'


def sample_map(query_xy, valid, bounds, reference):
    """Bilinearly sample a cropped map only when each contributing pixel is usable."""
    xy, valid = np.asarray(query_xy), np.asarray(valid)
    ref = np.asarray(reference, dtype=np.float64)
    x0, y0, x1, y1 = bounds
    if (xy.shape != (y1-y0, x1-x0, 2) or xy.dtype != np.float32
            or valid.shape != xy.shape[:2] or valid.dtype != bool):
        raise ValueError('invalid geometry crop arrays')
    if ref.ndim != 2 or ref.shape[1] != 2:
        raise ValueError('reference must be Nx2')
    result = np.full_like(ref, np.nan)
    finite = np.isfinite(ref).all(axis=1)
    x = np.where(finite, ref[:, 0] - x0, 0)
    y = np.where(finite, ref[:, 1] - y0, 0)
    h, w = valid.shape
    inside = finite & (x >= 0) & (x <= w-1) & (y >= 0) & (y <= h-1)
    x, y = np.clip(x, 0, w-1), np.clip(y, 0, h-1)
    left, top = np.floor(x).astype(int), np.floor(y).astype(int)
    right, bottom = np.minimum(left+1, w-1), np.minimum(top+1, h-1)
    dx, dy = x-left, y-top
    weights = ((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy)
    indexes = ((top,left), (top,right), (bottom,left), (bottom,right))
    usable = inside.copy()
    values = np.zeros((len(ref), 2), dtype=np.float64)
    for weight, (iy, ix) in zip(weights, indexes):
        point = xy[iy, ix]
        point_ok = valid[iy, ix] & np.isfinite(point).all(axis=1)
        usable &= (weight == 0) | point_ok
        values += weight[:, None] * np.where(point_ok[:, None], point, 0)
    result[usable] = values[usable]
    return result, usable, inside


def error_stats(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if not len(errors):
        return {'count': 0, 'median_px': None, 'p90_px': None}
    if not np.isfinite(errors).all():
        raise ValueError('nonfinite errors')
    return {'count': len(errors), 'median_px': float(np.median(errors)),
            'p90_px': float(np.percentile(errors, 90))}


def compare_points(reference, query, box, raw_h, xy, valid, bounds):
    reference, query = np.asarray(reference, float), np.asarray(query, float)
    if reference.shape != query.shape or reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError('expected matching Nx2 source vertices')
    finite = np.isfinite(reference).all(axis=1) & np.isfinite(query).all(axis=1)
    x0, y0, x1, y1 = box
    region = finite & (reference[:, 0] >= x0) & (reference[:, 0] < x1)
    region &= (reference[:, 1] >= y0) & (reference[:, 1] < y1)
    ref, truth = reference[region], query[region]
    base, base_ok, status = project_homography(raw_h, ref)
    candidate, cand_ok, inside = sample_map(xy, valid, bounds, ref)
    common = base_ok & cand_ok
    be = np.linalg.norm(base[base_ok] - truth[base_ok], axis=1)
    ce = np.linalg.norm(candidate[cand_ok] - truth[cand_ok], axis=1)
    b_common = np.linalg.norm(base[common] - truth[common], axis=1)
    c_common = np.linalg.norm(candidate[common] - truth[common], axis=1)
    total = len(ref)
    return {'source_vertices': len(reference), 'nonfinite_vertices': int((~finite).sum()),
            'reference_box_vertices': total, 'baseline_status': status,
            'baseline_available': int(base_ok.sum()), 'candidate_available': int(cand_ok.sum()),
            'candidate_missing_or_occluded': int((inside & ~cand_ok).sum()),
            'candidate_outside_crop': int((~inside).sum()), 'common_vertices': int(common.sum()),
            'candidate_evaluated_fraction': float(cand_ok.mean()) if total else None,
            'baseline_evaluated_fraction': float(base_ok.mean()) if total else None,
            'common_fraction': float(common.mean()) if total else None,
            'baseline': error_stats(be), 'candidate': error_stats(ce),
            'baseline_common': error_stats(b_common), 'candidate_common': error_stats(c_common)}


def sha(path):
    return prior.sha(path)


def load(path):
    return json.loads(Path(path).read_text())


def _payload_hash(record):
    data = {k: v for k, v in record.items() if k != 'record_payload_sha256'}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def summarize(records):
    axes = [(), ('engine',), ('environment',), ('level',), ('state',),
            ('candidate_outcome',), ('baseline_outcome',),
            ('engine', 'environment', 'level', 'state', 'candidate_outcome'),
            ('engine', 'environment', 'level', 'state', 'baseline_outcome')]
    output = []
    for axis in axes:
        groups = defaultdict(list)
        for record in records:
            groups[tuple(record[k] for k in axis)].append(record)
        for key, rows in sorted(groups.items()):
            total = sum(row['geometry']['reference_box_vertices'] for row in rows)
            counts = {name: sum(row['geometry'][name] for row in rows) for name in
                      ('baseline_available', 'candidate_available', 'candidate_missing_or_occluded',
                       'candidate_outside_crop', 'common_vertices')}
            metrics = {}
            for name in ('baseline', 'candidate', 'baseline_common', 'candidate_common'):
                values = [row['geometry'][name]['median_px'] for row in rows if row['geometry'][name]['count']]
                metrics[name + '_case_median_px'] = float(np.median(values)) if values else None
            output.append({'group_by': list(axis), 'group': dict(zip(axis, key)), 'cases': len(rows),
                           'reference_box_vertices': total,
                           'baseline_matrix_missing_cases': sum(row['geometry']['baseline_status'] == 'missing_matrix' for row in rows),
                           'baseline_matrix_invalid_cases': sum(row['geometry']['baseline_status'] == 'invalid_matrix' for row in rows),
                           'baseline_projection_invalid_cases': sum(row['geometry']['baseline_status'] == 'invalid_projection' for row in rows),
                           'no_reference_box_vertices_cases': sum(row['geometry']['reference_box_vertices'] == 0 for row in rows),
                           **counts,
                           'candidate_evaluated_fraction': counts['candidate_available']/total if total else None,
                           'common_fraction': counts['common_vertices']/total if total else None, **metrics})
    return output


def run(analysis_path, cohort_path, prepared_dir, baseline_dir, candidate_dir, assets_dir, weight_records, output_dir):
    analysis_path, cohort_path, prepared_dir, baseline_dir, candidate_dir, assets_dir, output_dir = map(
        lambda p: Path(p).resolve(), (analysis_path, cohort_path, prepared_dir, baseline_dir, candidate_dir, assets_dir, output_dir))
    weight_records = [Path(p).resolve() for p in weight_records]
    if not weight_records:
        raise ValueError('at least one weight verification record is required')
    analysis, cohort = load(analysis_path), load(cohort_path)
    prep_path, ledger_path = prepared_dir/'preparation.json', prepared_dir/'ledger.json'
    prep, ledger = load(prep_path), load(ledger_path)
    jobs_path = prepared_dir/'verifier/primary/jobs.json'
    jobs_list = load(jobs_path)
    jobs, ledger_by_id = {r['id']: r for r in jobs_list}, {r['id']: r for r in ledger}
    if (cohort.get('split') != 'pilot' or analysis.get('study_id') != cohort.get('study_id')
            or sha(prep_path) != analysis.get('preparation_sha256') or sha(ledger_path) != prep.get('ledger_sha256')
            or sha(cohort_path) != prep.get('cohort_sha256') or sha(jobs_path) != prep['jobs_sha256']['primary']
            or len(jobs) != 1200 or len(jobs_list) != 1200 or len(ledger_by_id) != 1200 or len(ledger) != 1200
            or set(jobs) != set(ledger_by_id)):
        raise ValueError('cohort, analysis, preparation, jobs or ledger mismatch')
    old_manifest, cand_manifest = baseline_dir/'manifest.json', candidate_dir/'manifest.json'
    old, cand = load(old_manifest), load(cand_manifest)
    if (sha(old_manifest) != analysis['control_manifest_sha256']['relaxed_guard'] or old['jobs_sha256'] != sha(jobs_path)
            or cand['jobs_sha256'] != sha(jobs_path) or cand['reference_manifest_sha256'] != sha(old_manifest)
            or cand['config']['method'] != 'geometry'):
        raise ValueError('baseline or candidate binding mismatch')
    checkpoint = cand['provider']['provenance']['checkpoint_sha256']
    model_revision = cand['provider']['provenance']['model_revision']
    if not any(load(path).get('sha256') == checkpoint and load(path).get('revision') == model_revision
               for path in weight_records):
        raise ValueError('no weight verification record matches candidate provenance')
    rows_list = analysis['rows']['relaxed_guard']
    rows = {r['id']: r for r in rows_list}
    if len(rows) != 1200 or len(rows_list) != 1200 or set(rows) != set(jobs):
        raise ValueError('baseline analysis rows incomplete')
    for folder, suffix in ((baseline_dir/'records', '.json'), (candidate_dir/'records', '.json'),
                           (candidate_dir/'artifacts', '.json'), (candidate_dir/'artifacts', '.npz')):
        if {p.stem for p in folder.glob('*'+suffix)} != set(jobs):
            raise ValueError(f'incomplete output inventory: {folder} {suffix}')
    vertices, cameras, asset_hashes, arrangement_hashes = {}, {}, {}, {}
    for item in cohort['arrangements']:
        base = item['base_id']
        for level, entry in item['levels'].items():
            path = (cohort_path.parent/entry['path']).resolve()
            if path.parent != cohort_path.parent or sha(path) != entry['sha256']:
                raise ValueError('arrangement hash mismatch')
            arr = load(path)
            arrangement_hashes[str(path)] = entry['sha256']
            if base not in vertices:
                target = arr['objects']['target']
                if target['kind'] == 'ycbv':
                    source = assets_dir/target['ply']
                    asset_hashes[str(source)] = sha(source)
                    vertices[base] = geometry.object_mesh(target, assets_dir)['vertices']
                else:
                    vertices[base] = geometry.source_geometry(arr, assets_dir)['target']['vertices']
            for stop in cohort['stops']:
                ref, _ = geometry.project(vertices[base], geometry.camera_record(arr, stop))
                query, _ = geometry.project(vertices[base], geometry.camera_record(arr, stop, revisit=True))
                cameras[(base, level, stop)] = (prior.sample_index(ref, (1200,675)), prior.sample_index(query, (1200,675)))
    records = []
    for ident in sorted(jobs):
        item, row, job = ledger_by_id[ident], rows[ident], jobs[ident]
        for key in ('engine','environment','base_id','level','stop','state','physical_state'):
            if row[key] != item[key]: raise ValueError('analysis row/ledger mismatch')
        old_path, new_path = baseline_dir/'records'/f'{ident}.json', candidate_dir/'records'/f'{ident}.json'
        a, b = load(old_path), load(new_path)
        if (a.get('id') != ident or a.get('binding_sha256') != sha(old_manifest) or a.get('failure') is not False
                or b.get('id') != ident or b.get('binding_sha256') != sha(cand_manifest) or b.get('failure') is not False
                or b.get('record_payload_sha256') != _payload_hash(b) or a.get('answer', row['answer']) != row['answer']):
            raise ValueError('prediction record mismatch')
        meta_path = candidate_dir/'artifacts'/f'{ident}.json'
        npz_path = candidate_dir/'artifacts'/f'{ident}.npz'
        meta = load(meta_path)
        x0,y0,x1,y1 = job['box']
        expected_bounds = [max(0,x0-1), max(0,y0-1), min(1200,x1+1), min(675,y1+1)]
        if (meta.get('id') != ident or meta.get('binding_sha256') != sha(cand_manifest)
                or meta.get('npz_sha256') != sha(npz_path) or meta.get('bounds_xyxy') != expected_bounds
                or meta.get('query_xy_dtype') != 'float32' or meta.get('valid_dtype') != 'bool'):
            raise ValueError('candidate geometry artifact mismatch')
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {'query_xy','valid'}: raise ValueError('unexpected geometry arrays')
            xy, mask = archive['query_xy'], archive['valid']
        ref, query = cameras[(item['base_id'], item['level'], item['stop'])]
        fit = compare_points(ref, query, job['box'], a.get('raw_H_query_to_reference'), xy, mask, expected_bounds)
        outcome = lambda answer: 'unknown' if answer == 'UNKNOWN' else ('correct' if answer == item['physical_state'] else 'incorrect')
        if b['answer'] not in ('UNKNOWN','PRESENT','REMOVED'): raise ValueError('unknown candidate answer')
        records.append({**{k: item[k] for k in ('id','engine','environment','base_id','level','stop','state','physical_state')},
                        'baseline_outcome': outcome(row['answer']), 'candidate_outcome': outcome(b['answer']),
                        'reference_type': 'virtual_expected_location' if item['physical_state'] == 'REMOVED' else 'projected_target_vertices',
                        'baseline_record_sha256': sha(old_path), 'candidate_record_sha256': sha(new_path),
                        'artifact_sha256': sha(npz_path), 'artifact_metadata_sha256': sha(meta_path), 'geometry': fit})
    result = {'purpose':'post-scoring evaluator-only diagnostic; no predictor input or changed decisions',
              'direction':'reference to query for both methods; inverse of raw baseline query-to-reference H',
              'target_vertices':'all original source vertices whose nominal projection lies in the reference box; not area weighted',
              'removed_note':'removed target vertices are virtual expected locations, not ground-truth observed correspondences',
              'candidate_missing_note':'missing includes predicted occlusion, invalid projection and other unusable samples; they are not zero error',
              'analysis_sha256':sha(analysis_path), 'cohort_sha256':sha(cohort_path),
              'preparation_sha256':sha(prep_path), 'ledger_sha256':sha(ledger_path), 'jobs_sha256':sha(jobs_path),
              'baseline_manifest_sha256':sha(old_manifest), 'candidate_manifest_sha256':sha(cand_manifest),
              'weight_record_sha256':{str(p):sha(p) for p in weight_records}, 'assets_sha256':asset_hashes,
              'arrangement_sha256':arrangement_hashes, 'source_sha256':sha(__file__),
              'geometry_source_sha256':sha(geometry.__file__), 'prior_source_sha256':sha(prior.__file__),
              'case_count':len(records), 'summary':summarize(records), 'records':records}
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir/'geometry_residuals.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (output_dir/'output_sha256.txt').write_text(sha(output)+'  geometry_residuals.json\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ('analysis','cohort','prepared-dir','baseline-dir','candidate-dir','assets-dir','output-dir'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--weight-record', action='append', required=True)
    args = parser.parse_args()
    result = run(args.analysis, args.cohort, args.prepared_dir, args.baseline_dir, args.candidate_dir,
                 args.assets_dir, args.weight_record, args.output_dir)
    print(json.dumps({'case_count': result['case_count'], 'output': str(Path(args.output_dir)/'geometry_residuals.json')}))


if __name__ == '__main__':
    main()
