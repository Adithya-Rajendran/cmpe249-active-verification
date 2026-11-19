"""Evaluator-only image correspondence and local cue measurements.

The returned measurements do not decide whether a visual judgment is supported.
`row` contains opaque id, engine, base_id, level, stop, state, condition,
prepared_reference, prepared_query, their SHA-256 fields, and primary box_xyxy.
`votes` maps `terra` and `main` to records containing answer, point, target,
anchor and cue. Engine roots contain the accepted native capture directories.
"""
from __future__ import annotations
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image

NATIVE=(2304,1296)
PRIMARY=(1200,675)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rgb_primary(path):
    with Image.open(path) as image:
        if image.size!=NATIVE:raise ValueError(f'native RGB dimensions: {path}')
        rgb=image.convert('RGB').resize(PRIMARY,Image.Resampling.LANCZOS)
        return np.asarray(rgb).copy()


def mask_native(path):
    with Image.open(path) as image:
        if image.size!=NATIVE:raise ValueError(f'native mask dimensions: {path}')
        arr=np.asarray(image)
    if arr.ndim==3:
        if arr.shape[2] not in (3,4):raise ValueError('unexpected mask channels')
        arr=np.any(arr[:,:,:3]!=0,axis=2)
    elif arr.ndim==2:arr=arr!=0
    else:raise ValueError('unexpected mask dimensions')
    return arr


def mask_primary(native):
    image=Image.fromarray(native.astype(np.uint8)*255)
    return np.asarray(image.resize(PRIMARY,Image.Resampling.NEAREST))>0


def _prepared(path,wanted):
    path=Path(path)
    if not path.is_absolute() or sha(path)!=wanted:raise ValueError('prepared image binding mismatch')
    with Image.open(path) as image:
        if image.mode!='RGB' or image.size!=PRIMARY or image.info:raise ValueError('prepared image format or metadata mismatch')
        return np.asarray(image).copy()


def _bbox(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError('empty nominal target mask')
    return [int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1]


def _evidence_for_point(point,mask,present,removed,clear,repeat):
    result={'point':point,'patch_size':[7,7],'required_mask_pixels':25,'mask_pixels':None,
            'present_removed_mean_abs_channel_delta':None,'clear_repeat_mean_abs_channel_delta':None,
            'mask_majority':False,'intervention_change_ge_8':False,'intervention_minus_repeat_ge_4':False}
    if point is None:result['status']='missing_point';return result
    if not isinstance(point,(list,tuple)) or len(point)!=2 or any(type(v) is not int for v in point):result['status']='invalid_point';return result
    x,y=point;h,w=mask.shape
    if not (3<=x<w-3 and 3<=y<h-3):result['status']='patch_outside_image';return result
    sl=(slice(y-3,y+4),slice(x-3,x+4));inside=mask[sl];count=int(inside.sum());result['mask_pixels']=count
    result['mask_majority']=count>=25
    if not count:result['status']='empty_masked_patch';return result
    delta=np.abs(present[sl].astype(np.int16)-removed[sl].astype(np.int16))[inside]
    noise=np.abs(clear[sl].astype(np.int16)-repeat[sl].astype(np.int16))[inside]
    d,n=float(delta.mean()),float(noise.mean());result.update(status='measured',present_removed_mean_abs_channel_delta=d,
        clear_repeat_mean_abs_channel_delta=n,intervention_change_ge_8=d>=8,intervention_minus_repeat_ge_4=d-n>=4)
    return result


def case_evidence(row,votes,engine_roots):
    """Return evaluator-only certificates and per-reader patch measurements.

    Definite-answer support and inter-reader agreement remain a separate review.
    Query points are integer pixel indices; out-of-image patches are not scored.
    """
    if set(votes)!={'terra','main'}:raise ValueError('both blinded reviewers required')
    if row['engine'] not in engine_roots or row['state'] not in ('clear_present','clear_removed','placed_present','placed_removed'):raise ValueError('unknown engine or state')
    if row['condition'] not in ('partial','clear','hidden') or (row['condition']=='clear')!=row['state'].startswith('clear_'):raise ValueError('condition/state mismatch')
    root=Path(engine_roots[row['engine']]).resolve();base=row['base_id'];level=row['level'];stop=row['stop']
    if not base.isalnum() or level not in ('D0','D6') or stop not in ('P1','P2','P3','P4','P5'):raise ValueError('invalid capture identity')
    nominal=root/base/'D0';folder=root/base/level;phase='clear' if row['state'].startswith('clear_') else 'placed'
    files={'reference_rgb':nominal/'rgb'/f'reference_{stop}.png','reference_mask':nominal/'masks'/f'target_reference_{stop}.png',
           'present_rgb':folder/'rgb'/f'{phase}_present_{stop}.png','removed_rgb':folder/'rgb'/f'{phase}_removed_{stop}.png',
           'present_mask':folder/'masks'/f'target_{phase}_present_{stop}.png','removed_mask':folder/'masks'/f'target_{phase}_removed_{stop}.png',
           'blocker_present_mask':folder/'masks'/f'blocker_{phase}_present_{stop}.png','blocker_removed_mask':folder/'masks'/f'blocker_{phase}_removed_{stop}.png',
           'clear_rgb':folder/'rgb'/f'clear_present_{stop}.png','clear_mask':folder/'masks'/f'target_clear_present_{stop}.png',
           'repeat_rgb':folder/'rgb'/f'repeat_clear_present_{stop}.png','repeat_mask':folder/'masks'/f'target_repeat_clear_present_{stop}.png',
           'alone_mask':folder/'masks'/f'alone_{stop}.png'}
    for path in files.values():
        if root not in path.resolve().parents or not path.is_file():raise ValueError(f'capture file missing: {path}')
    reference=rgb_primary(files['reference_rgb']);present=rgb_primary(files['present_rgb']);removed=rgb_primary(files['removed_rgb'])
    clear=rgb_primary(files['clear_rgb']);repeat=rgb_primary(files['repeat_rgb'])
    target_native=mask_native(files['present_mask']);mask=mask_primary(target_native)
    removed_native=mask_native(files['removed_mask']);blocker_present=mask_native(files['blocker_present_mask']);blocker_removed=mask_native(files['blocker_removed_mask'])
    reference_mask=mask_native(files['reference_mask']);clear_mask=mask_native(files['clear_mask']);repeat_mask=mask_native(files['repeat_mask']);alone_mask=mask_native(files['alone_mask'])
    prepared_ref=_prepared(row['prepared_reference'],row['reference_sha256']);prepared_query=_prepared(row['prepared_query'],row['query_sha256'])
    query=present if row['state'].endswith('_present') else removed
    if not np.array_equal(reference,prepared_ref) or not np.array_equal(query,prepared_query):raise ValueError('prepared image does not match native source')
    native_box=_bbox(reference_mask);expected_box=[round(x*PRIMARY[j%2]/NATIVE[j%2]) for j,x in enumerate(native_box)]
    if row['box_xyxy']!=expected_box:raise ValueError('prepared reference box does not match source target mask')
    same_query=bool(np.array_equal(present,removed))
    pixel_digest=lambda image:hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    result={'id':row['id'],'exact_opposite_input_certificate':same_query,'same_reference_and_box':True,'same_primary_query_pixels':same_query,
            'primary_pixel_sha256':{'reference':pixel_digest(reference),'present':pixel_digest(present),'removed':pixel_digest(removed)},
            'source_sha256':{name:sha(path) for name,path in files.items()},'prepared_reference_sha256':sha(row['prepared_reference']),
            'prepared_query_sha256':sha(row['prepared_query']),'reference_box_native_xyxy':native_box,'reference_box_primary_xyxy':expected_box,
            'mask_counts_native':{'present_target':int(target_native.sum()),'removed_target':int(removed_native.sum()),'clear_target':int(clear_mask.sum()),
                                  'repeat_target':int(repeat_mask.sum()),'alone_target':int(alone_mask.sum()),'present_blocker':int(blocker_present.sum()),'removed_blocker':int(blocker_removed.sum()),
                                  'blocker_mask_equal':bool(np.array_equal(blocker_present,blocker_removed))},
            'reviewer_points':{}}
    for reviewer,vote in votes.items():
        if not isinstance(vote,dict) or any(k not in vote for k in ('answer','point','target','anchor','cue')):raise ValueError('reviewer vote fields missing')
        result['reviewer_points'][reviewer]=_evidence_for_point(vote['point'],mask,present,removed,clear,repeat)
    return result
