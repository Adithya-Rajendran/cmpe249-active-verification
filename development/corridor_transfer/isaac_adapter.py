"""Isolated Isaac capture primitives for cross-simulator development.

This module does not launch a study. A reviewed driver must supply the inputs,
settings and output contract before calling capture_arrangement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim import isaac_batch as batch
from sim import isaac_rig as rig

STATES = {
    'reference': (True, 'parked', 'nominal'),
    'clear_present': (True, 'parked', 'revisit'),
    'clear_removed': (False, 'parked', 'revisit'),
    'placed_present': (True, 'placed', 'revisit'),
    'placed_removed': (False, 'placed', 'revisit'),
}


def observe_stage(stage, built, include_geometry=True):
    """Read transforms and geometry from USD after the intervention is applied."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    records = {}
    for prim in stage.TraverseAll():
        if not prim.IsA(UsdGeom.Xformable):
            continue
        path = str(prim.GetPath())
        if not (path.startswith('/World/Objects') or path.startswith('/World/Shelf')
                or path.startswith('/World/Cameras')):
            continue
        xform = cache.GetLocalToWorldTransform(prim)
        item = {'world_from_local_row': [[float(xform[i][j]) for j in range(4)] for i in range(4)],
                'type': prim.GetTypeName(),
                'visibility': str(UsdGeom.Imageable(prim).ComputeVisibility())}
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            item.update(focal_length=float(cam.GetFocalLengthAttr().Get()),
                        horizontal_aperture=float(cam.GetHorizontalApertureAttr().Get()),
                        vertical_aperture=float(cam.GetVerticalApertureAttr().Get()),
                        horizontal_aperture_offset=float(cam.GetHorizontalApertureOffsetAttr().Get()),
                        vertical_aperture_offset=float(cam.GetVerticalApertureOffsetAttr().Get()),
                        clipping_range=[float(v) for v in cam.GetClippingRangeAttr().Get()])
        if include_geometry and prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            item['points_local'] = [[float(c) for c in p] for p in mesh.GetPointsAttr().Get()]
            item['face_vertex_counts'] = list(mesh.GetFaceVertexCountsAttr().Get())
            item['face_vertex_indices'] = list(mesh.GetFaceVertexIndicesAttr().Get())
        elif prim.IsA(UsdGeom.Cube):
            item['cube_size'] = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        records[path] = item
    return records


def capture_arrangement(app, arr, output, assets_dir, tags_dir, textures_dir,
                        *, lighting='L1', settle=100, capture_updates=100,
                        rt_subframes=16, qualification=False, include_reference=True):
    """Capture one arrangement's nominal and actual revisit cameras.

    The caller validates and freezes all inputs, authorizes this call, and owns
    transactional writes, provenance, repeat selection and resume verification.
    """
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf

    batch.check_arrangement(arr)
    batch.check_scene(arr)
    batch.check_render_inputs([(arr, b'')], [lighting], textures_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'rgb').mkdir()
    (output / 'masks').mkdir()
    built = batch.build_stage(arr, Path(assets_dir), Path(tags_dir), {}, textures_dir)
    stage = omni.usd.get_context().get_stage()
    rig.set_lighting(stage, built, lighting)
    cams = {c['name']: c for c in arr['scene']['cameras']}
    if len(cams) != 3:
        raise ValueError('Expected three unique camera stops')
    roles = {built['target']: 1, built['occluder']: 2}
    roles.update({p: 10 + i for i, p in enumerate(built['distractors'])})

    def grab(cam_set, *, rgb, updates, subframes):
        paths = built['cameras' if cam_set == 'nominal' else 'cameras_revisit']
        products, annotators = [], {}
        try:
            for name, path in paths.items():
                c = cams[name]
                rp = rep.create.render_product(path, (c['width'], c['height']))
                seg = rep.AnnotatorRegistry.get_annotator('instance_id_segmentation', init_params={'colorize': False})
                seg.attach(rp)
                color = rep.AnnotatorRegistry.get_annotator('rgb') if rgb else None
                if color is not None:
                    color.attach(rp)
                products.append(rp)
                annotators[name] = (color, seg)
            for _ in range(updates):
                app.update()
            rep.orchestrator.step(rt_subframes=subframes)
            data = {}
            for name, (color, seg) in annotators.items():
                raw = seg.get_data()
                cls = batch.class_map(raw['data'], raw['info'].get('idToLabels', {}), roles)
                image = None if color is None else np.asarray(color.get_data())[..., :3].copy()
                data[name] = (image, cls)
            return data
        finally:
            for color, seg in annotators.values():
                if color is not None:
                    color.detach()
                seg.detach()
            for product in products:
                product.destroy()

    grab('nominal', rgb=True, updates=settle, subframes=rt_subframes)
    captures = list(STATES) if include_reference else list(STATES)[1:]
    if qualification:
        captures.extend(f'repeat{n}_{state}' for n in (1, 2) for state in list(STATES)[1:])
        captures.extend(f'distractor_hidden_{state}' for state in list(STATES)[1:])
    observations = {}
    for index, state in enumerate(captures):
        actual_state = state.removeprefix('repeat1_').removeprefix('repeat2_').removeprefix('distractor_hidden_')
        batch._visible(stage, built['distractors'][0], not state.startswith('distractor_hidden_'))
        present, pose, camera = STATES[actual_state]
        batch._visible(stage, built['target'], present)
        built['occluder_op'].Set(Gf.Matrix4d(*built['occluder_poses'][pose].reshape(-1).tolist()))
        prefix = state
        data = grab(camera, rgb=True, updates=capture_updates, subframes=rt_subframes)
        observation = observe_stage(stage, built, include_geometry=(index == 0))
        observations[prefix] = observation
        for stop, (image, classes) in data.items():
            Image.fromarray(image).save(output / 'rgb' / f'{prefix}_{stop}.png')
            for name, mask in [('target', classes == 1), ('blocker', classes == 2)]:
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'masks' / f'{name}_{prefix}_{stop}.png')
    for p in [built['shelf'], built['occluder'], *built['distractors']]:
        batch._visible(stage, p, False)
    batch._visible(stage, built['target'], True)
    for camera in ('revisit',):
        data = grab(camera, rgb=False, updates=batch.ALONE_UPDATES, subframes=1)
        for stop, (_, classes) in data.items():
            Image.fromarray((classes == 1).astype(np.uint8) * 255).save(output / 'masks' / f'alone_{stop}.png')

    return observations
