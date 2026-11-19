"""Build an unqualified MuJoCo scene from one Isaac arrangement.

This module only constructs a model. It does not implement the capture protocol.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from sim import isaac_rig as ir
from development.cross_sim.mujoco_geometry import camera_axes, fovy_degrees, local_ycb


def _numbers(a):
    return ' '.join(f'{float(x):.12g}' for x in np.asarray(a).ravel())


def _ints(a):
    return ' '.join(str(int(x)) for x in np.asarray(a).ravel())


def _yaw(yaw):
    a = math.radians(float(yaw)) / 2
    return _numbers([math.cos(a), 0, 0, math.sin(a)])


def build_xml(arr, assets_dir, tags_dir, textures_dir):
    """Return MJCF and an inventory of source assets and scene bindings.

    Texture wrap and color-space conversion are not yet validated. This function
    must not be used for qualification capture until those checks are complete.
    """
    spec = arr['scene']
    cams = spec['cameras'] + spec['revisit_cameras']
    sizes = {(int(c['width']), int(c['height'])) for c in cams}
    if len(sizes) != 1:
        raise ValueError('all cameras must share dimensions')
    width, height = next(iter(sizes))
    root = ET.Element('mujoco', model='cross_sim_development')
    ET.SubElement(root, 'compiler', angle='radian')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'quality', offsamples='0')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.45 .45 .45', specular='0 0 0')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', contype='0', conaffinity='0')
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    inventory = {'files': [], 'bindings': {}, 'limitations': [
        'Lighting is an approximate MuJoCo headlight, not the Isaac lighting.',
        'Texture edge sampling and colour-space response require validation.',
        'MuJoCo materials do not directly reproduce USD roughness and metallic.']}

    def mat(name, rgba, texture=None):
        attrs = {'name': name, 'rgba': _numbers(rgba)}
        if texture:
            path = Path(texture).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            tid = 'tex_' + name
            ET.SubElement(asset, 'texture', name=tid, type='2d', file=str(path))
            attrs['texture'] = tid
            inventory['files'].append(str(path))
        ET.SubElement(asset, 'material', **attrs)
        return name

    materials = {}
    for name, (rgb, _) in ir.MATERIALS.items():
        materials[name] = mat('material_' + name, [*rgb, 1])
    for name, m in spec.get('materials', {}).items():
        tex = m.get('texture')
        verified = ir.verify_texture(tex, textures_dir) if tex else None
        if verified and (verified['params']['uv_transform'] or verified['params']['wrap'] == 'mirror'):
            raise ValueError(f'unsupported texture transform or mirror wrap for {name}')
        path = verified['path'] if verified else None
        materials[name] = mat('spec_' + ir._safe(name), [1, 1, 1, 1] if path else m.get('rgba', [.6, .6, .6, 1]), path)

    def material(name):
        return materials.get(name, materials['grey'])

    def mesh(parent, name, vertices, faces, mat_name, uv=None, normals=None):
        attrs = {'name': name, 'vertex': _numbers(vertices), 'face': _ints(faces), 'inertia': 'shell'}
        if uv is not None:
            if len(uv) != len(vertices):
                raise ValueError(f'{name}: UV count differs from vertex count')
            texcoord = np.array(uv, dtype=float, copy=True)
            texcoord[:, 1] = 1.0 - texcoord[:, 1]
            attrs['texcoord'] = _numbers(texcoord)
        if normals is not None:
            if len(normals) != len(vertices):
                raise ValueError(f'{name}: normal count differs from vertex count')
            attrs['normal'] = _numbers(normals)
        ET.SubElement(asset, 'mesh', **attrs)
        ET.SubElement(parent, 'geom', name=name, type='mesh', mesh=name, material=mat_name)
        inventory['bindings'][name] = mat_name

    for i, b in enumerate(spec.get('boxes', [])):
        if b.get('role') == 'occluder':
            raise ValueError('occluder must be an arrangement object')
        name = f'box_{i}'
        ET.SubElement(world, 'geom', name=name, type='box', pos=_numbers(b['center']), size=_numbers(b['half']),
                      quat=_numbers(b.get('quat_wxyz', [1, 0, 0, 0])), material=material(b.get('material', 'grey')))
        inventory['bindings'][name] = material(b.get('material', 'grey'))
    for i, m in enumerate(spec.get('meshes', [])):
        if ir.is_static_mesh(m):
            ir.check_mesh_uv(spec, m)
            mesh(world, f'static_{i}', m['vertices'], m['faces'], material(m.get('material', 'grey')),
                 m.get('uv'), m.get('normals'))
    if spec.get('tags_enabled', True):
        for t in spec.get('tags', []):
            name = f'tag_{t["id"]}'
            tm = mat('material_' + name, [1, 1, 1, 1], Path(tags_dir) / f'aruco_4x4_50_id{t["id"]}.png')
            mesh(world, name, ir.tag_quad(t['corners']), [[0, 3, 2], [0, 2, 1]], tm,
                 [[0, 1], [1, 1], [1, 0], [0, 0]])

    def object_body(name, pose):
        return ET.SubElement(world, 'body', name=name, pos=_numbers(pose['base']), quat=_yaw(pose['yaw_deg']))

    def ycb(body, name, rec):
        path = Path(assets_dir) / rec['ply']
        d = ir.read_ply(path)
        inventory['files'].append(str(path.resolve()))
        v, n = local_ycb(d['vertices'], rec['rest_matrix'], d['normals'])
        tex = path.parent / d['texture'] if d['texture'] else None
        material_name = mat('material_' + name, [1, 1, 1, 1] if tex else [.8, .8, .8, 1], tex)
        mesh(body, name, v, d['faces'], material_name, d['uv'], n)

    objs = arr['objects']
    target = objs['target']
    if target['kind'] == 'ycbv':
        ycb(object_body('target', target['pose']), 'target_mesh', target)
    elif target['kind'] == 'mesh':
        body = ET.SubElement(world, 'body', name='target')
        for i, m in enumerate(spec['meshes']):
            if m['name'] in target['meshes']:
                mesh(body, f'target_mesh_{i}', m['vertices'], m['faces'], material(m.get('material', 'mug')))
    else:
        raise ValueError(f'unsupported target kind {target["kind"]!r}')
    blocker = objs['occluder']
    body = object_body('blocker', blocker['poses']['parked'])
    if blocker['kind'] == 'box':
        size = np.asarray(blocker['size_m'], float)
        ET.SubElement(body, 'geom', name='blocker_box', type='box', pos=_numbers([0, 0, size[2] / 2]),
                      size=_numbers(size / 2), material=material(blocker.get('material', 'occluder')))
    elif blocker['kind'] == 'ycbv':
        ycb(body, 'blocker_mesh', blocker)
    else:
        raise ValueError(f'unsupported blocker kind {blocker["kind"]!r}')
    for i, d in enumerate(objs['distractors']):
        ycb(object_body(f'distractor_{i}', d['pose']), f'distractor_mesh_{i}', d)
    for kind, group in [('nominal', spec['cameras']), ('revisit', spec['revisit_cameras'])]:
        for c in group:
            right, up, _ = camera_axes(c['forward'], c['up'])
            ET.SubElement(world, 'camera', name=f'{kind}_{c["name"]}', pos=_numbers(c['position']),
                          xyaxes=_numbers([right, up]), fovy=str(fovy_degrees(c['f_px'], c['height'])))
    return ET.tostring(root, encoding='unicode'), inventory


def compiled_geom_bounds(model, data, geom_id):
    """Return world AABB from compiled mesh vertices or exact box corners."""
    import mujoco
    kind = int(model.geom_type[geom_id])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[geom_id])
        start, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[start:start + count], dtype=float)
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        from itertools import product
        vertices = np.array(list(product((-1, 1), repeat=3)), dtype=float) * model.geom_size[geom_id]
    else:
        raise ValueError(f'unsupported compiled geometry type {kind}')
    rotation = np.asarray(data.geom_xmat[geom_id]).reshape(3, 3)
    points = vertices @ rotation.T + data.geom_xpos[geom_id]
    return [points.min(axis=0).tolist(), points.max(axis=0).tolist()]


def compiled_observations(model, data, arr):
    """Read effective camera, body and geometry values from a forwarded model."""
    import mujoco
    mujoco.mj_forward(model, data)
    spec = arr['scene']
    result = {'cameras': {'nominal': {}, 'revisit': {}}, 'objects': {}, 'static_bounds': {}}
    for kind, cams in [('nominal', spec['cameras']), ('revisit', spec['revisit_cameras'])]:
        for cam in cams:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f'{kind}_{cam["name"]}')
            if cid < 0:
                raise ValueError('compiled camera missing')
            rotation = np.asarray(data.cam_xmat[cid]).reshape(3, 3)
            fovy = float(model.cam_fovy[cid])
            h, w = int(cam['height']), int(cam['width'])
            result['cameras'][kind][cam['name']] = {'position': data.cam_xpos[cid].tolist(),
                'forward': (-rotation[:, 2]).tolist(), 'up': rotation[:, 1].tolist(), 'right': rotation[:, 0].tolist(),
                'f_px': h / (2 * math.tan(math.radians(fovy) / 2)), 'height': h, 'width': w,
                'principal_point_px': [w / 2, h / 2]}
    for name in ['target', 'blocker'] + [f'distractor_{i}' for i in range(len(arr['objects']['distractors']))]:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f'compiled body {name} missing')
        geom_ids = np.flatnonzero(model.geom_bodyid == bid)
        if not len(geom_ids):
            raise ValueError(f'compiled body {name} has no geometry')
        bounds = np.array([compiled_geom_bounds(model, data, int(g)) for g in geom_ids])
        result['objects'][name] = {'position': data.xpos[bid].tolist(),
            'rotation': data.xmat[bid].reshape(3, 3).tolist(),
            'bounds': [bounds[:, 0].min(axis=0).tolist(), bounds[:, 1].max(axis=0).tolist()],
            'enabled_geom_groups': [int(model.geom_group[g]) for g in geom_ids]}
    for prefix, items in [('box', spec.get('boxes', [])), ('static', spec.get('meshes', []))]:
        for i, item in enumerate(items):
            if prefix == 'static' and not ir.is_static_mesh(item):
                continue
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{prefix}_{i}')
            if gid < 0:
                raise ValueError(f'compiled static geometry {prefix}_{i} missing')
            result['static_bounds'][item['name']] = compiled_geom_bounds(model, data, gid)
    return result


def apply_state(model, data, arr, state):
    """Set one intervention and return the forwarded, observed state."""
    import mujoco
    allowed = {'clear_present', 'clear_removed', 'placed_present', 'placed_removed', 'reference'}
    if state not in allowed:
        raise ValueError(f'unknown state {state!r}')
    blocker_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'blocker')
    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'target')
    if min(blocker_id, target_id) < 0:
        raise ValueError('missing intervention body')
    pose_key = 'placed' if state.startswith('placed_') else 'parked'
    pose = arr['objects']['occluder']['poses'][pose_key]
    model.body_pos[blocker_id] = pose['base']
    a = math.radians(float(pose['yaw_deg'])) / 2
    model.body_quat[blocker_id] = [math.cos(a), 0, 0, math.sin(a)]
    target_groups = np.flatnonzero(model.geom_bodyid == target_id)
    if len(target_groups) == 0:
        raise ValueError('missing target geometry')
    model.geom_group[target_groups] = 5 if state.endswith('_removed') else 0
    observed = compiled_observations(model, data, arr)
    observed['state'] = {'target_enabled': bool(np.all(model.geom_group[target_groups] == 0)),
                         'target_geom_groups': model.geom_group[target_groups].tolist(),
                         'blocker': observed['objects']['blocker']}
    return observed
