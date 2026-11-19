"""Generate independent service corridor arrangements for the transfer study."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from sim.rig import Camera
from sim.arrange import arrangement_hash

ROOT = Path(__file__).resolve().parents[2]
STATES = ['clear_present', 'clear_removed', 'placed_present', 'placed_removed']
STOPS = ['P1', 'P2', 'P3']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera(name, position, heading, pitch, roll=0):
    return Camera(name, tuple(position), heading, pitch, roll, 1731.26, 2304, 1296)


def camera_record(c):
    right, up, forward = c.axes()
    return dict(name=c.name, position=list(c.position), forward=forward.tolist(), up=up.tolist(),
                right=right.tolist(), f_px=c.f_px, width=c.width, height=c.height,
                principal_point_px=[1152., 648.], fovy_deg=c.fovy_deg,
                heading_deg=c.heading_deg, pitch_down_deg=c.pitch_down_deg, roll_deg=c.roll_deg)


def level_camera(c, edge):
    uv = c.project(np.array(edge))
    # Camera's positive roll rotates the image content counter-clockwise.
    angle = math.degrees(math.atan2(uv[1, 1]-uv[0, 1], uv[1, 0]-uv[0, 0]))
    return camera('P2', c.position, c.heading_deg, c.pitch_down_deg, angle)


def box(name, center, size, material, role='corridor'):
    return dict(name=name, role=role, label=role, center=list(center),
                half=[v/2 for v in size], material=material, quat_wxyz=[1, 0, 0, 0])


def object_record(obj_id, xyz, yaw, role):
    cfg = json.loads((ROOT/'configs/sim/arrange_shelf_v3.json').read_text())
    # Configuration supplies native YCB metadata only, not scene geometry.
    def find(node):
        if isinstance(node, dict):
            if str(obj_id) in node and isinstance(node[str(obj_id)], dict) and 'ply' in node[str(obj_id)]: return node[str(obj_id)]
            for v in node.values():
                found = find(v)
                if found: return found
        return None
    obj = find(cfg)
    if not obj: raise ValueError(f'missing YCB metadata {obj_id}')
    return dict(name=role, role=role, label=obj['name'], kind='ycbv',
                size_m=[v/1000 for v in obj['size_mm']], obj_id=obj_id, ycb=obj['ycb'], ply=obj['ply'],
                rest='upright', rest_matrix=[[1,0,0],[0,1,0],[0,0,1]],
                pose=dict(base=list(xyz), yaw_deg=yaw))


def build(seed, base_id, level):
    i = seed - 4171 if seed != 4169 else 8
    sign = 1 if (i // 2) % 2 == 0 else -1
    width = 1.65 + 0.10*(i % 4)
    door_y = 0.15 + 0.07*(i % 3)
    target_x = sign*(0.08 + 0.035*(i % 3))
    target_y = door_y + 0.32 + 0.04*(i % 4)
    target_id = 12 if i % 2 == 0 else 15
    target = object_record(target_id, [target_x, target_y, 0], 30*i, 'target')
    # Door remains in the same plane while sliding sideways along its track.
    door_width = 0.95
    door = dict(name='occluder', role='occluder', label='sliding_door', kind='box',
                size_m=[door_width, 0.045, 1.65], material='door', family='sliding_door',
                poses={'placed':dict(base=[0,door_y-0.08,0], yaw_deg=0),
                       'parked':dict(base=[-sign*0.98,door_y-0.08,0], yaw_deg=0)})
    stripe_y = door_y - 0.33
    edge = [[-0.5, stripe_y-0.0175, 0.003], [0.5, stripe_y-0.0175, 0.003]]
    boxes = [box('floor', [0,0.2,-0.035], [width+1.8,4.5,0.07], 'floor'),
             box('left_wall', [-width/2-0.05,0.65,1.05], [0.1,3.0,2.1], 'wall'),
             box('right_wall', [width/2+0.05,0.65,1.05], [0.1,3.0,2.1], 'wall'),
             box('back_wall', [0,2.15,1.05], [width,0.1,2.1], 'back'),
             box('door_post_left', [-0.48,door_y,0.9], [0.10,0.12,1.8], 'frame'),
             box('door_post_right', [0.48,door_y,0.9], [0.10,0.12,1.8], 'frame'),
             box('door_return_left', [-(width/2+0.53)/2,door_y+0.02,0.9], [width/2-0.53,0.08,1.8], 'wall'),
             box('door_return_right', [(width/2+0.53)/2,door_y+0.02,0.9], [width/2-0.53,0.08,1.8], 'wall'),
             box('door_header', [0,door_y,1.87], [width,0.14,0.14], 'frame'),
             box('door_track', [0,door_y-0.08,1.70], [2.95,0.09,0.08], 'track'),
             box('floor_transverse_mark', [0,stripe_y,0.0015], [1.0,0.035,0.003], 'mark'),
             box('wall_fixture', [sign*(width/2-0.025),1.28,0.62+0.1*(i%3)], [0.05,0.32,0.35], 'fixture'),
             box('back_panel', [-sign*0.42,2.08,0.75], [0.35,0.035,0.52], 'fixture')]
    distractors = [object_record(16, [-sign*0.53, target_y+0.36, 0], 15*i, 'distractor_0')]
    # Point P1 at the remembered target. P3 is independently frontal.
    pos = np.array([target_x-sign*0.72, -1.30-0.05*(i%3), 0.68])
    aim = np.array([target_x,target_y,0.13])
    d = aim-pos
    heading = math.degrees(math.atan2(d[0],d[1]))
    pitch = math.degrees(math.atan2(-d[2], math.hypot(d[0],d[1])))
    frontal_pos = np.array([0,-1.40-0.05*(i%3),0.68])
    frontal_pitch = math.degrees(math.atan2(0.55,target_y-frontal_pos[1]))
    nominal1 = camera('P1',pos,heading,pitch)
    nominal = [nominal1,level_camera(nominal1,edge),camera('P3',frontal_pos,0,frontal_pitch)]
    if level == 'D6':
        revisit1 = camera('P1',pos+np.array([0.06*sign,0,0]),heading+3*sign,pitch)
        revisit = [revisit1,level_camera(revisit1,edge),camera('P3',frontal_pos+np.array([0.06*sign,0,0]),0,frontal_pitch)]
    else:
        revisit = nominal
    materials = {k:{'rgba':v} for k,v in {'floor':[.49,.50,.48,1], 'wall':[.67,.68,.65,1],
        'back':[.58,.60,.59,1], 'frame':[.30,.33,.34,1], 'track':[.23,.25,.26,1],
        'door':[.38,.44,.47,1], 'mark':[.90,.77,.30,1], 'fixture':[.40,.43,.40,1]}.items()}
    scene = dict(schema='cmpe249.sim-scene.v1', units='m', frame='x lateral, y corridor depth, z height',
        camera_model='pinhole, square pixels, centered principal point', labels=['corridor','target','occluder'],
        materials=materials, boxes=boxes, meshes=[], tags=[], tags_enabled=False,
        cameras=[camera_record(c) for c in nominal], revisit_cameras=[camera_record(c) for c in revisit],
        lighting={'aim':[0,0.4,0], 'presets':{'L1':{'dome_intensity':500,
          'lights':[{'name':'corridor_ceiling','position':[0,-0.3,2.5], 'size_m':1.5,'intensity':4000,'on':True}]}}},
        visibility_classes={'H_max':0.05,'C_min':0.9})
    # Conservative axis-aligned bounding box encloses every yawed target vertex.
    radius=math.hypot(*target['size_m'][:2])/2
    corners=np.array([[target_x+x,target_y+y,z] for x,y,z in itertools.product([-radius,radius],[-radius,radius],[0,target['size_m'][2]])])
    checks={}
    for label, cams in [('nominal',nominal),('revisit',revisit)]:
        checks[label]={}
        for c in cams:
            margin=c.frame_margin(corners)
            uv=c.project(np.array(edge)); slope=float(uv[1,1]-uv[0,1])
            if margin < 20: raise ValueError(f'{base_id} {level} {label} {c.name} margin {margin}')
            # Check the center and conservative target corners through the aperture plane.
            points = np.vstack([corners, aim])
            t = (door_y-c.position[1])/(points[:,1]-c.position[1])
            aperture_x = c.position[0] + t*(points[:,0]-c.position[0])
            aperture_margin = float(0.43 - max(abs(aperture_x)))
            if aperture_margin <= 0: raise ValueError(f'{base_id} {c.name} target ray misses aperture')
            checks[label][c.name]={'target_bbox_frame_margin_px':margin,'stripe_endpoint_dy_px':slope,
                                  'aperture_ray_margin_m':aperture_margin}
        if abs(checks[label]['P2']['stripe_endpoint_dy_px']) > 1e-6: raise ValueError('stripe not level')
    arr = dict(schema='cmpe249.sim-arrangement.v1', arrangement_id=base_id+level, base_id=base_id,
        study_id='corridor-transfer-v1', level=level, seed=seed, split='dev', kind='independent', environment='corridor',
        target_id=target['label'], occluder_family='sliding_door', objects={'target':target,'occluder':door,'distractors':distractors},
        stops=STOPS, start_stop='P3', lightings=['L1'], scene=scene,
        revisit={'schema':'cmpe249.stress-revisit.v1','level':{'id':level,'translation_m':0 if level=='D0' else 0.06,
          'yaw_deg':0 if level=='D0' else 3},'stops':{}},
        analytic={'floor_mark_edge_world_m':edge, 'projection_checks':checks,
                  'note':'Frame bounds are analytic, not rendered visibility labels.'},
        design={'corridor_width_m':width,'door_depth_m':door_y,'target_side':sign})
    arr['arrangement_sha256']=arrangement_hash(arr)
    return arr


def generate(output, fixture=False):
    out=Path(output)
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    entries=[]
    specs=[('CORFIX',4169)] if fixture else [(f'COR{i:02}',4170+i) for i in range(1,9)]
    for base,seed in specs:
        entry={'base_id':base,'environment':'corridor','seed':seed,'levels':{}}
        for level in ['D0','D6']:
            path=out/(base+level+'.json');path.write_text(json.dumps(build(seed,base,level),indent=2)+'\n')
            entry['levels'][level]={'path':path.name,'sha256':digest(path)}
        entries.append(entry)
    manifest={'study_id':'corridor-transfer-v1','split':'pilot','stops':STOPS,'states':STATES,
              'levels':['D0','D6'],'lighting':'L1','reference':'nominal_clear_present','arrangements':entries}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--fixture',action='store_true')
    a=p.parse_args();generate(a.out,a.fixture)
