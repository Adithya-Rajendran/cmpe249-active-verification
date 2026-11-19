"""Use the paper capture routines with this study's moving door and two mounted lights."""
import hashlib
import sys
import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sim import isaac_rig as rig


def digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def door_parts(arr):
    door=arr['objects']['occluder']; width,depth,height=door['size_m']
    # Move the core rearwards. Its front sheet and pull fit inside the declared envelope.
    core=[width,depth-.016,height]; offset=.008
    dressing=arr['door_dressing']
    return core,offset,[dressing['skin'],dressing['badge']],dressing['boxes']


def isaac_stage(original, arr, *args, **kwargs):
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade
    from sim import isaac_batch as batch
    built=original(arr,*args,**kwargs);stage=omni.usd.get_context().get_stage();parent=built['occluder']
    core,offset,planes,boxes=door_parts(arr)
    prim=stage.GetPrimAtPath(parent+'/box');mat=batch.box_local_matrix(core);mat[3,1]=offset
    UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(Gf.Matrix4d(*mat.ravel().tolist()))
    def surface(material):
        return UsdShade.Material.Get(stage,'/World/Looks/spec_'+rig._safe(material))
    for p in planes:
        prim=rig._mesh(stage,parent+'/'+p['name'],p['vertices'],p['faces'],p['normals'],p['uv']);rig._bind(prim,surface(p['material']))
    for b in boxes:
        matrix=rig.box_matrix(b['center'],b['half'],b['quat_wxyz']);batch._cube(stage,parent+'/'+b['name'],matrix,surface(b['material']))
    return built


def mj_scene(original, arr, *args, **kwargs):
    from development.cross_sim.mujoco_scene import _numbers, _ints
    xml,inventory=original(arr,*args,**kwargs);root=ET.fromstring(xml);world=root.find('worldbody');asset=root.find('asset')
    body=world.find("body[@name='blocker']");slab=body.find("geom[@name='blocker_box']")
    core,offset,planes,boxes=door_parts(arr)
    slab.set('size',_numbers(np.asarray(core)/2));slab.set('pos',_numbers([0,offset,core[2]/2]))
    for p in planes:
        uv=np.asarray(p['uv'],dtype=float).copy();uv[:,1]=1-uv[:,1];name=p['name']
        ET.SubElement(asset,'mesh',name=name,vertex=_numbers(p['vertices']),face=_ints(p['faces']),normal=_numbers(p['normals']),texcoord=_numbers(uv),inertia='shell')
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,material='spec_'+p['material'])
    for b in boxes:
        ET.SubElement(body,'geom',name=b['name'],type='box',pos=_numbers(b['center']),size=_numbers(b['half']),material='spec_'+b['material'])
    visual=root.find('visual');visual.find('quality').attrib.update(offsamples='4',shadowsize='2048');visual.find('headlight').attrib.update(ambient='.27 .28 .29',diffuse='.10 .10 .10',specular='0 0 0')
    lighting=arr['scene']['lighting'];aim=np.asarray(lighting['aim'])
    for i,light in enumerate(lighting['presets']['L1']['lights']):
        pos=np.asarray(light['position']);direction=aim-pos;direction/=np.linalg.norm(direction)
        color=['.82 .78 .71','.69 .75 .82','.31 .32 .34'][i]
        ET.SubElement(world,'light',name=light['name'],pos=_numbers(pos),dir=_numbers(direction),
                      cutoff='78',exponent='1',attenuation='.66 .07 .12',castshadow='true' if i<2 else 'false',
                      diffuse=color,ambient='.075 .075 .07',specular='.13 .13 .13' if i<2 else '0 0 0')
    for name,props in arr['scene']['materials'].items():
        node=asset.find("material[@name='spec_"+rig._safe(name)+"']")
        if node is not None:
            rough=props.get('roughness',.7);node.set('shininess',str(round(1-rough,3)));node.set('specular',str(round(.09+.34*props.get('metallic',0),3)))
    inventory['workshop_lights']=[x['name'] for x in lighting['presets']['L1']['lights']];inventory['moving_parts']=[p['name'] for p in planes+boxes]
    inventory['limitations']=['MuJoCo uses spotlights and shadow maps; Isaac uses ray-traced area lights. Surface reflectance is renderer specific.']
    return ET.tostring(root,encoding='unicode'),inventory


def add_isaac_binding(original,*args,**kwargs):
    spec,items,binding=original(*args,**kwargs);binding['workshop_renderer_sha256']=digest();return spec,items,binding


def add_mj_binding(original,*args,**kwargs):
    binding=original(*args,**kwargs);binding['workshop_renderer_sha256']=digest();return binding


def main():
    engine=sys.argv.pop(1)
    if engine=='isaac':
        from development.paper_audit import isaac_driver as driver
        from sim import isaac_batch as engine_api
        with patch.object(engine_api,'build_stage',partial(isaac_stage,engine_api.build_stage)), patch.object(driver,'inspect',partial(add_isaac_binding,driver.inspect)):
            driver.main()
    elif engine=='mujoco':
        from development.paper_audit import mujoco_capture as driver
        from development.cross_sim import mujoco_scene as engine_api
        with patch.object(engine_api,'build_xml',partial(mj_scene,engine_api.build_xml)), patch.object(driver,'identity',partial(add_mj_binding,driver.identity)):
            driver.main()
    else:
        raise SystemExit('Specify isaac or mujoco, followed by the original paper capture options.')


if __name__=='__main__': main()
