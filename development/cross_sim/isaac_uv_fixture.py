"""Render an asymmetric texture through the project USD mesh/material path."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from sim import isaac_rig as rig


def capture(app, output, *, settle=200, updates=100, subframes=16):
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, UsdGeom, UsdLux

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    texture = np.zeros((128, 128, 3), dtype=np.uint8)
    texture[:64, :64] = [255, 0, 0]
    texture[:64, 64:] = [0, 255, 0]
    texture[64:, :64] = [0, 0, 255]
    texture[64:, 64:] = [255, 255, 0]
    tex_path = (output / 'asymmetric.png').resolve()
    Image.fromarray(texture).save(tex_path)
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, '/World')
    prim = rig._mesh(stage, '/World/quad', [[-0.5,0,-0.5],[0.5,0,-0.5],[0.5,0,0.5],[-0.5,0,0.5]],
                     [[0,1,2],[0,2,3]], uv=[[0,0],[1,0],[1,1],[0,1]])
    UsdGeom.Mesh(prim).GetDoubleSidedAttr().Set(True)
    rig._bind(prim, rig._material(stage, 'fixture', (1,1,1), 0.6, str(tex_path)))
    dome = UsdLux.DomeLight.Define(stage, '/World/dome')
    dome.GetIntensityAttr().Set(1000.0)
    cam = UsdGeom.Camera.Define(stage, '/World/camera')
    aperture, vaperture = rig.apertures(300, 256, 256)
    cam.GetFocalLengthAttr().Set(rig.FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(aperture)
    cam.GetVerticalApertureAttr().Set(vaperture)
    matrix = rig.camera_matrix([0,-2,0], [0,1,0], [0,0,1])
    UsdGeom.Xformable(cam.GetPrim()).AddTransformOp().Set(Gf.Matrix4d(*matrix.reshape(-1).tolist()))
    rp = rep.create.render_product('/World/camera', (256,256))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach(rp)
    try:
        for _ in range(settle + updates):
            app.update()
        rep.orchestrator.step(rt_subframes=subframes)
        image = np.asarray(rgb.get_data())[...,:3].copy()
        Image.fromarray(image).save(output / 'observed.png')
        points = {'top_left': (90,90), 'top_right': (90,166),
                  'bottom_left': (166,90), 'bottom_right': (166,166)}
        signature = {k: image[y-5:y+6,x-5:x+6].mean(axis=(0,1)).tolist() for k,(y,x) in points.items()}
        (output / 'signature.json').write_text(json.dumps({'mean_rgb': signature, 'points_yx': points,
            'expected': {'top_left':'red','top_right':'green','bottom_left':'blue','bottom_right':'yellow'},
            'settings': {'settle':settle,'capture_updates':updates,'rt_subframes':subframes}}, indent=2) + '\n')
        return signature
    finally:
        rgb.detach()
        rp.destroy()
