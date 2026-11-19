"""Render a synthetic asymmetric UV fixture using the adapter texture path."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
from development.cross_sim.mujoco_scene import build_xml


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (128, 128))
    draw = ImageDraw.Draw(img)
    for box, colour, label in [((0, 0, 63, 63), 'red', 'TL'), ((64, 0, 127, 63), 'green', 'TR'),
                               ((0, 64, 63, 127), 'blue', 'BL'), ((64, 64, 127, 127), 'yellow', 'BR')]:
        draw.rectangle(box, fill=colour)
        draw.text((box[0] + 5, box[1] + 5), label, fill='white')
    texture = out / 'canonical.png'
    img.save(texture)
    sha = hashlib.sha256(texture.read_bytes()).hexdigest()
    c = dict(name='P1', width=256, height=256, position=[0, 0, 3], forward=[0, 0, -1], up=[0, 1, 0], f_px=250)
    quad = dict(name='quad', role='static', material='fixture', vertices=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
                faces=[[0,1,2],[0,2,3]], uv=[[0,0],[1,0],[1,1],[0,1]], normals=[[0,0,1]]*4)
    arr = {'scene': {'cameras':[c], 'revisit_cameras':[c], 'boxes':[], 'meshes':[quad], 'tags_enabled':False,
                     'materials': {'fixture': {'rgba':[1,1,1,1], 'texture': {'file':str(texture.resolve()),
                                                 'sha256':sha, 'wrap':'clamp','colour_space':'sRGB'}}}},
           'objects': {'target': {'kind':'mesh','meshes':[]},
                       'occluder': {'kind':'box','size_m':[.1,.1,.1], 'poses':{'parked':{'base':[3,0,0],'yaw_deg':0}}},
                       'distractors':[]}}
    xml, inventory = build_xml(arr, out, out, out)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 256, 256)
    try:
        renderer.update_scene(data, camera='nominal_P1')
        rgb = renderer.render().copy()
    finally:
        renderer.close()
    Image.fromarray(rgb).save(out / 'observed.png')
    samples = {'top_left':(87,87), 'top_right':(87,169), 'bottom_left':(169,87), 'bottom_right':(169,169)}
    expected = {'top_left':'red', 'top_right':'green', 'bottom_left':'blue', 'bottom_right':'yellow'}
    observed = {k:rgb[y,x].tolist() for k,(y,x) in samples.items()}
    report = {'fixture':'synthetic', 'texture_sha256':sha, 'expected_quadrants':expected, 'observed_rgb':observed,
              'sample_yx':samples, 'source_uv':'bottom-left (0,0), top-right (1,1)',
              'actual_adapter_texture_ingestion':True, 'mujoco_version':mujoco.__version__, 'inventory':inventory}
    (out / 'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
