"""Independent, clipped, pixel-center depth oracle for source-scene visibility."""
from __future__ import annotations
import numpy as np


def _camera(camera):
    fwd=np.array(camera['forward'],float); up=np.array(camera['up'],float)
    if np.linalg.norm(fwd)<1e-12 or np.linalg.norm(up)<1e-12: raise ValueError('zero camera axis')
    fwd/=np.linalg.norm(fwd); right=np.cross(fwd,up)
    if np.linalg.norm(right)<1e-12: raise ValueError('parallel camera axes')
    right/=np.linalg.norm(right); up=np.cross(right,fwd)
    width,height=int(camera['width']),int(camera['height']); fx=float(camera['f_px']); fy=float(camera.get('fy_px',camera.get('f_y_px',fx)))
    cx,cy=camera.get('principal_point_px',[width/2,height/2]); params=np.array([fx,fy,cx,cy],float)
    if width<=0 or height<=0 or fx<=0 or fy<=0 or not np.isfinite(params).all(): raise ValueError('invalid camera intrinsics')
    return np.asarray(camera['position'],float),np.stack((right,up,fwd)),params,width,height


def _clip_near(points, near):
    """Clip one camera-space triangle to z >= near; preserve winding."""
    polygon=[]
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        ai=a[2]>=near; bi=b[2]>=near
        if ai: polygon.append(a)
        if ai != bi:
            t=(near-a[2])/(b[2]-a[2]); polygon.append(a+t*(b-a))
    if len(polygon)<3: return []
    return [np.asarray([polygon[0],polygon[i],polygon[i+1]]) for i in range(1,len(polygon)-1)]


def _depth(mesh,camera,near,roi=None):
    origin,axes,(fx,fy,cx,cy),width,height=_camera(camera)
    vertices=(np.asarray(mesh['vertices'],float)-origin)@axes.T
    faces=np.asarray(mesh['faces'],int)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all() or np.any(faces<0) or np.any(faces>=len(vertices)): raise ValueError('invalid mesh')
    if roi is None: x0,y0,x1,y1=0,0,width,height
    else: x0,y0,x1,y1=roi
    depth=np.full((y1-y0,x1-x0),np.inf,dtype=float)
    for face in faces:
        triangle=vertices[face]
        if np.all(triangle[:,2]<near): continue
        pieces=[triangle] if np.all(triangle[:,2]>=near) else _clip_near(triangle,near)
        for tri in pieces:
            x=cx+fx*tri[:,0]/tri[:,2]; y=cy-fy*tri[:,1]/tri[:,2]
            lo=np.maximum(np.ceil([x.min()-.5,y.min()-.5]).astype(int),[x0,y0])
            hi=np.minimum(np.floor([x.max()-.5,y.max()-.5]).astype(int),[x1-1,y1-1])
            if np.any(lo>hi): continue
            denom=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2])
            if abs(denom)<1e-12: continue
            px,py=np.meshgrid(np.arange(lo[0],hi[0]+1)+.5,np.arange(lo[1],hi[1]+1)+.5)
            a=((y[1]-y[2])*(px-x[2])+(x[2]-x[1])*(py-y[2]))/denom
            b=((y[2]-y[0])*(px-x[2])+(x[0]-x[2])*(py-y[2]))/denom
            c=1-a-b; inside=(a>=-1e-10)&(b>=-1e-10)&(c>=-1e-10)
            reciprocal=a/tri[0,2]+b/tri[1,2]+c/tri[2,2]
            valid=inside&(reciprocal>0); patch=depth[lo[1]-y0:hi[1]+1-y0,lo[0]-x0:hi[0]+1-x0]
            candidate=np.full_like(reciprocal,np.inf); candidate[valid]=1/reciprocal[valid]; np.minimum(patch,candidate,out=patch)
    return depth


def visible_mask(subject, foreground_meshes, camera, near_m=1e-4, depth_tolerance_m=1e-6):
    """Return independently rasterized silhouette and its source-geometry-visible subset.

    Geometry at identical depth is not treated as foreground. Meshes must omit
    the subject and include all physical occluders in the evaluated state.
    """
    if near_m<=0 or depth_tolerance_m<0: raise ValueError('invalid depth tolerance')
    _,_,_,width,height=_camera(camera)
    source_depth=_depth(subject,camera,near_m); isolated=np.isfinite(source_depth)
    visible=np.zeros_like(isolated); occluded=np.zeros_like(isolated)
    if isolated.any():
        ys,xs=np.nonzero(isolated); roi=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1)
        x0,y0,x1,y1=roi; front=np.full((y1-y0,x1-x0),np.inf)
        for mesh in foreground_meshes:
            np.minimum(front,_depth(mesh,camera,near_m,roi),out=front)
        local=isolated[y0:y1,x0:x1]; depths=source_depth[y0:y1,x0:x1]
        lost=local & (front+depth_tolerance_m<depths)
        occluded[y0:y1,x0:x1]=lost; visible=isolated & ~occluded
    return {'isolated':isolated,'visible':visible,'occluded':occluded,'isolated_pixels':int(isolated.sum()),
            'visible_pixels':int(visible.sum()),'occluded_pixels':int(occluded.sum()),
            'fully_occluded':bool(isolated.any() and not visible.any()),'near_m':near_m,'depth_tolerance_m':depth_tolerance_m}
