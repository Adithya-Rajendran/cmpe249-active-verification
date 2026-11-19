"""Independent source-geometry and native-pixel mask oracle for development qualification."""
from __future__ import annotations
import math
import struct
from pathlib import Path
import numpy as np

_TYPES = {'char':'b','int8':'b','uchar':'B','uint8':'B','short':'h','int16':'h','ushort':'H','uint16':'H',
          'int':'i','int32':'i','uint':'I','uint32':'I','float':'f','float32':'f','double':'d','float64':'d'}
_BOX_FACES = np.array([[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],
                       [2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]],dtype=int)


def read_ply(path):
    """Read vertex positions and polygon indices without renderer mesh helpers."""
    with Path(path).open('rb') as stream:
        if stream.readline().strip() != b'ply': raise ValueError('invalid PLY magic')
        fmt=None; elements=[]; current=None
        for _ in range(500):
            raw=stream.readline()
            if not raw: raise ValueError('truncated PLY header')
            line=raw.decode('ascii').strip().split()
            if not line: continue
            if line[0]=='format': fmt=line[1]
            elif line[0]=='element':
                current={'name':line[1],'count':int(line[2]),'props':[]}; elements.append(current)
            elif line[0]=='property':
                if current is None: raise ValueError('property without element')
                current['props'].append(line[1:])
            elif line[0]=='end_header': break
        else: raise ValueError('unterminated PLY header')
        if fmt not in ('ascii','binary_little_endian'): raise ValueError('unsupported PLY encoding')
        vertices=[]; polygons=[]
        for element in elements:
            for _ in range(element['count']):
                parts=iter(stream.readline().decode('ascii').split()) if fmt=='ascii' else None
                values={}
                def scalar(kind):
                    if kind not in _TYPES: raise ValueError('unsupported PLY type')
                    if parts is not None: return float(next(parts)) if _TYPES[kind] in 'fd' else int(next(parts))
                    code='<'+_TYPES[kind]; size=struct.calcsize(code); raw=stream.read(size)
                    if len(raw)!=size: raise ValueError('truncated PLY body')
                    return struct.unpack(code,raw)[0]
                for prop in element['props']:
                    if prop[0]=='list':
                        count=scalar(prop[1]); values[prop[3]]=[scalar(prop[2]) for _ in range(count)]
                    else: values[prop[1]]=scalar(prop[0])
                if element['name']=='vertex': vertices.append([values[k] for k in ('x','y','z')])
                if element['name']=='face':
                    indices=values.get('vertex_indices',values.get('vertex_index'))
                    if indices is None or len(indices)<3: raise ValueError('invalid PLY face')
                    polygons.extend([[indices[0],indices[i],indices[i+1]] for i in range(1,len(indices)-1)])
    v=np.asarray(vertices,dtype=float); f=np.asarray(polygons,dtype=int)
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not len(v) or not len(f) or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v): raise ValueError('invalid PLY geometry')
    return v,f


def _mesh(vertices, faces):
    v=np.asarray(vertices,float); f=np.asarray(faces,int)
    if v.ndim!=2 or v.shape[1]!=3 or not len(v) or f.ndim!=2 or f.shape[1]!=3 or not len(f) or not np.isfinite(v).all() or f.min()<0 or f.max()>=len(v): raise ValueError('invalid mesh')
    return {'vertices':v,'faces':f,'bounds':np.stack([v.min(0),v.max(0)])}


def _rotation_yaw(degrees):
    a=math.radians(float(degrees)); c,s=math.cos(a),math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]],float)


def _quaternion(q):
    w,x,y,z=np.asarray(q,float); norm=math.sqrt(w*w+x*x+y*y+z*z)
    if norm<1e-12: raise ValueError('zero quaternion')
    w,x,y,z=np.array([w,x,y,z])/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _box(center, half, quat=(1,0,0,0)):
    half=np.asarray(half,float)
    if half.shape!=(3,) or np.any(half<=0): raise ValueError('invalid box half size')
    local=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],float)*half
    return _mesh(local@_quaternion(quat).T+np.asarray(center,float),_BOX_FACES)


def object_mesh(record, assets_dir, pose=None):
    """Build one target, blocker, or distractor from the original record."""
    pose=pose or record['pose']; base=np.asarray(pose['base'],float); rotation=_rotation_yaw(pose['yaw_deg'])
    if record['kind']=='box':
        size=np.asarray(record['size_m'],float)
        if size.shape!=(3,) or np.any(size<=0): raise ValueError('invalid object size')
        local=_box([0,0,size[2]/2],size/2)
        return _mesh(local['vertices']@rotation.T+base,local['faces'])
    if record['kind']!='ycbv': raise ValueError('unsupported object kind')
    v,f=read_ply(Path(assets_dir)/record['ply']); r=np.asarray(record['rest_matrix'],float)
    if r.shape!=(3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(r),1,atol=1e-6): raise ValueError('invalid rest matrix')
    v=(v/1000)@r.T; lo=v.min(0); hi=v.max(0); v-=[(lo[0]+hi[0])/2,(lo[1]+hi[1])/2,lo[2]]
    return _mesh(v@rotation.T+base,f)


def source_geometry(arr, assets_dir):
    """Return world meshes for each independent source object and static surface."""
    spec=arr['scene']; objects=arr['objects']; result={}
    target=objects['target']
    if target['kind']=='mesh':
        matches=[m for m in spec['meshes'] if m['name'] in target['meshes']]
        if len(matches)!=len(target['meshes']): raise ValueError('target mesh missing')
        vertices=[]; faces=[]; offset=0
        for m in matches:
            v=np.asarray(m['vertices'],float); vertices.extend(v); faces.extend(np.asarray(m['faces'],int)+offset); offset+=len(v)
        result['target']=_mesh(vertices,faces)
    else: result['target']=object_mesh(target,assets_dir)
    occ=objects['occluder']
    for state in ('placed','parked'): result['blocker_'+state]=object_mesh(occ,assets_dir,occ['poses'][state])
    for i,obj in enumerate(objects['distractors']): result[f'distractor_{i}']=object_mesh(obj,assets_dir)
    for i,box in enumerate(spec.get('boxes',[])):
        if box.get('role') in ('target','occluder'): continue
        name=box['name']
        if name in result: raise ValueError('duplicate geometry name')
        result[name]=_box(box['center'],box['half'],box.get('quat_wxyz',[1,0,0,0]))
    for i,mesh in enumerate(spec.get('meshes',[])):
        if mesh.get('role') in ('target','occluder') or mesh.get('name') in target.get('meshes',[]): continue
        name=mesh['name']
        if name in result: raise ValueError('duplicate geometry name')
        result[name]=_mesh(mesh['vertices'],mesh['faces'])
    return result


def camera_record(arr, stop, revisit=False):
    matches=[x for x in arr['scene']['revisit_cameras' if revisit else 'cameras'] if x['name']==stop]
    if len(matches)!=1: raise ValueError('camera missing or duplicate')
    return matches[0]


def project(points, camera):
    """Project world points with +u right, +v down, and positive forward depth."""
    forward=np.asarray(camera['forward'],float); up=np.asarray(camera['up'],float)
    forward/=np.linalg.norm(forward); right=np.cross(forward,up); right/=np.linalg.norm(right); up=np.cross(right,forward)
    if not np.isfinite(np.r_[forward,right,up]).all(): raise ValueError('invalid camera axes')
    delta=np.asarray(points,float)-np.asarray(camera['position'],float)
    depth=delta@forward
    if np.any(depth<=1e-8): raise ValueError('geometry intersects or crosses camera plane')
    f=float(camera['f_px']); width=int(camera['width']); height=int(camera['height'])
    if f<=0 or width<=0 or height<=0: raise ValueError('invalid pinhole')
    cx,cy=camera.get('principal_point_px',[width/2,height/2]); fy=float(camera.get('fy_px',camera.get('f_y_px',f)))
    if fy<=0 or not np.isfinite([cx,cy,fy]).all(): raise ValueError('invalid pinhole intrinsics')
    xy=np.column_stack((cx+f*(delta@right)/depth,cy-fy*(delta@up)/depth))
    return xy,depth


def rasterize(mesh, camera):
    """Rasterize projected source triangles at pixel centers, with no edge dilation."""
    xy,_=project(mesh['vertices'],camera); w,h=int(camera['width']),int(camera['height']); mask=np.zeros((h,w),bool)
    for ids in mesh['faces']:
        t=xy[ids]; lo=np.maximum(np.ceil(t.min(0)-.5).astype(int),0); hi=np.minimum(np.floor(t.max(0)-.5).astype(int),[w-1,h-1])
        if np.any(lo>hi): continue
        xs=np.arange(lo[0],hi[0]+1,dtype=float)+.5; ys=np.arange(lo[1],hi[1]+1,dtype=float)+.5
        x,y=np.meshgrid(xs,ys)
        edges=[]
        for a,b in ((0,1),(1,2),(2,0)):
            edges.append((t[b,0]-t[a,0])*(y-t[a,1])-(t[b,1]-t[a,1])*(x-t[a,0]))
        area=(t[1,0]-t[0,0])*(t[2,1]-t[0,1])-(t[1,1]-t[0,1])*(t[2,0]-t[0,0])
        if abs(area)<1e-12: continue
        inside=np.logical_and.reduce([e>=-1e-10 for e in edges]) if area>0 else np.logical_and.reduce([e<=1e-10 for e in edges])
        mask[lo[1]:hi[1]+1,lo[0]:hi[0]+1] |= inside
    return mask


def _dilate(mask, radius):
    out=np.asarray(mask,bool).copy(); h,w=out.shape
    # Chebyshev-radius pixel band; includes both sides of each raster edge.
    src=np.asarray(mask,bool); padded=np.pad(src,radius)
    for dy in range(2*radius+1):
        for dx in range(2*radius+1): out |= padded[dy:dy+h,dx:dx+w]
    return out


def _erode(mask, radius):
    src=np.asarray(mask,bool); h,w=src.shape; padded=np.pad(src,radius,constant_values=False); out=src.copy()
    for dy in range(2*radius+1):
        for dx in range(2*radius+1): out &= padded[dy:dy+h,dx:dx+w]
    return out


def compare_mask(observed, expected, excess_only=False, band_px=2, threshold=.005):
    """Compare mask areas beyond the band around the opposite mask boundary."""
    obs=np.asarray(observed,bool); exp=np.asarray(expected,bool)
    if obs.ndim!=2 or obs.shape!=exp.shape or not isinstance(band_px,int) or band_px<0: raise ValueError('mask shape or band mismatch')
    area=int(exp.sum())
    excess=int((obs & ~_dilate(exp,band_px)).sum())
    missing=0 if excess_only else int((_erode(exp,band_px) & ~obs).sum())
    error=(excess+missing)/area if area else None
    return {'expected_pixels':area,'observed_pixels':int(obs.sum()),'excess_outside_band':excess,'missing_outside_band':missing,
            'error_fraction':error,'band_px':band_px,'band_metric':'Chebyshev','excess_only':excess_only,'threshold':threshold,'pass':error<=threshold if area else not obs.any()}
