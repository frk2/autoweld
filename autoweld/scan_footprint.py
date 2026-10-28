"""Calibrated laser/reference-plane illumination geometry, without device I/O.

This predicts where an assumed-on laser illuminates a fixed reference plane.
It never creates measured surface XYZ or certifies physical pose timing.
"""
import hashlib
import json
import math

import numpy as np


def _pose(value):
    matrix=np.asarray(value,float)
    if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3],[0,0,0,1],atol=1e-9,rtol=0)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6,rtol=0)
            or np.linalg.det(matrix[:3,:3])<.999999):
        raise ValueError('Footprint requires a finite rigid transform')
    return matrix


def _line(n1,d1,n2,d2):
    normals=np.stack((n1,n2))
    direction=np.cross(n1,n2)
    length=np.linalg.norm(direction)
    if length<1e-9:
        raise ValueError('Laser and reference planes do not define a unique line')
    origin=normals.T@np.linalg.solve(normals@normals.T,-np.array([d1,d2]))
    return origin,direction/length


def _project_visible(points,config):
    import cv2
    uv=cv2.projectPoints(np.asarray(points,float),np.zeros(3),np.zeros(3),
        np.asarray(config['camera_matrix'],float),np.asarray(config['distortion'],float))[0].reshape(-1,2)
    margin=config['sensor_margin_px']
    width,height=config['image_size']
    return (np.isfinite(uv).all(1)&(points[:,2]>.001)&(uv[:,0]>=margin)&(uv[:,0]<=width-1-margin)
            &(uv[:,1]>=margin)&(uv[:,1]<=height-1-margin))


def _visible_intervals(point,direction,low,high,config):
    if high<low:
        return []
    count=max(2,int(math.ceil((high-low)/config['sample_step_m']))+1)
    if count>20000:
        raise ValueError('Footprint sampling interval is unbounded')
    values=np.linspace(low,high,count)
    visible=_project_visible(point+values[:,None]*direction,config)
    indices=np.flatnonzero(visible)
    if not len(indices):
        return []
    groups=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    # Endpoints are INSIDE the conservative sensor margin. Do not extend beyond
    # a tested visible interval or bridge disconnected camera-FoV intervals.
    return [(float(values[group[0]]),float(values[group[-1]])) for group in groups if len(group)>=2]


def _clip_xy(point,direction,width,height):
    low,high=-float('inf'),float('inf')
    for index,extent in enumerate((width,height)):
        if abs(direction[index])<1e-12:
            if abs(point[index])>extent/2+1e-12:
                return None
        else:
            ends=sorted(((-extent/2-point[index])/direction[index],(extent/2-point[index])/direction[index]))
            low,high=max(low,ends[0]),min(high,ends[1])
    return (low,high) if np.isfinite([low,high]).all() and low<=high else None


def _segment_cells(start,end,width_mm,height_mm):
    """Exact grid-crossing partition; no invented measurements between frames."""
    start,end=np.asarray(start,float)[:2],np.asarray(end,float)[:2]
    half=np.array([width_mm,height_mm])/2000.
    intervals=np.array([round(width_mm),round(height_mm)],int)
    delta=end-start
    parameters=[0.,1.]
    for axis in range(2):
        if abs(delta[axis])>1e-15:
            boundaries=(np.arange(intervals[axis])+.5)/1000.-half[axis]
            values=(boundaries-start[axis])/delta[axis]
            parameters.extend(values[(values>0)&(values<1)].tolist())
    parameters=np.unique(parameters)
    probes=np.r_[parameters,(parameters[:-1]+parameters[1:])/2]
    cells=set()
    for value in probes:
        xy=start+value*delta
        grid=(xy+half)*1000.
        choices=[]
        for axis in range(2):
            center=int(np.rint(grid[axis]))
            candidates={center}
            floor=int(np.floor(grid[axis]))
            if abs(grid[axis]-(floor+.5))<1e-8:
                candidates.update((floor,floor+1))
            choices.append([i for i in candidates if 0<=i<=intervals[axis]])
        for x in choices[0]:
            for y in choices[1]:
                cells.add(int(y*(intervals[0]+1)+x))
    return cells


def plan_footprint(calibration,baseline_optical,fixed_optical_rotation,*,reference_plane_z_m,
                   width_mm=100.,height_mm=100.,margin_px=12.,sample_step_m=.00025):
    """Plan carrier orientation/pitch from ACTIVE native calibration, in metres.

    ``laser_plane`` is n.dot(XYZ_mm)+D=0. Its offset alone is divided by1000.
    ``baseline_optical`` is the ORIGINAL measured Tube optical frame in base.
    A fixed optical roll must preserve that frame's optical Z direction.
    """
    baseline=_pose(baseline_optical)
    rotation=np.asarray(fixed_optical_rotation,float)
    test=np.eye(4)
    test[:3,:3]=rotation
    _pose(test)
    if not np.allclose(rotation[:,2],baseline[:3,2],atol=1e-6,rtol=0):
        raise ValueError('Fixed scan view must preserve original optical axis')
    matrix=np.asarray(calibration['camera_matrix'],float)
    distortion=np.asarray(calibration['distortion'],float)
    image=np.asarray(calibration['image_size'])
    plane=np.asarray(calibration['laser_plane'],float)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0
            or distortion.ndim!=1 or len(distortion) not in (4,5,8,12,14) or not np.isfinite(distortion).all()
            or image.shape!=(2,) or np.any(image<=0) or not np.equal(image,np.rint(image)).all()
            or plane.shape!=(4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3])<1e-9):
        raise ValueError('Footprint requires valid active camera/laser calibration')
    if (not np.isfinite([reference_plane_z_m,width_mm,height_mm,margin_px,sample_step_m]).all()
            or not 1<=width_mm<=100 or not 1<=height_mm<=100
            or width_mm!=round(width_mm) or height_mm!=round(height_mm)
            or not 0<sample_step_m<=.00025 or not 0<=margin_px<min(image)/2):
        raise ValueError('Invalid footprint ROI, sensor margin or native sample interval')
    normal=plane[:3]/np.linalg.norm(plane[:3])
    offset=plane[3]/1000./np.linalg.norm(plane[:3])
    calibration_id=hashlib.sha256(json.dumps(calibration,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    config=dict(schema='autoweld.reference-illumination.v1',baseline_optical=baseline.tolist(),
        camera_matrix=matrix.tolist(),distortion=distortion.tolist(),image_size=image.astype(int).tolist(),
        laser_plane_camera_m=[*normal.tolist(),float(offset)],laser_plane_native_mm=plane.tolist(),
        source_id=calibration.get('source_id'),calibration_id=calibration_id,
        fixed_optical_rotation=rotation.tolist(),reference_plane_z_m=float(reference_plane_z_m),
        width_mm=float(width_mm),height_mm=float(height_mm),sensor_margin_px=float(margin_px),
        sample_step_m=float(sample_step_m),standoff_m=.140,
        cell_definition='1mm squares centered on original ROI grid points; per-frame visible line intersection, no inter-frame interpolation',
        assumptions=dict(laser_assumed_on=True,reference_plane_is_measured_surface=False,
            native_xyz_is_unchanged=True,physical_sample_timing_qualified=False,firmware_sample_age_s=None))
    point,direction=_line(normal,offset,np.array([0.,0.,1.]),-.140)
    intervals=_visible_intervals(point,direction,-.5,.5,config)
    if not intervals:
        raise ValueError('Calibrated laser has no conservative sensor span at140mm')
    # The closest point on the infinite line to the optical axis must be in the
    # actual finite sensor segment, otherwise centering is not supported.
    containing=[interval for interval in intervals if interval[0]<=0<=interval[1]]
    if not containing:
        raise ValueError('Closest laser-line point to optical axis is outside sensor FoV')
    low,high=max(containing,key=lambda span:span[1]-span[0])
    if low<=-.5+sample_step_m or high>=.5-sample_step_m:
        raise ValueError('Laser FoV extends beyond bounded calibration search')
    relative=baseline[:3,:3].T@rotation
    endpoints=(point+np.array([low,high])[:,None]*direction)@relative.T
    closest=relative@point
    shift=-closest[:2]
    compensated=endpoints.copy()
    compensated[:,:2]+=shift
    compensated[:,2]+=reference_plane_z_m-.140
    span=np.abs(compensated[1,:2]-compensated[0,:2])
    stripe_major=int(np.argmax(span))
    sweep_axis='y' if stripe_major==0 else 'x'
    pitch=min(.020,float(span[stripe_major])/2)
    if pitch<.001:
        raise ValueError('Visible laser cross-lane span cannot support1mm coverage')
    lane_extent=(width_mm if stripe_major==0 else height_mm)/1000.
    lanes=int(math.ceil(lane_extent/pitch))+1
    pitch=lane_extent/(lanes-1)
    config.update(origin_compensation_xy_m=shift.tolist(),camera_xy_shift_m=shift.tolist(),
        closest_line_point_camera_m=point.tolist(),uncompensated_line_offset_xy_m=closest[:2].tolist(),
        camera_line_segment_m=(point+np.array([low,high])[:,None]*direction).tolist(),
        nominal_segment_original_m=compensated.tolist(),stripe_axis_xy=(relative@direction)[:2].tolist(),
        sweep_axis=sweep_axis,lane_spacing_m=pitch,lane_spacing_mm=pitch*1000.,lanes=lanes,
        projected_cross_lane_span_m=float(span[stripe_major]))
    return config


def validate_footprint_config(config):
    """Validate static persisted geometry before reuse/audit of measured poses."""
    if config.get('schema')!='autoweld.reference-illumination.v1':
        raise ValueError('Unsupported illumination footprint schema')
    native=np.asarray(config['laser_plane_native_mm'],float)
    plane=np.asarray(config['laser_plane_camera_m'],float)
    matrix=np.asarray(config['camera_matrix'],float)
    distortion=np.asarray(config['distortion'],float)
    image=np.asarray(config['image_size'],float)
    if (native.shape!=(4,) or plane.shape!=(4,) or not np.isfinite(np.r_[native,plane]).all()
            or np.linalg.norm(native[:3])<1e-9 or matrix.shape!=(3,3) or not np.isfinite(matrix).all()
            or matrix[0,0]<=0 or matrix[1,1]<=0 or distortion.ndim!=1
            or len(distortion) not in (4,5,8,12,14) or not np.isfinite(distortion).all()
            or image.shape!=(2,) or not np.isfinite(image).all() or np.any(image<=0)):
        raise ValueError('Invalid persisted camera/laser footprint geometry')
    expected=native.copy()
    expected[3]/=1000.
    expected/=np.linalg.norm(native[:3])
    if not np.allclose(expected,plane,atol=1e-12,rtol=0):
        raise ValueError('Footprint laser-plane units differ from original native calibration')
    if (not np.isfinite([config['sample_step_m'],config['reference_plane_z_m'],config['sensor_margin_px']]).all()
            or not 0<config['sample_step_m']<=.00025
            or not 0<=config['sensor_margin_px']<min(image)/2):
        raise ValueError('Invalid persisted footprint sampling or sensor margin')
    for key in ('width_mm','height_mm'):
        if not 1<=config[key]<=100 or config[key]!=round(config[key]):
            raise ValueError('Invalid persisted footprint ROI')
    _pose(config['baseline_optical'])


def footprint_cells(config,base_from_camera):
    """Actual measured-pose illumination on the fixed reference plane, not XYZ."""
    validate_footprint_config(config)
    baseline=_pose(config['baseline_optical'])
    camera=_pose(base_from_camera)
    relative=baseline[:3,:3].T@camera[:3,:3]
    translation=baseline[:3,:3].T@(camera[:3,3]-baseline[:3,3])
    laser=np.asarray(config['laser_plane_camera_m'],float)
    point,direction=_line(laser[:3],laser[3],relative[2,:],translation[2]-config['reference_plane_z_m'])
    original=relative@point+translation
    axis=relative@direction
    clip=_clip_xy(original,axis,config['width_mm']/1000.,config['height_mm']/1000.)
    segments=[]
    cells=set()
    if clip is not None:
        for low,high in _visible_intervals(point,direction,*clip,config):
            segment=original+np.array([low,high])[:,None]*axis
            segments.append(segment.tolist())
            cells.update(_segment_cells(segment[0],segment[1],config['width_mm'],config['height_mm']))
    return dict(schema='autoweld.reference-illumination-frame.v1',segments_original_m=segments,
        cells=sorted(cells),reference_plane_z_m=config['reference_plane_z_m'],
        coverage_basis='assumed_on_calibrated_laser_intersection_with_fixed_reference_plane_and_sensor_FoV',
        measured_surface_occupancy=False,physical_sample_timing_qualified=False)
