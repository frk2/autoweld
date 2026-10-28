"""Analytic calibration geometry, no hardware or SDK ownership."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.scan_footprint import footprint_cells, plan_footprint


def calibration():
    return dict(source_id='camera',camera_matrix=[[4000.,0.,719.5],[0.,4000.,539.5],[0.,0.,1.]],
        distortion=[0.]*5,image_size=[1440,1080],laser_plane=[1.,0.,.2,-20.])


def config(rotation=None,baseline=None):
    baseline=np.eye(4) if baseline is None else baseline
    rotation=baseline[:3,:3] if rotation is None else rotation
    return plan_footprint(calibration(),baseline,rotation,reference_plane_z_m=.147)


def camera_pose(cfg,xy=(0.,0.),rotation=None):
    base=np.asarray(cfg['baseline_optical'])
    pose=np.eye(4)
    pose[:3,:3]=np.asarray(cfg['fixed_optical_rotation']) if rotation is None else rotation
    pose[:3,3]=base[:3,3]+base[:3,:3]@np.r_[np.asarray(cfg['origin_compensation_xy_m'])+xy,.007]
    return pose


def test_plane_units_negative_offset_compensation_and_center_are_exact():
    cfg=config()
    np.testing.assert_allclose(cfg['closest_line_point_camera_m'],[-.008,0.,.14],atol=1e-12)
    np.testing.assert_allclose(cfg['origin_compensation_xy_m'],[.008,0.],atol=1e-12)
    assert cfg['sweep_axis']=='x' and 6<=cfg['lanes']<=8
    assert .001<=cfg['lane_spacing_m']<=.020
    assert cfg['lane_spacing_m']<=cfg['projected_cross_lane_span_m']/2
    result=footprint_cells(cfg,camera_pose(cfg))
    assert 50*101+50 in result['cells']
    assert not result['measured_surface_occupancy']
    for segment in result['segments_original_m']:
        np.testing.assert_allclose(np.asarray(segment)[:,2],.147,atol=1e-12)


def test_roll_swaps_sweep_axis_preserves_original_roi_and_reference_plane():
    baseline=np.eye(4)
    baseline[:3,:3]=Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix()
    baseline[:3,3]=[.2,-.3,.1]
    rolled=baseline[:3,:3]@Rotation.from_euler('z',90,degrees=True).as_matrix()
    cfg=config(rolled,baseline)
    assert cfg['sweep_axis']=='y'
    np.testing.assert_allclose(cfg['baseline_optical'],baseline)
    assert 5100 in footprint_cells(cfg,camera_pose(cfg))['cells']


def test_actual_measured_pose_changes_illumination_and_sensor_bounds_limit_ends():
    cfg=config()
    one=footprint_cells(cfg,camera_pose(cfg))
    shifted=footprint_cells(cfg,camera_pose(cfg,xy=(.01,0.)))
    assert one['cells']!=shifted['cells']
    assert np.ptp(np.asarray(one['segments_original_m'])[0,:,1])<.04
    assert not footprint_cells(cfg,camera_pose(cfg,xy=(.2,0.)))['cells']
    tilted=Rotation.from_euler('x',1,degrees=True).as_matrix()
    actual=footprint_cells(cfg,camera_pose(cfg,rotation=tilted))
    assert actual['cells']!=one['cells']
    np.testing.assert_allclose(np.asarray(actual['segments_original_m'])[:,:,2],.147,atol=1e-12)


def test_1mm_illumination_coverage_uses_line_not_camera_origin_rows():
    cfg=config()
    seen=set()
    # Seven stripe lanes, each carrier moves along X; no101 origin rows.
    lanes=np.linspace(-.05,.05,cfg['lanes'])
    for lane in lanes:
        for sweep in np.arange(-.05,.05001,.001):
            seen.update(footprint_cells(cfg,camera_pose(cfg,xy=(sweep,lane)))['cells'])
    assert len(seen)==10201


def test_active_distortion_and_sensor_margin_reduce_supported_span():
    native=calibration()
    native['distortion']=[-.1,.01,.002,-.001,0.]
    small=plan_footprint(native,np.eye(4),np.eye(3),reference_plane_z_m=.14,margin_px=12.)
    large=plan_footprint(native,np.eye(4),np.eye(3),reference_plane_z_m=.14,margin_px=100.)
    assert large['projected_cross_lane_span_m']<small['projected_cross_lane_span_m']


def test_degenerate_or_outside_sensor_line_is_rejected():
    native=calibration()
    native['laser_plane']=[0.,0.,1.,-140.]
    with pytest.raises(ValueError,match='unique line'):
        plan_footprint(native,np.eye(4),np.eye(3),reference_plane_z_m=.14)
    native['laser_plane']=[1.,0.,0.,-200.]
    with pytest.raises(ValueError,match='no conservative sensor span'):
        plan_footprint(native,np.eye(4),np.eye(3),reference_plane_z_m=.14)
