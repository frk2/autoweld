"""Hardware-free proof of true world-line geometry and inherited protections."""
from dataclasses import replace

import numpy as np
import pytest
from i2rt.robots.waypoint_motion import TimedJointPath
from scipy.spatial.transform import Rotation

from autoweld.continuous_raster import raster_path_digest
from autoweld.motion import MotionLimits
from autoweld.vertical_scan import VerticalLine, WorldLinePlanBuilder, build_vertical_plan


def line_builder(**kwargs):
    start = np.eye(4)
    start[:3, :3] = Rotation.from_euler('xyz', [31., -22., 43.], degrees=True).as_matrix()
    start[:3, 3] = [.2, -.1, .4]
    handeye = np.eye(4)
    handeye[:3, :3] = Rotation.from_euler('y', .2).as_matrix()
    handeye[:3, 3] = [.01, -.02, .04]
    seed = np.r_[(start @ np.linalg.inv(handeye))[:3, 3],
                 Rotation.from_matrix((start @ np.linalg.inv(handeye))[:3, :3]).as_rotvec(), .4]
    calls = {'solve': 0, 'collision': 0, 'validate': 0}

    def solve(pose, prior):
        calls['solve'] += 1
        return np.r_[pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec(), prior[6]]

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def collision(q):
        calls['collision'] += 1
        return False

    def validate(q):
        calls['validate'] += 1

    options = dict(baseline_optical=start, fixed_optical_rotation=start[:3, :3],
                   tcp_from_camera=handeye, solve=solve, fk=fk, jacobian=lambda q: np.eye(6),
                   bounds=np.tile([-3., 3.], (6, 1)), collision=collision, validate=validate)
    options.update(kwargs)
    builder = WorldLinePlanBuilder(VerticalLine(), seed, **options)
    return builder, start, handeye, calls


@pytest.fixture(scope='module')
def validated_line():
    builder, start, handeye, calls = line_builder()
    return builder.run(), builder, start, handeye, calls


def test_line_is_monotonic_c3_rest_profile_with_requested_peak_feed():
    line = VerticalLine()
    assert line.duration == pytest.approx(17.5)
    times = line.sample_times(.05)
    assert np.max(np.diff(times)) <= .05000000001
    fraction = np.array([line.fraction(t) for t in times])
    assert fraction[0] == 0. and fraction[-1] == 1.
    assert np.all(np.diff(fraction) > 0.)
    # Analytic derivatives of the C3 smoothstep vanish at both endpoints.
    from numpy.polynomial import Polynomial
    p = Polynomial([0, 0, 0, 0, 35, -84, 70, -20])
    for order in (1, 2, 3):
        np.testing.assert_allclose(p.deriv(order)([0., 1.]), 0., atol=1e-12)
    assert p.deriv()(0.5) * line.length_m / line.duration == pytest.approx(.010)


def test_every_joint_knot_and_dense_interior_follows_world_minus_z_with_fixed_view(validated_line):
    plan, builder, start, handeye, calls = validated_line
    assert type(plan.path) is TimedJointPath
    assert plan.report['kind'] == 'Continuous laser vertical line'
    assert plan.report['station_count'] == 81
    scope = plan.report['scan_scope']
    assert scope['direction_base'] == [0., 0., -1.]
    np.testing.assert_allclose(np.asarray(scope['end_base_from_camera'])[:3, 3], [.2, -.1, .32])
    assert plan.report['validation_proof']['schema'] == 'autoweld.continuous_world_line.v1'
    assert plan.report['validation_proof']['path_sha256'] == raster_path_digest(plan.path)
    assert plan.report['validation_actual_max_period_s'] <= .01
    assert calls['collision'] == calls['validate']
    assert calls['collision'] >= len(plan.path.knots) + plan.report['validation_samples']
    assert 'coverage_cells' not in plan.report
    for t in np.unique(np.r_[builder.validation_times, plan.path.knots * plan.path.duration]):
        q = plan.path.position(t)
        camera = builder.fk(q) @ handeye
        np.testing.assert_allclose(camera[:2, 3], start[:2, 3], atol=2e-10)
        np.testing.assert_allclose(camera[:3, :3], start[:3, :3], atol=2e-10)
        expected = start[2, 3] - .08 * builder.raster.fraction(t / builder.scale)
        assert camera[2, 3] == pytest.approx(expected, abs=2e-7)
        assert q[6] == pytest.approx(.4)
    assert np.max(np.abs(np.diff(plan.path.positions[:, :6], axis=0))) <= .05
    for key, limit in vars(MotionLimits()).items():
        assert np.isfinite(plan.report['planned_peaks'][key])
        assert plan.report['planned_peaks'][key] <= limit * .95


def test_interior_collision_and_tcp_deviation_reject_before_result():
    builder, *_ = line_builder()
    while builder.phase == 'ik':
        builder.advance(max_samples=1)
    builder.collision = lambda q: True
    with pytest.raises(Exception, match='self-collision'):
        builder.advance(max_samples=1)
    assert builder.result is None

    builder, *_ = line_builder()
    fk = builder.fk
    def displaced_fk(q):
        pose = fk(q)
        pose[0, 3] += .0003
        return pose
    builder.fk = displaced_fk
    with pytest.raises(Exception, match='residual'):
        builder.run()
    assert builder.result is None


def test_all_nine_limits_retime_and_budget_remains_enforced():
    limits = replace(MotionLimits(), linear_velocity=.005)
    builder, *_ = line_builder(limits=limits)
    plan = builder.run()
    assert plan.report['retime_factor'] > 2.
    assert plan.report['planned_peaks']['linear_velocity'] < .005
    assert plan.report['validation_actual_max_period_s'] <= .01 + 1e-12
    builder, *_ = line_builder(max_duration_s=10.)
    with pytest.raises(Exception, match='budget'):
        builder.run()
    assert builder.result is None


def test_cancel_is_checked_before_solving():
    builder, _, _, calls = line_builder()
    with pytest.raises(Exception, match='cancelled'):
        builder.run(cancelled=lambda: True)
    assert calls['solve'] == 0 and builder.result is None


@pytest.mark.parametrize('kwargs', [dict(length_m=0), dict(length_m=.101),
                                    dict(feed_m_s=0), dict(feed_m_s=np.nan)])
def test_bad_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        VerticalLine(**kwargs)


def test_line_cannot_be_deformed_or_have_a_different_start_rotation():
    with pytest.raises(ValueError, match='deformation'):
        line_builder(depth=lambda x, y: 0.)
    with pytest.raises(ValueError, match='frozen start camera rotation'):
        line_builder(fixed_optical_rotation=np.eye(3))


def test_helper_rejects_invalid_inputs_without_a_planner_or_model():
    args = dict(start_base_from_camera=np.eye(4), tcp_from_camera=np.eye(4),
                seed_q=np.zeros(6), bounds=np.tile([-3., 3.], (6, 1)))
    with pytest.raises(ValueError, match='bounds'):
        build_vertical_plan(None, **{**args, 'bounds': [[0., 0.]] * 6})
    with pytest.raises(ValueError, match='gripper'):
        build_vertical_plan(None, **{**args, 'seed_q': np.r_[np.zeros(6), .5]})
    with pytest.raises(ValueError, match='seed exceeds'):
        build_vertical_plan(None, **{**args, 'seed_q': np.full(6, 4.)})
