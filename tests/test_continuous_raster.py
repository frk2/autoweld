from dataclasses import replace

import numpy as np
import pytest
from i2rt.robots.waypoint_motion import TimedJointPath

from autoweld.continuous_raster import ContinuousRaster, RasterPlanBuilder, RasterSettings


def test_full_square_timing_coverage_and_envelope():
    raster = ContinuousRaster()
    assert raster.rows == raster.columns == 101
    assert 600 < raster.duration < 750
    events = raster.cell_events()
    assert len(events) == 10201
    assert len({tuple(e[3].round(9)) for e in events}) == 10201
    xy = np.array([e[3] for e in events])
    np.testing.assert_allclose(xy.min(axis=0), [-.05, -.05])
    np.testing.assert_allclose(xy.max(axis=0), [.05, .05])
    for row in range(101):
        row_events = events[row * 101:(row + 1) * 101]
        assert all(a[0] < b[0] for a, b in zip(row_events, row_events[1:]))
        for event in row_events[::10]:
            np.testing.assert_allclose(raster.sample(event[0])[0], event[3], atol=1e-10)
    # Only hairpins use the existing +1 mm guard; square coverage remains exact.
    for segment in raster.segments:
        if segment.kind == 'turn':
            points = np.array([segment.sample(t)[0] for t in np.linspace(segment.start, segment.start + segment.duration, 51)])
            assert np.max(np.abs(points[:, 0])) < .0505
            assert np.min(points[:, 1]) >= -.05 - 1e-12
            assert np.max(points[:, 1]) <= .05 + 1e-12


def test_all_joins_are_c3_without_intermediate_stops():
    raster = ContinuousRaster()
    for left, right in zip(raster.segments, raster.segments[1:]):
        a, b = left.sample(left.start + left.duration), right.sample(right.start)
        for derivative in range(4):
            np.testing.assert_allclose(a[derivative], b[derivative], atol=2e-11)
        assert np.linalg.norm(a[1]) >= .000999
    for t in (0, raster.duration):
        for derivative in raster.sample(t)[1:]:
            np.testing.assert_allclose(derivative, 0, atol=1e-11)
    for segment in raster.segments:
        if segment.kind == 'turn':
            for t in np.linspace(segment.start, segment.start + segment.duration, 101):
                assert np.linalg.norm(raster.sample(t)[1]) > .0008


def test_analytic_cartesian_derivatives_respect_installed_linear_limits():
    raster = ContinuousRaster()
    peaks = np.zeros(3)
    length = 0.
    for segment in raster.segments:
        times = np.linspace(segment.start, segment.start + segment.duration, 101)
        values = np.array([segment.sample(t) for t in times])
        peaks = np.maximum(peaks, np.max(np.linalg.norm(values[:, 1:], axis=2), axis=0))
        length += np.sum(np.linalg.norm(np.diff(values[:, 0], axis=0), axis=1))
    assert 10.1 < length < 10.4
    assert peaks[0] <= .0200000001
    assert peaks[1] < .15
    assert peaks[2] < 1.


def test_optical_xy_frame_remains_original_after_fixed_roll():
    raster = ContinuousRaster()
    original = np.eye(4)
    original[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    original[:3, 3] = [.2, .1, .3]
    rolled = np.eye(3)
    pose = raster.pose(0, original, rolled, .023)
    np.testing.assert_allclose(pose[:3, 3], [.25, .05, .323])
    np.testing.assert_array_equal(pose[:3, :3], rolled)


def test_bounded_lookahead_and_sampling_preserve_every_join():
    raster = ContinuousRaster()
    times = raster.sample_times(.05)
    assert np.max(np.diff(times)) <= .0500000001
    assert times[0] == 0 and times[-1] == raster.duration
    assert all(segment.start in times for segment in raster.segments)
    future_t, future_xy = raster.lookahead(10, 2., .05)
    assert len(future_t) <= 42
    assert future_t[0] == 10 and future_t[-1] == 12
    np.testing.assert_allclose(future_xy[0], raster.sample(10)[0])
    end_t, _ = raster.lookahead(raster.duration - .1, 2)
    assert end_t[-1] == raster.duration


def test_locally_timed_joint_path_preserves_nonuniform_feed_and_rest_endpoints():
    times = np.array([0, .2, .4, .6, 1., 2., 3., 4., 4.2, 4.4, 4.6, 5.])
    positions = np.column_stack((np.sin(times), times / 5, np.zeros((len(times), 5))))
    path = TimedJointPath(times / 5, positions, 5.)
    for t, expected in zip(times, positions):
        np.testing.assert_allclose(path.sample(t)[0], expected, atol=1e-12)
    for t in (0., 5.):
        for derivative in path.sample(t)[1:]:
            np.testing.assert_allclose(derivative, 0., atol=1e-10)
    for t in times[1:-1]:
        before, after = path.sample(t - 1e-8), path.sample(t + 1e-8)
        for derivative in range(4):
            np.testing.assert_allclose(before[derivative], after[derivative], atol=2e-4)
    six = replace(path, positions=path.positions[:, :6])
    assert six.sample(2.)[0].shape == (6,)
    np.testing.assert_allclose(six.sample(2.)[0], path.sample(2.)[0][:6])
    positions[:] = 123
    # SDK reference owns its data and cannot diverge from validated positions.
    assert not path.positions.flags.writeable
    assert not path.knots.flags.writeable
    with pytest.raises(ValueError):
        path.positions[:] = 456
    np.testing.assert_allclose(six.sample(2.)[0], path.sample(2.)[0][:6])
    assert np.max(path.sample(2.)[0]) < 2


@pytest.mark.parametrize('kwargs', [dict(spacing_m=0), dict(cruise_m_s=np.nan),
                                    dict(turn_m_s=.03), dict(width_m=.0005), dict(height_m=.1005)])
def test_invalid_geometry_rejected(kwargs):
    with pytest.raises(ValueError):
        RasterSettings(**kwargs)


def builder_fixture(**options):
    from autoweld.motion import MotionLimits

    settings = options.pop('settings', RasterSettings(width_m=.020, height_m=.001))
    raster = ContinuousRaster(settings)
    calls = {'solve': 0, 'validate': 0, 'collision': 0}

    def solve(pose, seed):
        calls['solve'] += 1
        q = seed.copy()
        q[:3] = pose[:3, 3]
        return q

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    def validate(q):
        calls['validate'] += 1

    def collision(q):
        calls['collision'] += 1
        return False

    defaults = dict(baseline_optical=np.eye(4), fixed_optical_rotation=np.eye(3),
                    tcp_from_camera=np.eye(4), solve=solve, fk=fk,
                    jacobian=lambda q: np.diag([1., 1., 1., 0., 0., 0.]),
                    bounds=np.tile([-3., 3.], (6, 1)), limits=MotionLimits(),
                    validate=validate, collision=collision)
    defaults.update(options)
    return RasterPlanBuilder(raster, np.zeros(7), **defaults), calls


def test_incremental_builder_full_interior_guards_and_derivatives():
    builder, calls = builder_fixture(depth=lambda x, y: .020 + .2 * x)
    first = builder.advance(max_samples=3, max_wall_s=1.)
    assert first['phase'] == 'ik'
    assert first['ik_index'] == 3
    assert calls['solve'] == 3
    assert builder.result is None
    result = builder.run()
    assert builder.state()['ready']
    assert isinstance(result.path, TimedJointPath)
    assert calls['solve'] == result.report['ik_knots']
    assert calls['collision'] == calls['validate']
    assert calls['validate'] == result.report['ik_knots'] + result.report['validation_samples']
    assert np.max(np.diff(builder.validation_times)) <= .01000000001
    assert result.report['tcp_path_error_m'] < .0002
    assert result.report['tcp_path_error_rad'] < .0002
    assert all(result.report['planned_peaks'][k] <= v * .95
               for k, v in result.report['limits'].items())
    for t in builder.validation_times[::31]:
        q = result.path.sample(t)[0]
        desired = builder.target(t / builder.scale)
        np.testing.assert_allclose(q[:3], desired[:3, 3], atol=1e-7)


def test_global_retiming_preserves_geometry_and_final_derivative_gates():
    from autoweld.motion import MotionLimits

    builder, calls = builder_fixture(limits=MotionLimits(joint_velocity=.010))
    result = builder.run()
    assert result.report['retime_factor'] > 2.
    assert result.report['retimes'] == 1
    assert result.report['geometric_validation_passes'] == 1
    assert result.report['validation_samples'] == len(builder.validation_times)
    assert calls['collision'] == result.report['ik_knots'] + result.report['validation_samples']
    assert result.report['planned_peaks']['joint_velocity'] < .010 * .95
    for fraction in np.linspace(0, 1, 35):
        q = result.path.sample(fraction * result.path.duration)[0]
        np.testing.assert_allclose(q[:2], builder.raster.sample(fraction * builder.raster.duration)[0], atol=1e-7)
    assert np.max(np.diff(builder.validation_times)) <= .010000001


def test_retiming_rejects_budget_without_partial_executable_result():
    from autoweld.motion import MotionLimits

    builder, _ = builder_fixture(limits=MotionLimits(joint_velocity=.010), max_duration_s=5.)
    with pytest.raises(ValueError, match='budget'):
        builder.run()
    assert builder.phase == 'error'
    assert builder.error['code'] == 'duration_budget'
    assert builder.result is None


def test_collision_between_ik_knots_is_rejected():
    # A narrow region excludes every IK knot, while an interior 10 ms sample hits it.
    reference, _ = builder_fixture()
    chosen = reference.times[len(reference.times) // 4]
    middle = chosen + .025
    bad_x = reference.target(middle)[0, 3]
    builder, _ = builder_fixture(collision=lambda q: abs(q[0] - bad_x) < .00004)
    assert all(abs(builder.target(t)[0, 3] - bad_x) > .00004 for t in builder.times)
    with pytest.raises(ValueError, match='collision'):
        builder.run()
    assert builder.error['code'] == 'self_collision'
    assert builder.phase == 'error' and builder.result is None


def test_spline_interior_residual_gate_does_not_accept_exact_knots_alone():
    builder, _ = builder_fixture(ik_period_s=.9)
    # Coarse exact IK misses rapid ramps/turn geometry; existing 0.2 mm gate rejects it.
    with pytest.raises(ValueError, match='residual'):
        builder.run()
    assert builder.error['code'] == 'tcp_residual'
    assert builder.result is None


def test_ik_branch_bounds_and_source_callback_failures_reject():
    builder, _ = builder_fixture()
    original = builder.solve

    def jumping(pose, seed):
        q = original(pose, seed)
        if builder.index:
            q[5] += .051
        return q

    builder.solve = jumping
    with pytest.raises(ValueError, match='continuity'):
        builder.run()
    assert builder.error['code'] == 'joint_step'
    builder, _ = builder_fixture(bounds=np.tile([-.005, .005], (6, 1)))
    with pytest.raises(ValueError, match='bounds'):
        builder.run()
    assert builder.error['code'] == 'joint_bounds'
    builder, _ = builder_fixture(solve=lambda pose, seed: np.full(7, np.nan))
    with pytest.raises(ValueError, match='Non-finite'):
        builder.run()
    assert builder.error['code'] == 'joint_nonfinite'


def test_depth_callback_has_no_invented_fallback_and_hard_envelope():
    for bad in (np.nan, .060001, -.061):
        builder, _ = builder_fixture(depth=lambda x, y: bad)
        with pytest.raises(ValueError, match='depth|Depth'):
            builder.run()
        assert builder.error['code'] == 'depth_envelope'
        assert builder.result is None
    builder, _ = builder_fixture(depth=lambda x, y: None)
    with pytest.raises(TypeError):
        builder.run()
    assert builder.phase == 'error' and builder.result is None


def test_cancelled_worker_never_exposes_partial_plan_or_keeps_solving():
    builder, calls = builder_fixture()
    builder.advance(max_samples=4, max_wall_s=1.)
    before = calls['solve']
    with pytest.raises(ValueError, match='cancelled'):
        builder.advance(cancelled=lambda: True)
    assert builder.phase == 'cancelled'
    assert builder.result is None
    builder.advance()
    assert calls['solve'] == before


def test_preflight_deadline_and_total_budget_are_enforced():
    builder, _ = builder_fixture(max_preflight_s=.001)
    builder.started -= 1.
    with pytest.raises(ValueError, match='preflight exceeded'):
        builder.run()
    assert builder.error['code'] == 'preflight_budget'
    builder, _ = builder_fixture(max_total_s=1.)
    with pytest.raises(ValueError, match='total budget'):
        builder.run()
    assert builder.error['code'] == 'total_budget'


def test_proper_transforms_and_validation_density_required():
    with pytest.raises(ValueError, match='10 ms'):
        builder_fixture(validation_period_s=.011)
    with pytest.raises(ValueError, match='proper'):
        builder_fixture(tcp_from_camera=np.zeros((4, 4)))
    with pytest.raises(ValueError, match='proper'):
        builder_fixture(fixed_optical_rotation=np.diag([1., 1., -1.]))


def test_sdk_dispatch_keeps_local_clock_and_immutable_reference(monkeypatch):
    import time

    from test_integral import sdk_stub

    clock = [100.]
    monkeypatch.setattr('i2rt.robots.motor_chain_robot.time.monotonic', lambda: clock[0])
    robot = sdk_stub()
    times = np.linspace(0, 2, 31)
    positions = np.zeros((len(times), 7))
    positions[:, 0] = .003 * np.sin(times * np.pi / 2)
    path = TimedJointPath(times / 2, positions, 2.)
    robot.command_joint_motion(path, np.zeros(7), np.ones(7))
    robot.update()
    assert isinstance(robot._joint_motion[0], TimedJointPath)
    assert robot._joint_motion[0] is not path
    assert not robot._joint_motion[0].positions.flags.writeable
    for i in range(1, 201):
        clock[0] = 100 + i * .01
        robot._joint_state.timestamp = time.time()
        robot.update()
        np.testing.assert_allclose(robot.sent[1].pos, path.sample(i * .01)[0], atol=1e-12)
        np.testing.assert_allclose(robot.sent[1].vel, path.sample(i * .01)[1], atol=1e-12)
    assert robot.get_motion_fault() is None


def test_spline_overshoot_cannot_escape_installed_bounds_between_knots():
    builder, _ = builder_fixture(bounds=np.tile([-.050, .050], (6, 1)))
    original = builder.solve

    def alternating(pose, seed):
        q = original(pose, seed)
        q[5] = .049 if builder.index % 2 else 0.
        return q

    builder.solve = alternating
    with pytest.raises(ValueError, match='bounds'):
        builder.run()
    assert builder.error['code'] == 'joint_bounds'
    assert builder.phase == 'error' and builder.result is None
    assert all(0 <= q[5] <= .049 for q in builder.positions)


def test_nonconstant_jacobian_chain_rule_and_fk_residual():
    builder, _ = builder_fixture()
    original = builder.solve

    def solve(pose, seed):
        q = original(pose, seed)
        q[0] = np.arcsin(q[0])
        return q

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = [np.sin(q[0]), q[1], q[2]]
        return pose

    builder.solve, builder.fk = solve, fk
    builder.jacobian = lambda q: np.diag([np.cos(q[0]), 1., 1., 0., 0., 0.])
    result = builder.run()
    assert result.report['tcp_path_error_m'] < 1e-7
    assert result.report['planned_peaks']['linear_velocity'] == pytest.approx(.020, abs=1e-6)
    assert result.report['planned_peaks']['linear_acceleration'] == pytest.approx(.07291666, abs=2e-4)
    assert result.report['planned_peaks']['linear_jerk'] == pytest.approx(.4174, abs=.001)


def test_optional_30mm_s_profile_shortens_budget_without_raising_ramp_limits():
    # Candidate only: live native frame footprint/coverage and installed model
    # validation must authorize selecting this feed; default remains20mm/s.
    raster = ContinuousRaster(RasterSettings(cruise_m_s=.030, ramp_s=.900))
    assert 500 < raster.duration < 510
    peaks = np.zeros(3)
    for segment in (raster.segments[0], raster.segments[1], raster.segments[2],
                    raster.segments[3], raster.segments[-1]):
        samples = np.array([segment.sample(t) for t in np.linspace(segment.start, segment.start + segment.duration, 301)])
        peaks = np.maximum(peaks, np.max(np.linalg.norm(samples[:, 1:], axis=2), axis=0))
    assert peaks[0] <= .030000001
    assert peaks[1] < .073
    assert peaks[2] < .279
    assert RasterSettings().cruise_m_s == .020


def test_from_planner_reuses_given_single_model_kinematics_without_construction():
    from types import SimpleNamespace

    template, _ = builder_fixture()
    calls = []

    def ik(target, site, *, init_q, **kwargs):
        calls.append((site, kwargs))
        q = init_q.copy()
        q[:3] = target[:3, 3]
        return True, q

    kin = SimpleNamespace(ik=ik)
    planner = SimpleNamespace(kin=kin, site_name='weld_tcp', limits=template.limits,
                              _configuration=lambda q: q[:6].copy(),
                              fk=template.fk, jacobian=template.jacobian)
    builder = RasterPlanBuilder.from_planner(
        planner, template.raster, np.zeros(7), baseline_optical=np.eye(4),
        fixed_optical_rotation=np.eye(3), tcp_from_camera=np.eye(4), bounds=template.bounds)
    result = builder.run()
    assert len(calls) == result.report['ik_knots']
    assert all(site == 'weld_tcp' for site, _ in calls)
    assert all(options['pos_threshold'] == 1e-8 and options['ori_threshold'] == 1e-8 for _, options in calls)
    assert planner.kin is kin


def test_global_duration_change_preserves_exact_spline_geometry_and_scales_derivatives():
    times = np.linspace(0., 1., 31)
    positions = np.c_[np.sin(times), times, np.zeros((len(times), 5))]
    original = TimedJointPath(times, positions, 3.)
    factor = 1.10487465
    stretched = replace(original, duration=original.duration * factor)
    np.testing.assert_array_equal(original._spline.c, stretched._spline.c)
    for fraction in np.linspace(0., 1., 101):
        before = original.sample(fraction * original.duration)
        after = stretched.sample(fraction * stretched.duration)
        for order in range(4):
            np.testing.assert_allclose(after[order], before[order] / factor**order, atol=2e-12)


def test_budget_reserved_samples_allow_retime_without_second_geometry_pass():
    from autoweld.motion import MotionLimits

    builder, calls = builder_fixture(limits=MotionLimits(joint_velocity=.0195))
    plan = builder.run()
    assert 1 < plan.report['retime_factor'] < 2
    assert plan.report['retimes'] == 1
    assert plan.report['geometric_validation_passes'] == 1
    assert builder.validation_count == len(builder.validation_times)
    assert calls['collision'] == builder.index + len(builder.validation_times)
    assert np.max(np.diff(builder.validation_times)) <= .01 + 1e-12


def test_line_footprint_lanes_are_separate_from_one_mm_roi_grid():
    raster = ContinuousRaster(RasterSettings(lane_spacing_m=.020))
    assert raster.rows == 6
    assert raster.coverage_rows == raster.coverage_columns == 101
    assert raster.lane_pitch_m == pytest.approx(.020)
    events = raster.cell_events()
    assert len(events) == 606  # Carrier events, not fabricated10201 views.
    np.testing.assert_allclose(sorted({round(e[3][1], 9) for e in events}), np.linspace(-.05, .05, 6))
    assert 40 < raster.duration < 50
    for left, right in zip(raster.segments, raster.segments[1:]):
        for a, b in zip(left.sample(left.start + left.duration), right.sample(right.start)):
            np.testing.assert_allclose(a, b, atol=2e-11)
    for segment in raster.segments:
        if segment.kind == 'turn':
            values = np.array([segment.sample(t) for t in np.linspace(segment.start, segment.start + segment.duration, 101)])
            assert np.max(np.linalg.norm(values[:, 1], axis=1)) <= .020001
            assert np.max(abs(values[:, 0, 0])) <= .051
            assert np.min(np.linalg.norm(values[:, 1], axis=1)) > 0


def test_line_lane_pitch_rounds_down_to_preserve_maximum_measured_overlap_spacing():
    raster = ContinuousRaster(RasterSettings(lane_spacing_m=.019))
    assert raster.rows == 7
    assert raster.lane_pitch_m <= .019
    assert raster.lane_pitch_m == pytest.approx(.1 / 6)
    with pytest.raises(ValueError, match='20 mm'):
        RasterSettings(lane_spacing_m=.020001)
