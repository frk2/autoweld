"""Tracking geometry, timing, feed preservation, SDK execution and measured criteria."""

import json
from pathlib import Path

import numpy as np
import pytest
from i2rt.robots.waypoint_motion import FeedJointPath
from test_control import FakeRobot
from test_integral import sdk_stub

from autoweld.control import Controller
from autoweld.models import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import TrackCurve, TrackingMonitor, load_tracks, plan_tracking, tracking_sample


@pytest.fixture(scope="module")
def planner():
    model = tool_model()
    yield MotionPlanner(model)


@pytest.fixture(scope="module")
def line_plan(planner):
    seed = np.array(json.loads(Path("assets/stage3/manifest.json").read_text())[0]["seed_q"])
    return plan_tracking(planner, seed, load_tracks("assets/stage3/lines.step")[0])


def test_feed_timing_derivatives_and_ramp_boundaries():
    path = FeedJointPath([0, 1], [[0.0], [1.0]], np.float64(6.0), np.float64(1 / 6))
    assert type(path.duration) is float and type(path.ramp_fraction) is float
    h = 1e-5
    for t in np.linspace(0.01, 5.99, 51):
        analytic = np.array(path.sample(t)[1:])
        numeric = (np.array(path.sample(t + h)[:3]) - np.array(path.sample(t - h)[:3])) / (2 * h)
        np.testing.assert_allclose(analytic, numeric, atol=1e-7)
    for t in [-1, 0, 6, 7]:
        np.testing.assert_allclose(path.sample(t)[1:], 0, atol=1e-10)
    for t in [1, 5]:
        np.testing.assert_allclose(path.sample(t - 1e-8), path.sample(t + 1e-8), atol=1e-6)
    for t in [1, 2, 3, 4, 5]:
        assert path.sample(t)[1][0] == pytest.approx(0.2)


def test_step_fixture_geometry():
    lines = load_tracks("assets/stage3/lines.step")
    circles = load_tracks("assets/stage3/circles.step")
    arcs = load_tracks("assets/stage3/quarter-arcs.step")
    assert len(lines) == len(circles) == len(arcs) == 5
    for line, circle, arc in zip(lines, circles, arcs):
        assert line.length == pytest.approx(0.02)
        assert circle.radius == pytest.approx(0.0127)
        assert circle.length == pytest.approx(2 * np.pi * 0.0127)
        assert arc.sweep == pytest.approx(np.pi / 2)
        assert circle.distance(circle.point(0.37)) < 1e-9
        assert line.distance(line.point(1.5)) == pytest.approx(0.01)
        for s in [0, 0.3, 1]:
            R = arc.pose(s, np.eye(3), "Center facing")[:3, :3]
            np.testing.assert_allclose(-R[:, 2], (arc.center - arc.point(s)) / arc.radius, atol=1e-10)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)
            R = line.pose(s, np.eye(3), "Perpendicular")[:3, :3]
            assert abs(np.dot(-R[:, 2], line.direction)) < 1e-10


def test_requested_line_has_constant_feed_and_accurate_reference(planner, line_plan):
    plan = line_plan
    for t in np.linspace(plan.ramp, plan.motion.path.duration - plan.ramp, 211):
        q, v, _, _ = plan.motion.path.sample(t)
        row = tracking_sample(plan, planner, q, v, t)
        assert row["tracking_error_mm"] < 0.01
        assert row["speed_m_s"] == pytest.approx(plan.feed, abs=1e-6)
    assert plan.curve_fraction(plan.ramp) == pytest.approx(0)
    assert plan.curve_fraction(plan.motion.path.duration - plan.ramp) == pytest.approx(1)
    for k, peak in plan.motion.report["planned_peaks"].items():
        assert peak <= plan.motion.report["limits"][k]


def test_position_velocity_compares_reference_at_its_own_time(planner, line_plan):
    monitor = TrackingMonitor(line_plan)
    for elapsed in np.arange(.02, .42, .02):
        q, v, _, _ = line_plan.motion.path.sample(elapsed)
        row = tracking_sample(line_plan, planner, q, v, elapsed)
        monitor.add_velocity_diagnostics(row, planner, q, np.full(6, 100+elapsed), 100+elapsed+.004)
    estimate = row['position_velocity']
    assert estimate is not None
    assert estimate['elapsed_s'] < row['elapsed_s']-.03
    assert estimate['reference_speed_m_s'] == pytest.approx(
        line_plan.motion.path.phase(estimate['elapsed_s'])[1]
        * (line_plan.curve.length+2*line_plan.lead))
    assert estimate['reference_speed_m_s'] < line_plan.motion.path.phase(elapsed)[1]*(line_plan.curve.length+2*line_plan.lead)
    expected = planner.jacobian(estimate['joint_position_rad'])[:3]@estimate['joint_velocity_rad_s']
    assert estimate['speed_m_s'] == pytest.approx(np.linalg.norm(expected))


def test_invalid_velocity_timing_cannot_qualify(planner, line_plan):
    monitor = TrackingMonitor(line_plan)
    t = line_plan.ramp+.1
    q,v,_,_=line_plan.motion.path.sample(t)
    for stamps,now in [(np.full(6,100.),100.001),(np.full(6,100.),100.002),
                       (100.1+np.arange(6)*.003,100.12),(np.full(6,100.2),100.26)]:
        row=tracking_sample(line_plan,planner,q,v,t)
        monitor.add_velocity_diagnostics(row,planner,q,stamps,now)
        monitor.rows.append(row)
    assert monitor.rows[0]['velocity_sample_valid']
    assert all(not r['velocity_sample_valid'] and not r['speed_pass'] for r in monitor.rows[1:])
    assert monitor.summary()['invalid_velocity_samples']==3
    assert not monitor.summary()['sampling_ok']


def test_tcp_jacobian_matches_position_difference_across_poses(planner):
    for q in np.random.default_rng(31).uniform(-.3,.3,(20,6)):
        J=planner.jacobian(q)[:3]
        numeric=np.column_stack([(planner.fk(q+np.eye(6)[j]*1e-6)[:3,3]
                                 -planner.fk(q-np.eye(6)[j]*1e-6)[:3,3])/2e-6 for j in range(6)])
        np.testing.assert_allclose(J,numeric,atol=2e-9)


def test_inside_ramps_stay_on_finite_edge_and_score_only_constant_feed(planner):
    import copy
    from dataclasses import replace
    planner = copy.copy(planner)
    planner.limits = replace(planner.limits, linear_velocity=.02)
    seed = np.array(json.loads(Path("assets/stage3/manifest.json").read_text())[0]["seed_q"])
    original = load_tracks("assets/stage3/lines.step")[0]
    curve = replace(original, start=original.point(.5)-.04*original.direction, length=.08)
    plan = plan_tracking(planner, seed, curve, .02, ramp_placement="Inside edge")
    assert plan.lead == 0
    assert plan.curve_fraction(0) == pytest.approx(0)
    assert plan.curve_fraction(plan.motion.path.duration) == pytest.approx(1)
    monitor = TrackingMonitor(plan)
    for t in np.linspace(0, plan.motion.path.duration+1, 1001):
        q, v, _, _ = plan.motion.path.sample(t)
        row = tracking_sample(plan, planner, q, v, t)
        assert row["cross_track_mm"] < .01
        if row["on_edge"]:
            assert row["speed_m_s"] == pytest.approx(.02, abs=1e-6)
        monitor.rows.append(row)
    result = monitor.summary(completed=True)
    assert result["passed"]
    json.dumps(result, allow_nan=False)
    assert 0 < result["feed_interval_edge_fraction"][0] < result["feed_interval_edge_fraction"][1] < 1
    with pytest.raises(ValueError, match="Ramp placement"):
        plan_tracking(planner, seed, curve, ramp_placement="bad")
    with pytest.raises(ValueError, match="too short"):
        plan_tracking(planner, seed, replace(curve, length=.001), .02, ramp_placement="Inside edge")


def test_numpy_ramp_summary_is_json_serializable(line_plan):
    from dataclasses import replace
    plan = replace(line_plan, ramp=np.float64(line_plan.ramp))
    monitor = TrackingMonitor(plan)
    monitor.rows = [{"elapsed_s": float(t), "edge_fraction":float(plan.curve_fraction(t)),
        "on_edge": bool(plan.ramp <= t <= plan.motion.path.duration-plan.ramp),
        "tracking_error_mm":0.,"cross_track_mm":0.,"orientation_error_deg":0.,
        "speed_m_s":plan.feed,"speed_pass":True,"orientation_pass":True}
        for t in np.linspace(0,plan.motion.path.duration+1,1001)]
    result=monitor.summary(completed=True)
    assert result['passed']
    assert type(result['edge_coverage']) is bool
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('mode', ['Path tracking', 'Returning to rest'])
def test_reporting_exception_never_blocks_hold_or_rest_disable(mode):
    from types import SimpleNamespace

    from autoweld.legacy_app import App
    app=object.__new__(App)
    calls=[]
    app.ctrl=SimpleNamespace(mode=mode,closed=False,hold=lambda:calls.append('hold'),
                             tick=lambda:calls.append('tick'))
    app.tracking_monitor=object()

    def fail():
        raise TypeError('serialization')

    app.update_tracking=fail
    with pytest.raises(TypeError,match='serialization'):
        app.tick_motion()
    assert calls==(['hold','tick'] if mode=='Path tracking' else ['tick'])
    assert app.tracking_monitor is None
    assert not app.tracking_result['running']
    json.dumps(app.tracking_result)


def test_monitor_never_calls_incomplete_or_overspeed_trace_passed(planner, line_plan):
    monitor = TrackingMonitor(line_plan)
    for t in np.linspace(0, line_plan.motion.path.duration + 1.0, 501):
        q, v, _, _ = line_plan.motion.path.sample(t)
        monitor.rows.append(tracking_sample(line_plan, planner, q, v, t))
    assert monitor.summary(completed=True)["passed"]
    assert not monitor.summary()["passed"]
    next(r for r in monitor.rows if r["on_edge"])["speed_pass"] = False
    assert not monitor.summary(completed=True)["passed"]


def test_sdk_preserves_feed_timing_and_copies_input():
    r = sdk_stub()
    path = FeedJointPath([0, 1], [np.zeros(7), np.r_[0.02, np.zeros(6)]], 3.0, 0.2)
    r.command_joint_motion(path, np.ones(7), np.ones(7))
    copied = r._joint_motion[0]
    assert isinstance(copied, FeedJointPath)
    np.testing.assert_allclose(copied.sample(1)[1], path.sample(1)[1])
    path.positions[0, 0] = 20
    assert copied.positions[0, 0] == 0


def test_tracking_needs_approach_and_preserves_automatic_bounds(planner, line_plan):
    robot = FakeRobot()
    robot.q[6] = line_plan.motion.path.positions[0, 6]
    c = Controller(robot, np.tile([-3, 3], (6, 1)))
    c.motion_planner = planner
    with pytest.raises(ValueError, match="Move to path start"):
        c.command_tracking(line_plan)
    assert not robot.commands
    robot.q = line_plan.motion.path.positions[0].copy()
    c.command_tracking(line_plan)
    assert c.mode == "Path tracking"
    assert c.motion_plan is line_plan.motion

    c.motion_plan = None
    c.live = True
    c.health = lambda: None
    c.ready_for_waypoint = lambda: True
    c.rest[0] = robot.q[0] - 0.5
    with pytest.raises(ValueError, match="0.49"):
        c.command_tracking(line_plan, automatic=True)
    assert not robot.commands


def test_tracking_preview_pins_captured_gripper_despite_encoder_noise(planner, line_plan, monkeypatch):
    from types import SimpleNamespace

    from autoweld.legacy_app import App

    robot = FakeRobot()
    robot.q[6] = 0.5
    c = Controller(robot, np.tile([-3, 3], (6, 1)))
    c.motion_planner = planner
    c.motion_collision = None
    robot.q[6] = 0.501  # The measured gripper must not become a new commanded target.
    app = object.__new__(App)
    app.ctrl = c
    app.tracks = [line_plan.curve]
    app.tracking_monitor = None
    app.track_select = SimpleNamespace(value="1: line")
    app.track_feed = SimpleNamespace(value=5.0)
    app.track_orientation = SimpleNamespace(value="Unconstrained")
    app.server = SimpleNamespace(scene=SimpleNamespace(add_line_segments=lambda *a, **kw: None))

    def capture(_planner, seed, *args, **kwargs):
        assert seed[6] == c.rest[6] == 0.5
        return line_plan

    monkeypatch.setattr("autoweld.legacy_app.plan_tracking", capture)
    app.tracking_request({"action": "track_preview"}, False)
    assert app.tracking_preview[1] is line_plan
    assert not robot.commands


@pytest.mark.parametrize("error_mm, accepted", [(9.999, True), (10.0, True), (10.001, False)])
def test_global_ten_mm_boundary_used_for_sample_and_summary(line_plan, error_mm, accepted):
    from types import SimpleNamespace

    from autoweld.tracking_limits import POSITION_TOLERANCE_M

    assert POSITION_TOLERANCE_M == 0.010
    from dataclasses import replace

    curve = TrackCurve("line", np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.02)
    line_plan = replace(line_plan, curve=curve)
    actual = np.eye(4)
    actual[:3, 3] = [error_mm / 1000, 0.01, 0]
    planner = SimpleNamespace(fk=lambda q: actual, jacobian=lambda q: np.zeros((6, 6)))
    row = tracking_sample(line_plan, planner, np.zeros(7), np.zeros(7), 0)
    assert row["position_pass"] is accepted
    assert row["position_tolerance_mm"] == 10
    monitor = TrackingMonitor(line_plan)
    monitor.rows = [
        {
            "elapsed_s": float(t),
            "edge_fraction": float(np.clip(line_plan.curve_fraction(t), 0, 1)),
            "on_edge": line_plan.ramp <= t <= line_plan.motion.path.duration - line_plan.ramp,
            "tracking_error_mm": 100.0,  # Along-path timing error never determines qualification.
            "cross_track_mm": error_mm,
            "orientation_error_deg": 0.0,
            "speed_m_s": line_plan.feed,
            "speed_pass": True,
            "orientation_pass": True,
        }
        for t in np.linspace(0, line_plan.motion.path.duration + 1.0, 601)
    ]
    assert monitor.summary(completed=True)["passed"] is accepted


@pytest.mark.parametrize("error_mm, accepted", [(9.999, True), (10.0, True), (10.001, False)])
def test_path_start_uses_same_ten_mm_boundary(line_plan, error_mm, accepted):
    from types import SimpleNamespace

    robot = FakeRobot()
    robot.q = line_plan.motion.path.positions[0].copy()
    controller = Controller(robot, np.tile([-3, 3], (6, 1)))
    from dataclasses import replace

    curve = TrackCurve("line", np.zeros(3), np.array([0.0, 1.0, 0.0]), 0.02)
    line_plan = replace(line_plan, curve=curve)
    actual = np.eye(4)
    actual[:3, 3] = [error_mm / 1000, 0.01, 0]
    controller.motion_planner = SimpleNamespace(fk=lambda q: actual)
    if accepted:
        controller.command_tracking(line_plan)
        assert controller.motion_plan is line_plan.motion
    else:
        with pytest.raises(ValueError, match="Path start edge distance must be at most 10 mm"):
            controller.command_tracking(line_plan)
        assert controller.motion_plan is None
    assert not robot.commands


def test_along_edge_lag_is_recorded_without_position_failure(line_plan):
    from types import SimpleNamespace

    actual = line_plan.expected(0).copy()
    actual[:3, 3] = line_plan.curve.point(0.5)
    planner = SimpleNamespace(fk=lambda q: actual, jacobian=lambda q: np.zeros((6, 6)))
    row = tracking_sample(line_plan, planner, np.zeros(7), np.zeros(7), 0)
    assert row["tracking_error_mm"] > 10
    assert row["cross_track_mm"] < 1e-9
    assert row["position_pass"]
    assert row["position_metric"] == "edge_distance"


@pytest.mark.parametrize('sdk_fault', [False, True])
def test_app_records_geometric_deviation_without_hold_but_preserves_sdk_stop(
        planner, line_plan, monkeypatch, tmp_path, sdk_fault):
    from dataclasses import replace
    from types import SimpleNamespace

    from autoweld.legacy_app import App

    plan = replace(line_plan, orientation='Perpendicular')
    elapsed = [.2]
    monkeypatch.setattr('autoweld.legacy_app.time.monotonic', lambda: elapsed[0])
    original = tracking_sample

    def deviated_sample(*args):
        row = original(*args)
        row.update(cross_track_mm=15., orientation_error_deg=12., position_pass=False,
                   orientation_pass=False, deviating=True)
        return row

    monkeypatch.setattr('autoweld.legacy_app.tracking_sample', deviated_sample)

    def unexpected_hold():
        pytest.fail('Geometric deviation must not command Hold')

    app = object.__new__(App)
    app.ctrl = SimpleNamespace(live=False, closed=False, mode='Path tracking', motion_started=0.,
                               motion_planner=planner, motion_fault=lambda: None, hold=unexpected_hold)
    app.tracking_monitor = TrackingMonitor(plan)
    app.tracking_log = tmp_path / 'tracking.jsonl'
    app.tracking_md = SimpleNamespace(content='')
    app.update_tracking()
    assert app.tracking_result['running']
    assert app.tracking_result['deviated']
    assert app.tracking_result['first_deviation_s'] == .2
    assert 'motion continues' in app.tracking_md.content
    if sdk_fault:
        app.ctrl.motion_fault = lambda: {'reason': 'watchdog'}
    elapsed[0] = plan.motion.path.duration + 1.01
    app.update_tracking()
    result = app.tracking_result
    assert result['completed'] is (not sdk_fault)
    assert not result['running'] and not result['passed']
    assert result['position_deviated'] and result['orientation_deviated']
    assert result['deviation_action'] == 'record_only'
    assert result['reason'] is not None if sdk_fault else result['reason'] is None
    assert app.tracking_monitor is None


@pytest.mark.parametrize("guard", ["bounds", "collision", "derivatives"])
def test_tracking_keeps_safety_checks_without_knot_step_gate(planner, line_plan, monkeypatch, guard):
    from dataclasses import replace

    from autoweld.diagnostics import MotionRejection

    kwargs = {}
    if guard == "bounds":
        kwargs["bounds"] = np.tile([-0.001, 0.001], (6, 1))
        code = "joint_bounds"
    elif guard == "collision":
        kwargs["collision"] = lambda q: True
        code = "self_collision"
    else:
        monkeypatch.setattr(planner, "limits", replace(planner.limits, joint_velocity=1e-6))
        code = "derivative_limit"
    with pytest.raises(MotionRejection) as exc:
        plan_tracking(planner, line_plan.motion.path.positions[0], line_plan.curve, **kwargs)
    assert exc.value.diagnostic["code"] == code


def test_unconstrained_path_releases_unrequested_orientation(planner, line_plan):
    free = line_plan
    constrained = plan_tracking(planner, free.motion.path.positions[0], free.curve,
                                orientation='Perpendicular')
    wrist_free = np.ptp(free.motion.path.positions[:, 3:6], axis=0)
    wrist_fixed = np.ptp(constrained.motion.path.positions[:, 3:6], axis=0)
    assert max(wrist_free) < max(wrist_fixed)*.1
    assert free.motion.report['tcp_path_error_m'] < .0002
    assert free.motion.report['orientation_policy'] == 'position only, minimum wrist motion'
    # Motion limits apply to actual changing orientation too.
    for key, peak in free.motion.report['planned_peaks'].items():
        assert peak <= free.motion.report['limits'][key]
