"""Stage 2 independent numerical checks and hardware-free execution checks."""

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import DMChainCanInterface
from i2rt.robots.waypoint_motion import JointPath, time_scale
from scipy.spatial.transform import Rotation
from test_control import FakeRobot
from test_integral import sdk_stub

from autoweld.app import tool_model
from autoweld.control import Controller
from autoweld.motion import MotionLimits, MotionPlanner


@pytest.fixture(scope="module")
def planner():
    path = tool_model()
    yield MotionPlanner(path)
    Path(path).unlink()


def sample_path():
    start = np.array([0.1, 0.4, 0.8, 0.1, -0.15, 0.2, 0.5])
    return start, start + [0.005, 0.01, 0.015, 0, 0, 0, 0]


def test_time_scaling_boundary_and_derivatives():
    for u in (-1, 0, 1, 2):
        s, v, a, j = time_scale(u)
        assert s == (0 if u <= 0 else 1)
        np.testing.assert_equal(np.array([v, a, j]), 0)
    h = 1e-5
    for u in np.linspace(0.01, 0.99, 31):
        numeric = (np.array(time_scale(u + h)[:3]) - np.array(time_scale(u - h)[:3])) / (2 * h)
        np.testing.assert_allclose(numeric, time_scale(u)[1:], atol=1e-6)


def test_joint_derivatives_and_end_hold():
    q = np.array([[0.0, 1.0], [0.15, 0.9], [0.3, 1.2], [0.2, 1.3]])
    path = JointPath(np.linspace(0, 1, 4), q, 2.3)
    h = 1e-5
    for t in np.linspace(0.1, 2.2, 31):
        numeric = (np.array(path.sample(t + h)[:3]) - np.array(path.sample(t - h)[:3])) / (2 * h)
        np.testing.assert_allclose(numeric, path.sample(t)[1:], atol=2e-6)
    np.testing.assert_allclose(path.sample(3)[0], q[-1])
    np.testing.assert_equal(np.array(path.sample(3)[1:]), 0)
    with pytest.raises(ValueError):
        JointPath([0, 0.5, 0.5, 1], q, 1)
    with pytest.raises(ValueError):
        JointPath([0, 1], q[:2], 0)


def test_tcp_chain_rule_against_independent_fk_differences(planner):
    path = JointPath([0, 1], sample_path(), 1.7)
    t, h = 0.73, 1e-4
    q = path.sample(t)[0]
    plus, minus = (planner.fk(path.sample(t + sign * h)[0]) for sign in (1, -1))
    v, a, j = planner.derivatives(path, t)
    np.testing.assert_allclose(v[:3], (plus[:3, 3] - minus[:3, 3]) / (2 * h), atol=1e-8)
    angular = Rotation.from_matrix(plus[:3, :3] @ minus[:3, :3].T).as_rotvec() / (2 * h)
    np.testing.assert_allclose(v[3:], angular, atol=1e-8)
    numeric_a = (plus[:3, 3] - 2 * planner.fk(q)[:3, 3] + minus[:3, 3]) / h**2
    np.testing.assert_allclose(a[:3], numeric_a, atol=2e-7)
    p, m = planner.derivatives(path, t + h), planner.derivatives(path, t - h)
    np.testing.assert_allclose(a, (p[0] - m[0]) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(j, (p[1] - m[1]) / (2 * h), atol=1e-7)


@pytest.mark.parametrize("cartesian", [False, True])
@pytest.mark.parametrize("large_move", [False, True])
def test_dense_limits_and_ik_path(planner, cartesian, large_move):
    start, goal = sample_path()
    if large_move:
        goal = start.copy()
        goal[0] += 0.7
    target = planner.fk(goal) if cartesian else None
    robot = FakeRobot()
    robot.q = start.copy()
    controller = Controller(robot, np.tile([-3.0, 3.0], (6, 1)))
    controller.motion_planner = planner
    controller.command(goal, cartesian=target)
    plan = controller.motion_plan
    # Independent, offset denser grid than planner's 801 evaluation samples.
    tcp = np.array([planner.derivatives(plan.path, t) for t in np.linspace(0, plan.path.duration, 1802)])
    joints = np.array([plan.path.sample(t)[1:] for t in np.linspace(0, plan.path.duration, 1802)])
    raw = np.r_[
        np.max(np.linalg.norm(tcp[:, :, :3], axis=2), axis=0),
        np.max(np.linalg.norm(tcp[:, :, 3:], axis=2), axis=0),
        np.max(abs(joints), axis=(0, 2)),
    ]
    assert np.all(raw <= np.array(list(plan.report["limits"].values())))
    if cartesian:
        initial = planner.fk(start)
        for s in np.linspace(0, 1, 102):
            actual = planner.fk(plan.path.spline(s))
            np.testing.assert_allclose(
                actual[:3, 3], initial[:3, 3] + s * (target[:3, 3] - initial[:3, 3]), atol=2e-7
            )
        np.testing.assert_allclose(planner.fk(plan.path.sample(plan.path.duration)[0]), target, atol=2e-7)


def test_invalid_inputs_and_interior_collision(planner):
    start, goal = sample_path()
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            MotionLimits(linear_jerk=bad)
    with pytest.raises(ValueError, match="bounds"):
        planner.plan(start, goal, bounds=np.tile([-0.01, 0.01], (6, 1)))
    with pytest.raises(ValueError, match="collision"):
        planner.plan(start, goal, collision=lambda q: start[0] + 0.002 < q[0] < start[0] + 0.003)


def test_sdk_streams_reference_without_resetting_integral(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: now[0])
    r = sdk_stub()
    path = JointPath([0, 1], [np.zeros(7), [0.01, 0.02, 0.03, 0, 0, 0, 0]], 1.0)
    session = r.command_joint_motion(path, np.ones(7) * 10, np.ones(7))
    r.update()
    np.testing.assert_allclose(r.sent[1].pos, 0)
    for i in range(1, 121):
        now[0] = 100 + i * 0.01
        r._joint_state.timestamp = time.time()
        r.update()
        assert r._commands.integral_session == session
        np.testing.assert_allclose(r.sent[1].pos, path.sample(i * 0.01)[0], atol=1e-12)
        np.testing.assert_allclose(r.sent[1].vel, path.sample(i * 0.01)[1], atol=1e-12)
        if i == 50:
            assert r.get_integral_torque()[2] > 0
            assert r.sent[1].vel[2] > 0
    assert r.get_motion_fault() is None
    assert r.get_motion_reference()["active"] is False
    np.testing.assert_equal(r.sent[1].vel, 0)
    # Even identical final target cancels streaming, preserving ordinary PID behavior.
    r.command_joint_pos_pid(path.positions[-1], np.ones(7), np.ones(7))
    assert r._joint_motion is None


@pytest.mark.parametrize("interruption", ["pd", "stale", "gap"])
def test_sdk_motion_interruption(monkeypatch, interruption):
    now = [100.0]
    monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: now[0])
    r = sdk_stub()
    path = JointPath([0, 1], [np.zeros(7), [0.01, 0, 0, 0, 0, 0, 0]], 1.0)
    r.command_joint_motion(path, np.ones(7), np.ones(7))
    r.update()
    if interruption == "pd":
        r.command_joint_pos(np.zeros(7))
    elif interruption == "stale":
        r._joint_state.timestamp -= 0.2
    else:
        now[0] += 2  # Also catch a gap that skips over the entire remaining path.
    r.update()
    assert r._joint_motion is None
    np.testing.assert_equal(r.sent[1].pos, 0)
    np.testing.assert_equal(r.sent[1].vel, 0)
    np.testing.assert_equal(r.get_integral_torque(), 0)
    if interruption != "pd":
        assert "stalled" in r.get_motion_fault()["reason"]


@pytest.mark.parametrize('grabbing', [False, True])
def test_controller_queued_disable_and_hold(planner, monkeypatch, grabbing):
    now = [100.0]
    monkeypatch.setattr("autoweld.control.time.monotonic", lambda: now[0])
    c = Controller(FakeRobot(), np.tile([-3.0, 3.0], (6, 1)))
    if grabbing:
        c.close_gripper()
    c.motion_planner = planner
    goal = c.rest + [0.01, 0.01, 0.02, 0, 0, 0, 0]
    goal[6] = c.gripper_target
    c.command(goal)
    with pytest.raises(RuntimeError):
        c.command(goal)
    with pytest.raises(RuntimeError):
        c.configure_gravity(np.ones(6))
    c.request_disable()
    assert not c.closed
    for _ in range(100):
        now[0] += 0.05
        c.tick()
    assert c.closed and c.rest_verified
    np.testing.assert_allclose(c.positions()[:6], c.rest[:6])
    assert c.positions()[6] == c.gripper_target
    if grabbing:
        assert all(q[6] == 0 for q in c.robot.commands)


def test_worker_is_joined_before_socket_close():
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.running = True
    finished = threading.Event()

    def worker():
        while chain.running:
            time.sleep(0.001)
        time.sleep(0.02)  # Model an I/O call already in progress.
        finished.set()

    chain._control_thread = threading.Thread(target=worker)
    chain.motor_interface = SimpleNamespace(
        close=lambda: finished.wait(0) or pytest.fail("closed bus during I/O")
    )
    chain._control_thread.start()
    chain.close()
    assert finished.is_set() and not chain._control_thread.is_alive()


def test_repeated_target_preserves_completed_motion(planner, monkeypatch):
    now = [100.]
    monkeypatch.setattr('autoweld.control.time.monotonic', lambda: now[0])
    c = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    c.motion_planner = planner
    goal = c.rest + [.01, .01, .02, 0, 0, 0, 0]
    c.command(goal)
    now[0] += c.motion_plan.path.duration+.01
    c.tick()
    calls = len(c.robot.commands)
    c.command(goal)
    assert c.motion_plan is None
    assert len(c.robot.commands) == calls


def test_duration_responds_to_jerk_limit(planner):
    start, goal = sample_path()
    saved = planner.limits
    try:
        planner.limits = MotionLimits(linear_jerk=.01)
        first = planner.plan(start, goal)
        planner.limits = MotionLimits(linear_jerk=.01/8)
        second = planner.plan(start, goal)
        assert second.path.duration == pytest.approx(2*first.path.duration)
    finally:
        planner.limits = saved


@pytest.mark.parametrize('failure', ['ik', 'step', 'tolerance'])
def test_path_rejection_has_location_and_measured_failure(planner, monkeypatch, failure):
    from autoweld.diagnostics import MotionRejection
    start, goal = sample_path()
    target = planner.fk(goal)
    count = [0]
    def solution(*args, **kwargs):
        count[0] += 1
        q = planner._configuration(start)
        if failure == 'ik':
            return False, q
        if failure == 'step':
            q[2] += .08
        # For tolerance test the solver claims success but remains at start:
        # independent interpolated FK validation must still catch its error.
        return True, q
    monkeypatch.setattr(planner.kin, 'ik', solution)
    with pytest.raises(MotionRejection) as caught:
        planner.plan(start, goal, cartesian=target)
    d = caught.value.diagnostic
    if failure == 'ik':
        assert d['stage'] == 'path_ik' and d['path_fraction'] == .025
        assert not d['checks']['position']['passed']
    elif failure == 'step':
        assert d['stage'] == 'path_continuity' and d['joint'] == 3
        assert d['step_rad'] == pytest.approx(.08)
        assert d['limit_rad'] == .05 and d['path_fraction'] == .025
    else:
        assert d['stage'] == 'path_tolerance'
        assert d['checks']['position']['error'] > d['checks']['position']['limit']
        assert d['checks']['position']['path_fraction'] == 1
        assert not d['checks']['position']['passed']
