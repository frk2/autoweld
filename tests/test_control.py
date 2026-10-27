import time
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.control import ANGLE_TOL, REST_TOL, Controller


class FakeRobot:
    def __init__(self):
        self.q = np.array([0, 0.2, 0.3, 0, 0, 0, 0.4], dtype=float)
        self.commands = []
        self.closed = False
        self.gravity = False

    def get_joint_pos(self):
        return self.q.copy()

    def get_robot_info(self):
        return {"gripper_limits": [3.0, -3.57]}

    def get_observations(self):
        return {'joint_pos': self.q.copy(), 'joint_vel': getattr(self, 'velocity', np.zeros(7))}

    def command_joint_pos(self, q):
        self.commands.append(q.copy())
        self.q = q.copy()

    def close(self):
        self.closed = True

    def command_joint_pos_pid(self, q, ki, limits, **kwargs):
        self.command_joint_pos(q)

    def enter_gravity_comp_idle(self):
        self.gravity = True


@pytest.fixture
def c():
    return Controller(FakeRobot(), np.tile([-3.0, 3.0], (6, 1)))


def authorize_live(c):
    c.live = True
    c.health = lambda: None


def test_operator_control_has_no_agent_excursion_cap(c):
    c.live = True
    c.health = lambda: None
    q = c.rest.copy()
    q[0] += 0.4
    c.command(q)
    q[0] += 0.4
    c.command(q)
    assert c.target[0] == 0.8


def test_test_scope_cannot_walk_outward(c):
    authorize_live(c)
    q = c.rest.copy()
    q[0] += 0.3
    c.command(q, automatic=True)
    q[0] += 0.3
    with pytest.raises(ValueError, match="0.49"):
        c.command(q, automatic=True)
    assert len(c.robot.commands) == 1


def test_long_cruise_exception_is_disabled_only_bounded_and_reversible(c):
    assert c.test_excursion == .49
    with pytest.raises(RuntimeError, match="disabled"):
        c.configure_long_cruise_test(True)
    c.closed = True
    c.disabled_q = c.rest.copy()
    c.configure_long_cruise_test(True)
    assert c.test_excursion == 1.55
    c.closed = False
    authorize_live(c)
    q = c.rest.copy()
    q[0] += 1.49
    c.validate(q, automatic=True)
    q[0] = c.rest[0]+1.55
    with pytest.raises(ValueError, match="1.55"):
        c.validate(q, automatic=True)
    # Prior target remains checked independently of rest and current.
    c.target[0] = c.rest[0]-.8
    q[0] = c.rest[0]+.8
    with pytest.raises(ValueError, match="prior target"):
        c.validate(q, automatic=True)
    c.closed = True
    c.configure_long_cruise_test(False)
    assert c.test_excursion == .49
    assert Controller(FakeRobot(), c.limits).test_excursion == .49


def test_long_cruise_keeps_measured_excursion_hold(c):
    c.closed = True
    c.configure_long_cruise_test(True)
    c.closed = False
    authorize_live(c)
    c.motion_plan = SimpleNamespace(path=SimpleNamespace(duration=10))
    c.motion_started = time.monotonic()
    c.motion_automatic = True
    c.robot.q[0] = c.rest[0]+1.55
    with pytest.raises(RuntimeError, match="1.55"):
        c.tick()
    assert c.mode == "Hold"
    assert c.robot.commands


@pytest.mark.parametrize("delta", [0.49, 0.5, -0.5, float("nan"), float("inf")])
def test_reject_invalid_or_oversize_commands(c, delta):
    authorize_live(c)
    q = c.rest.copy()
    q[0] += delta
    with pytest.raises(ValueError):
        c.command(q, automatic=True)
    assert not c.robot.commands


def test_gripper_fixed(c):
    q = c.rest.copy()
    q[6] += 0.01
    with pytest.raises(ValueError, match="Gripper"):
        c.command(q)


@pytest.mark.parametrize("delta", [0.5, -0.5, 0.500001, -0.500001, 1.2, -1.2])
def test_ui_accepts_large_waypoints_within_joint_range(c, delta):
    q = c.rest.copy()
    q[:6] += delta
    c.command(q)
    np.testing.assert_array_equal(c.target, q)


@pytest.mark.parametrize("delta", [3.000001, -3.000001])
def test_ui_rejects_joint_range_violation_without_advancing_target(c, delta):
    q = c.rest.copy()
    q[4] += delta
    with pytest.raises(ValueError, match="URDF joint range"):
        c.command(q)
    assert not c.robot.commands


def test_closed_grip_survives_arm_move_hold_and_rest_without_overwriting_rest(c):
    saved = c.rest.copy()
    c.close_gripper()
    assert c.gripper_target == 0
    q = c.target.copy()
    q[0] += .1
    c.command(q, automatic=True)
    c.hold()
    c.return_rest(automatic=True)
    assert all(q[6] == 0 for q in c.robot.commands)
    np.testing.assert_array_equal(c.rest, saved)
    # Contact blocks full closure, but never blocks an explicit disable at rest.
    c.robot.q[6] = .08
    assert c.at_rest()
    with pytest.raises(RuntimeError, match="Rest verification"):
        c._disable_at_rest()  # Still requires the one-second rest check.
    assert not c.robot.closed
    with pytest.raises(ValueError, match="Gripper"):
        c.command(saved)
    c.request_disable()
    c.robot.q[6] = .08  # Simulate continued contact with the object.
    c.tick()
    assert not c.closed
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed and c.robot.closed
    assert all(q[6] == 0 for q in c.robot.commands)
    np.testing.assert_array_equal(c.rest, saved)


def test_grip_does_not_allow_disable_away_from_rest(c):
    c.close_gripper()
    c.request_disable()
    c.robot.q[0] += .1
    c.tick()
    assert c.disable_pending and not c.closed and c.settled_since is None


def test_disable_waits_for_coasting_arm_then_returns(c):
    authorize_live(c)
    c.robot.q[0] += .1
    c.robot.velocity = np.r_[.1, np.zeros(6)]
    c.request_disable()
    assert c.disable_pending and not c.robot.commands
    c.tick()
    assert not c.robot.commands
    c.robot.velocity.fill(0)
    c.tick()
    assert c.mode == 'Returning to rest'
    c.tick()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed


def test_fault_at_rest_can_disable_without_clearing_fault_or_commanding_motion(c):
    c.close_gripper()
    c.robot.q[6] = .08
    fault = {'hold_q': c.robot.q.tolist()}
    c.robot.get_motion_fault = lambda: fault
    count = len(c.robot.commands)
    c.request_disable()
    c.tick()
    assert c.disable_pending and not c.closed
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed and c.robot.closed
    assert c.motion_fault() is fault
    assert len(c.robot.commands) == count


def test_fault_away_from_rest_keeps_disable_pending_until_explicit_recovery(c):
    c.robot.q[0] += .1
    c.robot.get_motion_fault = lambda: {'hold_q': c.robot.q.tolist()}
    c.request_disable()
    c.tick()
    assert c.disable_pending and not c.closed and not c.robot.commands
    assert c.settled_since is None
    c.robot.get_motion_fault = lambda: None  # Operator's separate SDK recovery.
    c.tick()
    assert c.mode == 'Returning to rest'
    c.tick()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    assert c.closed


def test_fault_disable_still_requires_fresh_feedback(c):
    c.robot.get_motion_fault = lambda: {'hold_q': c.robot.q.tolist()}
    c.request_disable()
    c.tick()
    c.settled_since = time.monotonic()-1.1
    def stale():
        raise RuntimeError('SDK feedback is stale')
    c._feedback_health = stale
    with pytest.raises(RuntimeError, match='stale'):
        c.tick()
    assert not c.closed and not c.robot.closed


def test_grab_and_shutdown_dispatch_use_the_same_rest_disable(c, tmp_path):
    from autoweld.legacy_app import App
    app = object.__new__(App)
    app.ctrl, app.runtime = c, tmp_path
    app.target_handle = SimpleNamespace()
    app.tracking_monitor = None
    app.enable_button = SimpleNamespace()
    app.disable_button = SimpleNamespace()
    app.grab_button = SimpleNamespace()
    app.dispatch({'action': 'gripper_close'})
    app.update_lifecycle_buttons()
    assert not app.disable_button.disabled  # Grab never locks out shutdown.
    assert (tmp_path/'gripper-hold.json').read_text().find('true') >= 0
    app.dispatch({'action': 'shutdown'})
    assert app.exit_after_disable and c.disable_pending
    c.tick()
    c.settled_since = time.monotonic()-1.1
    c.tick()
    app.update_lifecycle_buttons()
    assert c.closed and not app.enable_button.disabled and app.grab_button.disabled


def test_disable_button_available_during_motion_and_fault(c):
    from autoweld.legacy_app import App
    app = object.__new__(App)
    app.ctrl = c
    app.tracking_monitor = None
    app.enable_button = SimpleNamespace()
    app.disable_button = SimpleNamespace()
    app.grab_button = SimpleNamespace()
    c.motion_plan = object()
    c.gripper_hold_closed = True
    c.robot.get_motion_fault = lambda: {'hold_q': c.robot.q.tolist()}
    app.update_lifecycle_buttons()
    assert not app.disable_button.disabled and app.grab_button.disabled


def test_grip_close_preserves_current_pose_away_from_rest(c):
    c.robot.q[0] += .05
    before = c.robot.q[:6].copy()
    c.close_gripper()
    np.testing.assert_array_equal(c.target[:6], before)
    assert c.gripper_target == 0


def test_grip_close_rejects_active_motion(c):
    c.motion_plan = object()
    with pytest.raises(RuntimeError, match="arm idle"):
        c.close_gripper()
    assert not c.gripper_hold_closed
    np.testing.assert_array_equal(c.target, c.rest)


def test_ui_has_no_displacement_cap_from_target_feedback_or_rest(c):
    c.live = True
    c.health = lambda: None
    q = c.rest.copy()
    q[0] = 0.4
    c.command(q)
    c.robot.q[0] = 0.0
    q[0] = 1.2
    c.command(q)
    assert c.target[0] == 1.2


@pytest.mark.parametrize("action", ["joint", "pose"])
@pytest.mark.parametrize("automatic", [False, True])
def test_dispatch_separates_browser_and_automatic_commands(c, action, automatic):
    from autoweld.legacy_app import App

    c.live = True
    c.health = lambda: None
    app = object.__new__(App)
    app.ctrl = c
    app.target_handle = SimpleNamespace()
    app._data = SimpleNamespace(qpos=np.zeros(8))
    app._mirror_robot = lambda: None
    app._has_self_collision = lambda *_: False
    for angle in (0.4, 1.2):
        arm = c.rest[:6].copy()
        arm[0] = angle
        app._kin = SimpleNamespace(ik=lambda *_args, **_kwargs: (True, np.r_[arm, 0, 0]))
        req = {"action": action, "q": arm, "position": [0, 0, 0], "rpy": [0, 0, 0]}
        if automatic:
            req["_id"] = "test-request"
        if automatic and angle == 1.2:
            with pytest.raises(ValueError, match="0.49"):
                app.dispatch(req)
            assert c.target[0] == 0.4
        else:
            app.dispatch(req)
            assert c.target[0] == angle


def test_one_click_return_then_disable(c):
    q = c.rest.copy()
    q[0] += 0.1
    c.command(q)
    c.request_disable()
    assert not c.closed and c.disable_pending
    np.testing.assert_array_equal(c.target, c.rest)
    c.tick()
    assert not c.closed
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed and c.robot.closed


def test_disable_has_no_camera_confirmation_dependency(c):
    authorize_live(c)
    c.request_disable()
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed


def test_drift_invalidates_rest_verification(c):
    c.return_rest()
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.rest_verified
    c.robot.q[0] += 2 * REST_TOL
    c.tick()
    assert not c.rest_verified
    assert not c.closed


def test_gravity_then_hold_does_not_replay_old_target(c):
    with pytest.raises(ValueError, match="six-axis"):
        c.gravity()
    assert not c.robot.gravity
    c.robot.q[0] = 0.2
    c.hold()
    assert c.target[0] == 0.2
    assert c.mode == "Hold"


@pytest.mark.parametrize("degrees, accepted", [(0.49, True), (0.5, True), (0.5001, False)])
def test_rest_tolerance_boundary(c, degrees, accepted):
    c.return_rest()
    c.robot.q[0] = c.rest[0] + np.deg2rad(degrees)
    assert c.at_rest() is accepted


def test_rest_can_verify_without_passing_joint_accuracy(c):
    c.request_disable()
    c.robot.q[0] += np.deg2rad(0.4)
    assert abs(c.robot.q[0] - c.target[0]) > ANGLE_TOL[0]
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed


def test_distant_disable_targets_rest_and_waits_for_feedback(c):
    q = c.rest.copy()
    q[0] += 0.4
    c.command(q)
    q[0] += 0.4
    c.command(q)
    c.request_disable()
    np.testing.assert_array_equal(c.target, c.rest)
    c.robot.q[0] = 0.8  # The return command alone cannot verify rest.
    c.tick()
    assert not c.closed and c.settled_since is None
    c.robot.q = c.target.copy()
    c.tick()
    np.testing.assert_array_equal(c.target, c.rest)
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed
    assert c.robot.commands[-2][0] - c.robot.commands[-1][0] == pytest.approx(0.8)


def test_failed_return_never_disables(c):
    c.request_disable()
    c.robot.q[0] += 0.05
    c.return_step_started = time.monotonic() - 16
    with pytest.raises(RuntimeError, match="remain enabled"):
        c.tick()
    assert not c.closed and not c.robot.closed and not c.disable_pending


def test_pending_disable_rejects_motion_and_duplicate_click_does_not_reset_timer(c):
    c.request_disable()
    c.tick()
    started = c.settled_since
    c.request_disable()
    assert c.settled_since == started
    with pytest.raises(RuntimeError, match="wait for completion"):
        c.command(c.rest)
    with pytest.raises(RuntimeError, match="wait for completion"):
        c.gravity()


def test_enable_reconnects_holds_current_and_preserves_rest_and_calibration(c):
    original_rest = c.rest.copy()
    calls = []
    new_robot = FakeRobot()
    new_robot.q[0] = 0.8  # Arm moved by hand while disabled.

    def factory(limits, last_q, **kwargs):
        assert kwargs == {"arm_control_mode": "POS_VEL"}
        calls.append((limits, last_q))
        return new_robot

    c.robot_factory = factory
    c.request_disable()
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed
    c.enable()
    assert not c.closed and c.mode == "Hold"
    assert c.target[0] == 0.8
    np.testing.assert_array_equal(c.rest, original_rest)
    np.testing.assert_array_equal(calls[0][0], [3.0, -3.57])
    c.enable()
    assert len(calls) == 1
    # A second complete cycle uses the same UI state machine.
    c.request_disable()
    c.tick()
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()
    assert c.closed


def test_failed_reconnect_keeps_disabled_state(c):
    c.request_disable()
    c.tick()
    c.settled_since = time.monotonic() - 1.1
    c.tick()

    def fail(*_, **_kwargs):
        raise RuntimeError("SDK unavailable")

    c.robot_factory = fail
    with pytest.raises(RuntimeError, match="SDK unavailable"):
        c.enable()
    assert c.closed and c.mode == "Disabled"


@pytest.mark.parametrize("degrees, accepted", [(0.49, True), (0.51, False)])
def test_gripper_rest_tolerance_is_in_motor_degrees(c, degrees, accepted):
    span = abs(np.diff(c.robot.get_robot_info()["gripper_limits"])[0])
    c.robot.q[6] += np.deg2rad(degrees) / span
    assert c.at_rest() is accepted


@pytest.mark.parametrize("joint", range(6))
@pytest.mark.parametrize("factor, accepted", [(0.9999, True), (1.0, True), (1.0001, False)])
def test_app_reports_joint_accuracy_separately_from_rest(c, joint, factor, accepted):
    from autoweld.legacy_app import App

    app = object.__new__(App)
    app.ctrl = c
    app.camera = None
    app.message = "test"
    app.fault = None
    app._ee_pose_4x4 = lambda: np.eye(4)
    degrees = (0.2 if joint < 3 else 1.0) * factor
    c.robot.q[joint] += np.deg2rad(degrees)
    result = app.state()
    assert result["accuracy_pass"] is accepted
    assert result["at_rest"] is (degrees <= 0.5)
    assert result["joint_accuracy_pass"][joint] is accepted
    assert result["accuracy_tolerance_deg"] == pytest.approx([0.2] * 3 + [1.0] * 3)
    assert result["rest_tolerance_deg"] == pytest.approx(0.5)


def test_endpoint_ik_rejection_keeps_previous_target_and_sends_nothing(c):
    from autoweld.diagnostics import MotionRejection
    from autoweld.legacy_app import App
    app = object.__new__(App)
    app.ctrl = c
    app._data = SimpleNamespace(qpos=np.zeros(8))
    app._mirror_robot = lambda: None
    app._kin = SimpleNamespace(ik=lambda *a, **kw: (False, np.zeros(8)),
                               fk=lambda *a: np.eye(4))
    previous = c.target.copy()
    with pytest.raises(MotionRejection) as caught:
        app.dispatch({'action': 'pose', 'position': [.4, .2, .3], 'rpy': [0, 0, 0]})
    assert caught.value.diagnostic['stage'] == 'endpoint_ik'
    assert caught.value.diagnostic['checks']['position']['error'] > .5
    assert not c.robot.commands
    np.testing.assert_equal(c.target, previous)
