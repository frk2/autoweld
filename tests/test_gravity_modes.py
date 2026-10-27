"""Six-axis operation-mode transitions are delegated exclusively to the SDK."""
import time

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.control import Controller


class SwitchingRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.q = self.q[:6]
        self.events = []
        self.arm_control_mode = 'POS_VEL'
        self.fault = None
        self.fail_switch = None
        self.fail_gravity = False
        self.temperature = 35.
        self.velocity = np.zeros(6)

    def switch_arm_control_mode(self, mode):
        self.events.append(('switch', mode))
        if mode == self.fail_switch:
            raise RuntimeError('SDK transition refused')
        self.arm_control_mode = mode
        self.gravity = False
        self.velocity.fill(0.)  # SDK captures measured PD hold and damps before mode commit.

    def configure_gravity(self, factors, *, model='sdk', reference=None):
        assert self.arm_control_mode == 'MIT'
        self.events.append(('gravity_config', np.asarray(factors).copy(), model))
        if self.fail_gravity:
            raise RuntimeError('SDK model configuration failed')

    def enter_gravity_comp_idle(self):
        assert self.arm_control_mode == 'MIT'
        self.events.append(('gravity_idle',))
        self.gravity = True

    def command_joint_pos(self, q):
        assert self.arm_control_mode == 'POS_VEL'
        assert len(q) == 6
        self.events.append(('position', q.copy()))
        super().command_joint_pos(q)

    def get_motion_fault(self):
        return self.fault

    def acknowledge_motion_fault(self):
        self.events.append(('ack',))
        self.fault = None

    def get_temperature_state(self):
        return {'mos_c': [self.temperature]*6, 'rotor_c': [30.]*6}


@pytest.fixture
def c():
    return Controller(SwitchingRobot(), np.tile([-3., 3.], (6, 1)))


def test_gravity_switches_sdk_once_with_exact_current_tool_model(c):
    c.gravity()
    assert c.control_law == c.robot.arm_control_mode == 'MIT'
    assert c.mode == 'Gravity compensation' and c.robot.gravity
    assert c.robot.events[0] == ('switch', 'MIT')
    assert c.robot.events[1][0] == 'gravity_config'
    np.testing.assert_array_equal(c.robot.events[1][1], np.ones(6))
    assert c.robot.events[1][2] == 'sdk'
    assert c.robot.events[2] == ('gravity_idle',)
    c.gravity()
    assert len(c.robot.events) == 3
    assert not c.robot.closed and not c.mode_transition


@pytest.mark.parametrize('mode', ['Hold', 'IK'])
def test_hold_and_ik_capture_hand_moved_pose_without_replaying_old_target(c, mode):
    c.gravity()
    old_rest = c.rest.copy()
    c.robot.q[0] += .2
    current = c.robot.q.copy()
    c.select_mode(mode)
    assert c.mode == mode
    assert c.control_law == c.robot.arm_control_mode == 'POS_VEL'
    np.testing.assert_array_equal(c.target[:6], current)
    np.testing.assert_array_equal(c.rest, old_rest)
    assert c.robot.events[-2] == ('switch', 'POS_VEL')
    assert c.robot.events[-1][0] == 'position'
    assert not c.robot.closed


def test_gravity_configuration_cached_in_position_mode_applied_on_entry(c):
    gains = np.array([1., 1.01, 1.02, 1., 1., 1.])
    c.configure_gravity(gains)
    assert not c.robot.events
    c.gravity()
    np.testing.assert_array_equal(c.robot.events[1][1], gains)
    c.configure_gravity(np.ones(6))
    assert c.robot.events[-1][0] == 'gravity_config'
    np.testing.assert_array_equal(c.gravity_factors, np.ones(6))


@pytest.mark.parametrize('gains', [[1.]*5, [0.]*6, [-1.]*6, [float('nan')]*6])
def test_invalid_gravity_configuration_does_not_mutate_sdk_or_policy(c, gains):
    with pytest.raises(ValueError):
        c.configure_gravity(gains)
    assert not c.robot.events
    np.testing.assert_array_equal(c.gravity_factors, np.ones(6))


@pytest.mark.parametrize('blocked', ['disable', 'fault', 'cooldown', 'stale', 'disabled'])
def test_gravity_refuses_invalid_lifecycle_without_switching(c, blocked):
    if blocked == 'disable':
        c.disable_pending = True
    elif blocked == 'fault':
        c.robot.fault = {'hold_q': c.robot.q.tolist()}
    elif blocked == 'cooldown':
        c.cooldown_active = True
    elif blocked == 'stale':
        c.health = lambda: None
        c.feedback_warning = 'stale'
    else:
        c.closed = True
    with pytest.raises(RuntimeError):
        c.gravity()
    assert not c.robot.events and not c.robot.closed
    assert c.control_law == 'POS_VEL'


def test_failed_switch_keeps_current_mode_and_exposes_diagnostic(c):
    c.robot.fail_switch = 'MIT'
    with pytest.raises(RuntimeError, match='SDK transition refused'):
        c.gravity()
    assert c.control_law == c.robot.arm_control_mode == 'POS_VEL'
    assert c.mode == 'Hold' and not c.mode_transition
    assert c.mode_transition_error == 'SDK transition refused'
    assert not c.robot.closed


def test_gravity_setup_failure_recovers_position_hold_without_poweroff(c):
    c.robot.fail_gravity = True
    with pytest.raises(RuntimeError, match='model configuration failed'):
        c.gravity()
    assert c.control_law == c.robot.arm_control_mode == 'POS_VEL'
    assert c.mode == 'Hold'
    assert [event[:2] for event in c.robot.events if event[0] == 'switch'] == [('switch', 'MIT'), ('switch', 'POS_VEL')]
    assert not c.robot.closed


def test_failed_recovery_never_claims_position_mode_or_disables(c):
    c.robot.fail_gravity = True
    c.robot.fail_switch = 'POS_VEL'
    with pytest.raises(RuntimeError, match='model configuration failed'):
        c.gravity()
    assert c.control_law == c.robot.arm_control_mode == 'MIT'
    assert c.mode == 'Mode transition failed'
    assert 'recovery failed' in c.mode_transition_error
    assert not c.robot.closed


@pytest.mark.parametrize('operation', ['hold', 'rest', 'disable', 'cooldown', 'joint'])
def test_all_position_operations_restore_sdk_position_speed(c, operation):
    c.gravity()
    c.robot.q[0] += .1
    if operation == 'hold':
        c.hold()
    elif operation == 'rest':
        c.return_rest()
    elif operation == 'disable':
        c.request_disable()
    elif operation == 'cooldown':
        c.robot.temperature = 60.
        c.tick()
        assert c.cooldown_active
    else:
        c.command(c.rest + [.02, 0., 0., 0., 0., 0., 0.])
    assert c.control_law == c.robot.arm_control_mode == 'POS_VEL'
    assert not c.robot.closed
    if operation == 'disable':
        c.tick()
        c.settled_since = time.monotonic()-1.1
        c.tick()
        assert c.robot.closed and c.closed and c.rest_verified


def test_moving_gravity_exit_delegates_damped_stop_to_sdk(c):
    c.gravity()
    c.live = True
    c.health = lambda: None
    c.robot.velocity[:] = .2
    c.hold()
    assert c.control_law == 'POS_VEL'
    assert np.max(abs(c.robot.velocity)) == 0.


def test_motion_fault_keeps_sdk_latch_until_explicit_ack_then_position_hold(c):
    c.gravity()
    c.robot.fault = {'hold_q': c.robot.q.tolist()}
    before = len(c.robot.events)
    c.tick()
    assert c.mode == 'Oscillation fault'
    assert c.control_law == 'MIT'
    assert len(c.robot.events) == before
    c.acknowledge_fault()
    assert c.mode == 'Hold' and c.control_law == 'POS_VEL'
    assert c.robot.events[-2:] == [('ack',), ('switch', 'POS_VEL')]


def test_invalid_joint_target_does_not_exit_gravity(c):
    c.gravity()
    before = len(c.robot.events)
    bad = c.rest.copy()
    bad[0] = 100.
    with pytest.raises(ValueError):
        c.command(bad)
    assert len(c.robot.events) == before and c.control_law == 'MIT'


def test_simulator_runs_same_six_axis_gravity_lifecycle_and_shared_model():
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    from autoweld.models import compiled_model, tool_model
    model = compiled_model()
    robot = get_yam_robot(sim=True, gripper_type=GripperType.NO_GRIPPER,
                          xml_path_override=tool_model(), compiled_model=model,
                          arm_joint_limits_override=model.jnt_range[:6],
                          arm_motor_offsets_override=np.zeros(6))
    try:
        q = np.array([.05, .3, .6, .02, -.1, -2.8])
        robot.command_joint_pos(q)
        c = Controller(robot, model.jnt_range[:6])
        c.strict_joint_limits = {5}
        c.start_hold()
        c.gravity()
        assert c.control_law == robot.arm_control_mode == 'MIT'
        assert robot.get_gravity_info()['enabled']
        assert robot.get_gravity_info()['model'] == 'sdk'
        np.testing.assert_array_equal(robot.get_gravity_info()['factors'], np.ones(6))
        thread = robot._physics_thread
        c.gravity()
        assert robot._physics_thread is thread
        c.hold()
        assert c.control_law == robot.arm_control_mode == 'POS_VEL'
        assert not robot.get_gravity_info()['enabled']
        assert robot._model is model
        np.testing.assert_allclose(c.target[:6], q, atol=.001)
    finally:
        robot.close()
    assert not thread.is_alive()


def test_gravity_selection_cancels_future_path_at_measured_pose(c):
    c.motion_plan = object()
    c.target[0] = .4
    c.robot.q[0] = .1
    c.gravity()
    assert c.motion_plan is None and c.mode == 'Gravity compensation'
    assert c.robot.events[0][0] == 'position'
    assert c.robot.events[0][1][0] == .1
    assert c.robot.events[1] == ('switch', 'MIT')
    assert c.target[0] == .1


def test_gravity_entry_delegates_stationary_wait_to_sdk(c):
    c.live = True
    c.health = lambda: None
    c.robot.velocity[:] = .2
    c.gravity()
    assert c.control_law == 'MIT' and c.robot.gravity
    assert np.max(abs(c.robot.velocity)) == 0.


def test_hold_captures_pose_after_sdk_damped_stop(c):
    c.gravity()
    switch = c.robot.switch_arm_control_mode
    def stop(mode):
        switch(mode)
        if mode == 'POS_VEL':
            c.robot.q[0] += .02  # Measured drift during the bounded damped stop.
    c.robot.switch_arm_control_mode = stop
    c.robot.q[0] = .1
    c.hold()
    assert c.target[0] == pytest.approx(.12)
    assert c.robot.q[0] == pytest.approx(.12)
    assert c.robot.commands[-1][0] == pytest.approx(.12)


def test_hold_reconciles_actual_sdk_mode_after_partial_transition(c):
    # A failed rollback can leave firmware in a different mode from the last
    # successfully selected controller. Hold must consult SDK protocol state.
    c.robot.arm_control_mode = 'MIT'
    c.robot.get_arm_control_modes = lambda: [c.robot.arm_control_mode] * 6
    assert c.control_law == 'POS_VEL'
    c.hold()
    assert c.robot.arm_control_mode == 'POS_VEL'
    assert ('switch', 'POS_VEL') in c.robot.events
    assert c.mode == 'Hold'
