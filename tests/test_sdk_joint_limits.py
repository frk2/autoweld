"""Calibrated arm bounds/branch overrides are checked before any CAN access."""
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots import get_robot as factory
from i2rt.robots.utils import ArmType, GripperType


def limits():
    result = np.tile([-2., 2.], (6, 1))
    result[5] = [-5.36, -2.10]
    return result


@pytest.mark.parametrize('bad', [np.zeros((5, 2)), np.zeros((7, 2)), np.zeros((6,)),
                                 np.zeros((6, 2)), np.tile([1., -1.], (6, 1)),
                                 np.full((6, 2), np.nan), np.full((6, 2), np.inf), {'bad': 1}])
def test_bad_limits_fail_before_can_constructor(monkeypatch, bad):
    monkeypatch.setattr(factory, 'DMChainCanInterface', lambda *a, **kw: pytest.fail('Invalid bounds opened CAN'))
    with pytest.raises(ValueError, match='Arm joint limits'):
        factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_joint_limits_override=bad)


@pytest.mark.parametrize('bad', [[0.]*5, [0.]*7, np.zeros((6, 1)), [float('nan')]*6,
                                [float('inf')]*6, {'bad': 1}])
def test_bad_offsets_fail_before_can_constructor(monkeypatch, bad):
    monkeypatch.setattr(factory, 'DMChainCanInterface', lambda *a, **kw: pytest.fail('Invalid offsets opened CAN'))
    with pytest.raises(ValueError, match='Arm motor offsets'):
        factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER,
                              arm_joint_limits_override=limits(), arm_motor_offsets_override=bad)


@pytest.mark.parametrize('overrides', [{'arm_joint_limits_override': limits()},
                                      {'arm_motor_offsets_override': np.zeros(6)}])
def test_arm_overrides_rejected_for_gripper_only_before_device_factory(monkeypatch, overrides):
    monkeypatch.setattr(factory, '_get_gripper_only_robot', lambda **kw: pytest.fail('Opened gripper'))
    with pytest.raises(ValueError, match='require an arm'):
        factory.get_yam_robot(arm_type=ArmType.NO_ARM, **overrides)


@pytest.fixture
def capture_factory(monkeypatch):
    captured = {}
    raw = np.array([.1, .2, .3, .04, -.05, -5.39])
    class Chain:
        def __init__(self, motors, offsets, directions, *args, **kwargs):
            self.motor_offset = np.asarray(offsets, float).copy()
            self.motor_direction = np.asarray(directions, float).copy()
            self.state = [SimpleNamespace(position=float(q)) for q in raw]
            self.commands = [SimpleNamespace(pos=0.) for _ in raw]
            self.position_speed_motor_ids = set(range(1, 7))
            captured.update(chain=self, options=kwargs, initial_offsets=self.motor_offset.copy())
        def _joint_position_real_to_sim_idx(self, position, index):
            return (position-self.motor_offset[index])*self.motor_direction[index]
        def read_states(self):
            return [SimpleNamespace(pos=self._joint_position_real_to_sim_idx(q, i)) for i, q in enumerate(raw)]
        def start_thread(self):
            captured['worker_start_calls'] = captured.get('worker_start_calls', 0)+1
    monkeypatch.setattr(factory, 'DMChainCanInterface', Chain)
    monkeypatch.setattr('i2rt.robots.position_speed_robot.PositionSpeedRobot', lambda **kwargs: kwargs)
    monkeypatch.setattr(factory, 'MotorChainRobot', lambda **kwargs: kwargs)
    return captured


def test_explicit_limits_and_zero_offsets_preserve_negative_j6_branch(capture_factory, monkeypatch):
    monkeypatch.setattr(factory, '_load_joint_limits_from_xml', lambda *a: pytest.fail('Explicit bounds loaded stock ranges'))
    supplied, offsets, model = limits(), np.zeros(6), object()
    robot = factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
                                  xml_path_override='unused.xml', compiled_model=model,
                                  arm_joint_limits_override=supplied, arm_motor_offsets_override=offsets)
    np.testing.assert_array_equal(robot['joint_limits'], supplied)
    np.testing.assert_array_equal(capture_factory['initial_offsets'], offsets)
    np.testing.assert_array_equal(capture_factory['chain'].motor_offset, offsets)
    assert capture_factory['options']['fixed_branch'] is True
    assert capture_factory['chain'].commands[5].pos == pytest.approx(-5.39)
    assert robot['motor_chain'].read_states()[5].pos == pytest.approx(-5.39)
    assert robot['compiled_model'] is model
    assert capture_factory['worker_start_calls'] == 1  # Mock only: no worker/device exists.
    supplied[5] = [-1., 1.]
    offsets[5] = 10.
    np.testing.assert_array_equal(robot['joint_limits'][5], [-5.36, -2.10])
    assert capture_factory['chain'].motor_offset[5] == 0.


def test_explicit_nonzero_arm_offsets_are_used_before_first_command(capture_factory):
    offsets = np.array([.01, .02, .03, .04, .05, -.2])
    robot = factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
                                  xml_path_override='unused.xml', arm_joint_limits_override=limits(),
                                  arm_motor_offsets_override=offsets)
    np.testing.assert_array_equal(capture_factory['initial_offsets'], offsets)
    np.testing.assert_array_equal(robot['motor_chain'].motor_offset, offsets)
    assert robot['motor_chain'].commands[5].pos == pytest.approx(-5.19)


def test_default_factory_keeps_legacy_range_expansion_and_boot_normalization(capture_factory):
    expected = factory._load_joint_limits_from_xml(ArmType.YAM.get_xml_path(), GripperType.NO_GRIPPER.get_xml_path())[:6]
    expected[:, 0] -= .15
    expected[:, 1] += .15
    robot = factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='POS_VEL',
                                  xml_path_override='unused.xml')
    np.testing.assert_array_equal(robot['joint_limits'], expected)
    assert 'fixed_branch' not in capture_factory['options']
    assert capture_factory['chain'].motor_offset[5] == pytest.approx(-2*np.pi)
    assert capture_factory['chain'].commands[5].pos == pytest.approx(-5.39+2*np.pi)


def test_explicit_limits_also_reach_simulator_without_expansion(monkeypatch):
    monkeypatch.setattr('i2rt.robots.sim_robot.SimRobot', lambda **kwargs: kwargs)
    monkeypatch.setattr(factory, 'DMChainCanInterface', lambda *a, **kw: pytest.fail('Simulation opened CAN'))
    robot = factory.get_yam_robot(sim=True, gripper_type=GripperType.NO_GRIPPER,
                                  xml_path_override='unused.xml', arm_joint_limits_override=limits(),
                                  arm_motor_offsets_override=np.zeros(6))
    np.testing.assert_array_equal(robot['joint_limits'], limits())
    assert robot['n_dofs'] == 6 and robot['gripper_index'] is None


def test_explicit_bounds_reach_mit_base_controller(capture_factory):
    robot = factory.get_yam_robot(gripper_type=GripperType.NO_GRIPPER, arm_control_mode='MIT',
                                  xml_path_override='unused.xml', arm_joint_limits_override=limits(),
                                  arm_motor_offsets_override=np.zeros(6))
    np.testing.assert_array_equal(robot['joint_limits'], limits())
    assert robot['motor_chain'].read_states()[5].pos == pytest.approx(-5.39)
