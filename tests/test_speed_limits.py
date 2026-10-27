"""The single UI command must not leave planner/firmware limits half-applied."""
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from test_control import FakeRobot

from autoweld.backend import RobotService
from autoweld.control import Controller
from autoweld.motion import MotionLimits


@pytest.fixture
def service(tmp_path):
    s = object.__new__(RobotService)
    s.runtime = tmp_path
    s.ctrl = Controller(FakeRobot(), np.tile([-3., 3.], (6, 1)))
    s.planner = SimpleNamespace(limits=MotionLimits(joint_velocity=.35))
    s.ctrl.motion_planner = s.planner
    s.message, s.fault = 'Old rejection', 'Old request error'
    return s


@pytest.mark.parametrize('joint_speed', [2., 3.])
def test_atomic_speed_apply_accepts_requested_default_and_sdk_maximum(service, joint_speed):
    previous = asdict(service.planner.limits)
    result = service.dispatch({'action': 'speed_limits', 'linear_velocity': 3., 'joint_velocity': joint_speed})
    assert service.planner.limits.joint_velocity == service.ctrl.position_speed_limit == joint_speed
    assert service.planner.limits.linear_velocity == 3.
    assert result['position_speed_limit_rad_s'] == joint_speed
    for key, value in previous.items():
        if key not in ('linear_velocity', 'joint_velocity'):
            assert asdict(service.planner.limits)[key] == value
    assert json.loads((service.runtime/'motion-limits.json').read_text()) == asdict(service.planner.limits)
    assert service.fault is None and service.message == 'Speed limits applied'


@pytest.mark.parametrize('linear,joint', [(0., .5), (float('nan'), .5), (3., 3.01), (3., float('inf'))])
def test_invalid_speed_pair_preserves_both_limits_and_disk(service, linear, joint):
    previous = service.planner.limits
    with pytest.raises(ValueError):
        service.configure_speed_limits(linear, joint)
    assert service.planner.limits is previous
    assert service.ctrl.position_speed_limit == 2.
    assert not (service.runtime/'motion-limits.json').exists()


@pytest.mark.parametrize('reason', ['gravity_mode', 'moving', 'disable_pending'])
def test_guard_rejection_preserves_planner_speed(service, reason):
    previous = service.planner.limits
    if reason == 'gravity_mode':
        service.ctrl.control_law = 'MIT'
    elif reason == 'moving':
        service.ctrl.motion_plan = object()
    else:
        service.ctrl.disable_pending = True
    with pytest.raises((RuntimeError, ValueError)):
        service.configure_speed_limits(5., .5)
    assert service.planner.limits is previous
    assert service.ctrl.position_speed_limit == 2.
    assert not (service.runtime/'motion-limits.json').exists()


def test_failed_save_rolls_back_accepted_in_memory_settings(service, monkeypatch):
    previous = service.planner.limits
    def fail(*args):
        raise OSError('Disk unavailable')
    monkeypatch.setattr('autoweld.backend.atomic_json', fail)
    with pytest.raises(OSError, match='Disk'):
        service.configure_speed_limits(5., .5)
    assert service.planner.limits is previous
    assert service.ctrl.position_speed_limit == 2.
