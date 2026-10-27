"""Calibration motion is local to the user's baseline; cooldown owns rest return."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.backend import RobotService


@pytest.fixture
def service():
    s = object.__new__(RobotService)
    q = np.array([-1.9, 1., .4, .5, -.1, -2.3])
    s.commands = []
    s.ctrl = SimpleNamespace(cooldown_active=False, positions=lambda: q.copy(),
        command=lambda target, mode, **kw: s.commands.append((target, mode, kw)))
    s.calibration = SimpleNamespace(return_pose=np.eye(4))
    s._calibration_reference = None
    s.validate_pose = lambda p: np.r_[q + np.r_[p[:3, 3], [0.]*3], 0.]
    return s


def test_calibration_far_from_rest_uses_local_envelope_and_normal_path(service):
    pose = np.eye(4)
    pose[0, 3] = .02
    service.move_calibration_pose(pose)
    assert len(service.commands) == 1
    assert service.commands[0][2]['automatic'] is False
    np.testing.assert_array_equal(service.commands[0][2]['cartesian'], pose)


@pytest.mark.parametrize('reason', ['translation', 'orientation', 'joint', 'cooldown'])
def test_local_calibration_guards_precede_motor_command(service, reason):
    pose = np.eye(4)
    if reason == 'translation':
        pose[0, 3] = .081
    elif reason == 'orientation':
        pose[:3, :3] = Rotation.from_euler('x', 21, degrees=True).as_matrix()
    elif reason == 'joint':
        service.validate_pose = lambda p: np.r_[service.ctrl.positions()+.5, 0.]
    else:
        service.ctrl.cooldown_active = True
    with pytest.raises((ValueError, RuntimeError)):
        service.move_calibration_pose(pose)
    assert not service.commands
