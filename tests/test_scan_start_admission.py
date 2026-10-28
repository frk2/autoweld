"""Explicit scan admission uses the unchanged bounded SDK quiet/fresh wait."""
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.backend import RobotService

pytest_plugins = ("test_post_validation_quiet",)


def service_for(rig):
    service=RobotService.__new__(RobotService)
    service.ctrl=rig.ctrl
    service.ctrl.mode='Hold'
    service.exit_after_disable=False
    service.pending_pose=None
    service.calibration=service.laser=None
    service.scan=SimpleNamespace(busy=False,dispatch=lambda r:dict(accepted=r['action']))
    return service


def test_explicit_start_waits_for_new_quiet_feedback_after_transient_stale_sample(rig):
    service=service_for(rig)
    rig.robot.feedback=lambda:dict(timestamp=rig.clock.wall(),age_s=.08 if rig.robot.diagnostics<3 else .001)
    assert service.dispatch(dict(action='scan_start'))=={'accepted':'scan_start'}
    assert 0 < rig.clock.elapsed < .5
    assert not rig.robot.commands


@pytest.mark.parametrize('condition',['stale','moving'])
def test_persistent_stale_or_moving_feedback_never_admits_scan(rig,condition):
    service=service_for(rig)
    if condition=='stale':
        rig.robot.feedback=lambda:dict(timestamp=rig.clock.wall(),age_s=.020001)
    else:
        rig.robot.velocity=lambda:np.full(7,.050001)
    with pytest.raises(RuntimeError,match='no motion started'):
        service.dispatch(dict(action='scan_start'))
    assert .5 <= rig.clock.elapsed < .503
    assert not rig.robot.commands


def test_readiness_and_state_poll_never_wait_or_command(rig):
    service=service_for(rig)
    rig.robot.feedback=lambda:dict(timestamp=rig.clock.wall(),age_s=.101)
    with pytest.raises(RuntimeError,match='Fresh stationary'):
        service.can_start_scan()
    assert service.dispatch(dict(action='scan_state'))=={'accepted':'scan_state'}
    assert rig.clock.elapsed==0 and not rig.robot.commands


@pytest.mark.parametrize('blocker',['shutdown','moving','disabled','busy','mode'])
def test_structural_start_blockers_reject_before_sdk_wait(rig,blocker):
    service=service_for(rig)
    if blocker=='shutdown':
        service.ctrl.shutdown_requested=True
    elif blocker=='moving':
        service.ctrl.motion_plan=object()
    elif blocker=='disabled':
        service.ctrl.closed=True
    elif blocker=='busy':
        service.scan.busy=True
    else:
        service.ctrl.mode='Gravity compensation'
    with pytest.raises(RuntimeError):
        service.dispatch(dict(action='scan_start'))
    assert rig.clock.elapsed==0 and not rig.robot.commands


def test_scan_feedback_clock_anchor_follows_expensive_health_check(rig):
    service=service_for(rig)
    rig.ctrl.health=lambda:rig.clock.sleep(.15)
    rig.robot.feedback=lambda:dict(timestamp=rig.clock.wall(),age_s=.005)
    result=service.scan_feedback()
    assert result['fresh']
    assert rig.clock.monotonic()-result['measured_at']==pytest.approx(.005)
    assert not rig.robot.commands


def test_scan_feedback_fresh_flag_accounts_for_sdk_read_latency(rig):
    service=service_for(rig)
    rig.robot.feedback=lambda:dict(timestamp=rig.clock.wall(),age_s=.005)
    def observations():
        rig.clock.sleep(.15)
        return dict(joint_vel=np.zeros(7))
    rig.robot.get_observations=observations
    result=service.scan_feedback()
    assert not result['fresh']
    assert rig.clock.monotonic()-result['measured_at']==pytest.approx(.155)
    assert not rig.robot.commands
