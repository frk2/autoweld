"""Regression for Viser gesture phases and a released target during active motion."""
import asyncio
import queue
import time
from types import SimpleNamespace

import pytest

from autoweld.app import App


@pytest.fixture
def app(tmp_path):
    a = object.__new__(App)
    a.events = queue.Queue()
    a.runtime = tmp_path
    a.pending_pose = None
    a.request_status = None
    a.ctrl = SimpleNamespace(mode='IK', closed=False, disable_pending=False, motion_plan=None,
                             motion_fault=lambda: None, ready_for_waypoint=lambda: True)
    a.sent = []
    a.dispatch = a.sent.append
    return a


def test_gizmo_grab_and_updates_do_not_start_a_move(app):
    handle = SimpleNamespace(position=[0, 0, 0], wxyz=[1, 0, 0, 0])
    for phase in ['start', 'update', 'update']:
        asyncio.run(app._target_changed(SimpleNamespace(phase=phase, target=handle)))
    assert app.events.empty()
    handle.position = [.1, .2, .3]
    asyncio.run(app._target_changed(SimpleNamespace(phase='end', target=handle)))
    request = app.events.get_nowait()
    assert request['position'] == [.1, .2, .3]
    assert app.events.empty()


def test_latest_release_waits_and_executes_once(app):
    app.ctrl.motion_plan = object()
    for x in [.1, .2, .3]:
        result = app.handle_request({'action':'pose', 'position':[x, 0, 0], 'rpy':[0, 0, 0]})
        assert result['status'] == 'queued'
    assert not app.sent
    app.process_pending_pose()
    assert not app.sent
    app.ctrl.motion_plan = None
    app.ctrl.ready_for_waypoint = lambda: False
    app.process_pending_pose()
    assert not app.sent
    app.ctrl.ready_for_waypoint = lambda: True
    app.process_pending_pose()
    app.process_pending_pose()
    assert len(app.sent) == 1
    assert app.sent[0]['position'][0] == .3
    assert app.pending_pose is None


@pytest.mark.parametrize('action', ['mode', 'rest', 'disable', 'shutdown', 'joint', 'gripper_close'])
def test_control_actions_cancel_pending_target(app, action):
    app.pending_pose = ({'action':'pose'}, time.monotonic())
    app.handle_request({'action':action})
    assert app.pending_pose is None


def test_pending_target_expires_or_cancels_on_fault(app):
    app.pending_pose = ({'action':'pose'}, time.monotonic()-16)
    app.process_pending_pose()
    assert app.pending_pose is None
    assert app.request_status['status'] == 'expired'
    assert not app.sent
    app.pending_pose = ({'action':'pose'}, time.monotonic())
    app.ctrl.motion_fault = lambda: {'reason':'fault'}
    app.process_pending_pose()
    assert app.pending_pose is None and not app.sent


def test_automatic_requests_do_not_queue_and_errors_are_journaled(app):
    app.ctrl.motion_plan = object()
    def reject(_):
        raise RuntimeError('Motion in progress')
    app.dispatch = reject
    result = app.handle_request({'action':'pose', '_id':'test'})
    assert result['status'] == 'rejected'
    assert app.pending_pose is None
    assert 'Motion in progress' in (app.runtime/'command-events.jsonl').read_text()


def test_diagnostics_visible_and_preserved_in_journal_and_response(app):
    import json

    from autoweld.diagnostics import MotionRejection
    app.pose_status_md = SimpleNamespace(content='')
    def reject(_):
        raise MotionRejection('path_continuity', 'joint_step',
                              'J3 changes 0.08 rad at 25% (limit 0.05 rad)',
                              joint=3, path_fraction=.25, step_rad=.08, limit_rad=.05)
    app.dispatch = reject
    result = app.handle_request({'action': 'pose'})
    assert result['diagnostic']['joint'] == 3
    assert 'path continuity' in app.pose_status_md.content
    assert 'J3 changes 0.08 rad' in app.pose_status_md.content
    record = json.loads((app.runtime/'command-events.jsonl').read_text())
    assert record['diagnostic'] == result['diagnostic'] == app.request_status['diagnostic']
    app.dispatch = app.sent.append
    app.handle_request({'action': 'pose'})
    assert 'diagnostic' not in app.request_status
    assert 'J3 changes' not in app.pose_status_md.content
