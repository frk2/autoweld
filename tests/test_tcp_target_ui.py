"""IK targets begin at the measured TCP without issuing a motion command."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
from test_laser_ui import Gui, Scene, app_fixture

import autoweld.app as frontend


def measured_state(mode='Hold'):
    return dict(mode=mode, closed=False, disable_pending=False, live=False,
        q_rad=[.1, .2, .3, .4, .5, .6], target_rad=[0.]*6, error_deg=[0.]*6,
        tcp_position_m=[.2, -.1, .3], tcp_wxyz=[.5, .5, .5, .5], tcp_rpy_rad=[np.pi/2, 0., np.pi/2],
        controller='POS_VEL', motion_limits={'linear_velocity': 3.}, position_speed_limit_rad_s=2.,
        feedback_source='test measured feedback', rest_verified=False, rest_tolerance_deg=.5)


def render_fixture():
    app, sent, gui, scene = app_fixture()
    app.visual = SimpleNamespace(update_cfg=lambda _q: None)
    app.ee_frame = SimpleNamespace()
    app.target_handle = SimpleNamespace()
    app.mode_select.options = ['Hold', 'Gravity compensation', 'IK']
    app._synced = False
    app._saved_names = app._speed_limits_seen = app._gravity_config_signature = None
    app._cone_handles, app._cone_signature = [], None
    incoming = measured_state()
    app.connection.snapshot = lambda: json.dumps(incoming)
    return app, sent, incoming


def test_constructor_uses_hidden_placeholders_until_measured_pose_arrives_without_commands(monkeypatch):
    gui, scene, sent, transforms = Gui(), Scene(), [], []
    gui.configure_theme = lambda **_kwargs: None
    scene.add_grid = lambda *_args, **_kwargs: None
    scene.add_frame = lambda *_args, **_kwargs: SimpleNamespace()

    def target(_name, **kwargs):
        transforms.append(kwargs)
        return SimpleNamespace(**kwargs, on_update=lambda callback: callback)

    scene.add_transform_controls = target
    server = SimpleNamespace(gui=gui, scene=scene, on_client_connect=lambda callback: callback)
    monkeypatch.setattr(frontend.viser, 'ViserServer', lambda **_kwargs: server)
    monkeypatch.setattr(frontend, 'ViserUrdf', lambda *_args, **_kwargs: SimpleNamespace())
    app = frontend.App(SimpleNamespace(send=sent.append), '/tmp/unused-test.urdf')
    assert not transforms[0]['visible']
    assert transforms[0]['wxyz'] == (1., 0., 0., 0.)
    assert app.rpy.value == (0., 0., 0.)
    assert app.xyz.value == (0., 0., 0.)  # Measured XYZ arrives in the first snapshot.
    assert not sent


def test_initial_and_mode_transition_targets_match_full_measured_tcp_pose_and_joints():
    app, sent, incoming = render_fixture()
    for mode in ('Hold', 'Gravity compensation', 'Hold', 'IK'):
        incoming.update(mode=mode, tcp_position_m=[.21, -.12, .34], q_rad=[.2]*6)
        app.render()
        assert app.target_handle.wxyz == incoming['tcp_wxyz']
        assert app.target_handle.wxyz == app.ee_frame.wxyz
        assert app.rpy.value == tuple(incoming['tcp_rpy_rad'])
        assert app.xyz.value == (.21, -.12, .34)
        assert app.target_handle.position == incoming['tcp_position_m']
        assert [handle.value for handle in app.joint_inputs] == [.2]*6
    assert not sent
    # Same-mode telemetry must not overwrite an operator's staged target, even
    # when the measured arm pose changes while a target is being edited.
    app.target_handle.position = (.3, .2, .4)
    app.target_handle.wxyz = (1., 0., 0., 0.)
    app.xyz.value, app.rpy.value = (.3, .2, .4), (.7, .8, .9)
    incoming.update(tcp_position_m=[.4, .1, .5], tcp_wxyz=[0., 0., 0., 1.],
        tcp_rpy_rad=[0., 0., np.pi], q_rad=[.3]*6)
    app.render()
    assert app.rpy.value == (.7, .8, .9) and not sent
    assert app.xyz.value == (.3, .2, .4)
    assert app.target_handle.position == (.3, .2, .4)
    assert app.target_handle.wxyz == (1., 0., 0., 0.)


def test_reentering_ik_discards_old_target_and_copies_new_measured_pose_without_motion():
    app, sent, incoming = render_fixture()
    incoming['mode'] = 'IK'
    app.render()
    app.target_handle.position = (.8, .7, .6)
    app.target_handle.wxyz = (1., 0., 0., 0.)
    app.xyz.value, app.rpy.value = (.8, .7, .6), (.7, .8, .9)
    incoming['mode'] = 'Hold'
    app.render()
    incoming.update(mode='IK', tcp_position_m=[.4, .1, .5], tcp_wxyz=[0., 0., 0., 1.],
        tcp_rpy_rad=[0., 0., np.pi], q_rad=[.3]*6)
    app.render()
    assert app.target_handle.position == incoming['tcp_position_m']
    assert app.target_handle.wxyz == incoming['tcp_wxyz']
    assert app.xyz.value == (.4, .1, .5)
    assert app.rpy.value == (0., 0., np.pi)
    assert [handle.value for handle in app.joint_inputs] == [.3]*6
    assert not sent


def test_explicit_snap_copies_exact_measured_orientation_and_does_not_send_motion():
    app, sent, _incoming = render_fixture()
    app.render()
    app.sync_button.click()
    assert app.target_handle.wxyz == [.5, .5, .5, .5]
    assert app.rpy.value == (np.pi/2, 0., np.pi/2)
    assert app.xyz.value == (.2, -.1, .3)
    assert [handle.value for handle in app.joint_inputs] == [.1, .2, .3, .4, .5, .6]
    app.render()
    assert app.rpy.value == (np.pi/2, 0., np.pi/2) and not sent


def test_measured_target_is_staged_until_explicit_send_and_server_gizmo_updates_never_move():
    app, sent, incoming = render_fixture()
    incoming['mode'] = 'IK'
    app.render()
    target = app.target_handle
    for phase in ('start', 'update', 'end'):
        asyncio.run(app._target_changed(SimpleNamespace(phase=phase, target=target, client_id=None)))
    assert not sent
    app.pose_button.click()
    assert sent == [{'action': 'pose', 'position': [.2, -.1, .3], 'rpy': [np.pi/2, 0., np.pi/2]}]
    asyncio.run(app._target_changed(SimpleNamespace(phase='end', target=target, client_id='operator')))
    assert sent[-1] == {'action': 'pose', 'position': [.2, -.1, .3], 'wxyz': [.5, .5, .5, .5]}
