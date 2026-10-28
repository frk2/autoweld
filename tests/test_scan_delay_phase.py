"""Synthetic exposure/encoder experiment with real fit and service callbacks."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.motion_registration import RegistrationRejected
from autoweld.scan_delay_phase import run_delay_experiment


class ProbeScene:
    def __init__(self, monkeypatch):
        self.now = 1000.
        self.moves = []
        self.commands = []
        self.target_rotations = []
        self.delay = .037
        self.bad_interval = None
        monkeypatch.setattr('autoweld.scan_delay_phase.time.monotonic', lambda: self.now)
        self.event = SimpleNamespace(is_set=lambda: False, wait=self.wait)
        self.history_id = 'measured-owner'
        self.max_timing_uncertainty_s = .002

    def wait(self, seconds):
        self.now += seconds

    def position(self, stamp):
        for start, duration, first, last in reversed(self.moves):
            if stamp >= start:
                s = np.clip((stamp-start)/duration, 0., 1.)
                return first+(last-first)*(10*s**3-15*s**4+6*s**5)
        return 0.

    def command(self, action, payload, context):
        self.commands.append(action)
        if action == 'move_pose':
            self.target_rotations.append(np.asarray(payload['pose'])[:3, :3])
            self.moves.append((self.now, .65, self.position(self.now), payload['pose'][2][3]))

    def snapshot(self):
        active = bool(self.moves and self.now < self.moves[-1][0]+self.moves[-1][1])
        pose = np.eye(4)
        pose[2, 3] = self.position(self.now)
        return dict(tcp_pose=pose, motion_active=active, feedback=dict(fresh=True, stationary=not active))

    def cloud(self):
        frame = int(self.now*60)
        stamp = frame/60.-.008
        depth = .140-self.position(stamp)
        n = 60
        return dict(source_id='camera', optical_intrinsics_id='optics', acquisition_session_id='session',
            sequence=frame, coordinate_frame='camera', units='m', image_size=[1440, 1080],
            frame_product=dict(kind='single_frame'),
            # A tilted fixed laser plane moves its stripe across the static
            # target as camera Z changes, supplying a real 2D surface footprint.
            xyz_m=np.column_stack([np.linspace(-.01, .01, n), np.full(n, .5*(depth-.140)), np.full(n, depth)]),
            uv=np.column_stack([np.linspace(200., 440., n), np.full(n, 300.)]),
            strength=np.full(n, 30.), width_px=np.full(n, 2.),
            timing=dict(motion_pairing_supported=True, sensor_shutter_mode='global',
                device_frame_id=frame, chunk_frame_id=frame, exposure_end_frame_id=frame,
                exposure_host_mid_s=stamp, exposure_time_uncertainty_s=.0002,
                exposure_duration_us=8000., clock_mapping=dict(status='qualified', mapping_id='clock')))

    def raw_series(self, start, end):
        # Reference angle samples lag the physical camera by the unknown delay.
        rows = np.column_stack([np.arange(start, end, .004), np.zeros((len(np.arange(start, end, .004)), 4))])
        rows[:, 3] = .0001
        return dict(axes=(rows,)*6)

    def provisional_interpolate(self, stamp):
        if self.bad_interval and self.bad_interval[0] <= stamp <= self.bad_interval[1]:
            raise RegistrationRejected('Combined exposure/motor timing uncertainty exceeds limit')
        return SimpleNamespace(joint_pos=np.array([self.position(stamp-self.delay)]),
            timing_uncertainty_s=.0001)

    def pose_and_jacobian(self, q):
        pose = np.eye(4)
        pose[2, 3] = q[0]
        return pose, np.zeros((6, 6))

    def validate_delay_qualification(self, record):
        assert record['history_id'] == self.history_id


@pytest.mark.parametrize('bad_interval', [None, (1001.5, 1001.51)])
def test_service_experiment_recovers_delay_and_returns_to_baseline(monkeypatch, tmp_path, bad_interval):
    scene = ProbeScene(monkeypatch)
    scene.bad_interval = bad_interval
    baseline = np.eye(4)
    angle = .004
    baseline[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    record = run_delay_experiment(dict(cancel_event=scene.event,
        data={'baseline': {'fixed_optical_rotation': np.eye(3)}}), command=scene.command,
        snapshot=scene.snapshot, cloud=scene.cloud, history=scene, fk=scene,
        baseline_optical=baseline, handeye={'tcp_from_camera': np.eye(4)}, evidence_dir=tmp_path)
    assert record['heldout_passed'] and abs(record['delay_s']-scene.delay) < .001
    assert record['uncertainty_s'] < .001
    assert record['acquisition_session_id'] == 'session'
    assert record['history_id'] == scene.history_id
    assert scene.moves[-1][-1] == 0. and scene.commands[-1] == 'hold'
    assert len(list(tmp_path.glob('delay-*.json'))) == 1
    assert len(list(tmp_path.glob('probe-*.npz'))) == 1
    for rotation in scene.target_rotations:
        np.testing.assert_array_equal(rotation, np.eye(3))


def test_cancelled_experiment_never_submits_a_target(monkeypatch, tmp_path):
    scene = ProbeScene(monkeypatch)
    event = threading.Event()
    event.set()
    with pytest.raises(RuntimeError, match='cancelled'):
        run_delay_experiment(dict(cancel_event=event), command=scene.command,
            snapshot=scene.snapshot, cloud=scene.cloud, history=scene, fk=scene,
            baseline_optical=np.eye(4), handeye={'tcp_from_camera': np.eye(4)}, evidence_dir=tmp_path)
    assert not scene.commands
