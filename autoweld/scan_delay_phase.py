"""Service-owned optical/encoder delay experiment; no device owner is created."""

import copy
import json
import time
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from autoweld.laser_scanning import segment_object
from autoweld.motion_delay import DelayCalibrationRejected
from autoweld.motion_registration import RegistrationRejected


def run_delay_experiment(context, *, command, snapshot, cloud, history, fk,
                         baseline_optical, handeye, evidence_dir):
    """Move via the service, fit only measured outputs, persist qualification.

    baseline_optical is the CURRENT measured settled optical pose at working
    depth, including fixed scan roll. The original square center is unchanged.
    command(action,payload,context) marshals short operations to the service main
    thread. snapshot/cloud return detached caches; neither opens hardware.
    """
    baseline = np.asarray(baseline_optical, float).copy()
    h = np.asarray(handeye['tcp_from_camera'], float)
    if baseline.shape != (4, 4) or h.shape != (4, 4) or not np.isfinite(np.r_[baseline.ravel(), h.ravel()]).all():
        raise ValueError('Delay experiment needs finite optical and measured hand-eye transforms')
    fixed_rotation = np.asarray(context.get('data', {}).get('baseline', {}).get(
        'fixed_optical_rotation', baseline[:3, :3]), float)
    if (fixed_rotation.shape != (3, 3) or not np.isfinite(fixed_rotation).all()
            or not np.allclose(fixed_rotation.T@fixed_rotation, np.eye(3), atol=1e-8, rtol=0)
            or np.linalg.det(fixed_rotation) < .99999999):
        raise ValueError('Delay experiment needs a proper fixed optical rotation')
    cancel = context['cancel_event']
    started = time.monotonic()
    frames, seen = [], set()
    binding = None
    previous = None

    def check():
        if cancel.is_set():
            raise RuntimeError('Delay experiment cancelled')
        if time.monotonic()-started > 28.:
            raise DelayCalibrationRejected('Bounded delay experiment exceeded 28 seconds')
        state = snapshot()
        if state.get('fault') or state.get('motion_fault') or state.get('cooldown_active'):
            raise RuntimeError('Physical protection interrupted delay experiment')
        shutdown = state.get('shutdown') or {}
        if shutdown.get('requested') or shutdown.get('active') or state.get('closed'):
            raise RuntimeError('Arm shutdown interrupted delay experiment')
        return state

    def observe(cycle):
        nonlocal binding, previous
        check()
        native = cloud()
        if not isinstance(native, dict):
            return
        timing = native.get('timing') or {}
        device = timing.get('device_frame_id')
        key = (native.get('acquisition_session_id'), device)
        if key in seen:
            return
        if (timing.get('motion_pairing_supported') is not True
                or timing.get('sensor_shutter_mode') != 'global'
                or timing.get('clock_mapping', {}).get('status') != 'qualified'
                or type(device) is not int or device != timing.get('exposure_end_frame_id')
                or device != timing.get('chunk_frame_id')):
            return
        stamp = timing.get('exposure_host_mid_s')
        uncertainty = timing.get('exposure_time_uncertainty_s')
        if (stamp is None or uncertainty is None or not np.isfinite([stamp, uncertainty]).all()
                or uncertainty < 0 or not 0 <= time.monotonic()-stamp <= .3):
            return
        current_binding = dict(source_id=native.get('source_id'),
            optical_intrinsics_id=native.get('optical_intrinsics_id'),
            acquisition_session_id=native.get('acquisition_session_id'),
            clock_mapping_id=timing['clock_mapping'].get('mapping_id'))
        if not all(isinstance(v, str) and v for v in current_binding.values()):
            return
        if binding is not None and binding != current_binding:
            raise DelayCalibrationRejected('Camera identity or acquisition clock changed during delay experiment')
        try:
            segment = segment_object(native, previous)
        except ValueError:
            return
        center = np.asarray(segment['centroid_camera_m'])
        if not .080 <= center[2] <= .220:
            return
        binding, previous = current_binding, center
        seen.add(key)
        if cycle >= 0:
            selected = np.asarray(segment['indices'], int)
            selected = selected[np.linspace(0, len(selected)-1, min(64, len(selected)), dtype=int)]
            frames.append(dict(time_s=float(stamp), depth_m=float(center[2]), cycle=int(cycle),
                uncertainty_s=float(uncertainty), device_frame_id=device,
                exposure_duration_us=timing.get('exposure_duration_us'),
                native_sequence=native.get('sequence'), quality=segment['quality'],
                xyz_camera_m=np.asarray(native['xyz_m'])[selected].tolist(),
                uv=np.asarray(native['uv'])[selected].tolist(),
                strength=np.asarray(native['strength'])[selected].tolist(),
                width_px=np.asarray(native['width_px'])[selected].tolist(),
                sampled_native_indices=selected.tolist(), centroid_camera_m=center.tolist(),
                timing=copy.deepcopy(timing), read_started=native.get('read_started'),
                read_completed=native.get('read_completed')))

    def pause(seconds, cycle=-1):
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            observe(cycle)
            cancel.wait(.005)

    def move(offset, cycle):
        optical = baseline.copy()
        optical[:3, :3] = fixed_rotation
        optical[:3, 3] += fixed_rotation[:, 2]*offset
        target = optical @ np.linalg.inv(h)
        check()
        command('move_pose', {'pose': target.tolist()}, context)
        deadline = time.monotonic()+4.
        while True:
            observe(cycle)
            state = check()
            pose = np.asarray(state.get('tcp_pose'), float)
            feedback = state.get('feedback') or {}
            if pose.shape == (4, 4):
                position_error = np.linalg.norm(pose[:3, 3]-target[:3, 3])
                orientation_error = np.linalg.norm(Rotation.from_matrix(target[:3, :3] @ pose[:3, :3].T).as_rotvec())
                if (not state.get('motion_active') and feedback.get('fresh') and feedback.get('stationary')
                        and position_error <= .0005 and orientation_error <= .005):
                    break
            if time.monotonic() >= deadline:
                raise DelayCalibrationRejected('Measured delay probe endpoint did not converge quietly')
            cancel.wait(.005)

    # Warm reference history on both sides of the full signed delay search.
    pause(.25)
    move(-.004, -1)
    pause(.20)
    for cycle, amplitude in enumerate((.004, .005, .0035)):
        # Different displacement profiles give different measured velocities
        # without changing any configured speed, acceleration or jerk ceiling.
        move(amplitude, cycle)
        move(-amplitude, cycle)
    move(0., -1)
    command('hold', {}, context)
    pause(.25)
    if len(frames) < 60 or binding is None:
        raise DelayCalibrationRejected('Too few unique qualified laser frames during bidirectional probes')
    times = np.asarray([f['time_s'] for f in frames])
    if np.any(np.diff(times) <= 0):
        raise DelayCalibrationRejected('Native delay frames are not strictly ordered')
    raw = history.raw_series(times[0]-.20, times[-1]+.20)
    axes = raw['axes']
    if any(len(axis) < 20 for axis in axes):
        raise DelayCalibrationRejected('Insufficient measured encoder history for delay fit')
    begin = max(axis[0, 0] for axis in axes)
    end = min(axis[-1, 0] for axis in axes)
    output_times = np.arange(begin, end, .004)
    positions = np.full(len(output_times), np.nan)
    poses = np.full((len(output_times), 4, 4), np.nan)
    uncertainties = np.full(len(output_times), np.nan)
    invalid = []
    for index, stamp in enumerate(output_times):
        try:
            measured = history.provisional_interpolate(float(stamp))
            tcp, _ = fk.pose_and_jacobian(measured.joint_pos)
            camera = tcp @ h
            poses[index] = camera
            positions[index] = float(baseline[:3, 2] @ (camera[:3, 3]-baseline[:3, 3]))
            uncertainties[index] = measured.timing_uncertainty_s
        except RegistrationRejected as error:
            invalid.append(dict(time_s=float(stamp), reason=str(error)))
    valid = np.isfinite(positions)
    directory = Path(evidence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trial_id = uuid.uuid4().hex
    np.savez_compressed(directory/f'probe-{trial_id}.npz',
        output_reference_times_s=output_times, measured_camera_z_m=positions,
        measured_camera_poses=poses, baseline_optical=baseline, tcp_from_camera=h,
        measured_timing_uncertainty_s=uncertainties,
        frame_times_s=np.asarray([f['time_s'] for f in frames]),
        frame_depth_m=np.asarray([f['depth_m'] for f in frames]),
        frame_cycles=np.asarray([f['cycle'] for f in frames]),
        frames_json=np.asarray(json.dumps(frames, allow_nan=False)),
        **{f'raw_measured_joint_{i+1}': axis for i, axis in enumerate(axes)})
    diagnostic = dict(binding=binding, history_id=history.history_id, input_frames=len(frames),
        reference_clock=getattr(history, 'reference_clock', 'request_midpoint'),
        invalid_reference_samples=invalid)
    diagnostic_path = directory/f'probe-{trial_id}.json'
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    if valid.sum() < 20:
        raise DelayCalibrationRejected('Too few qualified measured camera poses for geometry timing')
    from autoweld.motion_delay_geometry import fit_geometry_motion_delay

    # A broad, explicitly UNQUALIFIED hint may cross bad intervals. Its only
    # purpose is selecting a bounded search neighborhood; it never publishes a
    # pose or establishes qualification. The final fit uses only actual valid
    # samples and one fixed image set whose entire trial window avoids holes.
    coarse_times = output_times[(output_times >= output_times[valid][0]) & (output_times <= output_times[valid][-1])]
    coarse_poses = np.repeat(np.eye(4)[None], len(coarse_times), axis=0)
    for axis in range(3):
        coarse_poses[:, axis, 3] = np.interp(coarse_times, output_times[valid], poses[valid, axis, 3])
    coarse_poses[:, :3, :3] = Slerp(output_times[valid], Rotation.from_matrix(poses[valid, :3, :3]))(coarse_times).as_matrix()
    coarse_frames = [f for f in frames if f['time_s']-.3 >= coarse_times[0] and f['time_s']+.3 <= coarse_times[-1]]
    coarse_frames = coarse_frames[::max(1, len(coarse_frames)//120)]
    hint = fit_geometry_motion_delay([f['time_s'] for f in coarse_frames],
        [f['xyz_camera_m'] for f in coarse_frames], [f['cycle'] for f in coarse_frames],
        coarse_times, coarse_poses, baseline_optical=baseline, search_s=(-.3, .3),
        max_points_per_frame=20, diagnostic_only=True)
    search = (float(hint.delay_s)-.030, float(hint.delay_s)+.030)
    keep, used = [], np.zeros(len(output_times), dtype=bool)
    for frame in frames:
        left = np.searchsorted(output_times, frame['time_s']+search[0], side='right')-1
        right = np.searchsorted(output_times, frame['time_s']+search[1], side='left')
        if left < 0 or right >= len(output_times) or not valid[left:right+1].all():
            continue
        keep.append(frame)
        used[left:right+1] = True
    diagnostic.update(retained_frames=len(keep), unqualified_coarse_hint=hint.as_dict(),
        qualified_search_interval_s=list(search))
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2))
    if len(keep) < 60:
        raise DelayCalibrationRejected('Too few frames have fully qualified measured brackets throughout the delay search')
    frames = keep
    times = np.asarray([f['time_s'] for f in frames])
    output_times, positions, poses = output_times[valid], positions[valid], poses[valid]
    camera_uncertainty = max(f['uncertainty_s'] for f in frames)
    motor_uncertainty = float(uncertainties[used].max())
    available = history.max_timing_uncertainty_s-camera_uncertainty-motor_uncertainty
    if available <= 0:
        raise DelayCalibrationRejected('Clock and measured encoder brackets exhaust timing budget')
    if cancel.is_set():
        raise RuntimeError('Delay experiment cancelled during timing fit')
    fit = fit_geometry_motion_delay(times, [f['xyz_camera_m'] for f in frames], [f['cycle'] for f in frames],
        output_times, poses, baseline_optical=baseline, search_s=search, max_uncertainty_s=available)
    qualification_id = uuid.uuid4().hex
    path = directory/f'delay-{qualification_id}.json'
    record = dict(schema='autoweld.motion-delay.v1', status='qualified',
        qualification_id=qualification_id, history_id=history.history_id,
        reference_clock=getattr(history, 'reference_clock', 'request_midpoint'),
        **binding, **fit.as_dict(), evidence_path=str(path.resolve()),
        camera_uncertainty_s=camera_uncertainty, motor_bracket_uncertainty_s=motor_uncertainty,
        probe_evidence_path=str((directory/f'probe-{trial_id}.npz').resolve()))
    path.write_text(json.dumps(dict(qualification=record, frames=frames,
        measured_reference_times_s=output_times.tolist(), measured_camera_z_m=positions.tolist(),
        baseline_optical=baseline.tolist(), commanded_fixed_optical_rotation=fixed_rotation.tolist(),
        tcp_from_camera=h.tolist()), allow_nan=False, indent=2))
    history.validate_delay_qualification(record)
    return record
