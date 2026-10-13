"""Offline comparison of timestamp-aligned encoder FK and camera shaking."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, '/home/faraz/Code/posetracker')
from posetracker.analysis import AnalysisConfig, read_poses, split_and_filter  # noqa: E402

from autoweld.app import tool_model  # noqa: E402
from autoweld.motion import MotionPlanner  # noqa: E402


def rms(x):
    return float(np.sqrt(np.mean(np.sum(x*x, axis=1))))


def corr(x, y):
    x, y = x - x.mean(axis=0), y - y.mean(axis=0)
    den = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.sum(x*y) / den) if den > 1e-12 else None


def main(folder):
    p = Path(folder)
    dest = p / 'encoder-comparison'
    dest.mkdir(exist_ok=True)
    timing = json.loads((p / 'timing.json').read_text())
    wall0 = timing['wall_minus_monotonic_s'] + timing['first_monotonic_s']
    rows = [json.loads(line) for line in (p / 'complete-control.jsonl').read_text().splitlines()]
    ct, cp, cg = read_poses(p / 'board-center.csv')
    cs, _ = split_and_filter(ct, cp, cg, AnalysisConfig(high_hz=20))
    assert len(cs) == 1
    c = cs[0]
    t = c['t']
    q = np.empty((len(t), 6))
    good = np.ones(len(t), bool)
    quality = []
    for j in range(6):
        ts = np.array([r['feedback_timestamps'][j] - wall0 for r in rows])
        values = np.array([r['q'][j] for r in rows])
        ts, ix = np.unique(ts, return_index=True)
        values = values[ix]
        q[:, j] = np.interp(t, ts, values)
        hi = np.searchsorted(ts, t, side='right').clip(1, len(ts)-1)
        valid = (t >= ts[0]) & (t <= ts[-1]) & ((ts[hi] - ts[hi-1]) <= .05)
        good &= valid
        gaps = [(float(a), float(b)) for a, b in zip(ts, ts[1:]) if b-a > .05 and b>t[0] and a<t[-1]]
        quality.append(dict(joint=j+1, median_sample_rate_hz=float(1/np.median(np.diff(ts))), gaps=gaps))
    model = tool_model()
    try:
        planner = MotionPlanner(model)
        transforms = np.array([planner.fk(qi) for qi in q])
        trace = [json.loads(line) for line in (p / 'circle60-50mms.jsonl').read_text().splitlines()]
        fk_check = max(float(np.linalg.norm(planner.fk(r['joint_position_rad'])[:3, 3]
                                           - r['actual_m']) * 1000) for r in trace[::50])
        assert fk_check < 1e-6, 'Offline FK differs from the recorded robot path model'
    finally:
        Path(model).unlink()
    rotation = transforms[:, :3, :3]
    tcp = transforms[:, :3, 3] * 1000
    mount = tcp + rotation[:, :, 2] * 150
    pose = np.c_[tcp, Rotation.from_matrix(rotation).as_rotvec()]
    es, eq = split_and_filter(t, pose, good, AnalysisConfig(high_hz=20))
    lo, hi = timing['circle_start_s'], timing['circle_end_s']
    plan = json.loads((p / 'circle60-50mms.json').read_text())['result']['plan']
    phases = dict(before=(1, lo-1), circle=(lo, hi),
                  cruise=(lo+plan['ramp_s']+1, hi-plan['ramp_s']-1), after=(hi+1, t[-1]))
    # Estimate only the camera/base axis rotation from slow path motion. No
    # high-frequency fitting; board-center and TCP are different physical points.
    lp = butter(4, 1., fs=c['fs'], output='sos')
    etrend = sosfiltfilt(lp, tcp, axis=0)
    mask = (t >= lo) & (t <= hi)
    x = etrend[mask] - etrend[mask].mean(axis=0)
    y = c['trend'][mask] - c['trend'][mask].mean(axis=0)
    u, _, vh = np.linalg.svd(x.T @ y)
    camera_rotation = u @ np.diag([1., 1., np.linalg.det(u @ vh)]) @ vh
    alignment_error = rms(x @ camera_rotation - y)
    result = dict(method='Per-joint feedback timestamps; native camera filter reproduced; '
                        'FK resampled onto camera grid; gaps >50 ms excluded plus 1 s filter edges; '
                        'axis rotation fitted only to <1 Hz slow trajectory, no scale fitting',
                  encoder_timing=quality, encoder_filter_quality=eq,
                  slow_path_rigid_alignment_rms_mm=alignment_error,
                  slow_path_rotation=camera_rotation.tolist(), bands={},
                  max_fk_reproduction_error_mm=fk_check,
                  point_warning='Camera board center differs from model TCP (150 mm from mount); no calibrated hand-eye/board-TCP transform')
    samples = []
    for high in (20., 10.):
        summaries = {}
        for phase, (a, b) in phases.items():
            parts = []
            for s in es:
                st = s['t']
                filt = butter(4, [2, high], btype='bandpass', fs=s['fs'], output='sos')
                eb = sosfiltfilt(filt, s['raw'], axis=0)
                cb_all = sosfiltfilt(butter(4, [2, high], btype='bandpass', fs=c['fs'], output='sos'), c['raw'], axis=0)
                cb = np.column_stack([np.interp(st, t, cb_all[:, j]) for j in range(3)])
                mb = sosfiltfilt(filt, np.column_stack([np.interp(st, t, mount[:, j]) for j in range(3)]), axis=0)
                qb = sosfiltfilt(filt, np.column_stack([np.interp(st, t, q[:, j]) for j in range(6)]), axis=0)
                keep = s['trusted'] & (st >= a) & (st <= b)
                if keep.any():
                    parts.append((st[keep], eb[keep], cb[keep], mb[keep], qb[keep]))
            if not parts:
                continue
            pt = np.concatenate([r[0] for r in parts])
            eb, cb, mb = (np.concatenate([r[i] for r in parts]) for i in (1, 2, 3))
            aligned = eb @ camera_rotation
            qb = np.concatenate([r[4] for r in parts])
            axis_pairs = [[corr(eb[:, i:i+1], cb[:, j:j+1]) for j in range(3)] for i in range(3)]
            lag_results = []
            # Positive lag means camera observations follow encoder observations.
            for lag in np.arange(-.15, .1501, .005):
                shifted = np.column_stack([np.interp(pt+lag, t, cb_all[:, j]) for j in range(3)])
                lag_results.append(dict(camera_delay_s=float(lag), vector_correlation=corr(aligned, shifted)))
            item = dict(samples=len(pt), coverage_s=float(len(pt)/c['fs']),
                        camera_rms_mm=rms(cb), encoder_tcp_rms_mm=rms(eb), encoder_mount_rms_mm=rms(mb),
                        rms_ratio=rms(eb)/rms(cb), camera_axis_rms_mm=np.sqrt(np.mean(cb**2, axis=0)).tolist(),
                        encoder_axis_rms_mm=np.sqrt(np.mean(eb**2, axis=0)).tolist(),
                        encoder_joint_band_rms_deg=np.rad2deg(np.sqrt(np.mean(qb**2, axis=0))).tolist(),
                        encoder_approx_camera_axis_rms_mm=np.sqrt(np.mean(aligned**2, axis=0)).tolist(),
                        approximate_same_axis_correlations=[corr(aligned[:, j:j+1], cb[:, j:j+1]) for j in range(3)],
                        aligned_vector_correlation=corr(aligned, cb),
                        amplitude_correlation=corr(np.linalg.norm(eb,axis=1)[:,None], np.linalg.norm(cb,axis=1)[:,None]),
                        axis_correlation_matrix=axis_pairs,
                        best_delay=max(lag_results, key=lambda r: r['vector_correlation']), delay_scan=lag_results)
            summaries[phase] = item
            if high == 20 and phase == 'circle':
                samples = np.c_[pt, aligned, cb]
        result['bands'][f'2-{high:g}Hz'] = summaries
    result['camera_full_circle_original_rms_mm'] = rms(c['band'][mask & c['trusted']])
    previous = json.loads((p / 'summary.json').read_text())
    expected = next(m['vibration']['band_rms_mm'] for m in previous['camera_phases'] if m['id'] == 'circle_50mms')
    assert abs(expected - result['camera_full_circle_original_rms_mm']) < 1e-12
    (dest / 'metrics.json').write_text(json.dumps(result, indent=2))
    np.savetxt(dest / 'circle-signals.csv', samples, delimiter=',',
               header='time_s,encoder_approx_camera_x_mm,encoder_approx_camera_y_mm,encoder_approx_camera_z_mm,camera_x_mm,camera_y_mm,camera_z_mm', comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), constrained_layout=True)
    for j, ax in enumerate(axes[:3]):
        ax.plot(samples[:, 0]-lo, samples[:, j+4], label='Camera board center', color='#0072b2', lw=1)
        ax.plot(samples[:, 0]-lo, samples[:, j+1], label='Encoder TCP (approx. camera axes)', color='#d55e00', lw=1)
        ax.set(ylabel=f'{"XYZ"[j]} band motion (mm)')
        ax.legend(loc='upper right')
    for key, label in [('camera_rms_mm','Camera board center'), ('encoder_tcp_rms_mm','Encoder TCP'), ('encoder_mount_rms_mm','Encoder mount')]:
        axes[3].plot(list(phases), [result['bands']['2-20Hz'][p][key] for p in phases], 'o-', label=label)
    axes[3].set(ylabel='2–20 Hz vector RMS (mm)', xlabel='Matched valid intervals')
    axes[3].legend()
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_title('Camera versus encoder-derived motion: same 2–20 Hz band and matched times\nDifferent measurement points; no fitted amplitude scale; first ≈1 s excluded after encoder log gap')
    axes[2].set_xlabel('Seconds since circle reference start')
    fig.savefig(dest / 'comparison.png', dpi=150)
    plt.close(fig)
    print(json.dumps({band:{phase:{k:v for k,v in r.items() if k in ('camera_rms_mm','encoder_tcp_rms_mm','encoder_mount_rms_mm','aligned_vector_correlation','amplitude_correlation','best_delay','coverage_s')} for phase,r in phases.items()} for band,phases in result['bands'].items()}, indent=2))
    print('Slow alignment error', alignment_error)


if __name__ == '__main__':
    main(sys.argv[1])
