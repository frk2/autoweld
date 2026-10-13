"""PoseTracker analysis with robot event alignment and explicit missing data."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/home/faraz/Code/posetracker')
from posetracker.analysis import AnalysisConfig, amplitude_metrics, analyze, write_report  # noqa: E402


def main(dest):
    p = Path(dest)
    rows = list(csv.DictReader((p / 'poses.csv').open()))
    events = [json.loads(line) for line in (p / 'events.jsonl').read_text().splitlines()]
    first = int(rows[0]['host_monotonic_ns']) / 1e9
    offset = np.median([e['wall_time'] - e['monotonic_ns'] / 1e9 for e in events])
    trace = [json.loads(line) for line in (p / 'circle60-50mms.jsonl').read_text().splitlines()]
    experiment = json.loads((p / 'circle60-50mms.json').read_text())
    plan = experiment['result']['plan']
    start = float(np.median([min(r['feedback_timestamps']) - r['elapsed_s']
                            for r in trace if r['elapsed_s'] > .1]) - offset - first)
    end = start + plan['duration_s']
    times = np.array([float(r['time_s']) for r in rows])
    good = np.array([r['valid'] == '1' for r in rows])

    def event_time(name):
        return next(e['monotonic_ns'] / 1e9 - first for e in events if e['event'].startswith(name))

    phases = [dict(id='initial_hold', start_s=1., end_s=min(6., start - 1))]
    if start - 6 > phases[-1]['end_s']:
        phases.append(dict(id='circle_start_hold', start_s=start - 6, end_s=start - 1))
    phases.extend([dict(id='circle_50mms', start_s=start, end_s=end),
                   dict(id='post_circle_hold', start_s=end + 1,
                        end_s=min(event_time('post-circle dwell'), float(times[-1])))])
    if any(e['event'].startswith('verified arm') for e in events):
        phases.append(dict(id='final_rest_hold', start_s=event_time('verified arm'),
                           end_s=float(times[-1])))
    with (p / 'moves.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'start_s', 'end_s'])
        w.writeheader()
        w.writerows(phases)
    timing = dict(circle_start_s=start, circle_end_s=end, duration_s=plan['duration_s'],
                  wall_minus_monotonic_s=float(offset), first_monotonic_s=first, intervals=phases,
                  timing_limit='Host delivery timestamps, not exposure timestamps; no hardware synchronization')
    (p / 'timing.json').write_text(json.dumps(timing, indent=2))
    b = json.loads((p / 'board.json').read_text())
    center = np.array([b['squares_x'] * b['square_m'] / 2, b['squares_y'] * b['square_m'] / 2, 0]) * 1000
    fields = list(rows[0])
    for r in rows:
        if r['valid'] == '1':
            rot = Rotation.from_rotvec([float(r[k]) for k in ['rx_rad', 'ry_rad', 'rz_rad']])
            xyz = rot.apply(center) + [float(r[k]) for k in ['x_mm', 'y_mm', 'z_mm']]
            for key, value in zip(['x_mm', 'y_mm', 'z_mm'], xyz):
                r[key] = value
    with (p / 'board-center.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    report, segments = analyze(p / 'board-center.csv', AnalysisConfig(high_hz=20), p / 'moves.csv')
    report['measurement_point'] = 'ChArUco board center; board-to-TCP transform unmeasured'
    report['cruise_inner'] = amplitude_metrics(segments, start + plan['ramp_s'] + 1,
                                              end - plan['ramp_s'] - 1)
    report['cruise_inner']['start_s'] = start + plan['ramp_s'] + 1
    report['cruise_inner']['end_s'] = end - plan['ramp_s'] - 1
    write_report(report, segments, p / 'analysis')
    raw = np.array([[float(r[k]) if r[k] != '' else np.nan for k in ['x_mm', 'y_mm', 'z_mm']] for r in rows])
    mask = (times >= start) & (times <= end)
    summary = dict(camera_average_fps=float((len(times)-1)/(times[-1]-times[0])),
                   circle_valid_frames=int(sum(good & mask)), circle_frames=int(sum(mask)),
                   circle_valid_fraction=float(np.mean(good[mask])),
                   encoder_result=experiment['result'], camera_phases=report['moves'],
                   cruise_inner=report['cruise_inner'], quality=report['quality'])
    (p / 'summary.json').write_text(json.dumps(summary, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    colors = ['#0072b2', '#d55e00', '#009e73']
    for i, (axis, color) in enumerate(zip('XYZ', colors)):
        axes[0].plot(times, raw[:, i] - raw[good][0, i], lw=.7, color=color, label=f'Camera {axis}')
        labeled = False
        for seg in segments:
            axes[1].plot(seg['t'], np.where(seg['trusted'], seg['band'][:, i], np.nan),
                         lw=.7, color=color, label=axis if not labeled else None)
            labeled = True
    axes[0].set(title='Board-center displacement — raw PoseTracker observations', ylabel='Displacement (mm)')
    axes[1].set(title='Apparent shaking, 2–20 Hz — gaps excluded; includes camera/pose noise', ylabel='Band displacement (mm)')
    axes[2].plot(times, [int(r['corners']) for r in rows], lw=.6, label='Detected ChArUco corners')
    axes[2].scatter(times[~good], np.zeros(sum(~good)), s=3, color='red', label='Invalid pose')
    axes[2].set(title=f'Capture quality — {summary["camera_average_fps"]:.1f} fps; circle {summary["circle_valid_fraction"]:.1%} valid', ylabel='Corners')
    rt = np.array([r['elapsed_s'] for r in trace]) + start
    axes[3].plot(rt, [r['cross_track_mm'] for r in trace], label='Distance to circle')
    axes[3].plot(rt, [r['tracking_error_mm'] for r in trace], label='Timed position error', alpha=.6)
    axes[3].set(title='Encoder/FK path error — separate measurement from camera board pose', ylabel='Error (mm)')
    for ax in axes:
        ax.axvspan(start, end, alpha=.1, color='gray', label='Circle reference')
        ax.set_xlabel('Camera recording time (s)')
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', ncol=4, fontsize=8)
    fig.savefig(p / 'overview.png', dpi=150)
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ['camera_average_fps', 'circle_valid_fraction', 'cruise_inner']}, indent=2))
    print(json.dumps([{'id': m['id'], 'vibration': m['vibration']} for m in report['moves']], indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
