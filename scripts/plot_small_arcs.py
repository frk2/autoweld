"""Offline finite-arc tracking, center-facing orientation and velocity plots."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks
from scripts.circle_metrics import circle_coordinates
from scripts.plot_velocity_windows import invalid_intervals, running_velocity

ROOT = Path('evidence/small-oriented-arcs')
WINDOWS = tuple(i / 10 for i in range(2, 11))


def main():
    assert json.loads(Path('runtime/live/status.json').read_text())['closed']
    out = ROOT / 'plots'
    out.mkdir(exist_ok=True)
    items = json.loads((ROOT / 'preflight.json').read_text())
    model = tool_model()
    planner = MotionPlanner(model)
    records = []
    try:
        for path in sorted(ROOT.glob('arc-*.jsonl')):
            if path.stem.endswith('-control'):
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            trial_record = json.loads(path.with_suffix('.json').read_text())
            result = trial_record['result']
            item = next(x for x in items if path.stem.startswith(x['name'] + '-'))
            curve = load_tracks(item['asset'])[0]
            plan = result['plan']
            stamps = np.array([r['feedback_timestamps'] for r in rows])
            ts = stamps - stamps[0, 0]
            elapsed = np.array([r['elapsed_s'] for r in rows])
            offset = np.median(elapsed[elapsed > 0] - ts[elapsed > 0, 0])
            common = np.unique(ts[:, 0][(ts[:, 0] >= max(ts[0])) & (ts[:, 0] <= min(ts[-1]))])
            q = np.array([r['joint_position_rad'] for r in rows])
            v = np.array([r['joint_velocity_raw_rad_s'] for r in rows])
            qs = np.column_stack([np.interp(common, ts[:, j], q[:, j]) for j in range(6)])
            vs = np.column_stack([np.interp(common, ts[:, j], v[:, j]) for j in range(6)])
            t = common + offset
            transforms = np.array([planner.fk(q) for q in qs])
            pos, torch = transforms[:, :3, 3], -transforms[:, :3, 2]
            geo = circle_coordinates(pos, curve.center, curve.normal, curve.start)
            bearing = curve.center - pos
            bearing /= np.linalg.norm(bearing, axis=1)[:, None]
            aim = np.rad2deg(np.arccos(np.clip(np.sum(torch * bearing, axis=1), -1, 1)))
            jac = np.array([planner.jacobian(q)[:3] for q in qs])
            raw = np.einsum('nij,nj,ni->n', jac, vs, geo['tangent'])
            ref_arc = np.interp(t, elapsed, np.array([r['edge_fraction'] for r in rows]) * curve.length)
            bad = invalid_intervals(rows)
            averages = {w: running_velocity(t, geo['tangential_m'], raw, ref_arc, w, invalid=bad) for w in WINDOWS}
            a, b = plan['ramp_s'], plan['duration_s'] - plan['ramp_s']
            final = t >= max(t[0], t[-1] - .2)
            metric = {
                'name': path.stem, 'radius_mm': curve.radius * 1000,
                'angle_deg': float(np.rad2deg(curve.sweep)), 'length_mm': curve.length * 1000,
                'feed_mm_s': plan['feed_m_s'] * 1000, 'completed': result['completed'],
                'passed': result['passed'], 'reason': result['reason'],
                'sdk_fault_reason': trial_record.get('sdk_fault_reason'),
                'max_edge_mm': result['max_cross_track_mm'],
                'max_orientation_deg': max(r['orientation_error_deg'] for r in rows),
                'max_actual_center_aim_error_deg': float(max(aim)),
                'final_endpoint_mm': float(np.mean(np.linalg.norm(pos[final] - curve.point(1), axis=1)) * 1000),
                'measured_sweep_deg': float(np.rad2deg(np.mean(geo['theta_rad'][final]) - geo['theta_rad'][0])),
                'minimum_phase_deg': float(np.rad2deg(min(geo['theta_rad']))),
                'reference_cruise_s': b - a, 'sampling_ok': result['sampling_ok'], 'windows': {},
            }
            settle_path = ROOT / f'{path.stem}-settle.json'
            settle = None
            if settle_path.exists():
                settle = json.loads(settle_path.read_text())
                settle_pos = np.array([planner.fk(x['q_rad'])[:3, 3] for x in settle])
                settle_t = np.array([x['timestamp'] - stamps[0, 0] + offset for x in settle])
                settle_geo = circle_coordinates(settle_pos, curve.center, curve.normal, curve.start)
                metric['settled_endpoint_mm'] = float(np.linalg.norm(settle_pos[-1] - curve.point(1)) * 1000)
                metric['settled_phase_deg'] = float(np.rad2deg(settle_geo['theta_rad'][-1]))
                metric['settled_elapsed_s'] = float(settle_t[-1])
                metric['post_run_hold_max_edge_mm'] = max(curve.distance(x) for x in settle_pos) * 1000
            for w, avg in averages.items():
                mask = avg['valid'] & (avg['end_s'] - w >= a) & (avg['end_s'] <= b)
                last = np.flatnonzero(mask)
                i = last[-1] if len(last) else None
                metric['windows'][str(w)] = {
                    'latest_cruise_avg_mm_s': float(avg['position_mm_s'][i]) if i is not None else None,
                    'matching_command_mm_s': float(avg['command_mm_s'][i]) if i is not None else None,
                    'window_end_s': float(avg['end_s'][i]) if i is not None else None,
                    'valid_cruise_windows': int(mask.sum()),
                }
            records.append(metric)
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            ideal = np.array([curve.point(f) for f in np.linspace(0, 1, 101)])
            ax = axes[0, 0]
            ideal_uv = np.column_stack(((ideal - curve.start) @ geo['u'], (ideal - curve.start) @ geo['v'])) * 1000
            actual_uv = (geo['uv_m'] - [curve.radius, 0]) * 1000
            ax.plot(*ideal_uv.T, 'k--', label='Requested finite arc')
            ax.plot(*actual_uv.T, label='Measured encoder/FK path')
            ax.scatter(*actual_uv[[0, -1]].T, c=['green', 'red'], s=30)
            ax.set(xlabel='Radial displacement from start (mm)', ylabel='Tangential coordinate (mm)', title='Path in arc plane; green=start, red=end')
            ax.axis('equal')
            ax = axes[0, 1]
            ax.plot(elapsed, [r['cross_track_mm'] for r in rows], label='Finite-edge distance')
            ax.plot(t, geo['plane_m'] * 1000, label='Signed out-of-plane error', alpha=.7)
            ax.axhline(5, c='red', ls='--', label='5 mm guard')
            ax.set(xlabel='Time (s)', ylabel='mm', title=f"Maximum edge deviation: {metric['max_edge_mm']:.3f} mm")
            ax = axes[0, 2]
            ax.plot(elapsed, [r['orientation_error_deg'] for r in rows], label='Full pose orientation error')
            ax.plot(t, aim, label='Actual torch-to-center aim error')
            ax.axhline(5, c='red', ls='--', label='5° full-pose criterion')
            ax.set(xlabel='Time (s)', ylabel='degrees', title='Torch points toward arc center')
            ax = axes[1, 0]
            ax.plot(t, np.rad2deg(geo['theta_rad']), label='Measured angular position')
            ax.plot(t, np.rad2deg(ref_arc / curve.radius), 'k--', label='Reference angular position')
            if settle is not None:
                ax.plot(settle_t, np.rad2deg(settle_geo['theta_rad']), ':', label='Post-run settling observation')
            ax.set(xlabel='Time (s)', ylabel='degrees', title=f"Measured sweep: {metric['measured_sweep_deg']:.2f}°; endpoint: {metric['final_endpoint_mm']:.3f} mm")
            ax = axes[1, 1]
            ax.plot(t, raw * 1000, c='.75', alpha=.5, lw=.7, label='Raw tangential velocity')
            for w in (.2, .5, 1.):
                avg = averages[w]
                ax.plot(avg['center_s'], avg['position_mm_s'], label=f'{w:g} s running avg')
            avg = averages[.5]
            ax.plot(avg['center_s'], avg['command_mm_s'], 'k--', label='0.5 s command avg')
            ax.set(xlabel='Window midpoint (s)', ylabel='mm/s', title='Signed tangential velocity; timing gaps omitted')
            ax = axes[1, 2]
            ax.plot(elapsed, [r['tracking_error_mm'] for r in rows], label='Timed position error')
            ax.set(xlabel='Time (s)', ylabel='mm', title='Timed error is informational')
            for ax in axes.flat:
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            status = ('Completed' if result['completed'] else 'SDK watchdog interruption'
                      if trial_record.get('sdk_fault_reason') else 'Stopped by edge guard')
            title = f"R {curve.radius*1000:g} mm · {np.rad2deg(curve.sweep):g}° · {plan['feed_m_s']*1000:g} mm/s · {status}"
            fig.suptitle(title + '\nEncoder/FK tracking estimates; no external metrology or force-deflection measurement', fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, .94))
            fig.savefig(out / f'{path.stem}-tracking.png', dpi=140)
            plt.close(fig)
            fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
            # Keep the full raw range in the tracking figure. Here use the full
            # running-average range so the commanded feed and its band are legible.
            values = np.concatenate([np.r_[v['position_mm_s'], v['command_mm_s']] for v in averages.values()])
            values = values[np.isfinite(values)]
            low = min(0., float(min(values))) if len(values) else -1.
            high = max(plan['feed_m_s'] * 1200, float(max(values))) if len(values) else 1.
            padding = max(.1, (high - low) * .1)
            for ax, w in zip(axes.flat, WINDOWS):
                avg = averages[w]
                ax.plot(t, raw * 1000, c='.8', alpha=.5, lw=.6, label='Raw tangential velocity')
                ax.plot(avg['center_s'], avg['position_mm_s'], label='Measured running avg')
                ax.plot(avg['center_s'], avg['command_mm_s'], 'k--', label='Command avg, same window')
                ax.fill_between(avg['center_s'], avg['command_mm_s']*.8, avg['command_mm_s']*1.2, alpha=.1, color='green')
                info = metric['windows'][str(w)]
                val = info['latest_cruise_avg_mm_s']
                note = 'No valid full cruise window'
                if val is not None:
                    note = f"Latest cruise avg: {val:.3f} mm/s\nCommand: {info['matching_command_mm_s']:.3f} mm/s"
                    ax.plot(info['window_end_s'] - w/2, val, 'o', ms=4)
                ax.text(.98, .04, note, transform=ax.transAxes, ha='right', fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
                ax.set(title=f'{w:.1f} s window', xlabel='Window midpoint (s)', ylabel='mm/s')
                ax.set_ylim(low - padding, high + padding)
                ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .915), ncol=3)
            fig.suptitle(title + '\nNine full sampling windows; measured travel uses actual radius'
                         '\nZoomed to running averages; raw outliers can extend beyond axes', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .875))
            fig.savefig(out / f'{path.stem}-windows.png', dpi=140)
            plt.close(fig)
            print({k: v for k, v in metric.items() if k != 'windows'}, flush=True)
        (ROOT / 'metrics.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    finally:
        Path(model).unlink()


if __name__ == '__main__':
    main()
