"""Complete-trace error and nine-window velocity plots for guard-stop repeats."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks
from scripts.circle_metrics import circle_coordinates
from scripts.plot_velocity_windows import invalid_intervals, running_velocity

ROOT = Path('evidence/path-deviation-repeat')
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
        for path in sorted(ROOT.glob('*-attempt*.jsonl')):
            if not re.fullmatch(r'.+-attempt\d+', path.stem):
                continue
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            result = json.loads(path.with_suffix('.json').read_text())['result']
            item = next(x for x in items if path.stem.startswith(x['name'] + '-attempt'))
            curve = load_tracks(item['asset'])[item['index']]
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
            normal_error = None
            if curve.kind == 'arc':
                geo = circle_coordinates(pos, curve.center, curve.normal, curve.start)
                travel, tangent = geo['tangential_m'], geo['tangent']
                progress = geo['theta_rad'] / curve.sweep
                if item['orientation'] == 'Center facing':
                    bearing = curve.center - pos
                    bearing /= np.linalg.norm(bearing, axis=1)[:, None]
                    normal_error = np.rad2deg(np.arccos(np.clip(np.sum(torch * bearing, axis=1), -1, 1)))
            else:
                travel = (pos - curve.start) @ curve.direction
                tangent = np.tile(curve.direction, (len(pos), 1))
                progress = travel / curve.length
                if item['orientation'] == 'Perpendicular':
                    normal_error = np.rad2deg(np.arcsin(np.clip(abs(torch @ curve.direction), 0, 1)))
            jac = np.array([planner.jacobian(q)[:3] for q in qs])
            raw = np.einsum('nij,nj,ni->n', jac, vs, tangent)
            ref_travel = np.interp(t, elapsed, np.array([r['edge_fraction'] for r in rows]) * curve.length)
            bad = invalid_intervals(rows)
            averages = {w: running_velocity(t, travel, raw, ref_travel, w, invalid=bad) for w in WINDOWS}
            a, b = plan['ramp_s'], plan['duration_s'] - plan['ramp_s']
            record = {'name': path.stem, 'source_runs': item['source_runs'], 'feed_mm_s': item['feed_mm_s'],
                      'curve_kind': curve.kind, 'length_mm': curve.length * 1000, 'orientation': item['orientation'],
                      **{k: result[k] for k in ('completed', 'passed', 'reason', 'deviated', 'position_deviated',
                                               'orientation_deviated', 'first_deviation_s', 'max_cross_track_mm',
                                               'max_tracking_error_mm', 'max_orientation_error_all_deg',
                                               'sampling_ok', 'invalid_velocity_samples')},
                      'final_recorded_phase_percent': float(progress[-1] * 100),
                      'last_endpoint_error_mm': float(np.linalg.norm(pos[-1] - curve.point(1)) * 1000),
                      'max_actual_tool_normal_error_deg': float(max(normal_error)) if normal_error is not None else None,
                      'observed_duration_s': float(t[-1]), 'reference_duration_s': plan['duration_s'], 'windows': {}}
            for w, avg in averages.items():
                mask = avg['valid'] & (avg['end_s'] - w >= a) & (avg['end_s'] <= b)
                ids = np.flatnonzero(mask)
                i = ids[-1] if len(ids) else None
                record['windows'][str(w)] = {
                    'latest_cruise_avg_mm_s': float(avg['position_mm_s'][i]) if i is not None else None,
                    'matching_command_mm_s': float(avg['command_mm_s'][i]) if i is not None else None,
                    'window_end_s': float(avg['end_s'][i]) if i is not None else None,
                    'valid_cruise_windows': int(mask.sum()),
                }
            records.append(record)
            fig = plt.figure(figsize=(16, 9))
            ideal = np.array([curve.point(f) for f in np.linspace(0, 1, 301)])
            ax = fig.add_subplot(231, projection='3d')
            local_ideal, local_pos = (ideal - curve.start) * 1000, (pos - curve.start) * 1000
            ax.plot(*local_ideal.T, 'k--', label='Requested edge')
            ax.plot(*local_pos.T, label='Encoder/FK TCP')
            ax.scatter(*local_pos[[0, -1]].T, c=['green', 'red'], s=25)
            points = np.vstack((local_ideal, local_pos))
            center = (points.max(axis=0) + points.min(axis=0)) / 2
            span = np.ptp(points, axis=0)
            span = np.maximum(span, max(float(max(span)) * .6, 5)) * 1.1
            # Matplotlib normalizes its aspect argument in place; keep limits in mm.
            ax.set_box_aspect(span.copy())
            for axis, lo, hi, set_lim in zip((ax.xaxis, ax.yaxis, ax.zaxis), center-span/2,
                                             center+span/2, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
                set_lim(lo, hi)
                axis.set_major_locator(MaxNLocator(3))
            ax.set(xlabel='ΔX mm', ylabel='ΔY mm', zlabel='ΔZ mm',
                   title='Relative to edge start; equal mm scale')
            ax = fig.add_subplot(232)
            ax.plot(elapsed, [r['cross_track_mm'] for r in rows], label='Finite-edge distance')
            ax.axhline(10, c='red', ls='--', label='10 mm flag; continues')
            ax.axhline(5, c='.6', ls=':', label='Historical 5 mm stop')
            ax.set(xlabel='Time (s)', ylabel='mm', title=f"Max deviation: {record['max_cross_track_mm']:.3f} mm")
            ax = fig.add_subplot(233)
            ax.plot(elapsed, [r['orientation_error_deg'] for r in rows], label='Full reference orientation error')
            if normal_error is not None:
                ax.plot(t, normal_error, label='Actual tool-to-path normal error')
            if item['orientation'] != 'Unconstrained':
                ax.axhline(5, c='red', ls='--', label='5° reference-orientation flag')
            ax.set(xlabel='Time (s)', ylabel='degrees', title=item['orientation'])
            ax = fig.add_subplot(234)
            ax.plot(t, progress * 100, label='Measured geometric phase')
            ax.plot(t, ref_travel / curve.length * 100, 'k--', label='Reference phase')
            ax.set(xlabel='Time (s)', ylabel='% of edge', title='Actual progress, including reversal / lag')
            ax = fig.add_subplot(235)
            ax.plot(t, raw * 1000, c='.75', alpha=.5, lw=.6, label='Raw signed tangential velocity')
            for w in (.2, .5, 1.):
                avg = averages[w]
                ax.plot(avg['center_s'], avg['position_mm_s'], label=f'{w:g} s running avg')
            avg = averages[.5]
            ax.plot(avg['center_s'], avg['command_mm_s'], 'k--', label='0.5 s command avg')
            ax.set(xlabel='Window midpoint (s)', ylabel='mm/s', title='Full raw range; invalid timing windows omitted')
            ax = fig.add_subplot(236)
            ax.plot(elapsed, [r['tracking_error_mm'] for r in rows], label='Timed position lag')
            ax.set(xlabel='Time (s)', ylabel='mm', title='Timed error remains informational')
            for ax in fig.axes:
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            status = 'Completed, DEVIATED' if result['completed'] and result['deviated'] else 'Completed' if result['completed'] else 'Interrupted'
            title = f"{item['name']} · {status}"
            fig.suptitle(title + '\n10 mm reporting-only policy; encoder/FK estimates, not external metrology', fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, .94))
            fig.savefig(out / f'{path.stem}-tracking.png', dpi=140)
            plt.close(fig)
            fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
            values = np.concatenate([np.r_[v['position_mm_s'], v['command_mm_s']] for v in averages.values()])
            values = values[np.isfinite(values)]
            low = min(0., float(min(values))) if len(values) else -1.
            high = max(item['feed_mm_s'] * 1.2, float(max(values))) if len(values) else 1.
            pad = max(.1, (high - low) * .1)
            for ax, w in zip(axes.flat, WINDOWS):
                avg = averages[w]
                ax.plot(t, raw * 1000, c='.8', alpha=.5, lw=.6, label='Raw tangential velocity')
                ax.plot(avg['center_s'], avg['position_mm_s'], label='Measured running avg')
                ax.plot(avg['center_s'], avg['command_mm_s'], 'k--', label='Command, same window')
                ax.fill_between(avg['center_s'], avg['command_mm_s'] * .8, avg['command_mm_s'] * 1.2,
                                alpha=.1, color='green')
                info = record['windows'][str(w)]
                val = info['latest_cruise_avg_mm_s']
                note = 'No valid full cruise window'
                if val is not None:
                    note = f"Latest cruise avg: {val:.2f} mm/s\nCommand: {info['matching_command_mm_s']:.2f} mm/s"
                    ax.plot(info['window_end_s'] - w / 2, val, 'o', ms=4)
                ax.text(.98, .04, note, transform=ax.transAxes, ha='right', fontsize=8,
                        bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none'})
                ax.set(title=f'{w:.1f} s window', xlabel='Window midpoint (s)', ylabel='mm/s', ylim=(low-pad, high+pad))
                ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .915), ncol=3)
            fig.suptitle(title + '\nNine full sampling windows, plotted at their midpoint'
                         '\nZoomed to running averages; raw outliers can extend beyond axes', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .875))
            fig.savefig(out / f'{path.stem}-windows.png', dpi=140)
            plt.close(fig)
            print(path.stem, result['completed'], result['deviated'], result['max_cross_track_mm'], flush=True)
        (ROOT / 'metrics.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    finally:
        Path(model).unlink()


if __name__ == '__main__':
    main()
