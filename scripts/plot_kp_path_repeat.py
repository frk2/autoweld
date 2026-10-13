"""Offline matched-geometry comparison and complete nine-window diagnostics."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks
from scripts import plot_path_deviation_repeat
from scripts.circle_metrics import circle_coordinates
from scripts.plot_velocity_windows import invalid_intervals, running_velocity

ROOT = Path('evidence/kp-path-repeat')
WINDOWS = tuple(i / 10 for i in range(2, 11))


def analyze(path, curve, planner):
    rows = [json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
    result = json.loads(path.read_text())['result']
    stamps = np.array([r['feedback_timestamps'] for r in rows])
    relative = stamps - stamps[0, 0]
    elapsed = np.array([r['elapsed_s'] for r in rows])
    offset = np.median(elapsed[elapsed > 0] - relative[elapsed > 0, 0])
    common = np.unique(relative[:, 0][(relative[:, 0] >= max(relative[0]))
                                     & (relative[:, 0] <= min(relative[-1]))])
    t = common + offset
    q = np.array([r['joint_position_rad'] for r in rows])
    v = np.array([r['joint_velocity_raw_rad_s'] for r in rows])
    qs = np.column_stack([np.interp(common, relative[:, j], q[:, j]) for j in range(6)])
    vs = np.column_stack([np.interp(common, relative[:, j], v[:, j]) for j in range(6)])
    pos = np.array([planner.fk(q)[:3, 3] for q in qs])
    geo = None
    if curve.kind == 'arc':
        geo = circle_coordinates(pos, curve.center, curve.normal, curve.start)
        travel, tangent = geo['tangential_m'], geo['tangent']
    else:
        travel = (pos - curve.start) @ curve.direction
        tangent = np.tile(curve.direction, (len(pos), 1))
    jac = np.array([planner.jacobian(q)[:3] for q in qs])
    raw = np.einsum('nij,nj,ni->n', jac, vs, tangent)
    reference = np.interp(t, elapsed, np.array([r['edge_fraction'] for r in rows]) * curve.length)
    bad = invalid_intervals(rows)
    plan = result['plan']
    a, b = plan['ramp_s'], plan['duration_s'] - plan['ramp_s']
    windows = {}
    for w in WINDOWS:
        avg = running_velocity(t, travel, raw, reference, w, invalid=bad)
        cruise = (avg['end_s'] - w >= a) & (avg['end_s'] <= b)
        valid = cruise & avg['valid']
        values = avg['position_mm_s'][valid]
        commands = avg['command_mm_s'][valid]
        final = None
        if b - a >= w and b <= t[-1] and b - w >= t[0]:
            exact = running_velocity(t, travel, raw, reference, w, ends=[b], invalid=bad)
            if exact['valid'][0]:
                final = {k: float(exact[k][0]) for k in
                         ('position_mm_s', 'command_mm_s', 'raw_mm_s', 'end_s')}
        windows[str(w)] = dict(valid_cruise_windows=int(valid.sum()),
                               available_cruise_windows=int(cruise.sum()),
                               median_cruise_mm_s=float(np.median(values)) if len(values) else None,
                               p05_cruise_mm_s=float(np.percentile(values, 5)) if len(values) else None,
                               p95_cruise_mm_s=float(np.percentile(values, 95)) if len(values) else None,
                               in_20_percent_fraction=float(np.mean(abs(values-commands) <= .2*commands))
                               if len(values) else None, final_cruise_window=final)
    edge = np.array([r['cross_track_mm'] for r in rows])
    lag = np.array([r['tracking_error_mm'] for r in rows])
    record = dict(path=str(path), completed=result['completed'], passed=result['passed'],
                  feed_mm_s=plan['feed_m_s']*1000, max_edge_mm=float(max(edge)),
                  rms_edge_mm=float(np.sqrt(np.trapezoid(edge**2, elapsed)/(elapsed[-1]-elapsed[0]))),
                  p95_edge_mm=float(np.percentile(edge, 95)), max_timed_error_mm=float(max(lag)),
                  sampling_ok=result.get('sampling_ok'), invalid_velocity_samples=result.get('invalid_velocity_samples'),
                  duration_s=plan['duration_s'], cruise_s=b-a, windows=windows,
                  max_feedback_age_s=max(r['feedback_age_s'] for r in rows),
                  max_feedback_skew_s=float(np.max(np.ptp(stamps, axis=1))))
    if geo is not None:
        record.update(min_radial_mm=float(min(geo['radial_m'])*1000),
                      max_radial_mm=float(max(geo['radial_m'])*1000),
                      max_abs_plane_mm=float(max(abs(geo['plane_m']))*1000),
                      final_phase_deg=float(np.rad2deg(geo['theta_rad'][-1])))
    return dict(record=record, t=t, elapsed=elapsed, edge=edge, lag=lag, pos=pos, geo=geo)


def main():
    assert json.loads(Path('runtime/live/status.json').read_text())['closed']
    plot_path_deviation_repeat.ROOT = ROOT
    plot_path_deviation_repeat.main()
    items = json.loads((ROOT/'preflight.json').read_text())
    model = tool_model()
    planner = MotionPlanner(model)
    records, datasets, settling = [], [], []
    try:
        for item in items:
            candidates = sorted(ROOT.glob(item['name']+'-attempt[0-9].json'))
            if not candidates:
                continue
            path = candidates[-1]
            curve = load_tracks(item['asset'])[item['index']]
            new = analyze(path, curve, planner)
            old = analyze(Path(item['source_runs'][0]), curve, planner) if item['source_runs'] else None
            record = dict(name=item['name'], suite=item['suite'], feed_mm_s=item['feed_mm_s'],
                          new=new['record'], old=old['record'] if old else None)
            if old:
                record['peak_edge_improvement_percent'] = 100*(1-new['record']['max_edge_mm']/old['record']['max_edge_mm'])
                record['peak_timed_error_improvement_percent'] = 100*(1-new['record']['max_timed_error_mm']/old['record']['max_timed_error_mm'])
            records.append(record)
            datasets.append(dict(item=item, curve=curve, new=new, old=old))
            tailpath = path.with_name(path.stem+'-settling-control.jsonl')
            rows = [json.loads(x) for x in tailpath.read_text().splitlines()]
            target = np.array(json.loads(path.read_text())['after']['target_rad'])
            assert all(np.max(abs(np.array(r['target'])-target)) < 1e-8 for r in rows)
            xyz = np.array([planner.fk(r['q'])[:3, 3] for r in rows])
            error = np.linalg.norm(xyz-planner.fk(target)[:3, 3], axis=1)*1000
            edge = np.array([curve.distance(x)*1000 for x in xyz])
            settling.append(dict(name=item['name'], samples=len(rows), final_target_error_mm=float(error[-1]),
                                 final_edge_mm=float(edge[-1]), max_tail_edge_mm=float(max(edge)),
                                 observed_s=rows[-1]['timestamp']-rows[0]['timestamp']))
            print('Compared', item['name'], record['new']['max_edge_mm'], flush=True)
        (ROOT/'comparison.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')
        (ROOT/'settling-metrics.json').write_text(json.dumps(settling, indent=2, allow_nan=False)+'\n')
        draw(datasets, records)
    finally:
        Path(model).unlink()


def draw(datasets, records):
    for suite in ('lines', 'circles'):
        data = [d for d in datasets if d['item']['suite'] == suite]
        fig, axes = plt.subplots(len(data), 2, figsize=(13, 2.3*len(data)), squeeze=False)
        for d, row in zip(data, axes):
            for key, label, color in [('old', 'Earlier Kp', '#c57a2c'), ('new', 'Selected Kp', '#0072b2')]:
                series = d[key]
                if series is None:
                    continue
                row[0].plot(series['elapsed'], series['edge'], color=color, label=label)
                row[1].plot(series['elapsed'], series['lag'], color=color, label=label)
            for ax in row:
                ax.set(xlabel='Time since reference start (s)', ylabel='mm')
                ax.grid(alpha=.2)
                ax.legend(fontsize=8)
            row[0].axhline(10, c='red', ls='--', lw=.8)
            row[0].set_title(f"{d['item']['feed_mm_s']:g} mm/s: finite-edge deviation")
            row[1].set_title('Timed position error (lag + path error)')
            if d['old'] is None:
                row[1].text(.98, .9, 'No earlier completed baseline', transform=row[1].transAxes,
                            ha='right', fontsize=8)
        fig.suptitle(f'{"500 mm line" if suite == "lines" else "R60 mm full circle"}: matched paths and feeds\n'
                     'Standard reference + 1 s traces; encoder/FK, unconstrained orientation', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(ROOT/'plots'/f'{suite}-comparison.png', dpi=140)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, suite in zip(axes, ('lines', 'circles')):
        for key, label, color in [('old', 'Earlier Kp', '#c57a2c'), ('new', 'Selected Kp', '#0072b2')]:
            group = [r for r in records if r['suite'] == suite and r[key] is not None]
            f = [r['feed_mm_s'] for r in group]
            row[0].plot(f, [r[key]['max_edge_mm'] for r in group], 'o-', color=color, label=label)
            row[1].plot(f, [r[key]['max_timed_error_mm'] for r in group], 'o-', color=color, label=label)
        row[0].axhline(10, color='red', ls='--', label='10 mm deviation flag')
        for ax, title in zip(row, ('Maximum finite-edge deviation', 'Maximum timed position error')):
            ax.set(xlabel='Commanded feed (mm/s)', ylabel='mm', title=f'{suite.capitalize()}: {title}')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle('Path adherence with the selected onboard position gains\nSame geometry and reference timing; encoder/FK estimates', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(ROOT/'plots/summary.png', dpi=160)
    plt.close(fig)
    circles = [d for d in datasets if d['item']['suite'] == 'circles']
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, d in zip(axes.flat, circles):
        theta = np.linspace(0, 2*np.pi, 361)
        ax.plot(60*np.cos(theta), 60*np.sin(theta), 'k--', label='Requested R60')
        for key, label, color in [('old', 'Earlier Kp', '#c57a2c'), ('new', 'Selected Kp', '#0072b2')]:
            if d[key] is not None:
                ax.plot(*(d[key]['geo']['uv_m']*1000).T, color=color, label=label)
        ax.set(aspect='equal', xlabel='Circle-plane u (mm)', ylabel='Circle-plane v (mm)',
               title=f"{d['item']['feed_mm_s']:g} mm/s")
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3)
    fig.suptitle('Full-circle geometry: corner cutting and plane projection', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(ROOT/'plots/circles-geometry.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    for ax, d in zip(axes.flat, circles):
        for key, label, color in [('old', 'Earlier', '#c57a2c'), ('new', 'Selected', '#0072b2')]:
            if d[key] is None:
                continue
            geo = d[key]['geo']
            phase = np.rad2deg(geo['theta_rad'])
            ax.plot(phase, geo['radial_m']*1000, color=color, label=f'{label}: radial')
            ax.plot(phase, geo['plane_m']*1000, color=color, ls=':', label=f'{label}: out of plane')
        ax.axhline(0, color='black', lw=.5)
        ax.set(title=f"{d['item']['feed_mm_s']:g} mm/s", xlabel='Measured circle phase (degrees)',
               ylabel='Signed error (mm)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle('Circle error components\nNegative radial error means inside the requested R60 circle; encoder/FK estimates', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(ROOT/'plots/circles-components.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
