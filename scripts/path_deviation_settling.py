"""Archive post-trace settling already recorded before each normal rest return."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks

ROOT = Path('evidence/path-deviation-repeat')


def main():
    assert json.loads(Path('runtime/live/status.json').read_text())['closed']
    items = json.loads((ROOT / 'preflight.json').read_text())
    coverage = json.loads((ROOT / 'coverage.json').read_text())
    events = [json.loads(x) for x in Path('runtime/live/command-events.jsonl').read_text().splitlines()]
    intervals = []
    for c in coverage:
        if not c['completed']:
            continue
        record_path = Path(c['result_path'])
        record = json.loads(record_path.read_text())
        rows = [json.loads(x) for x in record_path.with_suffix('.jsonl').read_text().splitlines()]
        last = rows[-1]
        epoch = np.median([min(r['feedback_timestamps']) - r['elapsed_s'] for r in rows if r['elapsed_s'] > 0])
        cap = next(e for e in events if e['timestamp'] > record['after']['timestamp']
                   and e['action'] == 'position_speed_limit' and e['request']['limit_rad_s'] == .15)
        intervals.append({'name': record_path.stem, 'item': items[c['index']], 'record': record,
                          'rows': rows, 'epoch': float(epoch), 'begin': max(last['feedback_timestamps']),
                          'end': cap['timestamp'] - cap['processing_s'], 'tail': []})
    # The append-only log is large; seek to this session by its wall-clock timestamp.
    earliest = min(x['begin'] for x in intervals)
    source = Path('runtime/live/control-telemetry.jsonl')
    with source.open('rb') as f:
        lo, hi = 0, source.stat().st_size
        while hi - lo > 65536:
            middle = (lo + hi) // 2
            f.seek(middle)
            f.readline()
            line = f.readline()
            if line and json.loads(line)['timestamp'] < earliest:
                lo = middle
            else:
                hi = middle
        f.seek(lo)
        if lo:
            f.readline()
        for line in f:
            row = json.loads(line)
            for interval in intervals:
                if interval['begin'] < row['timestamp'] < interval['end']:
                    stamps = row.get('feedback_timestamps')
                    if stamps and min(stamps) > interval['begin']:
                        interval['tail'].append(row)
    model = tool_model()
    planner = MotionPlanner(model)
    records = []
    fig, axes = plt.subplots(4, 4, figsize=(18, 13))
    try:
        for interval, ax in zip(intervals, axes.flat):
            rows = interval['tail']
            assert len(rows) >= 10, interval['name']
            target = np.array(interval['record']['after']['target_rad'])
            assert all(np.max(abs(np.array(r['target']) - target)) < 1e-8 for r in rows)
            (ROOT / (interval['name'] + '-settling.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            curve = load_tracks(interval['item']['asset'])[interval['item']['index']]
            xyz = np.array([planner.fk(r['q'])[:3, 3] for r in rows])
            expected = planner.fk(target)[:3, 3]
            t = np.array([min(r['feedback_timestamps']) - interval['epoch'] for r in rows])
            edge = np.array([curve.distance(p) * 1000 for p in xyz])
            target_error = np.linalg.norm(xyz-expected, axis=1)*1000
            endpoint = np.linalg.norm(xyz-curve.point(1), axis=1)*1000
            original = interval['rows']
            ax.plot([r['elapsed_s'] for r in original], [r['cross_track_mm'] for r in original],
                    label='Standard trace: edge distance')
            ax.plot(t, edge, label='Post-trace edge distance')
            ax.plot(t, target_error, '--', label='Post-trace final-target error')
            ax.axvline(original[-1]['elapsed_s'], c='.6', ls=':')
            ax.axhline(10, c='red', ls='--', lw=.7)
            ax.set(title=interval['name'].replace('-attempt1', '').replace('-attempt2', ''),
                   xlabel='Time since path start (s)', ylabel='mm')
            ax.title.set_fontsize(8)
            ax.grid(alpha=.2)
            ax.text(.98, .9, f'Final target error: {target_error[-1]:.3f} mm', transform=ax.transAxes,
                    ha='right', fontsize=8)
            record = {'name':interval['name'], 'samples':len(rows), 'end_elapsed_s':float(t[-1]),
                      'standard_trace_end_s':original[-1]['elapsed_s'],
                      'final_target_error_mm':float(target_error[-1]),
                      'final_finite_edge_distance_mm':float(edge[-1]),
                      'final_edge_endpoint_distance_mm':float(endpoint[-1]),
                      'max_settling_edge_distance_mm':float(max(edge)),
                      'max_edge_including_settling_mm':max(float(max(edge)), interval['record']['result']['max_cross_track_mm']),
                      'max_feedback_age_s':max(r['timestamp']-min(r['feedback_timestamps']) for r in rows),
                      'measurement':'SDK encoder/FK; post-trace observations, before rest return. No external metrology.'}
            records.append(record)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=3)
        fig.suptitle('Complete paths and subsequent settling before return to rest\n'
                     'Dotted divider: end of standard reference + 1 s hold; red line: 10 mm deviation threshold', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .905))
        fig.savefig(ROOT / 'plots/settling-overview.png', dpi=140)
        plt.close(fig)
        (ROOT / 'settling-metrics.json').write_text(json.dumps(records, indent=2)+'\n')
        print(json.dumps(records, indent=2))
    finally:
        Path(model).unlink()


if __name__ == '__main__':
    main()
