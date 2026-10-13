"""Offline encoder oscillation comparison; never imports or opens robot I/O."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def joint_signal(rows, joint, start, duration, session, high=20., fs=50.):
    """Align feedback and reference, split gaps, and discard filter boundaries."""
    t = np.array([r['feedback_timestamps'][joint] - start for r in rows])
    q = np.rad2deg([r['q'][joint] for r in rows])
    t, ix = np.unique(t, return_index=True)
    q = q[ix]
    refs = [r for r in rows if (r.get('motion_reference') or {}).get('session') == session]
    rt = np.array([r['timestamp'] - start for r in refs])
    rq = np.rad2deg([r['motion_reference']['q'][joint] for r in refs])
    rt, ix = np.unique(rt, return_index=True)
    rq = rq[ix]
    sample_rate = float(1 / np.median(np.diff(t)))
    segments = []
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > .05)+1, len(t)]
    for a, b in zip(cuts[:-1], cuts[1:]):
        if t[b-1]-t[a] < 2.5:
            continue
        grid = np.arange(max(t[a], -4), min(t[b-1], duration+5), 1/fs)
        if len(grid) < 125:
            continue
        position = np.interp(grid, t[a:b], q[a:b])
        reference = np.interp(grid, rt, rq)
        error = position-reference
        filtered = sosfiltfilt(butter(4, [2., high], btype='bandpass', fs=fs, output='sos'), error)
        trusted = (grid >= grid[0]+1) & (grid <= grid[-1]-1)
        segments.append(dict(t=grid, raw=position, reference=reference, error=error,
                             band=filtered, trusted=trusted))
    return segments, sample_rate


def measure(segments, lo, hi, high, fs=50.):
    values, powers, raw_values, errors, references = [], [], [], [], []
    for segment in segments:
        keep = segment['trusted'] & (segment['t'] >= lo) & (segment['t'] <= hi)
        v = segment['band'][keep]
        if len(v):
            values.append(v)
            raw_values.append(segment['raw'][keep])
            errors.append(segment['error'][keep])
            references.append(segment['reference'][keep])
        # Fixed four-second Welch windows for comparable 0.25 Hz resolution.
        if len(v) >= 4*fs:
            f, p = welch(v, fs=fs, nperseg=int(4*fs), noverlap=int(2*fs))
            powers.append((p, len(v)))
    if not values:
        return dict(coverage_s=0, rms_deg=None, peak_to_peak_deg=None, dominant_hz=None), None
    value = np.concatenate(values)
    raw = np.concatenate(raw_values)
    error = np.concatenate(errors)
    # A constant encoder code can leave tiny numerical/reference residuals.
    # Do not report their spectral maximum as a physical shaking frequency.
    no_encoder_motion = np.ptp(raw) < 1e-6
    result = dict(coverage_s=len(value)/fs, coverage_fraction=min(1., len(value)/fs/(hi-lo)),
                  rms_deg=float(np.sqrt(np.mean(value**2))), peak_to_peak_deg=float(np.ptp(value)),
                  dominant_hz=None, frequency_resolution_hz=.25,
                  mean_tracking_error_deg=float(error.mean()),
                  raw_position_span_deg=float(np.ptp(raw)),
                  reference_position_span_deg=float(np.ptp(np.concatenate(references))),
                  encoder_position_constant=bool(no_encoder_motion))
    spectrum = None
    if powers:
        p = np.average([x[0] for x in powers], weights=[x[1] for x in powers], axis=0)
        band = (f >= 2) & (f <= high)
        if not no_encoder_motion:
            result['dominant_hz'] = float(f[band][np.argmax(p[band])])
        if not no_encoder_motion:
            spectrum = (f[band], p[band])
    return result, spectrum


def main(folder='evidence/mixed-wrist'):
    dest = Path(folder)
    results, signals = [], {}
    tracking_details = []
    for path in sorted(dest.glob('*-r[12].json')):
        trial = json.loads(path.read_text())
        rows = [json.loads(line) for line in path.with_name(path.stem+'-control.jsonl').read_text().splitlines()]
        after = trial['after']
        ref = after['motion_reference']
        session, duration = ref['session'], ref['duration_s']
        reference_rows = [r for r in rows if (r.get('motion_reference') or {}).get('session') == session]
        start = float(np.median([r['timestamp']-r['motion_reference']['elapsed_s'] for r in reference_rows]))
        item = dict(name=path.stem, controller=after['controller'], fixture=trial['fixture']['name'],
                    feed_mm_s=trial['fixture']['feed_m_s']*1000, duration_s=duration,
                    max_edge_error_mm=trial['result']['max_cross_track_mm'],
                    max_timed_error_mm=trial['result']['max_tracking_error_mm'], bands={})
        trace = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        active_trace = [r for r in trace if r['elapsed_s'] <= duration]
        item['full_path_measured_joint_span_deg'] = np.rad2deg(np.ptp(
            [r['joint_position_rad'] for r in active_trace], axis=0)).tolist()
        item['full_path_reference_joint_span_deg'] = np.rad2deg(np.ptp(
            [r['motion_reference']['q'][:6] for r in reference_rows], axis=0)).tolist()
        if item['fixture'] == 'circle60':
            tracking_details.append((item, reference_rows, active_trace, start))
        for high in (20., 10.):
            joints = []
            for joint in range(6):
                segments, rate = joint_signal(rows, joint, start, duration, session, high)
                metrics, spectrum = measure(segments, 0, duration, high)
                metrics.update(joint=joint+1, native_sample_rate_hz=rate)
                joints.append(metrics)
                if high == 20 and joint in (3, 4):
                    signals[(path.stem, joint)] = (segments, spectrum)
            item['bands'][f'2-{high:g}Hz'] = joints
        results.append(item)
    if not results:
        raise RuntimeError('No completed mixed-wrist captures available; no comparison claimed')
    (dest/'oscillation-metrics.json').write_text(json.dumps(results, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for fixture in ('line120', 'circle60'):
        selected = [r for r in results if r['fixture'] == fixture]
        if not selected:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
        for item in selected:
            color = '#0072b2' if item['controller'] == 'POS_VEL' else '#d55e00'
            label = ('Position–speed' if item['controller'] == 'POS_VEL' else 'Mixed MIT/gravity')+' '+item['name'].rsplit('-', 1)[1]
            for j, joint in enumerate((3, 4)):
                segments, spectrum = signals[(item['name'], joint)]
                first = True
                for segment in segments:
                    keep = segment['trusted'] & (segment['t'] >= 0) & (segment['t'] <= item['duration_s'])
                    axes[j, 0].plot(segment['t'][keep], segment['band'][keep], color=color, alpha=.65,
                                    lw=.8, label=label if first else None)
                    first = False
                if spectrum is not None:
                    axes[j, 1].semilogy(*spectrum, color=color, alpha=.7, label=label)
                axes[j, 0].set(ylabel=f'J{joint+1} tracking error (°)', xlabel='Reference time (s)')
                axes[j, 1].set(ylabel='Power (°²/Hz)', xlabel='Frequency (Hz)', xlim=(2,20))
        if fixture == 'line120':
            axes[1, 0].set_ylim(-.025, .025)
            axes[1, 0].text(.5, .8, 'J5 encoder position stayed on one code in all line runs',
                            ha='center', transform=axes[1, 0].transAxes, fontsize=8)
            axes[1, 1].set_axis_off()
            axes[1, 1].text(.5, .5, 'J5: no encoder-resolved motion\nNo shaking frequency reported',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
        for ax in axes.flat:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f'{fixture}: encoder tracking-error oscillation, 2–20 Hz\n'
                     'Per-joint feedback timestamps; >50 ms gaps excluded with 1 s filter margins')
        fig.savefig(dest/f'{fixture}-oscillation.png', dpi=160)
        plt.close(fig)
    if tracking_details:
        fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True, sharex=True)
        for item, rows, trace, start in tracking_details:
            col = 0 if item['controller'] == 'POS_VEL' else 1
            label = item['name'].rsplit('-', 1)[1]
            for row, joint in enumerate((3, 4)):
                t = np.array([r['feedback_timestamps'][joint]-start for r in rows])
                q = np.rad2deg([r['q'][joint] for r in rows])
                rt = np.array([r['timestamp']-start for r in rows])
                reference = np.rad2deg([r['motion_reference']['q'][joint] for r in rows])
                use = (t >= 0) & (t <= item['duration_s'])
                axes[row,col].plot(t[use], q[use], color='#0072b2', alpha=.7, label=f'Measured {label}')
                axes[row,col].plot(rt[use], reference[use], '--', color='#d55e00', alpha=.7, label=f'Reference {label}')
                axes[row,col].set_ylabel(f'J{joint+1} angle (°)')
            axes[2,col].plot([r['elapsed_s'] for r in trace], [r['cross_track_mm'] for r in trace], label=label)
        for col, title in enumerate(('All position–speed', 'J4/J5 MIT + gravity')):
            axes[0,col].set_title(title)
            axes[2,col].set(xlabel='Reference time (s)', ylabel='Edge deviation (mm)')
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Circle at 50 mm/s: quieter mixed-mode wrists also under-track the commanded motion\n'
                     'Raw joint positions and reference; two repeats; encoder/FK geometry')
        fig.savefig(dest/'circle60-tracking.png', dpi=160)
        plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
