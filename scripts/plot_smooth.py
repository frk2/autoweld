"""Plot recorded calibration/tuning evidence; never connects to hardware."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs=3, help='Outward, reversal and rest trial names')
    args = parser.parse_args()
    out = Path('evidence/smooth')
    trials = sorted((json.loads(p.read_text()) for p in Path('evidence/pid').glob('smooth-*.json')
                     if not p.stem.endswith('-aborted')), key=lambda d: d['start'])
    (out/'trials.json').write_text(json.dumps(trials, indent=2)+'\n')
    colors = ['#1677b8', '#d47713', '#23935c']
    for filename, window in [('convergence', (0, 1.2)), ('full-hold', None)]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for name, color in zip(args.names, colors):
            summary = json.loads(Path(f'evidence/pid/{name}.json').read_text())
            records = [json.loads(line) for line in Path(f'evidence/pid/{name}.jsonl').read_text().splitlines()]
            t = np.array([r['timestamp']-summary['start'] for r in records])
            errors = np.rad2deg(np.array([r['q'] for r in records])-summary['target'])
            directed = errors*np.sign(summary['step_deg'])
            for j, ax in enumerate(axes.flat):
                ax.plot(t, directed[:, j], color=color, lw=1.3, label=name.removeprefix('smooth-'))
                ax.set_title(f'J{j+1}')
        for ax in axes.flat:
            ax.axhline(0, color='black', lw=.6)
            ax.axhline(.1, color='#bf3131', ls='--', lw=1, label='Overshoot ceiling')
            if window:
                ax.set_xlim(*window)
            ax.set_xlabel('Seconds from command')
            ax.set_ylabel('Error along move direction (deg)')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=7)
        fig.suptitle('Recorded joint convergence: positive error is beyond the waypoint')
        fig.savefig(out/f'{filename}.png', dpi=170)
        fig.savefig(out/f'{filename}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), constrained_layout=True)
    for i, trial in enumerate(trials):
        peak = max(trial['overshoot_deg'])
        stable = trial['settle_and_remain_s']
        color = '#23935c' if trial['speed_and_overshoot_pass'] else '#d47713'
        axes[0].bar(i+1, peak, color=color)
        axes[1].scatter(i+1, stable if stable is not None else trial['duration'],
                        marker='o' if stable is not None else 'x', color=color)
    axes[0].axhline(.1, color='#bf3131', ls='--')
    axes[0].set_ylabel('Maximum overshoot (deg)')
    axes[1].axhline(1., color='#bf3131', ls='--')
    axes[1].set_ylabel('Sustained convergence (s)')
    axes[1].set_xlabel('Trial number; × means not settled within the recording')
    fig.suptitle('Gravity isolation, friction comparison and PID search')
    fig.savefig(out/'comparison.png', dpi=170)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)
    print(f'Plotted {len(trials)} trials; selected traces: {args.names}')


if __name__ == '__main__':
    main()
