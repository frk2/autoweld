"""Plot archived Kp trials only, with motors disabled."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEST = Path('evidence/pos-speed-kp')


def main():
    state = json.loads(Path('runtime/live/status.json').read_text())
    if not state['closed']:
        raise RuntimeError('Run bulk plots only with motors disabled')
    rounds = sorted(DEST.glob('*/metrics.json'), key=lambda p: p.stat().st_mtime)
    summary = []
    for path in rounds:
        metrics = json.loads(path.read_text())
        status = json.loads((path.parent/'enabled-status.json').read_text())
        gains = [status['motor_mode_survey'][str(i)]['KP_APR'] for i in range(1, 7)]
        overshoot_limits = metrics[0].get('overshoot_limits_deg', [.1]*6)
        times = np.array([[j['settling_s'] or np.nan for j in m['joints']] for m in metrics])
        overshoot = np.array([[j['overshoot_deg'] for j in m['joints']] for m in metrics])
        error_path = path.parent/'error.json'
        summary.append(dict(name=path.parent.name, gains=gains, tests=len(metrics),
                            synchronized=sum(m['synchronized'] for m in metrics),
                            max_ratio=max((m['arrival_ratio'] for m in metrics if m['arrival_ratio']), default=None),
                            median_joint_settling_s=np.nanmedian(times, axis=0).tolist(),
                            max_overshoot_deg=float(overshoot.max()),
                            error=json.loads(error_path.read_text())['error'] if error_path.exists() else None,
                            rest_verified=(path.parent/'rest-status.json').exists(),
                            all_pass=len(metrics) == 10 and not error_path.exists()
                            and (path.parent/'rest-status.json').exists()
                            and all(m['synchronized'] and m['overshoot_pass'] and m['all_settled'] for m in metrics)))
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        x = np.arange(1, len(metrics)+1)
        for j in range(6):
            axes[0].plot(x, times[:, j], 'o-', label=f'J{j+1} (Kp {gains[j]:.2f})')
            axes[1].plot(x, overshoot[:, j], 'o-', label=f'J{j+1}')
        axes[0].plot(x, [m['duration_s'] for m in metrics], 'k--', label='Reference duration')
        axes[0].plot(x, np.nanmin(times, axis=1)*1.2, color='gray', linestyle=':', label='1.2 × earliest arrival')
        axes[0].set(ylabel='Settling time from SDK motion start (s)')
        axes[0].legend(ncol=4, fontsize=8)
        if overshoot_limits[:3] == overshoot_limits[3:]:
            axes[1].axhline(overshoot_limits[0], color='red', linestyle='--', label='Overshoot ceiling (all joints)')
        else:
            axes[1].axhline(overshoot_limits[0], color='red', linestyle='--', label='J1–J3 overshoot ceiling')
            axes[1].axhline(overshoot_limits[3], color='orange', linestyle='--', label='J4–J6 overshoot ceiling')
        axes[1].set(xlabel='Waypoint number', ylabel='Directed overshoot (degrees)', xticks=x)
        axes[1].legend(ncol=7, fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f'{path.parent.name}: {summary[-1]["synchronized"]}/{len(metrics)} arrivals within 20%\n'
                     'J1–J3 ±0.2°, J4–J6 ±1°; sustained for ≥1 s; encoder measurements')
        fig.tight_layout()
        fig.savefig(path.parent/'settling.png', dpi=150)
        plt.close(fig)
    (DEST/'summary.json').write_text(json.dumps(summary, indent=2))
    if summary:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        x = np.arange(len(summary))
        for j in range(6):
            axes[0].plot(x, [r['median_joint_settling_s'][j] for r in summary], 'o-', label=f'J{j+1}')
        axes[0].set(ylabel='Median settling time (s)')
        axes[0].legend(ncol=6)
        axes[1].plot(x, [r['max_ratio'] for r in summary], 'o-', label='Worst slowest/fastest arrival ratio')
        axes[1].axhline(1.2, color='green', linestyle='--', label='Requested ≤1.20')
        axes[1].set(ylabel='Arrival ratio', xticks=x, xticklabels=[r['name'] for r in summary])
        axes[1].tick_params(axis='x', labelrotation=30)
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle('Onboard position-loop Kp tuning: fixed waypoint sequence and motion limits')
        fig.tight_layout()
        fig.savefig(DEST/'comparison.png', dpi=150)
        plt.close(fig)
        latest = json.loads(rounds[-1].read_text())
        base = json.loads(rounds[0].read_text())
        if len(latest) == len(base) == 10:
            fig, axes = plt.subplots(1, 2, figsize=(13, 8), sharey=True)
            fig.subplots_adjust(top=.80, wspace=.30)
            latest_limits = latest[0].get('overshoot_limits_deg', [.1]*6)
            label = 'Confirmed profile' if summary[-1]['all_pass'] else 'Latest candidate'
            latest_title = (f"{label}: {summary[-1]['synchronized']}/10 timing passes\n"
                            f"Peak overshoot {summary[-1]['max_overshoot_deg']:.3f}°\nLimits: "
                            f"{latest_limits[0]:g}° J1–J3 / {latest_limits[3]:g}° J4–J6")
            for ax, data, title in zip(axes, [base, latest], ['Baseline: Kp 54 on all joints', latest_title]):
                values = np.array([[j['settling_s'] for j in row['joints']] for row in data])
                mesh = ax.imshow(values, aspect='auto', vmin=2.2, vmax=4.6, cmap='viridis_r')
                for i in range(10):
                    for j in range(6):
                        ax.text(j, i, f'{values[i, j]:.2f}', ha='center', va='center',
                                color='white' if values[i, j] > 3.6 else 'black', fontsize=10)
                ax.set(title=title, xticks=range(6), xticklabels=[f'J{j+1}' for j in range(6)],
                       yticks=range(10), yticklabels=[f'Pose {i+1}' for i in range(10)])
                ax.set_title(title, fontsize=11)
            fig.colorbar(mesh, ax=axes, label='Settling time from SDK motion start (s)', shrink=.75)
            fig.suptitle('Measured joint settling for the same ten waypoint moves\n'
                         '±0.2° on J1–J3, ±1° on J4–J6; sustained ≥1 s; encoder timestamps', y=.97, fontsize=14)
            fig.savefig(DEST/'settling-table.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
