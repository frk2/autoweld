"""Offline audit of recorded trajectories; never opens motors or sends commands."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from i2rt.robots.waypoint_motion import FeedJointPath
from scipy.optimize import brentq

from autoweld.app import tool_model
from autoweld.diagnostics import solve_pose
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks

ROOT = Path('evidence/path-deviation-repeat')
DEST = Path('evidence/trajectory-audit')


def main():
    manifest = json.loads((ROOT / 'preflight.json').read_text())
    model = tool_model()
    planner = MotionPlanner(model)
    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    try:
        for row_index, index in enumerate((0, 1)):
            item = manifest[index]
            name = item['name'] + '-attempt1'
            record = json.loads((ROOT / (name + '.json')).read_text())
            rows = [json.loads(x) for x in (ROOT / (name + '.jsonl')).read_text().splitlines()]
            control = [json.loads(x) for x in (ROOT / (name + '-control.jsonl')).read_text().splitlines()]
            report = record['result']['plan']
            curve = load_tracks(item['asset'])[item['index']]
            seed = np.r_[record['start']['target_rad'], record['before']['target_rad'][6:]]
            # Controller status has six arm targets; use the original fixed gripper.
            if len(seed) == 6:
                seed = np.r_[seed, item['seed_q'][6]]
            knots = np.linspace(0, 1, max(41, min(801, int(np.ceil(report['tcp_distance_m']/.0005))+1)))
            q = planner._configuration(seed)
            positions = []
            rotation = planner.fk(seed)[:3, :3].copy()
            for s in knots:
                fraction = (s*report['tcp_distance_m']-report['lead_in_out_m'])/curve.length
                pose = curve.pose(fraction, rotation, item['orientation'])
                q = solve_pose(planner.kin, pose, planner.site_name, q, stage='offline_reconstruction')
                positions.append(np.r_[q[:6], seed[6]])
            path = FeedJointPath(knots, np.array(positions), report['duration_s'], report['ramp_s']/report['duration_s'])
            session = record['after']['motion_reference']['session']
            control = [r for r in control if (r.get('motion_reference') or {}).get('session') == session]
            difference = np.array([path.sample(r['motion_reference']['elapsed_s'])[0][:6]-r['motion_reference']['q'][:6] for r in control])
            reconstruction_error = float(np.max(abs(difference)))
            assert reconstruction_error < 1e-5, (name, reconstruction_error)
            elapsed = np.array([r['elapsed_s'] for r in rows])
            actual_q = np.array([r['joint_position_rad'] for r in rows])
            reference_q = np.array([path.sample(t)[0][:6] for t in elapsed])
            maximum = int(np.argmax([r['cross_track_mm'] for r in rows]))
            command_span = np.rad2deg(path.positions[-1,:6]-path.positions[0,:6])
            error_at_max = np.rad2deg(actual_q[maximum]-reference_q[maximum])
            dense_t = np.linspace(0, path.duration, 4001)
            reference_fk = np.array([planner.fk(q)[:3,3] for q in reference_q])
            reference_edge = np.array([curve.distance(p)*1000 for p in reference_fk])
            # At interior knots q, dq/ds, d²q/ds² are continuous. The only finite
            # time-jerk jump is Δ(d³q/ds³) * (ds/dt)^3.
            jump_times = [brentq(lambda t: path.phase(t)[0]-s, 0, path.duration) for s in knots[1:-1]]
            jumps = np.array([(path.spline(s+1e-10,3)-path.spline(s-1e-10,3))[:6]*path.phase(t)[1]**3
                              for s,t in zip(knots[1:-1],jump_times)])
            derivatives = np.array([path.sample(t)[1:] for t in dense_t])[:,:,:6]
            timing = {key: {'p50_ms':float(np.median([r['sdk_timing'][key] for r in control])*1000),
                            'max_observed_ms':float(max(r['sdk_timing'][key] for r in control)*1000)}
                      for key in ('host_period_s','cycle_period_s','feedback_skew_s','command_age_s')}
            entry = {'name':name,'command_joint_delta_deg':command_span.tolist(),
                     'max_reference_reconstruction_error_rad':reconstruction_error,
                     'max_reference_edge_distance_mm':float(max(reference_edge)),
                     'worst_edge_time_s':float(elapsed[maximum]),
                     'max_measured_edge_mm':rows[maximum]['cross_track_mm'],
                     'joint_error_at_worst_edge_deg':error_at_max.tolist(),
                     'max_joint_jerk_rad_s3':float(np.max(abs(derivatives[:,2]))),
                     'max_cubic_knot_jerk_jump_rad_s3':float(np.max(abs(jumps))),
                     'max_reference_nominal_4ms_step_deg':float(np.rad2deg(np.max(abs(derivatives[:,0]))*.004)),
                     'actual_start_joint_error_deg':np.rad2deg(actual_q[0]-reference_q[0]).tolist(),
                     'sampled_timing':timing,
                     'note':'Reference reconstructed offline and checked against recorded SDK reference. Timing is sampled, not every CAN exchange.'}
            results.append(entry)
            ax=axes[row_index,0]
            ax.plot(elapsed,[r['cross_track_mm'] for r in rows],label='Measured edge distance')
            ax.plot(elapsed,reference_edge,'--',label='Reference edge distance')
            ax.set(title=f"R{curve.radius*1000:g} mm / 5° at {report['feed_m_s']*1000:g} mm/s",ylabel='mm',xlabel='Time (s)')
            ax=axes[row_index,1]
            for joint,color in ((0,'tab:blue'),(4,'tab:orange')):
                ax.plot(elapsed,np.rad2deg(reference_q[:,joint]-reference_q[0,joint]),'--',c=color,label=f'J{joint+1} reference')
                ax.plot(elapsed,np.rad2deg(actual_q[:,joint]-reference_q[0,joint]),c=color,label=f'J{joint+1} measured')
            ax.axvline(elapsed[maximum],c='.5',ls=':',label='Largest edge error')
            ax.set(title='Different joint responses distort the TCP path',ylabel='Joint change (degrees)',xlabel='Time (s)')
            ax=axes[row_index,2]
            peak_joint=int(np.argmax(np.max(abs(derivatives[:,2]),axis=0)))
            ax.plot(dense_t,derivatives[:,2,peak_joint],label=f'J{peak_joint+1} reference jerk')
            ax.scatter(jump_times,[path.sample(t)[3][peak_joint] for t in jump_times],s=8,label='Cubic spline knots')
            ax.set(title=f"Largest knot jerk jump: {entry['max_cubic_knot_jerk_jump_rad_s3']:.4f} rad/s³",ylabel='rad/s³',xlabel='Time (s)')
            print(json.dumps(entry,indent=2),flush=True)
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('Recorded center-facing arcs: reference smoothness versus physical joint tracking\nOffline replay only; measured motion is encoder/FK, without external tip metrology',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.92))
        fig.savefig(DEST/'reference-vs-tracking.png',dpi=140)
        (DEST/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    finally:
        Path(model).unlink()
        plt.close(fig)


if __name__ == '__main__':
    main()
