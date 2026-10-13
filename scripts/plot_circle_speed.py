"""Offline circle tracking geometry and nine-window velocity plots."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from scripts.circle_metrics import circle_coordinates
from scripts.plot_velocity_windows import invalid_intervals, running_velocity

ROOT=Path('evidence/circle-speed')
WINDOWS=tuple(i/10 for i in range(2,11))


def main():
    out=ROOT/'plots'
    out.mkdir(exist_ok=True)
    circle=json.loads((ROOT/'circle.json').read_text())
    center,normal,start=(np.array(circle[k]) for k in ('center_m','normal','start_m'))
    model=tool_model()
    planner=MotionPlanner(model)
    records=[]
    chart_data=[]
    try:
        for path in sorted(ROOT.glob('circle60-*mms.jsonl'), key=lambda p:float(p.stem.split('-')[1][:-3])):
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            result=json.loads(path.with_suffix('.json').read_text())['result']
            plan=result['plan']
            stamps=np.array([r['feedback_timestamps'] for r in rows])
            ts=stamps-stamps[0,0]
            elapsed=np.array([r['elapsed_s'] for r in rows])
            offset=np.median(elapsed[elapsed>0]-ts[elapsed>0,0])
            common=ts[:,0]
            common=np.unique(common[(common>=max(ts[0])) & (common<=min(ts[-1]))])
            q=np.array([r['joint_position_rad'] for r in rows])
            v=np.array([r['joint_velocity_raw_rad_s'] for r in rows])
            qs=np.column_stack([np.interp(common,ts[:,j],q[:,j]) for j in range(6)])
            vs=np.column_stack([np.interp(common,ts[:,j],v[:,j]) for j in range(6)])
            t=common+offset
            pos=np.array([planner.fk(q)[:3,3] for q in qs])
            geo=circle_coordinates(pos,center,normal,start)
            jac=np.array([planner.jacobian(q)[:3] for q in qs])
            raw=np.einsum('nij,nj,ni->n',jac,vs,geo['tangent'])
            ref_arc=np.interp(t,elapsed,np.array([r['edge_fraction'] for r in rows])*plan['edge_length_m'])
            ref_pos=np.column_stack([np.interp(t,elapsed,np.array([r['reference_m'] for r in rows])[:,j]) for j in range(3)])
            bad=invalid_intervals(rows)
            averages={w:running_velocity(t,geo['tangential_m'],raw,ref_arc,w,invalid=bad) for w in WINDOWS}
            a,b=plan['ramp_s'],plan['duration_s']-plan['ramp_s']
            metric={'name':path.stem,'feed_mm_s':plan['feed_m_s']*1000,'completed':result['completed'],
                    'reason':result['reason'],'reference_cruise_s':b-a,'observed_end_s':float(t[-1]),
                    'max_edge_online_mm':result['max_cross_track_mm'],
                    'max_edge_aligned_mm':float(max(geo['edge_m'])*1000),
                    'rms_edge_aligned_mm':float(np.sqrt(np.mean(geo['edge_m']**2))*1000),
                    'p95_edge_aligned_mm':float(np.quantile(geo['edge_m'],.95)*1000),
                    'radial_range_mm':(np.array([min(geo['radial_m']),max(geo['radial_m'])])*1000).tolist(),
                    'max_abs_plane_mm':float(max(abs(geo['plane_m']))*1000),
                    'max_timed_error_online_mm':result['max_tracking_error_mm'],
                    'observed_measured_rotation_deg':float(np.rad2deg(geo['theta_rad'][-1]-geo['theta_rad'][0])),
                    'raw_feed_pass':result['passed'],'invalid_velocity_samples':result['invalid_velocity_samples'],'windows':{}}
            for w,avg in averages.items():
                mask=avg['valid'] & (avg['end_s']-w>=a) & (avg['end_s']<=b)
                i=int(np.flatnonzero(mask)[-1]) if mask.any() else None
                last=None
                if b-a>=w and t[-1]>=b:
                    last_window=running_velocity(t,geo['tangential_m'],raw,ref_arc,w,ends=[b],invalid=bad)
                    if last_window['valid'][0]:
                        last=float(last_window['position_mm_s'][0])
                metric['windows'][str(w)]={'final_cruise_avg_mm_s':last,
                    'latest_valid_cruise_avg_mm_s':float(avg['position_mm_s'][i]) if i is not None else None,
                    'latest_window_end_s':float(avg['end_s'][i]) if i is not None else None,
                    'latest_command_avg_mm_s':float(avg['command_mm_s'][i]) if i is not None else None,
                    'valid_cruise_windows':int(mask.sum())}
            records.append(metric)
            chart_data.append((metric,t,geo,averages,plan))
            ideal_angles=np.linspace(0,2*np.pi,401)
            ideal=center+circle['radius_m']*(np.cos(ideal_angles)[:,None]*geo['u']+np.sin(ideal_angles)[:,None]*geo['v'])
            fig=plt.figure(figsize=(17,10))
            ax=fig.add_subplot(231,projection='3d')
            ax.plot(*(ideal*1000).T,'k--',lw=1,label='Requested circle')
            ax.plot(*(pos*1000).T,color='#0072b2',lw=1.4,label='Measured encoder/FK TCP')
            ax.scatter(*(pos[[0,-1]]*1000).T,c=['green','red'],s=25)
            ax.set(xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title='TCP path in robot-base coordinates')
            # This circle has a narrow X extent; fixed padding avoids overlapping
            # ticks and keeps the projection legible without magnifying X error.
            ax.set_xlim(center[0]*1000-20,center[0]*1000+20)
            ax.set_xticks([round(center[0]*1000)-10,round(center[0]*1000)+10])
            ax.set_box_aspect([40,140,140])
            ax.legend(fontsize=7)
            ax=fig.add_subplot(232)
            ax.plot(np.cos(ideal_angles)*circle['radius_m']*1000,np.sin(ideal_angles)*circle['radius_m']*1000,'k--',label='Requested')
            ax.plot(*(geo['uv_m']*1000).T,color='#0072b2',label='Measured')
            ax.scatter(*(geo['uv_m'][[0,-1]]*1000).T,c=['green','red'],s=25)
            ax.axis('equal')
            ax.set(xlabel='Circle-plane u (mm)',ylabel='Circle-plane v (mm)',title='Planar path agreement; green=start, red=end')
            ax.legend(fontsize=8)
            ax=fig.add_subplot(233)
            ax.plot(elapsed,[r['cross_track_mm'] for r in rows],color='#0072b2',label='Online finite-circle distance')
            ax.axhline(5,color='red',ls='--',label='5 mm Hold threshold')
            ax.set(xlabel='Elapsed time (s)',ylabel='Deviation (mm)',title=f"Path deviation · max {metric['max_edge_online_mm']:.3f} mm")
            ax.legend(fontsize=8)
            ax=fig.add_subplot(234)
            ax.plot(t,geo['radial_m']*1000,label='Radial: negative=inside circle')
            ax.plot(t,geo['plane_m']*1000,label='Signed out-of-plane')
            ax.axhline(0,color='k',lw=.5)
            ax.set(xlabel='Feedback time (s)',ylabel='Signed deviation (mm)',title='Radial and plane errors, aligned encoder/FK')
            ax.legend(fontsize=8)
            ax=fig.add_subplot(235)
            for j,label in enumerate('XYZ'):
                ax.plot(t,(pos[:,j]-ref_pos[:,j])*1000,alpha=.7,label=f'{label} error')
            ax.plot(elapsed,[r['tracking_error_mm'] for r in rows],'k',lw=1.3,label='Timed Euclidean error')
            ax.set(xlabel='Feedback time (s)',ylabel='Error (mm)',title='Timed position error; informational, not the stop metric')
            ax.legend(fontsize=8,ncol=2)
            ax=fig.add_subplot(236)
            ax.plot(t,raw*1000,color='.75',lw=.6,alpha=.5,label='Raw tangential speed')
            for w in (.2,.5,1.):
                avg=averages[w]
                ax.plot(avg['center_s'],avg['position_mm_s'],label=f'{w:g} s actual tangential avg')
            avg=averages[.5]
            ax.plot(avg['center_s'],avg['command_mm_s'],'k--',lw=1,label='0.5 s command avg')
            ax.set(xlabel='Window midpoint (s)',ylabel='Tangential velocity (mm/s)',title='Curvature-aware velocity from encoder positions')
            ax.legend(fontsize=7,ncol=2)
            for ax in fig.axes:
                ax.grid(alpha=.2)
            fig.suptitle(f"{path.stem}: radius 60 mm · {'Completed' if result['completed'] else 'INTERRUPTED — 5 mm edge guard'}\nEncoder/model tracking measurements; no force-deflection compliance measurement",fontsize=14)
            fig.tight_layout(rect=(0,0,1,.94))
            fig.savefig(out/f'{path.stem}-tracking.png',dpi=140)
            plt.close(fig)
            fig,axes=plt.subplots(3,3,figsize=(16,11),sharex=True,sharey=True)
            for ax,w in zip(axes.flat,WINDOWS):
                avg=averages[w]
                ax.plot(t,raw*1000,color='.75',lw=.6,alpha=.4,label='Raw tangential speed')
                ax.plot(avg['center_s'],avg['position_mm_s'],color='#0072b2',lw=1.6,label='Measured running avg')
                ax.plot(avg['center_s'],avg['command_mm_s'],'k--',lw=1,label='Command avg — same interval')
                ax.fill_between(avg['center_s'],avg['command_mm_s']*.8,avg['command_mm_s']*1.2,color='green',alpha=.1)
                ax.axvspan(a,min(b,t[-1]),color='green',alpha=.04)
                info=metric['windows'][str(w)]
                val=info['final_cruise_avg_mm_s']
                if val is not None:
                    end=b
                    note=f'Final cruise avg: {val:.2f} mm/s'
                else:
                    val=info['latest_valid_cruise_avg_mm_s']
                    end=info['latest_window_end_s']
                    note='No valid full cruise window' if val is None else f'Latest observed avg: {val:.2f} mm/s'
                if val is not None:
                    ax.plot(end-w/2,val,'o',color='#0072b2',ms=4)
                if not result['completed']:
                    note+='\nInterrupted before circle completion'
                ax.text(.98,.04,note,transform=ax.transAxes,ha='right',fontsize=8,bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
                ax.set_title(f'{w:.1f} s window · available {w/2:.2f} s after midpoint',fontsize=10)
                ax.set(xlabel='Window midpoint (s)',ylabel='Tangential mm/s')
                ax.grid(alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3)
            fig.suptitle(f"{path.stem}: nine velocity windows · {'Completed' if result['completed'] else 'INTERRUPTED'}\nActual tangential travel uses measured radius; the requested arc uses nominal radius",fontsize=14)
            fig.tight_layout(rect=(0,0,1,.91))
            fig.savefig(out/f'{path.stem}-windows.png',dpi=140)
            plt.close(fig)
            print(path.stem,metric['max_edge_online_mm'],flush=True)
        (out/'metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        for m,t,g,avg,plan in chart_data:
            label=f"{m['feed_mm_s']:g} mm/s"+(' (Hold)' if not m['completed'] else '')
            axes[0,0].plot(*(g['uv_m']*1000).T,label=label)
            angle=np.rad2deg(g['theta_rad'])
            axes[0,1].plot(angle,g['edge_m']*1000,label=label)
            axes[1,0].plot(angle,g['radial_m']*1000,label=label)
            a=avg[1.]
            axes[1,1].plot(a['center_s'],a['position_mm_s'],label=label)
        axes[0,0].plot(np.cos(ideal_angles)*60,np.sin(ideal_angles)*60,'k--',label='Requested circle')
        axes[0,0].set(xlabel='Circle-plane u (mm)',ylabel='Circle-plane v (mm)',title='Path agreement')
        axes[0,0].axis('equal')
        axes[0,1].axhline(5,color='red',ls='--',label='5 mm guard')
        axes[0,1].set(xlabel='Measured angle around circle (degrees)',ylabel='Deviation (mm)',title='Distance to requested circle')
        axes[1,0].set(xlabel='Measured angle around circle (degrees)',ylabel='Radial error (mm)',title='Negative means inside the requested circle')
        axes[1,1].set(xlabel='Window midpoint (s)',ylabel='Tangential speed (mm/s)',title='1 s running averages — actual tangential travel')
        for ax in axes.flat:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle('120 mm diameter circle: measured tracking by speed',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(out/'circle-comparison.png',dpi=150)
        plt.close(fig)
    finally:
        Path(model).unlink()


if __name__=='__main__':
    main()
