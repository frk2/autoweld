"""Offline oriented Stage 3 deviation, tool-normal and velocity evidence."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking import load_tracks

ROOT=Path('evidence/stage3-welding')


def main():
    items=json.loads((ROOT/'preflight.json').read_text())
    records=[]
    paths=sorted(ROOT.glob('perpendicular-line-*-20mms.jsonl'))
    model=tool_model()
    planner=MotionPlanner(model)
    fig,axes=plt.subplots(len(paths),3,figsize=(16,len(paths)*2.8),squeeze=False)
    geometry=plt.figure(figsize=(16,6))
    try:
        for i,path in enumerate(paths):
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            result=json.loads(path.with_suffix('.json').read_text())['result']
            item=next(x for x in items if path.stem.startswith(x['name']))
            curve=load_tracks(item['asset'])[0]
            t=np.array([r['elapsed_s'] for r in rows])
            T=np.array([planner.fk(r['joint_position_rad']) for r in rows])
            points=T[:,:3,3]
            torch=-T[:,:3,2]
            normal_error=np.rad2deg(np.arcsin(np.clip(abs(torch@curve.direction),0,1)))
            result_small={k:result[k] for k in ['completed','passed','reason','max_cross_track_mm','max_tracking_error_mm','max_orientation_error_deg','invalid_velocity_samples','sampling_ok']}
            record={'name':path.stem,**result_small,'max_tool_normal_error_deg':float(max(normal_error))}
            record['max_orientation_error_all_deg']=max(r['orientation_error_deg'] for r in rows)
            record['minimum_progress_mm']=float(min((points-curve.start)@curve.direction)*1000)
            records.append(record)
            ax=axes[i,0]
            ax.plot(t,[r['cross_track_mm'] for r in rows],label='Finite-edge distance')
            ax.plot(t,[r['tracking_error_mm'] for r in rows],alpha=.5,ls=':',label='Timed error; informational')
            ax.axhline(5,color='red',ls='--',label='5 mm guard')
            ax.set_ylabel('mm')
            ax=axes[i,1]
            ax.plot(t,[r['orientation_error_deg'] for r in rows],label='Full orientation error')
            ax.plot(t,normal_error,label='Deviation from 90° to line')
            ax.axhline(5,color='red',ls='--',label='5° criterion')
            ax.set_ylabel('degrees')
            ax=axes[i,2]
            ax.plot(t,[r['raw_along_speed_m_s']*1000 for r in rows],alpha=.4,label='Raw signed along-path speed')
            valid=[r['position_velocity'] for r in rows if r.get('position_velocity')]
            if valid:
                ax.plot([v['elapsed_s'] for v in valid],[v['along_speed_m_s']*1000 for v in valid],label='120 ms position slope')
            plan=result['plan']
            phase=np.clip(np.minimum(t,plan['duration_s']-t)/plan['ramp_s'],0,1)
            ax.plot(t,20*(10*phase**3-15*phase**4+6*phase**5),'k--',label='Reference')
            ax.set_ylabel('mm/s')
            for ax in axes[i]:
                ax.set_title(path.stem+(' — completed' if result['completed'] else ' — guard Hold'),fontsize=10)
                ax.set_xlabel('Time (s)')
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            ax=geometry.add_subplot(1,len(paths),i+1,projection='3d')
            ref=np.array([curve.point(s) for s in np.linspace(0,1,41)])
            ax.plot(*(ref*1000).T,'k--',label='Requested edge')
            ax.plot(*(points*1000).T,label='Measured TCP')
            ix=np.linspace(0,len(points)-1,5,dtype=int)
            ax.quiver(*(points[ix]*1000).T,*torch[ix].T,length=4,normalize=True,color='orange')
            ax.set(xlabel='X mm',ylabel='Y mm',zlabel='Z mm',title=f"Line {i+1}\nnormal error ≤{max(normal_error):.2f}°")
            ax.set_box_aspect([1,1,1])
            ax.tick_params(labelsize=6)
        fig.suptitle('Stage 3 · five 20 mm lines at 20 mm/s · perpendicular tool orientation',fontsize=15)
        fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(ROOT/'oriented-lines.png',dpi=140)
        plt.close(fig)
        geometry.suptitle('Requested / measured paths; orange arrows show measured torch direction',fontsize=15)
        geometry.tight_layout(rect=(0,0,1,.95))
        geometry.savefig(ROOT/'tool-orientation.png',dpi=150)
        plt.close(geometry)
        (ROOT/'metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
        print(json.dumps(records,indent=2))
    finally:
        Path(model).unlink()


if __name__=='__main__':
    main()
