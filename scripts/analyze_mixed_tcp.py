"""Offline timestamp-aligned FK shaking at the modeled TCP and gripper mount."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner

FS = 50.


def split_valid(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def vector_rms(value):
    return float(np.sqrt(np.mean(np.sum(value**2, axis=1))))


def filter_valid(t, values, valid, high):
    result = np.full_like(values, np.nan)
    trusted = np.zeros(len(t), bool)
    for a, b in split_valid(valid):
        if b-a < 125:
            continue
        result[a:b] = sosfiltfilt(butter(4, [2, high], btype='bandpass', fs=FS, output='sos'),
                                values[a:b], axis=0)
        trusted[a+50:b-50] = True
    return result, trusted


def load_trial(path, planner):
    trial = json.loads(path.read_text())
    rows = [json.loads(x) for x in path.with_name(path.stem+'-control.jsonl').read_text().splitlines()]
    session = trial['after']['motion_reference']['session']
    duration = trial['after']['motion_reference']['duration_s']
    refs = [r for r in rows if (r.get('motion_reference') or {}).get('session') == session]
    start = float(np.median([r['timestamp']-r['motion_reference']['elapsed_s'] for r in refs]))
    # Anchor every run's grid to its reference start, so common masks select
    # exactly the same path phases, not just similar total durations.
    t = np.arange(-200, int(np.ceil((duration+5)*FS))+1)/FS
    q = np.empty((len(t), 6))
    valid = np.ones(len(t), bool)
    timing = []
    for j in range(6):
        jt = np.array([r['feedback_timestamps'][j]-start for r in rows])
        jq = np.array([r['q'][j] for r in rows])
        jt, ix = np.unique(jt, return_index=True)
        jq = jq[ix]
        q[:,j] = np.interp(t, jt, jq)
        hi = np.searchsorted(jt, t, side='right').clip(1, len(jt)-1)
        valid &= (t >= jt[0]) & (t <= jt[-1]) & ((jt[hi]-jt[hi-1]) <= .05)
        timing.append(dict(joint=j+1, native_rate_hz=float(1/np.median(np.diff(jt))),
                           gaps_s=[[float(x),float(y)] for x,y in zip(jt,jt[1:]) if y-x>.05]))
    # Reference q carries its own executor elapsed time. Publication time can
    # jitter by milliseconds and would invent apparent reference shaking.
    rt, ix = np.unique([r['motion_reference']['elapsed_s'] for r in refs], return_index=True)
    rq = np.array([r['motion_reference']['q'][:6] for r in refs])[ix]
    reference_q = np.column_stack([np.interp(t, rt, rq[:,j]) for j in range(6)])
    transforms = np.array([planner.fk(x) for x in q])
    tcp = transforms[:,:3,3]*1000
    mount = tcp + transforms[:,:3,2]*150
    reference_tcp = np.array([planner.fk(x)[:3,3]*1000 for x in reference_q])
    trace = [json.loads(x) for x in path.with_suffix('.jsonl').read_text().splitlines()]
    check = max(np.linalg.norm(planner.fk(r['joint_position_rad'])[:3,3]-r['actual_m'])*1000
                for r in trace[::50])
    assert check < 1e-6, 'FK no longer reproduces the recorded TCP geometry'
    bands = {}
    for high in (20.,10.):
        # Include reference-subtracted and raw-motion bands to check that
        # intentional path motion is not being mistaken for shaking.
        filtered, trusted = filter_valid(t, np.c_[tcp, mount, tcp-reference_tcp, reference_tcp], valid, high)
        bands[high] = dict(filtered=filtered, trusted=trusted)
    return dict(name=path.stem, controller=trial['after']['controller'], fixture=trial['fixture']['name'],
                duration_s=duration, ramp_s=trial['result']['plan']['ramp_s'], t=t, bands=bands,
                max_fk_reproduction_error_mm=float(check), timing=timing,
                q=q, reference_q=reference_q, valid=valid, start=start)


def main(folder='evidence/mixed-wrist'):
    dest = Path(folder)/'tcp-shaking'
    dest.mkdir(exist_ok=True)
    model = tool_model()
    try:
        planner = MotionPlanner(model)
        runs = [load_trial(p, planner) for p in sorted(Path(folder).glob('*-r[12].json'))]
    finally:
        Path(model).unlink()
    records = []
    for fixture in ('line120','circle60'):
        matched = [r for r in runs if r['fixture']==fixture]
        for run in matched:
            np.testing.assert_array_equal(run['t'], matched[0]['t'])
        for high in (20.,10.):
            common = np.logical_and.reduce([r['bands'][high]['trusted'] for r in matched])
            for run in matched:
                duration, ramp, t = run['duration_s'], run['ramp_s'], run['t']
                # Cruise interior excludes a further one second around ramps,
                # matching the conservative convention used in camera analysis.
                for phase, (lo, hi) in {'path':(0,duration), 'cruise':(ramp+1,duration-ramp-1)}.items():
                    keep = common & (t>=lo) & (t<=hi)
                    signal = run['bands'][high]['filtered']
                    item = dict(name=run['name'], controller=run['controller'], fixture=fixture,
                                band_hz=[2,high], phase=phase, samples=int(keep.sum()),
                                coverage_s=float(keep.sum()/FS), interval_s=[lo,hi],
                                coverage_fraction=float(min(1.,keep.sum()/FS/(hi-lo))),
                                first_sample_s=float(t[keep][0]), last_sample_s=float(t[keep][-1]),
                                max_fk_reproduction_error_mm=run['max_fk_reproduction_error_mm'])
                    for key, sl in [('tcp',slice(0,3)), ('mount',slice(3,6)),
                                    ('tcp_error',slice(6,9)), ('reference',slice(9,12))]:
                        values = signal[keep,sl]
                        item[key+'_rms_mm'] = vector_rms(values)
                        item[key+'_axis_rms_mm'] = np.sqrt(np.mean(values**2,axis=0)).tolist()
                        item[key+'_axis_peak_to_peak_mm'] = np.ptp(values,axis=0).tolist()
                    powers=[]
                    for a,b in split_valid(keep):
                        if b-a>=200:
                            f,p = welch(signal[a:b,:3], fs=FS, nperseg=200, noverlap=100, axis=0)
                            powers.append((p.sum(axis=1),b-a))
                    item['tcp_peak_hz'] = None
                    if powers:
                        power = np.average([x[0] for x in powers], weights=[x[1] for x in powers], axis=0)
                        band=(f>=2)&(f<=high)
                        item['tcp_peak_hz']=float(f[band][np.argmax(power[band])])
                    records.append(item)
                if high==20:
                    run['common_mask']=common & (t>=0) & (t<=duration)
                    np.savetxt(dest/(run['name']+'-signals.csv'),
                        np.c_[t[run['common_mask']], run['bands'][high]['filtered'][run['common_mask']]],
                        delimiter=',',header='time_s,tcp_x_mm,tcp_y_mm,tcp_z_mm,mount_x_mm,mount_y_mm,mount_z_mm,'
                        'tcp_error_x_mm,tcp_error_y_mm,tcp_error_z_mm,reference_x_mm,reference_y_mm,reference_z_mm',comments='')
    (dest/'metrics.json').write_text(json.dumps(records,indent=2))
    (dest/'timing.json').write_text(json.dumps([{k:r[k] for k in ('name','timing')} for r in runs],indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for row,fixture in enumerate(('line120','circle60')):
        for run in [r for r in runs if r['fixture']==fixture]:
            color='#0072b2' if run['controller']=='POS_VEL' else '#d55e00'
            label=('Position–speed' if run['controller']=='POS_VEL' else 'Mixed MIT/gravity')+' '+run['name'].rsplit('-',1)[1]
            keep=run['common_mask']
            v=run['bands'][20.]['filtered'][keep,:3]
            axes[row,0].plot(run['t'][keep],np.linalg.norm(v,axis=1),color=color,alpha=.7,lw=.9,label=label)
        axes[row,0].set(title='Line at 20 mm/s' if row==0 else 'Circle at 50 mm/s',
                        xlabel='Reference time (s)',ylabel='TCP 2–20 Hz magnitude (mm)')
        for i,mode in enumerate(('POS_VEL','MIXED_J45')):
            r=[x for x in records if x['fixture']==fixture and x['controller']==mode and x['band_hz']==[2,20.] and x['phase']=='path']
            y=[x['tcp_rms_mm'] for x in r]
            color='#0072b2' if i==0 else '#d55e00'
            axes[row,1].bar(i,np.mean(y),color=color,alpha=.8,width=.55)
            axes[row,1].plot([i-.07,i+.07],y,'ko',ms=5)
            axes[row,1].text(i,max(y)*1.04,f'{np.mean(y):.3f} mm',ha='center')
        axes[row,1].set(xticks=[0,1],xticklabels=['Position–speed','Mixed MIT/gravity'],
                        ylabel='3D TCP vector RMS (mm)',ylim=(0,None))
        row_metrics=[x['tcp_rms_mm'] for x in records if x['fixture']==fixture
                     and x['band_hz']==[2,20.] and x['phase']=='path']
        axes[row,1].set_ylim(0,max(row_metrics)*1.25)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in axes[:,0]:
        ax.legend(fontsize=8)
    fig.suptitle('Encoder-derived end-effector shaking: identical valid time windows across both modes\n'
                 '2–20 Hz; modeled TCP 150 mm from gripper mount; dots show two repeats; no camera measurement')
    fig.savefig(dest/'tcp-comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    main()
