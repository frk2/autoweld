import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
import cv2,numpy as np
cv2.setNumThreads(2)
root=Path('/home/faraz/Code/autoweld/evidence/reflective-tubes-20261030')
files=sorted(list((root/'raw-bracket').glob('*/snapshot.json'))+list((root/'raw-bracket-high').glob('*/snapshot.json')))
groups={}
for p in files:
 s=json.loads(p.read_text());controls=s['metadata'].get('controls')or{}
 key=(float(controls.get('exposure_us',-1)),float(controls.get('gain_db',-1)))
 groups.setdefault(key,[]).append(p)
result=[]
for key,paths in groups.items():
 if len(paths)<3:continue
 paths=paths[:3];snaps=[json.loads(p.read_text())for p in paths]
 a=np.stack([cv2.imread(str(p.with_name('native.png')),0)for p in paths]).astype(float)
 median=np.median(a,axis=0)
 stats={}
 regions={'left':(350,470,225,315),'right':(700,1200,215,280),'root_right_bridge':(475,550,230,265)}
 for name,(x0,x1,y0,y1)in regions.items():
  roi=median[y0:y1,x0:x1]
  peaks=roi.max(axis=0);positions=roi.argmax(axis=0)+y0
  bg=np.concatenate([a[:,180:210,x0:x1],a[:,300:330,x0:x1]],axis=1)
  baseline=np.median(bg,axis=(0,1));noise=float(np.std(bg[0]-bg[1])/np.sqrt(2))
  contrast=peaks-baseline
  temporal_pixel_noise=float(np.sqrt(np.mean(np.var(a[:,y0:y1,x0:x1],axis=0,ddof=1))))
  stats[name]={'native_roi':[x0,y0,x1,y1],'peak_p10_50_90':np.percentile(peaks,[10,50,90]).tolist(),'contrast_p10_50_90':np.percentile(contrast,[10,50,90]).tolist(),'bg_median_dn':float(np.median(baseline)),'bg_temporal_sigma_dn':noise,'signal_over_temporal_noise_p10_50_90':np.percentile(contrast/max(noise,.1),[10,50,90]).tolist(),'signal_roi_temporal_noise_rms_dn':temporal_pixel_noise,'peak_v_p10_50_90':np.percentile(positions,[10,50,90]).tolist(),'raw_saturated_fraction':float(np.mean(a[:,y0:y1,x0:x1]>=254))}
 clouds=[s.get('cloud')or{}for s in snaps]
 qs=np.asarray([json.loads(p.with_name('robot-status.json').read_text())['q_rad']for p in paths])
 result.append({'exposure_us':key[0],'gain_db':key[1],'frame_count':3,'paths':[str(p.relative_to(root))for p in paths],'accepted_counts':[c.get('count')for c in clouds],'within_group_max_joint_range_rad':float(np.ptp(qs,axis=0).max()),'stats':stats})
report={'schema':'offline-native-bracket-statistics.v1','diagnostic_only':True,'regions_are_manual_analysis_not_detection_roi':True,'all_blas_workers':1,'profiles':result}
(root/'offline-raw-statistics.json').write_text(json.dumps(report,indent=2)+'\n')
for r in result:
 print(f"{r['exposure_us']:7.0f}us {r['gain_db']:4.0f}dB accepted{r['accepted_counts']} qspread{r['within_group_max_joint_range_rad']:.6f}")
 for key,s in r['stats'].items():
  print(' ',key,'peaks',np.round(s['peak_p10_50_90'],1),'contrast',np.round(s['contrast_p10_50_90'],1),'noise',round(s['bg_temporal_sigma_dn'],2),'snr',np.round(s['signal_over_temporal_noise_p10_50_90'],1),'sat%',round(s['raw_saturated_fraction']*100,2),'v',np.round(s['peak_v_p10_50_90'],1))
