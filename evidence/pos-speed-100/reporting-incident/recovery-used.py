import json,time,os,signal
from pathlib import Path
import numpy as np
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
from i2rt.robots.utils import ArmType,_load_arm_config,JointMapper
root=Path('/home/faraz/Code/autoweld')
rest=json.loads((root/'runtime/live/rest.json').read_text())
mapper=JointMapper({6:tuple(rest['gripper_limits_motor_rad'])},7)
target=mapper.to_robot_joint_pos_space(np.r_[rest['arm_rad'],rest['gripper_normalized']])
hw=_load_arm_config(ArmType.YAM); assert hw.directions==(1,1,1,1,1,1)
assert max(abs(target))<np.pi
motors=list(hw.motor_list)+[(7,'DM4310')]
i=DMSingleMotorCanInterface(channel='can0',name='passive_sdk_recovery')
ids={i.receive_mode.get_receive_id(mid):(mid,kind) for mid,kind in motors}
latest={};records=[];since=None;verified=False;deadline=time.monotonic()+10
try:
 while time.monotonic()<deadline:
  msg=i.try_receive_message(timeout=.01)
  if msg is None or msg.arbitration_id not in ids: continue
  mid,kind=ids[msg.arbitration_id];state=i.parse_recv_message(msg,kind,ignore_error=True)
  latest[mid]=(state,float(msg.timestamp))
  if len(latest)!=7:continue
  q=np.array([latest[mid][0].position for mid,_ in motors]);v=np.array([latest[mid][0].velocity for mid,_ in motors]);error=np.rad2deg(q-target)
  age=time.time()-min(x[1] for x in latest.values());valid=age<.05 and max(abs(error))<=.5 and max(abs(v))<=.1 and all(x[0].error_code=='0x1' for x in latest.values())
  since=(since if since is not None else time.monotonic()) if valid else None
  if not records or time.time()-records[-1]['timestamp']>.05:
   records.append({'timestamp':time.time(),'age_s':age,'q_motor_rad':q.tolist(),'velocity_rad_s':v.tolist(),'error_deg':error.tolist(),'status':[latest[mid][0].error_code for mid,_ in motors]})
  if since is not None and time.monotonic()-since>=1:
   verified=True
   pid=58301
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   assert b'autoweld.app' in cmd and b'runtime/live' in cmd
   assert json.loads((root/'runtime/live/status.json').read_text())['fault']=='Object of type bool is not JSON serializable'
   os.kill(pid,signal.SIGSTOP)
   i._drain_bus(timeout_s=.05)
   i.motor_modes={mid:('POS_VEL' if mid<7 else 'MIT') for mid,_ in motors}
   for mid,_ in motors: i.motor_off(mid)
   final={mid:i.read_motor_status(mid,kind).error_code for mid,kind in motors}
   assert all(x=='0x0' for x in final.values()),final
   (root/'evidence/pos-speed-100/sdk-recovery-disabled.json').write_text(json.dumps({'timestamp':time.time(),'rest_verified_seconds':time.monotonic()-since,'max_motor_error_deg':float(max(abs(error))),'motor_status':final,'stopped_app_pid':pid},indent=2))
   os.kill(pid,signal.SIGKILL)
   print('REST VERIFIED AND ALL MOTORS DISABLED',max(abs(error)),final,flush=True)
   break
 if not verified:raise RuntimeError('Passive SDK rest verification failed')
finally:
 i.close();(root/'evidence/pos-speed-100/passive-sdk-rest.json').write_text(json.dumps({'verified':verified,'samples':records},indent=2))
