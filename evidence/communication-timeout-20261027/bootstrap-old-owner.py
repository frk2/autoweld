"""One-time bootstrap of an old failed service, using read-only SDK proof."""
import json,os,signal,time,xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
import numpy as np
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
from i2rt.motor_config_tool.dm_motor_registers import read_register_sample
from i2rt.robots.utils import ArmType,_load_arm_config
pid=1861887
root=Path('evidence/communication-timeout-20261027')
args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
assert b'autoweld.app' in args and b'--live' in args and b'runtime/agents-live' in args
s=json.loads(Path('runtime/agents-live/status.json').read_text())
assert s['shutdown']['phase']=='blocked' and 'SDK control thread stopped' in s['shutdown']['error']
assert time.time()-s['temperatures']['timestamp']>5
rest=np.asarray(json.loads(Path('runtime/agents-live/rest.json').read_text())['arm_rad'])
cal=json.loads(Path('assets/camera_mount/joint-calibration.json').read_text())
offsets=np.asarray(cal['fixed_motor_offsets_rad'])
hw=_load_arm_config(ArmType.YAM)
assert [mid for mid,_ in hw.motor_list]==list(range(1,7))
model=ET.parse('runtime/models/receiver-camera.xml').getroot()
bounds=np.array([[float(v) for v in model.find(f'.//joint[@name="joint{i}"]').attrib['range'].split()] for i in range(1,7)])
os.kill(pid,signal.SIGSTOP)
deadline=time.monotonic()+2
while not all('State:\tT' in p.read_text() for p in Path(f'/proc/{pid}/task').glob('*/status')):
    if time.monotonic()>deadline: raise RuntimeError('Old owner did not stop; no kill or motor mutation issued')
    time.sleep(.01)
reader=DMSingleMotorCanInterface(channel='can0',name='failed_owner_bootstrap_readonly')
records=[]
try:
    started=time.monotonic()
    while True:
        frames=[asdict(reader.read_motor_status(mid,kind,quiet_disabled=True)) for mid,kind in hw.motor_list]
        native=[read_register_sample(reader,mid,'p_m') for mid,_ in hw.motor_list]
        now=time.time()
        raw=np.array([f['position'] for f in frames]); q=(raw-offsets)*np.asarray(hw.directions)
        assert np.isfinite([[f['position'],f['velocity'],f['timestamp'],f['temperature_mos'],f['temperature_rotor']] for f in frames]).all()
        assert all(f['error_code'] in ('0x0','0xd') and 0<=now-f['timestamp']<=.2 and abs(f['velocity'])<=.05 for f in frames)
        assert all(0<=now-n['timestamp']<=.2 for n in native)
        assert max(abs(raw-np.array([n['value'] for n in native])))<=.02
        assert np.all(q>=bounds[:,0]) and np.all(q<=bounds[:,1])
        assert max(abs(q-rest))<=.15
        assert max(max(f['temperature_mos'],f['temperature_rotor']) for f in frames)<55
        records.append(dict(timestamp=now,q_rad=q.tolist(),rest_delta_rad=(q-rest).tolist(),frames=frames,native=native))
        if len(records)>1:
            assert max(np.ptp(np.array([r['q_rad'] for r in records]),axis=0))<=.005
        if time.monotonic()-started>=1.2:break
        time.sleep(.02)
    report=dict(old_pid=pid,old_threads_stopped=True,read_only=True,quiet_duration_s=time.monotonic()-started,samples=records,near_rest_limit_rad=.15,hardware_mutations=False)
    (root/'bootstrap-disabled-proof.json').write_text(json.dumps(report,indent=2))
    os.kill(pid,signal.SIGKILL)
    print(json.dumps(dict(event='failed_owner_stopped',old_pid=pid,samples=len(records),quiet_duration_s=report['quiet_duration_s'],q_rad=records[-1]['q_rad'],maximum_rest_delta_deg=float(np.rad2deg(np.max(abs(q-rest)))),motor_status=[f['error_code'] for f in frames],hardware_mutations=False)))
finally:
    reader.close()
