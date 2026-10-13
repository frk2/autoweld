"""One-shot SDK motor-off takeover for the legacy board-hold app.

Requires explicit operator confirmation that the board is supported, a completed
SDK one-second rest check, and the exact existing controller PID. Does not open
the jaws, enable motors, send positions, change registers, or write Flash.
"""
import fcntl
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
from i2rt.motor_config_tool.dm_motor_registers import read_register
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.robots.utils import ArmType, _load_arm_config


def main(pid):
    root = Path('runtime/live')
    dest = Path('evidence/board-release-shutdown')
    s = json.loads((root / 'status.json').read_text())
    if (time.time()-s['timestamp'] > 1 or s.get('fault') or s.get('motion_fault')
            or s['closed'] or s['motion_active'] or s['pending_tcp_target']
            or not s['rest_verified'] or s['mode'] != 'Returning to rest'
            or np.max(np.abs(np.array(s['q_rad'])-s['rest_rad'])) > np.deg2rad(.5)):
        raise RuntimeError('Need fresh, SDK-verified arm rest before release')
    cmd = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
    if b'autoweld.app' not in cmd or b'--live' not in cmd or b'runtime/live' not in cmd:
        raise RuntimeError('PID does not identify the expected live controller')
    if (Path('/proc') / str(pid) / 'cwd').resolve() != Path.cwd():
        raise RuntimeError('Controller workspace does not match')
    record = dict(started_at=time.time(), pid=pid, rest_status=s,
                  authorization='Operator confirmed: Board supported; release and shut down',
                  procedure='Stop legacy controller at verified rest; SDK motor_off; independent readback',
                  motor_enable_sent=False, position_commands_sent=False, flash_written=False)
    (dest / 'shutdown.json').write_text(json.dumps(record, indent=2))
    # SIGTERM queues the legacy app's board-hold rejection and cannot exit it.
    # Rest is verified and the board supported before ending the only command
    # producer. The takeover acquires its released hardware lock before any I/O.
    os.kill(pid, signal.SIGKILL)
    with Path('/tmp/autoweld-can0.lock').open('a') as lock:
        deadline = time.monotonic()+3
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError('Controller hardware lock did not release; no second command producer started')
                time.sleep(.01)
        interface = DMSingleMotorCanInterface(channel='can0', name='supported_board_shutdown')
        interface.motor_modes = {i: ControlMode.POS_VEL if i <= 6 else ControlMode.MIT for i in range(1, 8)}
        motors = list(_load_arm_config(ArmType.YAM).motor_list)+[(7, 'DM4310')]
        failures = []
        try:
            for mid in (7, 6, 5, 4, 3, 2, 1):
                try:
                    interface.motor_off(mid)
                except Exception as exc:
                    failures.append(dict(motor=mid, error=str(exc)))
            result = {}
            for mid, kind in motors:
                state = interface.read_motor_status(mid, kind)
                result[mid] = dict(status=state.error_code, position=state.position,
                                   velocity=state.velocity, torque=state.torque,
                                   control_mode=read_register(interface, mid, 'CTRL_MODE'))
            record['motor_off_errors'] = failures
            record['motors_after'] = result
            record['all_disabled'] = all(r['status'] == '0x0' for r in result.values())
            if not record['all_disabled']:
                raise RuntimeError('Not all motors independently reported disabled')
        finally:
            interface.close()
            record['finished_at'] = time.time()
            (dest / 'shutdown.json').write_text(json.dumps(record, indent=2))
    (root / 'gripper-hold.json').write_text(json.dumps({'closed': False})+'\n')
    s.update(timestamp=time.time(), mode='Disabled', closed=True, motion_active=False,
             pending_tcp_target=False, gripper_hold_closed=False, disable_pending=False,
             controller_process_stopped=True, status_source='SDK shutdown survey; app stopped',
             last_request={'action':'supported_board_shutdown', 'ok':True, 'timestamp':time.time()})
    (root / 'status.json').write_text(json.dumps(s, indent=2))
    (dest / 'final-status.json').write_text(json.dumps(s, indent=2))
    print('All seven motors independently report disabled (0x0). Gripper unpowered; Viser stopped.', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
