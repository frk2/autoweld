"""Exclusive, disabled-only KP_APR RAM adjustment using SDK register APIs."""

import fcntl
import json
import math
import time
from pathlib import Path

import tyro
from i2rt.motor_config_tool.dm_motor_registers import read_register, write_register
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
from i2rt.motor_drivers.position_speed import SURVEY_REGISTERS
from i2rt.motor_drivers.utils import MotorType
from i2rt.robots.utils import ArmType, _load_arm_config

REGISTERS = (*SURVEY_REGISTERS, 'UV_Value', 'OT_Value', 'OC_Value', 'OV_Value',
             'I_BW', 'MST_ID', 'ESC_ID', 'can_br', 'sw_ver')


def check_step(current, target):
    if len(current) != 6 or len(target) != 6:
        raise ValueError('Exactly six arm gains required')
    for old, new in zip(current, target):
        if not all(math.isfinite(v) and v > 0 for v in (old, new)):
            raise ValueError('KP_APR must be finite and positive')
        if abs(new/old - 1.) > .2000001:
            raise ValueError('KP_APR change exceeds 20% of current read-back')


def survey(interface, motors):
    result = {}
    for mid, kind in motors:
        state = interface.read_motor_status(mid, kind)
        if state.error_code != '0x0' or abs(state.velocity) > .05:
            raise RuntimeError(f'Motor {mid} must be disabled, fault-free and stationary')
        regs = {r: read_register(interface, mid, r) for r in REGISTERS}
        const = MotorType.get_motor_constants(kind)
        for r, value in [('PMAX', const.POSITION_MAX), ('VMAX', const.VELOCITY_MAX),
                         ('TMAX', const.TORQUE_MAX), ('CTRL_MODE', 2 if mid <= 6 else 1)]:
            if not math.isclose(regs[r], value, rel_tol=1e-5):
                raise RuntimeError(f'Motor {mid}: unexpected {r} scaling/mode')
        result[mid] = dict(status=state.error_code, position=state.position,
                           velocity=state.velocity, registers=regs,
                           temperature_mos_c=getattr(state, 'temperature_mos', None),
                           temperature_rotor_c=getattr(state, 'temperature_rotor', None))
    return result


def main(target: tuple[float, ...], expected: tuple[float, ...], output: str):
    check_step(expected, target)  # Reject malformed requests before opening CAN.
    dest = Path(output)
    if dest.exists():
        raise FileExistsError(dest)
    record = dict(requested=list(target), expected=list(expected), flash_written=False,
                  writes=[], started_at=time.time())
    with Path('/tmp/autoweld-can0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        motors = list(_load_arm_config(ArmType.YAM).motor_list) + [(7, 'DM4310')]
        interface = DMSingleMotorCanInterface(channel='can0', name='disabled_kp_apr_tuning')
        try:
            before = survey(interface, motors)
            record['before'] = before
            current = [before[i]['registers']['KP_APR'] for i in range(1, 7)]
            check_step(current, target)
            if not all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(current, expected)):
                raise RuntimeError('Current gains differ from expected; no writes issued')
            dest.write_text(json.dumps(record, indent=2))
            for mid, new in enumerate(target, 1):
                if math.isclose(new, current[mid-1], rel_tol=1e-7):
                    continue
                # Recheck the target actuator immediately before each mutation.
                status = interface.read_motor_status(mid, motors[mid-1][1])
                if status.error_code != '0x0' or abs(status.velocity) > .05:
                    raise RuntimeError('Disabled/stationary state changed before write')
                write_register(interface, mid, 'KP_APR', float(new))
                actual = read_register(interface, mid, 'KP_APR')
                record['writes'].append(dict(motor_id=mid, requested=new, readback=actual))
                dest.write_text(json.dumps(record, indent=2))
                if not math.isclose(actual, new, rel_tol=1e-6):
                    raise RuntimeError(f'Motor {mid}: KP_APR independent read-back mismatch')
            after = survey(interface, motors)
            record['after'] = after
            for mid, _ in motors:
                for reg in REGISTERS:
                    wanted = target[mid-1] if reg == 'KP_APR' and mid <= 6 else before[mid]['registers'][reg]
                    if not math.isclose(after[mid]['registers'][reg], wanted, rel_tol=1e-6):
                        raise RuntimeError(f'Motor {mid}: unrelated register {reg} changed')
            record['verified'] = True
            print('Verified KP_APR RAM:', [after[i]['registers']['KP_APR'] for i in range(1, 7)], flush=True)
        except Exception as exc:
            record['error'] = str(exc)
            raise
        finally:
            interface.close()
            record['finished_at'] = time.time()
            dest.write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    tyro.cli(main)
