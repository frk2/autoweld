"""Read-only SDK survey. Requires the live app to be disabled and stopped."""

import fcntl
import json
from pathlib import Path

import tyro
from i2rt.motor_config_tool.dm_motor_registers import read_register
from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface
from i2rt.motor_drivers.position_speed import SURVEY_REGISTERS
from i2rt.robots.utils import ArmType, _load_arm_config


def main(output: str = 'evidence/pos-speed/motor-registers-before.json'):
    with Path('/tmp/autoweld-can0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = list(_load_arm_config(ArmType.YAM).motor_list) + [(7, 'DM4310')]
        result = {}
        interface = DMSingleMotorCanInterface(channel='can0', name='position_speed_read_only_survey')
        try:
            for mid, kind in rows:
                state = interface.read_motor_status(mid, kind)
                result[mid] = {'motor_type': kind, 'status': state.error_code,
                    'position': state.position, 'velocity': state.velocity,
                    'registers': {r: read_register(interface, mid, r) for r in SURVEY_REGISTERS}}
                print(mid, result[mid], flush=True)
        finally:
            interface.close()
            Path(output).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    tyro.cli(main)
