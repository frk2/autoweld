"""Powered six-axis SDK transitions tested with fake motors only.

The fake transport rejects invalid per-axis command formats and records every
register/control exchange. No CAN interface, robot, or worker thread is opened.
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import MotorCmd
from i2rt.robots import arm_mode_switch as switching
from i2rt.robots.motor_chain_robot import JointCommands


class FakeClock:
    def __init__(self):
        self.now = 1000.
        self.on_sleep = None
    def time(self):
        return self.now
    def monotonic(self):
        return self.now
    def sleep(self, duration):
        self.now += duration
        if self.on_sleep:
            self.on_sleep()


class ThreadProbe:
    def is_alive(self):
        return True


class FakeInterface:
    def __init__(self, chain, mode):
        self.chain = chain
        self.motor_modes = {mid: mode for mid, _ in chain.motor_list}
        self.registers = {mid: 1 if mode == 'MIT' else 2 for mid, _ in chain.motor_list}
        self.flags = {mid: '0x1' for mid, _ in chain.motor_list}
        self.status_velocities = {mid: 0. for mid, _ in chain.motor_list}
        self.status_age = {}
        self.fail_write_motor = None
        self.fail_after_apply = False
        self.failed = False
        self.corrupt_readback_motor = None
        self.written = set()
        self.mutation_count = 0
        self.seed_error_motor = None
    def _forbidden(self, *args, **kwargs):
        raise AssertionError('Mode switching must never power off/on, zero, clear faults, stop threads or save Flash')
    motor_off = motor_on = clean_error = clear_errors = set_zero_position = _forbidden
    save_to_memory = save_to_flash = close = stop = _forbidden
    def read_motor_status(self, mid, kind):
        self.chain.events.append(('status', mid))
        return SimpleNamespace(position=self.chain.raw_position[mid - 1],
                               velocity=self.status_velocities[mid], error_code=self.flags[mid],
                               timestamp=switching.time.time() - self.status_age.get(mid, 0.))
    def set_control(self, motor_id, motor_type, pos, vel, kp, kd, torque, **kwargs):
        mid = motor_id
        firmware = self.registers[mid]
        assert self.motor_modes[mid] == ('MIT' if firmware == 1 else 'POS_VEL')
        assert np.isfinite([pos, vel, kp, kd, torque]).all()
        assert pos == pytest.approx(self.chain.raw_position[mid - 1])
        if firmware == 2:
            assert vel > 0 and kp == kd == torque == 0
        else:
            assert kp >= 0 and kd >= 0
        self.chain.events.append(('seed', mid, firmware, float(pos), float(vel), float(kp), float(kd), float(torque)))
        flag = '0x0' if self.seed_error_motor == mid else self.flags[mid]
        return SimpleNamespace(error_code=flag, timestamp=switching.time.time(),
                               position=pos, velocity=0.)
    def read_register(self, mid, name):
        assert name == 'CTRL_MODE'
        self.chain.events.append(('read', mid, name))
        if mid == self.corrupt_readback_motor and mid in self.written:
            self.corrupt_readback_motor = None
            return 99
        return self.registers[mid]
    def write_register(self, mid, name, value):
        assert name == 'CTRL_MODE'
        assert value in (1, 2)
        self.chain.events.append(('write', mid, name, value))
        self.mutation_count += 1
        failure = mid == self.fail_write_motor and not self.failed
        if not failure or self.fail_after_apply:
            self.registers[mid] = value
            self.written.add(mid)
        if failure:
            self.failed = True
            raise TimeoutError('Injected lost register reply')


class FakeChain:
    def __init__(self, mode='POS_VEL'):
        self.motor_list = [(mid, 'DM4340' if mid < 4 else 'DM4310') for mid in range(1, 7)]
        self.running = True
        self.enable_auto_recovery = False
        self._exchange_lock = threading.RLock()
        self.command_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.events = []
        self.motor_direction = np.array([1., -1., 1., -1., 1., -1.])
        self.motor_offset = np.linspace(-.2, .2, 6)
        self.q = np.array([.08, -.13, .22, .05, -.04, -.17])
        self.raw_position = self.q * self.motor_direction + self.motor_offset
        self.position_speed_motor_ids = set(range(1, 7)) if mode == 'POS_VEL' else set()
        self.motor_interface = FakeInterface(self, mode)
        self.commands = [MotorCmd(pos=self.q[i], vel=.15 if mode == 'POS_VEL' else 0.,
                                  kp=0. if mode == 'POS_VEL' else 10.,
                                  kd=0. if mode == 'POS_VEL' else 1., torque=0.) for i in range(6)]
        self._command_updated = 0.
        self.control_thread = ThreadProbe()
        self.state = []
    def __len__(self):
        return len(self.motor_list)
    def _joint_position_real_to_sim_idx(self, position, idx):
        return (position - self.motor_offset[idx]) * self.motor_direction[idx]
    def _update_absolute_positions(self, states):
        self.state = states
    def read_states(self):
        return list(self.state)
    def _set_commands(self, commands):
        self.events.append(('refresh',))
        results = []
        for idx, (mid, kind) in enumerate(self.motor_list):
            command = commands[idx]
            pos = command.pos * self.motor_direction[idx] + self.motor_offset[idx]
            vel = command.vel if mid in self.position_speed_motor_ids else command.vel * self.motor_direction[idx]
            results.append(self.motor_interface.set_control(mid, kind, pos, vel, command.kp, command.kd,
                                                            command.torque * self.motor_direction[idx]))
        return results


class FakeRobot:
    def __init__(self, mode='POS_VEL'):
        self.motor_chain = FakeChain(mode)
        self._command_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._joint_state = SimpleNamespace(pos=self.motor_chain.q.copy(), vel=np.zeros(6),
                                            timestamp=switching.time.time())
        self._kp, self._kd = np.array([80., 80., 80., 10., 10., 10.]), np.array([5., 5., 5., 1., 1., 1.])
        self.remapper = SimpleNamespace(to_robot_joint_pos_space=lambda q: q.copy())
        self._commands = JointCommands.init_all_zero(6)
        self._commands.pos = self.motor_chain.q.copy()
        self._joint_motion = self._motion_reference = object()
        self._integral_torque = np.ones(6)
        self.position_speed_limit = .15
        self.gravity_comp_factor = np.linspace(.9, 1.1, 6)
        self.gravity = np.arange(1., 7.) * .1
        self.kdl = SimpleNamespace(compute_inverse_dynamics=lambda *args: self.gravity.copy(),
                                   compute_gravity=lambda *args: self.gravity.copy())
        self.use_gravity_comp = mode == 'MIT'
        self.mit_arm_joint_indices = tuple(range(6)) if mode == 'MIT' else ()
        self.motion_fault = None
        self.paused = False
        self.pause_failure = False
        self.pause_callback = None
        self.control_thread = ThreadProbe()
    def get_motion_fault(self):
        return self.motion_fault
    def _motor_state_to_joint_state(self, states):
        return SimpleNamespace(pos=self.motor_chain.q.copy(), vel=np.zeros(6),
                               timestamp=min(s.timestamp for s in states))
    def pause_control_server(self, *, deadline):
        self.motor_chain.events.append(('pause',))
        self.paused = True
        if self.pause_callback:
            self.pause_callback()
        if self.pause_failure:
            raise TimeoutError('Injected host pause deadline')
    def resume_control_server(self):
        self.motor_chain.events.append(('resume',))
        self.paused = False


@pytest.fixture
def robot(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(switching, 'time', clock)
    robot = FakeRobot()
    monkeypatch.setattr(switching, 'read_register', lambda interface, mid, name: interface.read_register(mid, name))
    monkeypatch.setattr(switching, 'write_register', lambda interface, mid, name, value:
                        interface.write_register(mid, name, value))
    robot.clock = clock
    return robot


def assert_power_and_threads_preserved(robot):
    assert robot.motor_chain.running
    assert robot.motor_chain.control_thread.is_alive()
    assert robot.control_thread.is_alive()
    assert all(flag == '0x1' for flag in robot.motor_chain.motor_interface.flags.values())
    assert not robot.paused


@pytest.mark.parametrize('mode', ['MIT', 'POS_VEL'])
def test_six_axis_powered_ram_transition_seeds_each_axis_before_readback(robot, mode):
    if mode == 'POS_VEL':
        switching.switch_arm_control_mode(robot, 'MIT')
        robot.motor_chain.events.clear()
        robot.motor_chain.motor_interface.written.clear()
    switching.switch_arm_control_mode(robot, mode)
    chain, interface = robot.motor_chain, robot.motor_chain.motor_interface
    events = chain.events
    target = 1 if mode == 'MIT' else 2
    writes = [e for e in events if e[0] == 'write']
    assert writes == [('write', mid, 'CTRL_MODE', target) for mid in range(1, 7)]
    first_write = next(i for i, event in enumerate(events) if event[0] == 'write')
    surveyed = [event[1] for event in events[:first_write] if event[0] == 'read']
    assert surveyed == list(range(1, 7))
    for i, event in enumerate(events):
        if event[0] != 'write':
            continue
        mid = event[1]
        assert events[i + 1][:3] == ('seed', mid, target)
        assert events[i + 2] == ('read', mid, 'CTRL_MODE')
        if mode == 'MIT':
            assert events[i + 1][-1] == pytest.approx(
                robot.gravity[mid - 1] * robot.gravity_comp_factor[mid - 1] * chain.motor_direction[mid - 1])
    assert all(value == target for value in interface.registers.values())
    assert robot.use_gravity_comp == (mode == 'MIT')
    assert robot.mit_arm_joint_indices == (tuple(range(6)) if mode == 'MIT' else ())
    assert robot._joint_motion is robot._motion_reference is None
    np.testing.assert_array_equal(robot._integral_torque, 0.)
    np.testing.assert_allclose(robot._commands.pos, chain.q)
    chain._set_commands(chain.commands)  # Published commands match every retained firmware mode.
    assert events[0] == ('pause',)
    assert ('resume',) in events
    assert_power_and_threads_preserved(robot)


@pytest.mark.parametrize('ids', [list(range(1, 6)), list(range(1, 8)), [1, 2, 3, 4, 5, 5], [2, 1, 3, 4, 5, 6]])
def test_requires_exact_six_ordered_arm_motor_ids_before_any_exchange(robot, ids):
    robot.motor_chain.motor_list = [(mid, 'DM4310') for mid in ids]
    with pytest.raises(RuntimeError, match='exactly motors'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not robot.motor_chain.events


@pytest.mark.parametrize('fault', ['fault', 'chain_stopped', 'auto_recovery'])
def test_fault_stopped_chain_and_auto_recovery_reject_without_mutation(robot, fault):
    if fault == 'fault':
        robot.motion_fault = {'reason': 'injected'}
    elif fault == 'chain_stopped':
        robot.motor_chain.running = False
    else:
        robot.motor_chain.enable_auto_recovery = True
    with pytest.raises(RuntimeError, match='healthy'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not robot.motor_chain.events


def test_stale_host_feedback_never_changes_firmware(robot):
    robot._joint_state.timestamp -= 1.
    with pytest.raises(RuntimeError, match='Fresh SDK feedback'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not robot.motor_chain.events


def test_moving_measured_hold_must_settle_before_firmware_change(robot):
    robot._joint_state.vel[:] = .3
    robot.clock.on_sleep = lambda: setattr(robot._joint_state, 'timestamp', robot.clock.time())
    with pytest.raises(RuntimeError, match='stationary'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not robot.motor_chain.events


@pytest.mark.parametrize('state_problem', ['disabled', 'fault', 'moving', 'stale'])
def test_each_raw_motor_must_be_fresh_enabled_and_stationary(robot, state_problem):
    interface = robot.motor_chain.motor_interface
    if state_problem in ('disabled', 'fault'):
        interface.flags[4] = '0x0' if state_problem == 'disabled' else '0x9'
    elif state_problem == 'moving':
        interface.status_velocities[4] = .2
    else:
        interface.status_age[4] = 1.
    with pytest.raises(RuntimeError, match='freshly enabled and stationary'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not any(e[0] == 'write' for e in robot.motor_chain.events)
    assert not robot.paused


@pytest.mark.parametrize('after_apply', [False, True])
def test_failed_ram_write_rolls_back_even_a_lost_reply_without_poweroff(robot, after_apply):
    chain, interface = robot.motor_chain, robot.motor_chain.motor_interface
    interface.fail_write_motor = 3
    interface.fail_after_apply = after_apply
    with pytest.raises(RuntimeError, match='retaining measured hold'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert set(interface.registers.values()) == {2}
    assert set(interface.motor_modes.values()) == {'POS_VEL'}
    assert chain.position_speed_motor_ids == set(range(1, 7))
    assert not robot.use_gravity_comp
    assert not robot.mit_arm_joint_indices
    chain._set_commands(chain.commands)
    np.testing.assert_allclose(robot._commands.pos, chain.q)
    assert_power_and_threads_preserved(robot)
    writes = [(e[1], e[3]) for e in chain.events if e[0] == 'write']
    assert writes == [(1, 1), (2, 1), (3, 1), (3, 2), (2, 2), (1, 2)]


def test_bad_mode_readback_rolls_back_and_resumes_threads(robot):
    interface = robot.motor_chain.motor_interface
    interface.corrupt_readback_motor = 3
    with pytest.raises(RuntimeError, match='readback'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert set(interface.registers.values()) == {2}
    robot.motor_chain._set_commands(robot.motor_chain.commands)
    assert_power_and_threads_preserved(robot)


def test_firmware_disagrees_with_sdk_before_any_mode_write(robot):
    robot.motor_chain.motor_interface.registers[6] = 1
    with pytest.raises(RuntimeError, match='firmware mode differs'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not any(e[0] == 'write' for e in robot.motor_chain.events)
    assert_power_and_threads_preserved(robot)


def test_pause_timeout_always_clears_host_pause_latch(robot):
    robot.pause_failure = True
    with pytest.raises((RuntimeError, TimeoutError), match='pause'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not any(e[0] == 'write' for e in robot.motor_chain.events)
    assert_power_and_threads_preserved(robot)


def test_new_host_fault_during_pause_aborts_before_ram_mutation(robot):
    robot.pause_callback = lambda: setattr(robot, 'motion_fault', {'reason': 'fault while settling'})
    with pytest.raises(RuntimeError, match='fault|healthy'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert not any(e[0] == 'write' for e in robot.motor_chain.events)
    assert_power_and_threads_preserved(robot)


def test_rollback_seeds_immediately_and_verifies_each_restored_register(robot):
    interface = robot.motor_chain.motor_interface
    interface.fail_write_motor = 3
    with pytest.raises(RuntimeError):
        switching.switch_arm_control_mode(robot, 'MIT')
    events = robot.motor_chain.events
    for i, event in enumerate(events):
        if event[0] == 'write' and event[3] == 2:
            assert events[i + 1][:3] == ('seed', event[1], 2)
            assert events[i + 2] == ('read', event[1], 'CTRL_MODE')


def test_already_selected_mode_is_a_noop_without_ram_or_power_commands(robot):
    previous = robot._commands
    switching.switch_arm_control_mode(robot, 'POS_VEL')
    assert not robot.motor_chain.events
    assert robot._commands is previous
    assert_power_and_threads_preserved(robot)


def test_mixed_initial_modes_preserve_correct_frames_for_unchanged_axes(robot):
    chain, interface = robot.motor_chain, robot.motor_chain.motor_interface
    for mid in (4, 5):
        chain.position_speed_motor_ids.discard(mid)
        interface.registers[mid] = 1
        interface.motor_modes[mid] = 'MIT'
        chain.commands[mid - 1] = MotorCmd(pos=chain.q[mid - 1], vel=0., kp=10., kd=1., torque=0.)
    robot.mit_arm_joint_indices = (3, 4)
    robot.use_gravity_comp = True
    switching.switch_arm_control_mode(robot, 'MIT')
    writes = [event[1] for event in chain.events if event[0] == 'write']
    assert writes == [1, 2, 3, 6]
    chain._set_commands(chain.commands)
    assert set(interface.registers.values()) == {1}
    assert_power_and_threads_preserved(robot)


def test_invalid_requested_mode_rejects_before_any_exchange(robot):
    with pytest.raises(ValueError, match='MIT or POS_VEL'):
        switching.switch_arm_control_mode(robot, 'OFF')
    assert not robot.motor_chain.events
    assert_power_and_threads_preserved(robot)


def test_lost_rollback_reply_reconciles_actual_mode_before_resuming(robot, monkeypatch):
    chain, interface = robot.motor_chain, robot.motor_chain.motor_interface
    original_write = interface.write_register
    interface.fail_write_motor = 3
    lost_restore_reply = []
    def write(mid, name, value):
        original_write(mid, name, value)
        if mid == 2 and value == 2 and not lost_restore_reply:
            # The firmware restored POS_VEL, but that reply was lost. The SDK
            # must discover its actual mode before publishing continuing frames.
            lost_restore_reply.append(True)
            raise TimeoutError('Injected lost rollback reply after firmware apply')
    monkeypatch.setattr(interface, 'write_register', write)
    with pytest.raises(RuntimeError, match='retaining measured hold'):
        switching.switch_arm_control_mode(robot, 'MIT')
    assert lost_restore_reply
    chain._set_commands(chain.commands)
    assert_power_and_threads_preserved(robot)
