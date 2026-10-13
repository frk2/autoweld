"""Velocity diagnostics and command encoding: no hardware access."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, DMSingleMotorCanInterface
from i2rt.motor_drivers.mit_command import mit_codes
from i2rt.motor_drivers.utils import MotorType, ReceiveMode, float_to_uint, uint_to_float
from test_integral import sdk_stub

from autoweld.velocity import PositionVelocity


@pytest.mark.parametrize("motor", [MotorType.DM4340, MotorType.DM4310])
def test_compensated_native_pd_matches_continuous_reference(motor):
    c = MotorType.get_motor_constants(motor)
    rng = np.random.default_rng(82)
    for _ in range(1000):
        p, v, kp, kd, ff = rng.uniform([-0.5, -0.1, 10, 1, -2], [0.5, 0.1, 180, 5, 2])
        codes = mit_codes(p, v, kp, kd, ff, c, compensate=True)
        p_wire, v_wire, kp_wire, kd_wire, ff_wire = [
            uint_to_float(x, lo, hi, bits)
            for x, lo, hi, bits in zip(
                codes,
                [c.POSITION_MIN, c.VELOCITY_MIN, 0, 0, c.TORQUE_MIN],
                [c.POSITION_MAX, c.VELOCITY_MAX, 500, 5, c.TORQUE_MAX],
                [16, 12, 12, 12, 12],
            )
        ]
        # Arbitrary actual motor state cancels: the residual never needs stale host q/dq.
        q, dq = 0.12, -0.031
        requested = kp_wire * (p - q) + kd_wire * (v - dq) + ff
        delivered = kp_wire * (p_wire - q) + kd_wire * (v_wire - dq) + ff_wire
        assert abs(requested - delivered) <= (c.TORQUE_MAX - c.TORQUE_MIN) / 4095 / 2 + 1e-12


def test_legacy_encoding_unchanged_and_compensation_saturates():
    c = MotorType.get_motor_constants(MotorType.DM4310)
    args = [0.0032, -0.001, 29.85984, 2.592, 0.5]
    expected = tuple(
        float_to_uint(x, lo, hi, b)
        for x, lo, hi, b in zip(args, [-12.5, -30, 0, 0, -10], [12.5, 30, 500, 5, 10], [16, 12, 12, 12, 12])
    )
    assert mit_codes(*args, c) == expected
    assert mit_codes(100, 100, 1000, 9, 100, c, compensate=True) == (65535, 4095, 4095, 4095, 4095)
    with pytest.raises(ValueError):
        mit_codes(float("nan"), *args[1:], c, compensate=True)


def test_receive_timestamp_survives_cached_reads():
    import can

    interface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    interface.receive_mode = ReceiveMode.p16
    frame = interface.parse_recv_message(
        can.Message(arbitration_id=17, timestamp=42.0, data=[0x11, 0x80, 0, 0x80, 0, 0, 20, 20]),
        MotorType.DM4310,
    )
    assert frame.timestamp == 42.0
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.state_lock = threading.Lock()
    chain.motor_list = [(1, MotorType.DM4310)]
    chain.state = [frame]
    chain.absolute_positions = [0.0]
    chain.motor_offset = [0.0]
    chain.motor_direction = [1.0]
    for _ in range(4):
        assert chain.read_states()[0].timestamp == 42.0


def test_velocity_fit_quantization_irregular_sampling_and_skew():
    estimator = PositionVelocity()
    velocity = np.array([0.02, -0.012, 0.006, -0.003, 0.024, 0])
    t = 100.0
    for dt in np.random.default_rng(7).uniform(0.008, 0.022, 100):
        t += dt
        stamps = t - np.arange(6) * 0.0005
        q = np.round((stamps - 100) * velocity / (25 / 65535)) * (25 / 65535)
        result = estimator.update(stamps, q, t + 0.002)
    assert result is not None
    error = abs(np.array(result["joint_velocity_rad_s"]) - velocity)
    assert np.all(error <= result["joint_quantization_bound_rad_s"])
    assert 0.035 < result["delay_s"] < 0.09
    assert estimator.update(stamps, q, t + 0.003) is None
    assert estimator.update(stamps, q, t + 0.2) is None
    assert len(estimator.rows) == 0
    assert estimator.update(np.full(6, t + 1), q, t + 1) is None


def test_encoder_stationary_has_zero_slope_despite_raw_velocity_floor():
    estimator = PositionVelocity()
    for t in np.arange(100, 100.3, 0.01):
        result = estimator.update(np.full(6, t), np.full(6, 0.1), t)
    np.testing.assert_allclose(result["joint_velocity_rad_s"], 0)


def test_acceleration_estimate_has_common_pose_and_time_with_skew_and_jitter():
    estimator = PositionVelocity()
    rng = np.random.default_rng(2026)
    initial = np.linspace(-.2, .2, 6)
    velocity = np.linspace(-.3, .3, 6)
    acceleration = np.linspace(.5, -.5, 6)
    t = 0.
    for dt in rng.uniform(.012, .023, 60):
        t += dt
        times = t - np.arange(6)*.0007
        positions = initial + velocity*times + .5*acceleration*times**2
        result = estimator.update(times+100, positions, t+100+.004)
        if result:
            center = result['timestamp']-100
            np.testing.assert_allclose(result['joint_velocity_rad_s'], velocity+acceleration*center, atol=1e-11)
            np.testing.assert_allclose(result['joint_position_rad'],
                                       initial+velocity*center+.5*acceleration*center**2, atol=1e-12)
            assert result['delay_s'] > .03


def test_quadratic_velocity_quantization_bound_and_skew_rejection():
    estimator = PositionVelocity()
    step = 25/65535
    for t in np.arange(0., .4, .013):
        times = t-np.arange(6)*.0003
        q = np.floor((.1+.1*times+.15*times**2)/step)*step
        result = estimator.update(times+100, q, t+100+.002)
        if result:
            true = .1+.3*(result['timestamp']-100)
            assert np.all(abs(np.array(result['joint_velocity_rad_s'])-true)
                          <= np.array(result['joint_quantization_bound_rad_s'])+1e-10)
    assert estimator.update(101-np.arange(6)*.004, np.ones(6), 101) is None
    assert not estimator.rows


def test_smoothing_opt_in_does_not_change_original_pd_or_fault_hold():
    r = sdk_stub()
    r.configure_command_quantization(True)
    r.command_joint_pos_pid(np.ones(7) * 0.01, np.ones(7), np.ones(7))
    assert r._commands.compensate_quantization
    r.command_joint_pos(np.zeros(7))
    assert not r._commands.compensate_quantization
    r.command_joint_pos_pid(np.ones(7) * 0.01, np.ones(7), np.ones(7))
    r._oscillation_guard.observe = lambda *args: {"reason": "test", "timestamp": 1.0}
    r.update()
    assert not r.sent[1].compensate_quantization


def test_oldest_motor_timestamp_controls_freshness():
    r = sdk_stub()
    r.remapper.to_command_joint_pos_space = lambda q: q
    r.remapper.to_command_joint_vel_space = lambda q: q
    states = [
        SimpleNamespace(pos=0, vel=0, eff=0, temp_mos=0, temp_rotor=0, timestamp=t)
        for t in [100, 100, 99, 100, 100, 100, 100]
    ]
    result = r._motor_state_to_joint_state(states)
    assert result.timestamp == 99
    assert result.motor_timestamps[2] == 99


def test_sdk_scheduler_skips_missed_deadlines_without_catchup(monkeypatch):
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    now, starts = [100.], []
    monkeypatch.setattr("i2rt.robots.motor_chain_robot.time.monotonic", lambda: now[0])
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot.motor_chain = SimpleNamespace(running=True)
    robot._stop_event = SimpleNamespace(is_set=lambda: len(starts) == 3,
                                       wait=lambda seconds: now.__setitem__(0, now[0]+seconds))

    def update():
        starts.append(now[0]-100.)
        now[0] += .006 if len(starts) == 1 else .001

    robot.update = update
    robot.start_server()
    np.testing.assert_allclose(starts, [0., .008, .012], atol=1e-10)
    assert robot._host_timing["host_missed_deadlines"] == 1
