"""No robot, camera, SDK owner or installed YAM model is constructed."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.motion_registration import (
    HostClockAnchor,
    JointOutputHistory,
    MovingCloudRegistrar,
    RegistrationRejected,
    SharedModelFK,
)

BOUNDS = np.tile([-2., 2.], (6, 1))
BOUNDS[5] = [-5.37, -2.09]
Q = np.array([0., 0., 0., 0., 0., -3.])


def history(**kwargs):
    return JointOutputHistory(BOUNDS, encoder_sample_age_bound_s=kwargs.pop('encoder_sample_age_bound_s', 0.),
                              encoder_quantization_rad=0., **kwargs)


def sample(h, t, q=None, velocity=None, skew=None, errors=None, **kwargs):
    skew = np.zeros(6) if skew is None else np.asarray(skew)
    receive_mono = t+skew
    bounds = np.column_stack((receive_mono-.0004, receive_mono+.0001))
    return h.append(Q if q is None else q, np.zeros(6) if velocity is None else velocity,
                    receive_mono+1000., errors=['0x1']*6 if errors is None else errors,
                    request_bounds_monotonic_s=bounds,
                    anchor=HostClockAnchor(1000.+t+.001, t+.001, 0.), **kwargs)


def fill(h, q_func=None, v=None):
    for t in np.arange(10., 10.1001, .005):
        q = Q.copy() if q_func is None else q_func(t)
        assert sample(h, t, q=q, velocity=v)
    return h


def test_interpolates_actual_outputs_per_motor_stagger_never_targets():
    h = history()
    skew = np.arange(6)*.0001
    for t in [10., 10.01]:
        q = Q + .1*((t+skew-.00015)-10.)
        assert sample(h, t, q=q, velocity=np.full(6, .1), skew=skew)
    got = h.interpolate(10.005)
    np.testing.assert_allclose(got.joint_pos, Q+.0005, atol=1e-13)
    np.testing.assert_allclose(got.joint_vel, .1)
    assert np.ptp(got.bracket_monotonic_s[:, 0]) > .0004
    assert got.max_gap_s == pytest.approx(.01)


def test_unknown_encoder_sample_age_cannot_qualify_receive_time():
    h = fill(history(encoder_sample_age_bound_s=None))
    with pytest.raises(RegistrationRejected, match='sample-age bound is unknown'):
        h.interpolate(10.03)


@pytest.mark.parametrize('time', [9., 10.2])
def test_never_extrapolates(time):
    h = fill(history())
    with pytest.raises(RegistrationRejected, match='no measured motor bracket'):
        h.interpolate(time)


def test_gap_gate_and_timing_gate_independent():
    h = history()
    sample(h, 10.)
    sample(h, 10.04)
    with pytest.raises(RegistrationRejected, match='maximum gap'):
        h.interpolate(10.02)
    h = fill(history())
    with pytest.raises(RegistrationRejected, match='timing uncertainty'):
        h.interpolate(10.03, timing_uncertainty_s=.002)


@pytest.mark.parametrize('bad', ['fault', 'disabled', 'nan', 'branch', 'reordered', 'cachedchange', 'badbracket', 'wallstep'])
def test_invalid_data_invalidates_epoch_and_no_interpolation_across_it(bad):
    h = fill(history())
    generation = h.state()['generation']
    q, errors, stamp = Q.copy(), ['0x1']*6, 10.11
    kw = {}
    if bad == 'fault':
        kw['motion_fault'] = {'kind': 'watchdog'}
    if bad == 'disabled':
        errors[3] = '0x0'
    if bad == 'nan':
        q[2] = np.nan
    if bad == 'branch':
        q[5] += 2*np.pi
    if bad == 'reordered':
        stamp = 10.02
    if bad == 'cachedchange':
        stamp, q[2] = 10.1, .01
    if bad in ('badbracket', 'wallstep'):
        wall = 1010.11 if bad == 'badbracket' else 2010.11
        anchor = HostClockAnchor(1010.111 if bad == 'badbracket' else 2010.111, 10.111, 0.)
        assert not h.append(q, np.zeros(6), np.full(6, wall), errors=errors,
            request_bounds_monotonic_s=np.tile([10.2, 10.3] if bad == 'badbracket' else [10.1096, 10.1101], (6, 1)), anchor=anchor)
    else:
        assert not sample(h, stamp, q=q, errors=errors, **kw)
    assert h.state()['generation'] == generation+1
    with pytest.raises(RegistrationRejected):
        h.interpolate(10.03)


def test_bounded_storage_copies_inputs_and_duplicates_do_not_churn():
    h = history(max_samples=4)
    q = Q.copy()
    for t in np.arange(10., 10.1, .005):
        assert sample(h, t, q=q)
    latest = h.state()['samples_per_joint']
    assert latest == [4]*6
    assert sample(h, t, q=q)
    assert h.state()['samples_per_joint'] == latest
    q[0] = 1.
    assert h.interpolate(t-.002).joint_pos[0] == 0.


def test_retention_and_thread_safety():
    h = history(max_samples=100, retention_s=.04)
    failures = []
    def producer():
        for t in np.arange(10., 10.5, .001):
            if not sample(h, t):
                failures.append(h.state())
    def reader():
        for _ in range(500):
            h.state()
            try:
                h.interpolate(10.2)
            except RegistrationRejected:
                pass
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda f: f(), [producer, reader]))
    assert not failures
    assert max(h.state()['samples_per_joint']) <= 42


class LinearFK:
    def pose_and_jacobian(self, q):
        transform = np.eye(4)
        transform[0, 3] = q[0]
        jac = np.zeros((6, 6))
        jac[0, 0] = 1.
        return transform, jac


def handeye():
    return dict(tcp_from_camera=np.eye(4), units='m', coordinate_frame='tcp',
                source_id='camera-a', optical_intrinsics_id='optical-a')


def cloud(**timing):
    return dict(xyz_m=[[0., 0., .14], [.01, 0., .14]], units='m', coordinate_frame='camera',
                source_id='camera-a', optical_intrinsics_id='optical-a', sequence=31,
                timing=dict(host_clock_domain='host_monotonic', motion_pairing_supported=True,
                    exposure_host_start_s=10.0495, exposure_host_midpoint_s=10.05,
                    exposure_host_end_s=10.0505, exposure_time_uncertainty_s=.0001,
                    exposure_duration_us=1000., sensor_shutter_mode='global',
                    clock_mapping={'status': 'qualified', 'mapping_id': 'map-1'}, **timing))


def test_registration_uses_measured_exposure_pose_and_keeps_input_immutable():
    h = fill(history(), q_func=lambda t: Q+np.array([.01*(t-10), 0, 0, 0, 0, 0]), v=np.array([.01, 0, 0, 0, 0, 0]))
    c = cloud()
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    result = registrar.register(c)
    # Request brackets locate physical sample center 0.15ms before receive.
    np.testing.assert_allclose(result['points_world_m'][:, 0], [.0005015, .0105015], atol=1e-12)
    assert c['xyz_m'][0][0] == 0.
    assert result['registration_method'] == 'measured_output_interpolation'
    assert result['spatial_error_bound_m'] < .0005
    assert result['sequence'] == 31
    assert result['clock_mapping_id'] == 'map-1'


@pytest.mark.parametrize('kind', ['unsupported', 'unqualified', 'wallclock', 'noexposure', 'receiptfallback', 'identity', 'units', 'nonfinite', 'motionblur'])
def test_registration_never_promotes_unqualified_cloud(kind):
    h = fill(history(), q_func=lambda t: Q+np.array([.5*(t-10), 0, 0, 0, 0, 0]), v=np.array([.5, 0, 0, 0, 0, 0]))
    c = cloud()
    if kind == 'unsupported':
        c['timing']['motion_pairing_supported'] = False
    if kind == 'unqualified':
        c['timing']['clock_mapping']['status'] = 'unqualified'
    if kind == 'wallclock':
        c['timing']['host_clock_domain'] = 'host_wall'
    if kind == 'noexposure':
        c['timing']['exposure_host_midpoint_s'] = None
    if kind == 'receiptfallback':
        c['timing'] = {'read_started': 10.05}
    if kind == 'identity':
        c['source_id'] = 'other-camera'
    if kind == 'units':
        c['units'] = 'mm'
    if kind == 'nonfinite':
        c['xyz_m'][0][0] = np.nan
    if kind == 'motionblur':
        c['timing'].update(exposure_host_start_s=10.045, exposure_host_end_s=10.055, exposure_duration_us=10000.)
    with pytest.raises(RegistrationRejected):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register(c)


def test_clock_anchor_capture_and_mapping():
    anchor = HostClockAnchor.capture()
    assert anchor.uncertainty_s >= 0
    assert anchor.to_monotonic(anchor.wall_s) == pytest.approx(anchor.monotonic_s)


def test_point_error_bound_uses_actual_rotational_point_velocity():
    jacobian = np.zeros((6, 6))
    jacobian[0, 0] = .2
    jacobian[4, 0] = 1.
    points = np.array([[0., 0., -.14], [0., 0., .14]])
    errors = MovingCloudRegistrar._point_error_bound(points, np.eye(4), jacobian,
        np.array([.001, 0, 0, 0, 0, 0]))
    # +Y angular velocity moves the first point toward -X and the second +X.
    np.testing.assert_allclose(errors, [.00006, .00034])


def test_generic_shared_model_fk_uses_same_model_private_data():
    import mujoco
    bodies = ''.join(f'<body pos="0 0 .02"><joint name="j{i}" type="hinge" axis="0 1 0"/><geom size=".005"/>' for i in range(6))
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody>'+bodies+'<site name="weld_tcp" pos="0 0 .02"/>'+'</body>'*6+'</worldbody></mujoco>')
    a, b = SharedModelFK(model), SharedModelFK(model)
    assert a.model is model and b.model is model and a.data is not b.data
    pose, jac = a.pose_and_jacobian(np.zeros(6))
    saved = pose.copy()
    b.pose_and_jacobian(np.ones(6)*.1)
    np.testing.assert_array_equal(pose, saved)
    assert jac.shape == (6, 6)
    reference = mujoco.MjData(model)
    reference.qpos[:] = .1
    mujoco.mj_forward(model, reference)
    got, _ = a.pose_and_jacobian(np.ones(6)*.1)
    np.testing.assert_array_equal(got[:3, 3], reference.site_xpos[a.site])
    np.testing.assert_array_equal(got[:3, :3], reference.site_xmat[a.site].reshape(3, 3))


def test_sdk_state_listener_subscribes_once_and_failure_does_not_change_writer():
    from i2rt.robots.motor_chain_robot import MotorChainRobot
    robot = MotorChainRobot.__new__(MotorChainRobot)
    import threading
    robot._state_lock = threading.Lock()
    robot._joint_state_listeners = ()
    def callback(*_args, **_kwargs):
        pass
    detach = robot.add_joint_state_listener(callback)
    robot.add_joint_state_listener(callback)
    assert robot._joint_state_listeners == (callback,)
    detach()
    detach()
    assert robot._joint_state_listeners == ()


def test_sdk_existing_exchange_publishes_per_motor_monotonic_brackets():
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface
    from i2rt.motor_drivers.utils import FeedbackFrameInfo
    calls = []
    def set_control(**kwargs):
        calls.append(kwargs)
        return FeedbackFrameInfo(kwargs['motor_id'], '0x1', 'enabled', 0., 0., 0., 25., 25., 1000.)
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain.motor_list = [(i, 'fake') for i in range(1, 7)]
    chain.motor_direction = [1]*6
    chain.motor_interface = SimpleNamespace(set_control=set_control)
    chain._joint_position_sim_to_real_idx = lambda value, _i: value
    cmd = SimpleNamespace(torque=0., pos=.1, vel=.2, kp=0., kd=0., compensate_quantization=False)
    before = __import__('time').monotonic()
    states = chain._set_commands([cmd]*6)
    after = __import__('time').monotonic()
    assert len(calls) == 6  # No additional status/register requests.
    assert [c['motor_id'] for c in calls] == list(range(1, 7))
    for state in states:
        assert before <= state.request_started_monotonic <= state.response_completed_monotonic <= after
        assert state.timestamp == 1000.  # Never replace kernel receipt timestamp.


def test_sdk_publication_preserves_state_when_observer_raises():
    import threading

    from i2rt.motor_drivers.utils import MotorInfo
    from i2rt.robots.motor_chain_robot import MotorChainRobot
    robot = MotorChainRobot.__new__(MotorChainRobot)
    calls = []
    motors = [MotorInfo(id=i+1, error_code='0x1', pos=Q[i], timestamp=1000.+i*.001,
                       request_started_monotonic=10.+i*.001,
                       response_completed_monotonic=10.0005+i*.001) for i in range(6)]
    robot.motor_chain = SimpleNamespace(set_commands=lambda *_a, **_k: motors)
    robot.remapper = SimpleNamespace(to_command_joint_pos_space=lambda x: x,
                                    to_command_joint_vel_space=lambda x: x)
    robot._gripper_index = None
    robot._motion_fault = None
    robot._mcap_lock = threading.Lock()
    robot._mcap_recorder = robot._joint_state_saver = None
    robot._check_current_qpos_in_joint_limits = lambda **_kw: None
    robot._joint_limit_buffer_rad = .01
    def fail(*_args, **_kwargs):
        raise RuntimeError('broken observer')
    robot._joint_state_listeners = (fail, lambda state, **_kw: calls.append(state))
    command = SimpleNamespace(pos=Q, vel=np.zeros(6), kp=np.zeros(6), kd=np.zeros(6), compensate_quantization=False)
    robot._update_joint_state(np.zeros(6), command)
    assert len(calls) == 1
    assert robot._joint_state is calls[0]
    assert robot._joint_state_listener_error == 'RuntimeError: broken observer'
    np.testing.assert_array_equal(robot._joint_state.pos, Q)
    assert robot._joint_state.motor_errors == ('0x1',)*6
    assert robot._joint_state.motor_request_bounds_monotonic.shape == (6, 2)
    np.testing.assert_allclose(robot._joint_state.motor_timestamps, 1000.+np.arange(6)*.001)


def test_fixed_installed_branch_cannot_be_replaced_by_stock_j6_bounds():
    bounds = BOUNDS.copy()
    bounds[5] = [-np.pi, np.pi]
    with pytest.raises(ValueError, match='fixed J6'):
        JointOutputHistory(bounds)


def test_xml_round_trip_bounds_are_intersected_with_exact_installed_branch():
    from autoweld.joint_calibration import j6_operating_range
    installed = j6_operating_range()
    bounds = BOUNDS.copy()
    bounds[5] = [np.nextafter(installed[0], -np.inf), np.nextafter(installed[1], np.inf)]
    measured = JointOutputHistory(bounds)
    np.testing.assert_array_equal(measured.bounds[5], installed)


def test_cached_sample_after_small_clock_slew_is_noop():
    h = fill(history())
    count = h.state()['samples_per_joint']
    # Same immutable SDK packet, new publication anchor differs by 1us.
    assert h.append(Q, np.zeros(6), np.full(6, 1010.1), errors=['0x1']*6,
                    request_bounds_monotonic_s=np.tile([10.0996, 10.1001], (6, 1)),
                    anchor=HostClockAnchor(1010.101001, 10.101, 0.))
    assert h.state()['samples_per_joint'] == count


def qualification(h, **updates):
    result = dict(schema='autoweld.motion-delay.v1', status='qualified', qualification_id='fit-1',
        history_id=h.history_id, source_id='camera-a', optical_intrinsics_id='optical-a',
        acquisition_session_id='acq-1', clock_mapping_id='map-1', delay_s=.01,
        uncertainty_s=.0001, confidence_interval_s=[.0099, .0101], evidence_path='/tmp/evidence.json',
        method='bidirectional_static_surface_depth', heldout_passed=True)
    result.update(updates)
    return result


def test_provisional_unknown_timing_stays_calibration_only_and_no_extrapolation():
    h = fill(history(encoder_sample_age_bound_s=None))
    result = h.provisional_interpolate(10.05)
    assert result.calibration_only
    assert h.state()['motion_pairing_supported'] is False
    with pytest.raises(RegistrationRejected):
        h.interpolate(10.05)
    with pytest.raises(RegistrationRejected, match='extrapolation'):
        h.provisional_interpolate(11.)
    raw = h.raw_series(10.02, 10.04)
    assert raw['calibration_only'] is True and raw['motion_pairing_supported'] is False
    assert raw['history_id'] == h.history_id
    assert len(raw['axes']) == 6
    assert len(raw['axes'][0]) > 1
    assert raw['axes'][0].flags.writeable is False
    # Read-only data belongs to the diagnostic, never aliases producer storage.
    raw['axes'][0].flags.writeable = True
    raw['axes'][0][0, 1] = 100.
    assert h.provisional_interpolate(10.03).joint_pos[0] == 0.


@pytest.mark.parametrize('delay', [.01, -.01])
def test_evidenced_signed_delay_applies_exposure_plus_offset(delay):
    h = fill(history(encoder_sample_age_bound_s=None),
        q_func=lambda t: Q+np.array([.1*(t-10), 0, 0, 0, 0, 0]), v=np.array([.1, 0, 0, 0, 0, 0]))
    record = qualification(h, delay_s=delay, confidence_interval_s=[delay-.0001, delay+.0001])
    result = h.interpolate_with_delay(10.05, record, timing_uncertainty_s=.0002)
    assert not result.calibration_only
    assert result.query_offset_s == delay
    assert result.monotonic_s == pytest.approx(10.05+delay)
    assert result.timing_uncertainty_s == pytest.approx(.00055)
    assert result.joint_pos[0] == pytest.approx(.1*(.05+delay+.00015))


@pytest.mark.parametrize('bad', [dict(status='candidate'), dict(heldout_passed=False),
    dict(history_id='old-owner'), dict(evidence_path=''), dict(uncertainty_s=-1),
    dict(confidence_interval_s=[.009, .011]), dict(uncertainty_s=.01), dict(delay_s=float('nan'))])
def test_unqualified_delay_cannot_promote_registration(bad):
    h = fill(history(encoder_sample_age_bound_s=None))
    record = qualification(h, **bad)
    with pytest.raises(RegistrationRejected):
        h.interpolate_with_delay(10.05, record)
    with pytest.raises(RegistrationRejected):
        MovingCloudRegistrar(h, LinearFK(), handeye()).set_delay_qualification(record)


def test_qualified_registration_binds_camera_session_mapping_and_applies_offset():
    h = fill(history(encoder_sample_age_bound_s=None),
        q_func=lambda t: Q+np.array([.01*(t-10), 0, 0, 0, 0, 0]), v=np.array([.01, 0, 0, 0, 0, 0]))
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    c = cloud()
    c['acquisition_session_id'] = 'acq-1'
    with pytest.raises(RegistrationRejected, match='sample-age bound'):
        registrar.register(c)
    record = qualification(h)
    registrar.set_delay_qualification(record)
    # Installed evidence is copied: outside mutation cannot change applied delay.
    record['delay_s'] = 1.
    got = registrar.register(c)
    assert got['delay_qualification_id'] == 'fit-1'
    assert got['applied_delay_s'] == .01
    assert got['points_world_m'][0, 0] == pytest.approx(.0006015)
    c['acquisition_session_id'] = 'acq-2'
    with pytest.raises(RegistrationRejected, match='session/identity'):
        registrar.register(c)
    c['acquisition_session_id'] = 'acq-1'
    c['timing']['clock_mapping']['mapping_id'] = 'map-2'
    with pytest.raises(RegistrationRejected, match='session/identity'):
        registrar.register(c)


def test_delay_plus_camera_uncertainty_cannot_exceed_combined_limit():
    h = fill(history(encoder_sample_age_bound_s=None))
    with pytest.raises(RegistrationRejected, match='Combined'):
        h.interpolate_with_delay(10.05, qualification(h), timing_uncertainty_s=.0019)


def test_cannot_apply_empirical_delay_and_firmware_age_correction_twice():
    h = fill(history())
    with pytest.raises(RegistrationRejected, match='both'):
        h.interpolate_with_delay(10.05, qualification(h))


def test_malformed_sdk_observation_clears_prior_bracket_without_writer_failure():
    h = fill(history())
    assert not h.observe(SimpleNamespace(pos=Q, vel=np.zeros(6), motor_timestamps=None))
    assert h.state()['samples_per_joint'] == [0]*6
    assert h.state()['last_error']


def test_encoder_discontinuity_inside_bounds_cannot_be_interpolated():
    h = fill(history())
    q = Q.copy()
    q[2] = 1.
    assert not sample(h, 10.105, q=q)
    assert 'discontinuity' in h.state()['last_error']


@pytest.mark.parametrize('change', [dict(sensor_shutter_mode='rolling'), dict(sensor_shutter_mode='unknown'),
    dict(exposure_duration_us=None), dict(exposure_duration_us=2000.), dict(exposure_duration_us=-1.)])
def test_native_supported_flag_cannot_override_actual_exposure_or_shutter_gate(change):
    h = fill(history())
    c = cloud()
    c['timing'].update(change)
    with pytest.raises(RegistrationRejected, match='shutter'):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register(c)


def test_native_mid_alias_and_disagreement_rejection():
    h = fill(history())
    c = cloud()
    c['timing']['exposure_host_mid_s'] = c['timing'].pop('exposure_host_midpoint_s')
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    assert registrar.register(c)['exposure_monotonic_s'] == 10.05
    c['timing']['exposure_host_midpoint_s'] = 10.06
    with pytest.raises(RegistrationRejected, match='aliases disagree'):
        registrar.register(c)


def receive_sample(h, stamp, *, lag=.0001, offset=1000., anchor_offset=None,
                   anchor_error=0., q=None, velocity=None):
    """Synthetic kernel RX event + separately delayed Python return, no devices."""
    anchor_offset = offset if anchor_offset is None else anchor_offset
    completed = stamp + lag
    return h.append(Q if q is None else q, np.zeros(6) if velocity is None else velocity,
                    np.full(6, stamp + offset), errors=['0x1'] * 6,
                    request_bounds_monotonic_s=np.tile([stamp - .0004, completed], (6, 1)),
                    anchor=HostClockAnchor(completed + .0001 + anchor_offset,
                                           completed + .0001, anchor_error))


def kernel_history(**kwargs):
    return history(encoder_sample_age_bound_s=None, reference_clock='kernel_receive', **kwargs)


def test_optional_kernel_reference_is_independent_of_python_return_lag_and_retains_delay_sign():
    kernel = kernel_history()
    midpoint = history(encoder_sample_age_bound_s=None)
    physical_to_kernel = .004
    stamps = [10., 10.01, 10.02, 10.035, 10.045, 10.055]
    for stamp in stamps:
        lag = .008 if stamp == 10.02 else .0001
        measured = Q + .1 * (stamp - physical_to_kernel - 10.)
        for h in (kernel, midpoint):
            assert receive_sample(h, stamp, lag=lag, q=measured, velocity=np.full(6, .1))
    provisional = kernel.provisional_interpolate(10.03)
    assert provisional.calibration_only
    assert provisional.timing_uncertainty_s < 1e-6
    with pytest.raises(RegistrationRejected, match='timing uncertainty'):
        midpoint.provisional_interpolate(10.03)
    with pytest.raises(RegistrationRejected, match='sample-age bound is unknown'):
        kernel.interpolate(10.03)
    record = qualification(kernel, reference_clock='kernel_receive', delay_s=physical_to_kernel,
        confidence_interval_s=[physical_to_kernel - .0001, physical_to_kernel + .0001])
    physical = kernel.interpolate_with_delay(10.03, record)
    assert not physical.calibration_only
    assert physical.query_offset_s == physical_to_kernel
    np.testing.assert_allclose(physical.joint_pos, Q + .003, atol=1e-12)
    assert not kernel.state()['motion_pairing_supported']
    with pytest.raises(RegistrationRejected, match='extrapolation'):
        kernel.interpolate_with_delay(10.10, record)


def test_kernel_reference_qualification_must_explicitly_bind_matching_clock():
    h = fill(kernel_history())
    with pytest.raises(RegistrationRejected, match='reference clock'):
        h.interpolate_with_delay(10.05, qualification(h))
    with pytest.raises(RegistrationRejected, match='reference clock'):
        h.interpolate_with_delay(10.05, qualification(h, reference_clock='request_midpoint'))
    old_owner = kernel_history()
    with pytest.raises(RegistrationRejected, match='different SDK history owner'):
        old_owner.validate_delay_qualification(qualification(h, reference_clock='kernel_receive'))
    default = fill(history(encoder_sample_age_bound_s=None))
    with pytest.raises(RegistrationRejected, match='reference clock'):
        default.validate_delay_qualification(qualification(default, reference_clock='kernel_receive'))


def test_reference_default_and_known_physical_age_semantics_are_unchanged():
    h = fill(history(encoder_sample_age_bound_s=.001))
    assert h.reference_clock == 'request_midpoint'
    assert h.interpolate(10.05).timing_uncertainty_s == pytest.approx(.00075)
    with pytest.raises(ValueError, match='unknown sample age'):
        history(reference_clock='kernel_receive', encoder_sample_age_bound_s=.001)
    with pytest.raises(ValueError, match='Reference clock'):
        history(reference_clock='kernel_hardware', encoder_sample_age_bound_s=None)


def test_rx_clock_uncertainty_includes_observed_wall_mono_offset_drift():
    h = kernel_history()
    assert receive_sample(h, 10.)
    assert receive_sample(h, 10.01)
    # A50us offset change is below the epoch-step threshold, but must NEVER
    # silently become an exact timestamp. Receipt remains inside request bounds.
    assert receive_sample(h, 10.02, anchor_offset=1000. - .00005)
    row = h.raw_series(10.019, 10.021)['axes'][0][-1]
    assert row[0] == pytest.approx(10.02005)
    assert row[3] >= .00005 - 1e-10
    assert row[0] - row[3] <= 10.02 + 1e-10


def test_rx_clock_uncertainty_includes_binary64_epoch_and_nanosecond_resolution():
    h = kernel_history()
    epoch_offset = 1_800_000_000.
    for stamp in (10., 10.01, 10.02):
        assert receive_sample(h, stamp, offset=epoch_offset)
    result = h.provisional_interpolate(10.015)
    assert result.timing_uncertainty_s >= abs(np.spacing(epoch_offset)) / 2
    assert result.timing_uncertainty_s < 1e-5
    raw = h.raw_series(10.005, 10.025)
    assert raw['reference_clock'] == 'kernel_receive'
    assert raw['columns'][3] == 'receive_clock_uncertainty_s'
    assert 'firmware sample age unknown' in raw['timestamp_semantics']


@pytest.mark.parametrize('bad', ['fault', 'disabled', 'branch', 'badbracket', 'wallstep', 'changedcached'])
def test_rx_reference_retains_packet_provenance_fault_and_epoch_guards(bad):
    h = kernel_history()
    for stamp in (10., 10.01, 10.02):
        assert receive_sample(h, stamp)
    generation = h.state()['generation']
    q = Q.copy()
    received, rb = np.full(6, 1010.03), np.tile([10.0296, 10.0301], (6, 1))
    errors, fault, anchor = ['0x1'] * 6, None, HostClockAnchor(1010.031, 10.031, 0.)
    if bad == 'fault':
        fault = {'kind': 'watchdog'}
    elif bad == 'disabled':
        errors[2] = '0x0'
    elif bad == 'branch':
        q[5] += 2 * np.pi
    elif bad == 'badbracket':
        rb[:] = [10.031, 10.032]
    elif bad == 'wallstep':
        received += 1
        anchor = HostClockAnchor(1011.031, 10.031, 0.)
    else:
        received[:] = 1010.02
        rb[:] = [10.0196, 10.0201]
        q[1] += .01
    assert not h.append(q, np.zeros(6), received, errors=errors, motion_fault=fault,
                        request_bounds_monotonic_s=rb, anchor=anchor)
    assert h.state()['generation'] == generation + 1
    assert not h._receive_anchors
    with pytest.raises(RegistrationRejected):
        h.provisional_interpolate(10.015)


def test_rx_reference_retains_two_millisecond_total_timing_gate():
    h = kernel_history()
    for stamp in (10., 10.01, 10.02):
        assert receive_sample(h, stamp, anchor_error=.0015)
    with pytest.raises(RegistrationRejected, match='timing uncertainty'):
        h.provisional_interpolate(10.015, timing_uncertainty_s=.0006)


def test_kernel_reference_clock_anchors_are_bounded_and_duplicate_values_do_not_churn():
    h = kernel_history(max_samples=4)
    for stamp in np.arange(10., 10.301, .001):
        assert receive_sample(h, float(stamp))
    assert len(h._receive_anchors) == 256
    before = h.state()['samples_per_joint']
    assert receive_sample(h, float(stamp))
    assert h.state()['samples_per_joint'] == before == [4] * 6
    assert len(h._receive_anchors) == 256


def test_reference_clock_cannot_be_switched_inside_an_existing_history_epoch():
    h = kernel_history()
    with pytest.raises(AttributeError):
        h.reference_clock = 'request_midpoint'
    assert h.reference_clock == 'kernel_receive'


def receive_cloud():
    value = cloud()
    value.update(acquisition_session_id='acq-1', generation=7, frame_product={'kind': 'single_frame'})
    value['timing'].update(device_frame_id=31, chunk_frame_id=31, exposure_end_frame_id=31,
                           acquisition_session_id='acq-1')
    return value


def test_receive_reference_registration_uses_actual_exposure_angles_without_physical_proof():
    h = fill(kernel_history(), q_func=lambda t: Q + np.array([.01*(t-10), 0, 0, 0, 0, 0]),
             v=np.array([.01, 0, 0, 0, 0, 0]))
    native = receive_cloud()
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    got = registrar.register_receive_reference(native)
    np.testing.assert_allclose(got['points_world_m'][:, 0], [.0005, .0105], atol=1e-12)
    assert got['registration_reference'] == 'measured_kernel_rx'
    assert got['registration_method'] == 'measured_output_receive_reference'
    assert got['reference_registered'] and got['reference_timing_qualified']
    assert got['hardware_exposure_timing_qualified']
    assert got['physical_sample_timing_qualified'] is False
    assert got['timing_qualified'] is False
    assert got['firmware_sample_age_s'] is None
    assert got['physical_spatial_error_bound_m'] is None
    assert got['physical_timing_uncertainty_s'] is None
    assert got['timing_uncertainty_s'] is got['spatial_error_bound_m'] is got['joint_error_rad'] is None
    assert got['applied_empirical_delay_s'] == got['applied_delay_s'] == 0.
    assert got['delay_qualification_id'] is None
    assert got['no_extrapolation']
    assert got['reference_clock_uncertainty_s'] < 1e-6
    assert got['reference_timing_uncertainty_s'] >= got['camera_clock_uncertainty_s'] == .0001
    assert 0 <= got['reference_spatial_error_bound_m'] <= got['max_reference_spatial_error_m'] == .001
    assert 0 <= got['reference_timing_uncertainty_s'] <= got['max_reference_timing_uncertainty_s'] == .002
    assert got['reference_joint_error_rad'].shape == (6,)
    assert native['xyz_m'][0][0] == 0.
    assert not h.state()['motion_pairing_supported']
    with pytest.raises(RegistrationRejected, match='sample-age bound is unknown'):
        registrar.register(native)


def test_receive_reference_ignores_installed_empirical_offset_instead_of_faking_zero_qualification():
    h = fill(kernel_history(), q_func=lambda t: Q + np.array([.01*(t-10), 0, 0, 0, 0, 0]),
             v=np.array([.01, 0, 0, 0, 0, 0]))
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    registrar.set_delay_qualification(qualification(h, reference_clock='kernel_receive'))
    reference = registrar.register_receive_reference(receive_cloud())
    strict = registrar.register(receive_cloud())
    assert reference['delay_qualification_id'] is None and reference['applied_delay_s'] == 0.
    assert strict['delay_qualification_id'] == 'fit-1' and strict['applied_delay_s'] == .01
    assert reference['joint_pos'][0] == pytest.approx(.0005)
    assert strict['joint_pos'][0] == pytest.approx(.0006)


@pytest.mark.parametrize('h', [lambda: fill(history()), lambda: fill(history(encoder_sample_age_bound_s=None))])
def test_receive_reference_cannot_relabel_other_history_clock_modes(h):
    with pytest.raises(RegistrationRejected, match='kernel_receive history'):
        MovingCloudRegistrar(h(), LinearFK(), handeye()).register_receive_reference(receive_cloud())


@pytest.mark.parametrize('bad', ['device', 'chunk', 'event', 'mapping', 'session', 'singleframe',
                                'sequence', 'generation', 'shutter', 'actualexposure', 'optical', 'units',
                                'exposureclock', 'nanclock', 'no_exposure'])
def test_receive_reference_retains_exact_native_identity_exposure_and_units_gates(bad):
    h = fill(kernel_history())
    native = receive_cloud()
    if bad in ('device', 'chunk', 'event'):
        key = {'device': 'device_frame_id', 'chunk': 'chunk_frame_id', 'event': 'exposure_end_frame_id'}[bad]
        native['timing'][key] = 32
    elif bad == 'mapping':
        native['timing']['clock_mapping']['mapping_id'] = None
    elif bad == 'session':
        native['timing']['acquisition_session_id'] = 'old'
    elif bad == 'singleframe':
        native['frame_product']['kind'] = 'merged'
    elif bad in ('sequence', 'generation'):
        native[bad] = -1
    elif bad == 'shutter':
        native['timing']['sensor_shutter_mode'] = 'rolling'
    elif bad == 'actualexposure':
        native['timing']['exposure_duration_us'] = 1200.
    elif bad == 'optical':
        native['optical_intrinsics_id'] = 'old'
    elif bad == 'units':
        native['units'] = 'mm'
    elif bad == 'exposureclock':
        native['timing']['clock_mapping']['status'] = 'unqualified'
    elif bad == 'nanclock':
        native['timing']['exposure_time_uncertainty_s'] = np.nan
    else:
        native['timing'].pop('exposure_host_start_s')
    with pytest.raises(RegistrationRejected):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(native)


def test_receive_reference_no_return_is_empty_measured_coverage_not_invented_geometry():
    h = fill(kernel_history())
    native = receive_cloud()
    native['xyz_m'] = []
    got = MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(native)
    assert got['points_world_m'].shape == (0, 3)
    assert got['reference_spatial_error_bound_m'] == 0.
    assert got['physical_spatial_error_bound_m'] is None
    assert got['joint_pos'].shape == (6,)


def test_receive_reference_preserves_no_extrapolation_epoch_clock_and_spatial_reference_metrics():
    from dataclasses import replace

    h = fill(kernel_history(), q_func=lambda t: Q + np.array([.5*(t-10), 0, 0, 0, 0, 0]),
             v=np.array([.5, 0, 0, 0, 0, 0]))
    native = receive_cloud()
    native['timing'].update(exposure_host_start_s=10.045, exposure_host_end_s=10.055, exposure_duration_us=10000.)
    with pytest.raises(RegistrationRejected, match='reference spatial metric'):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(native)
    native = receive_cloud()
    native['timing']['exposure_time_uncertainty_s'] = .002
    with pytest.raises(RegistrationRejected, match='timing uncertainty'):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(native)
    native = receive_cloud()
    native['timing'].update(exposure_host_start_s=10.1495, exposure_host_midpoint_s=10.15, exposure_host_end_s=10.1505)
    with pytest.raises(RegistrationRejected, match='extrapolation'):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(native)
    original = h.interpolate_receive_reference
    epoch = [0]
    def changing_epoch(*args, **kwargs):
        epoch[0] += 1
        return replace(original(*args, **kwargs), generation=epoch[0])
    h.interpolate_receive_reference = changing_epoch
    with pytest.raises(RegistrationRejected, match='epoch changed'):
        MovingCloudRegistrar(h, LinearFK(), handeye()).register_receive_reference(receive_cloud())


def test_receive_reference_gap_policy_is_separate_from_strict_and_experimental_twenty_ms():
    h = kernel_history()
    assert receive_sample(h, 10.)
    assert receive_sample(h, 10.070)
    with pytest.raises(RegistrationRejected, match='maximum gap'):
        h.provisional_interpolate(10.035)
    received = h.interpolate_receive_reference(10.035)
    assert received.max_gap_s == pytest.approx(.070)
    assert received.calibration_only  # Still no physical timing qualification.
    assert h.max_gap_s == .020
    known = history()
    assert receive_sample(known, 10.)
    assert receive_sample(known, 10.070)
    with pytest.raises(RegistrationRejected, match='maximum gap'):
        known.interpolate(10.035)
    with pytest.raises(RegistrationRejected, match='kernel_receive history'):
        known.interpolate_receive_reference(10.035)


def test_receive_reference_gap_limit_never_extrapolates_or_exceeds_explicit_hundred_ms():
    h = kernel_history()
    assert receive_sample(h, 10.)
    assert receive_sample(h, 10.1001)
    with pytest.raises(RegistrationRejected, match='maximum gap'):
        h.interpolate_receive_reference(10.05)
    with pytest.raises(RegistrationRejected, match='100'):
        h.interpolate_receive_reference(10.05, max_gap_s=.101)
    with pytest.raises(RegistrationRejected, match='extrapolation'):
        h.interpolate_receive_reference(10.2)


def test_receive_registrar_exposes_hundred_ms_one_mm_reference_only_metrics():
    h = kernel_history()
    assert receive_sample(h, 10.)
    assert receive_sample(h, 10.070)
    native = receive_cloud()
    native['timing'].update(exposure_host_start_s=10.0345, exposure_host_midpoint_s=10.035,
                            exposure_host_end_s=10.0355)
    registrar = MovingCloudRegistrar(h, LinearFK(), handeye())
    got = registrar.register_receive_reference(native)
    assert got['max_reference_gap_s'] == .100
    assert got['robot_max_gap_s'] == pytest.approx(.070)
    assert .0005 < got['reference_spatial_error_bound_m'] < .001
    assert got['max_reference_spatial_error_m'] == .001
    assert got['physical_spatial_error_bound_m'] is None
    assert got['physical_sample_timing_qualified'] is False
    assert got['firmware_sample_age_s'] is None
    assert got['max_reference_timing_uncertainty_s'] == .002
    # Callers can explicitly choose the narrower reference policy too, without
    # mutating strict history/registrar limits or substituting any pose.
    with pytest.raises(RegistrationRejected, match='maximum gap'):
        MovingCloudRegistrar(h, LinearFK(), handeye(), max_reference_gap_s=.020).register_receive_reference(native)
    with pytest.raises(RegistrationRejected, match='reference spatial metric'):
        MovingCloudRegistrar(h, LinearFK(), handeye(), max_reference_spatial_error_m=.0005).register_receive_reference(native)
    assert registrar.max_spatial_error_m == .0005
    native['timing']['exposure_time_uncertainty_s'] = .002
    with pytest.raises(RegistrationRejected, match='timing uncertainty'):
        registrar.register_receive_reference(native)
