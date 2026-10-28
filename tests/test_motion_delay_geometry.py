import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.motion_delay import DelayCalibrationRejected, fit_motion_delay
from autoweld.motion_delay_geometry import fit_geometry_motion_delay


def camera_pose(times):
    times = np.asarray(times)
    result = np.tile(np.eye(4), (len(times), 1, 1))
    result[:, :3, 3] = np.column_stack((.002 * np.sin(2*np.pi*times/1.8 + .3),
        .0015*np.sin(2*np.pi*times/1.1),
        .004*np.sin(2*np.pi*times/1.5) + .001*np.sin(2*np.pi*times/.57)))
    rotvec = np.column_stack((.020*np.sin(2*np.pi*times/1.7),
        .035*np.sin(2*np.pi*times/1.3 + .2), .010*np.sin(2*np.pi*times/.9)))
    result[:, :3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return result


def scene(delay=.017, frame_noise=2e-6, point_noise=2e-6):
    outputs = np.arange(0., 9., .004)
    times = np.arange(.5, 8.5, .04)
    cycles = np.floor((times - .5) / 2).astype(int)
    physical = camera_pose(times)
    native = []
    rng = np.random.default_rng(4)
    for i, stamp in enumerate(times):
        x = np.linspace(-.015, .015, 12)
        y = .5*x + .004*np.sin(2*np.pi*stamp/.8)
        z = .14 + .2*x + .1*y + 10*x*x + 4*x*y + 8*y*y
        world = np.column_stack((x, y, z))
        points = (world - physical[i, :3, 3]) @ physical[i, :3, :3]
        points[:, 2] += rng.normal(0., frame_noise) + rng.normal(0., point_noise, len(points))
        native.append(points)
    return times, native, cycles, outputs, camera_pose(outputs - delay)


@pytest.mark.parametrize('delay', [.017, -.023])
def test_full_pose_recovers_signed_delay_with_rotating_translating_camera_curved_surface(delay):
    args = scene(delay)
    result = fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.060, .060))
    assert abs(result.delay_s - delay) < .0005
    assert result.confidence_interval_s[0] <= delay <= result.confidence_interval_s[1]
    assert result.uncertainty_s <= .002
    assert result.qualified and result.heldout_passed and not result.diagnostic_only
    assert result.heldout_rms_m < 2e-5
    assert result.direction_bias_m < 2e-5
    assert result.method == 'full_pose_static_quadratic_surface'
    # Z-only fitting discards the actual XY and R and must not be substituted.
    times, native, cycles, outputs, poses = args
    try:
        old = fit_motion_delay(times, [np.median(p[:, 2]) for p in native], cycles,
                               outputs, poses[:, 2, 3], search_s=(-.060, .060))
    except DelayCalibrationRejected:
        pass
    else:
        assert abs(old.delay_s - delay) > .005


def test_coarse_hint_is_never_a_qualified_record():
    result = fit_geometry_motion_delay(*scene(), baseline_optical=np.eye(4),
                                      search_s=(-.15, .15), diagnostic_only=True)
    assert abs(result.delay_s - .017) < .001
    assert result.diagnostic_only and not result.qualified and not result.heldout_passed
    assert result.uncertainty_s is None and result.confidence_interval_s is None


def test_native_point_count_does_not_multiply_frame_confidence():
    args = list(scene())
    first = fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(.005, .030))
    args[1] = [np.repeat(p, 4, axis=0) for p in args[1]]
    repeated = fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(.005, .030))
    assert repeated.delay_s == pytest.approx(first.delay_s, abs=2e-7)
    assert repeated.uncertainty_s == pytest.approx(first.uncertainty_s, abs=2e-7)
    assert repeated.samples == first.samples


def test_changing_surface_is_rejected_by_independent_cycles():
    args = list(scene())
    args[1] = [points + [0, 0, (.001 if cycle % 2 else 0)] for points, cycle in zip(args[1], args[2])]
    with pytest.raises(DelayCalibrationRejected):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.040, .040))


def test_cycle_dependent_latency_cannot_average_into_false_precision():
    args = list(scene())
    times, _, cycles, _, _ = args
    # Create native data under different physical poses per cycle but retain one
    # reference trace; there is no single empirically valid delay.
    nominal = camera_pose(times)
    varying = camera_pose(times + np.where(cycles % 2, .008, -.008))
    clouds = []
    for points, before, after in zip(args[1], nominal, varying):
        world = points @ before[:3, :3].T + before[:3, 3]
        clouds.append((world - after[:3, 3]) @ after[:3, :3])
    args[1] = clouds
    with pytest.raises(DelayCalibrationRejected):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.040, .040))


def test_actual_query_gap_and_extrapolation_reject_but_irrelevant_trace_gap_is_allowed():
    args = list(scene())
    # No frame near this hole after removing its entire +/-40ms trial window.
    outputs = args[3]
    keep_outputs = (outputs < 4.) | (outputs > 4.1)
    keep_frames = (args[0] < 3.94) | (args[0] > 4.16)
    args[0], args[1], args[2] = args[0][keep_frames], [p for p, keep in zip(args[1], keep_frames) if keep], args[2][keep_frames]
    args[3], args[4] = outputs[keep_outputs], args[4][keep_outputs]
    got = fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.040, .040))
    assert got.qualified
    args = list(scene())
    keep = (args[3] < 4.) | (args[3] > 4.1)
    args[3], args[4] = args[3][keep], args[4][keep]
    with pytest.raises(DelayCalibrationRejected, match='gap'):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.040, .040))
    args = list(scene())
    args[3], args[4] = args[3][300:], args[4][300:]
    with pytest.raises(DelayCalibrationRejected, match='bracket'):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.040, .040))


def test_static_camera_and_invalid_pose_cannot_qualify():
    args = list(scene())
    args[4] = np.tile(np.eye(4), (len(args[3]), 1, 1))
    with pytest.raises(DelayCalibrationRejected, match='observable'):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.04, .04))
    args = list(scene())
    args[4][20, 0, 0] = 2.
    with pytest.raises(DelayCalibrationRejected, match='proper'):
        fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(-.04, .04))


def test_original_baseline_frame_and_arbitrary_robot_world_transform_are_respected():
    args = list(scene())
    baseline = np.eye(4)
    baseline[:3, :3] = Rotation.from_euler('xyz', [.6, -.8, 1.]).as_matrix()
    baseline[:3, 3] = [.3, -.2, .4]
    args[4] = baseline @ args[4]
    got = fit_geometry_motion_delay(*args, baseline_optical=baseline, search_s=(.005, .030))
    assert abs(got.delay_s - .017) < .0005
    assert got.qualified and got.heldout_rms_m < 2e-5


def test_laser_line_need_not_identify_all_six_surface_coefficients():
    args = list(scene(frame_noise=0., point_noise=0.))
    physical = camera_pose(args[0])
    x = np.linspace(-.015, .015, 12)
    world = np.column_stack((x, .5*x, .140 + .2*x + 10*x*x))
    args[1] = [(world - pose[:3, 3]) @ pose[:3, :3] for pose in physical]
    got = fit_geometry_motion_delay(*args, baseline_optical=np.eye(4), search_s=(.005, .030))
    assert abs(got.delay_s - .017) < .0005
    assert got.qualified
