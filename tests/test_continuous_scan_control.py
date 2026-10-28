import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots.waypoint_motion import JointPath, TimedJointPath
from scipy.spatial.transform import Rotation
from test_control import FakeRobot

from autoweld.continuous_raster import ContinuousRaster, RasterPlanBuilder, RasterSettings
from autoweld.control import FACTORY_KD, FACTORY_KP, PID_LIMIT, Controller
from autoweld.motion import MotionLimits
from autoweld.vertical_scan import VerticalLine, WorldLinePlanBuilder


@pytest.fixture(scope='module')
def validated_plans():
    result = {}
    for gripper in (0., .4):
        seed = np.zeros(7)
        seed[6] = gripper

        def solve(pose, prior):
            q = prior.copy()
            q[:3] = pose[:3, 3]
            return q

        def fk(q):
            pose = np.eye(4)
            pose[:3, 3] = q[:3]
            pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
            return pose

        builder = RasterPlanBuilder(
            ContinuousRaster(RasterSettings(width_m=.020, height_m=.001)), seed,
            baseline_optical=np.eye(4), fixed_optical_rotation=np.eye(3), tcp_from_camera=np.eye(4),
            solve=solve, fk=fk, jacobian=lambda q: np.eye(6), bounds=np.tile([-3., 3.], (6, 1)),
            validate=lambda q: None, collision=lambda q: False)
        result[gripper] = (builder.run(), fk)
    return result


def controller_and_plan(validated_plans, six=False, live=False):
    plan, fk = validated_plans[0. if six else .4]
    robot = FakeRobot()
    robot.q = plan.path.positions[0, :6 if six else 7].copy()
    robot.moves = []
    robot.command_joint_motion = lambda path, ki, limits, **kwargs: robot.moves.append((path, ki, limits, kwargs))
    controller = Controller(robot, np.tile([-3., 3.], (6, 1)))
    controller.motion_planner = SimpleNamespace(limits=MotionLimits(), fk=fk)
    controller.live = live
    events = []
    controller.health = lambda: events.append('health')
    controller._wait_for_fresh_motion_start = lambda: events.append('fresh_after_validation')
    return controller, copy.deepcopy(plan), events


@pytest.mark.parametrize('six', [False, True])
def test_scan_uses_named_local_timing_normal_pid_and_fixed_main_owner_state(validated_plans, six):
    controller, plan, events = controller_and_plan(validated_plans, six=six, live=True)
    controller.command_continuous_scan(plan)
    assert controller.mode == 'Laser scanning'
    assert controller.motion_automatic is False
    assert controller.motion_plan is not plan
    assert controller.motion_plan.path is not plan.path
    assert 'fresh_after_validation' in events
    path, ki, limits, gains = controller.robot.moves[0]
    assert type(path) is TimedJointPath
    count = 6 if six else 7
    assert path.positions.shape[1] == count
    np.testing.assert_array_equal(ki, np.zeros(count))
    np.testing.assert_allclose(limits, np.r_[PID_LIMIT, 0.][:count])
    np.testing.assert_allclose(gains['kp'], np.r_[FACTORY_KP, 5.][:count])
    np.testing.assert_allclose(gains['kd'], np.r_[FACTORY_KD, 0.][:count])
    np.testing.assert_allclose(controller.target, plan.path.positions[-1])


@pytest.mark.parametrize('tamper', ['kind', 'limits', 'nan_peak', 'collision', 'constraints', 'digest',
                                  'bounds', 'samples', 'residual', 'scope', 'axis', 'plain_path'])
def test_unproved_or_mutated_path_never_submits(validated_plans, tamper):
    controller, plan, _ = controller_and_plan(validated_plans, live=True)
    if tamper == 'kind':
        plan.report['kind'] = 'Joint'
    elif tamper == 'limits':
        plan.report['limits']['linear_acceleration'] = 1.
    elif tamper == 'nan_peak':
        plan.report['planned_peaks']['joint_jerk'] = np.nan
    elif tamper in ('collision', 'constraints'):
        key = 'collision_checked' if tamper == 'collision' else 'command_constraints_checked'
        plan.report['validation_proof'][key] = False
    elif tamper == 'digest':
        plan.path = replace(plan.path, duration=plan.path.duration + .1)
        plan.report['duration_s'] = plan.path.duration
    elif tamper == 'bounds':
        controller.limits[0, 0] += .1
    elif tamper == 'samples':
        plan.report['validation_samples'] = 1
    elif tamper == 'residual':
        plan.report['tcp_path_error_m'] = .00021
    elif tamper == 'scope':
        plan.report['scan_scope']['width_m'] = .101
    elif tamper == 'axis':
        plan.report['scan_scope']['fixed_optical_rotation'] = Rotation.from_euler('x', .01).as_matrix().tolist()
    else:
        plan.path = JointPath(plan.path.knots, plan.path.positions, plan.path.duration)
    with pytest.raises(ValueError):
        controller.command_continuous_scan(plan)
    assert not controller.robot.moves and controller.motion_plan is None


@pytest.mark.parametrize('offset,expected', [([.009, 0, 0, 0, 0, 0], 'joint mismatch'),
                                           ([.0006, 0, 0, 0, 0, 0], '0.5 mm'),
                                           ([0, 0, 0, .006, 0, 0], '0.005 rad')])
def test_actual_fresh_start_must_match_both_joint_and_tcp_endpoint(validated_plans, offset, expected):
    controller, plan, _ = controller_and_plan(validated_plans, live=True)
    controller.robot.q[:6] += offset
    with pytest.raises(ValueError, match=expected):
        controller.command_continuous_scan(plan)
    assert not controller.robot.moves


@pytest.mark.parametrize('blocker', ['cooldown', 'shutdown', 'disable', 'busy', 'stale', 'speed'])
def test_normal_admission_and_fresh_after_validation_gates_preserved(validated_plans, blocker):
    controller, plan, _ = controller_and_plan(validated_plans, live=True)
    if blocker == 'cooldown':
        controller.cooldown_active = True
    elif blocker == 'shutdown':
        controller.shutdown_requested = True
    elif blocker == 'disable':
        controller.disable_pending = True
    elif blocker == 'busy':
        controller.motion_plan = object()
    elif blocker == 'speed':
        controller.robot.velocity = np.ones(7) * .051
    else:
        def stale():
            raise RuntimeError('No fresh SDK feedback after path validation')
        controller._wait_for_fresh_motion_start = stale
    with pytest.raises(RuntimeError):
        controller.command_continuous_scan(plan)
    assert not controller.robot.moves


def test_worker_cannot_call_controller_path_submission_directly(validated_plans):
    from concurrent.futures import ThreadPoolExecutor

    controller, plan, _ = controller_and_plan(validated_plans, live=True)
    with ThreadPoolExecutor(max_workers=1) as pool:
        error = pool.submit(controller.command_continuous_scan, plan).exception(timeout=1)
    assert isinstance(error, RuntimeError)
    assert 'main owner' in str(error)
    assert not controller.robot.moves


@pytest.fixture(scope='module')
def validated_vertical_plan():
    seed = np.zeros(7)
    seed[:3] = [.03, .04, .20]
    seed[3:6] = [.10, -.20, .30]

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def solve(pose, prior):
        result = prior.copy()
        result[:3] = pose[:3, 3]
        result[3:6] = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        return result

    start = fk(seed)
    plan = WorldLinePlanBuilder(VerticalLine(), seed, baseline_optical=start,
        fixed_optical_rotation=start[:3, :3], tcp_from_camera=np.eye(4),
        solve=solve, fk=fk, jacobian=lambda q: np.eye(6),
        bounds=np.tile([-3., 3.], (6, 1)), validate=lambda q: None,
        collision=lambda q: False).run()
    return plan, fk


def vertical_controller(validated_vertical_plan):
    plan, fk = validated_vertical_plan
    robot = FakeRobot()
    robot.q = plan.path.positions[0].copy()
    robot.moves = []
    robot.command_joint_motion = lambda *args, **kwargs: robot.moves.append((args, kwargs))
    controller = Controller(robot, np.tile([-3., 3.], (6, 1)))
    controller.motion_planner = SimpleNamespace(limits=MotionLimits(), fk=fk)
    controller.live = True
    controller.health = lambda: None
    controller._wait_for_fresh_motion_start = lambda: None
    return controller, copy.deepcopy(plan)


def test_vertical_line_admits_proved_world_z_path_with_tilted_camera(validated_vertical_plan):
    controller, plan = vertical_controller(validated_vertical_plan)
    controller.command_continuous_scan(plan)
    assert len(controller.robot.moves) == 1
    assert controller.mode == 'Laser scanning'
    np.testing.assert_allclose(controller.target[:3], [.03, .04, .12], atol=1e-10)
    assert controller.last_motion['validation_proof']['schema'] == 'autoweld.continuous_world_line.v1'


@pytest.mark.parametrize('tamper', ['schema', 'profile', 'direction', 'length_zero', 'length_large',
    'length_bool', 'endpoint', 'rotation', 'off_line_knot', 'missing_scope', 'incomplete_end'])
def test_vertical_line_rejects_wrong_scope_even_with_matching_joint_digest(validated_vertical_plan, tamper):
    from autoweld.continuous_raster import raster_path_digest

    controller, plan = vertical_controller(validated_vertical_plan)
    scope = plan.report['scan_scope']
    if tamper == 'schema':
        plan.report['validation_proof']['schema'] = 'autoweld.continuous_raster.v1'
    elif tamper == 'profile':
        scope['scan_profile'] = 'area'
    elif tamper == 'direction':
        scope['direction_base'] = [0., -1., 0.]
    elif tamper.startswith('length_'):
        scope['length_m'] = {'length_zero': 0., 'length_large': .101, 'length_bool': True}[tamper]
    elif tamper == 'endpoint':
        scope['end_base_from_camera'][0][3] += .001
    elif tamper == 'rotation':
        scope['fixed_optical_rotation'] = np.eye(3).tolist()
    elif tamper == 'missing_scope':
        del scope['start_base_from_camera']
    else:
        positions = plan.path.positions.copy()
        if tamper == 'off_line_knot':
            positions[len(positions)//2, 0] += .001
        else:
            positions[-1, 2] += .001
        plan.path = TimedJointPath(plan.path.knots, positions, plan.path.duration)
        plan.report['validation_proof']['path_sha256'] = raster_path_digest(plan.path)
    with pytest.raises(ValueError):
        controller.command_continuous_scan(plan)
    assert not controller.robot.moves and controller.motion_plan is None
