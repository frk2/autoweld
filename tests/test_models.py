import cadquery as cq
import mujoco
import numpy as np
import yourdfpy
from i2rt.robots.kinematics import Kinematics

from autoweld.cad import load_step
from autoweld.models import URDF, camera_from_flange, tool_model


def test_step_units_multisolid_and_curves(tmp_path):
    box = cq.Workplane("XY").box(100, 80, 20)
    cylinder = cq.Workplane("XY").center(200, 0).circle(15).extrude(30)
    model = cq.Compound.makeCompound([box.val(), cylinder.val()])
    path = tmp_path / "fixture.step"
    cq.exporters.export(model, str(path))
    mesh = load_step(path)
    assert len(mesh.faces) > 20
    np.testing.assert_allclose(mesh.bounds[:, 0], [-0.05, 0.215], atol=1e-4)
    np.testing.assert_allclose(mesh.extents[1], 0.08, atol=1e-4)


def test_urdf_and_ik_tcp_agree_at_nonzero_poses():
    path = tool_model()
    model = mujoco.MjModel.from_xml_path(path)
    urdf = yourdfpy.URDF.load(str(URDF))
    kin = Kinematics(path, "weld_tcp")
    offset = camera_from_flange()
    for arm in ([0, 0, 0, 0, 0, 0], [0.12, 0.4, 0.8, 0.1, -0.15, 0.2]):
        q = np.zeros(model.nq)
        q[:6] = arm
        urdf.update_cfg(dict(zip([f"joint{i + 1}" for i in range(6)], map(float, arm))))
        expected = urdf.get_transform("gripper") @ offset
        np.testing.assert_allclose(kin.fk(q), expected, atol=1e-5)


def test_installed_mount_has_six_joints_and_no_gripper():
    model = mujoco.MjModel.from_xml_path(tool_model())
    assert model.nq == 6
    assert model.njnt == 6
    urdf = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    assert len(urdf.actuated_joints) == 6
    assert "tip_left" not in urdf.link_map
    assert "camera_optical" in urdf.link_map
    assert any(name.startswith("mount_") for name in urdf.link_map)


def test_nominal_cad_camera_and_motion_tcp_are_distinct_consistent_frames():
    import json
    from pathlib import Path

    manifest = json.loads((Path(URDF).parent/'manifest.json').read_text())
    urdf = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    model = mujoco.MjModel.from_xml_path(tool_model())
    data = mujoco.MjData(model)
    data.qpos[:] = [-.05, .1, .2, -.1, .1, -2.5]
    mujoco.mj_forward(model, data)
    urdf.update_cfg(dict(zip([f'joint{i+1}' for i in range(6)], data.qpos)))
    flange = urdf.get_transform('gripper')
    for site, frame in (('weld_tcp', 'active_tcp'), ('camera_optical', 'camera_optical')):
        expected = flange @ np.asarray(manifest['frames'][frame])
        np.testing.assert_allclose(data.site(site).xpos, expected[:3, 3], atol=1e-5)
        np.testing.assert_allclose(data.site(site).xmat.reshape(3, 3), expected[:3, :3], atol=1e-5)
        np.testing.assert_allclose(urdf.get_transform(site), expected, atol=1e-10)


def test_mount_alignment_preserves_calibrated_motion_reference_and_vertical_stack():
    import json
    from pathlib import Path

    manifest = json.loads((Path(URDF).parent/'manifest.json').read_text())
    np.testing.assert_array_equal(camera_from_flange(), [
        [0., 1., 0., -.0695], [1., 0., 0., .03209562859406243],
        [0., 0., -1., -.068782516614], [0., 0., 0., 1.]])
    reference = manifest['display_alignment']
    urdf = yourdfpy.URDF.load(str(URDF), load_meshes=False)
    urdf.update_cfg(dict(zip([f'joint{i+1}' for i in range(6)], reference['reference_joint_rad'])))
    flange = urdf.get_transform('gripper')
    delta = flange[:3, :3] @ (np.asarray(reference['camera_center_flange_m'])
                              - np.asarray(reference['laser_center_flange_m']))
    np.testing.assert_allclose(delta[:2], 0., atol=1e-10)
    assert delta[2] > .07


def test_sdk_ik_reaches_small_known_waypoint():
    path = tool_model()
    model = mujoco.MjModel.from_xml_path(path)
    kin = Kinematics(path, "weld_tcp")
    q = np.zeros(model.nq)
    q[:6] = [0.1, 0.4, 0.8, 0.1, -0.15, -2.8]
    goal = q.copy()
    goal[0] += 0.04
    goal[3] += 0.03
    target = kin.fk(goal)
    ok, result = kin.ik(target, "weld_tcp", init_q=q)
    assert ok
    np.testing.assert_allclose(kin.fk(result), target, atol=2e-4)


def test_sdk_planner_and_reconnect_share_one_compiled_model():
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType
    from i2rt.utils.mujoco_utils import MuJoCoKDL

    from autoweld.control import Controller
    from autoweld.kinematics import shared_kinematics
    from autoweld.models import compiled_model
    from autoweld.motion import MotionPlanner

    model = compiled_model()
    assert compiled_model() is model
    # The supplied model is authoritative, so dynamics never opens this path.
    dynamics = MuJoCoKDL('/does-not-exist/shared-model.xml', model=model)
    assert dynamics.model is model
    before = model.geom_contype.copy(), model.geom_conaffinity.copy(), model.jnt_limited.copy()
    robots = []
    def connect(*args, **kwargs):
        robot = get_yam_robot(sim=True, gripper_type=GripperType.NO_GRIPPER,
                              xml_path_override=tool_model(), compiled_model=model)
        robots.append(robot)
        return robot
    controller = Controller(connect(), model.jnt_range[:6], robot_factory=connect)
    try:
        planner = MotionPlanner(tool_model(), kin=shared_kinematics(controller.robot))
        assert planner.model is planner.kin._configuration.model is controller.robot._model is model
        controller.disabled_q = controller.positions()
        controller.rest_verified = True
        controller._disable_at_rest()
        controller.enable()
        assert len(robots) == 2
        assert controller.robot._model is planner.model is model
        np.testing.assert_array_equal(model.geom_contype, before[0])
        np.testing.assert_array_equal(model.geom_conaffinity, before[1])
        np.testing.assert_array_equal(model.jnt_limited, before[2])
    finally:
        for robot in robots:
            robot.close()
