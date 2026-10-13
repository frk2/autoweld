import cadquery as cq
import mujoco
import numpy as np
import yourdfpy
from i2rt.robots.kinematics import Kinematics

from autoweld.app import URDF, tool_model
from autoweld.cad import load_step


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
    offset = np.eye(4)
    offset[2, 3] = -0.15
    for arm in ([0, 0, 0, 0, 0, 0], [0.12, 0.4, 0.8, 0.1, -0.15, 0.2]):
        q = np.zeros(model.nq)
        q[:6] = arm
        urdf.update_cfg(dict(zip([f"joint{i + 1}" for i in range(6)], map(float, arm))))
        expected = urdf.get_transform("gripper") @ offset
        np.testing.assert_allclose(kin.fk(q), expected, atol=1e-5)


def test_sdk_ik_reaches_small_known_waypoint():
    path = tool_model()
    model = mujoco.MjModel.from_xml_path(path)
    kin = Kinematics(path, "weld_tcp")
    q = np.zeros(model.nq)
    q[:6] = [0.1, 0.4, 0.8, 0.1, -0.15, 0.2]
    goal = q.copy()
    goal[0] += 0.04
    goal[3] += 0.03
    target = kin.fk(goal)
    ok, result = kin.ik(target, "weld_tcp", init_q=q)
    assert ok
    np.testing.assert_allclose(kin.fk(result), target, atol=2e-4)
