"""Reuse the SDK-owned MuJoCo model for planning without compiling another."""

import mink
from i2rt.robots.kinematics import Kinematics


def shared_kinematics(robot, site="weld_tcp"):
    model = robot.kdl.model if hasattr(robot, "kdl") else robot._model
    kin = Kinematics.__new__(Kinematics)
    kin._configuration = mink.Configuration(model)
    kin._site_name = site
    return kin
