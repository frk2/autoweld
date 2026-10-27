"""Current SDK YAM arm with the installed fixed receiver/camera geometry.

Only mount assets/frames were imported from the old worktree. J6 travel is
measured for this assembly; other arm geometry remains from the pinned SDK.
"""

import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from autoweld.joint_calibration import j6_operating_range

ROOT = Path(__file__).resolve().parents[1]
MOUNT = ROOT / "assets/camera_mount"
URDF = MOUNT / "camera_mount.urdf"


def camera_from_flange():
    """Return flange-to-camera transform (metres), independent of calibration."""
    return np.asarray(json.loads((MOUNT / "manifest.json").read_text())["flange_from_camera"], float)


def _numbers(values):
    return " ".join(format(float(v), ".14g") for v in values)


@lru_cache(maxsize=1)
def tool_model():
    """Build one model description; callers share its compiled kinematics model."""
    source = ROOT / "vendor/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"
    tree = ET.parse(source)
    tree.find("compiler").set("meshdir", str(source.parent / "assets"))
    tree.find(".//joint[@name='joint6']").set("range", _numbers(j6_operating_range()))
    mount = tree.find(".//body[@name='gripper']")
    manifest = json.loads((MOUNT / "manifest.json").read_text())
    # Conservative mount envelopes participate in self-collision checks. Mesh
    # visuals are rendered from URDF, avoiding a second set of compiled meshes.
    for entry in manifest["envelopes"]:
        xyzw = Rotation.from_euler("xyz", entry["rpy"]).as_quat()
        ET.SubElement(mount, "geom", name="mount_" + entry["name"], type="box",
                      pos=_numbers(entry["xyz"]), size=_numbers(np.asarray(entry["size"]) / 2),
                      quat=_numbers(np.roll(xyzw, 1)), rgba="0.21 0.49 0.62 0.3", mass="0")
    transform = np.asarray(manifest["flange_from_camera"])
    quat = np.roll(Rotation.from_matrix(transform[:3, :3]).as_quat(), 1)
    for name in ("weld_tcp", "camera_optical"):
        ET.SubElement(mount, "site", name=name, pos=_numbers(transform[:3, 3]),
                      quat=_numbers(quat), size="0.004")
    folder = ROOT / "runtime/models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "receiver-camera.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return str(path)


@lru_cache(maxsize=1)
def compiled_model():
    """One model shared across reconnects; payload updates pause SDK execution."""
    import mujoco

    return mujoco.MjModel.from_xml_path(tool_model())
