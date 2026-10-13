"""Exercise the running offline app's real command loop, never hardware."""

import json
import time
from pathlib import Path

import mujoco
import numpy as np
from i2rt.robots.kinematics import Kinematics
from scipy.spatial.transform import Rotation

from autoweld.app import atomic_json, tool_model

runtime = Path("runtime/sim")


def state():
    result = json.loads((runtime / "status.json").read_text())
    assert result.get("live") is False, "This smoke check must never control live hardware"
    assert time.time() - result["timestamp"] < 2, "Offline viewer is not running"
    return result


def request(payload):
    state()
    name = f"smoke-{time.time_ns()}"
    atomic_json(runtime / "requests" / f"{name}.json", payload)
    response = runtime / "responses" / f"{name}.json"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if response.exists():
            time.sleep(0.25)  # Allow the telemetry publication tick.
            result = json.loads(response.read_text())
            if result["ok"]:
                finish = time.monotonic()+30
                while state().get("motion_active") or state().get("disable_pending"):
                    if time.monotonic() > finish:
                        raise TimeoutError("Waypoint/disable did not finish")
                    time.sleep(.05)
            return result
        time.sleep(0.05)
    raise TimeoutError(payload)


def main():
    assert state()["closed"] is False
    q = [0.1, 0.4, 0.8, 0.1, -0.15, 0.2]
    midway = q.copy()
    midway[2] = 0.4
    assert request({"action": "joint", "q": midway})["ok"]
    assert request({"action": "joint", "q": q})["ok"]
    np.testing.assert_allclose(state()["q_rad"], q)
    path = tool_model()
    model = mujoco.MjModel.from_xml_path(path)
    kinematics = Kinematics(path, "weld_tcp")
    goal = np.zeros(model.nq)
    goal[:6] = q
    goal[0] += 0.04
    T = kinematics.fk(goal)
    assert request(
        {
            "action": "pose",
            "position": T[:3, 3].tolist(),
            "rpy": Rotation.from_matrix(T[:3, :3]).as_euler("xyz").tolist(),
        }
    )["ok"]
    np.testing.assert_allclose(state()["tcp_position_m"], T[:3, 3], atol=2e-4)
    before = state()["target_rad"]
    assert request({"action": "pose", "position": [10, 10, 10], "rpy": [0, 0, 0]})["ok"] is False
    np.testing.assert_array_equal(state()["target_rad"], before)
    assert request({"action": "mode", "mode": "Gravity compensation"})["ok"]
    assert state()["mode"] == "Gravity compensation"
    assert request({"action": "mode", "mode": "Hold"})["ok"]
    assert request({"action": "step", "path": "assets/demo_joint.step"})["ok"]
    original_rest = state()["rest_rad"]
    assert request({"action": "disable"})["ok"]
    time.sleep(1.5)
    assert state()["closed"]
    assert request({"action": "enable"})["ok"]
    assert not state()["closed"] and state()["mode"] == "Hold"
    np.testing.assert_array_equal(state()["rest_rad"], original_rest)
    assert request({"action": "disable"})["ok"]
    time.sleep(1.5)
    assert state()["closed"]
    Path("evidence/offline-smoke.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": [
                    "six-joint waypoint",
                    "TCP IK waypoint",
                    "unreachable IK rejected without command",
                    "gravity and hold modes",
                    "STEP import",
                    "one-click return and disable",
                    "enable without restart and disable again",
                ],
                "final_state": state(),
            },
            indent=2,
        )
    )
    print("Offline end-to-end checks passed; simulation is disabled.")


if __name__ == "__main__":
    main()
