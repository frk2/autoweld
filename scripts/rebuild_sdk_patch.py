"""Rebuild the SDK extension patch and verify it against the pinned pristine source."""

import io
import subprocess
import tarfile
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sdk = root / "vendor/i2rt"
revision = (root / "SDK_REVISION").read_text().strip()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=sdk, text=True).strip() == revision
tracked = ["i2rt/robots/motor_chain_robot.py", "i2rt/motor_drivers/dm_driver.py", "i2rt/motor_drivers/utils.py", "i2rt/robots/kinematics.py"]
tracked.append("i2rt/robots/get_robot.py")
tracked.append("i2rt/robots/sim_robot.py")
tracked.extend(["i2rt/utils/mujoco_utils.py", "i2rt/motor_drivers/can_interface.py",
                "i2rt/motor_config_tool/dm_motor_registers.py", "i2rt/motor_config_tool/dm_motor_registers.md",
                "i2rt/motor_config_tool/set_timeout.py"])
added = [
    f"i2rt/robots/{name}.py"
    for name in [
        "integral_control",
        "oscillation_guard",
        "gravity_feedforward",
        "friction_feedforward",
        "waypoint_motion",
        "encoder_teaching",
        "disabled_recovery",
        "gripper_motion",
        "arm_mode_switch",
        "payload_gravity",
    ]
]
added.append("i2rt/motor_drivers/mit_command.py")
added.extend(["i2rt/motor_drivers/position_speed.py", "i2rt/robots/position_speed_robot.py"])
patch = subprocess.check_output(["git", "diff", "--", *tracked], cwd=sdk)
for path in added:
    result = subprocess.run(
        ["git", "diff", "--no-index", "--", "/dev/null", path], cwd=sdk, capture_output=True
    )
    assert result.returncode == 1, result.stderr
    patch += result.stdout
with tempfile.TemporaryDirectory(prefix="autoweld-patch-") as work:
    archive = subprocess.check_output(["git", "archive", revision], cwd=sdk)
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        source.extractall(work, filter="data")
    subprocess.run(["git", "apply", "--check", "-"], cwd=work, input=patch, check=True)
    subprocess.run(["git", "apply", "-"], cwd=work, input=patch, check=True)
    for path in tracked + added:
        assert (Path(work) / path).read_bytes() == (sdk / path).read_bytes(), path
(root / "patches/i2rt-integral.patch").write_bytes(patch)
print("Patch applies to pinned source and reproduces all modified/added SDK files exactly.")
