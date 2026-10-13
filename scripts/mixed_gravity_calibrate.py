"""Collect one authorized fixed-gravity round; fitting is done while disabled."""

import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro

from scripts import mixed_wrist_trial as io
from scripts.kp_path_repeat import check_gains

DEST = Path("evidence/mixed-gravity")
BASE = np.array([1.0, 1.04, 1.31, 1.32, 1.0, 1.0])


def factors_to(target):
    """Only J4/J5 may change, each by at most 20% per idle application."""
    target = np.array(target, dtype=float)
    if target.shape != (6,) or not np.isfinite(target).all() or np.any(target <= 0):
        raise ValueError("Invalid gravity factors")
    if not np.array_equal(target[[0, 1, 2, 5]], BASE[[0, 1, 2, 5]]):
        raise ValueError("Only J4/J5 may be tuned")
    while True:
        current = np.array(io.healthy()["gravity_factors"])
        if np.allclose(current, target, atol=1e-12, rtol=0):
            return
        step = np.clip(target, 0.8 * current, 1.2 * current)
        io.send(dict(action="gravity_feedforward", factors=step.tolist(), model="linear"))
        if not io.state()["closed"]:
            io.observe(2, "MIXED_J45")


def motion_done():
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        s = io.healthy("MIXED_J45")
        if not s["motion_active"]:
            return
        time.sleep(0.05)
    raise TimeoutError("SDK waypoint did not complete")


def main(label: str, j4: float = 1.32, j5: float = 1.0, poses: tuple[str, ...] = ()):
    if not 0.8 <= j4 <= 1.8 or not 0.5 <= j5 <= 2.0:
        raise ValueError("Local gravity calibration search bounds exceeded")
    io.DEST = DEST
    name = f"calibration-{label}-{time.time_ns()}"
    spec = json.loads((DEST / "calibration-preflight.json").read_text())
    factors = BASE.copy()
    factors[3:5] = [j4, j5]
    record = dict(name=name, requested_factors=factors.tolist(), holds=[])
    offset = io.TELEMETRY.stat().st_size
    with (io.ROOT / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            assert io.healthy()["closed"], "Start disabled at captured rest"
            io.send(dict(action="controller", controller="MIXED_J45"))
            factors_to(BASE)
            io.send(dict(action="position_speed_limit", limit_rad_s=0.15))
            record["enabled"] = io.send(dict(action="enable"))
            check_gains(record["enabled"])
            assert record["enabled"]["gravity"]["active_arm_joints"] == [4, 5]
            io.observe(3, "MIXED_J45")
            io.send(dict(action="rest"))
            motion_done()
            io.observe(3, "MIXED_J45")
            # Baseline gravity is retained at supported rest. No rest torque or
            # position sample enters the free-space fit.
            io.send(dict(action="joint", q=spec["seed"][:6]))
            motion_done()
            io.observe(3, "MIXED_J45")
            factors_to(factors)
            for item in spec["sequence"]:
                if poses and item["pose"] not in poses:
                    continue
                io.send(dict(action="joint", q=item["q"][:6]))
                motion_done()
                io.observe(2.5, "MIXED_J45")
                if item["kind"] == "measure":
                    begin = time.time()
                    io.observe(3.0, "MIXED_J45")
                    record["holds"].append(
                        dict(
                            pose=item["pose"],
                            side=item["side"],
                            target=item["q"][:6],
                            begin=begin,
                            end=time.time(),
                            status=io.healthy("MIXED_J45"),
                        )
                    )
                    print(
                        "MEASURED",
                        label,
                        item["pose"],
                        item["side"],
                        "error deg",
                        record["holds"][-1]["status"]["error_deg"][3:5],
                        flush=True,
                    )
            # Restore the already validated supported-rest profile before return.
            # This is contact handling, not fitting free-space gravity at rest.
            factors_to(BASE)
            io.disable(name)
            record["final"] = io.state()
        except Exception as exc:
            record["error"] = str(exc)
            record["error_status"] = io.state()
            print("STOPPED", str(exc), flush=True)
            s = io.state()
            if not s["closed"] and not s["fault"] and not s["motion_fault"]:
                if s["motion_active"]:
                    io.send(dict(action="mode", mode="Hold"))
                factors_to(BASE)
                io.disable(name)
            raise
        finally:
            io.archive_tail(offset, DEST / (name + "-control.jsonl"))
            (DEST / (name + ".json")).write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
