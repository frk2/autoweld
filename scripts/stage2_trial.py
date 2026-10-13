"""One small Stage 2 trial through the app inbox; SDK remains the sole robot owner."""

import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from scripts.tune_pid import request


def main(name: str, delta: tuple[float, ...], kind: str = "joint", duration: float = 4.0,
         evidence_dir: str = "evidence/stage2"):
    root = Path("runtime/live")
    evidence = Path(evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    with (root / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def state():
            s = json.loads((root / "status.json").read_text())
            if time.time() - s["timestamp"] > 1:
                raise RuntimeError("App status stale; inspect before further commands")
            return s

        before = state()
        if (
            before["closed"]
            or before["fault"]
            or before.get("motion_active")
            or not before["oscillation_guard_active"]
            or before["friction_enabled"]
        ):
            raise RuntimeError("Need idle, healthy Stage 2 controller, friction off")
        target = np.array(before["rest_rad"]) + np.array(delta)
        if len(delta) != 6 or not np.isfinite(delta).all() or max(abs(np.array(delta))) >= 0.49:
            raise ValueError("Expected six small finite offsets from rest")
        for reference in (before["q_rad"], before["target_rad"]):
            if max(abs(target - reference)) >= 0.49:
                raise ValueError("Test step too large")
        payload = {"action": "joint", "q": target.tolist()}
        if kind == "pose":
            model = tool_model()
            try:
                T = MotionPlanner(model).fk(np.r_[target, 0])
            finally:
                Path(model).unlink()
            payload = {
                "action": "pose",
                "position": T[:3, 3].tolist(),
                "rpy": Rotation.from_matrix(T[:3, :3]).as_euler("xyz").tolist(),
            }
        elif kind != "joint":
            raise ValueError("kind must be joint or pose")
        log = root / "control-telemetry.jsonl"
        offset = log.stat().st_size
        start = time.time()
        metadata = {"before": before, "payload": payload, "requested_at": start}
        (evidence / f"{name}.json").write_text(json.dumps(metadata, indent=2))
        request(root, payload)
        accepted_at = time.time()
        # Planning runs before motion is installed, and can span a status publication.
        # Require one fresh post-ack status; do not mistake the pre-plan status for feedback.
        fresh_deadline = time.monotonic() + 1
        while True:
            published = json.loads((root / "status.json").read_text())
            if published["timestamp"] >= start and time.time() - published["timestamp"] < 1:
                break
            if time.monotonic() > fresh_deadline:
                raise RuntimeError("No fresh status after waypoint acceptance")
            time.sleep(0.02)
        duration = max(duration, published["motion_plan"]["duration_s"]+5)
        deadline = time.monotonic() + duration
        try:
            while time.monotonic() < deadline:
                s = state()
                if s["fault"]:
                    raise RuntimeError(s["fault"])
                time.sleep(0.05)
        finally:
            with log.open() as source:
                source.seek(offset)
                rows = [json.loads(line) for line in source if line.endswith("\n")]
            (evidence / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = {
                "before": before,
                "after": json.loads((root / "status.json").read_text()),
                "payload": payload,
                "requested_at": start,
                "acceptance_latency_s": accepted_at-start,
            }
            (evidence / f"{name}.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "name": name,
                    "samples": len(rows),
                    "plan": result["after"]["motion_plan"],
                    "error_deg": result["after"]["error_deg"],
                    "fault": result["after"]["fault"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    tyro.cli(main)
