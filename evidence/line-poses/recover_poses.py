"""Recover the first recorded line's TCP orientations using only saved joint data."""

import hashlib
import json
from pathlib import Path

import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.teach import LineRecorder
from autoweld.tracking import load_tracks


def main():
    evidence = Path(__file__).resolve().parent
    state = json.loads((evidence / "disabled-status.json").read_text())
    original = state["line_recording"]
    path = Path(original["saved_path"])
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    model = tool_model()
    try:
        planner = MotionPlanner(model)
        recorder = LineRecorder()
        errors = {}
        for label, point in original["points"].items():
            pose = planner.fk(point["joint_rad"])
            error = float(np.linalg.norm(pose[:3, 3] - point["position_m"]))
            if error > 1e-10:
                raise ValueError(f"{label}: saved position does not match this TCP model")
            recorder.record(label, pose, point["joint_rad"], point["captured_at"], point["feedback_timestamp"])
            errors[label] = error
        sidecar = recorder.save_poses(path)
        curves = load_tracks(path)
        assert curves[0].recorded_points is not None
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
        report = dict(step_path=str(path), pose_path=str(sidecar), step_sha256=before_hash,
                      endpoint_reconstruction_error_m=errors, length_mm=curves[0].length*1000,
                      source="FK of original recorded joint_rad, using unchanged modeled mount -Z 150 mm TCP",
                      robot_commands_sent=False, points=recorder.snapshot()["points"])
        (evidence / "recovered-poses.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "points"}, indent=2))
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
