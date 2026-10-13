"""Record full measured TCP poses and save a STEP line with bound pose metadata."""

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation


def pose_file(path):
    return Path(path).with_suffix(".poses.json")


class LineRecorder:
    def __init__(self):
        self.points = {}
        self.saved_path = None

    def record(self, label, pose, joints, timestamp, feedback_timestamp):
        if label not in ("A", "B"):
            raise ValueError("Choose point A or B")
        pose, joints = np.asarray(pose, dtype=float), np.asarray(joints, dtype=float)
        if (pose.shape != (4, 4) or joints.shape != (6,)
                or not np.isfinite(np.r_[pose.ravel(), joints, timestamp, feedback_timestamp]).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-9, rtol=0)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-9, rtol=0)):
            raise ValueError("A recorded point needs a rigid TCP pose, six measured joint angles and finite timestamps")
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.points[label] = dict(position_m=pose[:3, 3].tolist(), quaternion_wxyz=quat.tolist(),
                                  joint_rad=joints.tolist(),
                                  captured_at=float(timestamp), feedback_timestamp=float(feedback_timestamp))
        self.saved_path = None

    @property
    def length(self):
        if len(self.points) != 2:
            return None
        return float(np.linalg.norm(np.array(self.points["B"]["position_m"]) - self.points["A"]["position_m"]))

    def snapshot(self):
        return dict(points=copy.deepcopy(self.points), length_mm=None if self.length is None else self.length*1000,
                    saved_path=self.saved_path, pose_path=str(pose_file(self.saved_path)) if self.saved_path else None,
                    coordinate_frame="robot base", tcp="mount -Z, 150 mm")

    def _check_curve(self, curves):
        if (self.length is None or self.length <= 1e-6 or len(curves) != 1 or curves[0].kind != "line"
                or any(not np.allclose(curves[0].point(i), self.points[k]["position_m"], atol=1e-9, rtol=0)
                       for i, k in enumerate(("A", "B")))):
            raise ValueError("Pose file must match the STEP's directed A-to-B line")

    def _metadata(self, path):
        return dict(format="autoweld.recorded-line", version=1, coordinate_frame="robot base",
                    tcp="mount -Z, 150 mm", position_unit="m", joint_unit="rad",
                    orientation_format="quaternion_wxyz", points=copy.deepcopy(self.points),
                    step_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    @classmethod
    def load(cls, path, curves):
        """Return None for legacy geometry-only STEP; reject mismatched/malformed sidecars."""
        sidecar = pose_file(path)
        if not sidecar.exists() and not sidecar.is_symlink():
            return None
        try:
            data = json.loads(sidecar.read_text())
            expected = dict(format="autoweld.recorded-line", version=1, coordinate_frame="robot base",
                            tcp="mount -Z, 150 mm", position_unit="m", joint_unit="rad",
                            orientation_format="quaternion_wxyz")
            if any(data[k] != v for k, v in expected.items()):
                raise ValueError("unsupported format, units or TCP frame")
            if data["step_sha256"] != hashlib.sha256(Path(path).read_bytes()).hexdigest():
                raise ValueError("STEP contents differ from the saved pose file")
            if set(data["points"]) != {"A", "B"}:
                raise ValueError("both endpoint poses are required")
            recorder = cls()
            for label, point in data["points"].items():
                quat = np.asarray(point["quaternion_wxyz"], dtype=float)
                position = np.asarray(point["position_m"], dtype=float)
                if (quat.shape != (4,) or position.shape != (3,) or not np.isfinite(quat).all()
                        or not np.isclose(np.linalg.norm(quat), 1, atol=1e-9, rtol=0)):
                    raise ValueError("expected a unit quaternion and three position coordinates")
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                pose[:3, 3] = position
                recorder.record(label, pose, point["joint_rad"], point["captured_at"], point["feedback_timestamp"])
            recorder._check_curve(curves)
            recorder.saved_path = str(Path(path).resolve())
            return recorder
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            raise ValueError(f"Cannot load {sidecar.name}: {exc}") from exc

    def save_poses(self, path):
        """Attach recovered poses to an existing matching STEP without changing its geometry."""
        from autoweld.tracking import load_tracks

        path = Path(path).resolve()
        self._check_curve(load_tracks(path))
        sidecar = pose_file(path)
        with tempfile.TemporaryDirectory(prefix=".poses-", dir=path.parent) as work:
            temporary = Path(work) / "poses.json"
            temporary.write_text(json.dumps(self._metadata(path), indent=2, allow_nan=False))
            try:
                os.link(temporary, sidecar)
            except FileExistsError:
                raise ValueError(f"{sidecar.name} already exists; choose a different file name") from None
        self.saved_path = str(path)
        return sidecar

    def save(self, directory, filename):
        from autoweld.tracking import load_tracks

        if self.length is None or self.length <= 1e-6:
            raise ValueError("Record distinct points A and B before saving the line")
        name = filename.strip()
        if not name or name in (".", "..") or any(c in name for c in '/\\') or any(ord(c) < 32 for c in name):
            raise ValueError("Enter a file name, without folders")
        if len(name.encode("utf-8")) > 200:
            raise ValueError("Use a shorter file name (at most 200 bytes)")
        suffix = Path(name).suffix.lower()
        if not suffix:
            name += ".step"
        elif suffix not in (".step", ".stp"):
            raise ValueError("Use a .step or .stp file name")
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        sidecar = pose_file(path)
        if any(p.exists() or p.is_symlink() for p in (path, sidecar)):
            raise ValueError(f"{name} or its pose file already exists; choose a different file name")
        a, b = (np.array(self.points[k]["position_m"]) for k in ("A", "B"))
        edge = cq.Edge.makeLine(cq.Vector(*(a*1000)), cq.Vector(*(b*1000)))
        with tempfile.TemporaryDirectory(prefix=".line-", dir=directory) as work:
            temporary = Path(work) / "line.step"
            cq.exporters.export(edge, str(temporary), exportType="STEP")
            curves = load_tracks(temporary)
            self._check_curve(curves)
            temporary_poses = Path(work) / "poses.json"
            temporary_poses.write_text(json.dumps(self._metadata(temporary), indent=2, allow_nan=False))
            try:
                # Publish poses first: a newly visible STEP always has its companion.
                # Neither file may overwrite another recording, including concurrent saves.
                os.link(temporary_poses, sidecar)
                try:
                    os.link(temporary, path)
                except Exception:
                    if os.path.samefile(temporary_poses, sidecar):
                        sidecar.unlink()
                    raise
            except FileExistsError:
                raise ValueError(f"{name} or its pose file already exists; choose a different file name") from None
        self.saved_path = str(path)
        curves[0].recorded_points = copy.deepcopy(self.points)
        return path, curves
