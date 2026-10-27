"""Measured-point capture must never command motion; STEP round trips keep base units and direction."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_control import FakeRobot

from autoweld.control import Controller
from autoweld.legacy_app import App, tool_model
from autoweld.motion import MotionPlanner
from autoweld.teach import LineRecorder, pose_file
from autoweld.tracking import TrackCurve, load_tracks, plan_tracking, tracking_sample


def pose(position, angles=(0, 0, 0)):
    T = np.eye(4)
    T[:3, 3] = position
    T[:3, :3] = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
    return T


@pytest.fixture
def recorder():
    r = LineRecorder()
    r.record("A", pose([.23, -.1, .32], (13, -8, 175)), np.zeros(6), 1, 1)
    r.record("B", pose([.21, -.04, .29], (13, -8, -175)), np.ones(6)*.01, 2, 2)
    return r


def test_named_step_preserves_measured_endpoints_units_and_direction(tmp_path, recorder):
    path, curves = recorder.save(tmp_path, "Weld seam 1")
    assert path.name == "Weld seam 1.step"
    for curve in (curves[0], load_tracks(path)[0]):
        np.testing.assert_allclose(curve.point(0), [.23, -.1, .32], atol=1e-10)
        np.testing.assert_allclose(curve.point(1), [.21, -.04, .29], atol=1e-10)
        assert curve.length == pytest.approx(.07)
    original = path.read_bytes()
    metadata = pose_file(path).read_bytes()
    restored = LineRecorder.load(path, curves)
    for label in ("A", "B"):
        for key in recorder.points[label]:
            np.testing.assert_allclose(restored.points[label][key], recorder.points[label][key], atol=1e-15)
    for s, label in ((0, "A"), (1, "B")):
        np.testing.assert_allclose(load_tracks(path)[0].pose(s, np.eye(3), "Recorded poses")[:3, :3],
            Rotation.from_quat(np.array(recorder.points[label]["quaternion_wxyz"])[[1, 2, 3, 0]]).as_matrix(), atol=1e-15)
    with pytest.raises(ValueError, match="already exists"):
        recorder.save(tmp_path, path.name)
    assert path.read_bytes() == original
    assert pose_file(path).read_bytes() == metadata


@pytest.mark.parametrize("name", ["", "..", "../line.step", "/tmp/line.step", "sub/line.step", "sub\\line.step", "line.obj", "a\x00.step"])
def test_filename_cannot_escape_directory_or_use_wrong_format(tmp_path, recorder, name):
    with pytest.raises(ValueError):
        recorder.save(tmp_path, name)
    assert not list(tmp_path.iterdir())


def test_missing_or_coincident_points_cannot_export(tmp_path):
    r = LineRecorder()
    with pytest.raises(ValueError, match="distinct"):
        r.save(tmp_path, "line")
    for label in ("A", "B"):
        r.record(label, pose([.1, .2, .3]), np.zeros(6), 1, 1)
    with pytest.raises(ValueError, match="distinct"):
        r.save(tmp_path, "line")
    assert not list(tmp_path.iterdir())


@pytest.fixture
def app(tmp_path):
    model = tool_model()
    a = object.__new__(App)
    a.runtime = tmp_path
    robot = FakeRobot()
    a.ctrl = Controller(robot, np.tile([-3, 3], (6, 1)))
    a.ctrl.motion_planner = MotionPlanner(model)
    a.ctrl.mode = 'Gravity compensation'
    a.line_recorder = LineRecorder()
    a.line_markers = {}
    a.tracking_pose_markers = {}
    a.tracking_monitor = None
    a.tracking_preview = object()
    a.tracking_result = {"old": True}
    a.drawn = {}

    def draw(name, *args, **kwargs):
        a.drawn[name] = kwargs
        return SimpleNamespace(remove=lambda: a.drawn.pop(name))

    a.server = SimpleNamespace(scene=SimpleNamespace(add_icosphere=draw, add_label=draw, add_line_segments=draw,
                                                    add_frame=draw))
    for name in ("line_a_button", "line_b_button", "line_save_button", "line_clear_button",
                 "line_points_md", "track_file", "track_select", "track_feed", "tracking_md",
                 "track_ramp", "track_orientation", "line_name"):
        setattr(a, name, SimpleNamespace(value=None))
    yield a


def test_record_save_load_is_read_only_and_uses_actual_pose_not_target(app):
    original_target = app.ctrl.target.copy()
    expected = []
    for label, offset in (("A", .04), ("B", -.03)):
        # Simulate the user hand-guiding the arm; no command is issued.
        app.ctrl.robot.q[0] += offset
        expected.append(app.ctrl.motion_planner.fk(app.ctrl.robot.q).copy())
        app.dispatch(dict(action="line_record", point=label))
    assert not app.line_save_button.disabled
    app.dispatch(dict(action="line_save", name="my seam.step"))
    assert app.ctrl.mode == "Gravity compensation"
    assert not app.ctrl.robot.commands
    np.testing.assert_array_equal(app.ctrl.target, original_target)
    assert app.tracking_preview is None and app.tracking_result is None
    assert app.track_feed.value == 20
    assert app.track_ramp.value == "Inside edge"
    assert app.track_orientation.value == "Recorded poses"
    for i, label in enumerate(("A", "B")):
        np.testing.assert_allclose(app.tracks[0].pose(i, np.eye(3), "Recorded poses"), expected[i], atol=1e-9)
        np.testing.assert_allclose(app.drawn[f"/tracking/poses/{label}"]["wxyz"],
                                   app.line_recorder.points[label]["quaternion_wxyz"])
    assert Path(app.track_file.value).exists()
    app.dispatch(dict(action="line_clear"))
    assert not app.line_recorder.points
    assert app.line_save_button.disabled
    assert all(name.startswith("/tracking/") for name in app.drawn)  # Loaded poses survive clearing the draft.
    assert "/tracking/poses/A" in app.drawn
    app.dispatch(dict(action="track_load", path=app.track_file.value))
    assert set(app.line_recorder.points) == {"A", "B"}
    assert app.track_orientation.value == "Recorded poses"
    assert not app.ctrl.robot.commands
    app.dispatch(dict(action="track_load", path="assets/stage3/lines.step"))
    assert app.track_orientation.value == "Unconstrained"
    assert "Recorded poses" not in app.track_orientation.options
    assert "/tracking/poses/A" not in app.drawn


@pytest.mark.parametrize("state", ["disabled", "wrong_mode", "moving", "fault", "stale"])
def test_invalid_capture_does_not_store_cached_pose_or_send_command(app, state):
    if state == "disabled":
        app.ctrl.closed = True
    elif state == "wrong_mode":
        app.ctrl.mode = "Hold"
    elif state == "moving":
        app.ctrl.motion_plan = object()
    elif state == "fault":
        app.ctrl.motion_fault = lambda: {"reason": "latched"}
    else:
        original = app.ctrl.robot.get_observations
        app.ctrl.robot.get_observations = lambda: dict(**original(), feedback_timestamp=time.time()-1)
    with pytest.raises(RuntimeError):
        app.dispatch(dict(action="line_record", point="A"))
    assert not app.line_recorder.points
    assert not app.ctrl.robot.commands


@pytest.mark.parametrize("fault", ["hash", "quaternion", "position", "units", "missing", "invalid_json"])
def test_corrupt_metadata_is_rejected_instead_of_discarding_poses(tmp_path, recorder, fault):
    path, _ = recorder.save(tmp_path, "seam")
    sidecar = pose_file(path)
    data = json.loads(sidecar.read_text())
    if fault == "hash":
        data["step_sha256"] = "wrong"
    elif fault == "quaternion":
        data["points"]["A"]["quaternion_wxyz"] = [0, 0, 0, 0]
    elif fault == "position":
        data["points"]["A"]["position_m"][0] += .001
    elif fault == "units":
        data["position_unit"] = "mm"
    elif fault == "missing":
        del data["points"]["B"]["quaternion_wxyz"]
    sidecar.write_text("{" if fault == "invalid_json" else json.dumps(data))
    with pytest.raises(ValueError, match="Cannot load"):
        load_tracks(path)


def test_orphan_pose_file_cannot_be_overwritten(tmp_path, recorder):
    sidecar = tmp_path / "seam.poses.json"
    sidecar.write_text("existing poses")
    with pytest.raises(ValueError, match="already exists"):
        recorder.save(tmp_path, "seam")
    assert not (tmp_path / "seam.step").exists()
    assert sidecar.read_text() == "existing poses"


def test_legacy_geometry_load_and_recover_pose_file(tmp_path, recorder):
    path, _ = recorder.save(tmp_path, "seam")
    original = path.read_bytes()
    pose_file(path).unlink()
    assert load_tracks(path)[0].recorded_points is None
    recorder.save_poses(path)
    assert path.read_bytes() == original
    assert load_tracks(path)[0].recorded_points is not None
    with pytest.raises(ValueError, match="already exists"):
        recorder.save_poses(path)


def test_recorded_rotation_takes_shortest_arc_without_ramp_discontinuities(tmp_path, recorder):
    _, curves = recorder.save(tmp_path, "seam")
    curve = curves[0]
    rotations = [Rotation.from_matrix(curve.pose(s, np.eye(3), "Recorded poses")[:3, :3])
                 for s in (-.1, 0, .5, 1, 1.1)]
    for a, b, span in ((0, 1, 1), (1, 2, 5), (2, 3, 5), (3, 4, 1)):
        assert np.rad2deg((rotations[a].inv() * rotations[b]).magnitude()) == pytest.approx(span)
    # Flip quaternion sign: same physical rotation and same interpolation.
    curve.recorded_points["B"]["quaternion_wxyz"] = (-np.array(curve.recorded_points["B"]["quaternion_wxyz"])).tolist()
    np.testing.assert_allclose(curve.pose(.5, np.eye(3), "Recorded poses")[:3, :3], rotations[2].as_matrix(), atol=1e-15)


def test_recorded_pose_reference_obeys_both_endpoints_and_motion_limits(app):
    planner = app.ctrl.motion_planner
    seed = np.array(json.loads(Path("assets/stage3/manifest.json").read_text())[0]["seed_q"])
    goal = seed.copy()
    goal[:3] += [.01, .015, .02]
    recorder = LineRecorder()
    for label, q in (("A", seed), ("B", goal)):
        recorder.record(label, planner.fk(q), q[:6], 1, 1)
    path, _ = recorder.save(app.runtime, "oriented")
    curve = load_tracks(path)[0]
    plan = plan_tracking(planner, seed, curve, .005, "Recorded poses", ramp_placement="Inside edge")
    for t, q in ((0, seed), (plan.motion.path.duration, goal)):
        np.testing.assert_allclose(plan.expected(t), planner.fk(q), atol=1e-9)
    for t in np.linspace(0, plan.motion.path.duration, 51):
        q, v, _, _ = plan.motion.path.sample(t)
        row = tracking_sample(plan, planner, q, v, t)
        assert row["tracking_error_mm"] < .2
        assert row["orientation_error_deg"] < np.rad2deg(.0002)
    for name, peak in plan.motion.report["planned_peaks"].items():
        assert peak <= plan.motion.report["limits"][name] + 1e-6
    assert plan.motion.report["orientation_policy"] == "pose constrained"
    with pytest.raises(ValueError, match="both endpoint poses"):
        plan_tracking(planner, seed, TrackCurve("line", curve.start, curve.direction, curve.length),
                      orientation="Recorded poses")
