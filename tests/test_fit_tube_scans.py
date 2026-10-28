"""Synthetic geometry and unchanged-file analysis; no hardware/model/camera."""

import hashlib
import json

import numpy as np
import pytest

from scripts.fit_tube_scans import balance_weights, compare, fit_cylinder, load_scan, main, radial_residual_mm


def cylinder(*, axis=(0.3, 0.4, 0.8660254), frames=12, angles=40, arc_deg=220.0, noise_mm=0.025, seed=3):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    u = np.cross(axis, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(-arc_deg / 2, arc_deg / 2, angles) * np.pi / 180.0
    center = np.array([0.19, -0.08, 0.24])
    pieces = []
    for axial in np.linspace(-0.04, 0.04, frames):
        points = center + axial * axis + 0.0127 * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
        points += rng.normal(0, noise_mm / 1000.0, points.shape)
        pieces.append(points)
    return np.concatenate(pieces), np.repeat(np.arange(frames), angles), axis, center


def axis_error(model, truth):
    return np.degrees(np.arccos(np.clip(abs(np.asarray(model["axis_direction_world"]) @ truth), 0, 1)))


def test_fixed_radius_recovers_arbitrary_tilt_without_mutating_points():
    points, frames, axis, center = cylinder()
    original = points.copy()
    result = fit_cylinder(points, frames)
    assert axis_error(result["model"], axis) < 0.15
    assert result["model"]["radius_mm"] == 12.7
    np.testing.assert_array_equal(points, original)
    assert 0.015 < result["statistics"]["rms_mm"] < 0.04
    fitted = np.asarray(result["model"]["axis_point_world_m"])
    assert np.linalg.norm(np.cross(fitted - center, axis)) * 1000 < 0.04
    assert not result["statistics"]["warnings"]


def test_robust_fit_retains_outliers_in_all_point_statistics():
    points, frames, axis, _ = cylinder(frames=14, noise_mm=0.02)
    rng = np.random.default_rng(9)
    bad = rng.choice(len(points), len(points) // 20, replace=False)
    points[bad] += rng.normal(0, 0.003, (len(bad), 3))
    result = fit_cylinder(points, frames)
    assert axis_error(result["model"], axis) < 0.4
    assert result["statistics"]["rms_mm"] > 0.25
    assert result["statistics"]["inlier_rms_mm"] < 0.08
    assert result["statistics"]["outlier_count"] >= len(bad) * 0.65
    raw = radial_residual_mm(points, result["model"])
    assert result["statistics"]["rms_mm"] == pytest.approx(np.sqrt(np.mean(raw**2)))
    assert result["statistics"]["point_count"] == len(points)


def test_frame_slice_weights_remove_duplicate_point_density_bias():
    points, frames, axis, _ = cylinder(frames=5, angles=24, noise_mm=0.0)
    weights = balance_weights(points * 1000.0, frames, axis)
    duplicated = np.repeat(np.arange(24), 20)
    extra = np.r_[np.arange(len(points)), duplicated]
    order = np.argsort(frames[extra], kind="stable")
    extra = extra[order]
    more = balance_weights(points[extra] * 1000.0, frames[extra], axis)
    for frame in np.unique(frames):
        expected = weights[frames == frame].sum() / weights.sum()
        actual = more[frames[extra] == frame].sum() / more.sum()
        assert actual == pytest.approx(expected)


def test_partial_arc_warns_even_if_fit_residual_looks_small():
    points, frames, _, _ = cylinder(frames=8, angles=24, arc_deg=8.0, noise_mm=0.015)
    result = fit_cylinder(points, frames, free_radius=True)
    assert result["statistics"]["warnings"]
    assert any("partial" in warning.lower() for warning in result["statistics"]["warnings"])
    assert result["statistics"]["rms_mm"] < 0.1


def write_archive(folder, points, frame_ids, *, legacy=False, phases=None):
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps({"scan_protocol": "synthetic"}))
    records = []
    for frame in np.unique(frame_ids):
        world = points[frame_ids == frame]
        all_world = np.r_[world, [[9.0, 9.0, 9.0]]]
        indices = np.arange(len(world), dtype=np.int64)
        name = f"frame-{frame:08d}.npz"
        np.savez(
            folder / name,
            xyz_world_m=world if legacy else all_world,
            xyz_camera_m=all_world,
            selected_indices=indices,
            base_from_tcp=np.eye(4),
            tcp_from_camera=np.eye(4),
        )
        rec = dict(
            status="captured" if legacy else "accepted",
            acquisition_identity=["session", int(frame)],
            ordinal=int(frame),
            cloud=dict(source_id="camera", sequence=int(frame)),
        )
        if legacy:
            rec["raw_snapshot_path"] = str((folder / name).resolve())
        else:
            rec["raw_file"] = name
        if phases is not None and phases[int(frame)] is not None:
            rec["scan_context"] = {"acquisition_phase": phases[int(frame)]}
        records.append(rec)
    (folder / ("cells.jsonl" if legacy else "frames.jsonl")).write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_loader_uses_selected_raw_views_without_changing_source(tmp_path, legacy):
    points, frames, _, _ = cylinder(frames=3, angles=20, noise_mm=0.0)
    folder = tmp_path / "motion-scans" / "synthetic"
    write_archive(folder, points, frames, legacy=legacy)
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}
    scan = load_scan(tmp_path, "synthetic")
    np.testing.assert_array_equal(scan.points_m, points)
    assert scan.excluded_nonfinite == 0 and len(scan.frames) == 3
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}
    assert before == after


def test_cli_exports_json_csv_png_and_states_timing_observability_limit(tmp_path, capsys):
    points, frames, _, _ = cylinder(frames=4, angles=24, noise_mm=0.02)
    write_archive(tmp_path / "motion-scans" / "fast", points, frames)
    write_archive(tmp_path / "motion-scans" / "precision", points, frames)
    out = tmp_path / "comparison"
    assert (
        main(["--runtime", str(tmp_path), "--fast", "fast", "--precision", "precision", "--output", str(out)])
        == 0
    )
    report = json.loads((out / "comparison.json").read_text())
    assert report["datasets"]["fast"]["statistics"]["point_count"] == len(points)
    assert report["relative_comparison"]["axis_angle_deg"] < 1e-5
    assert any("entirely unobservable" in message for message in report["limitations"])
    assert (out / "comparison.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (
        (out / "frames.csv").exists()
        and (out / "axial-slices.csv").exists()
        and (out / "summary.csv").exists()
    )
    assert json.loads(capsys.readouterr().out)["fast"]["rms_mm"] < 0.05


def test_oblique_native_ellipse_becomes_circle_only_after_axis_normal_projection():
    from scripts.fit_tube_scans import native_axis_normal_statistics, native_curve_statistics

    theta = np.linspace(-1.8, 1.8, 120)
    x = 0.0127 * np.cos(theta)
    y = 0.0127 * np.sin(theta)
    points = np.c_[x, y, -0.7 * x - 0.3 * y]  # Oblique planar section of a Z-axis cylinder.
    raw = native_curve_statistics(points)
    projected = native_axis_normal_statistics(points, [0.0, 0.0, 1.0], 12.7)
    assert projected["native_axis_normal_rms_mm"] < 1e-9
    assert raw["native_curve_rms_mm"] > 0.05


def test_axis_parallel_time_shift_is_invisible_to_cylinder_residual():
    points, frames, axis, _ = cylinder(frames=7, angles=28, noise_mm=0.0)
    fitted = fit_cylinder(points, frames)
    shifted = points + axis * 0.006
    np.testing.assert_allclose(
        radial_residual_mm(points, fitted["model"]), radial_residual_mm(shifted, fitted["model"]), atol=1e-7
    )


@pytest.mark.parametrize("main_phase", ["moving_main", "stationary_main"])
def test_labelled_anchors_excluded_before_fit_with_counts_and_unchanged_raw_files(tmp_path, main_phase):
    points, frames, _, _ = cylinder(frames=5, angles=32, noise_mm=0.02)
    points[(frames == 0) | (frames == 4)] += np.array([0.004, 0.0, 0.0])  # Biased anchor positions.
    folder = tmp_path / "motion-scans" / "labelled"
    write_archive(
        folder, points, frames, phases=["start_anchor", main_phase, main_phase, main_phase, "end_anchor"]
    )
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}
    selected = load_scan(tmp_path, "labelled", acquisition_phase=main_phase)
    np.testing.assert_array_equal(selected.points_m, points[(frames > 0) & (frames < 4)])
    choice = selected.metadata["phase_selection"]
    assert choice["mode"] == "labelled_phase_only" and not choice["legacy_unlabelled_fallback"]
    assert choice["excluded_phase_frame_counts"] == {"start_anchor": 1, "end_anchor": 1}
    assert choice["excluded_phase_selected_points"] == 64
    clean = fit_cylinder(selected.points_m, selected.frame_ids)
    mixed = fit_cylinder(points, frames)
    assert clean["statistics"]["rms_mm"] < 0.05
    assert mixed["statistics"]["rms_mm"] > 0.5
    after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.iterdir()}
    assert before == after


def test_mixed_unlabelled_frames_are_excluded_when_main_labels_exist(tmp_path):
    points, frames, _, _ = cylinder(frames=3, angles=24, noise_mm=0.0)
    write_archive(
        tmp_path / "motion-scans" / "mixed", points, frames, phases=[None, "moving_main", "end_anchor"]
    )
    scan = load_scan(tmp_path, "mixed", acquisition_phase="moving_main")
    assert len(scan.points_m) == 24
    assert scan.metadata["phase_selection"]["excluded_phase_frame_counts"] == {
        "unlabelled": 1,
        "end_anchor": 1,
    }


def test_legacy_phase_fallback_is_explicit_and_wrong_labelled_phase_does_not_fallback(tmp_path):
    points, frames, _, _ = cylinder(frames=3, angles=24, noise_mm=0.0)
    write_archive(tmp_path / "motion-scans" / "old", points, frames)
    old = load_scan(tmp_path, "old", acquisition_phase="moving_main")
    assert old.metadata["phase_selection"]["mode"] == "legacy_unlabelled_all_accepted"
    assert old.metadata["phase_selection"]["legacy_unlabelled_fallback"]
    write_archive(tmp_path / "motion-scans" / "wrong", points, frames, phases=["stationary_main"] * 3)
    with pytest.raises(ValueError, match="moving_main"):
        load_scan(tmp_path, "wrong", acquisition_phase="moving_main")


def test_compare_enforces_main_phase_for_unfiltered_programmatic_inputs(tmp_path):
    points, frames, _, _ = cylinder(frames=5, angles=24, noise_mm=0.02)
    points[(frames == 0) | (frames == 4)] += [0.004, 0.0, 0.0]
    scans = {}
    for label, phase in (("fast", "moving_main"), ("precision", "stationary_main")):
        write_archive(
            tmp_path / "motion-scans" / label,
            points,
            frames,
            phases=["start_anchor", phase, phase, phase, "end_anchor"],
        )
        scans[label] = load_scan(tmp_path, label)
    compare(scans, tmp_path / "comparison")
    report = json.loads((tmp_path / "comparison" / "comparison.json").read_text())
    for label in scans:
        result = report["datasets"][label]
        assert result["statistics"]["point_count"] == 72
        assert result["statistics"]["excluded_phase_frames"] == 2
        assert result["phase_selection"]["excluded_phase_selected_points"] == 48
        assert result["statistics"]["rms_mm"] < 0.05


def test_primary_comparison_rejects_in_progress_archive_before_creating_output(tmp_path):
    points, frames, _, _ = cylinder(frames=3, angles=24)
    scans = {}
    for label in ("fast", "precision"):
        folder = tmp_path / "motion-scans" / label
        write_archive(folder, points, frames)
        (folder / "summary.json").write_text(json.dumps({"status": "scanning"}))
        scans[label] = load_scan(tmp_path, label)
    with pytest.raises(ValueError, match="complete archive"):
        compare(scans, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_ordinary_fit_uses_every_point_and_improves_unweighted_objective():
    from scripts.fit_tube_scans import fit_cylinder_ordinary

    points, frames, _, _ = cylinder(frames=8, angles=32)
    points[::10] += [0.003, 0.0, 0.0]
    before = points.copy()
    robust = fit_cylinder(points, frames)
    ordinary = fit_cylinder_ordinary(points, robust["model"])
    assert ordinary["statistics"]["point_count"] == len(points)
    assert ordinary["statistics"]["rms_mm"] < robust["statistics"]["rms_mm"]
    assert ordinary["statistics"]["rms_mm"] > 0.3
    assert ordinary["model"]["radius_mm"] == 12.7
    np.testing.assert_array_equal(points, before)
