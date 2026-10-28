from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.motion import MotionLimits
from autoweld.scan_plan_probe import (
    build_scan_plan,
    compact_plan_summary,
    private_planner,
    private_self_collision,
)


def fake_planner():
    model = SimpleNamespace(geom_type=[], geom_bodyid=[], body_parentid=[])
    data = SimpleNamespace(ncon=0, contact=[])

    def ik(target, site, *, init_q, **kwargs):
        q = init_q.copy()
        q[:3] = target[:3, 3]
        return True, q

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose

    return SimpleNamespace(model=model, data=data, kin=SimpleNamespace(ik=ik), site_name='weld_tcp',
                           limits=MotionLimits(), fk=fk, jacobian=lambda q: np.diag([1, 1, 1, 0, 0, 0]),
                           _configuration=lambda q: q[:6].copy())


def parameters(**overrides):
    values = dict(baseline_pose=np.eye(4), tcp_from_camera=np.eye(4), seed_q=np.zeros(6),
                  bounds=np.tile([-3., 3.], (6, 1)), optical_roll_deg=0., width_mm=20., height_mm=1.,
                  depth_offset_m=-.024, model_fingerprint='same-installed-model')
    values.update(overrides)
    return values


def test_probe_builds_same_validated_raster_without_controller_or_devices():
    planner = fake_planner()
    plan = build_scan_plan(planner, **parameters())
    summary = compact_plan_summary(plan)
    assert summary['read_only'] and summary['validated']
    assert summary['model_fingerprint'] == 'same-installed-model'
    assert summary['validation_proof']['collision_checked']
    assert summary['validation_proof']['command_constraints_checked']
    assert summary['diagnostic_constant_depth']
    assert summary['entry_pose'][2][3] == pytest.approx(-.024)
    assert summary['feed_mm_s'] == 20.
    assert 'path' not in summary and 'positions' not in summary
    np.testing.assert_array_equal(plan.path.positions[:, 6], 0)
    assert planner.model is not None


def test_private_planner_shares_existing_model_and_creates_only_private_data(monkeypatch):
    original_model = object()
    configuration = SimpleNamespace(model=original_model)
    monkeypatch.setattr('mink.Configuration', lambda model: configuration if model is original_model else None)
    seen = []

    def make_planner(path, *, site, kin):
        seen.append((path, site, kin))
        return SimpleNamespace(model=kin._configuration.model, data=object(), kin=kin)

    monkeypatch.setattr('autoweld.motion.MotionPlanner', make_planner)
    original = SimpleNamespace(model=original_model, data=object(), site_name='weld_tcp', limits=MotionLimits())
    worker = private_planner(original)
    assert worker.model is original.model
    assert worker.data is not original.data
    assert worker.kin._configuration is configuration
    assert worker.limits is original.limits
    assert seen[0][0] is None


def test_private_collision_matches_existing_plane_adjacent_and_penetration_rules():
    import mujoco

    planner = fake_planner()
    model, data = planner.model, planner.data
    model.geom_type = np.array([mujoco.mjtGeom.mjGEOM_PLANE, 2, 2, 2])
    model.geom_bodyid = np.array([0, 1, 2, 3])
    model.body_parentid = np.array([0, 0, 1, 0])
    for first, second, depth, expected in ((0, 1, -.01, False), (1, 2, -.01, False),
                                           (1, 3, -.0005, False), (1, 3, -.002, True)):
        data.ncon = 1
        data.contact = [SimpleNamespace(geom1=first, geom2=second, dist=depth)]
        assert private_self_collision(planner, np.zeros(7)) is expected


@pytest.mark.parametrize('changes', [dict(feed_mm_s=31), dict(depth_offset_m=.061),
                                     dict(width_mm=101), dict(seed_q=np.ones(6) * 4)])
def test_probe_rejects_outside_installed_geometry_limits_before_any_ik(changes):
    planner = fake_planner()
    planner.kin.ik = lambda *args, **kwargs: pytest.fail('Invalid request must not start IK')
    with pytest.raises(ValueError):
        build_scan_plan(planner, **parameters(**changes))


def test_failed_probe_has_compact_nonrecursive_progress_diagnostic():
    import json

    with pytest.raises(ValueError) as caught:
        build_scan_plan(fake_planner(), **parameters(max_preflight_s=.000001))
    diagnostic = caught.value.diagnostic
    assert diagnostic['code'] == 'preflight_budget'
    assert 'ik_index' in diagnostic['preflight_progress']
    assert diagnostic['preflight_progress']['phase'] == 'error'
    assert diagnostic['reference_duration_s'] > 0
    assert len(json.dumps(diagnostic)) < 2500


def test_registration_budget_matches_registrar_and_point_lever_math():
    from autoweld.scan_plan_probe import registration_budget

    jacobian = np.zeros((6, 6))
    jacobian[0, 0] = 1.
    jacobian[5, 1] = 1.
    planner = SimpleNamespace(fk=lambda q: np.eye(4), jacobian=lambda q: jacobian)
    path = SimpleNamespace(duration=2., sample=lambda t: (np.zeros(7),))
    report = registration_budget(planner, path, np.eye(4))
    half = 25. / 65535 / 2
    lever = np.sqrt(.140**2 + 2 * .025**2)
    for row in report['samples']:
        assert row['registrar_quantization_bound_m'] == pytest.approx((1 + lever) * half)
        assert row['point_specific_center_bound_m'] == pytest.approx(half)
        assert row['point_specific_max_bound_m'] == pytest.approx((1 + np.sqrt(2) * .025) * half)
        assert row['registrar_quantization_budget_passed']
    assert [row['phase'] for row in report['samples']] == ['entry', 'middle', 'end']
    assert report['encoder_half_quantization_rad'] == pytest.approx(.00019073777)
    assert report['diagnostic_only']


def test_registration_diagnostic_can_expose_stationary_quantization_budget_failure():
    from autoweld.scan_plan_probe import registration_budget

    planner = SimpleNamespace(fk=lambda q: np.eye(4), jacobian=lambda q: np.diag([1., 1., 1., 0., 0., 0.]))
    path = SimpleNamespace(duration=1., sample=lambda t: (np.zeros(7),))
    report = registration_budget(planner, path, np.eye(4))
    assert report['worst_registrar_quantization_bound_m'] > .0005
    assert all(not row['registrar_quantization_budget_passed'] for row in report['samples'])


def test_collision_fk_is_reused_for_same_point_residual_including_after_retime():
    from dataclasses import replace

    planner = fake_planner()
    planner.limits = replace(planner.limits, joint_velocity=.010)
    forwards = []
    original = planner.fk

    def fk(q):
        forwards.append(q.copy())
        return original(q)

    planner.fk = fk
    plan = build_scan_plan(planner, **parameters())
    assert plan.report['retimes'] == 1
    # Every unique collision sample runs forward once. Residual reuses that exact
    # pose; entry report and three quantization diagnostics add four forwards.
    assert len(forwards) == plan.report['ik_knots'] + plan.report['validation_samples'] + 4


def test_y_sweep_compensation_preserves_actual_camera_rotation_and_original_roi_depth_queries():
    planner = fake_planner()
    baseline, handeye = np.eye(4), np.eye(4)
    baseline[:3, 3] = [.2, .1, .3]
    handeye[:3, 3] = [.02, 0, .01]
    queried = []
    shift = [.005, -.004]

    def depth(x, y):
        queried.append((x, y))
        return -.020 + .1*x + .2*y

    plan = build_scan_plan(planner, **parameters(
        baseline_pose=baseline, tcp_from_camera=handeye, width_mm=20., height_mm=30.,
        lane_spacing_mm=10., camera_xy_shift_m=shift, sweep_axis='y', depth=depth))
    scope = plan.report['scan_scope']
    assert plan.report['rows'] == 3
    assert plan.report['coverage_cells'] == 21 * 31
    assert scope['roi_width_m'] == .020 and scope['roi_height_m'] == .030
    np.testing.assert_allclose(scope['original_roi_baseline_optical'], baseline @ handeye)
    np.testing.assert_allclose(np.array(scope['baseline_optical'])[:3, 3], [.225, .096, .310])
    np.testing.assert_allclose(np.array(scope['baseline_optical'])[:3, :3], [[0,-1,0],[1,0,0],[0,0,1]], atol=1e-12)
    np.testing.assert_allclose(scope['fixed_optical_rotation'], np.eye(3))
    np.testing.assert_allclose(queried[0], [.015, -.019], atol=1e-12)
    # Smooth turns may extend<1mm along the fast axis; depth still receives
    # actual compensated camera coordinates there instead of clipped ROI bins.
    assert np.min(queried, axis=0)[0] == pytest.approx(-.005)
    assert np.max(queried, axis=0)[0] == pytest.approx(.015)
    assert -.020 < np.min(queried, axis=0)[1] <= -.019
    assert .011 <= np.max(queried, axis=0)[1] < .012
    expected_z = .3 + depth(.015, -.019)
    np.testing.assert_allclose(np.array(plan.report['entry_pose'])[:3, 3], [.215, .081, expected_z], atol=1e-10)


def test_x_sweep_shift_keeps_fixed_camera_r_and_separate_spatial_coverage():
    plan = build_scan_plan(fake_planner(), **parameters(
        width_mm=30., height_mm=20., lane_spacing_mm=10., camera_xy_shift_m=[.005, -.004]))
    assert plan.report['rows'] == 3
    assert plan.report['coverage_cells'] == 31 * 21
    np.testing.assert_allclose(plan.report['entry_q'][:2], [-.010, -.014], atol=1e-12)
    np.testing.assert_allclose(plan.report['scan_scope']['fixed_optical_rotation'], np.eye(3))
    assert plan.report['scan_scope']['camera_xy_shift_m'] == [.005, -.004]


@pytest.mark.parametrize('changes', [dict(camera_xy_shift_m=[0., np.nan]), dict(sweep_axis='diagonal'),
                                     dict(lane_spacing_mm=21.)])
def test_invalid_line_compensation_rejected_before_any_ik(changes):
    planner = fake_planner()
    planner.kin.ik = lambda *args, **kwargs: pytest.fail('Invalid footprint settings must not start IK')
    with pytest.raises(ValueError):
        build_scan_plan(planner, **parameters(**changes))
