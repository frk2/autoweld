"""Read-only IK rejection diagnostics with known residuals and no hardware."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from autoweld.diagnostics import MotionRejection, solve_pose


@pytest.mark.parametrize('orientation', [False, True])
def test_failed_ik_identifies_failed_tolerance_and_units(orientation):
    target = np.eye(4)
    if orientation:
        target[:3, :3] = Rotation.from_euler('x', 0.1).as_matrix()
    else:
        target[0, 3] = .012
    kin = SimpleNamespace(ik=lambda *a, **kw: (False, np.zeros(8)), fk=lambda *a: np.eye(4))
    with pytest.raises(MotionRejection) as caught:
        solve_pose(kin, target, 'tcp', np.zeros(8), stage='endpoint_ik')
    d = caught.value.diagnostic
    assert d['stage'] == 'endpoint_ik' and d['code'] == 'ik_nonconvergence'
    assert d['iterations'] == 300
    assert d['checks']['position']['passed'] is orientation
    assert d['checks']['orientation']['passed'] is not orientation
    assert d['checks']['orientation']['error'] == pytest.approx(.1 if orientation else 0)
    assert d['checks']['position']['error'] == pytest.approx(0 if orientation else .012)
    assert 'does not prove' in str(caught.value)


def test_solver_exception_is_not_misreported_as_nonconvergence():
    def broken(*a, **kw):
        raise RuntimeError('QP failed')
    with pytest.raises(MotionRejection) as caught:
        solve_pose(SimpleNamespace(ik=broken), np.eye(4), 'tcp', np.zeros(8),
                   stage='path_ik', path_fraction=.25)
    assert caught.value.diagnostic['code'] == 'ik_solver_error'
    assert caught.value.diagnostic['path_fraction'] == .25
    assert 'QP failed' in str(caught.value)


def test_near_failure_continues_from_owned_candidate_with_identical_strict_options():
    calls = []
    target = np.eye(4)
    target[0, 3] = .01
    first = np.zeros(8)
    first[0] = .009903  # The live failure's97micrometre translation residual.
    solved = first.copy()
    solved[0] = .01
    def ik(pose, site, *, init_q, **kwargs):
        calls.append((init_q.copy(), kwargs))
        if len(calls) == 1:
            return False, first
        np.testing.assert_array_equal(init_q, first)
        init_q[:] = 99  # SDK mutation cannot overwrite a previous attempt/seed.
        return True, solved
    def fk(q, site):
        pose = np.eye(4)
        pose[0, 3] = q[0]
        return pose
    seed = np.zeros(8)
    cost = np.arange(8) * .01
    result = solve_pose(SimpleNamespace(ik=ik, fk=fk), target, 'tcp', seed,
                        stage='path_ik', path_fraction=.671, joint_motion_cost=cost)
    np.testing.assert_array_equal(result, solved)
    np.testing.assert_array_equal(seed, 0.)
    np.testing.assert_array_equal(first[0], .009903)
    assert [c[1]['max_iters'] for c in calls] == [300, 100]
    for _, options in calls:
        assert options['pos_threshold'] == options['ori_threshold'] == 1e-8
        np.testing.assert_array_equal(options['joint_motion_cost'], cost)
        assert 'damping' not in options


def test_continuation_remains_bounded_and_reports_every_failed_residual(monkeypatch):
    import autoweld.diagnostics as diagnostics
    monkeypatch.setattr(diagnostics.time, 'monotonic', lambda: 1.)
    calls = []
    def ik(*args, **options):
        calls.append(options)
        return False, np.zeros(8)
    target = np.eye(4)
    target[0, 3] = .000097
    with pytest.raises(MotionRejection) as caught:
        solve_pose(SimpleNamespace(ik=ik, fk=lambda *a: np.eye(4)), target, 'tcp', np.zeros(8), stage='path_ik')
    detail = caught.value.diagnostic
    assert detail['iterations'] == sum(c['max_iters'] for c in calls) == 900
    assert detail['attempt_count'] == len(detail['residual_history']) == 7
    assert all(row['position_error_m'] == pytest.approx(.000097) for row in detail['residual_history'])
    assert detail['checks']['position']['limit'] == 1e-8
    assert detail['checks']['position']['passed'] is False


def test_solver_wall_budget_prevents_additional_continuation(monkeypatch):
    import autoweld.diagnostics as diagnostics
    stamps = iter([0., .301, .302])
    monkeypatch.setattr(diagnostics.time, 'monotonic', lambda: next(stamps))
    calls = []
    def ik(*args, **options):
        calls.append(options)
        return False, np.zeros(8)
    target = np.eye(4)
    target[0, 3] = .000097
    with pytest.raises(MotionRejection) as caught:
        solve_pose(SimpleNamespace(ik=ik, fk=lambda *a: np.eye(4)), target, 'tcp', np.zeros(8), stage='path_ik')
    assert len(calls) == 1
    assert caught.value.diagnostic['continuation_budget_exhausted'] is True
    assert caught.value.diagnostic['iterations'] == 300


@pytest.mark.parametrize('kind', ['translation', 'rotation', 'nonfinite'])
def test_distant_or_invalid_candidate_never_uses_continuation(kind):
    calls = []
    target = np.eye(4)
    q = np.zeros(8)
    if kind == 'translation':
        target[0, 3] = .00101
    elif kind == 'rotation':
        target[:3, :3] = Rotation.from_euler('z', .00101).as_matrix()
    else:
        q[0] = np.nan
    def ik(*args, **kwargs):
        calls.append(kwargs)
        return False, q
    with pytest.raises(MotionRejection):
        solve_pose(SimpleNamespace(ik=ik, fk=lambda *a: np.eye(4)), target, 'tcp', np.zeros(8), stage='path_ik')
    assert len(calls) == 1
