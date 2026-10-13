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
