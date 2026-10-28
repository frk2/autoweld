"""Exact existing pose/quality gates; no hardware, model, or live process."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_scanning_coordinator import capture_partial, start
from test_scanning_coordinator import rig as rig


def arrive_next_cell(rig):
    capture_partial(rig)
    rig.advance(.001)
    assert rig.scan.status == 'moving'
    target = [event[1] for event in rig.events if event[0] == 'move'][-1]
    rig.pose[:] = target
    rig.advance(.001)
    assert rig.scan.status == 'settling' and rig.scan._capture_arrival
    return len([e for e in rig.events if e[0] == 'move']), len([e for e in rig.events if e[0] == 'hold'])


def test_optical_xy_must_converge_before_one_raster_hold(rig):
    start(rig)
    rig.scan._handeye[:3, 3] = [.05, 0., 0.]
    rig.scan._goal = np.eye(4)
    rig.scan._move_started = rig.clock[0]
    rig.scan.status = 'moving'
    rig.scan._after_move_status = 'settling'
    rig.pose[:3, :3] = Rotation.from_euler('z', .004).as_matrix()
    rig.pose[1, 3] = .0004
    assert not rig.scan._endpoint()
    assert rig.scan.endpoint_error['position_mm'] == pytest.approx(.4)
    assert rig.scan.endpoint_error['rotation_rad'] < .005
    assert rig.scan.endpoint_error['optical_xy_mm'] > .5
    rig.pose[1, 3] = .0002
    assert rig.scan._endpoint()
    assert rig.scan.endpoint_error['optical_xy_mm'] <= .5


def test_post_hold_jitter_finishes_quiet_interval_without_repeated_motion_or_hold(rig):
    moves, holds = arrive_next_cell(rig)
    rig.pose[0, 3] = .00055
    rig.advance(.02)
    since = rig.scan._stable_since
    assert since is not None
    for _ in range(2):
        rig.advance(.1)
    assert len([e for e in rig.events if e[0] == 'move']) == moves
    assert rig.scan._saved['cells_visited'] == 1
    rig.pose[0, 3] = .0004
    rig.advance(.21)
    assert rig.scan._stable_since == since
    assert rig.scan._pending_kind == 'append'
    rig.advance(.001)
    assert rig.scan._saved['cells_visited'] == 2
    assert len([e for e in rig.events if e[0] == 'move']) == moves
    assert len([e for e in rig.events if e[0] == 'hold']) == holds
    assert rig.scan._saved['last_cell']['base_from_tcp'][0][3] == pytest.approx(.0004)


def test_persistent_grid_miss_requires_quiet_and_bounded_multiple_fresh_observations(rig):
    moves, holds = arrive_next_cell(rig)
    rig.pose[0, 3] = .00055
    rig.advance(.01)
    rig.advance(.41)  # Quiet qualified; first out-of-tolerance acquisition.
    assert rig.scan._grid_miss_samples == 1
    assert len([e for e in rig.events if e[0] == 'move']) == moves
    rig.advance(.1)
    assert rig.scan.status == 'settling'
    rig.advance(.11)
    assert rig.scan.status == 'moving'
    assert len([e for e in rig.events if e[0] == 'move']) == moves+1
    assert len([e for e in rig.events if e[0] == 'hold']) == holds
    assert rig.scan._saved['cells_visited'] == 1


def test_healthy_arrival_has_no_new_debounce_dwell(rig):
    moves, holds = arrive_next_cell(rig)
    rig.advance(.01)
    rig.advance(.41)
    assert rig.scan._pending_kind == 'append'
    assert rig.scan._grid_miss_since is None
    assert len([e for e in rig.events if e[0] == 'move']) == moves
    assert len([e for e in rig.events if e[0] == 'hold']) == holds


def test_previous_cell_qualified_hold_moves_to_next_nominal_cell_without_new_quiet_dwell(rig):
    capture_partial(rig)
    assert not rig.scan._capture_arrival
    before = len([e for e in rig.events if e[0] == 'move'])
    rig.advance(.001)
    assert rig.scan.status == 'moving'
    assert len([e for e in rig.events if e[0] == 'move']) == before+1


def test_stale_feedback_breaks_persistent_miss_evidence_without_commanding_motion(rig):
    moves, holds = arrive_next_cell(rig)
    rig.pose[0, 3] = .00055
    rig.advance(.01)
    rig.advance(.41)
    assert rig.scan._grid_miss_since is not None
    rig.measured['fresh'] = False
    rig.advance(.3)
    assert rig.scan._grid_miss_since is None and rig.scan._stable_since is None
    rig.measured['fresh'] = True
    rig.advance(.01)
    rig.advance(.41)
    assert rig.scan.status == 'settling'
    assert rig.scan._grid_miss_samples == 1
    assert len([e for e in rig.events if e[0] == 'move']) == moves
    assert len([e for e in rig.events if e[0] == 'hold']) == holds


def test_end_timeout_and_capture_timeout_stay_bounded(rig):
    start(rig)
    rig.scan._handeye[:3, 3] = [.05, 0., 0.]
    rig.scan._goal = np.eye(4)
    rig.scan._move_started = rig.clock[0]-30.01
    rig.scan.status = 'moving'
    rig.scan._after_move_status = 'settling'
    rig.pose[:3, :3] = Rotation.from_euler('z', .004).as_matrix()
    rig.pose[1, 3] = .0004
    with pytest.raises(ValueError, match='endpoint did not converge'):
        rig.scan._endpoint()
    rig.scan._begin_settle(capture_arrival=True)
    rig.clock[0] += 15.01
    rig.scan.tick()
    assert rig.scan.status == 'error' and '15 s' in rig.scan.error
