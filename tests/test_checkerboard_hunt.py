"""Deterministic image-plane hunt geometry; no detector, SDK or camera imports."""
import json

import numpy as np
import pytest

from autoweld.checkerboard_hunt import STEP_M, CheckerboardHunt, centroid_correction


def pose():
    value = np.eye(4)
    value[:3, 3] = [.3, -.2, .4]
    return value


def corners(offset=(20., 10.)):
    # Symmetric lattice keeps its centroid exactly at the requested offset.
    return np.array([[x, y] for x in (-3., 3.) for y in (-3., 3.)])+[100., 50.]+offset


def advance(hunt, current, *, points=None, now=1., full=False):
    result = hunt.next_step(current, points, [200, 100], now=now, full_board=full)
    json.dumps(result, allow_nan=False)
    if result['status'] == 'move':
        target = np.asarray(result['target_pose'])
        delta = np.asarray(result['translation_base_m'])
        np.testing.assert_allclose(target[:3, 3]-current[:3, 3], delta, atol=1e-15)
        assert np.linalg.norm(delta) == pytest.approx(STEP_M, abs=1e-15)
        np.testing.assert_array_equal(target[:3, :3], current[:3, :3])
        assert abs(delta@current[:3, 2]) < 1e-14
        assert np.linalg.norm(target[:3, 3]-hunt.anchor[:3, 3]) <= .03+1e-12
        assert result['path_m'] <= .20+1e-12
        return result, target
    assert result['target_pose'] is None and result['translation_base_m'] is None
    return result, current


def test_all_identified_corners_contribute_to_centroid():
    points = np.array([[80., 40.], [100., 40.], [80., 60.], [180., 90.]])
    delta, metadata = centroid_correction(points, [200, 100], pose())
    centroid = points.mean(axis=0)
    offset = centroid-[100., 50.]
    np.testing.assert_allclose(metadata['centroid_px'], centroid)
    np.testing.assert_allclose(metadata['offset_px'], offset)
    np.testing.assert_allclose(delta, np.r_[offset/np.linalg.norm(offset)*.005, 0.])
    assert metadata['corner_count'] == 4


@pytest.mark.parametrize('offset, expected', [((20., 0.), [.005, 0., 0.]),
                                             ((-20., 0.), [-.005, 0., 0.]),
                                             ((0., 20.), [0., .005, 0.]),
                                             ((0., -20.), [0., -.005, 0.]),
                                             ((30., 40.), [.003, .004, 0.])])
def test_pixel_axes_map_to_exact_five_mm_camera_translation(offset, expected):
    delta, metadata = centroid_correction(corners(offset), [200, 100], pose())
    np.testing.assert_allclose(delta, expected, atol=1e-15)
    assert metadata['step_mm'] == 5.


def test_camera_orientation_maps_direction_into_robot_base_without_depth():
    camera = pose()
    camera[:3, :3] = [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]
    delta, metadata = centroid_correction(corners((0., 20.)).reshape(-1, 1, 2), [200, 100], camera)
    np.testing.assert_allclose(delta, [0., 0., .005], atol=1e-15)
    assert delta@camera[:3, 2] == 0
    assert metadata['corner_count'] == 4


@pytest.mark.parametrize('points, reason', [(None, 'insufficient_corners'), ([], 'insufficient_corners'),
                                           ([[1., 2.]]*3, 'insufficient_corners'),
                                           (corners((0., 0.)), 'partial_centered')])
def test_undefined_or_centered_partial_direction_is_not_invented(points, reason):
    delta, metadata = centroid_correction(points, [200, 100], pose())
    assert delta is None
    assert metadata['centroid_reason'] == reason


def test_full_board_stops_before_hunting_even_with_offset_and_spent_budget():
    hunt = CheckerboardHunt(pose(), started_at=0., max_steps=1)
    first, current = advance(hunt, pose(), points=corners())
    assert first['status'] == 'move'
    result, _ = advance(hunt, current, points=corners((50., 30.)), now=60., full=True)
    assert result['status'] == 'complete' and result['reason'] == 'full_board'
    assert result['steps'] == 1 and not result['reacquiring']
    assert result['corner_count'] == 4
    assert hunt.next_step(current, [], [200, 100], now=61.)['status'] == 'complete'


def test_no_corner_search_is_anchor_camera_xy_square_spiral():
    anchor = pose()
    anchor[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    hunt, current = CheckerboardHunt(anchor, started_at=0.), anchor.copy()
    expected = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1],
                         [0, -1], [1, -1], [2, -1], [2, 0]])*.005
    for index, xy in enumerate(expected):
        result, current = advance(hunt, current, now=index)
        actual = anchor[:3, :3].T@(current[:3, 3]-anchor[:3, 3])
        np.testing.assert_allclose(actual, np.r_[xy, 0.], atol=1e-14)
        assert result['reacquiring'] and result['reason'] == 'reacquisition'
        assert result['corner_count'] == 0 and result['offset_px'] is None


def test_centered_partial_board_reacquires_and_never_returns_zero_step():
    hunt = CheckerboardHunt(pose(), started_at=0.)
    result, target = advance(hunt, pose(), points=corners((0., 0.)))
    assert result['status'] == 'move' and result['reacquiring']
    assert result['centroid_reason'] == 'partial_centered'
    assert np.linalg.norm(target[:3, 3]-pose()[:3, 3]) == pytest.approx(.005)


def test_centroid_reversal_does_not_revisit_anchor_or_oscillate():
    hunt, current = CheckerboardHunt(pose(), started_at=0.), pose()
    seen = [current[:3, 3].copy()]
    for index in range(20):
        offset = (20., 0.) if index % 2 == 0 else (-20., 0.)
        result, current = advance(hunt, current, points=corners(offset), now=index)
        if result['status'] != 'move':
            break
        assert min(np.linalg.norm(current[:3, 3]-prior) for prior in seen) >= .001-1e-12
        seen.append(current[:3, 3].copy())
    assert len(seen) > 3


def test_off_grid_centroid_then_lost_board_uses_exact_step_connectors():
    hunt, current = CheckerboardHunt(pose(), started_at=0.), pose()
    _, current = advance(hunt, current, points=corners((30., 40.)))
    seen = [p.copy() for p in hunt.visited]
    for index in range(1, 16):
        result, current = advance(hunt, current, now=index)
        assert result['status'] == 'move'
        assert min(np.linalg.norm(current[:3, 3]-prior) for prior in seen) >= .001-1e-12
        seen.append(current[:3, 3].copy())


def test_default_search_never_exceeds_forty_moves_or_twenty_cm():
    hunt, current = CheckerboardHunt(pose(), started_at=0.), pose()
    visited = [current[:3, 3].copy()]
    for index in range(41):
        result, current = advance(hunt, current, now=index)
        if index < 40:
            assert result['status'] == 'move'
            visited.append(current[:3, 3].copy())
        else:
            assert result['status'] == 'exhausted' and result['reason'] == 'step_limit'
    assert result['steps'] == 40 and result['path_m'] == pytest.approx(.20)
    distances = np.linalg.norm(np.diff(np.array(visited), axis=0), axis=1)
    np.testing.assert_allclose(distances, .005, atol=1e-15)


def test_radial_limit_blocks_outward_centroid_then_uses_bounded_search():
    hunt, current = CheckerboardHunt(pose(), started_at=0., max_radius_m=.01), pose()
    points = corners((20., 0.))
    reacquired = False
    for index in range(41):
        result, current = advance(hunt, current, points=points, now=index)
        reacquired |= result["reacquiring"]
        if result['status'] != 'move':
            break
        assert np.linalg.norm(current[:3, 3]-pose()[:3, 3]) <= .01+1e-12
    assert result['status'] == 'exhausted'
    assert result['reason'] in ('search_space_exhausted', 'step_limit')
    assert reacquired


@pytest.mark.parametrize('limits, now, reason', [({'timeout_s': 2.}, 2., 'timeout'),
                                                ({'max_path_m': .004}, 0., 'path_limit'),
                                                ({'max_radius_m': .004}, 0., 'search_space_exhausted')])
def test_exhaustion_never_proposes_an_out_of_budget_move(limits, now, reason):
    hunt = CheckerboardHunt(pose(), started_at=0., **limits)
    result, _ = advance(hunt, pose(), now=now)
    assert result['status'] == 'exhausted' and result['reason'] == reason
    assert result['steps'] == 0 and result['path_m'] == 0.


def test_snapshot_and_cancel_do_not_consume_moves_or_advance_spiral():
    hunt = CheckerboardHunt(pose(), started_at=10.)
    before = hunt.snapshot(11.)
    before['offset_px'] = [999., 999.]
    assert hunt.snapshot(12.)['steps'] == 0
    assert hunt.snapshot(12.)['offset_px'] is None
    stopped = hunt.cancel(13.)
    assert stopped['status'] == 'cancelled' and stopped['steps'] == 0
    result = hunt.next_step(pose(), corners(), [200, 100], now=14.)
    assert result['status'] == 'cancelled' and result['target_pose'] is None


@pytest.mark.parametrize('kwargs', [{'max_steps': 41}, {'max_steps': 1.5}, {'max_steps': True},
                                    {'max_radius_m': .031}, {'max_path_m': .201},
                                    {'timeout_s': 61.}, {'timeout_s': 0.},
                                    {'center_tolerance_px': float('nan')}])
def test_unsafe_hunt_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CheckerboardHunt(pose(), started_at=0., **kwargs)


@pytest.mark.parametrize('size, points', [([0, 100], corners()), ([200, float('nan')], corners()),
                                        ([200, 100], [[float('inf'), 1.]]*4),
                                        ([200, 100], [1., 2., 3.])])
def test_bad_image_geometry_is_rejected(size, points):
    with pytest.raises(ValueError):
        centroid_correction(points, size, pose())


def test_pose_and_time_validation_does_not_accept_rotations_or_backwards_time():
    invalid = pose()
    invalid[0, 0] = 2.
    with pytest.raises(ValueError, match='rigid rotation'):
        CheckerboardHunt(invalid, started_at=0.)
    hunt = CheckerboardHunt(pose(), started_at=10.)
    with pytest.raises(ValueError, match='no earlier'):
        hunt.snapshot(9.)
    outside = pose()
    outside[0, 3] += .031
    result = hunt.next_step(outside, corners(), [200, 100], now=11.)
    assert result['reason'] == 'radius_limit' and result['target_pose'] is None
