"""Gap tolerance never upgrades unobserved or competing returns to a robot path."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

import autoweld.camera as bridge
from autoweld.weld_seam_service import WeldSettings


@pytest.mark.parametrize('value', [True, None, -1., 10.01, float('nan'), float('inf')])
def test_profile_gap_is_bounded_independently_of_surface_fit_error(value):
    with pytest.raises(ValueError, match='profile gap'):
        WeldSettings(max_profile_gap_mm=value).validate()


@pytest.mark.parametrize('gap_status,reflection_status,expected', [
    ('not_observed', 'resolved_optical', 'found'),
    ('inferred_gap', 'resolved_optical', 'invalid'),
    ('not_observed', 'ambiguous', 'ambiguous'),
    ('not_observed', 'candidate_overflow', 'ambiguous'),
    ('not_observed', 'no_supported_returns', 'invalid'),
    ('ambiguous', 'resolved_optical', 'ambiguous'),
])
def test_optical_ambiguity_and_missing_intervals_never_become_measured_seams(
        monkeypatch, gap_status, reflection_status, expected):
    cloud = {'reflection': {'status': reflection_status}, 'xyz_mm': [[0., 0., 150.]]}
    original = copy.deepcopy(cloud)
    calls = []
    def find(cloud, **kwargs):
        return dict(status='found', found=True, seam_point_mm=[0., 0., 150.],
            camera_from_torch=[[1., 0., 0., 0.]], flank_lines_mm=[[[0., 0., 150.]]],
            quality={'good': True})
    def gap(cloud, **kwargs):
        calls.append(kwargs)
        return dict(status=gap_status, inferred_center_mm=[0., 0., 150.],
            counts_as_measured_seam=False, physical_gap_confirmed=False)
    module = SimpleNamespace(find_weld_seam=find, find_fitup_gap=gap,
        validate_reflective_seam_support=lambda *_, **__: {'qualified': True})
    monkeypatch.setattr(bridge, '_weldfinder_module', lambda name: module)
    result = bridge.CameraService.analyze_reflective_joint(cloud, max_profile_gap_mm=4.)
    assert cloud == original
    assert result['status'] == expected
    assert calls == [{'settings': {'max_gap_mm': 4.}, 'search_hint_mm': None}]
    assert result['fitup_gap']['counts_as_measured_seam'] is False
    if expected != 'found':
        assert result['found'] is False and result['seam_point_mm'] is None
        assert result['camera_from_torch'] is None and result['flank_lines_mm'] == []


def test_zero_gap_limit_is_allowed_without_changing_existing_default_joint_kind():
    settings = WeldSettings(max_profile_gap_mm=0.).validate()
    assert settings.joint_kind == 'local' and settings.reflective_surface is False


def test_native_candidate_archive_lookup_is_bound_to_exact_preview_not_latest_exposure():
    fields = dict(source_id='camera', optical_intrinsics_id='optics', calibration_id='laser',
        acquisition_session_id='session', generation=2, sequence=10, read_started=100.,
        xyz_mm=[[0., 0., 150.]], uv=[[500., 100.]], controls={'exposure_us': 1000.},
        timing={'device_frame_id': 70})
    preview = {**copy.deepcopy(fields), 'reflection': {'status': 'ambiguous'}}
    native = {**fields, 'xyz_mm': np.array(fields['xyz_mm']), 'uv': np.array(fields['uv']),
        'reflection': {'status': 'ambiguous', 'candidate_table': {'uv': [[500., 100.], [500., 130.]]}}}
    newer = {**fields, 'sequence': 11, 'timing': {'device_frame_id': 71}}
    cursors = []
    def history(cursor):
        cursors.append(cursor)
        return {'clouds': [native, newer]}
    camera = object.__new__(bridge.CameraService)
    camera._backend, camera._operation = SimpleNamespace(recent_laser_clouds=history), None
    restored = camera.native_laser_cloud(preview)
    assert restored['sequence'] == 10 and cursors == [('session', 69)]
    assert len(restored['reflection']['candidate_table']['uv']) == 2
    restored['reflection']['candidate_table']['uv'][0][0] = 999.
    assert native['reflection']['candidate_table']['uv'][0][0] == 500.
    native['uv'][0, 1] += 1
    with pytest.raises(ValueError, match='differs from the exact published exposure'):
        camera.native_laser_cloud(preview)
