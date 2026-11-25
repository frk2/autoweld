"""View predictions must use the same workspace/calibration as native frames."""

import json
from types import SimpleNamespace

import pytest

import autoweld.camera as bridge


def test_projection_uses_live_runtime_model_and_rejects_obsolete_calibration(tmp_path, monkeypatch):
    camera = object.__new__(bridge.CameraService)
    camera.workspace, camera.serial = tmp_path/'live', '25440983'
    model = dict(source_id='spinnaker:25440983', schema_version=1, synthetic=False,
                 image_size=[1440, 1080], camera_matrix=[[4000, 0, 603], [0, 4000, 392], [0, 0, 1]],
                 distortion=[0, 0, 0, 0, 0], camera_rms_px=.5,
                 board=dict(columns=5, rows=7, square_mm=4.))
    installed = camera._intrinsic_path()
    installed.parent.mkdir(parents=True)
    installed.write_text(json.dumps(model))
    obsolete = {**model, 'camera_matrix': [[4000, 0, 479], [0, 4000, 392], [0, 0, 1]]}
    other = tmp_path/'runtime/flir/data'/installed.name
    other.parent.mkdir(parents=True)
    other.write_text(json.dumps(obsolete))
    calls = []
    def project(points, calibration, **kwargs):
        calls.append((points, calibration, kwargs))
        return {'uv': [[603, 392]], 'projection_only': True}
    modules = {'calibration': SimpleNamespace(Calibration=SimpleNamespace(from_dict=lambda data: data)),
               'geometry': SimpleNamespace(project_camera_points=project)}
    monkeypatch.setattr(bridge, '_weldfinder_module', modules.__getitem__)
    current_id, _ = bridge._intrinsic_identity(model, model['source_id'])
    old_id, _ = bridge._intrinsic_identity(obsolete, model['source_id'])
    optical = bridge._optical_intrinsics_id(model)
    result = camera.project_camera_points([[0, 0, 150]], source_id=model['source_id'],
        optical_intrinsics_id=optical, calibration_id=current_id)
    assert result['projection_only'] and result['calibration_id'] == current_id
    assert calls == [([[0, 0, 150]], model, dict(source_id=model['source_id'], image_size=[1440, 1080]))]
    for binding in (dict(calibration_id=old_id, optical_intrinsics_id=optical),
                    dict(calibration_id=current_id, optical_intrinsics_id=bridge._optical_intrinsics_id(obsolete))):
        with pytest.raises(ValueError, match='does not match the native acquisition'):
            camera.project_camera_points([[0, 0, 150]], source_id=model['source_id'], **binding)
    assert len(calls) == 1  # No SDK call on mismatched optics.
