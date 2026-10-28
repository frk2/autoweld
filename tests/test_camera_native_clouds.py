"""Read-only bridge never constructs a backend or advances lifecycle/state."""
import copy
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from autoweld.camera import CameraService


def native_cloud(device=1):
    return dict(xyz_mm=np.array([[100.,200.,300.]]),uv=np.zeros((1,2)),strength=np.ones(1),
        width_px=np.ones(1),coordinate_frame='camera',units='mm',acquisition_session_id='session',
        timing={'device_frame_id':device},frame_product={'kind':'single_frame'})


def test_off_getter_does_not_construct_or_run_state_or_lifecycle(tmp_path):
    constructed=[]
    service=CameraService(tmp_path,backend_factory=lambda *a,**k:constructed.append(True))
    try:
        service._next_stream=True
        assert service.latest_laser_cloud() is None
        assert service.recent_laser_clouds()['clouds']==()
        assert service._next_stream is True and service._backend is None and not constructed
    finally:
        service._next_stream=None
        service.close()


def test_native_ring_getter_converts_once_preserves_order_and_overrun_without_state(tmp_path):
    service=CameraService(tmp_path)
    clouds=(native_cloud(1),native_cloud(2))
    requests=[]
    def recent(key):
        requests.append(key)
        return dict(clouds=clouds,overrun=True,session_changed=False,capacity=128,
            retained_keys=(('session',1),('session',2)),oldest_key=('session',1),newest_key=('session',2))
    service._backend=SimpleNamespace(enabled=True,recent_laser_clouds=recent,close=lambda:None)
    try:
        result=service.recent_laser_clouds(('session',0))
        assert requests==[('session',0)] and result['overrun']
        assert [x['timing']['device_frame_id'] for x in result['clouds']]==[1,2]
        saved=result['clouds'][0]
        np.testing.assert_allclose(saved['xyz_m'],[[.1,.2,.3]])
        assert saved['xyz_m_units']=='m' and saved['units']=='mm'
        assert service.recent_laser_clouds()['clouds'][0] is saved
        with pytest.raises(TypeError):
            saved['timing']['device_frame_id']=9
        with pytest.raises(ValueError):
            saved['xyz_m'][0,0]=7
        private=copy.deepcopy(saved)
        private['timing']['device_frame_id']=7
        assert saved['timing']['device_frame_id']==1
        assert service.latest_laser_cloud()['timing']['device_frame_id']==2
    finally:
        service.close()


def test_pending_or_stream_off_returns_empty_without_advancing_completed_future(tmp_path):
    service=CameraService(tmp_path)
    calls=[]
    service._backend=SimpleNamespace(enabled=False,recent_laser_clouds=lambda key:calls.append(key),close=lambda:None)
    try:
        assert not service.recent_laser_clouds()['clouds'] and not calls
        service._backend.enabled=True
        done=Future()
        done.set_result('finished')
        service._operation=done
        service._operation_name='starting'
        assert service.latest_laser_cloud() is None and not calls
        assert service._operation is done and service._operation_name=='starting'
    finally:
        service._operation=None
        service.close()
