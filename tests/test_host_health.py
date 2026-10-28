"""Pressure admission uses proc fixtures, never robot or camera hardware."""

import json

import pytest

from autoweld.host_health import HostHealthGuard


@pytest.fixture
def healthy_proc(tmp_path):
    (tmp_path / 'self').mkdir()
    (tmp_path / 'pressure').mkdir()
    (tmp_path / 'meminfo').write_text('MemTotal: 16000000 kB\nMemAvailable: 2097152 kB\n')
    (tmp_path / 'self/status').write_text('Name: fixture\nVmSwap: 0 kB\n')
    for source in ('memory', 'io'):
        (tmp_path / 'pressure' / source).write_text(
            'some avg10=10.00 avg60=0.00 avg300=0.00 total=100\n'
            'full avg10=0.00 avg60=0.00 avg300=0.00 total=0\n')
    return tmp_path


def test_safe_status_is_finite_and_memory_uses_mib(healthy_proc):
    guard = HostHealthGuard(healthy_proc, wall_time=lambda: 123.)
    status = guard.check()
    assert status == dict(safe=True, reasons=[], memory_available_mb=2048.,
                          process_swap_mb=0., memory_full_avg10=0., io_full_avg10=0.,
                          timestamp=123.)
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize('path,text,reason', [
    ('meminfo', 'MemAvailable: 1048575 kB\n', 'MemAvailable'),
    ('self/status', 'VmSwap: 16384 kB\n', 'Owner VmSwap'),
    ('pressure/memory', 'full avg10=1.0\n', 'Memory PSI'),
])
def test_each_threshold_rejects_with_specific_reason(healthy_proc, path, text, reason):
    (healthy_proc / path).write_text(text)
    guard = HostHealthGuard(healthy_proc)
    status = guard.sample()
    assert not status['safe']
    assert len(status['reasons']) == 1
    assert reason in status['reasons'][0]
    with pytest.raises(RuntimeError, match=reason):
        guard.check()


def test_safe_threshold_sides(healthy_proc):
    (healthy_proc / 'meminfo').write_text('MemAvailable: 1048576 kB\n')
    (healthy_proc / 'self/status').write_text('VmSwap: 16383 kB\n')
    (healthy_proc / 'pressure/memory').write_text('full avg10=0.999\n')
    (healthy_proc / 'pressure/io').write_text('full avg10=4.999\n')
    assert HostHealthGuard(healthy_proc).check()['safe']


@pytest.mark.parametrize('pressure', [0., 5., 60., 100.])
def test_io_pressure_is_telemetry_only_and_never_blocks_live_work(healthy_proc, pressure):
    (healthy_proc / 'pressure/io').write_text(f'full avg10={pressure}\n')
    status = HostHealthGuard(healthy_proc).check()
    assert status['safe'] and status['reasons'] == []
    assert status['io_full_avg10'] == pressure


def test_missing_io_telemetry_does_not_block_live_work(healthy_proc):
    (healthy_proc / 'pressure/io').unlink()
    status = HostHealthGuard(healthy_proc).check()
    assert status['safe'] and status['io_full_avg10'] is None
    assert status['reasons'] == []
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-1', '100.1', 'broken'])
def test_invalid_io_telemetry_does_not_block_live_work(healthy_proc, value):
    (healthy_proc / 'pressure/io').write_text(f'full avg10={value}\n')
    status = HostHealthGuard(healthy_proc).check()
    assert status['safe'] and status['io_full_avg10'] is None
    assert status['reasons'] == []
    json.dumps(status, allow_nan=False)


def test_all_pressure_reasons_are_reported_together(healthy_proc):
    (healthy_proc / 'meminfo').write_text('MemAvailable: 1 kB\n')
    (healthy_proc / 'self/status').write_text('VmSwap: 100000 kB\n')
    (healthy_proc / 'pressure/memory').write_text('full avg10=50\n')
    (healthy_proc / 'pressure/io').write_text('full avg10=60\n')
    status = HostHealthGuard(healthy_proc).sample()
    assert len(status['reasons']) == 3
    assert not status['safe']


@pytest.mark.parametrize('path', ['meminfo', 'self/status', 'pressure/memory'])
def test_missing_required_metric_fails_closed(healthy_proc, path):
    (healthy_proc / path).unlink()
    guard = HostHealthGuard(healthy_proc)
    assert not guard.sample()['safe']
    with pytest.raises(RuntimeError, match='unavailable'):
        guard.check()


@pytest.mark.parametrize('text', [
    '', 'MemFree: 1 kB\n', 'MemAvailable: nan kB\n', 'MemAvailable: -1 kB\n',
    'MemAvailable: 1.5 kB\n', 'MemAvailable: 2097152 MB\n',
    'MemAvailable: 2097152 kB\nMemAvailable: 2097152 kB\n',
])
def test_malformed_memory_is_unknown_and_rejected(healthy_proc, text):
    (healthy_proc / 'meminfo').write_text(text)
    status = HostHealthGuard(healthy_proc).sample()
    assert not status['safe'] and status['memory_available_mb'] is None
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-1', '100.1', 'broken'])
def test_invalid_psi_is_unknown_and_rejected(healthy_proc, value):
    (healthy_proc / 'pressure/memory').write_text(f'full avg10={value}\n')
    status = HostHealthGuard(healthy_proc).sample()
    assert not status['safe'] and status['memory_full_avg10'] is None
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize('text', [
    'some avg10=0\n', 'full avg60=0\n', 'full avg10=0 avg10=0\n',
    'full avg10=0\nfull avg10=0\n',
])
def test_ambiguous_or_missing_full_pressure_rejects(healthy_proc, text):
    (healthy_proc / 'pressure/memory').write_text(text)
    status = HostHealthGuard(healthy_proc).sample()
    assert not status['safe'] and status['memory_full_avg10'] is None


def test_missing_proc_tree_remains_serializable_and_fails_closed(tmp_path):
    guard = HostHealthGuard(tmp_path)
    status = guard.sample()
    assert len(status['reasons']) == 3
    json.dumps(status, allow_nan=False)
    with pytest.raises(RuntimeError):
        guard.check()


def test_sampling_throttles_until_deadline_and_force_bypasses_cache(healthy_proc):
    ticks = [10.]
    guard = HostHealthGuard(healthy_proc, monotonic=lambda: ticks[0], wall_time=lambda: ticks[0]+100.)
    assert guard.status()['safe']
    (healthy_proc / 'self/status').write_text('VmSwap: 20000 kB\n')
    ticks[0] = 10.999
    assert guard.check()['safe']
    ticks[0] = 11.
    assert not guard.sample()['safe']
    (healthy_proc / 'self/status').write_text('VmSwap: 0 kB\n')
    assert not guard.sample()['safe']
    assert guard.check(force=True)['safe']
    assert guard.status()['timestamp'] == 111.


def test_returned_status_cannot_mutate_cached_reasons(healthy_proc):
    guard = HostHealthGuard(healthy_proc)
    status = guard.sample()
    status['reasons'].append('injected')
    status['safe'] = False
    assert guard.check()['safe']
    assert guard.sample()['reasons'] == []


def test_nonfinite_timestamp_stays_json_finite_and_fails_closed(healthy_proc):
    status = HostHealthGuard(healthy_proc, wall_time=lambda: float('nan')).sample()
    assert not status['safe'] and status['timestamp'] is None
    json.dumps(status, allow_nan=False)


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf')])
def test_invalid_sampling_interval_refused(interval):
    with pytest.raises(ValueError, match='interval'):
        HostHealthGuard(sample_interval_s=interval)
