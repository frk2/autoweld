"""Read-only host pressure admission checks; these do not guarantee deadlines."""

import math
import threading
import time
from pathlib import Path

MIN_MEMORY_AVAILABLE_BYTES = 1024**3
MAX_PROCESS_SWAP_BYTES = 16*1024**2
MAX_MEMORY_FULL_AVG10 = 1.


def _kilobytes(text, key):
    rows = [line.split() for line in text.splitlines() if line.split()[:1] == [key+':']]
    if len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != 'kB':
        raise ValueError(f'expected one {key} value in kB')
    token = rows[0][1]
    if not token.isascii() or not token.isdigit():
        raise ValueError(f'invalid {key} value')
    value = int(token)*1024
    if not math.isfinite(value):
        raise ValueError(f'nonfinite {key} value')
    return value


def _full_pressure(text):
    rows = [line.split() for line in text.splitlines() if line.split()[:1] == ['full']]
    if len(rows) != 1:
        raise ValueError('expected one full pressure row')
    values = [token.split('=', 1)[1] for token in rows[0][1:] if token.startswith('avg10=')]
    if len(values) != 1:
        raise ValueError('missing or duplicate full avg10')
    value = float(values[0])
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('full avg10 must be finite and within 0–100 percent')
    return value


class HostHealthGuard:
    """Fail closed on memory pressure or missing required memory metrics.

    Samples are cached for one second by default. ``force=True`` bypasses that
    cache, for example immediately before enabling motors. Memory status fields
    use MiB. I/O PSI is optional telemetry and never gates admission or shutdown.
    No SDK, actuator, process-priority or OS settings are touched.
    """

    def __init__(self, proc_root='/proc', *, sample_interval_s=1., monotonic=None, wall_time=None):
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
            raise ValueError('Host health sampling interval must be finite and positive')
        self.proc_root = Path(proc_root)
        self.sample_interval_s = float(sample_interval_s)
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._lock = threading.Lock()
        self._sample_at = None
        self._status = None

    def sample(self, *, force=False):
        with self._lock:
            now = (self._monotonic or time.monotonic)()
            elapsed = None if self._sample_at is None else now-self._sample_at
            if (not force and self._status is not None and elapsed is not None
                    and 0 <= elapsed < self.sample_interval_s):
                return {**self._status, 'reasons': list(self._status['reasons'])}
            reasons = []
            status = dict(safe=False, reasons=reasons, memory_available_mb=None,
                          process_swap_mb=None, memory_full_avg10=None, io_full_avg10=None,
                          timestamp=None)
            metrics = (
                ('memory_available_mb', 'meminfo', lambda text: _kilobytes(text, 'MemAvailable')/1024**2),
                ('process_swap_mb', 'self/status', lambda text: _kilobytes(text, 'VmSwap')/1024**2),
                ('memory_full_avg10', 'pressure/memory', _full_pressure),
                ('io_full_avg10', 'pressure/io', _full_pressure),
            )
            for key, path, parse in metrics:
                try:
                    status[key] = parse((self.proc_root / path).read_text())
                except (OSError, ValueError, OverflowError) as error:
                    if key != 'io_full_avg10':
                        reasons.append(f'{key} unavailable: {error}')
            available, swapped = status['memory_available_mb'], status['process_swap_mb']
            memory = status['memory_full_avg10']
            if available is not None and available < MIN_MEMORY_AVAILABLE_BYTES/1024**2:
                reasons.append(f'MemAvailable {available:.1f} MiB is below 1024 MiB reserve')
            if swapped is not None and swapped >= MAX_PROCESS_SWAP_BYTES/1024**2:
                reasons.append(f'Owner VmSwap {swapped:.1f} MiB reaches the 16 MiB limit')
            if memory is not None and memory >= MAX_MEMORY_FULL_AVG10:
                reasons.append(f'Memory PSI full avg10 {memory:g}% reaches the 1% limit')
            try:
                stamp = float((self._wall_time or time.time)())
                if not math.isfinite(stamp):
                    raise ValueError('nonfinite timestamp')
                status['timestamp'] = stamp
            except (ValueError, OverflowError, TypeError) as error:
                reasons.append(f'Host clock unavailable: {error}')
            status['safe'] = not reasons
            self._sample_at, self._status = now, status
            return {**status, 'reasons': list(reasons)}

    def status(self, *, force=False):
        return self.sample(force=force)

    def check(self, *, force=False):
        status = self.sample(force=force)
        if not status['safe']:
            raise RuntimeError('Host pressure blocks live work: '+'; '.join(status['reasons']))
        return status
