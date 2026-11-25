"""The service bounds numerical worker pools before loading scientific libraries."""

import json
import os
import subprocess
import sys


def test_fresh_service_import_bounds_all_blas_backends_without_constructing_robot():
    code = '''
import autoweld
import numpy as np
import scipy.linalg
from threadpoolctl import threadpool_info
import json
assert np.isfinite(scipy.linalg.solve(np.eye(6), np.ones(6))).all()
print(json.dumps([{'name': p['prefix'], 'threads': p['num_threads']} for p in threadpool_info() if p['user_api'] == 'blas']))
'''
    process = subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True,
        env={**os.environ, 'OPENBLAS_NUM_THREADS': '16', 'OMP_NUM_THREADS': '16', 'MKL_NUM_THREADS': '16'})
    pools = json.loads(process.stdout)
    assert pools and all(pool['threads'] == 1 for pool in pools)
