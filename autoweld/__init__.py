"""AutoWeld robot service and JSON-driven visualization."""

import os

# Configure before NumPy/SciPy/OpenCV import. Small six-axis calculations and
# geometric fits do not need large independent BLAS pools; their worker stacks
# and scratch buffers consume memory needed by the camera and SDK owner.
for _backend_threads in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_backend_threads] = '1'
del _backend_threads
