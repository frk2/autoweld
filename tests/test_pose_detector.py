import numpy as np

from scripts.pose_detector import ScaledBoardDetector


def test_scaled_crop_pixel_center_mapping_round_trip():
    detector = object.__new__(ScaledBoardDetector)
    detector.roi = (100, 20, 400, 420)
    detector.scale = 2
    native = np.array([[[150.125, 85.75]], [[311.23, 297.11]]], dtype=np.float32)
    enlarged = (native - [100, 20] + .5) * 2 - .5
    np.testing.assert_allclose(detector.native_points(enlarged), native, atol=1e-6)
