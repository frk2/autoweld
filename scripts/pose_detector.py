"""Decode small ChArUco markers at 2x, returning native calibrated coordinates."""
import cv2
import numpy as np


class ScaledBoardDetector:
    def __init__(self, calibration, roi=(100, 0, 400, 420), scale=2):
        self.roi = roi
        self.scale = scale
        x, y, _, _ = roi
        matrix = calibration.matrix.copy()
        matrix[0, 2] -= x
        matrix[1, 2] -= y
        matrix[:2] *= scale
        # OpenCV resize maps pixel centres, not image edges.
        matrix[:2, 2] += (scale - 1) / 2
        parameters = cv2.aruco.CharucoParameters()
        parameters.cameraMatrix = matrix
        parameters.distCoeffs = calibration.distortion
        self.detector = cv2.aruco.CharucoDetector(calibration.board.build(), parameters)

    def native_points(self, points):
        return (points + .5) / self.scale - .5 + np.asarray(self.roi[:2], np.float32)

    def detectBoard(self, gray):
        x, y, width, height = self.roi
        crop = gray[y:y + height, x:x + width]
        enlarged = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, markers, marker_ids = self.detector.detectBoard(enlarged)
        if corners is not None:
            corners = self.native_points(corners)
        markers = tuple(self.native_points(m) for m in markers)
        return corners, ids, markers, marker_ids
