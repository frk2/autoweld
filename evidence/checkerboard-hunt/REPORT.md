# Automatic checkerboard hunting

Implemented baseline and per-capture checkerboard acquisition for both intrinsic
and hand-eye calibration. The UI sends JSON and displays hunt progress; image
processing stays in Weldfinder and robot motion stays in the existing controller.

The camera adapter exposes all measured full/partial grid corners in native image
pixels, their centroid, image-center offset, frame identity and acquisition-start
timestamp. Partial detections supply a direction but never qualify for capture,
board axes or a calibration fit. Existing reduced-preview OpenCV SB grid detection
is reused; smaller fragments without a validated connected 3×3 corner lattice
produce no direction.

For a partial view, the backend computes
`offset = mean(corners) - [width/2, height/2]`, normalizes this vector, and rotates
`[dx, dy, 0] * 0.005` into robot-base coordinates using the current optical TCP
orientation. Every proposed correction is exactly 5 mm, with orientation and
camera depth unchanged. Each move passes the usual IK, full-path, joint-limit
and collision checks. The next correction requires a newly acquired image after
the move and settling interval; delayed processing of an older exposure cannot
authorize movement or capture.

No corners, a centered partial view, or a centroid direction that would revisit
a prior point uses bounded square-spiral reacquisition. Limits per hunt are 40
proposed steps, 30 mm radius, 200 mm accumulated path and 60 seconds, including
in-flight motion. Camera stop/error, stale or nonadvancing frames, source/session
changes, cancellation and rejected motion end the hunt in hold. A fresh-frame
grace period begins after each move settles.

Cone planning is deferred until the baseline is fully visible. Individual capture
poses can also be corrected, and accepted samples retain the actual corrected
robot pose. Completion returns to the originally requested baseline. A partial
view that reappears during native capture reenters the hunt. Manual captures
continue to require a full checkerboard.

The Viser panel shows full/partial detection, corner count, centroid offset,
baseline/capture search phase, proposed steps and reacquisition state. Cancel
remains available; manual robot commands, disable and camera stop take ownership
from the automatic hunt.

Validation:

- 475 AutoWeld tests passed, including geometry, coordinator and UI checks.
- 74 Weldfinder camera/native-calibration/target-pose/pipeline tests passed.
- A clipped raster checkerboard was measured by the real OpenCV detector, then
  recovered into full view by a virtual moving camera using repeated 5 mm steps.
- Cases cover rotated camera frames, no/partial corners, reversal prevention,
  bounds/timeouts, pre-move exposures processed later, camera identity changes,
  cancellation, rejected paths, capture eligibility and corrected-pose metadata.
- Ruff passed. Read-only Playwright checks verified the new UI instructions and
  controls with zero browser console errors.

No physical robot motion or camera acquisition was used for these tests. The
updated service remains at http://127.0.0.1:8082 with motors disabled and camera
acquisition off; physical checkerboard calibration has not been run.

Main implementation: `autoweld/calibration.py`, `autoweld/checkerboard_hunt.py`,
`autoweld/camera.py`, and `autoweld/app.py`. The reusable camera implementation
and its adapter tests are in
`/home/faraz/Code/Weld-seam-finder/app/weldfinder/calibration_camera.py` and
`/home/faraz/Code/Weld-seam-finder/app/tests/test_calibration_camera_adapter.py`.
