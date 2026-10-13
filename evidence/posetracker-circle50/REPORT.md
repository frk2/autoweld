# PoseTracker circle at 50 mm/s — 2026-09-21

Two R60 mm YZ circles completed at 50 mm/s using the selected POS_VEL gains, with the ChArUco board clamped throughout motion. The final recording is in `../posetracker-circle50-repeat3/`. It provides useful observations, but does **not establish a calibrated mechanical shaking amplitude**: camera depth variation dominates, stationary raised-pose variation is comparable, and only 37.2% of circle time survives the conservative vibration-analysis exclusions.

**Final hardware state:** arm at its original verified rest, maximum arm error 0.02186°, motors **enabled** to keep the board gripped, closed gripper target 0, measured normalized opening about 0.033. SDK faults absent. Joint speed ceiling restored to 0.15 rad/s; automatic excursion remains 0.49 rad. The stored rest record is byte-for-byte unchanged. Do not use normal disable while the board depends on powered grip.

## Final measurement

| Quantity | Result |
|---|---:|
| Reference circle radius / duration | 60 mm / 8.131 s |
| Requested feed | 50 mm/s |
| Camera | See3CAM_CU135, `/dev/video1`, 640×480 MJPG |
| Exposure | Manual, driver control 150; independently read back |
| Actual processed rate | 66.70 frames/s, zero recorder-skipped frames |
| Pose-estimation processing | 8.08 ms mean, 14.36 ms maximum |
| Valid poses across recording | 4134/4224, 97.87% |
| Valid poses during circle | 461/542, 85.06% |
| Trusted 2–20 Hz coverage during circle | 37.25%, approximately 3.03 s |
| Encoder/FK peak distance to requested circle | 1.376 mm |
| Encoder/FK maximum timed position error | 10.963 mm |

Encoder errors concern the existing modeled TCP, not the camera board center. The unconstrained-orientation reference and original acceleration/jerk planner were retained. The run completed without an SDK communication, oscillation or execution-watchdog fault, or a 10 mm path-deviation flag. This is not welding-normal orientation qualification or a stage completion.

### Apparent board shaking, 2–20 Hz

Position below is the center of the printed 75×105 mm checkerboard, transformed from PoseTracker's board-origin pose. Camera X is right, Y down, Z depth. The board-to-tool transform is unmeasured.

| Interval | Vector RMS (mm) | X peak-to-peak (mm) | Y peak-to-peak (mm) | Z peak-to-peak (mm) | Trusted time |
|---|---:|---:|---:|---:|---:|
| Initial rest hold | 0.121 | 0.117 | 0.101 | 0.641 | 100% |
| Raised hold before circle | 0.396 | 0.222 | 0.538 | 2.315 | 33.3% |
| Circle | **0.474** | **0.653** | **1.303** | **2.852** | **37.2%** |
| Raised hold after circle | 0.511 | 0.185 | 0.652 | 2.695 | 94.9% |
| Final rest hold | 0.120 | 0.056 | 0.090 | 0.654 | 83.2% |

Within the usable circle portions the largest vector band excursion was 1.767 mm and angular RMS was 0.248°. A spectral maximum near 7.26 Hz comes from only 3.03 seconds of trusted observations; it is **not an established robot resonance**. Excluding the reference ramps plus one second on each side leaves 2.35 seconds of trusted cruise observations: 0.402 mm vector RMS and X/Y/Z peak-to-peak 0.371/1.048/1.955 mm. Missing intervals can contain unobserved larger peaks.

At the raised stationary holds J1–J3 encoder positions were unchanged at packet resolution; J4–J6 varied by at most one count (0.02186°). Camera depth still varied substantially. This is consistent with a substantial pose-estimation contribution, but does not rule out board flex, downstream compliance or motion below encoder resolution. These holds include any real robot vibration and are not an independent camera-noise calibration. No backlash value is inferred from this circle.

## Acquisition and detector correction

PoseTracker was launched before each acquisition and recorded across the circle and dwell periods. Host monotonic camera-delivery timestamps were aligned to SDK feedback using wall/monotonic anchors. This is software clock alignment, not hardware exposure synchronization; USB buffering, rolling shutter and exposure timing remain measurement limitations.

The existing calibration uses 15 mm squares, 10.5 mm markers, DICT_4X4_50, and a 5×7 board; stored reprojection RMS is 0.253 px. Physical print scale and the board-to-TCP transform were not newly measured. Focus, native resolution and exposure stayed unchanged. No temporal smoothing was applied to recorded poses.

The stock detector often found too few ChArUco corners in the small VGA marker image, even after the user repositioned the camera. An internal 2× enlargement of a fixed image region `(x=100,y=0,width=400,height=420)` recovered valid poses in all 79 saved audit frames, versus 32/79 with the original detector. Camera intrinsics were transformed for this internal crop/resize; returned corners were mapped back using OpenCV's pixel-center convention before the original PoseTracker pose fit. This does not create additional optical resolution. The original minimum of six corners and 1.5 px reprojection threshold were retained. Runtime tracking remained about 67 fps with no skipped frames.

Offline analysis uses PoseTracker's 1 Hz smooth trajectory and 2–20 Hz band, fourth-order Butterworth forward/backward filtering, separate continuous valid runs, and one-second edge exclusions. Failed detections are not interpolated across. The usable coverage, filter attenuation and rolling-shutter effects limit interpretation of amplitudes.

## Attempt history

- `grip-check-rejected/`: first close request rejected because the initial implementation unnecessarily required the arm already at the captured rest. Fresh J6 was approximately 1.7° from that rest. No circle was sent. The user explicitly supported the board and authorized the verified-rest disable/restart; closing now preserves any healthy stationary arm pose.
- `./`: first completed circle with stock PoseTracker; 44.75% circle poses valid and no continuous circle segment qualified for default vibration filtering. Encoder peak edge distance 1.336 mm. Normal return retained powered closure.
- `../posetracker-circle50-repeat/`: camera warm-up coverage failed a cumulative initial check; no seed/approach/circle was sent. Changed the startup check to use the most recent two seconds.
- `../posetracker-circle50-repeat2/`: after the user's camera repositioning, seed and approach completed but circle-start pose coverage was 0%; **no circle reference was sent**. Returned to rest with the gripper closed. These images identified the small-marker detection problem.
- `../posetracker-circle50-repeat3/`: detector correction, successful camera checks, complete second circle and closed-grip return. This is the primary result above.

The startup/approach camera checks concern data acquisition. They do not change the user's 10 mm record-only path-deviation policy or stop an executing circle for tracking deviation.

## Files and implementation

- [Overview](../posetracker-circle50-repeat3/overview.png): raw board position, band signal, detection coverage and encoder/FK path error.
- [PoseTracker plots](../posetracker-circle50-repeat3/analysis/report.png), [metrics](../posetracker-circle50-repeat3/summary.json), [raw poses](../posetracker-circle50-repeat3/poses.csv), [timestamps](../posetracker-circle50-repeat3/timing.json).
- Camera settings, calibration, sparse audit images, complete joint/gripper telemetry and both accepted circle traces are retained beside each attempt.
- `scripts/posetracker_circle.py`, `record_board_pose.py`, `pose_detector.py`, and `analyze_board_circle.py` reproduce acquisition/analysis using the existing PoseTracker project. The PoseTracker repository itself was not edited.
- The SDK app now carries a separate closed-grip target through Hold, arm waypoints, circle plans and return-to-rest. The SDK's existing gripper gains and 50 N force-limiter setting remain unchanged; measured gripper effort was 1.023–1.038 Nm in the final acquisition, not an independently calibrated grip force. Normal disable is blocked while this hold is active. `runtime/live/gripper-hold.json` preserves that intent. Arm rest verification is explicitly labeled arm-only under this condition. The former open-gripper rest target is not overwritten or replayed during motion.
- Existing webcam `/dev/video0` was absent, so the app uses `--camera ''` while PoseTracker owns video1. Current app invocation is `--live --start-disabled --camera '' --port 8081 --runtime runtime/live`. The source additionally rejects ordinary auto-calibrating startup while saved board-hold intent is active; use `--start-disabled`.
- Validation: full 211-test suite passed before the final idle-pose correction; the affected 60 control tests then passed, and the resize-coordinate regression passed. Ruff passes for all changed files. No bulk software test run was performed during motion.

A reliable physical shaking amplitude needs stronger optical validation: consistent board detection through the entire path, a rigid board and stationary reference at the working depth, and a check against a known physical displacement. Higher-resolution acquisition requires calibration at that native mode. The requested exposure 150 was preserved throughout these tests.
