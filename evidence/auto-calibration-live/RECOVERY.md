# Powered return to rest and service restart

The operator explicitly requested recovery to rest, permitting manual CAN
commands if necessary. The recovery used the SDK throughout; no raw CAN command
encoding, fault clearing, zeroing, mode change, Flash write or motor enable was
used during transit.

The stuck controller could not clear its old pending-disable latch. Killing it
at Calibration Pose was rejected by automatic approval review because holding
torque could disappear. A safer powered recovery was implemented and approved:

1. Verify fresh healthy six-axis feedback, POS_VEL mode and native TIMEOUT=8000.
2. Preload a complete joint-bound/collision-checked slow path to captured rest.
3. Pause all old writer threads; immediately seed the new SDK writer with a
   fixed hold at the measured pose. Never resume stale old targets after motion.
4. Follow the path with fresh feedback, speed/tracking/timing/temperature guards.
5. At rest, require position error ≤0.5° and speed ≤0.05 rad/s for one second.
6. Continue powered rest hold while killing the paused owners and acquiring the
   released hardware lock; then disable all six motors and verify disabled status.

The initial 49-second path hit the unchanged 0.04 rad tracking guard at 15.93 s.
The controller retained fixed powered hold and settled without a motor fault.
The remaining path was replanned at lower speed: 137.64 s, peak TCP speed
6.36 mm/s and peak joint speed 0.02477 rad/s. The guard was not widened.
A replacement controller acquired the fixed hold without a power transition;
handoff gaps were 6.27 ms initially and 8.15 ms for the replacement. Exactly one
writer was active during transit; all other controller threads remained stopped.

The slow path completed, rest was verified, and all six motors independently
reported disabled (0x0) in three fresh polls. Maximum final rest error was
0.02186°. Maximum observed temperature was 39°C; final maximum was 38°C. No
motor fault or thermal trip occurred.

The corrected service restarted on http://127.0.0.1:8082 with `--start-disabled`.
Its shutdown latch is clear; the arm remains at rest, motors disabled and camera
off. The disabled UI labels its pose as a saved rest display. Hardware verification
comes from `powered-recovery-result.json`, not that cached display.

Files:

- `powered-rest-preflight-initial.json` / `powered-rest-path-initial.npz`: initial path.
- `powered-rest-preflight.json` / `powered-rest-path.npz`: slower remaining path.
- `powered-recovery-survey.json`: native modes, watchdogs and feedback.
- `powered-recovery.jsonl`: handoffs, guarded hold, motion, rest and shutdown.
- `powered-recovery-result.json`: final fresh disabled feedback for IDs 1–6.

Intrinsic calibration and physical checkerboard hunting have not yet been
completed. The robot was left at rest to satisfy the operator's latest request.

Validation: 617 tests passed; Ruff passed. A fresh headless browser connected
with no console errors and confirmed Disabled, an enabled Enable motors button,
no pending shutdown, saved-rest labeling and Camera off. The test browser was
closed; the service remains running.
