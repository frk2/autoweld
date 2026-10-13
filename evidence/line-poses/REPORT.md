# Recorded paths and endpoint poses

The A/B recorder now captures full measured TCP transforms from the same fresh
SDK joint sample: position, orientation, six joint angles and capture/feedback
timestamps. The TCP remains the modeled mount −Z, 150 mm site in robot-base
coordinates. This is encoder/model FK, not external pose metrology.

**Save path + poses and load** creates a directed STEP line plus a same-name
`.poses.json` companion. STEP coordinates remain millimetres; the versioned pose
file explicitly uses metres, radians and unit wxyz quaternions. It records the
frame/TCP definition and STEP SHA-256. Loading validates these fields, both
endpoint poses and their direction against the geometric edge. Corrupt or
mismatched companions fail visibly. Neither file overwrites an existing file;
the complete pose file is published before the STEP. Legacy STEP-only paths
continue to load as geometry.

The viewer restores the A/B point displays and orientation axes, and selects
**Recorded poses** with inside-edge ramps. This mode interpolates the shortest
rotation from A to B by distance along the line. Equal rotations stay constant.
Optional outside-edge ramps extrapolate the same rotation, preserving angular
continuity at the endpoints. Existing full-pose IK, interpolation residual,
joint/collision and derivative checks apply; saving/loading never starts motion.
Other orientation modes remain explicit operator overrides.

## Existing recording preserved

`runtime/live/trajectories/my-line.step` is unchanged byte for byte. Both
orientations were recovered offline from the original recorded joint angles
using the unchanged TCP model. Reconstructed positions matched both original
positions exactly; STEP endpoint checks also passed. The line is
**101.77087466948 mm** long and now has `my-line.poses.json`, retaining the
original joint angles and timestamps. Evidence: `recovered-poses.json`; the
recovery procedure is `recover_poses.py` (refuses an existing companion).

## Validation and deployment

- **131 targeted tests passed** in 14.56 s: recorder, tracking, controller and IK
  request handling. Includes actual-feedback capture, both complete endpoint
  transforms after export/reload, invalid quaternion/unit/hash/geometry rejection,
  overwrite protection, legacy files, no-command capture/save/load, shortest-arc
  interpolation across ±180°, quaternion sign invariance and a full-pose planned
  line checked against endpoint poses and motion limits. `tests.log` retains output.
- Ruff and Python syntax checks pass for changed code/tests.
- The user approved return/rest disable and restart. By the next inspection,
  the UI's normal disable had completed: `rest_verified=true`, arm error
  **0.02185694°**, no SDK fault. No additional motion command was sent by this
  task. All numerical tests ran with motors disabled.
- Viser restarted with `--start-disabled` on port 8081, and the recorded line
  plus recovered poses was loaded. Position–speed selection, 1 rad/s ceiling,
  MIT gains, I=0, command encoding, exact gravity settings
  `[0.8,1.19,1.31,1.32,1,1]`, friction-off, motion limits and gripper intent match
  the pre-restart state. Rest, motion-limit, gripper-intent and STEP files are
  byte-for-byte unchanged. See `preserved-settings.json`,
  `preserved-file-hashes.json`, `disabled-status.json`, `deployed-status.json`.
- An initial attempt to restore inactive gravity settings while Position–speed
  was selected was rejected without a hardware command. Temporarily selecting
  Tuned PD while disabled allowed restoration, then Position–speed was restored.
  This task sent no motor enable, firmware writes or live path test after restart.
- Browser verification confirmed both endpoint quaternion displays, the new
  save button, the existing line and **Recorded poses** selection. The operator
  then enabled motors, selected IK and successfully previewed this 101.77 mm
  line at 20 mm/s with recorded poses. `final-status.json` reflects the resumed
  enabled session; this supersedes the disabled deployment snapshot. No further
  tests or robot commands were issued by this task during the resumed session.

No new physical tracking performance claim or stage qualification is made.
