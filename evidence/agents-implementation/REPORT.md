# AGENTS.md implementation validation — 2026-10-22

The user confirmed that motor 7 has been removed. AGENTS.md now requires six
motors and SDK NO_GRIPPER. Only receiver/camera geometry and transforms were
imported from autoweld-old. The current SDK arm supplies kinematics and limits.

## Automated validation

- AutoWeld full suite: 284 tests pass (final run recorded below).
- Ruff: `autoweld tests scripts` passes.
- SDK patch reconstruction applies to the pinned pristine checkout and reproduces
  every changed/added SDK file enumerated by the rebuild script.
- Offline JSON smoke passes joint motion, saved-pose recall, invalid-IK rejection,
  gravity-mode refusal, and two verified-rest disable/reconnect cycles.
- Headless Playwright passes UI controls, shared saved poses across two clients,
  camera-off-by-default, and verified-rest disable. Reproduce using
  `scripts/smoke_ui.js`; `scripts/smoke_offline.py` refuses live hardware.
- New sibling camera adapter tests pass. A broad sibling test run had 204 passes
  and one existing independent preview/cloud publication timing-race failure;
  this is not claimed as a fully passing sibling suite.

## Live validation

- Existing CAN owner was discovered and avoided. The user stopped it; its former
  UI still retained FLIR USB handles and was stopped before the final camera test.
- can0: gs_usb/CANable Pro, 1,000,000 bit/s, ERROR-ACTIVE (normal CAN state).
- SDK-only YAM NO_GRIPPER/POS_VEL startup captured the measured current rest.
- Bounded J2/J3 +0.02 rad move completed. Largest final joint target error was
  about 0.022 degrees. This is encoder/model evidence, not absolute metrology.
- UI Disable motors at rest returned the arm and verified rest before SDK close.
  First return's largest error from captured rest was about 0.175 degrees.
- Final service supports --start-disabled using a labeled saved display with no
  CAN connection until Enable. Its real SDK enable/disable cycle also passed.
- FLIR serial 25440983 streamed 1440×1080 at about 59–60 fps (reported sensor
  cadence ~60.38 fps), native continuous auto exposure, no camera warnings.
- Fixed/read-only correct Mono8 settings are accepted by readback; incompatible
  fixed settings remain errors. A host AE fallback is available for models that
  cannot provide native continuous AE; that fallback was tested synthetically.
- No full checkerboard was detected. Auto-calibration rejected the missing
  baseline without moving. No new physical intrinsic/hand-eye fit is claimed.
- Camera was stopped and motors finished disabled at verified rest. The UI at
  http://127.0.0.1:8082 remains available; feedback while disabled is labeled as
  cached rather than fresh hardware feedback.

## Limits

SDK position-speed accepts position and a positive speed ceiling, with no torque
feedforward. True gravity-compensation/free-drive is incompatible with the
position-speed-only requirement. The UI reports this and does not switch motor
firmware mode. Hold and IK switch without reconnecting. The TCP reference speed
limit is 3 m/s; the independent default joint ceiling remains 0.15 rad/s.

Cooldown, stale-data handling, shared model identity/reconnect, camera ownership,
cone geometry and measured hand-eye recovery were checked without heating the
physical motors or inducing communication faults. These guards do not guarantee
against physical communication/power/firmware failures.
