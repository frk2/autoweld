# Receiver/camera gravity commissioning — 2026-10-22

This records the initial small-pose commissioning. Subsequent
[wider-workspace tests](../gravity-workspace/REPORT.md) exposed wrist limitations;
the current tuning status is **limited**, and no broader candidate was promoted.

Selecting **Gravity compensation** now automatically switches the six YAM arm
motors (IDs 1–6) from POS_VEL into MIT torque control through the SDK. After
measured-pose PD has settled, gravity control uses zero position gains, damping
and exact-model gravity feedforward. Hold, IK, planned motion, rest, cooldown and
normal disable restore position–speed control from the measured pose.

Mode changes preserve motor power. SDK transitions read back each RAM mode and
immediately seed the corresponding command protocol, including recovery from
partial writes or lost replies. They issue no motor OFF/ON, fault clear, encoder
zero or Flash writes. The application does not communicate directly with CAN.
Motor 7 is absent and startup uses NO_GRIPPER.

## Installed model and selected profile

The SDK, IK and planner share one compiled model containing the receiver/camera
geometry and optical TCP. Payload changes replace the flange load while SDK
execution is paused; they do not create another model or add a second payload.
Gravity is computed from the model's bias forces at zero velocity, excluding
contact/constraint forces. Factors and payload persist with the model fingerprint
and are restored before the first MIT command after reconnect.
Fresh runtime directories load the bundled profile in
`assets/camera_mount/gravity/`; matching local runtime records override it.

- Joint gravity factors: **[1, 1, 1, 1, 1, 1.10]**.
- Receiver/camera payload: **0.45359237 kg**, the prior operator 1 lb estimate.
- Nominal COM in flange coordinates, meters:
  **[-0.0244443712, 0.0115145086, -0.0496884508]**.
- Payload mass, COM and inertia have **not** been physically identified. The
  configured inertia is a nominal isotropic approximation.

See [selected-profile.json](selected-profile.json) and
[applied-profile.json](applied-profile.json).

## Live measurements

All captures and commands went through the existing JSON service and its single
SDK connection. Rest/contact samples were excluded. Twenty-four quiet free-space
position holds were collected, with unstable windows rejected. Static fitting
showed substantial approach-dependent torque hysteresis: held-out payload-fit
RMS was 0.6466 Nm; joint-factor-fit RMS was 0.5853 Nm (worst joint 1.1912 Nm).
Both fits failed their acceptance gates and **neither was applied**. See
[samples.json](samples.json), [rejected-windows.json](rejected-windows.json) and
[static-fit-assessment.json](static-fit-assessment.json).

Commissioning then used bounded hands-off gravity probes. The SDK independently
restores powered measured-pose PD when the timer expires or the configured
excursion/speed guard is reached, even if the requesting client disappears.

| Check | Result |
| --- | --- |
| Initial tuning | Four profiles at three poses, 12 probes; unit model selected among encoder-equivalent results |
| Wrist correction | Four J6 factors at the wrist pose; ×1.10 reduced 1.5-second peak drift from 0.262° to 0.044° |
| Final held-out validation | Three different poses, 1.5 seconds each; peak drift 0.022°, all probes completed by timer |
| Normal UI mode action | Six actual MIT modes, zero estimated P torque throughout; peak drift 0.197° over 3.113 seconds at the wrist pose |
| Powered transitions | All six motor status flags remained 0x1 during probes and mode transitions |
| Motor/SDK faults observed | Zero |
| Highest observed motor temperature | 31 °C |

The final held-out drift is approximately one encoder bin; it is not a claim of
absolute physical accuracy. Normal entry initially showed about 0.7° wrist drift
in one second. Requiring a fresh, position-stable MIT PD window reduced this to
about 0.26° before applying the J6 correction.

Raw results: [initial trials](tuning-trials.json), [wrist trials](wrist-trials.json),
[final validation](validation-trials.json), and
[normal mode selection](normal-gravity-selection.json). The earlier unit-factor
validation is preserved in [unit-validation-trials.json](unit-validation-trials.json).
Reproduce bounded commissioning with `scripts/gravity_trials.py` against an
already enabled, healthy JSON service; `--wrist` runs wrist comparisons and
`--validation` uses the saved selected profile.

## Verification and limits

The full Python suite passed **371 tests**; Ruff passed. Tests cover automatic
mode selection/restoration, zero position stiffness, cancellation, fresh-state
settling, timeout recovery, bounded probe expiry, stale feedback, partial mode
transitions, lost replies, model sharing, payload/regressor math and JSON/UI
state. The SDK patch was checked against the pinned source and reproduces all
modified and added SDK files.
Read-only Playwright assertions also verified the displayed J6 factor, tuning
result, disabled/rest-verified state, and camera-off control. Browser console:
zero errors (software WebGL warnings in headless Chromium).

This is a limited live drift commissioning, not full-workspace load
identification or a human backdrive/force-feel assessment. Static friction can
mask residual gravity errors. The prior payload estimate remains explicit rather
than being presented as an accepted physical calibration. No software can
guarantee that hardware will never fault.

The arm was returned to captured rest and disabled only after fresh stationary
feedback qualified the rest check. Camera acquisition remained off. The Viser
service remains available at http://127.0.0.1:8082; re-enable motors for live
feedback. See [final-status.json](final-status.json). After this verified disable,
the updated service was restarted with `--start-disabled`: no CAN connection was
opened, and the persisted profile was confirmed in
[saved-display-status.json](saved-display-status.json).
