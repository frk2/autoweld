# Position–speed controller

Damiao implements this mode inside each actuator. Its position loop feeds a velocity
loop, which feeds the current loop. The host sends a target output angle (rad) and a
positive maximum speed (rad/s), encoded as two little-endian float32 values at
CAN ID `0x100 + motor_id`. Speed is a ceiling, not signed velocity feedforward.
Onboard position/velocity gains and acceleration/deceleration settings control the
response. These gains are different from the MIT frame's P/D fields.

Primary references:

- [Damiao DM-J4340 user manual, position–speed mode and frame format](https://damiao.enactic.ai/en/products/hardware/dm-j4340-2ec-v1.0/)
- [Damiao DM-J4310 user manual](https://wiki.aifitlab.com/damiao-docs/dm-j4310-2ec-v11-motor-instruction-manual)
- [Manufacturer's motor SDK](https://github.com/dmBots/motor-sdk)
- [i2rt SDK](https://github.com/i2rt-robotics/i2rt)

This is a plausible alternative for accurate positioning, not a guarantee of better
tracking. Float32 commands avoid MIT command quantization; the current feedback
still uses 16-bit position and 12-bit velocity. Encoder accuracy, mechanics,
firmware tuning and timing still matter. The manuals describe a potentially slower
response than MIT.

## UI operation

1. Click **Disable motors at rest** and wait for the one-second, 0.5° rest check.
2. Select **Position–speed** in **Controller**.
3. Click **Enable motors**. The arm holds its freshly measured position.
4. Use the existing Joint, IK, return-to-rest and STEP tracking controls.

Switch back to Factory PD or Tuned PD using the same disabled sequence. Gravity
mode requires MIT. Selecting a controller does not itself enable motors.
`--start-disabled` opens the live viewer using the saved rest/gripper calibration
record and makes no motor connection until Enable. The displayed position before
Enable is cached, not fresh feedback. Normal startup still selects Tuned PD.

**Grab something** closes the gripper at the current arm pose with the existing
SDK 50 N force limit. Closed intent persists through waypoints and rest returns.
**Disable motors at rest** queues behind an active move, returns to captured rest,
checks the six arm joints within 0.5° for one second, then disables all seven motors,
including the gripper. Object contact does not block arm rest verification. Support
a held object before disabling. The button remains available during motion and
latched faults: a quiet faulted arm already at rest can be disabled; away from rest,
explicit fault recovery is needed before the return. Stale feedback or failed rest
verification still prevents an unsafe disable. There is no camera gate or dialog.

## Mixed J4/J5 MIT controller

Select **Mixed: J4/J5 MIT + gravity** while disabled, then Enable. J1–J3 and J6
retain POS_VEL; J4/J5 use MIT native PD plus host gravity feedforward, and the
gripper retains its original MIT force-limited control. The same SDK disabled
survey, RAM-only CTRL_MODE write/readback and measured-position startup apply.
No onboard gain, ACC/DEC, watchdog or Flash changes are made.

J4/J5 receive signed planned velocity and the existing selected MIT gains:
Kp `[29.85984, 14.4]`, Kd `[2.592, 1.5]`. Host I and friction remain off. Gravity
uses the selected SDK/linear model and existing factors `[1.32, 1]` for J4/J5;
the other four joints receive no host torque field. The default linear model
falls back to exact SDK gravity beyond ±0.15 rad from captured rest. Other
joint entries in the MIT gain controls have no effect on their POS_VEL packets.
Onboard `KP_APR` values are retained for switching back to full position–speed.

The four POS_VEL axes receive the positive position-speed ceiling; MIT axes keep
signed velocity feedforward. The planner conservatively applies that ceiling to
all six reference velocities. Faults retain the physical mode assignment, with
factory MIT PD on J4/J5 and a measured-position hold on the POS_VEL axes. Free
gravity-compensation mode requires all-MIT and is unavailable in mixed mode.

Mixed mode is experimental. Hardware-free routing and fault tests do not establish
lower physical shaking. The matched experiment and its current status are in
[the mixed-wrist report](../evidence/mixed-wrist/REPORT.md).

The completed board-free comparison ran two repeats of a 120 mm line at 20 mm/s
and an R60 mm circle at 50 mm/s in each mode. Mixed mode reduced 2–20 Hz circle
encoder tracking-error RMS by 20% on J4 and 60% on J5; line J4 changed by only 5%.
However, mixed J5 moved just one encoder count (0.02186°) against a 0.359° commanded
circle variation, and J4 had about 0.4° mean tracking bias. Maximum line edge error
worsened from 0.140–0.162 to 1.254 mm; circle error from 1.314–1.347 to 3.585–3.589 mm.
Lower encoder-visible shaking therefore includes under-tracking; this profile is
not promoted as a precision-control improvement. All eight paths completed with
no SDK faults. Final normal disable and independent all-seven-off readback passed.

## Implementation

- All hardware traffic stays in the patched i2rt SDK. Arm IDs 1–6 use POS_VEL;
  gripper ID 7 retains MIT and its calibrated mapping and force limiter.
- A disabled-only startup survey reads every motor before any mode write. It
  checks feedback scaling and required onboard configuration. The selected mode
  is written to RAM and independently read back; nothing is saved to Flash.
  Existing onboard gains, ACC/DEC and watchdog values are preserved.
- Firmware mode switching can clear command registers. Each position-speed motor
  receives a measured-position seed **before** enable. Automatic gripper
  calibration is unavailable in this mode; existing MIT calibration is reused.
- `PositionSpeedRobot` uses the common SDK path executor and a positive 0.15 rad/s
  joint ceiling at startup. **Apply position-speed ceiling** can temporarily raise
  it while idle, up to 1 rad/s and no higher than the operator's joint-speed limit.
  The higher-speed commissioning sweep uses only the allowance required by each
  checked reference and restores 0.15 rad/s before rest return. This setting is not
  persisted. Joint reference plans are also capped at the effective value.
  Streaming targets remain
  subject to the firmware's own acceleration/deceleration behavior.
- There is no torque-feedforward field. Host gravity, friction, integral and MIT
  quantization compensation are therefore inactive on the arm. The former MIT
  settings remain available when switching back. Telemetry displays measured
  motor torque without inventing onboard P/I/current-loop contributions.
- The one-second oscillation monitor and 100 ms motion execution watchdog remain
  active. A fault latches a hold at measured position in the selected firmware
  mode; it does not attempt an energized mode switch. Existing recovery checks,
  joint/collision checks and automatic excursion limits remain enforced.
- After path validation, a planned live move requires a new feedback frame no
  older than 20 ms and measured joint speed at most 0.05 rad/s. A missing frame
  rejects the start within 0.5 s. Tracking-report failures request Hold and cannot
  prevent the controller lifecycle tick from processing rest verification/disable.
- Normal close explicitly disables motors after the application's verified-rest
  check, rather than relying solely on the firmware communication timeout.

Stage 2 derivative limits describe the planned reference; firmware interpolation
and physical tracking must be measured separately. Stage 3 now uses finite-edge
distance ≤10 mm globally, constrained orientation error ≤5°, and raw on-edge feed
within ±20%. Geometric deviations flag the trajectory without stopping it;
[SDK faults and other motion checks remain active](PATH_DEVIATION.md).
Timed position error remains informational. No stage is marked DONE automatically.

The tracking **Acceleration placement** selector defaults to **Outside edge**,
preserving the original constant-feed-over-the-whole-edge behavior. **Inside edge**
places acceleration and deceleration on the finite requested edge. It is useful
at higher feeds, where external lead-ins would exceed the global 10 mm edge-distance
tolerance. Feed qualification covers the constant-speed portion between the ramps;
global edge distance still covers ramps and final hold. Results record that feed
interval and its duration. Short high-speed intervals cannot establish sustained
convergence. Too-short edges and any joint/derivative violation are rejected.

Requested feed may equal the configured TCP speed limit; the planner tolerates
only 1 µm/s of numerical spline/Jacobian error at that boundary and retains its
5% reserve on the other planned derivatives.

## Reproduction

Hardware-free validation: `.venv/bin/python -m pytest -q`,
`.venv/bin/python -m ruff check autoweld scripts tests`, and
`.venv/bin/python scripts/rebuild_sdk_patch.py`.

The SDK-only read survey is `python -m scripts.inspect_position_speed` (app stopped,
motors disabled). Live commissioning requires the app to be enabled, healthy and
explicitly set to POS_VEL: `python -m scripts.pos_speed_suite --stage 1`, then stages
2 and 3. These issue real motion. Evidence is written under `evidence/pos-speed/`.

Follow-up evidence: [10/20 mm/s longer paths](../evidence/pos-speed-faster/REPORT.md)
and [20–100 mm/s, 120 mm paths](../evidence/pos-speed-100/REPORT.md). The latter
completed 20 paths inside the edge tolerance but did not qualify raw feed speed;
its 100 mm/s reference plateau was only 0.363 s. No faster default was selected.

The [velocity audit](../evidence/velocity-audit/REPORT.md) found a half-code-step
bias in raw motor velocity and corrected the position estimator's time/pose
alignment. Position derivatives now use a common fit time, the matching Jacobian
and reference, with a displayed quantization-only bound. Stale, duplicated or
excessively skewed feedback cannot qualify speed. Raw safety feedback and the raw
±20% criterion are unchanged. Corrected 100 mm/s plateau averages are about
55 mm/s; encoder displacement confirms an approximately 0.74 s following-response
time constant. This is measured controller lag, not a verified friction diagnosis.

[Larger-window plots](../evidence/velocity-windows/REPORT.md) compare 0.12, 0.5
and 1 s running averages from encoder displacement. The final 1 s cruise averages
reach 19.70–20.03 mm/s for a 20 mm/s command and 29.28–29.85 mm/s for 30 mm/s.
At 80 mm/s the last complete 0.5 s average is 53.85–56.31 mm/s; the 100 mm/s
plateau is too short for either larger window. These are offline plotting
windows, not changes to live guards or raw qualification.

## Operator-selected safe test feeds

The operator now selects **20 mm/s for lines** and **10 mm/s for circles/arcs**.
The Stage 3 UI shows these presets and provides **Use selected safe speed** for
the chosen edge. New paths retain all preview and tracking checks. See
[selected speed scope](SELECTED_SAFE_SPEEDS.md) and the
[oriented Stage 3 repeat](../evidence/stage3-welding/REPORT.md).

## Onboard position-gain tuning (2026-09-21)

The user selected RAM `KP_APR` values `[193.49176, 193.49176, 193.49176, 54, 77.76, 64.8]` for J1–J6 after bounded 20% tuning increments. The ten-pose timing suite reached a worst slowest/fastest arrival ratio of 1.19092, versus 1.77759 at baseline. J5 reached 0.11558° overshoot; the user explicitly accepted the new profile and changed the wrist overshoot limits to 0.5° on J4–J6. J1–J3 retain 0.1°. Settling tolerances remain 0.2° proximal / 1° wrist, sustained for one second. This changes the experiment’s acceptance limits, not the SDK oscillation or watchdog guards.

These gains are actuator RAM settings, preserved by normal controller startup/reconnect but not automatically applied or saved to Flash by the app. Flash values have not been verified. Current onboard gains appear in the startup motor survey. The gripper, speed-loop gains, integral, ACC/DEC, current limits and other protection settings were not changed. [Full evidence, per-joint settling times and limitations](../evidence/pos-speed-kp/REPORT.md).

## Larger paths with the selected gains

The [large-path repeat](../evidence/kp-path-repeat/REPORT.md) completed seven 500 mm lines at 20–100 mm/s and eight R60 mm circles at 5–80 mm/s. Peak edge deviation at line 80 mm/s improved from 5.752 to 1.083 mm; circle 40 mm/s improved from 6.588 to 1.051 mm. Worst new deviations were 1.414 mm for lines and 2.279 mm for circles. The circle 100 mm/s plan remains rejected by the existing feed-length/acceleration/jerk check.

Timed lag remains: line 100 mm/s reached 21.724 mm timed error, and circle 80 mm/s reached 16.856 mm. Valid one-second cruise averages agree with commanded speed within ±20%, but missing/invalid timing windows prevent full velocity qualification. All nine 0.2–1.0 s windows, coverage and before/after plots are retained. These are encoder/FK measurements with unconstrained orientation. The selected safe presets remain line 20 and circle 10 mm/s.

No actuator gain or other register changed during this repeat. The fixture-specific large-line excursion exception and temporary speed caps were restored; all motors finished disabled at verified rest. A recorder planning-reply timeout was fixed independently of the unchanged SDK motion watchdog. The report includes the failed approach attempt and successful retry.
