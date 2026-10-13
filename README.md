# AutoWeld — Stage 3 path tracking

Latest [path-deviation repeat report](evidence/path-deviation-repeat/REPORT.md):
10 mm record-only policy, 16 completed paths, two joint-bound preflight rejections, full plots and settling data.

Position–speed controller: select **Position–speed** after **Disable motors at rest**,
then **Enable motors**. This uses each actuator's onboard cascade controller, with
the original MIT gripper. See [documentation](docs/POS_SPEED.md) and
[commissioning results](evidence/pos-speed/REPORT.md). MIT remains available.

Fresh implementation based on the official i2rt `examples/control_with_viser` example. Viser renders
the YAM v1 URDF, and the SDK's MuJoCo/Mink kinematics solve the end-effector pose. All motor
communication, gravity compensation, friction compensation, and selectable PD/PID control run through i2rt.
There is no ROS runtime. Stage 2 adds timed, kinematically limited waypoint references inside the SDK.

**Stage 1 is not DONE.** PID defaults to the user-selected gravity-calibrated profile; guarded P/I/D tuning remains available. P/D adjustments are bounded to 20% per step.
Current guarded comparisons are in evidence/retune/. Earlier fast comparisons are historical in evidence/fast/. Older uniform-accuracy comparisons are in evidence/pi/. Only the user may mark a stage DONE. The user explicitly requested starting the updated Stage 2 (jerk, acceleration and velocity), which is now in progress.

## Stage 3: line and arc tracking

The **Path tracking · Stage 3** panel imports straight and circular STEP edges in robot-base
coordinates (STEP lengths in mm). This is independent of the workpiece placement controls.
`assets/stage3/lines.step` contains five 20 mm lines; `circles.step` contains five circles of
12.7 mm radius; `quarter-arcs.step` contains five 90° arcs of that radius. Individual fixtures
and their nominal seed joint poses are in the same directory.

1. **Load tracking edges**, select an edge, feed in mm/s, and orientation.
2. **Preview tracking path** checks IK, joint bounds, self-collision, interpolation error,
   and sampled TCP/joint velocity, acceleration and jerk limits, without moving.
3. **Move to path start** performs a separate planned approach. Wait for convergence.
4. **Run path tracking** checks the starting pose (at most 10 mm from the requested edge and within 0.01 rad per joint of the planned start),
   then streams joint position and velocity through the SDK. **Hold** interrupts immediately;
   Disable retains the normal verified-rest workflow.

Orange preview geometry includes lead-in/out extensions. Smooth speed ramps lie outside the
requested edge; every point on the requested edge has the selected reference feed. If limits
cannot accommodate that feed, planning rejects it rather than silently reducing speed.
There is no welder trigger output. Preview validates robot self-collision, not workpiece clearance.

**Unconstrained** follows position with minimal wrist motion and does not score orientation error.
**Perpendicular** points the torch axis (TCP −Z) perpendicular to a straight edge, choosing a
nearby orientation. **Center facing** points TCP −Z toward the circular center throughout an arc.
All options are preflighted; geometric feasibility and automatic-test limits can reject a path.

Each run writes `runtime/live/tracking-*.jsonl` samples and a matching `.json` summary. The panel
shows edge distance, time-aligned tip error, orientation error, and Jacobian-derived speed from joint
feedback. Distance over 10 mm from the requested finite edge, or constrained orientation
error above 5°, marks the trajectory **DEVIATED** and lets the planned motion finish.
SDK faults still interrupt motion. Edge distance is checked throughout ramps, traversal
and final hold; lead extensions therefore also consume this tolerance. Timed error is recorded only:
being ahead of or behind schedule does not by itself stop or fail a run. Qualification requires a
complete run plus a one-second final hold, adequate sample coverage, global edge distance at most
10 mm, on-edge feed within ±20%, and constrained orientation within 5°.
Ramp speed is explicitly excluded from the on-edge feed criterion. The raw velocity criterion
is not smoothed to hide violations. These are encoder/model measurements, not independent
absolute TCP metrology; the placeholder torch and base frame remain uncalibrated.

The latest user request authorizes repeating paths previously stopped by tracking guards
with the [10 mm reporting-only policy](docs/PATH_DEVIATION.md). Results remain separately qualified; failed preflights
are recorded without forcing a move. All automatic path points,
including lead extensions, normally remain inside the existing <0.49 rad test envelope.
The specifically authorized 500 mm line repeat used its temporary 1.55 rad exception,
then restored 0.49 rad after verified-rest disable. No gain,
gravity, friction or original PD behavior is changed by Stage 3.
The previous 0.05 rad adjacent-IK knot check has been removed at the user's request. Joint bounds,
self-collision, interpolation residuals, reference velocity/acceleration/jerk limits, automatic-test
excursion checks and SDK oscillation/watchdog guards remain active.

Run `python -m scripts.build_tracking_fixtures` to regenerate fixtures from the current rest
record (read-only hardware access), `python -m scripts.stage3_trial --index 0` for one bounded
live line trial, and `python -m scripts.plot_tracking` for evidence plots. Trials assume an
already-enabled idle controller. Mathematical/offline and live results are kept separately in
`evidence/stage3/`. The earlier 5 mm repeat is in
[evidence/stage3-5mm/REPORT.md](evidence/stage3-5mm/REPORT.md): 141 software tests passed;
15 live paths attempted, seven completed within position tolerance, none qualified on feed speed.
Five center-facing quarter-arcs were rejected in preflight. That historical run used timed error.
The historical [edge-distance repeat and raw speed-versus-time plots](evidence/stage3-edge/REPORT.md)
record 145 software tests passing and 10/15 live paths completing within edge tolerance
(5 lines, 1 circle, 4 perpendicular lines). No run qualified on feed speed; the five center-facing
arcs still fail IK preflight. Motors finished disabled at verified rest.
Stage 3 is not marked DONE.

## Stage 2: velocity, acceleration and jerk

The **Waypoint motion · Stage 2** panel sets TCP linear speed/acceleration/jerk limits
(m/s, m/s², m/s³), TCP angular limits (rad/s, rad/s², rad/s³), and per-joint limits.
Initial commissioning values are 0.03 / 0.15 / 1 for TCP translation, 0.25 / 1 / 5
for TCP rotation, and 0.35 / 1 / 5 for each joint. These are editable reference limits,
not measured hardware capabilities.

Cartesian commands follow a straight TCP position path and shortest rotation path.
SDK IK solves 41 path knots; a cubic spline gives joint positions and derivatives.
Joint commands follow a straight path in joint space. Both use seventh-degree time
scaling, with zero planned velocity, acceleration and jerk at the endpoints. The
planner checks 401 path samples for bounds and SDK self-collision, computes TCP
and joint derivatives at 801 times, and chooses a duration with a 10% time margin.
It rejects failed IK, large IK branch jumps and excessive interpolated pose error.
These sampled checks are not continuous collision or limit proofs. The existing
self-collision model excludes the ground and has no workpiece/contact model.

The SDK executes position **and planned velocity** in its own control loop.
Acceleration and jerk determine duration and are recorded; the motor protocol has
no acceleration/jerk command field, and no inertial torque feedforward is added.
Gravity stays on, friction stays off, and the selected PID gains are unchanged.
I resets once per user waypoint and remains continuous through its intermediate
reference samples. New waypoints require the previous reference to finish and
measured joint speeds below 0.05 rad/s. Releasing the TCP gizmo sends one waypoint;
grabbing and dragging only preview it. A release during motion retains only the latest
TCP target until the arm stops (15-second expiry). Hold, return, disable, mode changes
and faults cancel that pending target. Automatic test requests are never queued.
Online blending/replanning is not implemented.

**Stop and hold current pose** interrupts motion immediately; this exception is
not a jerk-limited stop. **Disable motors at rest** queues a planned return after
an active move, then performs the usual verified-rest disable. The one-second
oscillation guard remains active. A feedback/execution gap over 100 ms during a
move also latches the SDK factory-PD hold and cancels the reference.

The UI identifies the duration-limiting constraint and shows TCP request status beside
the pose controls. `command-events.jsonl` records accepted, queued and rejected requests
with rejection reasons. Applied motion limits persist in `runtime/<mode>/motion-limits.json`;
saved targets are never replayed on startup.

Status and `control-telemetry.jsonl` expose planned joint q/v/a/jerk and measured
q/velocity. UI planned peaks are reference values, not claims of actual tracking
limits. Duration is separate from settling: sufficiently low speed/jerk settings
can deliberately make a move longer than the earlier one-second settling aim.
See [Stage 2 evidence](evidence/stage2/REPORT.md) for mathematical and live checks.

## Run

From this directory:

```bash
bash scripts/setup.sh
# Offline: robot + demo STEP, no camera or CAN connection.
.venv/bin/python -m autoweld.app --step assets/demo_joint.step
# Live operator control (starts SDK calibration and motor hold).
.venv/bin/python -m autoweld.app --live --port 8081 --step assets/demo_joint.step
```

Open http://127.0.0.1:8080 for offline mode, or the selected live port. Use one live controller only.
The launcher checks for an active controller before opening the camera or SDK. Do not launch an
SDK example alongside the live app. The app binds to localhost and does not expose hardware control
to the LAN.

The environment is installed in `.venv`. `SDK_REVISION` pins the official SDK checkout;
`requirements.lock` records the tested Linux/Python 3.11 dependencies. `scripts/setup.sh` downloads
the pinned checkout if absent. Setup applies the versioned `patches/i2rt-integral.patch` idempotently. The original SDK PD command
function and its gains are unchanged; PID is a separate SDK extension selected by default.

## Controls

### Record a line by hand

Use **Factory PD** or **Tuned PD**, enable motors, then select **Gravity compensation**.
Switching from position–speed or mixed control requires **Disable motors at rest** first.
In **Record a line · A → B**:

1. Guide the tool to the first endpoint and click **Record point A**.
2. Guide it to the second endpoint and click **Record point B**.
3. Enter **Line file name**, then click **Save path + poses and load**.

The viewer shows labeled A/B orientation axes, a connecting line and its length. Either
point can be recorded again; **Clear recorded points** starts a new draft. Captures
use fresh measured joint feedback and the modeled TCP (150 mm along mount −Z),
not the joint target or CAD placement. Each point retains position, full TCP
orientation, six measured joint angles and capture/feedback timestamps.
Coordinates are in the robot-base frame; quaternion components are shown as w, x, y, z.
Recording and saving do not command motion or leave gravity mode.

The STEP file contains one directed A-to-B line, in millimetres. A companion
`<name>.poses.json` stores both endpoint poses (positions in metres, unit wxyz
quaternions), joint angles in radians, timestamps and the TCP/frame definition.
Both files are saved under
`runtime/live/trajectories/` (or the active runtime's `trajectories/` folder).
`.step` is appended when omitted. Existing files are never overwritten; choose
a new name to save another version. Unsaved point drafts last for the app session.
Keep both files together. **Tracking STEP path** automatically loads the matching
pose file, restores the endpoint displays and checks the STEP hash and directed
endpoints. Mismatched or malformed pose files are rejected. Older STEP files
without a companion still load as geometry only.

Saving also loads the line into **Path tracking · Stage 3**, using the selected
20 mm/s line feed, **Recorded poses** orientation and inside-edge ramps. Tool
orientation follows the shortest rotation from A to B, interpolated by distance
along the line; equal endpoint orientations remain constant. Outside-edge ramps,
if selected, smoothly extrapolate this rotation through the lead-in/out.
Full-pose IK and existing angular/joint derivative checks apply. Other orientation
options remain available when intentionally overriding the recorded poses. Use
**Preview tracking path**, **Move to path start**, and **Run path tracking** when
ready. Recording reachable endpoints does not establish that their connecting
line passes the existing full-path IK, joint and collision checks.

### Controller settings

- **Controller — PD / PID:** PD uses factory gains with I=0. The original SDK PD method is unchanged; Stage 2 moves use the separate SDK reference executor for either selection.
  PID uses the separate `command_joint_pos_pid()` method. P/I/D are editable; native D is at most 5. Factory D is
  `[5, 5, 5, 1.5, 1.5, 1.5]`. Click **Apply PID gains** to apply the fields together.
  Each P/D coefficient may change by at most 20% of its current PID setting per application;
  for example, 80 → 96 → 115.2. Invalid changes leave all settings untouched.
  Default P: `[80,96,152.8823808,29.85984,14.4,61.917364224]` Nm/rad; default I: `[60,60,75,3,6,6]`; default D: `[5,5,5,2.592,1.5,2.16]`.
  Earlier high-P/I selections remain historical in evidence/fast/ and are not startup defaults.
  The integral deadband is **0.06°**, as requested by the user.
  Integral torque is bounded at ±5 Nm on J1–J3 and ±1 Nm on J4–J6; gripper P/D stays factory
  and the gripper has no I term. Gravity stays enabled; friction compensation is now disabled by default at the user's request.
  I resets for a new waypoint and accumulates immediately on the next valid control interval, with no startup delay. Repeating an
  identical target preserves I. PD/gravity transitions and reconnect clear I; stale feedback
  is not integrated. Controller selection retains the current target. In gravity mode selection
  affects the next position command. Offline simulation uses ideal following, without PID dynamics.

- **Hold:** capture the current arm position as a position command.
- **Gravity compensation:** use the SDK's gravity idle mode, preserving the selected friction setting (off by default).
  Returning to Hold, Joint, or IK begins from measured joint positions.
- **Joint:** enter six joint angles in radians, then execute a smooth, timed joint-space waypoint through the SDK.
  UI commands have no displacement limit relative to the prior command or captured rest.
  Joint and IK paths retain joint-range/path checks and configured velocity, acceleration and jerk limits.
  Automatic tests retain their separate strict 0.49 rad excursion checks.
- **IK:** drag the TCP frame, or enter XYZ in metres and roll/pitch/yaw in radians. Rotations use
  `Rz(yaw) Ry(pitch) Rx(roll)`. Failed IK retains the previous command. The SDK example's robot
  self-collision check is retained for IK. This is not a workpiece collision planner.
  Rejected TCP poses show the failing stage beside the pose controls: endpoint IK, path IK,
  path continuity, joint/collision validation, or interpolated-path tolerance. IK failures report
  iteration count and separate position/orientation residuals versus thresholds. Path failures
  identify progress along the path and the offending joint/value where available. Residuals from
  IK use the SDK Mink SE(3) log metric; path residuals use Cartesian distance and rotation angle.
  A failed numerical search does not prove unreachability; the continuity check does not prove
  singularity. Structured details are saved under `diagnostic` in request responses,
  `status.json` (`last_request`), and `command-events.jsonl`.
- **Return to captured rest:** command the session's startup arm and gripper positions, using
  one velocity/acceleration/jerk-limited waypoint, then wait for measured rest verification.
- **Grab something:** close the gripper at the current, idle arm pose using the existing SDK
  force limiter. Keep its closed target through subsequent arm moves and rest returns.
- **Disable motors at rest:** one click returns to rest, waits until all joint feedback is within
  0.5° for one second, then calls SDK `robot.close()`. There is no camera checkbox or confirmation
  step. Grabbing an object never blocks this action: the gripper stays closed during return,
  object contact is excluded from the arm rest check, then **all seven motors lose power**.
  Closed grip intent remains saved for reconnect; no opening command is sent during disable.
  Active motion or residual movement queues the request. A latched SDK fault at quiet, verified
  rest can still disable without clearing the fault; away from rest the request waits for explicit
  fault recovery before returning. Motion controls are unavailable during this sequence. A step that fails to settle within
  15 seconds leaves motors enabled at the last target and reports the failure; it never disables
  away from verified rest. Repeated clicks do not reset the sequence.
- **Enable motors:** creates a fresh SDK connection and holds freshly measured joint positions.
  It reuses the session's gripper calibration and preserves its original rest pose and camera
  reference. It does not replay the previous IK/joint target. While disabled, the viewer explicitly
  shows the last measured position; it is not live motor feedback.

Ctrl+C / SIGTERM uses the same return-and-disable sequence, then exits. Camera images remain
available for monitoring but do not gate disable. An SDK communication fault follows SDK behavior
and cannot promise return to rest.

The gripper is fixed at its captured position for arm tests. SDK startup autocalibration is enabled,
as explicitly authorized. That is gripper stroke calibration, **not calibration of the six arm zeros**.
The displayed cylindrical torch is only a placeholder. Its TCP is 150 mm along the gripper mount's
negative Z axis, with mount orientation. Its mass is not added to the real robot's gravity model.

The motion accuracy criterion is **0.2° for J1–J3 and 1° for J4–J6**, with a **1-second convergence goal**.
The separate rest-verification criterion is **0.5° for every joint**. These thresholds are reported in status telemetry and
the UI. A pose can pass rest verification while failing motion accuracy. Rest verification requires
one second settled; normal disable remains blocked otherwise.

## STEP files

Load a local `.step` / `.stp` file or use Upload STEP. OpenCascade imports its geometry and units,
tessellates all imported shapes, and converts millimetres to the viewer's metres. Multiple solids and
curved surfaces are supported. Move/rotate the workpiece with its gizmo or numeric placement controls.
The demo is a base plate and vertical plate. Placement is manual; CAD-to-robot registration and joint
localization remain future work; the revised Stage 2 concerns motion derivatives. STEP colors/names are not preserved; imported faces use one material.

## Latest overshoot requirement

The user now requires **at most 0.1° overshoot**, retaining the per-joint accuracy targets and one-second aim.
The [21-trial lower-P search](evidence/overshoot/REPORT.md) did not establish a profile that meets both limits
in both directions. The earlier fast PID defaults below are historical and do not meet the new overshoot ceiling.
No failed experimental candidate has been promoted as a compliant replacement. The latest live session ended
in original PD, disabled at verified rest. The user declined trajectory shaping and authorized D tuning instead. Direct waypoints remain in use.
The fourth D trial developed wrist oscillation and lost motor-1 communication. The user reported bad noise/sticking and requested removal of all D changes. D overrides were temporarily removed after that incident; the latest request permits guarded full-PID tuning from factory defaults. A subsequent user-requested restart completed successfully with original PD and factory D; feedback and camera are fresh. See [incident report](evidence/damping/REPORT.md).

## Earlier PI tuning — per-joint targets

The requested goal is sustained convergence within one second: **±0.2° on J1–J3 and ±1° on J4–J6**.
The selected gains and live comparison are in [the current report](evidence/fast/REPORT.md).
`scripts/tune_pid.py` records new trials with per-joint thresholds, error at one second, and
uninterrupted settling through the entire recording. `scripts/plot_fast.py` reproduces the
current convergence and comparison plots. Older recordings and their original scoring remain intact.
Those historical trials retained the original PD command, factory D, 0.06° integral deadband, and 0.5-second integral delay;
friction compensation and the 0.5° verified-rest threshold were unchanged. The current controller removes that integral delay per the latest user instruction.

## Earlier PI tuning — uniform 0.1° criterion

The earlier PI tuning superseded the factory-P-only restriction: use 0.06° deadband, keep
D fixed, change P only in steps no larger than 20%, compare convergence and overshoot on small
live angles, and select the best measured PI profile. The original PD method remains unchanged.
The SDK extension is reconstructed by `patches/i2rt-integral.patch`.

The [PI report](evidence/pi/REPORT.md) records 34 completed trials and all convergence plots.
That earlier candidate reached 0.131° worst overshoot and 11–16 s settling on the standard
outbound/return cycle. Opposite-direction validation still reached 0.508° overshoot and failed
the 0.1° criterion during 40 s. It is not a fully validated fast controller.
Earlier [factory-P integral-only trials](evidence/pid/factory-only/REPORT.md) and the
[superseded high-P report](evidence/pid/REPORT.md) are historical; their timings are not results
for the current settings. Only the user may mark Stage 1 DONE.

`scripts/tune_pid.py` sends one bounded trial through the app inbox, enforces the 20% P step,
requires the SDK oscillation guard, enforces 20% P/D steps, and records approximately 50-Hz encoder/torque data. Only one trial
recorder can run at a time. It reports directed overshoot, final two-second error, first one-second
settling, and final uninterrupted settling. A brief pass followed by drift is not accepted as
convergence. `scripts/plot_pi.py` renders standalone PNG/PDF comparisons from saved trials.
These are sampled encoder measurements, not absolute physical joint calibration or TCP metrology.
All current live tests are close to captured rest with the current gripper/load.

Validation includes actual SDK update tests with fake feedback, unchanged original PD source,
native D bounds and fixed gripper P/D, atomic rejection of P/D changes beyond 20%, integral timing/reset/windup,
app lint, reproducible SDK patch application and offline end-to-end integration. Offline ideal
following does not validate physical PID convergence.

## Earlier commissioning evidence — 2026-09-14

The earlier commissioning restart on 2026-09-14 used port **8081** with UI/automatic-source
separation, **0.1° motion accuracy**, and **0.5° rest verification**. The SDK started in Hold mode;
the live UI can now switch modes normally.
The current state is in `runtime/live/status.json`; restart evidence is in `evidence/restart-result.json`.
The original session is archived in `evidence/live-before-restart`. Before stopping that obsolete
process, the arm was commanded to its original captured rest and independently verified within 0.35°
for over one second, with a fresh camera inspection. A forced process stop was needed because the old
process could not use the revised shutdown tolerance. The SDK then initialized a new session and
captured a new rest reference.

The subsequent one-click disable/enable update is deployed at the same URL. Its live test completed
disable → enable → disable through the running UI, preserving the session's original rest pose.
The test ended with motors **disabled** and Enable motors available; final rest error was approximately
0.372°. Evidence: `evidence/live-lifecycle-result.json`, `runtime/live/disabled-rest.jpg`.
The SDK logged an old control-thread/file-descriptor exception during connection shutdown; the
application's disable completed, and reconnect and the second disable both passed. This SDK shutdown
log behavior remains upstream of the app.

- SDK: `5b72c47239bd056d0fa6c1a39edeb0537c89443c`.
- YAM Standard/v1, six joints plus LINEAR_4310 motor 7, can0 at 1,000,000 bit/s, BRIO `/dev/video0`.
- Default SDK gains and friction compensation; no gain changes during commissioning.
- A +0.08 rad elbow waypoint moved approximately +0.0454 rad and settled about **1.983° short**.
- A nearby TCP target solved and was commanded; its arm target changes stayed below **0.081 rad**.
  Physical pose following retained approximately **1.983° maximum joint error**.
- After return, camera images show the folded rest pose. Maximum measured joint deviation remains
  approximately **0.4°**; the original 0.01° criterion had not passed. This arm error is within the
  user's subsequently revised 0.5° rest threshold (automatic verification also checks the gripper
  and settled time).
- SDK startup gripper calibration ran and reported `[0.249676, -2.702182]` motor radians, a span of
  **2.951858 rad**. This is smaller than the nominal 6.57 rad model stroke and needs validation before
  trusting gripper opening measurements. It was left fixed during arm tests.
- SDK p16 feedback uses a 25 rad range over 65,535 intervals: **0.0218569° per position increment**.
  This was coarser than the original 0.01° target, but is finer than the revised 0.1° target.
  The original PD approximately 2° tracking error exceeded the revised target; see PID results above. Encoder increments
  alone do not establish absolute physical accuracy.

See `evidence/live-results.json`, `evidence/captured-rest.jpg`, `evidence/returned-rest.jpg`, and
`evidence/captured-rest.json`. Runtime telemetry is recorded at approximately 5 Hz in
`runtime/live/telemetry.jsonl`; it is not a high-rate dynamics measurement. Motor temperature,
voltage under load, and absolute TCP accuracy were not validated. Gravity-mode switching is covered
offline; no live free-hand gravity sweep was performed.

## Tests and local integration

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check autoweld tests scripts
# With a fresh, running offline viewer only (this ends with the simulation disabled):
PYTHONPATH=. .venv/bin/python scripts/smoke_offline.py
```

Tests cover command scope, invalid targets, fixed gripper, automatic disable, rest-verification
invalidation, bounded return steps, failed return/reconnect, repeat enable/disable cycles,
gravity/hold transitions, STEP units and multiple solids, URDF/MuJoCo TCP agreement,
and a reachable SDK IK waypoint. Offline simulation has ideal position following and is not evidence
of physical accuracy.

The initial 15-test validation and offline end-to-end smoke run passed joint and TCP
commands, unreachable-IK rejection, mode switching, STEP loading, and the rest/verify/disable sequence.
The smoke result is saved in `evidence/offline-smoke.json`.

Current command-limit regression coverage accepts UI moves larger than 0.5 rad, retains URDF bounds,
and checks separate automatic-test limits for both joint and IK commands. Larger planned moves are
checked numerically against velocity, acceleration and jerk limits without commanding hardware.

Current validation after the motor lifecycle update: **38 tests pass**, Ruff passes, and the updated
offline smoke test passes including one-click return/disable, enable, and another disable. The
same near-rest cycle also passed on live hardware. Distant multi-waypoint return was tested offline.

For tooling, `scripts/request.py` submits a request to the running app's serial command loop. It never
opens another SDK or CAN connection. For example:

```bash
.venv/bin/python scripts/request.py --runtime runtime/live --payload '{"action":"snapshot"}'
.venv/bin/python scripts/request.py --runtime runtime/live --payload '{"action":"rest"}'
```

Other actions are `controller` (`controller`: `PD`/`PID`, optional six-element `ki`), `mode`, `joint` (six-element `q`), `pose` (`position` plus `rpy` or `wxyz`), `sync`,
`step` (`path`), `place_cad`, `enable`, `disable`, and `shutdown`. Requests and responses are
local JSON files. Inspect status after motion; an accepted request does not mean its target was reached.
For disable, wait until status reports `closed: true`; acceptance starts the return sequence.

Live commands delivered through that file inbox are **automatic requests** and additionally must stay
strictly within 0.49 rad of current, captured rest, and prior target. Browser events are distinguished
inside the app and never get that test-only rest envelope. There is no global `--testing` UI mode.

## Guarded factory-baseline retune

Following the wrist incident, defaults were reset to factory P `[80,80,80,10,10,10]`, factory D `[5,5,5,1.5,1.5,1.5]`, and zero I. The first integral experiment uses I=10 as explicitly requested; the 20% rule applies to P and D, not I. This reset preceded the later user-selected gravity-calibrated default. No trajectories are used.

The SDK monitors a full one-second window of fresh arm feedback. It requires at least six substantial velocity reversals (at least three in each half), speed above 0.15 rad/s, and at least 0.3° position span in each half. The second-half span must remain at least 70% of the first, so a decaying waypoint transient does not qualify. Reversals below 0.1° are ignored. A feedback gap over 0.1 s resets the observation window. Detection immediately substitutes factory PD holding the measured pose, clears integral correction, and latches a motion fault. Gravity/friction feedforward remains active. It does not turn motor power off or attempt a return movement. New commands remain overridden in the SDK while latched. The UI blocks waypoints and exposes **Acknowledge oscillation fault**; acknowledgment requires fresh feedback and one second with position span at most 0.1° and speed at most 0.1 rad/s, and resets selected gains to the factory baseline. Normal disable still requires verified rest.

This software guard is a response to detected oscillation, not a guarantee against every unsafe motion or a hardware safety stop. The original 0.3-second guard produced a user-identified false alarm; the revised one-second guard ignores that trace. Replay detects the bad-noise incident at about 1.10 s, before its later communication failure; live tuning must stop on a trip.

## Immediate integral retune

The latest [34-trial live search](evidence/instant/REPORT.md) used immediate I accumulation with reset on new waypoints, P/D changes bounded to 20%, small direct waypoints and the one-second oscillation guard. No image/camera checks were performed, as requested. The fastest passing individual outward trial settled in 0.222 s with 0.032° maximum recorded overshoot; no fixed profile met both limits across all tested directions. Non-rest reversals remained above the overshoot limit. That search ended with factory P/D and I=0 selected and motors disabled at verified rest. [Convergence plots](evidence/instant/convergence.png) and [full trial metrics](evidence/instant/trials.json) are saved.

## Tunable gravity and complete torque telemetry

Gravity and native SDK friction were already enabled in earlier trials. The UI now exposes six gravity multipliers and a choice of exact SDK geometry or a local linear gravity model around captured rest. The current default uses local linear mode with multipliers `[1,1.04,1.31,1.32,1,1]`; the previous exact-model tuning results remain in historical evidence. Linear mode uses the SDK downward gravity vector and coupled link-angle derivative; outside ±0.15 rad of its reference it uses the exact SDK model. It does not restrict UI waypoints or switch gravity off on return moves. The selected gravity settings are reapplied on Enable motors.

The torque table and recordings separate estimated P/D, I, gravity, native friction, estimated total and motor-reported torque. Estimated total includes gravity; the SDK's feedforward torque field excludes onboard PD. Disabled displays retain the labeled last active torque sample. Closed-state `gravity_factors` / `gravity_model` are the selected settings for next enable; the nested SDK `gravity` and `torque` snapshots describe the last connected robot.

The [33-trial gravity retune](evidence/gravity/REPORT.md) improved the matched raised-pose pair to 0.218/0.196 s but did not qualify a fixed profile across outward, reversal and rest under the strict 0.1° overshoot ceiling. The user subsequently selected the best measured candidate as the default, despite its recorded overshoot limits. [Gravity-inclusive torque plots](evidence/gravity/torque-breakdown.png) show the feedforward and PID contributions. Startup and Enable motors apply the selected PID and gravity settings at the measured pose; both gravity and SDK friction stay enabled. The candidate recorded 0.115° J6 reversal overshoot and 0.393° J4 rest overshoot, so it is not validated against the strict 0.1° ceiling. Original PD and oscillation-fault recovery retain factory P/D and zero I.

## Friction-disabled tuning

Latest user instruction supersedes the earlier always-enabled friction setting: run all remaining tests without friction compensation and keep it disabled in the selected configuration. `vTarget` remains zero; the proposed 1/2 rad/s velocity targets were withdrawn. The UI now exposes a separate friction enable switch, per-joint coefficients and an optional continuous tanh transition. A zero transition retains the native SDK sign rule. These options preserve gravity compensation and apply again on reconnect; friction is off by default.

`--gravity-calibration` starts the app with I=0, friction disabled and the exact SDK gravity model. `scripts/tune_pid.py` expects friction off by default; its explicit `--friction-compensation` flag is required for any future authorized friction-on experiment. Gravity-only calibration also requires I=0. Active test monitoring rejects unexpected changes to compensation flags. See `evidence/smooth/` for calibration, comparison and subsequent friction-off trials.

The [selected friction-off profile](evidence/smooth/selected-profile.json) replaces the earlier gravity profile. Repeated combined raised-pose moves settled in 0.197–0.244 s with maximum recorded overshoot 0.059°. Small isolated J1–J3 moves still took 1.07–2.03 s; the profile is not qualified across all moves. The user confirmed rest touches a stop/link/table, so contact-rest torque must not be used to fit free-space gravity. Normal disable still requires the separate 0.5° rest check for one second. [Results and plots](evidence/smooth/REPORT.md).
