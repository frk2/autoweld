# AutoWeld

Python/Viser control and camera calibration for the white i2rt YAM Standard/v1,
with the installed receiver and FLIR camera mount. There are **six physical motors
(IDs 1–6)**. Motor 7 and the linear gripper are absent; SDK startup uses `NO_GRIPPER`.
No ROS or application-level CAN commands are used.

```bash
bash scripts/setup.sh
# Offline ideal-following simulation
.venv/bin/python -m autoweld.app --port 8080
# Live: capture the current pose as rest and hold it using the SDK
.venv/bin/python -m autoweld.app --live --port 8082
# Reopen the UI with a previously captured rest display and motors disabled:
.venv/bin/python -m autoweld.app --live --port 8082 --start-disabled
```

Open the printed localhost URL. Startup refuses an occupied port or another
AutoWeld owner of the CAN channel. Configure can0 for 1,000,000 bit/s using the
CANable Pro. Camera acquisition remains off until **Stream camera** is selected.

The frontend sends JSON commands and reads JSON snapshots. One backend owns the
SDK controller, one shared compiled robot model, and one camera singleton across
browser connections. The SDK continues holding independently of browser rendering.
SDK, IK, and motion planning share the same model through motor reconnects. The
mount visuals and optical transform are the only assets imported from the old
worktree; arm geometry comes from the pinned SDK. Installed J6 limits come from
the measured receiver/camera assembly travel.

Use **Hold** to stop at the measured pose, or select **IK** and drag/release the TCP
gizmo or send a numeric pose. Joint waypoints and named saved poses are available.
TCP is the nominal installed camera optical frame, with +Z forward. The TCP speed
limit defaults to **3 m/s**; the default per-joint speed ceiling is **2 rad/s**,
with acceleration and jerk limits also enforced. This is a reference
limit, not a demonstrated hardware capability.
**Apply speed limits** updates TCP speed and the joint velocity/position-speed
ceiling together; a rejected change leaves both previous settings intact.

J6's measured endpoints are −308.83° and −118.58° in fixed SDK encoder
coordinates. Normal motion uses **−307.83° to −119.58°**, leaving 1° clearance.
The model, visualization, planner and SDK use this same range. Startup preserves
the measured encoder branch instead of wrapping J6 by 360°, and a read-only SDK
check rejects an endpoint pose before enabling power. Gravity compensation
switches to powered Hold when approaching the J6 travel margin. The supported
rest was recaptured inside the range. See [endpoint evidence](evidence/j6-limits/REPORT.md).

Selecting **Gravity compensation** automatically changes the six arm motors to
MIT torque mode through the SDK and applies the installed receiver/camera gravity
model. Hold, IK, planned motion, rest return and cooldown automatically capture a
measured-pose hold and restore position–speed control. No old target is replayed.
Transitions preserve power and use per-axis RAM mode readback and immediate hold
seeding. A measured-pose PD settling check precedes zero-position-stiffness
gravity control; transitions do not clear faults, zero encoders or save motor Flash.

The gravity panel exposes exact-model per-joint multipliers and measured torque.
Payload, factors and tuning evidence are saved with the model fingerprint and
restored on reconnect. Static fitting and bounded live commissioning probes are
available through the JSON backend. Supported rest/contact measurements are
excluded from fitting; a quiet static torque fit alone does not establish a
friction-free or physically calibrated arm.

Fresh runtime directories use the model-matched installed profile in
`assets/camera_mount/gravity/`; matching profiles saved in the selected runtime
directory take precedence.

The saved camera/receiver profile remains `[1, 1, 1, 1, 1, 1.10]`, using the prior
1 lb payload estimate and nominal flange COM. Its qualification is limited to the
[initial small-pose tests](evidence/gravity-camera/REPORT.md). Larger-angle testing
found configuration-dependent wrist errors: candidate J4 ×1.05/J6 ×1.35 passed
24 probes at 12 poses, including shoulder/elbow angles near 90°, then triggered
the J6 speed guard at a combined yaw pose. The candidate was not promoted; the
UI and bundled tuning status now report **limited**. See the
[wider-workspace results](evidence/gravity-workspace/REPORT.md). Physical payload
mass/COM identification and release-transient diagnosis remain unresolved.

**Disable motors at rest** plans a return to the session's captured rest, requires
fresh stationary feedback within 0.5° for one second, then closes the SDK.
**Enable motors** holds fresh measured positions without replaying old targets.
With `--start-disabled`, startup opens no CAN connection and labels the saved
pose as a display; Enable motors is required for fresh SDK feedback.
**Shutdown service at rest**, JSON `shutdown`, and process signals use a dedicated
shutdown path. SIGINT, SIGTERM and SIGHUP all request this shutdown; if the UI or
port is unresponsive, use `kill -HUP <service-pid>`. It takes ownership from calibration and queued motion, clears old
disable latches, and bypasses the automatic test excursion limit. It captures a
measured hold before planning a return using the configured speed ceilings
(defaults 3 m/s TCP and 2 rad/s per joint), with no hidden shutdown speed reduction.
Repeated requests preserve progress; a blocked return shows its cause and offers
**Retry shutdown at rest**. Hold pauses the return until an explicit retry.
Joint limits, collision checks, fresh feedback, thermal monitoring and SDK fault
handling remain enforced. Shutdown disables motors only after verified rest;
hardware or path failures expose a retryable blocker. A stopped SDK cannot
guarantee holding torque or establish the current motor power state.
At 60 °C the controller cancels
motion and returns to rest for cooldown; moves remain blocked until below 55 °C.
Old feedback warns and cannot qualify rest. A genuine stopped SDK thread or
communication failure still requires diagnosis; software cannot guarantee fault-free
hardware. The SDK's execution and oscillation guards preserve powered hold.

Live startup, Enable and new work are gated on Linux host health: at least
1 GiB MemAvailable, owner VmSwap below 16 MiB, memory PSI `full avg10` below
1%, and I/O PSI `full avg10` below 5%. Missing metrics reject admission. A
one-second pressure monitor requests the same dedicated shutdown if these
limits are crossed while powered; Hold and shutdown/retry stay available.
This is an early interlock, not a real-time deadline or anti-drop guarantee.
The October 27 motor 2 communication fault and prevention limits are recorded
in [the incident report](evidence/sdk-crash-20261027/REPORT.md).

JSON `sdk` telemetry includes worker liveness, feedback age and the first native
failure. Stopped/stale SDK readings are labeled cached, including motor codes
and temperatures. Motion-fault acknowledgement requires live, fresh feedback.
The first failure is archived under the runtime's `evidence/sdk-failure-*.json`
and `sdk-failure.json`; repeated tick errors produce a summary every five
seconds. Hardware faults are never ignored or automatically cleared.

Camera acquisition, checkerboard processing and intrinsics are reusable classes in
`~/Code/Weld-seam-finder/app/weldfinder/calibration_camera.py`, using the installed
Spinnaker SDK. Set `WELDFINDER_APP` if that checkout is elsewhere. The calibration
panel uses the actual backend checkerboard settings (default 5×7 inner corners,
4 mm squares). Change them to match the printed board before capture.

Automatic calibration starts with a fresh camera stream and hunts for a full
checkerboard at the baseline and at any capture pose where the board is partial.
The backend takes the centroid of all detected grid corners, subtracts the image
center, and maps that direction through the current camera optical TCP orientation.
Each correction translates exactly **5 mm** in the camera image plane, retaining
orientation and depth. It waits for settling and a newly acquired image before
recomputing. Full detection ends the hunt; partial detections never qualify as
calibration captures. The panel displays corner count, centroid offset and search
progress, and **Cancel calibration and hold** stops the search.

If no validated grid is visible (or its centroid gives no useful direction), a
bounded 5 mm square-spiral search reacquires it. Each hunt is capped at 40 proposed
steps, 30 mm radius, 200 mm total path and 60 seconds. Every move uses the existing
IK/path/joint/collision checks. Stale frames, camera identity changes, stream stop
or rejected motion stop the hunt. The partial-grid detector needs at least a
connected 3×3 corner lattice; smaller fragments use reacquisition.

After finding a full-board baseline, calibration previews/preflights 25 uniformly
spaced edge views on a 15° full-opening cone aimed at the point 150 mm forward of
the optical center. The fixed ring radius is 19.75 mm, and the baseline is not a
capture. Lines use screen-pixel thickness. It moves
sequentially, hunts when necessary, records actual corrected poses, returns to
the original requested baseline, then solves. The same pipeline supports intrinsics and
hand–eye. Hand–eye needs valid intrinsics, paired robot/camera poses, rotation about
two axes, and stationary-board residual checks. Manual motion, Hold, disable and
camera stop cancel the sequence. Unreachable/colliding views are rejected.
Calibration uses a local 80 mm / 20° / 0.49 rad envelope around its selected
baseline, retaining SDK motion, collision and joint-limit checks. At 60°C it
cancels without interrupting the controller's rest return; work resumes only
after cooldown below 55°C.
Paper-target fits retain the camera project's 20 accepted-view and 1 px intrinsic
save gates. An uncalibrated nominal CAD frame is not measured hand–eye calibration.

Hand–eye associates each checkerboard with the first accepted physical grid origin
before fitting, including deterministic correction of whole-grid 180° detector
reversals within the compact pose envelope. Ambiguous orientations and changed or
duplicate acquisition identities are rejected. `handeye.json` atomically stores
the result and its raw/normalized observations, acquisition metadata and measured
before/after TCP poses. The service reloads it with the camera off only when its
source and installed intrinsic identity match and its transform/residual checks
pass. The transform is `tcp_from_camera` relative to the active nominal `weld_tcp`;
compose it after `base_from_tcp`. See the [fresh live validation](evidence/handeye-live/REPORT.md).

Select **Laser plane calibration** in the calibration panel selector and turn on
the shared camera stream. **Auto laser calibration** uses the saved
**Calibration Pose**, assumes the laser is already on, and requests 25 discrete
views through the normal service planner at the configured motion limits.
Each move first verifies measured endpoint convergence within 0.5 mm / 0.005 rad,
captures one measured Hold, then waits for continuous quiet feedback and a fresh
image; the reference stays fixed throughout measurement.
Weldfinder owns stripe exposure, native board/stripe captures, plane fitting and
noise filtering; the panel also offers individual checks, captures, solve,
cancel, exposure/filter settings and a clean point-cloud preview. A plane is
saved only after its native view-count and fit-quality checks pass.
The laser line finder settings include auto-exposure bounds, manual exposure,
gain, preview gamma and brightness. The panel reports actual camera settings and
the effective exposure range at maximum frame rate. Disable smart stripe exposure
to hold a manual exposure. Gamma and brightness affect the preview only, including
when the point-cloud display is off; native measurements retain their original pixels.
Native full-resolution capture runs before capture-pose hunting, because laser
glare can make the fast preview report a partial grid. Only a native missing-board
result permits one bounded hunt and retry; other quality failures reject the view.

Camera XYZ is returned in metres and composed with the measured hand–eye and
mount transform to the center of J6. This reporting frame does not change the
installed nominal `weld_tcp` used for motion. The 3D preview registers points in
the robot base only with matching optical calibration, a fresh camera cloud and
fresh stationary SDK feedback. Stale or moving observations clear the preview.
After motion, world registration waits for an acquisition made inside the new
stationary interval, so a recent image from the preceding move cannot be reused.
The camera and point-cloud preview reuse the existing single frame grabber.

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check autoweld tests scripts
# Wider live gravity tests: start disabled; default only preflights the paths.
.venv/bin/python scripts/gravity_workspace.py --runtime runtime/agents-live
# Executes the preflighted sequence, then returns to rest and disables:
.venv/bin/python scripts/gravity_workspace.py --runtime runtime/agents-live --execute
# Against a running OFFLINE instance only:
.venv/bin/python scripts/smoke_offline.py --runtime runtime/sim
# Headless browser (playwright-cli installed):
playwright-cli -s=autoweld-smoke open http://127.0.0.1:8080
playwright-cli -s=autoweld-smoke run-code --filename=scripts/smoke_ui.js
playwright-cli -s=autoweld-smoke close
```

Wider gravity commissioning uses a temporary 1.8 rad excursion envelope,
configured only with motors disabled. Travel is split into ≤0.18 rad waypoints
at the existing speed ceiling. Each test pose must pass joint-margin,
self-collision, mount-clearance and predicted-torque checks on the shared model.
Half-second probes precede 1.5-second probes; SDK guard thresholds remain
0.04 rad drift and 0.3 rad/s, with a stricter 0.015 rad drift acceptance gate.
A guard event stops progression and restores powered hold; capture can overshoot
the trigger threshold while braking.
The script returns to verified rest, disables and restores the default envelope.

Local automation submits JSON files through `scripts/request.py`; it never opens
another SDK connection. For example:

```bash
.venv/bin/python scripts/request.py --runtime runtime/live --payload '{"action":"disable"}'
```

Acceptance starts the return sequence; wait for `status.json` to report
`closed: true` and `rest_verified: true`. Historical tuning/tracking/teach utilities
remain in `legacy_app.py` without a live launcher. Their old documentation is in
[docs/LEGACY_README.md](docs/LEGACY_README.md); it describes superseded controllers,
gripper hardware and earlier experiments, not current startup behavior.
