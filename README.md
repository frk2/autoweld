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

Open the mobile-compatible live URL: **https://faraz-precision-3640-tower.tail53558.ts.net:8082**.
Viser binds to localhost; a Tailscale Serve route explicitly accepts that hostname
and proxies HTTPS and WebSockets to `http://127.0.0.1:8082`. The installed Viser
has no `allowed_hosts` setting. Configure the route with
`tailscale serve --bg --https=8082 http://127.0.0.1:8082`; preserve existing Serve
routes. For a different Viser port, configure its corresponding HTTPS route.
Startup refuses an occupied port or another
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
The motion TCP preserves its original nominal camera reference, with +Z forward;
measured optical registration composes the saved hand-eye transform after it.
The October 29 CAD correction vertically stacks the camera and laser mechanical
centers at the saved rest while viewing sideways. URDF visuals, nominal CAD
`camera_optical` and collision envelopes share that correction; `weld_tcp` and
measured hand-eye remain unchanged. The TCP target selector independently starts
with +Z pointing down, RPY `[pi, 0, 0]`; **Snap target to measured TCP** copies the
actual orientation. Initializing this selector never commands the arm.
The TCP speed
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

To inspect or recapture an operator-positioned supported rest while disabled,
submit JSON `rest_inspect` or `rest_capture` through the running service. Its
single read-only SDK reader verifies disabled motors, fresh native agreement,
fixed encoder coordinates, installed bounds and one second of quiet feedback.
Capture also checks model collision and saves `rest.json` plus **Captured rest**;
inspection only refreshes the display. Neither action enables or moves the arm.
The [October 29 evidence](evidence/mount-orientation-20261029/) records the measured
rest, CAD alignment and unchanged calibration/dynamics checks.

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

Normal live startup, Enable and new work are gated on Linux host health: at least
1 GiB MemAvailable, owner VmSwap below 16 MiB, memory PSI `full avg10` below
1%. Missing required memory metrics reject admission. I/O PSI remains optional
telemetry and never blocks admission or requests shutdown, regardless of its value
or availability. A one-second pressure monitor requests the same dedicated shutdown if the memory
limits are crossed while normally powered; explicit controlled recovery uses
the host-pressure exception described below. Hold and shutdown/retry stay available.
This is an early interlock, not a real-time deadline or anti-drop guarantee.
The October 27 motor 2 communication fault and prevention limits are recorded
in [the incident report](evidence/sdk-crash-20261027/REPORT.md).

JSON `sdk` telemetry includes worker liveness, feedback age and the first native
failure. Stopped/stale SDK readings are labeled cached, including motor codes
and temperatures. Motion-fault acknowledgement requires live, fresh feedback.
The first failure is archived under the runtime's `evidence/sdk-failure-*.json`
and `sdk-failure.json`; repeated tick errors produce a summary every five
seconds. Hardware faults are never ignored or automatically cleared.

An agent can request an explicit controlled SDK restart through the service:

```bash
.venv/bin/python scripts/request.py --runtime runtime/agents-live \
  --payload '{"action":"recover_control"}'
```

Recovery refuses an active CAN writer, enabled motors, other hardware faults,
or a pose away from supported rest. It requires one second of fresh, quiet
six-axis feedback within 0.15 rad per joint of captured rest, installed joint
bounds and native encoder agreement. Explicit operator recovery bypasses host-pressure
admission and treats pressure as a warning for the recovered owner, so the host
monitor cannot immediately undo recovery. Normal new-work admission stays in
place. Recovery clears only disabled communication-timeout latches, then captures a measured
Hold through the normal SDK connection. Old targets are discarded and captured
rest is preserved. The 10-second intent deadline includes time spent queued;
the request helper supplies `requested_at` and the service also checks transport
enqueue time. A failed or expired request reports its blocker and cannot
enable later; retry explicitly after correcting the cause. A healthy idle Hold
is left in place. Submit one recovery request at a time and wait for its result
before submitting a new intent. Start a replacement service with `--start-disabled`; its saved
pose display alone is not fresh physical feedback.

The six motors' firmware communication `TIMEOUT` is **4000 ms**, applied and
read back as **80000 raw ticks** through the SDK while disabled during explicit enable or recovery.
This uses RAM settings without Flash writes. Fresh pre-restart readings on
October 27 were **8000 ticks = 400 ms**. The register uses **50 µs per tick**, so
**4000 ms = 80000 ticks**, a tenfold increase; raw 4000 would be only 200 ms.
See the [register-unit reference](https://damiao-motor.jia-xie.com/package-usage/cli-tool/#set-can-timeout). The
independent 400 ms motion watchdog and feedback freshness checks are unchanged.

Camera acquisition, checkerboard processing and intrinsics are reusable classes in
`~/Code/Weld-seam-finder/app/weldfinder/calibration_camera.py`, using the installed
Spinnaker SDK. Set `WELDFINDER_APP` if that checkout is elsewhere. The camera
performance and continuous-exposure timing changes are preserved in
`patches/weldfinder-continuous-scanning.patch`, based on `WELDFINDER_REVISION`.
Apply that patch to the matching sibling checkout with `git apply --check`
first, then apply `patches/weldfinder-seam-api.patch` for the seam API and tests;
preserve unrelated local edits. The installed SDK changes are reproduced
by `scripts/setup.sh` through `patches/i2rt-integral.patch`.
The calibration
panel uses the actual backend checkerboard settings (default 5×7 inner corners,
4 mm squares). Change them to match the printed board before capture.

Panel selection also selects the camera's processing work. **Camera calibration**
enables live checkerboard search; the laser panels use native stripe discovery
without that search. Automatic laser calibration enables checkerboard hunting
when needed. The choice is remembered across camera reopen and shared by all
clients. JSON clients select discovery with
`{"action":"camera_analysis","mode":"laser"}` before turning on the shared stream;
use `"checkerboard"` for live camera calibration. Native cloud `processing`
metadata reports queue, checkerboard, exposure and XYZ latency against the original
acquisition timestamps. Freshness and measurement-quality gates remain unchanged.

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

Select **Laser scanning**, enable the arm through the normal service control,
choose **Tube Scanning** and **Vertical down sweep**, then press **Fast scan** or
**Precision laser scan**. The default sweep is **80 mm along world −Z**, with the
saved viewing direction and **0° additional optical roll**. Fast scan records
while moving continuously. Precision visits all **81 positions at 1 mm
intervals**, verifies measured settling, captures one measured Hold, and records
fresh native exposures before advancing. The service owns the entire request,
export and verified-rest disable; the browser can be closed after submission.
The prepared start/end of the fast sweep are reused by the matched precision
sweep when the saved pose and calibrations agree.

The full line is validated before motion. Bounded roll alternatives can provide
joint clearance while preserving the camera origin and optical axis; the service
reports the selected roll and saves it for the matched Precision sweep. The
installed Tube pose uses **−15°**, because the 0° path reaches J2's lower bound.
Station capture waits for fresh post-exposure SDK telemetry at the same Hold;
an aged UI/service snapshot cannot qualify a measurement or cause an unnecessary
physical retry. Native exposure start must be after measured settling completes.

Live comparison on October 28 used Fast session `20261028-103716-9954fc77`
and Precision session `20261028-105741-613ffb54`, with identical frozen path and
calibrations. Fast completed in **107.76 s** (17.5 s continuous motion); Precision
completed in **274.10 s**, with exactly **81 × 3** accepted stationary exposures.
Separate ordinary least-squares cylinders of fixed **25.4 mm diameter**, with
free axes and centers, gave all-protocol-point radial RMS **0.33075 mm Fast** and
**0.09158 mm Precision**. All points were retained for these scores; Fast's six
stationary anchors were excluded from its moving-data comparison. Separate
robust frame/axial-slice-balanced diagnostics are labelled in the report.
Both independent native/pose/PLY audits passed. These are fitted-surface
consistency errors, not certified absolute 3D or timestamp accuracy. Reports and
plots are in `evidence/tube-fast-precision-20261028/comparison/`.

The file-only comparison is separate from the robot and camera processes. Install
the optional `analysis` dependencies and run:

```bash
.venv/bin/pip install -e '.[analysis]'
.venv/bin/python scripts/fit_tube_scans.py --runtime runtime/agents-live \
  --fast 20261028-103716-9954fc77 --precision 20261028-105741-613ffb54 \
  --diameter-mm 25.4 --output evidence/tube-fast-precision-20261028/comparison
```

Choose **Area** for the existing continuous area workflow, including surface
scouting, footprint coverage and automatic gap repair. A 100 × 100 mm square is centered on
the original measured Tube optical XY origin; aligning depth to **140 mm** and
applying the fixed optical roll do not recenter or rotate the square.

The continuous raster uses overlapping calibrated **laser-line footprints** and
up to **30 mm/s** feed, with smooth turns and a fixed viewing orientation. The
camera path compensates the stripe's optical offset while preserving the square
center. A 100 mm square typically needs six or seven lanes; six lanes at 20 mm
pitch take about 44 seconds before installed-model validation and retiming.
Preparation and the validated trajectory must fit the 900-second total budget
before launch. Normal
configured TCP/joint speed ceilings, acceleration, jerk, installed joint bounds,
collision, feedback, fault and temperature protections still apply. The service
scouts the visible surface to plan depth; unsupported areas remain explicit
missing geometry. It does not orbit the tube to gather other perspectives.

Moving point clouds use hardware exposure events matched to native image frame
IDs, a camera-to-host clock mapping and interpolation of actual measured SDK
encoder outputs. The scan uses each motor reply's kernel receive timestamp,
retaining its SDK request/reply provenance and bounded wall-to-monotonic clock
conversion; Python processing time does not become its sample timestamp.
The default exploratory scan uses this **measured kernel receive reference**;
firmware sample age remains unknown. The archive distinguishes qualified clocks
and bounded reference interpolation from physical sample-time qualification.
Command targets and image processing/receipt times cannot supply the feedback
reference. Missing measured brackets, calibration/session changes, invalid clocks
and excessive reference uncertainty reject the acquisition. Ordinary world
previews and laser calibration retain their stationary acquisition gates.

An optional bidirectional optical-depth experiment fits full measured camera
rotations/translations and the static native surface, then checks excluded cycles.
It can apply an empirical offset only after qualification. Calibration geometry
and timing can be confounded, so a rejected fit neither supplies a zero-age claim
nor blocks a receive-reference exploratory scan. No scale or hand-eye adjustment
is inferred merely to make an experimental fit pass.

Native acquisitions, measured poses, timing uncertainty and actual quality gaps
are retained in **`<runtime>/motion-scans/`**. The preview is capped at 10,000 points;
full and square-ROI PLY exports retain accepted native geometry. Completion must
include independently recomputed exposure-footprint coverage on the fixed 140 mm
reference plane at 1 mm resolution. Point occupancy is separate from reference
illumination coverage; missing returns do not establish an empty physical surface
and never become invented points. The service owns bounded coverage repairs.
**Cancel scan and hold** remains available; shutdown and cooldown
retain priority over scan ownership. Older discrete sessions remain available in
`<runtime>/laser-scans/`.

The live 100 × 100 mm Tube Scanning run `20261027-230823-c6cd589d` completed in
**575.1 seconds**, including preparation, six continuous lanes, automatic repairs,
export and verified-rest motor disable. Its independent file audit recomputed
all **10,201 reference-plane cells**, checked the native records and both PLY
exports, and reported no integrity errors. It retained **520,312 native points**,
including **437,863 points inside the square ROI**. Evidence is in
`evidence/square-scan-100mm-20261027/final-independent-audit.json`. This is
receive-reference registration; firmware encoder sample age remains unknown.

### Weld seam finding

Select **Weld seam finding** beside Laser scanning. Enable healthy robot control,
then press **Find weld seam**. One service request prepares the shared camera and
moves to the saved **Weld Finder** pose. A fresh calibrated laser cross-section
must show a supported valley or sharp discontinuity before tracking begins.
The minimum total explored scan span defaults to **25 mm**, split into a measured
minimum of **12.5 mm per direction**. The service follows both
directions in 1 mm stations, recenters the feature, and corrects the sensing
distance toward **150 mm**. Three fresh, good-quality cross-sections without the
feature confirm an endpoint. Missing or ambiguous laser data stop the operation
with a reason if bounded optical recovery fails. Recovery holds the arm at the
same measured pose and tries exposures within the actual sensor/frame-rate
bounds, then bounded gain adjustments if needed, retaining controls that resolve the seam. Native control changes
restart quiet/fresh-frame qualification and remain recorded with their generation.
The explored span comes from measured camera displacement perpendicular to the
laser scan plane and is reported separately from weld length. Travel is bounded;
unconfirmed endpoints produce an explicitly incomplete result. The other search
direction is still explored after bounded optical recovery fails at one end.

The camera-only SDK API `weldfinder.seam.find_weld_seam` returns the seam centroid,
two fitted flanks and a bisector torch reference in camera millimetres. AutoWeld
registers fresh stationary observations using measured robot poses and the
installed hand-eye transform. The panel draws **green** seam geometry and **red**
cross-section lines, retained after camera stop. Native observations, transforms,
identities and an ordered path are saved under `<runtime>/weld-paths/`.

**Replay weld path** is an explicit service operation. It checks the saved path
and installed calibration/model, preflights the normal motion plans, then follows
the recorded torch references with the installed camera optical frame. Reference
+Z points toward the seam from 150 mm away. This fixture has a receiver/camera;
a physical torch requires its own measured tool transform. Replay does not
activate welding output. An incomplete result can replay its continuous measured
section, with unconfirmed endpoints clearly reported. Operator Hold, cancellation, shutdown and cooldown keep
their normal priority.

```bash
.venv/bin/python scripts/request.py --runtime runtime/agents-live --payload '{"action":"weld_start","min_distance_mm":25,"start_pose":"Weld Finder"}'
.venv/bin/python scripts/request.py --runtime runtime/agents-live --payload '{"action":"weld_state"}'
.venv/bin/python scripts/request.py --runtime runtime/agents-live --payload '{"action":"weld_cancel"}'
# Explicit replay after a completed path:
.venv/bin/python scripts/request.py --runtime runtime/agents-live --payload '{"action":"weld_replay"}'
```

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
playwright-cli -s=autoweld-smoke run-code --filename=scripts/smoke_scanning_ui.js
playwright-cli -s=autoweld-smoke close
```

Close the automation browser after checks or scan submission. Headless software
WebGL can delay camera and SDK observations; the service continues scanning and
retaining its cloud independently of browser connections. Keep the mobile Viser
link available for inspection.

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
