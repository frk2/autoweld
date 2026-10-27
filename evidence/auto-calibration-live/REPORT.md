# Auto calibration repair and live validation

The live arm was already at saved `Calibration Pose`, with a fresh 1440×1080
FLIR stream and all 35 configured 5×7 inner corners detected. The target has
4 mm squares. Baseline preview and native-coordinate detections are retained
in `baseline.jpg`, `baseline-camera.json`, and `baseline-status.json`.
Temperatures were 29–32°C; cooldown was inactive.

The corrected cone has 25 unique, uniformly spaced camera origins, 150 mm height,
15° full opening / 7.5° half-angle, and 19.7479 mm radius. Every camera +Z axis
points at one fixed apex. There is no baseline capture, depth/radius modulation,
or added roll. Movement to the first edge point now precedes the first capture.
The UI explicitly uses pixel-width lines; the previous default interpreted
thickness 2 as 2 metres.

Offline IK preflight from the live calibration pose succeeded for all 25 views.
Maximum joint excursion was 0.39224 rad, with minimum joint margin 0.08108 rad.
`cone-preflight.json` retains the points and joint solutions.

Automatic calibration now uses a local 80 mm / 20° / 0.49 rad envelope around its
own baseline, retaining SDK motion, joint and collision checks. The old
rest-relative commissioning envelope could reject every move at this pose.
Cooldown cancels calibration without calling Hold over the controller's rest
return. The 60°C trigger and below-55°C resume threshold remain unchanged.

A service reload exposed a pending-disable race in the old running service.
A filesystem shutdown received while the arm was moving latched automatic
disable. When rest return was rejected by the 0.49 rad test limit, the pending
flag remained set and UI controls stayed disabled. SDK feedback confirmed fresh
POS_VEL powered hold. No process was killed and no second CAN connection was
opened. Safe external power handling was requested before replacing the old
process; a separate code fix and regression tests address this state failure.

Native intrinsics assessment is available through `scripts/assess_intrinsics.py`.
It exports exact stored PNGs and checks per-view errors, two-axis tilt excitation,
spatial coverage, parameter uncertainty, and held-out views. Production fit/save
thresholds have not been relaxed.

Status: geometry and offline checks are complete. Physical vector hunting,
25-view collection, installation and held-out validation are pending the reload.
No acceptable live intrinsic result is claimed yet.

The pending-disable fix now preflights before setting the motion barrier, clears
that barrier on a later rejected return while refreshing powered Hold, and
permits a normal operator retry to take ownership without expanding the
0.49 rad automatic test envelope. It never disables motors after a failed return.

Final offline validation: 554 tests passed; Ruff and compilation checks passed.
The full-suite run also exposed and repaired a test race that waited on a private
camera future after dispatch had already consumed it. The singleton test now
waits for the public enabled state. No production camera ownership gate changed.
The live service still runs the previous code and remains in SDK powered hold,
awaiting the operator's safe power-off confirmation for replacement.

The operator subsequently authorized service recovery without external support
when the arm is near rest; this rule is saved in AGENTS.md. Fresh SDK feedback
at that point still placed the arm at Calibration Pose, with J1 about 107° and
J2 about 63° from captured rest. The near-rest condition was therefore not
established. Clarification was requested on whether this particular current pose
is also physically safe for an unsupported restart. No restart or motor disable
was performed while that condition remained unresolved.

Recovery update: the arm has now returned to captured rest under continuous SDK
power, passed sustained fresh rest verification, and been disabled normally at
rest. The old stuck process was killed only then. The fixed service is restarted
and the shutdown latch is clear. See `RECOVERY.md` and
`powered-recovery-result.json` for physical evidence. Intrinsic calibration and
live hunting remain pending; the latest requested final arm state is rest.
