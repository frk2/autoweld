# Installed J6 travel calibration

The operator identified the existing pose as J6 maximum, then manually positioned
minimum with all six motors disabled, then returned the arm to supported rest
inside the maximum. SDK-only readouts confirmed disabled status for IDs 1–6 at
all three captures. No motor enable, motion, encoder-zero, mode change, error clear
or Flash write was issued during this calibration.

The three fresh packed SDK samples at each endpoint were cross-checked against
the native `p_m` and `xout` read-only registers. The command/feedback coordinate
used for limits is the median packed SDK joint position, corroborated by `p_m`;
`xout` is retained as separate evidence and is not mixed into those coordinates.

- Physical minimum: −5.390058747234303 rad (−308.8276°).
- Physical maximum: −2.069695582513161 rad (−118.5848°).
- Operating range: −307.8276° through −119.5848°, with 1° inward clearance.
- New supported rest J6: −2.274166475928894 rad (−130.3001°).
- The SDK read-only startup preflight passed at the new rest; power stayed off.

`assets/camera_mount/joint-calibration.json` records the measured endpoints and
fixed zero motor offsets. MuJoCo, URDF, planner, controller and SDK use the same
operating range. Explicit SDK offsets preserve J6's negative encoder branch at
startup and reconnect instead of adding 2π below −π. Overrides are validated
before constructing the SDK motor chain. A separate read-only preflight rejects
an endpoint pose before enabling; an old captured rest cannot expand J6 limits.
Gravity selection is rejected within 2° of an operating boundary; while active,
approaching that margin (including a 0.2 s velocity projection) switches to
powered Hold. This supplements the SDK's existing feedback monitor without
lowering other joints' feedback tolerance or powering motors off.

Only the joint range changed in the compiled model. Gravity profile fingerprints
were migrated with provenance because geometry, axes, inertials and payload
were unchanged. Previous gravity tuning remains limited; no broader gravity
qualification is claimed for this newly measured interval.

Raw snapshots: `maximum-read-only.json`, `minimum-read-only.json`,
`rest-read-only.json`. `prior-rest.json` retains the previous rest record.
These are manual endpoint measurements with encoder quantization, rather than
metrology-certified hard-stop angles. No powered endpoint sweep was performed.

Validation: 522 tests passed (`tests.txt`); Ruff and compilation checks passed.
A fresh headless browser connected to Viser and confirmed Disabled, saved-rest
feedback labeling, J6 −2.2742 rad, speed fields 5 m/s and 1 rad/s, and camera Off.
The test browser was closed. The service remains available on port 8082 with
motors disabled. No powered test of the newly calibrated range was performed.
