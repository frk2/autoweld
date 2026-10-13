# Large-path repeat with selected onboard Kp

User request: repeat the larger line and circular experiments to document path adherence with the newly selected gains. This is the same reviewed 500 mm Y line and R60 mm YZ full circle, with unconstrained orientation and ramps inside the requested edge. It does not repeat the smaller welding-oriented fixtures.

- Line: 20, 30, 40, 50, 60, 80, 100 mm/s. The request renews the prior 1.55 rad automatic-test exception for this exact 500 mm fixture during this sweep. It is restored to 0.49 rad after the line series. All circle tests use 0.49 rad.
- Circle: 5, 10, 20, 30, 40, 50, 60, 80, 100 mm/s preflight. The last feed remains rejected by the retained acceleration/jerk/feed-length check and will not be sent.
- Read actual onboard KP_APR before any test: J1–J6 `[193.49176025390625,193.49176025390625,193.49176025390625,54,77.76000213623047,64.80000305175781]`. Preserve all actuator registers, Flash, host settings and saved rest. Startup reads back the gains again before each run. No gain tuning is part of this repeat.
- SDK-only commands, unchanged oscillation/communication/100 ms watchdog, joint/collision/derivative checks, and fresh/idle-start gates. Finite-edge deviation above 10 mm is reporting-only; complete the reference. Timed position error is informational. The selected safe presets remain line 20 and circle 10 mm/s.
- Adaptive temporary POS_VEL speed ceiling is derived from planned joint peak speed, as in the earlier tests. Restore 0.15 rad/s before every normal return, verify rest within 0.5° for one second, and disable between runs.
- Archive the standard reference-plus-one-second trace and an additional endpoint hold (at least three seconds inside 0.005 rad error, or a labelled 30-second observation timeout). The extra observation gate never replaces or prevents the SDK's normal verified-rest disable.
- No heavy offline plots or bulk tests with motors enabled. Camera remains available; previously authorized camera-check skipping applies.

Velocity uses all six encoder feedback timestamps aligned to a common observed time, FK, and TCP Jacobians. Line speed is signed displacement along the line. Circle displacement is the signed integral of measured radius times angular displacement, avoiding a false speed increase from assuming the requested radius during corner cutting. Raw reported joint velocity projected through the Jacobian is plotted separately.

Each 0.2–1.0 s running velocity uses displacement divided by the exact full window duration; compare it with reference displacement over the identical interval. Reject windows crossing stale feedback (>50 ms), inter-joint skew (>10 ms), non-increasing joint timestamps, or gaps (>50 ms). Plot averages at window midpoints and label latest valid cruise averages. Separately report whether the exact final cruise window is valid, valid/available window counts and speed-band fractions; missing windows are not passing results. A longer average smooths noise but does not prove instantaneous speed regulation.

Before/after comparisons use the same STEP assets, seed targets, feeds and ramp policy from the manifest. Earlier complete line 20–60 and circle 5–30 traces used the historical 5 mm stop policy but completed without that stop; line 80 and circle 40 baselines use the subsequent complete 10 mm reporting-only repeats. There is no earlier completed baseline for line 100 or circle 50–80. These are single-run historical comparisons, not randomized repeated causal trials. All deviations are encoder/FK estimates; no external tip metrology or force-compliance measurement is available.

Reproduction:

1. With the app disabled, run `scripts.path_deviation_preflight` using this directory's manifest/output paths. Independently survey the disabled actuators through SDK register APIs with exclusive CAN ownership.
2. Start the SDK app disabled with the line-specific `--long-cruise-test` flag. Run `python -m scripts.kp_path_repeat --suite lines`, then `--suite circles` after the runner restores the usual excursion bound.
3. With motors disabled, run `python -m scripts.plot_kp_path_repeat`.

Register-access safety reference: [i2rt SDK register documentation](https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/motor_config_tool/dm_motor_registers.md). All standalone register surveys stop the app first and hold its exclusive CAN lock; no register writes or Flash saves are needed for this repeat.
