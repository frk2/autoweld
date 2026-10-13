# Integral-only live tuning — 2026-09-14

Historical report. The later user-authorized P/I comparison with fixed D and 0.06° deadband is in the [current PI report](../../pi/REPORT.md).
Fast, reliable six-joint convergence to 0.1° is **not achieved**. Factory P/D remained unchanged throughout these trials. The camera was restored and the complete arm was visible. All automatic waypoints stayed within 0.1 rad of the captured rest; commands used the existing app and SDK connection.

## Current configuration

- Factory Kp: `[80, 80, 80, 10, 10, 10]`; Kd: `[5, 5, 5, 1.5, 1.5, 1.5]`. Original PD function is unchanged.
- Last tested experimental Ki: `[40, 40, 60, 5, 5, 1]` Nm/(rad·s), saved as the UI/script candidate. This is not an accepted final profile.
- Integral deadband: 0.09°. Integral caps: ±5 Nm on J1–J3 and ±1 Nm on J4–J6; gripper I is zero.
- I resets on changed targets and waits 0.5 s before integration; identical repeated targets preserve I. Factory gravity/friction compensation remain enabled.
- Viewer at port 8081 is left in PD with motors disabled at verified rest. Maximum last measured arm/rest difference: 0.1311°. Disable verified all seven joints within 0.5° for one second.

## Findings

Factory PD baseline left approximately 1.87° elbow error on a +0.08-rad command. I substantially reduced this error, but settling depends on direction and prior motion. In the last isolated elbow trial, the elbow stayed within 0.1° after approximately 17 s and finished within 0.072°. Joint 4 intermittently read 0.109°, so the full six-joint criterion failed.

With Ki6=0, an already-settled positive target held all joints inside 0.1° for a 12-second observation. This was a hold, not a new movement or a zero-time convergence result. The same profile took approximately 28.48 s to settle on return and then left about 0.54° wrist error in the opposite direction. Stronger wrist I produced repeated overshoot/hunting instead of reliable convergence.

Late drift matters: several trials first entered the tolerance band and subsequently left it. The table reports the beginning of the final uninterrupted in-band interval only when at least one second remains. A dash means that criterion did not pass within the recorded window. Final two-second peaks can include a failure before the final uninterrupted interval starts. These measurements do not establish jerk-free behavior or absolute physical accuracy.

## Trial evidence

Paths below link to per-trial summaries; matching `.jsonl` files contain approximately 50-Hz encoder and integral-torque samples. Earlier trials precede the new-target reset, delay, or 0.09° deadband; see the chronology below.

| Trial | Ki J1–J6 | Duration (s) | Final sustained all-joint 0.1° start (s) | Worst last-2-s error (°) |
|---|---|---:|---:|---:|
| [factory-pd-baseline](../factory-pd-baseline.json) | 40, 40, 60, 5, 5, 5 | 6 | — | 1.873 |
| [factory-i60-elbow](../factory-i60-elbow.json) | 40, 40, 60, 5, 5, 5 | 15 | 7.83 | 0.087 |
| [factory-i60-w8-return](../factory-i60-w8-return.json) | 40, 40, 60, 8, 8, 8 | 15 | — | 0.109 |
| [factory-i80-w12-all](../factory-i80-w12-all.json) | 60, 60, 80, 12, 8, 8 | 15 | — | 0.618 |
| [factory-i30-w6-return](../factory-i30-w6-return.json) | 40, 40, 30, 6, 6, 6 | 25 | 16.06 | 0.087 |
| [factory-db09-all](../factory-db09-all.json) | 40, 40, 40, 8, 5, 3 | 20 | — | 0.418 |
| [factory-diag-w6-off](../factory-diag-w6-off.json) | 40, 40, 40, 8, 5, 0 | 12 | — | 0.106 |
| [factory-w6-i1-hold](../factory-w6-i1-hold.json) | 40, 40, 40, 8, 5, 1 | 15 | 7.37 | 0.084 |
| [factory-gentle-return](../factory-gentle-return.json) | 40, 40, 20, 6, 5, 1 | 25 | — | 0.131 |
| [factory-gentle-all](../factory-gentle-all.json) | 40, 40, 20, 6, 5, 1 | 30 | — | 0.565 |
| [factory-reset-all](../factory-reset-all.json) | 40, 40, 30, 6, 5, 2 | 25 | — | 0.368 |
| [factory-delayed-all](../factory-delayed-all.json) | 40, 40, 30, 6, 5, 1 | 25 | — | 0.728 |
| [factory-final-candidate-hold](../factory-final-candidate-hold.json) | 40, 40, 30, 6, 5, 0 | 12 | 0.00 | 0.084 |
| [factory-final-return](../factory-final-return.json) | 40, 40, 30, 6, 5, 0 | 30 | 28.48 | 0.109 |
| [factory-final-opposite](../factory-final-opposite.json) | 40, 40, 30, 6, 5, 0 | 30 | — | 0.543 |
| [factory-refine-opposite-hold](../factory-refine-opposite-hold.json) | 40, 40, 60, 10, 5, 5 | 30 | — | 0.681 |
| [factory-isolated-setup-rest](../factory-isolated-setup-rest.json) | 40, 40, 60, 5, 5, 1 | 5 | — | 0.350 |
| [factory-isolated-elbow-validate](../factory-isolated-elbow-validate.json) | 40, 40, 60, 5, 5, 1 | 30 | — | 0.109 |

The initial baseline and I60 trials used the earlier 0.075° deadband. `factory-db09-all` begins the 0.09° trials; `factory-reset-all` introduces I reset on changed targets; `factory-delayed-all` and subsequent trials use the current 0.5-second I delay. `factory-final-candidate-hold` and `factory-refine-opposite-hold` change gains while retaining the same target. `factory-isolated-setup-rest` is factory PD.

## Validation and remaining work

53 unit tests pass, including original-PD source equality, rejection of P/D overrides, integral state transitions, windup, stale feedback, deadband, gripper exclusion and activation delay. App/scripts/tests Ruff passes. The tracked SDK patch reverse-application check passes. The current-code offline end-to-end smoke passed and ended with simulation disabled. Offline dynamics are ideal and do not establish live convergence. The live browser shows read-only factory P/D, editable Ki/deadband, PD selection and Enable motors available.

A proposed 0.06° deadband trial **did not run**: automatic approval review rejected it because the local AGENTS.md records 0.09°. The proposed change would leave room for encoder quantization below 0.1° but would integrate more small errors, potentially increasing hunting. It requires approval before testing. No 0.06° result is claimed.

Evidence: [final state](final-state.json), [verified rest camera](verified-rest.jpg), [restored camera](camera-restored.jpg). The old `deployed-state.json` and `camera.jpg` in this directory are retained from the preceding camera-blocked run. The higher-P report in the parent directory remains superseded. Stage 1 is not DONE.
