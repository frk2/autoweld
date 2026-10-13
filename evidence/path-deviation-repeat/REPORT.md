# Path-deviation repeat — 20 September 2026

**16 paths completed; 4 were flagged as deviating and continued. Two other fixtures failed joint-bound preflight and were not sent.** The final state is Position–speed selected, motors disabled at verified rest, maximum joint error **0.044°**, no SDK fault, joint ceiling restored to 0.15 rad/s and automatic excursion bound restored to 0.49 rad.

Finite-edge deviation above **10 mm** now marks the trajectory without commanding Hold. Constrained world-orientation error above **5°** is also record-only. The SDK communication, 100 ms execution/feedback watchdog, oscillation, joint/collision, derivative and fresh/low-speed start checks remain active. User Hold and disable still interrupt. Timed position error remains informational. See [policy and UI behavior](../../docs/PATH_DEVIATION.md).

The [source inventory](guard-stop-inventory.json) maps 26 previously guard-stopped traces to [18 unique fixtures](manifest.json). All eight distinct POS_VEL guard-stopped fixtures ran again, as did eight of ten older MIT fixtures using the selected POS_VEL controller. Duplicate historical runs share one repeat. Original speeds, STEP geometry, orientation requests and ramp placement were retained; fresh preflight selected a sufficient host POS_VEL joint ceiling. This is not a controlled before/after controller comparison.

## Results and plots

Every completed path includes the entire reference plus a one-second hold. The [settling overview](plots/settling-overview.png) and archived `*-settling.jsonl` additionally cover the subsequent observations before return to rest. Final commanded-target error after that settling ranged from **0.051 to 0.258 mm** by encoder/FK. This does not make the preceding path accurate: transient edge errors remain included. Outside-edge ramps intentionally finish beyond the finite edge endpoint, so final-target error and edge-endpoint error differ.

The table's edge and orientation maxima cover the standard full trace. Additional settling introduced no larger edge maximum. All 16 fail the unchanged raw ±20% feed criterion; none qualifies Stage 3. “No geometric flag” is not an overall pass. Orientation maxima shown below apply only to constrained paths.

| Fixture | Feed mm/s | Max edge mm | Max world orientation ° | Recorded geometric flag | Final target error after settling mm | Plots |
|---|---:|---:|---:|---|---:|---|
| Center-facing R12.7 mm / 5° arc | 1 | 19.336 | 4.074 | position | 0.051 | [path](plots/repeat-arc-04-r12.7-a5-1mms-trial1-attempt1-tracking.png) · [windows](plots/repeat-arc-04-r12.7-a5-1mms-trial1-attempt1-windows.png) |
| Center-facing R200 mm / 5° arc | 5 | 6.825 | 3.016 | none | 0.079 | [path](plots/repeat-arc-10-r200-a5-5mms-trial4-attempt1-tracking.png) · [windows](plots/repeat-arc-10-r200-a5-5mms-trial4-attempt1-windows.png) |
| perpendicular-line-02-20mms | 20 | 7.474 | 3.150 | none | 0.175 | [path](plots/repeat-perpendicular-line-02-20mms-attempt1-tracking.png) · [windows](plots/repeat-perpendicular-line-02-20mms-attempt1-windows.png) |
| perpendicular-line-03-20mms | 20 | 9.332 | 5.055 | orientation | 0.122 | [path](plots/repeat-perpendicular-line-03-20mms-attempt1-tracking.png) · [windows](plots/repeat-perpendicular-line-03-20mms-attempt1-windows.png) |
| perpendicular-line-04-20mms | 20 | 10.923 | 6.112 | position + orientation | 0.107 | [path](plots/repeat-perpendicular-line-04-20mms-attempt1-tracking.png) · [windows](plots/repeat-perpendicular-line-04-20mms-attempt1-windows.png) |
| perpendicular-line-05-20mms | 20 | 10.212 | 5.594 | position + orientation | 0.102 | [path](plots/repeat-perpendicular-line-05-20mms-attempt1-tracking.png) · [windows](plots/repeat-perpendicular-line-05-20mms-attempt1-windows.png) |
| Unconstrained R60 mm circle | 40 | 6.588 | — | none | 0.147 | [path](plots/repeat-circle60-40mms-attempt1-tracking.png) · [windows](plots/repeat-circle60-40mms-attempt1-windows.png) |
| Older circles-01-5mms | 5 | — | — | Joint-bound preflight rejected | — | [rejection](preflight.json) |
| Older circles-02-5mms | 5 | — | — | Joint-bound preflight rejected | — | [rejection](preflight.json) |
| Older circles-03-5mms | 5 | 1.102 | — | none | 0.228 | [path](plots/legacy-circles-03-5mms-attempt1-tracking.png) · [windows](plots/legacy-circles-03-5mms-attempt1-windows.png) |
| Older circles-04-5mms | 5 | 1.292 | — | none | 0.203 | [path](plots/legacy-circles-04-5mms-attempt1-tracking.png) · [windows](plots/legacy-circles-04-5mms-attempt1-windows.png) |
| Older circles-05-5mms | 5 | 1.385 | — | none | 0.147 | [path](plots/legacy-circles-05-5mms-attempt1-tracking.png) · [windows](plots/legacy-circles-05-5mms-attempt1-windows.png) |
| Older lines-01-5mms | 5 | 1.540 | — | none | 0.102 | [path](plots/legacy-lines-01-5mms-attempt1-tracking.png) · [windows](plots/legacy-lines-01-5mms-attempt1-windows.png) |
| Older lines-03-5mms | 5 | 1.441 | — | none | 0.191 | [path](plots/legacy-lines-03-5mms-attempt1-tracking.png) · [windows](plots/legacy-lines-03-5mms-attempt1-windows.png) |
| Older lines-04-5mms | 5 | 1.418 | — | none | 0.070 | [path](plots/legacy-lines-04-5mms-attempt1-tracking.png) · [windows](plots/legacy-lines-04-5mms-attempt1-windows.png) |
| Older lines-05-5mms | 5 | 1.549 | — | none | 0.258 | [path](plots/legacy-lines-05-5mms-attempt2-tracking.png) · [windows](plots/legacy-lines-05-5mms-attempt2-windows.png) |
| Older perpendicular-lines-05-5mms | 5 | 4.097 | 2.298 | none | 0.083 | [path](plots/legacy-perpendicular-lines-05-5mms-attempt1-tracking.png) · [windows](plots/legacy-perpendicular-lines-05-5mms-attempt1-windows.png) |
| Unconstrained 500 mm line | 80 | 5.752 | — | none | 0.176 | [path](plots/repeat-line500-01-80mms-attempt1-tracking.png) · [windows](plots/repeat-line500-01-80mms-attempt1-windows.png) |

## What the completed traces reveal

- The **R12.7 mm / 5° arc at 1 mm/s** travels away from the requested arc before returning: maximum finite-edge distance **19.336 mm**, far larger than its approximately 1.108 mm requested arc length. The old stop hid most of this excursion. World-reference orientation error remains below 5°, but actual torch-to-center bearing error reaches **61.66°** because the tip itself has moved away from the circle. After additional settling, target error falls to 0.051 mm. Endpoint convergence does not validate the intervening welding path.
- The **R200 mm / 5° arc at 5 mm/s** completes with **6.825 mm** maximum edge distance and 3.016° world-reference orientation error. Its actual torch-to-center bearing error nevertheless reaches **7.80°**. The plotted actual bearing metric is separate from the unchanged reference-world-pose criterion. Final target error after settling is 0.079 mm.
- The **20 mm/s perpendicular lines** reach **7.474–10.923 mm** edge distance. Lines 03–05 exceed 5° world-orientation error; lines 04–05 also exceed 10 mm position deviation. All finish rather than stopping at the crossing. The first one-second hold is insufficient for full physical convergence: the recorded endpoint errors at that point are 4.40–6.44 mm, falling to 0.102–0.175 mm after further settling.
- The **R60 mm circle at 40 mm/s** completes with **6.588 mm** maximum edge distance. Its latest valid full 1 s cruise average is **35.45 mm/s** against a matching 40 mm/s command average.
- The **500 mm line at 80 mm/s** completes with **5.752 mm** maximum edge distance. Latest valid cruise averages are **78.53 mm/s over 0.5 s** and **78.34 mm/s over 1.0 s**, against 80 mm/s. Maximum timed lag is **61.343 mm**. Endpoint error is still 10.06 mm at the end of the standard trace, then falls to **0.176 mm** after the additional settling observations. Higher steady average speed does not remove phase lag or establish the raw-speed criterion.
- Older unconstrained 5 mm/s fixtures that passed preflight stay within **1.102–1.549 mm**. Their latest full 1 s cruise averages are approximately **4.79–5.03 mm/s**. The older perpendicular line finishes within **4.097 mm** and 2.298° world-orientation error.

The completed maxima must not be compared directly with old truncated maxima as though the controller became worse: the earlier traces ended at their guard crossing. Historical MIT traces also used a different controller and, in some runs, different position criteria.

## Measurement and coverage

[Aligned metrics](metrics.json) and 32 per-run figures include full finite-edge error, requested versus encoder/FK paths, world-orientation error, actual tool-normal/bearing error for constrained fixtures, geometric phase, timed lag, raw tangential velocity and nine running-average windows from **0.2 to 1.0 s**. Matching command averages use the identical window. Numerical latest-cruise averages are marked on each window chart. If no complete valid cruise window exists, the chart explicitly says so; the short 20 mm/s lines have no full 0.5/1.0 s cruise window.

Joint positions and reported velocities are aligned by their per-motor feedback timestamps before FK/Jacobian calculation. Line averages use signed along-line displacement. Circle averages integrate actual in-plane tangential displacement (measured radius × angular change), rather than multiplying by the nominal radius. Windows crossing stale/duplicate/widely separated feedback are omitted. Window plots show the complete range of valid averages, with raw outliers potentially outside that zoom; the companion tracking plots show the full raw range. Some standard traces also contain duplicate initial elapsed-zero observations, which conservatively fail `sampling_ok`; it was not relaxed. Invalid velocity sample counts remain in [coverage](coverage.json).

[Settling metrics](settling-metrics.json) use the separately archived SDK control telemetry after the standard trace, ending before the speed-limit reset and rest-return command. These observations were already recorded during acquisition; no extra live motions were needed for this analysis. The standard trace and subsequent settling are visibly distinguished in the overview. All final-target errors are kinematic estimates from encoders with the placeholder 150 mm TCP. There is no external tip metrology or applied-force measurement, so this is tracking data, not measured mechanical compliance.

Two older circles fail whole-path joint bounds at **41.2%** and **49.8%** in the standard seed preflight. Replanning from their historical start targets still fails at **35.8%** and **40.2%**; see [alternate preflight](alternate-preflight.json). This is a local planning rejection, not proof of geometric unreachability. No rejected path was issued, and no joint limit was relaxed.

The 500 mm / 80 mm/s repeat alone used the same previously reviewed **1.55 rad automatic-test exception**, renewed by the request to repeat all stopped tests. Its full preflight span is 1.498 rad and maximum rest excursion 0.931 rad. Every other run used 0.49 rad. The temporary bound was restored after normal verified-rest disable. Selected feed presets remain **20 mm/s lines / 10 mm/s arcs**.

## Execution and validation

- **192 software tests passed**, including exact 10 mm boundary behavior and an app regression proving geometric deviation continues while SDK faults interrupt. [Test output](tests.log). Ruff passes production, tests and the new evidence scripts.
- No SDK communication, oscillation or execution-watchdog fault occurred during these repeats. Each executed fixture ended with a normal verified-rest disable.
- Older line 05 initially received the unchanged SDK rejection “Arm moved during validation; wait for it to stop.” No tracking command ran on that attempt. After normal rest disable, a repeat with four seconds of start settling completed. Both attempt records are retained.
- An initial runner read saw a stale UI publication during reconnect; the runner now waits for the fresh post-request publication. No test motion occurred before that correction, and the enabled arm was returned and disabled normally. [Startup rest record](startup-publication-rest.json).
- [Settings validation](settings-validation.json) confirms captured rest/gripper file, controller selection, cached MIT gains/integral/gravity/friction settings, operator motion limits and selected feed presets are preserved. No onboard gains, ACC/DEC/TIMEOUT or Flash values were changed. The new 10 mm record-only policy is active.
- Viewer remains on port **8081**, Position–speed selected and disabled, Enable available. [Final SDK/app status](final-rest-status.json). No stage is marked DONE.

## Reproduction

Run preflight, tests and analysis with motors disabled. Live runner use requires operator authorization and the same independently checked geometry and environment.

```bash
.venv/bin/python -m scripts.path_deviation_preflight
.venv/bin/python -m scripts.path_deviation_repeat --index 0
MPLCONFIGDIR=/tmp/autoweld-matplotlib .venv/bin/python -m scripts.path_deviation_settling
MPLCONFIGDIR=/tmp/autoweld-matplotlib .venv/bin/python -m scripts.plot_path_deviation_repeat
```

The live runner refuses to overwrite existing evidence; a deliberate new attempt needs `--attempt N`. The exact long-line fixture additionally requires the dedicated long-cruise startup option and verifies restoration to the normal automatic bound after disable. [Coverage manifest](coverage.json) links every historical stopped trace to its completed result or exact preflight rejection.

