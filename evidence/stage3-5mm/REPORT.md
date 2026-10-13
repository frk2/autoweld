# Stage 3 repeat at 5 mm

The temporary global TCP tracking tolerance is now **strictly below 5 mm**, shared by the path-start gate, execution Hold guard, telemetry, UI and pass/fail summary. This supersedes the earlier 1 mm commissioning target. Orientation remains within 5 degrees and measured on-edge feed within 20% of the requested 5 mm/s. Numerical IK/interpolation tolerances, joint accuracy targets, rest verification and automatic-test bounds were not relaxed.

## Validation

- Full hardware-free suite: **141 passed**. Ruff passes. Boundary tests cover 4.999, 5.000 and 5.001 mm at both tracking evaluation and command entry.
- Live app restarted with the new threshold. Selected PID gains, linear gravity/factors, motion limits and friction-off configuration retained.
- All executed motions use the SDK, with sampled full paths including lead-in/out checked below 0.49 rad from current/rest/prior target, the SDK oscillation detector active, and no camera checks as requested.
- Tests use nominal encoder FK and Jacobian velocity, not external absolute TCP metrology. The torch geometry/base have not been physically calibrated. Speed figures below use every raw on-edge feedback sample; no smoothing was used to obtain a pass.

## Live results

Each completed run includes one second of final hold. A stopped path has only partial coverage and cannot pass. The five circles have radius 12.7 mm; straight lines are 20 mm. Perpendicular fixtures were placed with the initial tool axis perpendicular to their direction.

| Fixture | Completed | Max timed error (mm) | Max edge distance (mm) | Max orientation error (deg) | Max feed deviation (%) | Qualified |
|---|---|---:|---:|---:|---:|---|
| lines 1 | Yes | 1.637 | 1.517 | 0.314 | 490.3 | No |
| lines 2 | Yes | 2.920 | 2.587 | 0.396 | 461.8 | No |
| lines 3 | Stopped | 5.077 | 3.145 | 0.440 | 70.5 | No |
| lines 4 | Stopped | 5.048 | 3.740 | 0.551 | 338.8 | No |
| lines 5 | Yes | 3.468 | 3.450 | 0.654 | 521.2 | No |
| circles 1 | Stopped | 5.089 | 4.835 | 0.590 | 448.4 | No |
| circles 2 | Stopped | 5.087 | 4.794 | 0.534 | 503.5 | No |
| circles 3 | Stopped | 5.153 | 4.851 | 0.582 | 429.1 | No |
| circles 4 | Stopped | 5.073 | 4.810 | 0.610 | 460.4 | No |
| circles 5 | Stopped | 5.003 | 4.757 | 0.624 | 441.4 | No |
| perpendicular-lines 1 | Yes | 2.678 | 1.898 | 0.472 | 619.3 | No |
| perpendicular-lines 2 | Yes | 2.965 | 1.566 | 0.570 | 724.3 | No |
| perpendicular-lines 3 | Yes | 3.034 | 2.666 | 0.398 | 628.7 | No |
| perpendicular-lines 4 | Yes | 2.936 | 2.774 | 0.475 | 307.8 | No |
| perpendicular-lines 5 | Stopped | 5.008 | 4.608 | 0.587 | 371.6 | No |

`summary.json` also records feed percentiles and the fraction of raw speed samples within tolerance. Unconstrained trials log orientation for information only. Error-triggered Hold occurs on the first observed out-of-tolerance sample; this is a sampled guard, not a guarantee that the physical peak never exceeds 5 mm.

The first line's initial approach was rejected because the arm moved during planning. The healthy arm was allowed to settle, and that trial was repeated successfully through completion. Its earlier `-error.json` is preserved separately.

## Center-facing quarter-arcs: five rejected preflights

All five 90-degree, 12.7 mm-radius center-facing fixtures were attempted in the model and rejected for a joint step above 0.05 rad at 2.2% of the extended path. No robot commands were issued for these arcs. A denser diagnostic continuation with 251 points also exceeded the automatic 0.49 rad rest-relative excursion limit on J5 around 8.6% of the requested arc, then failed IK around 42–44%. This does not prove other placements unreachable. A larger allowed workspace or different feasible placements/orientation-roll strategy would need a new validated preflight; bypassing the rejection is not a test.

Details: `oriented-preflight.json`, `center-arc-diagnostics.json`. CAD fixtures and their generator are in `assets/stage3/` and `scripts/build_oriented_fixtures.py`.

## Evidence and reproduction

- One JSON summary and JSONL measured trace per live trial in this directory.
- `lines-tracking.png`, `circles-tracking.png`, `perpendicular-lines-tracking.png` and matching SVGs plot tracking error and raw feed against the planned feed/ramp.
- Runner: `.venv/bin/python -m scripts.stage3_trial --suite lines --index 0 --evidence-dir evidence/stage3-5mm` (live, requires a healthy enabled controller; other suites: circles, perpendicular-lines, center-arcs).
- Plotting: `MPLCONFIGDIR=/tmp/autoweld-mpl .venv/bin/python -m scripts.plot_tracking --evidence-dir evidence/stage3-5mm --pattern 'lines-*.jsonl' --output lines-tracking`.

The tolerance change permits more complete tracking runs, but it does not establish feed regulation. Stage 3 remains unqualified; no gain retuning or further tolerance relaxation was performed.

Final state: motors disabled, verified rest True, maximum joint error 0.262 degrees. SDK fault: None; motion fault: None. UI remains available on port 8081.
