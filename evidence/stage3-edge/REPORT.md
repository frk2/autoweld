# Stage 3: edge-distance acceptance and speed repeat

The adjacent-IK 0.05 rad knot-step gate is removed. Path planning still requires converged IK, joint bounds, robot self-collision checks, interpolation accuracy, and sampled TCP/joint velocity, acceleration and jerk limits. Automatic commands retain the full-path <0.49 rad excursion envelope. SDK oscillation and watchdog guards remain active. No selected gains, gravity/friction settings or motion limits changed.

Position qualification and execution Hold now use **distance to the requested finite edge, at most 5 mm** throughout lead-in, edge traversal, lead-out and final hold. Timed TCP error is recorded only, even if above 5 mm. The start gate uses edge distance plus the existing 0.01 rad joint match to the planned start. Orientation remains within 5 degrees for constrained paths and speed remains within ±20% of the requested 5 mm/s on the edge.

The full software suite passed **145 tests**, including the inclusive 5 mm edge boundary, large timed error without rejection, and retained bounds/collision/derivative guards. Ruff passed. The live app was restarted and reported `tracking_position_metric: edge_distance` before testing.

## Live results at 5 mm/s

Encoder FK and Jacobian velocity are used; these are not independent physical TCP measurements. The modeled torch/base remain uncalibrated. A completed run includes one second of final hold. Raw speed is neither smoothed nor omitted for qualification.

| Fixture | Completed | Max edge distance (mm) | Max timed error, informational (mm) | Max orientation error (deg) | Peak feed deviation (%) | Qualified |
|---|---|---:|---:|---:|---:|---|
| lines 1 | Yes | 2.375 | 1.728 | 0.314 | 491.2 | No |
| lines 2 | Yes | 3.442 | 3.782 | 0.420 | 582.6 | No |
| lines 3 | Yes | 2.953 | 5.567 | 0.614 | 962.7 | No |
| lines 4 | Yes | 4.768 | 6.348 | 0.739 | 1151.1 | No |
| lines 5 | Yes | 3.089 | 3.091 | 0.676 | 440.0 | No |
| circles 1 | Stopped | 5.096 | 5.322 | 0.650 | 437.9 | No |
| circles 2 | Stopped | 5.046 | 5.358 | 0.551 | 424.1 | No |
| circles 3 | Yes | 4.953 | 5.279 | 0.616 | 424.6 | No |
| circles 4 | Stopped | 5.057 | 5.365 | 0.805 | 469.0 | No |
| circles 5 | Stopped | 5.045 | 5.278 | 0.675 | 413.8 | No |
| perpendicular-lines 1 | Yes | 3.206 | 2.684 | 0.454 | 493.0 | No |
| perpendicular-lines 2 | Yes | 2.615 | 2.964 | 0.529 | 606.5 | No |
| perpendicular-lines 3 | Yes | 3.866 | 3.064 | 0.399 | 587.9 | No |
| perpendicular-lines 4 | Yes | 1.918 | 1.919 | 0.297 | 397.4 | No |
| perpendicular-lines 5 | Stopped | 5.050 | 5.801 | 0.695 | 285.8 | No |

Max edge distance above is over the whole recorded run, including lead-in/out and final hold. The additional on-edge-only maximum remains in JSON. Since an intended extension is outside the finite edge, its edge distance is nonzero even under ideal following. At 5 mm/s these fixtures use 1.25 mm extensions. The guard stops on the first observed violation; sample latency means measured error can exceed 5 mm before Hold.

The initial line-1 approach was rejected for motion during planning. After settling, its retry ran to completion. Its original rejection evidence is retained.

## Center-facing arcs

All five fixtures were re-preflighted without the 0.05 rad check. They now reach **44.4% of the extended path**, where IK exhausts 300 iterations. All final candidates have J4 at its model upper bound of **1.5708 rad**. Position residuals range from 0.707 to 4.642 mm; they do not satisfy the unchanged numerical IK tolerances. This is evidence about these local solves, not proof that all placements are unreachable. No center-facing arc command was sent to the robot. The new execution edge-distance threshold does not loosen numerical IK/path-model tolerances. Full diagnostics: `center-arcs-preflight.json`.

## Plots and reproduction

- [Speed overview: first trial from each live suite](speed-overview.png), with raw measured-model speed, planned ramp/feed, and the 4–6 mm/s acceptance band during the requested edge.
- All line trials: [speed](lines-speed.png), [position and speed](lines-tracking.png).
- All circles: [speed](circles-speed.png), [position and speed](circles-tracking.png).
- All perpendicular lines: [speed](perpendicular-lines-speed.png), [position and speed](perpendicular-lines-tracking.png).
- Matching SVGs, per-run JSON/JSONL and `summary.json` include full records, speed percentiles and percentage of samples within the feed band.
- Reproduce a bounded trial with `.venv/bin/python -m scripts.stage3_trial --suite lines --index 0` after enabling the controller. Other suites are `circles`, `perpendicular-lines`, and `center-arcs`; failed preflights must not be forced.
- Plot speed: `MPLCONFIGDIR=/tmp/autoweld-mpl .venv/bin/python -m scripts.plot_tracking --evidence-dir evidence/stage3-edge --pattern 'lines-*.jsonl' --output lines-speed --speed-only`.

No stage has been marked DONE. Position-only completion does not qualify feed-speed regulation.

Final state: motors disabled at verified rest, maximum joint error 0.262 degrees. SDK fault: None; motion fault: None. UI remains on port 8081.
