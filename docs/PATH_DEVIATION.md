# Tracking deviation acquisition policy

The operator requested complete traces of paths previously interrupted by the
path guard. Finite-edge distance now has a **10 mm reporting/acceptance threshold**.
Exceeding it marks the trajectory as deviating and does not command Hold. The
separate 5° constrained-orientation criterion is also recorded without stopping.
Unconstrained paths do not fail or flag orientation; timed position lag remains
informational. Raw feed acceptance remains ±20%.

The UI latches **DEVIATED — recording; motion continues** when a geometric
deviation occurs, even if the arm subsequently returns inside tolerance. Final
results distinguish `completed` from `passed` and record `deviated`,
`position_deviated`, `orientation_deviated`, `first_deviation_s`, sample count,
and `deviation_action: record_only`. A completed deviating path is not qualified.
Every row includes the instantaneous deviation flag and the same policy metadata.
Position and constrained-orientation deviations cover ramps and the final hold.

This changes geometric tracking-error handling, not SDK fault handling. Motor
communication, execution/feedback watchdog, oscillation, joint, collision,
reference derivative, automatic excursion and fresh/idle-start checks remain.
Reporting failure and explicit Hold/disable/fault still interrupt the path.
Normal disable still returns to captured rest, verifies ≤0.5° for one second,
and disables. The shared path-start edge-distance threshold is now 10 mm, with
the separate ≤0.01 rad joint-start match and fresh/low-speed checks unchanged.

The repeat inventory is in `evidence/path-deviation-repeat/manifest.json`:
eight position-speed runs plus ten distinct older MIT fixtures, all repeated
using the selected position-speed controller. Duplicate older runs are mapped
to the same fixture. Each fixture must pass a fresh preflight. The explicitly
requested repeat of the previously reviewed 500 mm / 80 mm/s line uses that
fixture's temporary 1.55 rad exception only; other trials use 0.49 rad. The
exception is restored after verified-rest disable. No rejected path is sent.

Earlier reports and plots retain their original 1/5 mm stopping policy and are
historical. Selected feed presets remain 20 mm/s for lines and 10 mm/s for arcs;
this acquisition-policy change does not validate those speeds for every pose.

The [completed repeat report](../evidence/path-deviation-repeat/REPORT.md) records
16 completed paths, four geometric deviation flags, and two retained joint-bound
preflight rejections. It links complete traces, 0.2–1.0 s velocity-window plots
and the subsequent settling observations before verified-rest disable.
