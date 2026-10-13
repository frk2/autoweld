# Nine velocity averaging windows: 0.2–1.0 seconds

Remade all **20 saved 120 mm line runs** at 20/30/40/50/60/80/100 mm/s using nine
windows: **0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 and 1.0 seconds**.
Each run has a 3×3 chart, one panel per window. These are reanalyses of existing
measurements, not new longer-path trials.

Blue is encoder-derived TCP displacement divided by window duration; black dashed
is commanded displacement divided by the same window duration. Gray shows raw
motor-reported velocity through the TCP Jacobian. Both averages are placed at the
window midpoint, and each panel states the resulting reporting delay. Raw
velocity bias is not silently corrected. Gaps exclude invalid feedback timing.

Circles/numeric labels show the final complete window entirely within constant
reference feed. If the window exceeds cruise duration, a diamond marks the
whole-trace peak average and the panel explicitly says no full cruise window.
Filtering and the green ±20% band are diagnostic; historical raw-speed acceptance
has not changed.

The old 100 mm/s reference has only 0.363 s of cruise, so only 0.2 and 0.3 s
windows fit inside it. Longer windows cannot establish steady tracking from that
recording. The other panels correctly compare averaged commands through ramps.

## Longer-path preparation

A new **500 mm Y line** passes offline IK, joint, self-collision and derivative
checks at all seven feeds, including checked approach and return. It gives
**4.163 s of reference cruise at 100 mm/s** and 5.502 s at 80 mm/s. Existing
automatic-test excursions below 0.49 rad prevent its execution: required maximum
joint change is 1.498 rad, and maximum rest-relative offset is 0.931 rad.

The user subsequently approved the reviewed exception. The [500 mm live report](../long-cruise/REPORT.md) records completed runs through 60 mm/s and an 80 mm/s edge-guard stop; 100 mm/s was not executed on the longer path. The temporary exception has ended and the arm is disabled at verified rest, with its normal limits restored.

## Charts

[Final cruise average versus window size, all speeds](window-comparison.png)

Y: [20 mm/s](line120-01-20mms-windows.png), [30 mm/s](line120-01-30mms-windows.png), [40 mm/s](line120-01-40mms-windows.png), [50 mm/s](line120-01-50mms-windows.png), [60 mm/s](line120-01-60mms-windows.png), [80 mm/s](line120-01-80mms-windows.png).

Z: [20 mm/s](line120-03-20mms-windows.png), [30 mm/s](line120-03-30mms-windows.png), [40 mm/s](line120-03-40mms-windows.png), [50 mm/s](line120-03-50mms-windows.png), [60 mm/s](line120-03-60mms-windows.png), [80 mm/s](line120-03-80mms-windows.png), [100 mm/s](line120-03-100mms-windows.png).

Y/Z diagonal: [20 mm/s](line120-04-20mms-windows.png), [30 mm/s](line120-04-30mms-windows.png), [40 mm/s](line120-04-40mms-windows.png), [50 mm/s](line120-04-50mms-windows.png), [60 mm/s](line120-04-60mms-windows.png), [80 mm/s](line120-04-80mms-windows.png), [100 mm/s](line120-04-100mms-windows.png).

Reproduce without hardware: `python -m scripts.plot_velocity_grid`. The script also accepts `--source` and `--dest` for subsequent recorded line runs. The time-weighted averaging helpers retain their three passing regression tests. Ruff passes. [Metrics](metrics.json) retain the numeric values for all nine windows.
