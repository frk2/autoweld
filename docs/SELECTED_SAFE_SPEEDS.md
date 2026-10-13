# Selected safe test speeds

On 2026-09-20 the operator selected these feeds for the position–speed controller:

| STEP path | Selected feed |
|---|---:|
| Straight lines | 20 mm/s |
| Circles and circular arcs | 10 mm/s |

The Stage 3 UI displays both values. Loading a tracking STEP initializes its feed
from the first edge's geometry. **Use selected safe speed** applies the selected
edge's preset and invalidates the old preview; it does not command motion.
The presets persist in source across restarts. Explicit feed entry remains
available for separately authorized experiments.

These are operator-selected operating/test presets, not universal safety or
tracking certification. The evidence used unconstrained paths in limited poses;
new constrained orientations and tighter radii require their own preflight and
measurements. Stage 3 now records finite-edge deviation over 10 mm and constrained
orientation error over 5° without stopping; see [acquisition policy](PATH_DEVIATION.md).
It retains the
raw ±20% speed criterion, SDK guards, collision/joint/derivative checks and
strictly below 0.49 rad automatic excursions. Normal disable still requires
verified rest. No stage is marked DONE.

Prior evidence: [long lines](../evidence/long-cruise/REPORT.md),
[circle speed sweep](../evidence/circle-speed/REPORT.md).
Oriented repeats: [Stage 3 welding-orientation evidence](../evidence/stage3-welding/REPORT.md).

The first constrained-orientation repeat did **not** qualify these presets for welding: at 20 mm/s, four of five perpendicular 20 mm lines hit the 5 mm path guard. Center-facing 12.7 mm quarter-arcs were not executed because no candidate passed all 10 mm/s preflight checks. Preserve the selected presets as the operator requested; do not represent them as validated for every orientation or radius.

The [smaller-arc follow-up](../evidence/small-oriented-arcs/REPORT.md) found a
200 mm radius, 5° center-facing arc that completed within the edge/orientation
criteria at 0.5–2 mm/s (including a repeated 1 mm/s trial). At 5 mm/s it hit the
edge guard. Raw feed accuracy remains unqualified. The subsequent 50 mm radius,
8° trial was interrupted by a watchdog fault. Guarded recovery and verified-rest
disable subsequently succeeded. These results do not change the selected presets.

The [10 mm record-only repeat](../evidence/path-deviation-repeat/REPORT.md) then
completed 16 previously stopped fixtures; two others still failed joint-bound
preflight. Four completed paths flagged geometric deviation. Full traces and
settling data retain the unfavorable transients; no run met raw feed qualification.
