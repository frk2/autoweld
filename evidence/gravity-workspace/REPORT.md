# Wider-angle gravity tests — 2026-10-22

The wider tests found a real limit in the existing compensation. Shoulder and
elbow poses near 90° could pass with empirical wrist corrections, but those
corrections did not generalize to a coupled base-yaw pose. **The workspace is
not fully qualified, and no candidate gains were saved as defaults.**

The arm finished at verified captured rest, motors disabled, camera off. There
were no observed motor faults, SDK motion faults or failed mode transitions.
The highest recorded motor temperature was 35 °C. The UI tuning status is now
**limited**, including on fresh startup from the bundled profile.

## Method and extent

Seventeen poses were preflighted while motors were disabled, using the existing
shared model and planner. Checks covered full-path joint limits/self-collision,
mount clearance above the nominal base surface, and predicted gravity torque.
Predicted path loads stayed below 6 Nm on the arm and 1.4 Nm on the wrist.
The fixed receiver/camera geometry was retained; physical payload mass and COM
remain estimates, not identified parameters.

Only the existing SDK connection controlled motors 1–6. A temporary, disabled-only
1.8 rad commissioning envelope allowed the wider poses. Actual position–speed
travel used increments no larger than 0.18 rad at the existing 0.15 rad/s ceiling.
Half-second gravity probes preceded 1.5-second probes at each new pose. Each
probe independently expired in the SDK into powered measured-pose PD, with
unchanged 0.04 rad excursion / 0.3 rad/s speed trigger thresholds. A stricter
0.015 rad drift acceptance gate also stopped progression.

There were **65 bounded probes** across the original profile, two diagnostic
comparisons and two candidate sweeps. The last candidate completed **24 probes
at 12 poses**, then stopped on the first probe at pose 13. Four subsequent poses
were not attempted. All 17 preflight passes therefore must not be reported as
17 live successes.

Completed live poses spanned measured shoulder angles **17.30–88.99°**, elbow
**20.82–89.56°**, wrist J4 **−43.75–30.57°**, J5 **−31.64–42.83°**, and J6
**−104.44–−24.12°**. Base yaw remained near −3° in the completed poses; the
attempted positive-yaw pose reached approximately +34° before its gravity probe
aborted. Negative yaw and the final three combined poses remain untested.

## Findings

The original saved factors `[1, 1, 1, 1, 1, 1.10]` passed the baseline and 35°
shoulder tests. At the 45° elbow pose, J6 crossed the speed guard after 0.218 s:
0.432 rad/s and 0.568° drift at the trigger. A later captured sample showed
approximately 0.831° total drift. Powered hold recovered without a motor fault.

J6 factors ×1.35 and ×1.50 both passed quarter-, half- and 1.5-second diagnostic
probes at that pose; ×1.20 still hit the speed guard. The smaller passing
correction, ×1.35, was selected for a temporary sweep. It passed both probe
lengths through the near-90° shoulder/elbow poses and both wrist-roll directions.
At positive wrist tilt, J4 drifted **1.617°** in half a second. The SDK timer
completed normally, but the script's stricter drift gate rejected the result.
This was a test acceptance failure, not an SDK fault or speed-guard event.

J4 corrections ×1.05, ×1.10 and ×1.15 all passed the local diagnostic probes
within one encoder step. The smallest correction, ×1.05, was combined with
J6 ×1.35 and the sequence was restarted from the beginning. That candidate's
24 successful probes had worst drift **0.175°** and peak recorded probe speed
**0.095 rad/s**. The formerly failing positive wrist-tilt pose improved to
0.066° / 0.044° for the half-/1.5-second probes. The available 391 active
telemetry samples confirmed all six actual MIT modes and zero estimated
position-gain torque, integral torque and friction feedforward.

At the coupled positive base-yaw pose, J6 moved in the positive direction and
tripped the speed guard after **0.0224 s**: **0.344 rad/s**, with **0.219°**
drift at the trigger. Subsequent capture telemetry reached **0.520 rad/s** and
approximately **0.415°** drift. Guard thresholds initiate recovery; they are
not hard mechanical velocity ceilings. This probe ended before the first
client telemetry sample, so its post-guard PD samples cannot establish the
free-drive P term. The SDK's event record identifies the bounded probe and its
guard response.

The opposite J6 motion directions at the two failing configurations show why a
single empirical multiplier cannot yet be called a workspace-wide solution.
Payload COM error and mode-release dynamics need further diagnosis. Static
motor effort close to commanded gravity is not independent physical validation;
post-guard braking effort is especially unsuitable for gravity fitting. The
earlier static payload fit was also rejected. No physical payload fit was
silently substituted and no guard threshold was increased.

## Result and evidence

Original factors `[1, 1, 1, 1, 1, 1.10]` were restored and remain saved, with
their earlier small-pose qualification explicitly limited. The temporary
commissioning envelope was reset to 0.49 rad. All six motors stayed enabled
through the probes and their recovery/mode changes; normal power-off happened
only after returning to rest and passing the stationary verification.

The complete Python suite passed **412 tests**; Ruff passed. The 41 new workspace
tests cover envelope restrictions, malformed inputs, interior path collision,
signed torque peaks, rotated mount clearance, payload/factor handling and model
sharing. These software tests support the test harness; they do not override
the failed live qualification.

- [Machine-readable summary](summary.json), [per-probe CSV](candidate-probes.csv).
- [Original-profile trials](trials.json), [J6 diagnostics](wrist-diagnostics.json).
- [J6-only candidate sweep](retuned-135/trials.json), [J4 diagnostics](j4-diagnostics.json).
- [Last candidate sweep](retuned-105-135/trials.json), [preflight](retuned-105-135/preflight.json).
- [Final disabled/rest-verified status](final-status.json), [envelope reset](envelope-reset.json), [saved tuning status](tuning-status.json).
- Reproduction harness: `scripts/gravity_workspace.py`; default is preflight
  only. `--execute` runs probes, and `--j4-factor` / `--j6-factor` select temporary
  comparisons that are restored after verified disable.

Encoder resolution is approximately 0.0219° per step. Small or zero recorded
drift is not proof of sub-step physical accuracy or friction-free hand guidance.
