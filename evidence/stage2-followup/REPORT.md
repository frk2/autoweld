# IK responsiveness and larger-move convergence

## Outcome

The dropped IK gesture bug is fixed. Larger Cartesian moves execute and complete, but **faster convergence with the strict overshoot ceiling and actual velocity compliance is not yet established**. No experimental gain/feedforward profile was promoted. Original selected gains, gravity factors, and the operator's motion limits were restored. D remained unchanged and friction remained off throughout.

## Why an IK move sometimes appeared to do nothing

The installed Viser TransformControls API calls `on_update` at gesture **start**, during updates, and at **end**. The previous callback commanded all three phases. Grabbing the handle therefore started a move to the unchanged pose, often just a tiny correction. Real drag/release targets arriving during that move were rejected by the rest-to-rest executor. The live pre-debug state explicitly showed “Motion in progress; wait for the waypoint to finish or select Hold” even after motion had completed.

The callback now commands **only the released pose**. It uses an async callback without awaits to preserve Viser phase order. If the arm is still moving, the latest operator TCP pose waits until the reference finishes and measured joint speed is <=0.05 rad/s. New releases replace the pending pose; there is no delayed sweep through intermediate drag samples. Pending poses expire after 15 s and cancel on Hold, mode/joint commands, return, disable, shutdown or fault. Automatic test commands are never queued.

IK still rejects unreachable poses, modeled self-collisions, invalid joint steps, IK branch jumps and other existing constraints. Those are deliberate rejections, not ignored commands. Endpoint IK now uses the path solver's 1e-8 tolerances instead of its loose default, preventing tiny requests from being declared solved before their target is reached. Planning/accepted/queued/rejected status is displayed beside the TCP controls. Requests and errors are recorded in `runtime/live/command-events.jsonl`.

The UI also names the constraint determining duration. Applied motion limits are persisted to `motion-limits.json`; targets are not replayed on startup.

## Live test conditions

Fifteen trials, including one raised-pose setup and seven outward/return pairs. All motion used the SDK, with one-second oscillation detection, execution/feedback watchdogs and automatic excursions below 0.49 rad. Image/camera checks remained skipped under the user's instruction. No guard or communication faults occurred.

Offsets from captured rest:

- A: `[.02, .06, .12, .01, 0, 0]` rad.
- B: `[.10, .16, .34, .035, .025, .025]` rad.

The Cartesian A/B path is approximately 93–95 mm long. These are substantially larger than the previous ~0.015–0.04 rad commissioning moves, while remaining within the existing test authorization. The maximum requested offset is 0.34 rad (~19.5°).

Operator limits at entry: TCP speed 0.1 m/s, acceleration 2 m/s², jerk 1 m/s³; angular speed/acceleration/jerk 1/2/5; joint speed/acceleration/jerk 1/2/5. Two comparison trials lowered TCP speed to 0.05 m/s only. The original 0.1 m/s setting was restored.

### Candidate profiles

Baseline: existing selected profile in `before.json`. D unchanged in every case.

| Profile | Changed P | I (J1–J6) | Gravity change |
|---|---|---|---|
| PI1 | J1=96, J2=115.2, J3=183.45885696 | 120,120,180,6,6,6 | none |
| PI2 | PI1 plus J4=35.831808 | 120,120,120,12,6,6 | none |
| PI3 | PI2 with J3 restored to 152.8823808 | 120,120,90,12,6,6 | none |
| PI4 | PI3 | 120,120,105,12,6,6 | J4 scale 1.32→1.45 |
| PI5 | PI3 | 120,120,90,12,6,6 | J4 scale 1.45 |

All P changes and rollbacks were within 20% of the then-current setting. The J4 gravity test was motivated by approximately +0.2 Nm of common residual holding correction at both A and B. It improved J4 behavior locally but did not yield a fully qualifying combined profile. No direction-switched gravity or friction compensation was introduced.

## Measurements

Settling is measured **from SDK motion start**, not button click: all joints must remain within ±0.2° on J1–J3 and ±1° on J4–J6 through the rest of the 7–10 second recording. This is not proof of indefinite stability. SDK planning/acceptance latency is saved separately in each trial JSON. Maximum directed overshoot is across all six joints, including small corrective motions. TCP speed is estimated from reported joint velocity and the geometry Jacobian; no external metrology was used.

| Trial | Planned travel (s) | Sustained settling (s) | Max directed overshoot (°) | Estimated measured TCP peak (mm/s) |
|---|---:|---:|---:|---:|
| baseline-large-out | 2.238 | 6.905 | 0.054 | 120.9 |
| baseline-large-return | 2.256 | 2.890 | 0.354 | 118.3 |
| baseline-raised | 1.496 | 3.394 | 0.144 | 90.4 |
| pi1-large-out | 2.271 | not sustained | 0.256 | 123.7 |
| pi1-large-return | 2.233 | 1.906 | 0.354 | 111.5 |
| pi2-large-out | 2.252 | not sustained | 0.278 | 127.4 |
| pi2-large-return | 2.242 | 1.950 | 0.289 | 115.5 |
| pi3-large-out | 2.264 | 4.951 | 0.076 | 124.7 |
| pi3-large-return | 2.260 | 2.013 | 0.333 | 114.6 |
| pi4-large-out | 2.257 | not sustained | 0.519 | 122.0 |
| pi4-large-return | 2.259 | 1.994 | 0.100 | 120.8 |
| pi5-large-out | 2.259 | 4.905 | 0.098 | 121.7 |
| pi5-large-return | 2.263 | 2.012 | 0.122 | 125.8 |
| slower-large-out | 4.570 | 6.359 | 0.076 | 70.0 |
| slower-large-return | 4.522 | 3.839 | 0.354 | 69.9 |

At 0.1 m/s, the smooth reference takes about 2.24–2.27 s for the ~94 mm move and reaches ~90.9 mm/s peak. Thus a one-second total move is not possible with this reference shape and those limits. That does not explain the additional settling tail: baseline outward settling was 6.905 s. PI5 reduced it to about 4.91 s (~29%), but the matched return still exceeded the strict 0.1° overshoot ceiling. Other aggressive-I trials failed to sustain the accuracy band even at the recording's end. They are not defaults.

Halving the speed cap approximately doubled planned duration to ~4.5 s, as expected mathematically. It did not eliminate endpoint error or reference tracking deviations. Actual peak estimates were roughly 0.12 m/s at the 0.10 m/s setting and 0.07 m/s at the 0.05 m/s setting. **These limits presently bound the planned reference, not measured physical speed.**

The traces show stationary offsets, delayed breakaway and jumps into/through the accuracy band. Friction, gravity-model error and compliance/backlash are plausible contributors; the trial data do not uniquely identify them. Faster I alone did not resolve the problem. Further work needs bidirectional holding/low-speed identification over the larger workspace and improved tracking control, rather than simply increasing gains or claiming the speed cap guarantees actual speed.

## Validation and final state

114 software tests passed, including gesture phase regression, latest-target replacement, wait-for-stop, cancellation, expiry, automatic-request rejection and command journaling. Existing derivative, SDK execution, guard, rest-disable and unchanged-PD tests remain passing. Ruff checks passed. The robot returned to verified rest after experiments (max error 0.284°) and disabled. The final source/UI deployment also leaves motors disabled; see `final-state.json`.

The original selected gains and gravity settings remain defaults. No stage is marked DONE.

## Reproduce and inspect

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.plot_stage2 evidence/stage2-followup
```

Each trial has a JSON metadata/configuration file, raw JSONL, and a convergence PNG. `metrics.json` contains per-joint errors, overshoot, speed estimates and timing. See [baseline outward](baseline-large-out.png), [baseline return](baseline-large-return.png), [best trial outward](pi5-large-out.png), [best trial return](pi5-large-return.png).
