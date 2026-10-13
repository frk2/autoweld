# Sub-second PI convergence — 2026-09-14

**All nine trials at the selected settings met the requested accuracy within 0.156–0.703 seconds and stayed inside it through the remainder of each 10–25 second recording.** The standard outward move improved from 3.780 seconds at the previous profile to 0.682 seconds. The comparison includes 50 completed live trials; earlier uniform-0.1° results remain historical.

The requested accuracy is ±0.2° on joints 1–3 and ±1° on joints 4–6. These are measured encoder tracking errors on small moves near the captured rest with the current gripper/load, not a claim about the entire workspace or absolute TCP accuracy. Only the user can mark Stage 1 DONE.

## Selected controller

| Joint | P (Nm/rad) | I (Nm/(rad·s)) | D (Nm·s/rad) | Accuracy (°) |
|---|---:|---:|---:|---:|
| 1 | 96 | 40 | 5 | 0.2 |
| 2 | 115.2 | 40 | 5 | 0.2 |
| 3 | 400 | 3600 | 5 | 0.2 |
| 4 | 35.831808 | 6 | 1.5 | 1 |
| 5 | 13.824 | 3 | 1.5 | 1 |
| 6 | 29.85984 | 6 | 1.5 | 1 |

Only the elbow P/I changed from the previous saved profile. Each experimental P adjustment stayed within 20% of the then-current value. Factory D and the original SDK PD function are unchanged. The integral deadband remains 0.06°, I resets on each new waypoint and starts after 0.5 seconds, and identical targets preserve I. The existing integral torque caps remain ±5 Nm on J1–J3 and ±1 Nm on J4–J6, with no gripper I. Gravity/friction compensation remain enabled. No trajectory or extra damping was added.

The larger I coefficient builds and retracts a **bounded torque correction** quickly after the 0.5-second delay. Increasing I at low P did not reliably solve the problem; some combinations developed late drift. P = 400 / I = 3600 gave the best worst-case convergence among the profiles tested across multiple directions and sizes, with less initial overshoot than the P = 450 profile. This is the best validated setting in this finite search, not proof of a global optimum.

## Validation at the selected gains

Settling is measured from submission of the waypoint to the final entry of **all six** errors inside their respective bands. Every subsequent sample must remain inside, with at least one second of recorded hold after entry. A brief pass followed by drift fails. Sampling is approximately 50 Hz; the largest interval across this session's recordings was 0.042 seconds. Small timing differences of about one sample should not be overinterpreted.

| Move | Sustained settling (s) | Recording (s) | Worst initial/late overshoot (°) |
|---|---:|---:|---:|
| return | 0.174 | 10 | 0.022 |
| larger | 0.662 | 20 | 1.419 |
| smaller | 0.666 | 20 | 0.731 |
| reset | 0.158 | 10 | 0.022 |
| reverse | 0.703 | 25 | 0.486 |
| ik | 0.156 | 20 | 0.161 |
| final-return | 0.239 | 10 | 0.087 |
| standard-out | 0.682 | 20 | 0.880 |
| direct-reverse | 0.661 | 20 | 0.541 |

The standard rest-relative offsets are A = `[0.035, 0.025, 0.060, 0.035, -0.025, 0.035]` rad and B = `[-0.035, 0.015, 0.035, -0.035, 0.025, -0.035]` rad. The larger target is `[0.05, 0.04, 0.08, 0.05, -0.04, 0.05]`; the smaller target is `[0.015, 0.01, 0.02, 0.015, -0.01, 0.015]`. The smaller trial approaches from the larger target. The TCP trial uses SDK FK/IK to reach a nearby pose derived from A; the solved target and every actual sample are saved with that trial. Direct waypoint transitions and returns are also included.

Worst measured absolute errors near the one-second mark, across the nine selected-profile trials: `[0.0944, 0.1241, 0.0999, 0.5628, 0.3614, 0.2349]` degrees.

**The accuracy bands describe settling, not the entire approach.** Transient overshoot still reached 0.731° on J3 and 1.419° on J4 during the larger-step tests, before the one-second deadline. The larger move's wrist overshoot was 1.660° at P3 = 450 and 1.419° at P3 = 400. The robot remains subject to abrupt waypoint transients with factory D and no trajectory shaping.

## Comparison and rejected settings

- Previous P3 = 138.24 / I3 = 60: standard outward settling 3.780 s; elbow error at one second 0.793°.
- P3 = 199.0656 / I3 = 720: 1.501 s outward; the remaining offset still persisted too long.
- P3 = 238.87872 / I3 = 60: standard outward 0.177 s, but the opposite direction failed to stay in tolerance over 25 s. Rejected despite its fast favorable case.
- P3 = 412.78242816 / I3 = 1800: opposite-direction settling 1.743 s; late overshoot of 0.202° exceeded the 0.2° elbow criterion.
- P3 = 495.338913792 / I3 = 1800: opposite direction 0.849 s and standard outward 0.727 s, with greater initial stiffness/overshoot.
- P3 = 450 / I3 = 3600: 0.657–0.719 s on the principal moving trials; lower P3 = 400 retained the speed and reduced the larger move's wrist overshoot.
- P3 = 360 / I3 = 3600: opposite direction 0.743 s with 0.180° elbow overshoot, leaving little margin.
- P3 = 360 / I3 = 7200: repeated drift outside the elbow tolerance through 20 s. Rejected.

One P3 = 495.338913792 / I3 = 180 trial used a -0.025 rad wrist-4 offset instead of B's -0.035; the exact command is preserved. It is not treated as an identical six-joint comparison.

## Artifacts and reproduction

- [Selected profile and measurements](selected-profile.json)
- [Convergence: first 3 seconds](convergence.png), [PDF](convergence.pdf)
- [Full recorded holds](hold.png), [PDF](hold.pdf)
- [All 50 trials](comparison.png), [PDF](comparison.pdf), [summary data](trials.json)
- Raw per-trial JSON/JSONL: `evidence/pid/fast-*.json*`; trial command, gains, target, timing and telemetry are retained.
- `scripts/tune_pid.py` performs one SDK/app-inbox trial with automatic-test guards. `scripts/plot_fast.py` regenerates the plots without hardware access.
- `evidence/fast/rescored-history.json` is a read-only rescore of earlier PI recordings at the new tolerances; those records are not counted as fresh live validation.

Across the 50 new trials, maximum commanded distance from rest was 0.0800 rad, maximum consecutive commanded step was 0.0800 rad, and maximum recorded distance from rest was 0.0919 rad. All remained below the automatic-test authorization. No SDK communication faults were reported by the app during the trials.

## Software and final robot state

The UI/status API now report the six requested accuracy tolerances, per-joint pass flags and a 1-second goal. PID defaults are saved in `autoweld/control.py`; the running UI shows P3 = 400 and I3 = 3600. The original PD remains selectable and retains its original behavior. Verified rest remains 0.5° for every motor for one second, independently of the motion bands.

All **68 tests passed**, including per-joint threshold boundaries, factory D, unchanged original SDK PD source, gain-step rejection, integral reset/delay, and rest verification. Lint passed. The offline end-to-end check passed for joint/IK commands, STEP import, gravity/hold, disable and reconnect. The offline simulator is ideal following and is not evidence of physical convergence.

The live app at port 8081 remains available. The arm returned to captured rest and verified the 0.5° tolerance for one second before normal SDK motor disable. Final maximum arm rest error was 0.437°. Motors are **disabled**, camera is available, and **Enable motors** is enabled in the UI. [Final state](final-state.json) · [rest camera](verified-rest.jpg).

The SDK logged its previously observed control-thread/closed-file-descriptor exception during shutdown after normal disable (`live.log`). The app completed verified-rest disable and remains available; this pre-existing SDK shutdown race was not changed by the tuning work.
