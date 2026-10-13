# Tunable gravity feedforward and live retuning

Gravity and native SDK Coulomb-friction compensation were already enabled in the earlier trials. This change exposes gravity calibration and records the complete torque breakdown; it does not add gravity twice or implement a second friction compensator.

No fixed profile met both requirements across outward, raised-pose reversal and rest moves. The best matched raised-pose pair settled in **0.218 / 0.196 s**, with maximum overshoot **0.076 / 0.115°**. That profile's rest move settled in **0.650 s** but overshot J4 by **0.393°**. It is an experimental candidate, not a compliant default.

## Model and torque accounting

The SDK model applies gravity [0,0,-9.81] m/s² to its link masses and computes compensation using inverse dynamics at zero velocity/acceleration. Gravity load depends on pose and lever arms; its sign does not switch simply because a waypoint is outward or returning. The compensating torque opposes the modeled downward load. The vertical J1 axis has essentially zero gravity torque. This follows the [MuJoCo dynamics model](https://mujoco.readthedocs.io/en/latest/computation/).

Linear mode uses `G(q) = diag(scale) [g(q_ref) + Jg(q_ref) (q - q_ref)]`. The full derivative matrix includes distal links loading proximal joints. `q_ref` is captured rest; `g` and its central-difference derivative come from the SDK model. Each joint's scale is tunable. This is a local first-order approximation, not a globally linear gravity law or a measured payload mass. Beyond ±0.15 rad of the reference in any joint, the SDK uses its exact geometry model; it does not restrict a UI waypoint. The model remains active in PD, PID and gravity mode, including return moves.

At the original rest used for the offline check, nominal SDK gravity was approximately [0,1.836,6.612,1.582,0,-0.001] Nm. The local model differed from exact gravity by less than 0.01 Nm at the A/B test poses. See [model check](model-check.json). Factory multipliers are [1,1.1,1.1,1.2,1,1]. The best raised-pose candidate used [1,1.1,1.35,1.36,1,1]. These are empirical compensation multipliers; they are not measurements of link masses.

Native friction remains `SDK coefficient * sign(measured velocity)`, using [0.3,0.3,0.3,0.06,0.06,0.06] Nm. No friction algorithm or coefficient was replaced.

Telemetry now reports P estimate, I, D estimate, scaled and nominal gravity, opposite modeled gravity load, SDK friction, external command torque, feedforward sent, estimated total and motor-reported torque. `total_estimate = P_estimate + D_estimate + feedforward_sent`; feedforward contains gravity + friction + I + external command torque. Onboard PD is estimated from feedback, so it is not an exact measured total or proof of the motor's internal clipping/quantization. Motor-reported torque is SDK feedback from the sampled cycle; sent torque is the ensuing command. Disabled UI values are labeled as the last active sample. The UI table shows the six arm joints.

## Procedure and findings

All commands used the SDK, direct small waypoints and the one-second oscillation guard. Camera/image checks were skipped as requested. PID I reset on changed waypoints and accumulated immediately, with 0.06° deadband. Repeated identical targets preserved I. P/D and experimental gravity-scale changes stayed within 20% per step. I was varied independently, starting at 10. The target remained sustained settling under 1 s at ±0.2° J1–J3 / ±1° J4–J6, with directed overshoot strictly below 0.1°. Trials held each target for 12–15 s so late overshoot also failed.

A = captured rest + [0.035,0.025,0.06,0.035,-0.025,0.035] rad. B = rest + [-0.035,0.015,0.035,-0.035,0.025,-0.035] rad. Automatic target/current/rest/prior checks stayed active; the largest planned joint step was 0.07 rad. Live records are approximately 50 Hz, so faster intersample peaks may be missed.

- The baseline with I=10 took 13.337 s to settle outward. J3 motor feedback averaged about 10.184 Nm near the end, while gravity feedforward was 7.199 Nm and I supplied 3.106 Nm.
- Increasing J3 gravity scale to 1.32 reduced the raised-pose reversal's J3 overshoot to 0.060° at factory P/D, but J3 took 1.578 s and J2 missed its position tolerance.
- Gravity scale 1.35, P3=199.0656 and I3=200 enabled fast raised-pose moves with little J3 overshoot. The rest-to-A move later revealed 0.562° J3 overshoot, preventing general qualification.
- Reducing P3 to 159.25248, I3 to 100 and gravity scale to 1.32 reduced rest-to-A J3 overshoot to 0.147°, still over the limit; the matched rest move was slow at 2.228 s.
- J6 P/D/I variations reduced but did not eliminate its excess overshoot. It reached 0.115° on the best final reversal checks. D6 never exceeded 2.592; other D values remained factory. Higher wrist P was not promoted to a default.
- The tested gravity factors improved reversal behavior substantially, but feedforward and fixed PID still did not meet every target across these poses. The load/friction model and behavior near rest remain unresolved; the measurements do not establish a physical cause.

## Trials

| Trial | Sustained settling s | Maximum overshoot ° | Both pass |
|---|---:|---:|---|
| grav-01-baseline-out | 13.337 | 0.3771 | No |
| grav-02-baseline-return | 0.377 | 0.6994 | No |
| grav-03-g132-out | 3.477 | 0.1905 | No |
| grav-04-g132-reverse | Not reached | 0.0711 | No |
| grav-05-g132-p96-out | 2.169 | 0.1031 | No |
| grav-06-g132-p96-reverse | 1.161 | 0.1803 | No |
| grav-07-g140-out | 6.905 | 0.6276 | No |
| grav-08-g140-reverse | 0.967 | 0.1803 | No |
| grav-09-g140-i0-out | Not reached | 0.0000 | No |
| grav-10-g140-i0-reverse | Not reached | 0.1585 | No |
| grav-11-g140-p166-out | Not reached | 0.0976 | No |
| grav-12-g135-p199-reverse | 0.225 | 0.1585 | No |
| grav-13-g135-p199-out | 0.219 | 0.0976 | Yes |
| grav-14-w36-reverse | 0.222 | 0.1366 | No |
| grav-15-w36-d18-out | 0.217 | 0.0976 | Yes |
| grav-16-w36-d18-reverse | 0.198 | 0.1366 | No |
| grav-17-w43-out | 0.222 | 0.0976 | Yes |
| grav-18-w43-reverse | 0.200 | 0.1366 | No |
| grav-19-w52-out | 0.220 | 0.0757 | Yes |
| grav-20-w52-d216-reverse | 0.197 | 0.1366 | No |
| grav-21-w52-d259-out | 0.219 | 0.0757 | Yes |
| grav-22-w52-d259-reverse | 0.199 | 0.1366 | No |
| grav-23-w52-i2-out | 0.227 | 0.0757 | Yes |
| grav-24-w52-i2-reverse | 0.199 | 0.1366 | No |
| grav-25-w62-out | 0.218 | 0.0757 | Yes |
| grav-26-w62-reverse | 0.196 | 0.1148 | No |
| grav-27-calibrated-rest | 0.650 | 0.3934 | No |
| grav-28-g4-130-w74-out | 1.152 | 0.5621 | No |
| grav-29-g4-130-w74-reverse | 0.186 | 0.1148 | No |
| grav-30-g132-p159-rest | 2.228 | 0.2404 | No |
| grav-31-g132-p159-out | 0.205 | 0.1468 | No |
| grav-32-g132-p159-reverse | 0.183 | 0.1148 | No |
| grav-33-final-rest | 1.025 | 0.2404 | No |

## Artifacts and status

[Convergence](convergence.png), [full hold](hold.png), [comparison](comparison.png), [gravity-inclusive torque plots](torque-breakdown.png), [trial metrics](trials.json), [experimental candidate](best-raised-pose-candidate.json). Raw command parameters and telemetry are in ../pid/grav-*.json and ../pid/grav-*.jsonl. Plots were generated without image inspection, as requested.

All 85 automated tests pass, including real-model linearization, component sums, SDK gravity/friction enablement, unchanged original PD and the oscillation guard. The versioned patch recreates all four changed/new SDK files. Every recorded torque sum and every gravity/friction flag was checked. See [validation](validation.json).

No profile was promoted as meeting all requirements. At the end of this search, startup defaults remained factory PD, factory P/D, I=0 and factory SDK gravity multipliers. Gravity controls and the local linear option remain available. All 33 trials completed without an oscillation guard fault or communication failure. Motors were normally disabled at verified rest with maximum joint error 0.240°. Selected UI settings were restored to factory P/D, I=0 and SDK gravity factors; the nested disabled SDK torque/gravity fields retain the last active sample. The final result is recorded in final-state.json. Stage 1 is not marked DONE.

## Subsequent default selection

The user requested the best profile as default. The candidate in [best-raised-pose-candidate.json](best-raised-pose-candidate.json) is now the startup PID profile, with its calibrated linear gravity factors and native friction enabled. The current UI selection is updated with motors disabled. Its recorded 0.115° J6 reversal and 0.393° J4 rest overshoot remain above the strict target; this selection does not change trial qualification. Original PD and fault recovery retain factory gains.
