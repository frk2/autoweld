# Lower-P overshoot retuning — 2026-09-14

**No fixed-P/I profile qualified across both movement directions.** The new requirement is at most **0.1° directed overshoot on every moving joint**, while retaining the aim of sustained convergence within **1 second** to ±0.2° on J1–J3 and ±1° on J4–J6. This report records **21 completed live trials**. Three return-to-rest trials individually passed both criteria; that does not qualify those profiles on the outward or direct-reversal moves.

The previously saved fast profile is historical and does not meet this new overshoot ceiling. No replacement profile is presented as compliant. Source PID defaults have not been promoted to a failed experimental candidate. The running app is left in original **PD**, motors **disabled at verified rest**. Its PID fields retain the final soft test values, which are not a qualified profile.

## What was tested

Elbow P was reduced from the previous 400 through successive adjustments of no more than 20%, down to 41.231686. Elbow I ranged from 5 to 1800. Base and most wrist P gains were reduced as well. Joint 5 needed more holding stiffness to reduce its delayed drift; its P was tested from 9 to 23.887872 through allowed increments, with I between 0 and 100. Integral accumulation on J1/J2/J4/J6 was removed in later trials when it was unnecessary for their accuracy bands or contributed to late motion.

Factory D stayed exactly `[5, 5, 5, 1.5, 1.5, 1.5]`. The original SDK PD function, friction/gravity compensation, 0.06° integral deadband, 0.5-second integral delay, waypoint reset behavior, and existing torque bounds were unchanged. No trajectory shaping or added damping was introduced.

All requests used the running app and SDK. A = `[0.035, 0.025, 0.06, 0.035, -0.025, 0.035]` rad and B = `[-0.035, 0.015, 0.035, -0.035, 0.025, -0.035]` rad relative to this session's captured rest. Trials include rest → A, A → rest and direct A → B. Testing both directions is essential; a favorable rest return alone was misleading. The camera was checked afresh after SDK startup.

## Findings

- A lower-P elbow setting around **P3 = 245.76 / I3 = 1800** settled the standard outward move in about **0.94–0.97 s**. Individual elbow overshoot ranged from **0.059° to 0.125°** across the outward repeats, so it lacked reliable margin below 0.1°.
- At that elbow setting, Joint 5's outward overshoot decreased from around **0.16–0.19°** to **0.076°** with more wrist holding stiffness. This did not solve the reversal problem elsewhere.
- Direct reversal with the lower-P 245.76 profile produced **0.607° elbow** and **0.945° wrist-4** overshoot. It settled rapidly, but failed the overshoot requirement.
- Reducing elbow P to **80.53** still produced **0.454° elbow overshoot before I activated** on reversal. Much of the initial violation therefore cannot be fixed by changing I alone.
- The softest **P3 = 41.23 / I3 = 5** outward trial stayed within **0.098° maximum overshoot**, but the elbow was still **3.307° short at one second**. The full six-joint accuracy criterion did not pass during 40 seconds (the final wrist-4 error remained just over 1°). Reversal still produced **0.388° elbow** and **0.945° wrist-4** overshoot.

This finite search does not prove that no other P/I combination could work. It does show that simply reducing P was insufficient in the tested directions and that the current requirements cannot yet be claimed as achieved. Late integral correction and initial direct-step response are distinct issues.

## All trials

Settling is the final entry of all six errors into their per-joint accuracy bands, with at least one second of subsequent data and no later excursion. “Both” additionally requires maximum directed overshoot ≤0.1° for every moving joint and settling ≤1 s. A dash means no sustained six-joint pass in the recording. All moving joints in these trials had substantial commanded steps; no zero-step hold is counted as a convergence success.

| Trial | Elbow P | Elbow I | Sustained settling (s) | Worst overshoot (°) | Both pass |
|---|---:|---:|---:|---:|---|
| p205-i600-out | 204.800 | 600 | 2.047 | 0.185 | no |
| p205-i900-return | 204.800 | 900 | 0.325 | 0.131 | no |
| p205-i900-out | 204.800 | 900 | 1.420 | 0.185 | no |
| p246-i1800-return | 245.760 | 1800 | 0.263 | 0.109 | no |
| p246-i1800-out | 245.760 | 1800 | 0.966 | 0.163 | no |
| p246-w17-return | 245.760 | 1800 | 0.284 | 0.066 | yes |
| p246-w17-out | 245.760 | 1800 | 0.942 | 0.163 | no |
| p246-w17-i100-return | 245.760 | 1800 | 0.282 | 0.087 | yes |
| p246-w17-i100-out | 245.760 | 1800 | 0.961 | 0.141 | no |
| p246-w24-direct-reverse | 245.760 | 1800 | 0.718 | 0.945 | no |
| p246-w24-return | 245.760 | 1800 | 0.222 | 0.175 | no |
| p246-w24-out | 245.760 | 1800 | 0.947 | 0.125 | no |
| transition-p197-return | 196.608 | 180 | 0.324 | 0.087 | yes |
| p157-i180-out | 157.286 | 180 | — | 0.125 | no |
| p157-i180-direct-reverse | 157.286 | 180 | — | 1.317 | no |
| p81-i60-return | 80.531 | 60 | 3.178 | 0.175 | no |
| p81-i60-out | 80.531 | 60 | — | 0.147 | no |
| p81-i60-direct-reverse | 80.531 | 60 | — | 1.098 | no |
| p81-reset | 80.531 | 60 | 3.150 | 0.175 | no |
| p41-i5-out | 41.232 | 5 | — | 0.098 | no |
| p41-i5-direct-reverse | 41.232 | 5 | — | 0.945 | no |

Recordings are approximately 50 Hz, with maximum sample spacing **0.028 s**. Encoder position resolution is approximately 0.0219°; no relaxation or rounding of the 0.1° ceiling was used. Overshoot is the maximum passage beyond the final target in the commanded direction, measured over the entire recording. A pass based on sampled encoder data is not an absolute physical/TCP metrology claim.

## Artifacts and proposed next step

- [Directional convergence plots](convergence.png), [PDF](convergence.pdf)
- [Full holds](hold.png), [PDF](hold.pdf)
- [All trial comparisons](comparison.png), [PDF](comparison.pdf)
- [Summary data](trials.json), [overshoot before I activation](pre-integral-overshoot.json)
- Raw commands/gains/targets/encoder traces: `evidence/pid/os-*.json` and matching `.jsonl` files.
- Reproduce plots with `scripts/plot_overshoot.py`; `scripts/tune_pid.py` now explicitly reports `overshoot_pass` and `speed_and_overshoot_pass`.

[The proposed next experiment](next-experiment.md) is a **150 ms smooth SDK position-reference transition** at a lower-P candidate, keeping D and the final waypoint/accuracy unchanged. It has not been implemented or run. Approval was requested because AGENTS.md explicitly says **“No trajectories or additional damping / limits.”** No improvement is claimed for this untested proposal. Arm D is already at the existing SDK motor protocol maximum of 5, so asking the motor command for a higher value would be clipped rather than provide extra damping.

## Final state and checks

All automatic targets and recorded positions remained under the 0.49-rad envelope. Maximum recorded distance from captured rest was **0.0626 rad**. There were no app-reported feedback faults during these trials. The old SDK shutdown thread/socket race was observed again after normal close; the app completed verified-rest disable.

Original PD was selected for the final return. The existing 0.5° all-motor rest check held for one second before SDK disable. Final maximum arm rest error was **0.393°**. The camera remains available. [Final state](final-state.json) · [Verified rest](verified-rest.jpg). Stage 1 remains not DONE.

Software validation: all 68 tests passed and Ruff passed. Changes add explicit overshoot scoring and plotting; no SDK control algorithm or original PD function was changed during this search.
