# Immediate-integral PID tuning

No fixed PID profile met both requested limits across the tested directions. Several individual moves passed; none is a validated replacement for the factory defaults. This is a bounded empirical search, not proof that no possible fixed gains can work.

The user authorized immediate accumulation after resetting I on new waypoints, 20% maximum changes to P/D, unrestricted I changes starting at 10, small direct waypoints and no image/camera checks. All trials used SDK-only commands, friction/gravity feedforward, the 0.06° integral deadband and existing integral torque bounds. Repeated identical targets preserved I. No trajectories or camera checks were used. The one-second oscillation guard remained active and did not trip in this session.

The target was sustained settling under 1 s at ±0.2° on J1–J3 and ±1° on J4–J6, with directed overshoot strictly below 0.1°. Trials observed 12–20 s of holding so late overshoot also counts as failure. Feedback was recorded at approximately 50 Hz; overshoot values are maxima of recorded samples, and faster intersample peaks may be missed.

## Findings

- Factory P/D with I=10 took 12.234 s to settle outward and overshot J4 by 0.377° and J5 by 0.207°.
- The fastest passing outward trial was `imm-15-d5-i350-out`: 0.222 s, maximum overshoot 0.032°. Its matching return `imm-14-d5-i350-return` overshot J3 by 0.131°. It is not qualified across directions.
- `imm-17-p120-i350-out` passed at 0.236 s and maximum overshoot 0.032°. Its matching return also overshot J3 by 0.131°.
- Reducing I from 350 to 300 allowed a 0.319 s return with 0.044° overshoot, but its outward trial developed 0.147° late J3 overshoot despite reaching the accuracy band in 0.279 s.
- Reducing J3 D from 5 to 4 worsened outward overshoot to 0.584°. D was restored through 4.8 to factory 5. No wrist D exceeded 2.592; J5/J6 D stayed factory 1.5.
- The smaller reversal between two non-rest poses remained difficult: J3 overshoot was 0.716–0.891°, with sustained settling of 2.194–9.864 s; J4 overshoot was 0.443–0.967°. Moderate increases in J3 P up to 191.103 did not eliminate this behavior.
- A low-overshoot return-to-outward trial at P3=159.25248, I3=400 nearly met speed (1.058 s, maximum overshoot 0.093°), but that does not resolve the opposite-direction failure.

The records show J3 temporarily sticking just short of target while I rises, followed by a position jump. On non-rest reversals, large overshoot often occurs near 0.2 s after the waypoint resets the holding correction. This is consistent with a direction-dependent load/friction problem, but this session did not identify its physical cause. A useful next investigation is the SDK gravity/friction feedforward and load model, retaining immediate I reset behavior, rather than assuming larger gains will fix it. No feedforward changes were made.

## Commands and gains

A = captured rest + [0.035,0.025,0.06,0.035,-0.025,0.035] rad.
B = captured rest + [-0.035,0.015,0.035,-0.035,0.025,-0.035] rad.
Return = captured rest. The maximum planned joint change is 0.07 rad. The harness additionally checked measured/current, captured-rest and prior-target test bounds. Full gain vectors and raw telemetry are saved for each trial; [search ranges](search-ranges.json) summarizes their extent.

| Trial | Sustained settling s | Maximum overshoot ° | Both pass |
|---|---:|---:|---|
| imm-01-i10-out | 12.234 | 0.3771 | No |
| imm-02-i10-return | 0.456 | 0.2841 | No |
| imm-03-p96-i120-out | 3.459 | 0.3333 | No |
| imm-04-p96-i120-return | 2.366 | 0.0000 | No |
| imm-05-p115-i120-out | Not reached | 0.0812 | No |
| imm-06-p138-i240-return | 0.303 | 0.1093 | No |
| imm-07-p138-i200-out | 2.473 | 0.1031 | No |
| imm-08-p115-i200-return | 0.296 | 0.0000 | Yes |
| imm-09-p115-i200-out | 2.108 | 0.1249 | No |
| imm-10-d4-i200-return | 0.315 | 0.1093 | No |
| imm-11-d4-i200-out | 5.989 | 0.5839 | No |
| imm-12-d48-i400-return | 0.298 | 0.0437 | Yes |
| imm-13-d48-i400-out | 0.202 | 0.1249 | No |
| imm-14-d5-i350-return | 0.324 | 0.1311 | No |
| imm-15-d5-i350-out | 0.222 | 0.0320 | Yes |
| imm-16-p120-i350-return | 0.319 | 0.1311 | No |
| imm-17-p120-i350-out | 0.236 | 0.0320 | Yes |
| imm-18-p120-i300-return | 0.319 | 0.0437 | Yes |
| imm-19-p120-i300-out | 0.279 | 0.1468 | No |
| imm-20-p120-i325-return | 0.293 | 0.1093 | No |
| imm-21-p120-i325-out | 0.264 | 0.1686 | No |
| imm-22-p120-i350-reverse | 3.131 | 0.9672 | No |
| imm-23-p144-d18-out | 1.167 | 0.0812 | No |
| imm-24-p144-i600-reverse | 2.194 | 0.8906 | No |
| imm-25-gentler-out | 0.702 | 0.1468 | No |
| imm-26-p92-i100-reverse | 9.864 | 0.7158 | No |
| imm-27-p92-i350-out | 0.559 | 0.1686 | No |
| imm-28-p92-i350-return | 0.321 | 0.0437 | Yes |
| imm-29-p110-i600-out | 0.700 | 0.8244 | No |
| imm-30-p132-i400-reverse | 2.715 | 0.8688 | No |
| imm-31-p159-i400-out | 1.058 | 0.0929 | No |
| imm-32-p191-i400-reverse | 3.079 | 0.8688 | No |
| imm-33-p191-i450-out | 1.147 | 0.0929 | No |
| imm-34-p191-i450-return | 0.262 | 0.1749 | No |

## Evidence and final state

[Convergence plots](convergence.png), [full holding interval](hold.png), [all-trial comparison](comparison.png), [trial metrics](trials.json). Raw data and command parameters are in ../pid/imm-*.jsonl and ../pid/imm-*.json. The plots were generated without image/camera inspection as requested. These tests validate measured behavior only for the listed small angles, rest capture and current setup.

All 34 trials completed without an oscillation guard fault or SDK communication failure. Normal disable verified rest at maximum joint error 0.328°. No tuned profile was promoted to the defaults. See final-state.json for verified rest, disabled motors and the restored factory P/D, I=0 selection. Stage 1 is not marked DONE.
