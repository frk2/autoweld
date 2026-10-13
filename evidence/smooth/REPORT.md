# Friction-off gravity and PID tuning

The user withdrew the nonzero velocity-target request: vTarget remains zero. The user then requested all further tests without friction compensation. The current selected profile and startup defaults therefore have friction disabled; gravity stays enabled. Motors were returned to captured rest, verified within 0.5° for one second, and closed through the SDK. The maximum final rest error was 0.240°.

## Result and scope

The selected friction-off profile passed repeated combined raised-pose outward/reversal tests: 0.197–0.244 s sustained convergence, with maximum recorded directed overshoot 0.059°. The per-joint accuracy thresholds remain ±0.2° on J1–J3 and ±1° on J4–J6. This is **not a fully qualified profile**: smaller isolated J1–J3 moves took 1.07–2.03 s, while the wrist checks passed both limits. Small wrist test steps were below the 1° accuracy band, so short reported convergence times there do not establish physical arrival at the exact target.

The user confirmed that rest touches a stop, another link or the table; the precise contact was not specified. Contact supports part of the weight, explaining why rest can require substantially less motor holding torque than the free-space gravity model predicts. Do not use contact-rest holding torque to fit free-space gravity, or increase gains/integral effort to force the tighter free-space target against contact. The separate normal-disable rest tolerance remains unchanged. Earlier B-to-rest recordings took 3.29–3.44 s to the tighter motion tolerance and reached 0.109° J4 overshoot; those remain in the evidence, not classified as successful free-space validation.

## Selected default

- P: `[80.0, 96.0, 152.8823808, 29.85984, 14.4, 61.917364224]`
- I: `[60.0, 60.0, 75.0, 3.0, 6.0, 6.0]`
- D: `[5.0, 5.0, 5.0, 2.592, 1.5, 2.16]`
- Exact SDK gravity factors: `[1.0, 1.04, 1.31, 1.32, 1.0, 1.0]`
- Friction: **off**; vTarget: **zero**. Integral deadband: **0.06°**; immediate accumulation and reset on each changed waypoint.

The original PD command and factory fault-recovery gains are unchanged. The selected settings are also saved in [selected-profile.json](selected-profile.json). The current UI selection matches this profile; the source defaults apply on the next app startup. Inactive friction coefficients/transition controls do not contribute torque while disabled.

## Calibration and comparisons

There were 65 completed small-angle trials: 10 initial gravity-isolation runs with friction and I off, 29 friction-on comparisons before the latest user instruction, and 26 friction-off trials after that instruction. All moves used direct waypoints, under the existing 0.49-rad automatic excursion bound, with the one-second oscillation guard. No camera/image inspection was performed.

At a shared raised pose approached from opposite directions, the calibrated gravity factors `[1,1.04,1.37,1.36,1,1]` reduced average directional position bias to approximately 0.001° on J2, 0.012° on J3, and 0.000° on J4. Static friction still produced differing holding torques in each direction. These are local estimates, not identified physical masses or centers of mass. Later empirical PID comparisons used the selected factors above; their successful scope is the recorded raised-pose tests. [Calibration measurements](gravity-calibration.json).

A separate SDK friction configuration API and UI were added. Zero transition retains native sign compensation; positive transition uses a continuous tanh function. Smooth compensation was compared before the user requested friction off for all remaining tests. It remains an optional implementation, disabled by default. Configuring gravity no longer silently re-enables friction. Tests explicitly verify friction settings, including across reconnect and throughout each live trial.

## Validation and limitations

91 automated tests pass, including unchanged original PD, integral behavior, oscillation fault recovery, friction isolation/continuity, default startup and reconnect. Ruff passes. The reproducible SDK patch was freshly applied to the pinned SDK and byte-compared across all five modified/new files. Recorded P/D adjustments stayed within 20% per step; every recorded compensation flag was checked, including zero friction torque in 7,285 friction-off samples. [Validation](validation.json).

No oscillation guard or communication fault interrupted the 65 trials. During SDK close after verified rest, the SDK's motor-chain worker logged a closed-socket exception (`file descriptor ... -1`). The current SDK close stops its loop flag and closes the bus without joining that worker; this is a separate shutdown race, not a tuning result. It remains to be fixed; do not mistake that log entry for a fault during a trial. The app reports closed at verified rest. [Final state](disabled-rest.json), [session log](live.log).

Recorded overshoot is limited by telemetry sampling and encoder quantization; it is not a bound on unobserved peaks. Holds were generally 3–5 seconds, not long-duration endurance tests. A universal under-one-second / below-0.1° profile has not been established. Next work should address free-space small-move stiction and convergence with friction off, then repeat across more poses. Keep contact-rest behavior separate. Stage 1 is not DONE.

## Plots and complete trial metrics

[Convergence](convergence.png), [full hold](full-hold.png), [comparison](comparison.png); matching PDFs are available. Plots were generated without image inspection. Raw telemetry and per-trial summaries are `evidence/pid/smooth-*.jsonl` and `.json`; [all summaries](trials.json).

| Trial | Friction | Sustained convergence (s) | Maximum directed overshoot (°) | Recorded thresholds pass |
|---|---|---:|---:|---|
| smooth-01-cal-high | Off | Not settled | 0.596 | No |
| smooth-02-cal-center-from-high | Off | Not settled | 0.022 | No |
| smooth-03-cal-low | Off | Not settled | 0.000 | No |
| smooth-04-cal-center-from-low | Off | Not settled | 0.000 | No |
| smooth-05-cal-rest | Off | Not settled | 0.962 | No |
| smooth-06-fit-high | Off | Not settled | 0.553 | No |
| smooth-07-fit-center-high | Off | Not settled | 0.000 | No |
| smooth-08-fit-low | Off | Not settled | 0.000 | No |
| smooth-09-fit-center-low | Off | Not settled | 0.000 | No |
| smooth-10-fit-rest | Off | Not settled | 0.918 | No |
| smooth-11-sign-out | On | 0.280 | 0.278 | No |
| smooth-12-sign-reverse | On | 0.195 | 0.000 | Yes |
| smooth-13-sign-rest | On | Not settled | 0.131 | No |
| smooth-14-smooth-out | On | 0.262 | 0.234 | No |
| smooth-15-smooth-reverse | On | 0.197 | 0.000 | Yes |
| smooth-16-smooth-rest | On | Not settled | 0.109 | No |
| smooth-17-p159-i-out | On | 1.260 | 0.311 | No |
| smooth-18-p159-i-reverse | On | 3.178 | 0.010 | No |
| smooth-19-p159-i-rest | On | 2.550 | 0.262 | No |
| smooth-20-j3-out | On | 2.517 | 0.348 | No |
| smooth-21-j3-rest | On | 0.180 | 0.000 | Yes |
| smooth-22-j4-out | On | 0.097 | 0.087 | Yes |
| smooth-23-j4-reverse | On | Not settled | 0.000 | No |
| smooth-24-j4-rest | On | 0.019 | 0.262 | No |
| smooth-25-p127-out | On | 1.783 | 0.431 | No |
| smooth-26-p127-reverse | On | 1.384 | 0.010 | No |
| smooth-27-p127-rest | On | 1.207 | 0.437 | No |
| smooth-28-f36-i0-out | On | 0.194 | 0.005 | Yes |
| smooth-29-f36-i0-reverse | On | Not settled | 0.010 | No |
| smooth-30-f36-i0-rest | On | Not settled | 0.087 | No |
| smooth-31-f43-i50-out | On | 0.199 | 0.000 | Yes |
| smooth-32-f43-i50-reverse | On | Not settled | 0.010 | No |
| smooth-33-f43-i50-rest | On | Not settled | 0.109 | No |
| smooth-34-f52-i100-out | On | 0.201 | 0.147 | No |
| smooth-35-f52-i100-reverse | On | 0.199 | 0.010 | Yes |
| smooth-36-f52-i100-rest | On | 2.205 | 0.284 | No |
| smooth-37-brake-out | On | 0.201 | 0.125 | No |
| smooth-38-brake-reverse | On | 0.196 | 0.038 | Yes |
| smooth-39-brake-rest | On | 2.612 | 0.262 | No |
| smooth-40-off-out | Off | 0.238 | 0.103 | No |
| smooth-41-off-reverse | Off | 2.491 | 0.029 | No |
| smooth-42-off-rest | Off | 2.515 | 0.284 | No |
| smooth-43-off-p153-out | Off | 0.256 | 0.212 | No |
| smooth-44-off-p153-reverse | Off | 0.203 | 0.029 | Yes |
| smooth-45-off-p153-rest | Off | 1.786 | 0.262 | No |
| smooth-46-off-i75-out | Off | 0.244 | 0.059 | Yes |
| smooth-47-off-i75-reverse | Off | 0.197 | 0.010 | Yes |
| smooth-48-off-i75-rest | Off | 3.439 | 0.109 | No |
| smooth-49-off-raised-start | Off | 0.234 | 0.059 | Yes |
| smooth-50-off-j1-plus | Off | 1.125 | 0.000 | No |
| smooth-51-off-j1-back | Off | 0.461 | 0.000 | Yes |
| smooth-52-off-j2-plus | Off | 2.031 | 0.069 | No |
| smooth-53-off-j2-back | Off | 1.067 | 0.000 | No |
| smooth-54-off-j3-plus | Off | 1.666 | 0.000 | No |
| smooth-55-off-j3-back | Off | 1.892 | 0.000 | No |
| smooth-56-off-j4-plus | Off | 0.058 | 0.000 | Yes |
| smooth-57-off-j4-back | Off | 0.019 | 0.000 | Yes |
| smooth-58-off-j5-plus | Off | 0.016 | 0.070 | Yes |
| smooth-59-off-j5-back | Off | 0.017 | 0.000 | Yes |
| smooth-60-off-j6-plus | Off | 0.018 | 0.000 | Yes |
| smooth-61-off-j6-back | Off | 0.017 | 0.060 | Yes |
| smooth-62-off-validation-rest | Off | 0.246 | 0.022 | Yes |
| smooth-63-off-validation-out | Off | 0.243 | 0.059 | Yes |
| smooth-64-off-validation-reverse | Off | 0.206 | 0.010 | Yes |
| smooth-65-off-validation-rest | Off | 3.293 | 0.109 | No |
