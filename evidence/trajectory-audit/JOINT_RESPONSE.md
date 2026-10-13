# Why the proximal joints lag in POS_VEL

21 September 2026. Analysis of existing recordings only; no motor commands or firmware writes.

**Measured response differs substantially between joint groups.** The most useful new finding is that J1–J3 behave almost exactly like first-order position followers with time constants near 0.74–0.75 s. Wrist responses are approximately 0.19–0.23 s in the available moving-path data. Their numerical response closely follows position-loop gain divided by gearbox ratio; that formula remains an experimentally supported hypothesis about the closed firmware, not a verified firmware equation.

[Moving-reference response plot](joint-moving-response.png) · [moving-fit data](joint-moving-response.json) · [independent fixed-target settling fit](joint-response.json)

| Joint | Recorded gear ratio | Recorded KP_APR | Fitted effective position gain, 1/s | Equivalent time constant, s | Moving-fit R² |
|---|---:|---:|---:|---:|---:|
| J1 | 40 | 54 | 1.337 | 0.748 | 0.999 |
| J2 | 40 | 54 | 1.345 | 0.744 | 0.999 |
| J3 | 40 | 54 | 1.332 | 0.751 | 0.995 |
| J4 | 10 | 54 | 5.058 | 0.198 | 0.852 |
| J5 | 10 | 54 | 5.165 | 0.194 | 0.995 |
| J6 | 10 | 54 | 4.299 | 0.233 | 0.872 |

A time constant describes gradual response, not a period with no movement. An ideal first-order follower reaches about 63% of a stationary step after one time constant. The wrist estimates, especially J4/J6, have less signal and a poorer simple-model fit; they are approximate responses in these recorded motions, not universal actuator specifications.

## Mechanism supported by the data

The useful empirical approximation is:

`actual joint velocity ≈ K_effective × (commanded position − actual position)`

with `tau = 1 / K_effective`. For a sustained moving target, following error is approximately `tau × requested joint velocity` once the response has settled and no limit is active. At 10°/s, a 0.75 s response needs roughly 7.5° of following error to produce that speed, while a 0.20 s response needs about 2°. When the target stops, the error decays and the final pose can become accurate. This explains why good endpoint accuracy coexists with poor path adherence.

The recorded position gains are 54 on every arm actuator. Recorded gear ratios are 40 for J1–J3 and 10 for J4–J6. Numerically, `54/40 = 1.35 /s` and `54/10 = 5.4 /s`, predicting time constants 0.741 and 0.185 s. These agree closely with the observed groups, particularly J1–J3 and J5. A controlled gain-change experiment or firmware confirmation is still required to establish the internal scaling conclusively. Do not edit the physical gearbox-ratio register to change response.

The independent constant-target settling analysis gives J1/J2/J3 time constants 0.735/0.736/0.743 s, with R² 0.997/0.997/0.979. It uses no changing reference and therefore corroborates the moving-path fit. Wrist residuals after stopping are mostly too small for that separate fit above the chosen quantization threshold; no unsupported wrist settling estimate is substituted.

The selected POS_VEL wrapper sends sampled positions and a positive maximum-speed field; it does not send the planned signed velocity as feedforward. The [manufacturer manual](https://damiao.enactic.ai/en/products/hardware/dm-j4340-2ec-v1.0/) describes position, speed and current loops in cascade, with a relatively long position-mode response. Onboard position/speed gains were surveyed and preserved when the mode was commissioned. The earlier MIT P/I/D tuning is a different controller and does not alter these onboard gains.

The arc's 0.35 rad/s command ceiling already exceeded its largest planned joint speed, 0.278 rad/s. The response fits also exclude mean measured speeds above 80% of the selected ceiling. Thus the identified proportional following behavior occurs below that ceiling; increasing the ceiling alone does not remove it. This does not rule out acceleration/current/voltage limits in other operating conditions. Torque/current saturation and firmware acceleration units have not been established by this fit.

J1–J3 can have greater Cartesian leverage and often drive greater effective inertia. Their tracking accuracy deserves an appropriately tight TCP-derived error budget. Their importance does not automatically make firmware respond faster. In these data, a repeatable gain/gearing relationship is a more specific explanation than attributing the lag solely to mass or friction.

## What should be matched

Matching stable closed-loop response across axes is a useful goal. Merely matching numeric speed, acceleration or jerk settings does not match that response:

- Velocity/acceleration/deceleration limits restrict permitted motion. They become relevant when active; they do not by themselves specify position-loop bandwidth.
- Position gain (`KP_APR`) and the inner speed-loop response determine how much error is needed to produce motion, along with plant dynamics, gearing and firmware conventions.
- All six recorded `ACC`/`DEC` values are already 2/−2, yet their response differs substantially. These raw firmware values must not be equated to host output-shaft acceleration units without verification.
- No documented firmware jerk-setting register was found in the current SDK register map. Our commanded jerk limits are currently implemented in the host planner.
- Equal *pure time delay* on every joint could preserve the same joint-space path at a shifted time. Equal first-order time constants are not an exact pure delay and cannot guarantee an arbitrary nonlinear multi-joint path. Final validation must use actual TCP path error and orientation.

The next diagnostic tuning target should be the onboard position response, starting with a modest, guarded `KP_APR` adjustment on J1 and a matched smooth-motion comparison. Preserve velocity/acceleration settings initially to isolate the effect. Check both directions, overshoot, oscillation, following error and speed-loop/torque headroom before extending to the other proximal joints. A fourfold gain jump merely because the ratio is four is not a validated setting. Slowing wrists to the present proximal response would sacrifice overall bandwidth and would not by itself meet the tracking and feed requirements.

If stable loop tuning cannot achieve sufficient tracking, plan one common path-progress rate within the measured capabilities and evaluate bounded feedforward/lead compensation. Any lead must use the analytic planned derivatives, include smooth endpoint handling and revalidate the modified command's joint/velocity/acceleration/jerk limits. The new evidence makes onboard loop-response characterization/tuning a more direct next step than immediately layering on host compensation.

## Identification method and limits

The moving-path fit uses all 16 completed repeat fixtures. For each joint, recorded reference samples are aligned with per-motor feedback time using the path's recorded elapsed-time origin. Over each 0.15 s interval, encoder displacement gives mean velocity and trapezoidal integration gives mean position error. Fitting their proportional relationship avoids differentiating noisy encoders sample by sample. Windows require same-sign endpoint errors above 0.12°, feedback gaps no larger than 50 ms, and mean speed below 80% of the commanded cap. Overlapping windows are correlated, so the window count is not an independent experiment count. Telemetry samples rather than every motor command are available; small reference/feedback alignment uncertainty remains.

Constant-target fits use the same displacement/integral method after the reference stops, before normal return to rest. Both are approximate local system identification with the existing configuration, not proof that the firmware consists of only one pole or that the parameters remain constant under other payloads and poses.

Reproduce with:

```bash
MPLCONFIGDIR=/tmp/autoweld-matplotlib .venv/bin/python -m scripts.audit_pos_speed_response
MPLCONFIGDIR=/tmp/autoweld-matplotlib .venv/bin/python -m scripts.audit_pos_speed_moving_response
```

Both scripts use only recorded files. Ruff passes. Production controller code and all motor settings remain unchanged.
