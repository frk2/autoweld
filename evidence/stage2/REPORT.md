# Stage 2 — bounded waypoint references and initial commissioning

Implemented the revised AGENTS.md Stage 2: end-effector velocity, acceleration and jerk control through a timed IK/joint reference. **The reference math passes validation; physical tracking is not yet qualified to the requested limits. Stage 2 is not marked DONE.**

## Delivered behavior

- UI limits for TCP translational and angular velocity/acceleration/jerk and joint velocity/acceleration/jerk. Defaults: TCP 0.03 m/s, 0.15 m/s², 1 m/s³; angular 0.25 rad/s, 1 rad/s², 5 rad/s³; each joint 0.35 rad/s, 1 rad/s², 5 rad/s³.
- Joint moves use a straight joint-space path. Cartesian moves use a straight TCP position path and shortest-axis rotation, with SDK IK at 41 knots. All moves start/end with zero reference speed, acceleration and jerk.
- IK branch jumps, path bounds, interpolated pose residual and SDK self-collision are checked before dispatch. Validation samples 401 path positions; derivative limits use 801 time samples and a 10% duration margin. These are sampled checks, not a continuous proof. Existing geometry excludes ground/adjacent-body contacts and does not model the workpiece.
- The SDK's own loop executes q and velocity references. Joint acceleration/jerk are derived and recorded, not sent as unsupported motor command fields. No inertial feedforward or PID retuning was added. Exact gravity stays enabled; friction stays disabled; selected gains remain unchanged.
- Integral resets once per user waypoint, accumulates immediately, and remains continuous across reference samples. An identical completed target preserves holding I. The original SDK PD function remains unchanged; Stage 2 PD selection uses the motion executor with factory P/D and I=0.
- Active moves reject new targets and gain/feedforward changes. Disable queues a planned rest return, verifies the usual 0.5° tolerance for one second, then disables. Stop-and-hold and fault-hold are immediate interruptions, not jerk-limited braking.
- Existing one-second oscillation guard retained. An execution/feedback gap over 100 ms during a waypoint latches factory PD hold and cancels motion. The SDK shutdown worker is joined before its CAN socket is closed; the live disable completed without the historical closed-socket exception.

## Math and software validation

For path parameter s and normalized time u=t/T:

`s(u) = 35u^4 - 84u^5 + 70u^6 - 20u^7`.

With spline q(s), the executor computes:

- `q_dot = q_s * s_dot`
- `q_ddot = q_ss * s_dot² + q_s * s_ddot`
- `q_dddot = q_sss * s_dot³ + 3*q_ss*s_dot*s_ddot + q_s*s_dddot`.

The TCP's world-aligned twist uses the MuJoCo site Jacobian. Along the path:

- `v_s = J*q_s`
- `a_s = J*q_ss + J_s*q_s`
- `j_s = J*q_sss + 2*J_s*q_ss + J_ss*q_s`.

Time scaling applies the same chain rule. J derivatives are central differences along the path. Angular quantities are angular velocity and its world-frame derivatives, not Euler-angle derivatives. Cubic path splines are C2; their third derivative is piecewise and can jump at knots. Jerk magnitude is checked; snap is not bounded.

Validation compares joint analytic derivatives with independent finite differences, TCP translation with finite differences of FK, rotation with relative rotation vectors, and acceleration/jerk with derivative differences. Independent 1,802-point grids pass all reference limits for joint and Cartesian cases. Reducing an active jerk limit by eight doubles planned duration. The example Cartesian path residual is about 2.7e-9 m and 5.5e-10 rad.

Hardware-free tests also cover SDK execution, integral continuity, endpoint zero velocity, PD interruption, stale/gapped execution, queued disable, repeated targets, bounds/collision rejection and worker shutdown ordering. Full suite: **105 tests passed**. Ruff checks pass. The patch reproduces all seven changed/added SDK files exactly from `SDK_REVISION`; the original PD method has a byte-for-byte regression check.

The offline end-to-end check exercised joint and Cartesian moves, unreachable IK rejection, gravity/hold, STEP loading, return/disable and reconnect. Browser inspection confirmed Stage 2 controls and an enabled Enable motors button while disabled.

References: [MuJoCo Jacobian API](https://mujoco.readthedocs.io/en/3.3.1/APIreference/APIfunctions.html), [Modern Robotics, trajectory generation](https://hades.mech.northwestern.edu/images/b/b2/MR-2up.pdf). The seventh-degree scaling and derivative checks are implemented locally.

## Initial live results

Four small planned moves, including normal return to rest; SDK-only communication, no camera checks, friction off, original selected gains unchanged. The largest requested offset from captured rest was J3 +0.04 rad. No guard or communication faults occurred.

| Move | Planned duration | Sustained accuracy from start | Maximum directed overshoot | Estimated measured TCP peak speed |
|---|---:|---:|---:|---:|
| Small joint move | 0.680 s | Not sustained in 5 s recording; final J3 error 0.204° | 0.087° | 14.8 mm/s |
| Cartesian outward | 0.920 s | 2.547 s | 0.047° | 46.0 mm/s |
| Cartesian reverse | 0.950 s | 2.864 s | 0.175° (J4) | 58.5 mm/s |
| Normal rest return | 0.700 s | Separate rest criterion | No directed crossing recorded | 28.8 mm/s |

Accuracy is ±0.2° J1–J3 and ±1° J4–J6. Settling requires staying inside those bands through the recording's end; it is not proof of indefinite stability. Directed overshoot is measured beyond the final target in the commanded direction from the initial reference. For unchanged nominal joints, small corrective moves and holding offsets can dominate this metric.

The Cartesian references stayed below 30 mm/s, but measured velocity transformed through the SDK geometry peaked at 46.0/58.5 mm/s. Reference tracking error reached 0.76° on J3. These results **do not establish physical velocity, acceleration or jerk compliance**, sub-second settling, or universal <0.1° overshoot. Quantized motor velocity and app telemetry (~50 Hz, maximum recorded gap ~34 ms) do not support a reliable physical jerk bound. No external TCP metrology was used; the torch TCP remains a placeholder.

The first Cartesian recorder read the old pre-planning status immediately after command acceptance and reported it stale. The SDK completed the move normally, with fresh feedback and no fault. Its first five seconds were recovered from the full app log by SDK reference session. The recorder now waits for a fresh post-acceptance status. This did not weaken the SDK feedback watchdog.

Rest touches a support, per the user. Normal return was accepted at max 0.350° error and held within 0.5° for one second before disabling; no tighter free-space accuracy was forced against contact. After a final source/UI refresh, another normal rest disable passed at 0.219° maximum error, with no fault or shutdown exception. The app remains available at port 8081, with motors disabled. Final status: `final-state.json`.

## Next work

1. Improve closed-loop tracking in free space with the reference executor: measure breakaway, gravity/holding bias and lag in both directions before changing gains. Preserve friction off and the 20% P/D step rule. Model-reference limits alone do not bound measured motion.
2. Repeat at multiple free-space poses, longer paths, rotations and near singularities; strengthen continuous path/derivative bounds if wider travel is required.
3. Validate physical TCP acceleration/jerk with suitable sampling/filtering or external measurements, document measurement bandwidth, and distinguish tracking transients from reference limits.
4. Add smooth interruption/replanning and waypoint blending if needed; current moves require near-rest starts and complete individually.

## Reproduce

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.plot_stage2
.venv/bin/python scripts/rebuild_sdk_patch.py
# With a healthy enabled app, authorized clear arm, and exclusive control:
.venv/bin/python -m scripts.stage2_trial --name joint-small --delta .005 .01 .015 0 0 0 --duration 5
```

[Reference derivatives](numerical-derivatives.png) · [Joint move](joint-small.png) · [Cartesian outward](cartesian-small.png) · [Cartesian reverse](cartesian-reverse.png) · [Return](return-rest.png) · [Metrics](metrics.json)
