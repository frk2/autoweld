# Trajectory generation and discontinuity audit

21 September 2026. Offline source inspection and replay of recorded feedback only; no motor commands, controller changes or new live trials.

The examined arc references are smooth in position, velocity and acceleration and have very small spline-knot jerk jumps. The dominant observed path error is associated with unequal physical joint response to a coordinated reference. Smoothing the curve alone will not address that response mismatch.

[Reference versus measured tracking](reference-vs-tracking.png) · [numerical audit](metrics.json) · [reproduction script](../../scripts/audit_trajectory_generation.py)

## Generation pipeline

1. `TrackCurve` extracts an analytic line or circular arc from STEP in robot-base coordinates. STEP millimetres become metres. The placeholder TCP is 150 mm from the gripper mount.
2. Orientation is attached to the curve. Unconstrained mode solves position only and penalizes wrist motion more than proximal joint motion. Perpendicular line orientation stays fixed during the line. Center-facing arcs continuously rotate the torch toward the center; they additionally lock torch roll by setting TCP X to the circle-plane normal. That roll choice is stricter than merely pointing the torch axis at the center.
3. Seeded SDK IK solves 41–801 spatial samples, aiming for 0.5 mm spacing with minimum/maximum sample counts. Every solve starts from the preceding solution. This promotes a consistent branch but is not a global branch/conditioning optimizer. The former adjacent-joint-sample rejection was removed; other checks remain.
4. A SciPy `CubicSpline` interpolates the resulting joint angles as a function of common path fraction, q(s). These are spline samples, not independent stop-and-go motor waypoints.
5. A time law supplies s(t). For STEP tracking, velocity rises with a quintic smoothstep, cruises, then falls with its mirrored ramp. Position, velocity, acceleration and jerk of that scalar time law join continuously; velocity/acceleration/jerk vanish at start/end. “Inside edge” consumes part of the requested edge for ramps. “Outside edge” adds lead-in/out of feed × ramp-time / 2 per end. The planner starts with at least 0.5 s ramps, lengthening them when necessary.
6. A separate rest-to-rest law serves joint moves, TCP waypoint moves and the approach/return: s(u)=35u⁴−84u⁵+70u⁶−20u⁷. It also has zero velocity, acceleration and jerk at both ends. A direct joint move uses a straight joint-space path; its TCP route need not be straight. Cartesian waypoints use intermediate IK solutions along a Cartesian line and smooth rotation.
7. Tracking preflight samples 1001 times for joint limits, collision/excursion checks, joint/TCP velocity/acceleration/jerk and interpolated pose error. These are sampled model checks, not continuous analytic bounds or physical tracking guarantees. The sampled positional interpolation tolerance is 0.2 mm.
8. The SDK evaluates the reference at a nominal 250 Hz and hands the latest commands to a separate CAN worker. Six arm commands and the gripper exchange are sequential. Delayed cycles can skip intermediate reference points rather than replay them in a burst.
9. In POS_VEL the arm receives sampled position plus the configured positive speed ceiling. The wrapper replaces the planner's signed velocity with that ceiling. Planned acceleration/jerk are not motor command fields. Each actuator's internal position/speed/current loops determine its response. The host advances the path using elapsed time; measured cross-track error currently annotates the run and does not modify the reference or slow path progress.

Source: [curve/orientation and tracking planner](../../autoweld/tracking.py), [waypoint planner](../../autoweld/motion.py), [spline/time laws](../../vendor/i2rt/i2rt/robots/waypoint_motion.py), [SDK executor](../../vendor/i2rt/i2rt/robots/motor_chain_robot.py), [POS_VEL wrapper](../../vendor/i2rt/i2rt/robots/position_speed_robot.py).

## Where discontinuities exist

| Location | What is discontinuous or mismatched | Interpretation |
|---|---|---|
| Interior cubic spline knots | Third derivative of q(s), hence generally time jerk | Position, velocity and acceleration are continuous. Limiting jerk magnitude does not make jerk continuous. |
| Scalar acceleration/cruise/deceleration joins | Fourth and higher derivatives can jump | The time law itself has no position, velocity, acceleration or jerk jump at these joins. |
| Digital host/CAN updates | Finite position target steps; variable spacing and sequential joint delivery | Expected sampled control. Timing irregularity can enlarge steps. Internal interpolation behavior is not fully observed. |
| Transition into a new move | A new rest-to-rest plan assumes zero reference velocity; actual low speed need not be exactly zero | A waypoint built from measured position can differ from the previous held reference. Tracking normally begins at the previous approach target, so its start error is not automatically a target jump. |
| Tracking start acceptance | Up to 0.01 rad per-joint mismatch to the planned start is allowed, plus 10 mm edge entry tolerance | These are acceptance bounds, not intentional steps. Observed initial errors in the two replayed arcs were below 0.03°, much smaller than their later tracking errors. |
| Hold, disable interruption or SDK fault response | Reference is replaced by a hold at measured pose | Deliberate interruption exception. The 10 mm path-deviation flag no longer causes this transition. |
| Separate STEP edges | No through-motion corner blending | Each edge runs its own approach/start/stop. Connecting edges at nonzero speed would need explicit blending within a geometric tolerance, or a stop at the corner. |

SciPy documents `CubicSpline` as C2: its first two derivatives are continuous, while the third generally is not. [Primary documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html).

The Damiao manual describes POS_VEL as cascaded position/speed/current loops with a maximum-speed input, acceleration/deceleration parameters and a comparatively long response time. The host time law therefore does not directly prescribe the actuator's physical acceleration/jerk. We have not established whether each streamed position update resets a firmware profile; that would require firmware-specific evidence. [Motor manual](https://damiao.enactic.ai/en/products/hardware/dm-j4340-2ec-v1.0/).

## What the replay shows

| Quantity | R12.7 mm, 5° arc, 1 mm/s | R200 mm, 5° arc, 5 mm/s |
|---|---:|---:|
| Requested TCP arc length | 1.108 mm | 17.453 mm |
| Commanded J1 travel | 13.09° | 19.16° |
| Commanded J5 travel | 17.54° | 24.01° |
| Maximum recorded edge error | 19.336 mm | 6.825 mm |
| J1 error at worst edge error | −6.236° | −3.088° |
| J5 error at worst edge error | −2.301° | −1.223° |
| Largest cubic-knot jerk jump, any joint | 0.000163 rad/s³ | 0.001215 rad/s³ |
| Peak reference joint jerk | 2.841 rad/s³ | 2.892 rad/s³ |
| Peak-speed position increment at nominal 4 ms | 0.0637° | 0.0287° |

The offline reconstruction matched recorded SDK reference positions within 8.33×10⁻⁸ rad for the tiny arc and 2.76×10⁻⁹ rad for the larger arc. Reference FK stayed on the requested circle to numerical precision at the checked times. This verifies the reconstructed commands; it does not establish equivalent physical TCP accuracy.

The knot jerk jumps in these cases are under 0.05% of peak reference jerk. They are not a persuasive explanation for the centimetre-scale excursions. Both joints move in the same positive joint direction, but their Cartesian effects must be coordinated to produce the small center-facing TCP arc. Their unequal progress breaks that coordination. A tiny TCP displacement is not necessarily a tiny motor move, especially with changing orientation and a 150 mm tool offset. The observed behavior supports prioritizing response matching and pose/path conditioning; it does not prove a particular firmware parameter, friction or singularity is the root cause.

Sampled host update periods were typically 4.3–4.4 ms with observations up to 9–11 ms. Sampled CAN cycle periods also reached roughly 11 ms. These are snapshots in telemetry, not a full packet-level capture, and cannot exclude unsampled longer gaps. They show execution is not an exact simultaneous 4 ms sample-and-hold system.

The raw feedback has quantization. It can make differentiated speed look rough without implying matching command roughness. POS_VEL commands use float32 positions; the historical MIT command quantization and host sign-switching friction compensation are not active here.

## Changes worth making, in order

1. **Choose easier coordinated motion.** Optimize path placement and torch roll while preserving the required torch direction and a smooth orientation profile. Penalize excessive joint travel, high Jacobian sensitivity and rapid changes of joint motion. The current center-facing frame fixes an extra roll degree of freedom that could be optimized. Kinematic feasibility alone does not predict tracking quality.
2. **Make feed scheduling reflect measured response.** Use per-joint tracking data and ramp dynamics, not only static velocity/acceleration/jerk ceilings. Progress along one common path parameter can be reduced smoothly when coordination error grows. Avoid stop/resume switching. Any slowing must be reported honestly: it may mean the originally requested ±20% speed condition cannot be met for that pose.
3. **Improve synchronization in POS_VEL.** Identify per-joint response across recorded motions and test bounded, smooth position-lead compensation in replay/simulation before live evaluation. Use derivatives of the planned reference, not noisy differentiated encoders. A candidate lead is q_command=q_reference+τ·q_reference_velocity, but this first-order approximation is not a validated controller or universal constant. Revalidate corrected command position, velocity, acceleration, jerk and overshoot; respect inner-loop speed/acceleration saturation. A low-bandwidth Cartesian correction is another option, but must be checked for interaction with the onboard loops.
4. **Remove remaining reference discontinuities.** Use a C3-or-better joint curve with explicit derivative constraints and state-matched transitions. A carefully constructed quintic spline can provide that continuity; merely replacing the interpolation degree without validating the interpolated TCP path is insufficient. Position-lead compensation increases derivative requirements, so jerk continuity matters more if that option is implemented. Keep fault holds available as deliberate exceptions.

I would start with geometry/roll optimization plus smooth feed scheduling, and evaluate response compensation next. The present evidence does not justify raising gains simply to reduce lag. Continuous jerk is useful cleanup, but it should accompany work on the observed joint-response mismatch.

Reproduce the audit with `MPLCONFIGDIR=/tmp/autoweld-matplotlib .venv/bin/python -m scripts.audit_trajectory_generation`. The script reads existing evidence and uses SDK kinematics only. Ruff passes; the numerical reconstruction assertion independently checks the reconstructed reference against recorded SDK output. No production behavior was modified.

Follow-up: [joint-response identification](JOINT_RESPONSE.md) finds approximately 0.74–0.75 s proximal and 0.19–0.23 s wrist response time constants. The relationship to identical onboard position gains and different gearbox ratios supplies a more specific tuning target than the initial generic response-matching recommendation.
