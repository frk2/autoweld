# Declined experiment — not run

The user requested native D tuning instead and explicitly retained the no-trajectories rule. This proposal is historical and will not be implemented.

Keep factory D, original PD, 0.06° integral deadband, 0.5-second integral delay, friction compensation, captured rest, and all automatic movement bounds unchanged.

Test the PID controller with a 150 ms smooth position-reference transition inside the SDK. Compare the same A/B small waypoints already recorded. Start from the lower-P candidate P [80, 92.16, 245.76, 20.736, 23.887872, 19.1102976], I [0, 0, 1800, 0, 100, 0]. Reach the exact final target at 150 ms; do not change the final target or its accuracy. Record at least 25 seconds after each move. Accept only if every moving joint stays below 0.1° directed overshoot and all joints settle to [0.2, 0.2, 0.2, 1, 1, 1] degrees within one second and remain there. Stop and return to verified rest under the existing rules if feedback faults or the automatic excursion bound is reached.

This experiment would change the AGENTS.md requirement "No trajectories or additional damping / limits." It has not been authorized or implemented. Its result cannot be promised from the current recordings. The current arm motor protocol already limits D to 5; simply requesting a higher D in the existing motor command would be clipped by the SDK protocol encoding.
