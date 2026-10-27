# Calibration Pose and default speed update

Used the existing right-panel Viser controls to enable all six motors and recall
the saved Calibration Pose. Fresh SDK feedback verified arrival with maximum
joint error 0.02186°, no motion fault and healthy temperatures.

The operator then requested default TCP speed 3 m/s and joint speed 2 rad/s.
Updated MotionLimits, Controller, GUI defaults/range, README and AGENTS.md. The
application accepts the SDK-supported range (0, 3] rad/s; acceleration, jerk,
collision, joint, feedback, thermal and fault protections remain enforced.
Validation: **745 tests passed in 28.99 s**, and changed Python files passed Ruff.

Deployment used the service's controlled verified-rest shutdown, followed by a
restart with motors disabled. The fresh process reported startup defaults of
3 m/s and 2 rad/s. Temporarily selected the previously validated conservative
limits for restoring Calibration Pose through Viser. Applied the requested
3 m/s / 2 rad/s through the UI only after stationary arrival.

Final service telemetry, persisted motion-limits.json and visible UI values all
agree. SDK telemetry reports `[2, 2, 2, 2, 2, 2]` rad/s ceilings. The arm remains
enabled in Hold at Calibration Pose, with no fault and camera off. The final
recorded maximum temperature is 32°C; maximum final joint error is 0.02186°.
No movement at the new higher ceilings was performed during this update.

See `new-default-startup.json`, `rest-before-restart.json`, `restored-pose-plan.json`
and `final.json` for the corresponding service records. Viser is open in the
right panel; its reconnect required browser focus because the installed client
gates disconnected retries on document.hasFocus().
