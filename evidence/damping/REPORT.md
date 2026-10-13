# D tuning stopped after wrist oscillation and communication failure

The user reported that the robot got stuck and made a bad noise on the fourth trial, and requested removal of all D changes. D overrides have been removed from the separate SDK PID method, the UI, and the trial runner. Factory D is again fixed in source at [5, 5, 5, 1.5, 1.5, 1.5]; the original PD function remains unchanged. The SDK patch was regenerated and checked; all 68 tests and Ruff pass.

The faulted live process was not restarted or sent further motor commands. Normal return to rest and motor disable could not be verified after communication was lost. Source rollback does not alter the faulted process's in-memory gains or establish the physical motor state. Camera snapshots after the fault were read-only.

## Last motion request

Time: 2026-09-14T23:21:15.196090-07:00. Direct SDK PID joint waypoint, no trajectory. Joints 1–6:

| Joint | Target rad | Change from prior target rad | P | I | D |
|---|---:|---:|---:|---:|---:|
| J1 | 0.021649119 | -0.070000 | 80 | 0 | 5 |
| J2 | 0.020912871 | -0.010000 | 92.16 | 0 | 5 |
| J3 | 0.056934844 | -0.025000 | 245.76 | 1800 | 5 |
| J4 | -0.103474861 | -0.070000 | 29.8598 | 10 | 3.1104 |
| J5 | 0.028624018 | +0.050000 | 23.8879 | 100 | 3.1104 |
| J6 | 0.020504692 | -0.070000 | 19.1103 | 0 | 3.1104 |

The preceding move used wrist D 2.592 and J4 P 24.8832. Immediately before this waypoint, wrist D increased 20% to 3.1104 and J4 P increased 20% to 29.85984. Other gains were unchanged. The gripper remained at its captured setting; I reset for the new waypoint and was delayed 0.5 s, with 0.06° deadband. Gravity and friction compensation remained enabled.

SDK telemetry showed repeated rapid wrist velocity reversals, reaching approximately 3.95 rad/s on J5 and 3.41 rad/s on J6. This is consistent with oscillation; the log alone cannot prove the cause of the reported sound or the subsequent communication failure. The SDK then raised `fail to communicate with the motor 1`, and both control threads exited. No valid sustained-convergence result is assigned to the aborted trial. The SDK continued sending the same waypoint with changing feedback/feedforward during the hold; this is the last motion request, not a captured final CAN packet.

## Completed trials

| Trial | Wrist D | Sustained settling s | Worst overshoot ° | Both requirements met? |
|---|---:|---:|---:|---|
| d216-out | 2.16 | 0.234 | 0.103 | No |
| d216-reverse | 2.16 | 0.723 | 0.836 | No |
| d259-out | 2.592 | 0.742 | 0.147 | No |

No D-tuned profile was selected. These near-rest tests do not validate other poses or loads. Targets stayed under the existing 0.49-rad automatic test bounds.

Evidence: `last-command.json` contains the last request and gains; `d311-aborted.jsonl` preserves the available fourth-trial telemetry including the gain-change sample before the waypoint; `d311-aborted.json` summarizes that incomplete record. `live.log` contains the SDK failure. Complete trial raw records remain under `evidence/pid/d*.jsonl`. The convergence/hold/comparison plots cover only the three completed trials; `aborted-wrist.png` shows the failed hold. No trajectory was implemented or tested.

## Subsequent user-requested restart

The user requested “restart everything.” The faulted process was terminated after normal shutdown could not complete. A fresh SDK/app/camera session completed initialization and gripper calibration with original PD selected and factory D restored. Startup held the new captured rest; the viewer was reloaded. The final saved state reports fresh feedback, no fault, and IK mode with original PD selected. No tuning trial was run during restart. See `restart-factory-d.log` and `restart-state.json`. This supersedes the earlier process-state note above.
