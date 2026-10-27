# Laser plane calibration implementation and live validation

The separate selectable Viser panel calls the AutoWeld JSON service. Weldfinder owns laser-ON native capture, stripe exposure, plane fitting and noise filtering. The existing camera owner and robot model are reused.

Live validation on 2026-10-26 visited all 25 unique cone edge poses through the service at the configured 3 m/s TCP / 2 rad/s joint ceilings. Endpoint convergence precedes a single measured Hold; fresh quiet feedback precedes native capture. No raw CAN or standalone arm owner was used. Maximum sampled motor temperature during calibration was 33 °C (34 °C during cleanup); no motor or SDK motion fault occurred.

21 native views were accepted; four were rejected without relaxing quality gates (three bracket-motion failures and one target-pose RMS failure). The installed laser plane has RMS **0.06796 mm**, with **10,060/10,060** plane inliers. Normal coverage is 15.26 degrees. Five-fold held-out accepted-view plane RMS is **0.06849 mm**. These are internal camera-plane consistency checks, not an absolute robot accuracy qualification.

The camera-frame equation uses millimetres: `0.1381561316*x - 0.8668251608*y - 0.4790897869*z + 63.90307849 = 0`.

Saved calibration: `runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json`. Native images, frame identities and captures: `runtime/agents-live/evidence/laser-plane-1793046122298456910`. Intrinsic K, distortion, resolution, source and board metadata were unchanged. The measured hand-eye remains bound to the same optical intrinsic identity after the plane-only save.

Live manual stripe check passed. Preview off cleared the cloud; preview on restored 984 clean points. A 985-point snapshot was finite and its J6-to-base composition matched the reported base transform exactly. World registration waits for a camera acquisition inside a fresh stationary interval. Noise filtering and stripe-aware auto exposure were enabled; stripe exposure stabilized near 562 microseconds with no measured saturation. Both calibrated image-plane directions were independently checked with bounded service/web movements.

Earlier live attempts exposed exposure-control whitelisting, premeasurement settling, premature measured Hold, preview toggle and manual acquisition-cutoff bugs. Their fixes have targeted regressions. Fast partial preview detection alone no longer authorizes capture-pose hunting; native missing-board detection allows one bounded retry. Failed attempts did not save a plane.

Validation: 866 AutoWeld tests passed; 67 focused Weldfinder facade/pipeline/adapter tests passed at the accepted calibration build. A separate full sibling run had one known asynchronous timing assertion failure; it was not used to loosen production gates.

Final robot/service state and startup filtering checks are recorded separately in this directory.

Final state: service restarted on port 8082 with motors disabled and camera off. The saved 21-view plane and matching 25-view hand-eye reloaded successfully with motors kept disabled. Default point-cloud filtering was confirmed enabled after restart. Shutdown verified fresh quiet rest before disabling; see `rest-disabled.json`. Startup displays the saved rest as cached, rather than claiming new SDK feedback. Motion defaults remain 3 m/s TCP and 2 rad/s per joint.
