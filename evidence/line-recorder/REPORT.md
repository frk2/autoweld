# Gravity-mode A/B line recorder

Deployed in the live Viser viewer on port 8081. **Record a line · A → B** provides
a file name field, **Record point A**, **Record point B**, **Save STEP and load**,
and **Clear recorded points**. Recorded points have labeled markers, a connecting
line, base-frame coordinates in millimetres and a displayed length. Either point
can be recorded again.

Capture requires enabled, healthy Gravity compensation mode, no planned motion
or pending disable, and feedback no older than 100 ms. It reads measured joints
through the SDK and applies the existing modeled TCP FK (150 mm along mount −Z).
It never sends an arm command, changes the controller mode or records the target
instead of the measured pose. The existing SDK guards are unchanged.

Saving creates one A-to-B STEP edge in millimetres under the active runtime's
`trajectories/` directory, checks its endpoints by reimporting it, then publishes
the complete file without overwriting an existing filename. Folder traversal,
invalid formats, missing endpoints and coincident points are rejected. No fake
surface is added to make a wire look like a solid. The existing tracking viewer
renders the edge in robot-base coordinates, independently of CAD placement.

The saved line is selected for tracking, with 20 mm/s selected line feed,
unconstrained orientation and inside-edge ramps. Saving does not preview or run
motion. The existing preview, joint/IK/collision and execution checks apply when
the operator chooses to approach/run. Draft points last for the current app
session; saved STEP files can be reloaded later.

## Validation

- **121 targeted tests passed**: recorder, controller, IK request handling and
  tracking. New tests verify actual-pose capture, no motion side effects, metre/
  millimetre and A-to-B round trips, filename containment, overwrite protection,
  missing/coincident points, disabled/wrong-mode/moving/faulted/stale capture
  rejection, and clearing a draft without deleting the loaded path.
- Ruff passes for the changed Python modules and new tests.
- A real offline App instance recorded a 7.720192 mm line, exported
  `offline/trajectories/Recorded demo line.step`, reimported the endpoints and
  remained in Gravity compensation with no active trajectory. Evidence:
  `offline-smoke.json`, `offline/command-events.jsonl`. Capture handlers took
  1.8–2.9 ms; export/check/load took 9.0 ms in this local simulation run.
- Browser verification confirmed the new controls in the deployed live viewer;
  capture/save buttons are correctly disabled before enabling gravity mode and
  recording valid endpoints. No live point-recording or trajectory test was run.

## Live deployment state

The user was manually using gravity mode when implementation finished and
explicitly approved return to rest and restart. The initial rest check was just
outside tolerance; a retry also missed it. Motors stayed enabled throughout,
with no guard/tolerance or gain change. Once the measured pose entered the
required band, normal one-second verification and disable succeeded at maximum
**0.437139°**. Both earlier failed checks are retained here.

Viser restarted with motors explicitly disabled. The user's current **Tuned PD**
selection, exact SDK gravity model and factors **[0.8,1.19,1.31,1.32,1,1]** were
restored, together with the observed 1 rad/s position-speed ceiling (inactive
under MIT). P/D, I=0, friction-off and command encoding match the saved status.
Saved rest, motion limits and gripper intent are byte-for-byte unchanged.
No motor enable was sent after restart. `deployed-status.json` and
`preserved-settings.json` capture the settings; `disabled-rest.json` records the
successful live rest verification. The offline test server was shut down.

Implementation: `autoweld/teach.py`, `autoweld/app.py`; usage in `README.md`.
