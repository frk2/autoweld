# Dedicated shutdown verification

`Controller.shutdown()` accepts shutdown independently of application mode,
calibration, stale disable state and the automatic commissioning excursion guard.
Its tick captures measured hold, waits for quiet fresh SDK feedback, plans a
conservative rest path, and verifies rest before closing the SDK. Physical checks
remain enforced. Errors become visible, retryable blockers with SDK control
retained. Healthy duplicate requests preserve progress; Hold pauses until retry.

The JSON service cancels calibration ownership and queued targets, prevents later
motion or parameter changes from stealing shutdown, and suppresses calibration
ticks during exit. Filesystem requests, the UI and process signals use this path.

Validation:

- Full suite: **669 passed in 26.95 seconds**.
- Ruff checks and compilation passed for changed Python files.
- Regression tests cover shutdown beyond the old excursion cap, stale latches,
  cleanup failure, duplicate requests, paused/retried returns, physical rejection,
  stale feedback, SDK faults and verified-rest motor disable.
- The updated service started using `--live --start-disabled` at port 8082.
  Playwright observed an enabled shutdown button with motors disabled, camera off,
  and the display correctly labeled as a saved rest record.
- Clicking that button submitted JSON `shutdown`, accepted in **0.0000484 seconds**.
  The service published phase `disabled` and exited with code 0. The browser click
  command timed out after the server exited; backend command and process records
  confirm delivery and completion, so the command was not resent.
- Restarted the updated service with `--start-disabled`; fresh application status
  shows no fault, no pending disable, shutdown phase `idle`, and camera off.

This service smoke test did not enable motors or command arm motion. Powered
return behavior was tested with SDK doubles, not a new live excursion. The final
UI pose is a saved rest record, not a new physical feedback measurement.

See [accepted UI request](ui-shutdown.json) and [final service state](final-state.json).

## SIGHUP follow-up

Registered SIGHUP with the same queued shutdown handler as SIGINT and SIGTERM.
Focused shutdown tests: **44 passed**; Ruff passed. Sent real SIGHUP to the
verified disabled service process after checking the handler was installed.
The backend accepted shutdown, published phase `disabled` without error, and
the process exited with code 0. Restarted the updated service in disabled mode.
No motors were enabled. See [accepted signal request](sighup-shutdown.json).
