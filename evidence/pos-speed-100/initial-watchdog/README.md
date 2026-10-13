The first attempt at vertical line 3 at 30 mm/s stopped at startup on the unchanged
100 ms SDK feedback/execution watchdog. Immediately before the stop, recorded
feedback age was 96.28 ms, then 104.59 ms; the new reference had advanced only
6.55–10.62 ms. The motor-chain exchange diagnostic also rose to 64.1 ms. This is
not evidence that a 30 mm/s speed limit was reached, and it was not an oscillation
event. The exact cause of the communication/scheduling gap is not established.

The SDK's one-second fresh/quiet recovery check passed. Normal return and disable
verified rest at 0.153 degrees. The application now requires a new feedback frame
after path validation, age <=20 ms and measured speed <=0.05 rad/s, before starting
any planned motion. Missing feedback rejects the start after at most 0.5 s. The
100 ms running watchdog and one-second oscillation guard are unchanged. Regression
tests cover stale feedback rejection, a fresh post-validation frame and motion
during validation. An interrupted attempt is archived here rather than overwritten
by its retry.
