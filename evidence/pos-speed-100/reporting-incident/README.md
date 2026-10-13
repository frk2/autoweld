# Initial 40 mm/s reporting failure and SDK recovery

This directory preserves the interrupted first 40 mm/s Y attempt. It is excluded
from the 20 completed sweep results; its retry is in the parent directory.

The acceleration/jerk ramp calculation introduced a NumPy float into SDK timing.
Comparisons produced NumPy booleans in motion/qualification fields, causing JSON
status writes to fail. The old app called tracking reporting before the controller
lifecycle tick, so a repeated reporting exception could prevent final rest
verification and motor disable from being processed.

Hold, restoration of the 0.15 rad/s ceiling and normal disable were requested.
The SDK accepted return to rest, but reporting blocked application finalization.
An attempted reporting-only process patch failed at OS debugger attach; no code
was injected. An SDK active-read observer also failed to collect a reliable full
rest check because replies were mixed with the running chain. Neither attempt
issued movement, enable or mode-change commands.

Recovery then used a passive SDK receiver: SDK `try_receive_message` and
`parse_recv_message`, with no transmitted frames. It checked all seven motors,
including the gripper through the existing SDK mapping, against captured rest.
Fresh feedback (age below 50 ms), speed at most 0.1 rad/s and error at most 0.5°
were required continuously for more than one second. After that verification,
the broken app process was stopped to end its command stream, motors were disabled
through SDK `motor_off`, and all seven status values were independently read as
`0x0`. Only then was the broken app process terminated.

The final verification interval was 1.018779 s, and maximum error across all
motors was 0.240426°. See [passive verification](../passive-sdk-rest.json),
[disabled readback](../sdk-recovery-disabled.json), and
[debugger attempt log](../reporting-recovery.log).

`recovery-used.py` is an **audit copy of the one-time recovery**, with the old
process ID embedded. Do not rerun it as a general recovery utility.

The permanent fixes convert path duration/ramp timing to native floats and result
fields to native booleans. App tracking-report exceptions now request Hold and
the controller tick runs in `finally`, preserving return/rest/disable processing.
Regression tests and actual offline 40/100 mm/s dispatch passed before the live
sweep resumed. No guard threshold was changed.
