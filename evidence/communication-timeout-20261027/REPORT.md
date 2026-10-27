# Controlled recovery and motor communication timeout

The corrected live recovery succeeded through the running service. All six
motors read TIMEOUT **80000 ticks = 4000 ms** and reported healthy enabled status
`0x1` in position-speed Hold. CAN and robot-server threads were alive with fresh
feedback. The camera remains off and captured rest was preserved.

A 15.1597-second monitor collected 51 snapshots: peak Hold error 0.02185694°,
peak reported joint speed 0.03663004 rad/s and peak temperature 30°C. The target
stayed constant, with no motion plan, hardware fault, SDK failure or shutdown
request. A duplicate recovery returned `already_controlled`, `changed=false`;
its target was unchanged. Rest, saved poses, hand-eye and optical calibration
SHA256 hashes still match their original files.

## Agent API and protections

Submit one explicit request at a time:

```bash
.venv/bin/python scripts/request.py --runtime runtime/agents-live \
  --payload '{"action":"recover_control"}'
```

The ten-second intent includes queue time, setup and physical write/enable
retries. Expired or cancelled requests cannot enable later. No background retry
is scheduled. SIGHUP/SIGINT/SIGTERM use a reentrant urgent inbox and enqueue the
normal controlled shutdown without SDK calls in the signal handler.

Recovery requires stopped previous SDK writers and a single owner. It proves
one second of fresh, stable, stationary six-axis feedback inside strict joint
bounds, agrees with native position registers and requires every joint within
0.15 rad of captured supported rest. Only disabled `0x0` or communication-timeout
`0xd` states qualify. Other faults and enabled or unknown owners remain refused.
Old targets and calibration ownership are discarded. Native fault, joint,
collision, thermal and deadline protections remain; the independent SDK motion
watchdog stays 400 ms. A retained failed owner can be released on a new explicit
request only after the same-interface fresh disabled proof, without motor writes.

Per the operator's latest instruction, host-pressure admission does not block
explicit recovery. The recovered owner gets an identity-bound warning-only host
monitor policy, cleared on disable or owner replacement. Ordinary enable and
new-motion admission remain host-gated. Explicit shutdown and thermal cooldown
remain available.

## Unit correction and earlier attempts

Native register 9 uses **50 µs ticks**, not milliseconds. Initial hardware
readbacks were raw 8000, meaning 400 ms. The first implementation incorrectly
labeled counts as milliseconds and wrote raw 4000, which is 200 ms. That attempt
failed startup qualification. Fresh subsequent SDK reads found motors 1/2 at
`0xd`, motors 3–6 at `0x0`, all stationary near supported rest. A further
1.2236-second fresh disabled proof preceded stopping that failed process.
No motion trajectory was submitted. The shortened timeout plausibly caused the
failure; the earlier response's `timeout_ms` label must not be treated as valid
hardware evidence. Raw records are preserved.

The corrected implementation converts the requested 4000 ms to raw 80000,
checks actual readbacks and reports both units. Raw 4000 is rejected even if
other metadata claims 4000 ms. No Flash write, encoder-zero or watchdog-disable
operation was used.

References: [DaMiao register conversion](https://damiao-motor.jia-xie.com/package-usage/cli-tool/#set-can-timeout),
[Seeed Damiao documentation](https://wiki.seeedstudio.com/damiao_series/).
The pinned i2rt README's factory 400 ms and its original raw 8000 setup value
also agree with 50 µs ticks. Earlier incident-report unit labels were corrected.

Before this implementation was deployed, the old failed process was stopped
only after 15 fresh SDK samples over 1.2247 seconds proved all motors already
disabled, quiet and near supported rest. That one-time bootstrap issued no motor
commands. Future supported recovery uses the service API above.

## Validation and records

**1052 tests passed in 28.54 seconds.** Targeted Ruff checks passed. The rebuilt
SDK patch applies to the pinned pristine source and reproduces its extensions.
Simulated tests cover unit conversion, timer expiry, queue age, cancellation,
partial ownership, rollback and physical gates. No deliberate live watchdog
fault was introduced to measure expiry.

Final live records:

- `corrected-units-recovery-response.json`
- `corrected-units-hold-monitor.json`
- `corrected-units-duplicate-response.json`
- `corrected-units-final-state.json`
- `corrected-units-calibration-hashes.json`
- `tests-corrected-units.txt`
