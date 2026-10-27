# SDK crash, October 27, 2026

Motor 2 (DM4340) returned the native **loss communication** fault. Its exception
terminated the SDK CAN worker; the robot server then terminated because the
chain was no longer running. The operator confirmed that the arm faulted and
fell. This was a physical loss of holding torque. The communication interruption
that triggered the motor fault is not proven by the available recording.

The last complete SDK sample was **10:16:55.139651 PDT**. The service continued
publishing a newer outer timestamp around frozen joint/temperature/power data.
The cached six motor codes were all `0x1`, although the following motor 2 fault
had already killed control. At 10:17:10, `acknowledge_fault` reported success:
there was no latched SDK motion fault, so its native acknowledgement was a no-op,
and the application lacked a worker/freshness check. This did not restore control.
The repeated later service tracebacks were secondary failures.

Evidence and diagnosis:

- [First failure](first-failure.log), lines 50 onward, identifies motor 2 and
  the native fault, followed by the two thread tracebacks. The old SDK discarded
  the failing packet before updating state; its exact packet timestamp, raw bytes
  and encoder position cannot be recovered from this log.
- [Host history](host-history.txt) shows swap 99.74%, 99.89%, and 99.52% full at
  10:00, 10:10, and 10:20. CPU I/O wait averaged 65.31% in the interval ending
  10:10 and 55.87% in the interval ending 10:20. System I/O PSI full stall time
  averaged 79.68% and 67.20% in those intervals. These are coarse interval
  measurements, not a trace of the precise fault second.
- [Service process](stopped-worker-process.txt) had 110,588 kB swapped out,
  about 249 MiB resident, 103 threads, and no locked memory when inspected.
  The observed host limits permitted only 8 MiB locked memory and no realtime
  scheduling priority. Host paging/I/O stalls are a credible trigger, but
  exact control-loop gaps were not preserved, so causation remains unconfirmed.
- [CAN counters](can-interface.txt) show ERROR-ACTIVE, 1 Mbit/s, zero RX/TX
  errors/drops and zero bus-off events. [Kernel log](kernel.log) shows no CANable
  disconnect or reset in the interval. These counters do not exclude missing
  application commands, an individual motor problem, or a power interruption.
- [Cached state](cached-service-status.json) records MOS/rotor temperatures at
  most 32°C in the last complete sample, well below the 60°C cooldown threshold.
  There is no recorded thermal trigger or motion in progress at the failure.
- Historical SDK motor surveys reported raw `TIMEOUT=8000` (**50 µs ticks,
  400 ms**; units corrected October 27 after checking the protocol reference). The exact current service survey was not exposed before this fix;
  the current watchdog value must not be inferred as freshly verified. The original register-unit interpretation as eight seconds was incorrect;
  the recording does not establish the exact control-command gap.
- The traceback representation names `LINEAR_4310` because of a constructor
  default in the SDK object representation. The service connection factory uses
  `NO_GRIPPER`; this representation is not evidence that motor 7 was commanded.

Implemented changes:

- Preserve the first native motor fault's parsed status, raw frame, partial
  exchange, cycle/command timing and startup watchdog/mode survey. The last
  healthy sample remains separate from this fault. Genuine faults still stop
  the worker; fault-clearing, power behavior and firmware watchdogs are unchanged.
- Expose thread liveness and sample age, label cached hardware state, reject
  acknowledgement with dead/stale control, invalidate cached rest verification,
  cancel queued application targets, archive first-failure evidence and suppress
  repeated traceback flooding. Logs no longer claim powered hold with dead SDK
  control.
- Add conservative host admission and early shutdown checks: MemAvailable at
  least 1 GiB, process VmSwap below 16 MiB, memory PSI full avg10 below 1%, I/O
  PSI full avg10 below 5%. Required metrics missing or malformed reject new work.
  Startup/Enable force a new check; active monitoring samples once per second
  and invokes the existing dedicated shutdown. Hold, shutdown and shutdown retry
  bypass admission. Normal motion speed ceilings and physical protections stay
  unchanged. These thresholds would reject the recorded hostile conditions;
  they are engineering interlocks, not validated realtime bounds.

Current physical and deployment state:

The old service was given one JSON shutdown request. It entered **Shutdown
blocked**, with "SDK control thread stopped; ... measured hold unavailable" as a
retryable blocker. See [response](shutdown-response.json) and
[blocked state](shutdown-blocked-status.json). Further application motion is
blocked by shutdown ownership. No fault clear, motor enable, direct SDK owner,
raw CAN command or physical recovery was issued during this investigation.
The arm's present pose and motor power cannot be established from cached SDK
data. The software changes have **not been deployed to that failed process**.

Remaining prevention work:

The new monitor shares the Python process and can itself stall. No successful
hardware or fault-injection qualification is claimed. Isolate the sole SDK
owner/controller/shared planner model from camera, vision and Viser work;
communicate through bounded IPC, reserve and successfully lock its working
memory, bound camera/cloud queues, and measure worst-case communication gaps
under representative load on a simulator/test fixture before supported live
qualification. Setting a Python CAN thread to FIFO alone cannot remove shared
GIL, locks or paging dependencies. Do not disable or enlarge the watchdog, ignore
native faults or blindly re-enable the fallen arm to hide this failure.

A guarantee against falling on loss of electrical torque requires mechanical
anti-drop protection independent of the host, SDK and power supply. The software
changes reduce exposure and improve diagnosis; they do not establish that the
robot can no longer fall.

Validation: **930 tests passed**, targeted Ruff checks passed, and the rebuilt SDK
patch reproduces the pinned-source modifications exactly. Results are recorded
in [tests.txt](tests.txt), [lint.txt](lint.txt) and
[sdk-patch-check.txt](sdk-patch-check.txt). All execution tests used fake
motors or simulation; there was no deliberate live fault reproduction.
