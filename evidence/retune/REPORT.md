# Guarded factory-baseline PID retuning

No fixed profile qualified across all tested directions. Fifteen direct-waypoint trials completed, plus one aborted trial identified by the user as a guard false alarm. Source defaults remain factory P [80,80,80,10,10,10], D [5,5,5,1.5,1.5,1.5], and I=0. The first integral trial used I=10. P/D changes stayed within 20% per step; I changes were not percentage-limited. Automatic motions stayed within 0.49 rad of current, rest and previous target.

Requirements: sustained settling below 1 s at ±0.2° (J1–J3) / ±1° (J4–J6), with strictly less than 0.1° directed overshoot. A passing individual trial does not establish a profile across directions or poses.

All results below used a 0.5 s integral delay. The latest user instruction removes that delay while retaining reset on each new waypoint. Immediate accumulation is now implemented and verified through the UI-to-SDK regression test; these historical gains have not been validated with the changed timing. Repeated identical targets preserve I. No carryover experiment was performed.

| Trial | Sustained settling s | Worst overshoot ° | Both pass |
|---|---:|---:|---|
| safe-factory-i0-out | Not reached | 0.0000 | No |
| safe-factory-i10-return | 1.226 | 0.2404 | No |
| safe-factory-i10-out | Not reached | 0.5738 | No |
| safe-p96-i60-return | 0.321 | 0.0437 | Yes |
| safe-p96-i60-out | 2.846 | 0.3552 | No |
| safe-p115-i180-return | 0.315 | 0.0656 | Yes |
| safe-p138-i400-out | 1.021 | 0.0929 | No |
| safe-p138-i500-reverse | 2.667 | 0.9672 | No |
| safe-d3-4-i800-out | 0.889 | 0.2124 | No |
| safe-d3-4-i600-reverse | 2.247 | 0.7814 | No |
| safe-p166-d48-out | 0.848 | 0.1148 | No |
| safe-p166-d48-reverse | 2.292 | 0.7158 | No |
| safe-p166-i4-3-out | 0.858 | 0.0757 | Yes |
| safe-1s-guard-out | 1.443 | 0.2069 | No |
| safe-1s-guard-return | 0.275 | 0.1311 | No |

The strongest outbound result settled in 0.858 s with 0.076° maximum overshoot, but its opposite-direction trial overshot J3 by 0.716° at about 0.202 s, before the old integral delay expired. After restart, repeated outward/return commands did not meet both requirements either. This evidence does not establish that removing the delay alone will solve convergence or overshoot.

## One-second oscillation guard

The original 0.3 s guard tripped on J4 about 74 ms into a return command. It cleared I and held the measured pose with factory PD; SDK communication remained healthy. The user identified it as a false alarm. The revised guard observes a full 1 s, requiring six substantial reversals, three per half, at least 0.3° span in both halves and a late span at least 70% of the early span. It rejects that false-alarm replay and still detects the earlier bad-noise recording about 1.103 s after its start, over ten seconds before communication loss. Two repeated live commands ran with the revised guard without a fault. No deliberate live oscillation was induced.

A fault stays latched and overrides commands with factory PD at measured pose, with gravity/friction retained. It does not power off the motors or move to rest. Acknowledgment requires fresh feedback and a full quiet second (position span ≤0.1°, speed ≤0.1 rad/s). See [guard validation](guard-validation.json).

## Artifacts and final state

[Convergence](convergence.png), [full hold](hold.png), [comparison](comparison.png), and [all trial metrics](trials.json). Raw recordings are in ../pid/safe-*.jsonl. All 81 tests pass, including recorded incident/false-alarm replay, fault latching/recovery, integral timing and unchanged original PD. The versioned SDK patch reproduces all three changed/new SDK files from the pinned base.

The tuning session ended with motors disabled at verified rest (maximum error 0.131°). The app was then restarted to load immediate integration with factory P/D and I=0; its post-restart disable result is recorded separately in instant-integral-state.json. Stage 1 is not marked DONE.
