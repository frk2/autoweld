# Smaller center-facing arcs

**Final hardware state: disabled at verified rest, maximum joint error 0.153°.** The seventh trial hit a watchdog fault and temporarily blocked recovery because J5 was not quiet enough. Feedback subsequently settled; the unchanged SDK quiet-window check accepted recovery, and normal return/rest verification/disable succeeded. No further motion tests were run. See `final-rest-status.json`, `trial7-fault-status.json`, `trial7-recovery-attempts.json`, and `trial7-rest-final.log`.

## Working geometry found

A **200 mm radius, 5° center-facing arc** (17.453 mm travel) completed at **0.5, 1, and 2 mm/s**, including two separate 1 mm/s runs. These runs satisfied the 5 mm finite-edge and 5° world-orientation criteria. Raw instantaneous velocity qualification still fails, so this is not full Stage 3 qualification or a welding-ready claim.

The best measured run was **0.5 mm/s: 1.114 mm maximum edge deviation, 0.472° maximum full orientation error**. After the normal one-second tracking hold and an additional settling observation, its endpoint error was **0.079 mm**, with measured phase **5.018°**. The repeated 1 mm/s run settled to **0.145 mm** endpoint error and **5.011°** phase. These endpoint observations show actual traversal of the small arc; merely remaining within 5 mm of a short edge would not establish that.

All positions/errors are encoder plus SDK kinematic-model estimates, not external metrology or an applied-force compliance measurement. A small settled endpoint error does not imply equally small error during travel.

| Radius | Sweep | Feed | Outcome | Max edge error | Max full orientation error |
|---:|---:|---:|---|---:|---:|
| 12.7 mm | 5° | 1 mm/s | Edge guard Hold | 5.021 mm | 0.545° |
| 200 mm | 5° | 2 mm/s | Completed | 3.135 mm | 1.354° |
| 200 mm | 5° | 1 mm/s | Completed | 1.917 mm | 0.769° |
| 200 mm | 5° | 5 mm/s | Edge guard Hold | 5.100 mm | 1.159° |
| 200 mm | 5° | 0.5 mm/s | Completed | 1.114 mm | 0.472° |
| 200 mm | 5° | 1 mm/s, repeat | Completed | 2.023 mm | 0.770° |
| 50 mm | 8° | 0.5 mm/s | SDK watchdog interruption | 2.992 mm before interruption | 0.706° before interruption |

The 5 mm/s failure ended the upward feed sweep; **10 mm/s was not physically attempted** on the 200 mm arc. The selected UI presets remain 20 mm/s for lines and 10 mm/s for circles/arcs, as the operator requested. The lower feeds above are individual commissioning trials, not a preset change.

## Search and reproducibility

The typo “1/&th” was interpreted as 1/8 circle, or 45°, with smaller sweeps also searched. Three search revisions recorded 259 candidate evaluations, including repeated geometries. Radii ranged from 12.7 to 200 mm and sweeps from 3° to 45°. The searches found 18 accepted plan records at 3°, 5°, and 8°, including repeated geometries; no searched 10°–45° candidate passed all checks. This is a bounded local search, not proof that larger oriented arcs are globally unreachable.

The final search ranked seed poses separately for each radius and sweep. Earlier ranking favored the smallest radius and missed useful shallow, larger-radius paths. The 150 mm tool offset and coupled joint motion mean that a short TCP arc can still require substantial joint travel. Short arcs also fail constant-feed planning when their length cannot accommodate the required acceleration/jerk ramps. Neither restriction was relaxed.

Working fixture: [`assets/arc-10-r200-a5.step`](assets/arc-10-r200-a5.step).

- Center in robot-base coordinates: `(477.879, 15.757, 285.479)` mm.
- Start: `(279.235, -7.432, 283.798)` mm.
- Plane normal: `(-0.020630, 0.246969, -0.968804)`.
- Orientation: **Center facing**, torch along TCP −Z, full pose constrained.
- Ramps: **Inside edge**.
- Planned maximum rest-relative excursion: 0.440 rad; maximum joint span: 0.419 rad.

`preflight.json` contains exact seed poses, geometry and derivative limits. `search-10mms-initial.json`, `search.json`, and `search-wide.json` preserve rejected candidates. Each modified-feed run has a separate `*-preflight.json`; raw tracking, SDK telemetry, result, settling and rest records use the same run prefix.

Scripts:

- `scripts/small_arc_preflight.py`: offline geometric and derivative search, requires disabled motors.
- `scripts/small_arc_trial.py`: one accepted fixture through the existing SDK app, followed by observed settling and normal verified-rest disable. SDK faults remain latched and stop the runner.
- `scripts/plot_small_arcs.py`: finite-arc deviation, orientation, actual angular progress, endpoint and nine velocity windows from 0.2–1.0 s. Requires disabled motors.

## Measured tracking and velocity

At 2 mm/s the TCP initially traveled backwards along the arc, reaching roughly −0.86° phase. Timed position lag reached 7.375 mm while finite-edge error remained below 5 mm. The latest valid one-second cruise average was 1.850 mm/s against 2.000 mm/s commanded. At the end of the standard one-second hold, measured sweep was 4.465° and endpoint error 1.878 mm. Additional settling was explicitly recorded for later trials.

The 12.7 mm arc also initially moved backwards and hit the edge guard. Its full orientation error stayed small, but the actual torch direction differed from the bearing to the circle center by up to 22.8° because the TCP displaced significantly relative to the small radius. Full timed-pose orientation and actual center-bearing error are distinct measurements; the plots label both.

For the completed R200 mm arcs, maximum actual torch-to-center bearing error was 1.154° at 0.5 mm/s, 1.886°/1.900° at 1 mm/s, and 3.438° at 2 mm/s. The interrupted R50 mm / 8° trace reached 5.144° actual center-bearing error before its watchdog stop, despite only 0.706° timed world-orientation error. That run is not an accepted center-facing result.

Lower feed improved geometry but did not establish smooth velocity. The latest valid one-second cruise averages were 0.049 mm/s at commanded 0.5 (window ending 31.205 s, before the 34.907 s cruise end), 1.226 and 1.034 mm/s at commanded 1, and 1.850 mm/s at commanded 2. These are individual local windows, not whole-run means; sampling gaps omit later windows in several runs. All nine windows and their aligned command averages are plotted, including these unfavorable measurements.

All seven online run summaries report `passed=false`. Raw ±20% feed accuracy is not qualified. The legacy initial duplicated elapsed-zero rows also make the online sampling summary conservative; additional invalid feedback-timing samples occurred in several runs. Offline window averages retain aligned encoder timestamps and omit invalid timing intervals; they do not replace the raw acceptance criterion.

## Guards, fault and recovery status

All hardware communication used the existing SDK app. Joint, self-collision, interpolation, reference velocity/acceleration/jerk, strictly-under-0.49-rad automatic excursion, 5 mm finite-edge, orientation, oscillation and 100 ms watchdog checks remained active. Friction remains disabled. No actuator gain, ACC/DEC, TIMEOUT or Flash values were changed. No production controller or guard code was changed.

The first six trials each ended with normal verified-rest disable at **0.131°** maximum rest error, POS_VEL ceiling restored to **0.15 rad/s**, automatic excursion **0.49 rad**. After the 5 mm/s edge Hold, the first return request was rejected while the arm was still settling. Waiting for stationary feedback allowed normal return and disable; the small-arc runner was then updated to observe settling before asking for return, preserving the SDK's independent raw-speed check.

The seventh trial, R50 mm / 8° / 0.5 mm/s, triggered **“Waypoint feedback or execution stalled for over 100 ms”** during tracking. The SDK latched position-speed Hold. No bulk tests, plotting or offline search were running during that live trial. The root cause of the scheduling/feedback gap has not been established. The UI labels its generic fault mode “Oscillation fault”; the recorded triggering event is the watchdog, not the oscillation detector.

Fresh communication resumed, but recovery was initially rejected by the existing quiet-window guard. Sampled J5 variation was 0.175–0.219° over one second; sampled speeds stayed below 0.1 rad/s. The first recovery request and ten subsequent guarded attempts were rejected; none bypassed the guard or issued a return command. Later feedback showed approximately 0.022° span on all joints and the SDK accepted recovery. An immediate helper read briefly saw the previous published fault status; after fresh status confirmed fault-free Hold, normal return and disable completed at **0.153°** maximum rest error. No guard was changed and no further trial was run.

Fault acknowledgement reset the UI's cached MIT gains to factory values. The original saved MIT profile was restored in at-most-20% steps while motors stayed disabled, then POS_VEL was reselected. This performed no actuator gain or hardware-mode writes. `settings-validation.json` verifies exact preservation of captured rest, motion limits, gravity, friction-off, MIT settings, automatic excursion, POS_VEL ceiling and selected safe-speed presets. The last held-state snapshot is historical: `held-status-before-recovery.json`.

## Artifacts and validation

[`online-result-summary.json`](online-result-summary.json) contains all seven outcomes. [`metrics.json`](metrics.json) adds aligned geometry, actual center-bearing error, angular coverage, post-run endpoint observations and nine velocity-window measurements. Fourteen figures cover every run, including the interrupted ones:

- [Best geometric run: R200 mm, 5°, 0.5 mm/s tracking](plots/arc-10-r200-a5-0.5mms-trial5-tracking.png)
- [R200 mm, 5°, 0.5 mm/s nine velocity windows](plots/arc-10-r200-a5-0.5mms-trial5-windows.png)
- [R200 mm, 5°, 1 mm/s repeat tracking](plots/arc-10-r200-a5-1mms-trial6-tracking.png)

- [R200 mm, 5°, 2 mm/s tracking](plots/arc-10-r200-a5-2mms-trial2-tracking.png)
- [R200 mm, 5°, 2 mm/s nine velocity windows](plots/arc-10-r200-a5-2mms-trial2-windows.png)
- [R12.7 mm, 5°, 1 mm/s interrupted tracking](plots/arc-04-r12.7-a5-1mms-trial1-tracking.png)

All figures were regenerated after verified-rest disable. Five circle/velocity-window tests and Ruff checks of all three new scripts pass. The best-run tracking figure was visually inspected. No production controller/SDK source was edited. No stage has been marked DONE.
