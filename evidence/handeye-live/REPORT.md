# Fresh live hand-eye validation

The completed fresh 25-pair run passes the existing hand-eye gates and all five heldout splits. Its transform closely reproduces the previous intrinsic-run-derived candidate: **0.65849 mm translation and 0.07169° rotation difference**. This independently confirms the recorded compact-cone frame relationship using fresh image/robot observations. The review did not install a calibration, modify active files, contact the service or access hardware.

## Provenance and audit checks

Active source archive: `runtime/agents-live/handeye.json`, schema 2, 25 paired samples, file modification epoch 1792799363.8258338. The exact reviewed archive is preserved as `archive-snapshot.json`, SHA256 `65086310825272fd9c034292d15ea3692f44248fb61e0c3a5fcde53915459fe2`.

Camera source is `spinnaker:25440983`, native resolution 1440×1080, 5×7 interior corners with nominal 4 mm spacing. All 25 pairs have full finite 35-corner native observations, positive in-image coordinates, native-validation flags, raw and normalized target poses/corners, grid/configuration, camera controls and acquisition metadata. Fixed intrinsics match `runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json`, file SHA256 `cb792766d770eda303d4b06a6b7d0092bd048dfd296db4359e64bed1f134a891`; canonical sorted-JSON calibration identity is `223fddd01c436d8d52d891a37f86d6ddaf9b0fa203a3e4bbe965a3e222bc2659`.

The samples share acquisition generation 3/session `a17ff3c89bea40d3ac69aca71b24fe4e`. Native sequence IDs advance uniquely from 12823 to 23740 over 184.860 seconds of host acquisition-start time. The timing session/frame IDs and host read-start/read-completion fields exactly match each sample's corresponding metadata. Recorded camera timestamps are present; no duplicate or regressing frames, reconnect flags or metadata errors were reported. The cumulative native frame-ID gap count is 1; this is not 25 contiguous sensor frames, because calibration samples intentionally select individual settled frames from a continuous stream.

All host acquisition starts are **48.15–286.13 ms after the required settled+0.4 s cutoff**. Every host read finishes before the recorded post-capture robot check. Maximum measured pre/post robot change is **0.26930 mm / 0.04887°**, below the coordinator limits of 0.5 mm / 0.005 rad, and all post-checks report ready. Acquisition-start-to-postcheck latency is 129.61–167.41 ms.

The frame's host read start precedes the host pre-pose reading by **1.82–17.21 ms** in this latest-frame path; the review does not claim the exposure is bracketed by those pose readings. Acquisition timing explicitly reports `camera_timestamp_semantics=image_timestamp_event_unknown`, unavailable camera-to-host clock mapping, unknown exposure host start/end and `motion_pairing_supported=false`. These checks establish host read freshness during a settled hold and bounded pre/post movement. They do **not** establish independently synchronized hardware exposure/robot timestamps or support calibration while moving.

Controls remain native auto exposure, 16562 µs exposure and 0 dB gain. Native target reprojection RMS is **0.57120–0.63281 px**. Fresh hand-eye archives preserve corner/pose metadata rather than raw fresh image pixels; this review cannot independently inspect fresh image blur, target flatness or paper dimensions from this archive.

## Deterministic checkerboard association

The live path reversed the entire grid only for views **3, 4 and 5**, using the first-view grid-X image axis as reference and selecting reversal before fitting when dot < −cos(20°) and actual TCP rotation from the reference is <20°. Other views require dot > cos(20°), with matching coherent Y orientation. Maximum recorded TCP rotation from the reference is 15.00654°.

The review recomputed the correction for every sample and verified:

```
normalized_camera_from_target = raw_camera_from_target @ board180
board180.rotation = diag(-1, -1, 1)
board180.translation_m = [0.016, 0.024, 0]
```

For unreversed views the pose remains unchanged; for reversed views the 35 corner indices reverse together. Every stored normalized pose matches this transform to 1e-12 and every normalized corner list matches its exact raw/reversed ordering. Original data remains in the archive. No fitted residual or expected calibration matrix was used to choose flips. The repeated three-view ordering discontinuity is therefore handled explicitly rather than silently treated as physical board motion.

## Full fit and heldout checks

Offline `solve_handeye` recomputation from the 25 archived normalized pairs matches the saved matrix to 1e-12. Full-data stationary-board RMS is **0.53076 mm / 0.35099°**. The unchanged production gates are at least 20 same-source/calibration paired observations, sufficient two-axis rotation, full-rank translation design, position RMS ≤3 mm and orientation RMS ≤1.5°.

Each interleaved fold fits 20 training pairs only and computes the stationary-board mean position/orientation from those training pairs. Its five withheld board poses are evaluated against that training reference without fitting either the transform or board reference to withheld pairs.

| Fold | Heldout RMS mm / deg | Heldout maximum mm / deg | Transform change vs full mm / deg |
|---|---|---|---|
| 0 | 0.50825 / 0.32488 | 0.68313 / 0.44339 | 0.07228 / 0.02037 |
| 1 | 0.51503 / 0.34811 | 0.72376 / 0.55557 | 0.04886 / 0.05428 |
| 2 | 0.43553 / 0.39255 | 0.58052 / 0.61473 | 0.09017 / 0.07306 |
| 3 | 0.58035 / 0.35446 | 0.77311 / 0.55149 | 0.14135 / 0.07681 |
| 4 | 0.63604 / 0.33808 | 0.72754 / 0.51242 | 0.18289 / 0.07344 |

Aggregate heldout RMS is **0.53937 mm / 0.35235°**; worst individual residual is **0.77311 mm / 0.61473°**. Every fold and every individual withheld observation is within 3 mm / 1.5°. Maximum fold transform change relative to the full fit is **0.18289 mm / 0.07681°**.

## Comparison with the earlier candidate

Earlier candidate: `evidence/handeye-review/candidate.json`, derived from the original 25 intrinsic images and their paired poses using the same installed intrinsics and documented image-geometry association. The fresh full-data fit differs by **0.65849 mm / 0.07169°**.

Applying the earlier candidate unchanged to all fresh pairs, while estimating only a fresh stationary-board reference, gives **0.52106 mm / 0.35111°** RMS and **0.76792 mm / 0.61334°** maximum. It passes unchanged gates on these new observations. The tiny translation-residual difference between old/new is not evidence to replace the fresh result: the production solver first estimates rotation and then translation from relative equations, rather than directly minimizing stationary-board position RMS alone.

Fresh `tcp_from_camera`, translations in metres:

```
[[-0.504717265, -0.857740435,  0.097682285,  0.025906753],
 [ 0.863282804, -0.501712173,  0.055024496,  0.022649280],
 [ 0.001811656,  0.112099250,  0.993695364, -0.012254233],
 [ 0,            0,            0,            1           ]]
```

This is a correction from the service's **nominal camera-optical weld_tcp** to the actual measured camera frame. It is not flange_from_camera and must not be applied twice. Its origin correction is 36.528 mm and total rotation 120.422°, primarily image-plane roll; actual optical +Z differs from nominal +Z by 6.437°. Independent fresh data reproduces the previously observed large roll, rather than explaining its nominal CAD/image-axis origin. Any physical mount or motion-frame update must use the correct transform composition.

## Scope

These independent fresh pairs substantiate repeatability and stationary-board consistency in the prescribed compact cone. The fixed intrinsic model was trained on the earlier dataset; fresh image/TCP observations are independent of that earlier run, but share its intrinsics and the same robot kinematic model. This is not a dimensional qualification of printed paper, independent robot kinematics, wider workspace or full sensor image extremes, nor hardware time synchronization.

`assessment.json` includes per-view metadata checks, movement deltas, fold results, old-candidate cross-checks and provenance. `assess_fresh_archive.py` reproduces the offline review. The active archive and hardware remained untouched by this reviewer.

## Service operation and final state

The primary agent ran this fresh sequence through service JSON requests and web
controls. No standalone arm control or separate path-tracking workflow was used.
The service's normal planner checked each movement. Approach and return took
approximately 174 seconds each; all 25 captures used the prescribed cone.
Maximum sampled motor temperature was **39°C**, with no SDK/motor fault and no
60°C cooldown trigger. No corrective checkerboard hunt steps were needed.

The corner-order fix passed the 701-test broader run and final 125-test focused
calibration/persistence/UI run; changed Python files passed Ruff. Original native
detections are retained alongside deterministic association and timing metadata.

After saving calibration, service shutdown returned to captured rest and verified
fresh stationary rest for one second before SDK motor disable. Maximum final
rest error was **0.02186°**. See `rest-disabled-state.json`. The service exited
with code 0, then restarted at port 8082 using `--start-disabled`; default planning
limits were restored. The new service reloaded the matching 25-view hand-eye
result with motors disabled and camera off, without broadcasting its raw archive.
`final-service-state.json` records that check; its UI pose is a saved rest display.

Installed archive: `runtime/agents-live/handeye.json`. Its SHA256 is
`65086310825272fd9c034292d15ea3692f44248fb61e0c3a5fcde53915459fe2`.
The compact measured transform is available in [handeye.json](handeye.json).
