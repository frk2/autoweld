# Recorded-pair hand-eye review

The 25 retained native image/TCP pairs can support a numerically valid hand-eye fit after a deterministic image-geometry correction of the checkerboard's 180° ordering symmetry in views 3, 4 and 5. No new robot motion or image acquisition was used. The original recorded ordering fails the production stationary-board gate; it must not be silently passed to the solver unchanged.

The normalized candidate passes the existing production gates and all five interleaved 20-train/5-heldout tests. It is saved only in this evidence directory, **not installed**. Its large correction from the nominal optical TCP must be interpreted correctly before using it in robot geometry.

## Data and frames

Source: `runtime/agents-live/evidence/robocam-1792796400030413285/checkpoint.json`; SHA256 `3ccc2c4fa3e60a858c0e0a7b652a492ecd2778d3fdd501f891dd40d935fef937`.

Intrinsics: `runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json`; file SHA256 `cb792766d770eda303d4b06a6b7d0092bd048dfd296db4359e64bed1f134a891`. Canonical intrinsic-calibration identity (SHA256 of sorted compact JSON) is `223fddd01c436d8d52d891a37f86d6ddaf9b0fa203a3e4bbe965a3e222bc2659`.

All 25 captures have 35 retained native corners, native-validation metadata, unique frame sequences and `robocam.base_from_tcp`; source `spinnaker:25440983`, resolution 1440×1080, checkerboard 5×7 interior corners at nominal 4 mm spacing. The same newly installed intrinsic model was used to recompute every target pose from these original native corner measurements. Camera/target geometry uses millimetres at OpenCV PnP; target translation is explicitly divided by 1000 before AX=XB. Robot transforms and fitted extrinsics use metres and column vectors.

`base_from_tcp` is the service's active `weld_tcp`, whose nominal geometry is the camera-optical frame in the mount manifest. Accordingly, the result is **nominal_tcp_from_actual_camera**, not flange_from_camera. The composition is:

```
base_from_target = base_from_nominal_tcp @ nominal_tcp_from_camera @ camera_from_target
flange_from_camera = flange_from_nominal_tcp @ nominal_tcp_from_camera
```

The original synchronized-capture coordinator records a measured pose before capture and checks that the robot remains ready and within 0.5 mm / 0.005 rad afterward. It also enforces a fresh frame acquired after settling. However, this checkpoint's persisted capture metadata omits read_started, capture cutoff, acquisition-session ID, before/after feedback and their timestamps. Therefore these files preserve paired native image/pose measurements and their acceptance provenance, but do not independently reconstruct hardware exposure/robot clock synchronization. The retrospective intrinsic identity is legitimate for recomputing PnP; it was not an installed calibration identity at each exposure.

## Corner association before fitting

A plain rectangular checkerboard has a 180° origin ambiguity. Raw native grid-X image angles in views 0–2 and 6–24 are +92.22° to +94.73°; views 3–5 instead report −87.86°, −88.26°, −88.43°. Those three swap the visible first/last physical corners while the robot moves only through a compact cone.

The association rule was fixed from image geometry, before solving: anchor grid X to view 0, reverse the entire 35-corner order only if its normalized X direction has dot product **< −cos(20°)** with the reference, and otherwise require **> cos(20°)**. This selects exactly views 3–5. Every normalized dot is at least 0.999169. No residual, nominal transform or hand-eye result was used to select a flip. Original images, original native corners and raw paired transforms are preserved.

The reference physical origin is the visible view-0 corner at pixel **[1038.8, 398.0]**, with opposite final corner **[287.2, 849.5]**. Normalized views 3–5 select that same side of the board:

| View | Raw first corner px | Raw final corner px | Normalized first corner |
|---|---|---|---|
| 3 | [325.4, 933.8] | [1077.3, 485.1] | raw final corner |
| 4 | [281.4, 958.1] | [1029.8, 509.4] | raw final corner |
| 5 | [245.7, 951.7] | [989.2, 503.8] | raw final corner |

Maximum adjacent centroid displacement is 90.47 px and maximum adjacent corresponding quadrilateral-corner displacement is 93.62 px; grid-X spans remain 447.69–483.86 px. The complete run's maximum robot pair rotation is 15.1335°, excluding a physical 180° turn. Exact first/last corners, centroids and quadrilaterals are in `assessment.json`. Native image evidence is retained in the checkpoint and exported unchanged under `../intrinsics-review/live-cone25/standard/native-*.png`.

The uncorrected samples fail at **12.06 mm / 59.64°** stationary-board RMS. Reversing every normalized view together simply chooses the opposite board origin and changes fitted hand-eye by only 0.00116 mm / 0.000457°. Injecting one isolated 180° reversal is rejected at 7.69 mm / 35.10°, confirming that mixed physical origins are not harmless.

## PnP and observability

All iterative PnP solutions have positive depth for every corner; minimum corner depths range 129.13–140.81 mm. Native reprojection RMS ranges 0.56636–0.64068 px, below the existing target-pose 1.5 px gate. Independent planar IPPE branches have best RMS 0.57479–0.66278 px and alternative RMS 1.56759–6.60804 px. The second planar branch is consistently inferior and exceeds that gate; no per-image branch was chosen using hand-eye residuals.

There are 300 relative robot pairs. Rotation excitation RMS singular directions are 7.68369°, 7.61690°, 0.35553°; the second direction exceeds the existing 2° requirement. Translation design singular values are 3.26366, 2.32028, 2.30018, condition number 1.41887 and full rank. The narrow cone therefore supplies useful two-axis hand-eye excitation for this dataset, although it does not replace independent wider-workspace validation.

An independent nonlinear rotation-matrix AX=XB least-squares solve initialized at identity, followed by independent translation least squares, agrees with the production rotation-vector solver within **0.0000282° / 0.0000758 mm**.

## Candidate and heldout validation

Production `solve_handeye` on the normalized 25 pairs gives stationary-board RMS **0.58868 mm / 0.32891°**, maximum per-view residual **0.92819 mm / 0.55552°**. The unchanged production gates are at least 20 paired views, one source/calibration identity, sufficient two-axis rotation, full-rank translation design, stationary-board translation RMS ≤3 mm and rotation RMS ≤1.5°.

Each fold fits only its 20 training pairs. Its board reference is the training set's mean board position and mean rotation; the five heldout board poses are tested against that training reference. No hand-eye transform or board reference is fitted to heldout samples.

| Fold | Heldout RMS mm / deg | Heldout maximum mm / deg | Transform change vs all25 mm / deg |
|---|---|---|---|
| 0 | 0.58317 / 0.32187 | 0.94343 / 0.46169 | 0.20790 / 0.07567 |
| 1 | 0.54891 / 0.33965 | 0.76543 / 0.49461 | 0.17634 / 0.03605 |
| 2 | 0.48553 / 0.35463 | 0.65599 / 0.56252 | 0.07637 / 0.06438 |
| 3 | 0.58442 / 0.33147 | 0.88038 / 0.51627 | 0.14312 / 0.04690 |
| 4 | 0.74679 / 0.30607 | 0.91605 / 0.46655 | 0.25221 / 0.12201 |

Aggregate heldout RMS is **0.59605 mm / 0.33114°**; every fold and even every individual heldout sample is within 3 mm / 1.5°. This is validation of the hand-eye fit conditional on fixed intrinsics trained on all 25 images; it is not a fully independent validation of the intrinsic model.

Candidate `tcp_from_camera`, translations in metres:

```
[[-0.503714470, -0.858373226,  0.097299218,  0.025975396],
 [ 0.863868569, -0.500727575,  0.054799554,  0.022594951],
 [ 0.001681932,  0.111657065,  0.993745375, -0.012906882],
 [ 0,            0,            0,            1           ]]
```

This is a **36.77 mm** origin correction and **120.35°** rotation from the nominal optical TCP. Its actual optical +Z is only 6.41° from nominal +Z, so the large rotation is principally image-plane roll. Low residuals and the independent solver support the recorded frame relationship, but do not identify why the nominal mount/image convention differs by that roll and translation. Applying this matrix as though it were flange_from_camera, or applying it twice to an already corrected TCP, would be wrong. Reconcile the active nominal TCP/mount convention before installation or geometry changes; do not reject or accept it solely on total rotation magnitude.

## Scope and retained artifacts

This supports reusing the existing pairs for an evidence-backed hand-eye candidate without another 25-view motion sequence. It does not establish independent millimetre accuracy, printed square dimensions, target flatness, absolute robot kinematic accuracy, full workspace performance or hardware clock synchronization. The same unmarked board cannot identify a universally named physical origin in an unrelated future run without an explicit origin convention; within this compact run, geometry makes the three ordering flips unambiguous.

`candidate.json` contains the normalized full-data result; `paired_samples.json` contains associated poses; `original_paired_samples.json` preserves uncorrected poses; `assessment.json` includes per-view, association, alternative-PnP, excitation and heldout diagnostics. `assess_recorded_pairs.py` reproduces the offline analysis from retained files. None of these is an active installed hand-eye calibration. No service, physical camera, robot connection or robot model was started.
