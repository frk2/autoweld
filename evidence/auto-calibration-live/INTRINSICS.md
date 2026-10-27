# Live automatic intrinsic calibration

Completed on 2026-10-23 using FLIR camera **25440983**, native **1440 × 1080**
images, and a **5 × 7 inner-corner checkerboard with 4 mm squares**.

The saved Calibration Pose brought all 35 corners into view. Automatic capture
accepted **25 native images** at the unique edge points of the 15° full-opening
cone, 150 mm high and 19.748 mm in radius. Every capture passed native corner
validation. The camera acquired at approximately 59 fps with automatic exposure.

The production solver saved and installed the standard model:

- Fit RMS: **0.606839 px**.
- Five-fold held-out aggregate RMS: **0.607169 px**.
- Worst held-out view RMS: **0.640651 px**.
- Focal-length variation across folds: **0.356% / 0.351%**.
- Estimated focal-length relative standard deviation: approximately **1.51%**.
- Measured board-normal separation: **15.319°**, with two-axis excitation.
- All existing quality gates passed; the offline assessor reported no warnings.

Camera matrix, in native pixels:

```text
3978.148250       0.000000  603.169730
   0.000000    4028.593813  391.819556
   0.000000       0.000000    1.000000
```

The installed model is
`runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json`.
The [portable copy](intrinsics.json) includes distortion, source identity, image
size and board dimensions. See [validation metrics](intrinsic-validation.json)
and the [independent review](../intrinsics-review/live-cone25/REPORT.md), including
the fixed-k3 comparison and limits from paper flatness and unmeasured sensor edges.
This is intrinsic calibration; hand-eye and laser-plane calibration remain separate.

Source checkpoint:
`runtime/agents-live/evidence/robocam-1792796400030413285/checkpoint.json`.
All 25 exported native PNGs were verified byte-identical to the checkpoint.
The [completed service snapshot](intrinsic-complete-state.json) records the saved
result before shutdown released calibration ownership. No corrective hunt steps
were needed, so this run does not establish a live vector-hunt correction test.

The approach and return used validated slow paths, each approximately 174 seconds,
with peak planned joint speed approximately 0.0235 rad/s. The maximum sampled
motor temperature was **39°C**; cooldown at 60°C was never triggered, and no SDK
or motor fault was observed. After calibration, the camera was switched off and
dedicated shutdown returned to the controller's captured rest, verified fresh
stationary rest for one second, and completed SDK motor disable. Maximum final
rest error was **0.02186°**. See the
[verified-rest shutdown snapshot](intrinsic-rest-disabled-state.json).

The service was restarted on port 8082 with `--start-disabled`, restoring its
default planning limits. The arm remains at rest with motors disabled and camera
off; its disabled UI displays the saved rest record rather than new SDK feedback.
