# Live 25-view intrinsic calibration assessment

The installed standard model passes the existing calibration gates and independent held-out corner checks on this dataset. The fixed-k3 comparison offers no measured quality advantage that warrants replacing it. No acceptance threshold was relaxed.

This report assesses retained camera data only. The assessment did not acquire camera frames, command the robot, alter the installed model, or change camera controls.

## Dataset and native-image provenance

- Source: `spinnaker:25440983`, monochrome native resolution **1440 × 1080**.
- Configured printed target: **5 × 7 inner corners, 4 mm square spacing**.
- **25 distinct capture IDs and 25 distinct frame sequences**, with all **35 measured native corners** accepted in every view: **875 corner measurements** total.
- Session: `runtime/agents-live/evidence/robocam-1792796400030413285/`.
- Checkpoint: `checkpoint.json`; session settings explicitly require **25 accepted views**.
- Capture interval: **2026-10-23 23:00:08.646–23:02:52.235 UTC**, frame sequences **12243–21886**.
- Every capture records `native_validation: true`. Live detection used half-size search, but accepted captures revalidated and measured the original native frame.
- Recorded controls are consistent across the set: **16,562 µs exposure, 0 dB gain**, native automatic exposure enabled. No saturated pixels were found in the retained images. Their 95th-percentile brightness is **33–37 DN** on the 8-bit scale; they remain relatively dark.
- All **25 exported PNGs under `standard/` match the retained native PNG bytes exactly**. Per-image SHA256 checks are in `comparison.json`. The checkpoint's measured corner arrays and original embedded images were preserved.

## Quality gates and measured diversity

The production SDK requires the configured view count, valid native full-grid captures, matching camera/resolution, at least **12° maximum board-normal separation**, and finite calibration parameters. The intrinsic solver rejects RMS above **1.5 px**; the integrated installation gate additionally requires **RMS ≤ 1 px**.

Both assessed models pass those gates. This dataset has meaningful variation beyond repeated centered images:

- Maximum recovered normal separation: **15.32°**.
- Second-axis normal spread: approximately **5.07°**.
- All measured corners span **66.35% of image width** and **68.74% of image height** in aggregate; their bounding coordinates are approximately **x = 165–1121, y = 246–989 px**.
- View centroids span **16.32% of image width** and **22.18% of image height**.
- The closest pair of views differs by approximately **15.52 px mean corner displacement**. There is no near-duplicate pair under the legacy 5 px criterion.
- Normalized homography constraints have fifth/first singular-value ratio **0.00707** and nullspace gap **83.73**, supporting a resolved intrinsic constraint system rather than an observed rank collapse.

Thus, this particular strict cone dataset does not exhibit the centered, low-excitation failure mode considered before acquisition.

## Standard versus fixed-k3 comparison

Five interleaved folds each fit **20 training views** and evaluate **five held-out views**. A held-out view contributes no measurements to its intrinsic fit. Its board pose is estimated from its own corners before calculating reprojection error; this tests consistency of the camera model on unseen views, rather than independently establishing camera-to-robot or dimensional accuracy.

| Metric | Standard, installed | Fixed k3, comparison only |
|---|---:|---:|
| Full-set RMS | 0.606839 px | 0.610218 px |
| Full-set per-view RMS range | 0.566361–0.640684 px | 0.568613–0.645188 px |
| Held-out aggregate RMS | **0.607169 px** | 0.610584 px |
| Worst held-out view RMS | **0.640651 px** | 0.645059 px |
| Focal-length relative standard deviation, fx / fy | 1.514% / 1.509% | 1.504% / 1.500% |
| Focal-length range across folds, fx / fy | 0.356% / 0.351% | 0.274% / 0.274% |
| fx range across folds | 3969.611–3983.771 px | 3951.504–3962.358 px |
| fy range across folds | 4019.875–4034.026 px | 4001.261–4012.238 px |
| Existing gates | Pass | Pass |
| Assessor heuristic warnings | None | None |

The standard model's small training advantage persists in held-out errors. Fixing k3 produces a slightly more stable but approximately **0.54% lower focal estimate**, with a negligible approximately **0.0034 px** held-out RMS disadvantage. Neither comparison supports a claim of exact optics or metric precision. It supports retaining the installed standard model within the measured region.

## Installed-model consistency

Installed file:

`runtime/agents-live/runtime/flir/data/calibration-efc57b2d88550b5f.json`

Its source, resolution and board settings match the checkpoint. The checkpoint's stored camera-model identity matches the installed file. Production and offline standard fits agree to **less than 8.4 × 10⁻⁷ px** in K and **1.2 × 10⁻⁷** in distortion coefficients; their RMS values agree to numerical precision.

```text
K = [[3978.1482500474913, 0, 603.1697298456295],
     [0, 4028.5938127964005, 391.8195559791731],
     [0, 0, 1]]

D = [0.3527889636820804, -22.428166284922618,
     -0.007246447142711492, -0.015231261928504972,
     358.0961751249878]
```

Installed-file SHA256 at assessment:

`cb792766d770eda303d4b06a6b7d0092bd048dfd296db4359e64bed1f134a891`

## Scope and physical sanity

Positive focal lengths, a principal point inside the image, resolved two-axis tilt, close held-out/training errors and stable focal estimates are internally coherent. A sampled 41 × 31 sensor-domain mapping check found no local fold: the standard model's distortion Jacobian determinant ranged from **0.974 to 1.488**, remaining positive at sampled points. Its approximate field of view is **20.52° × 15.27°**.

The high numerical k2/k3 values should not be judged in isolation: they multiply small powers of normalized ray radius and partially cancel within the measured region. However, they make extrapolation important. Standard and fixed-k3 models disagree by up to approximately **0.96° in inferred ray direction at unmeasured sensor extremes**, versus at most approximately **0.10° at retained corner locations**. Similar residuals on the available board do not validate the entire sensor border. The principal point is approximately **117 px left and 148 px above sensor center**; this is a model estimate, not an independently verified optical-center measurement.

Acceptance here means a coherent intrinsic model with approximately **0.61 px** reprojection and held-out error for the captured native views. It does not establish dimensional accuracy in millimeters, paper flatness or exact printed square spacing, full-sensor extrapolation accuracy, lens-distortion truth, or a hand-eye transform. Those require independent evidence. Reuse also depends on unchanged lens, focus, filter, crop/binning and sensor resolution.

## Retained analysis artifacts

- `standard/assessment.json`: standard fit, view errors, conditioning, uncertainties, coverage, folds and native-image metadata.
- `standard/native-*.png`: 25 exact native captures.
- `fix-k3/assessment.json`: fixed-k3 comparison using the same measured dataset.
- `comparison.json`: sampled mapping/extrapolation comparison, installed-model agreement and native PNG byte checks.
