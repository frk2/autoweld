# Encoder versus camera shaking

The encoder positions contain motion correlated with the camera shaking. Their modeled tool-tip displacement is about **0.204 mm RMS**, versus **0.341 mm RMS** from the camera over the same trusted portion of the circle: approximately 40% lower, but the same order of magnitude.

All values below are 3D vector RMS after the same fourth-order, zero-phase 2–20 Hz filter. They are displacement fluctuations, not absolute path error or peak-to-peak motion.

| Matched interval | Camera board center | Encoder-derived model TCP | Encoder-derived gripper mount |
|---|---:|---:|---:|
| Stationary before | 0.083 mm | 0.092 mm | 0.057 mm |
| Circle, 7.12 s trusted | 0.341 mm | 0.204 mm | 0.117 mm |
| Interior cruise, 4.95 s | 0.267 mm | 0.173 mm | 0.100 mm |
| Stationary afterward | 0.128 mm | 0.077 mm | 0.015 mm |

The original complete-circle camera value, **0.3386528 mm RMS**, was reproduced exactly. A 105–110 ms encoder-log gap immediately before reference motion requires splitting the encoder record and excluding a one-second filter boundary. The matched comparison consequently excludes about the first second of the circle; its camera value becomes 0.341 mm. No interpolation across this gap is included in the trusted result.

## Correlation

For illustration, a rigid rotation was fitted between the **slow (<1 Hz)** encoder TCP and camera board-center paths. High-frequency fluctuations were not fitted or amplitude-scaled. This is only an approximate coordinate alignment: the slow-path fit has 12.02 mm RMS residual because these are different physical points and the board/tool transform is uncalibrated.

On the matched circle, in these approximate camera axes:

| Axis | Camera RMS | Encoder TCP RMS | Same-time correlation |
|---|---:|---:|---:|
| Camera X, sideways | 0.217 mm | 0.184 mm | 0.770 |
| Camera Y, vertical in image | 0.113 mm | 0.057 mm | 0.704 |
| Camera Z, depth | 0.238 mm | 0.068 mm | 0.218 |

The combined vector correlation is 0.556 without a time adjustment. Searching camera delay from −150 to +150 ms in 5 ms steps yields 0.683 at +20 ms. This fitted delay is exploratory, not a calibrated exposure timestamp or an independently validated correlation. The same search over interior cruise yields 0.666 at +20 ms. The rotation-independent correlation between instantaneous displacement magnitudes is 0.485 for the circle and 0.463 for cruise.

Camera X also correlates directly with robot-world encoder Z at 0.772 (0.798 in cruise), without fitting a camera/base rotation. Together with the overlaid waveforms, this supports a shared motion component. Most additional camera variation appears in depth, where encoder agreement is weaker. It cannot yet be assigned uniquely to depth-estimation noise, rolling shutter, board/gripper flex or motion not resolved by the joint encoders.

## Sampling and model limits

- Each joint was independently placed on the common timeline using its feedback timestamp, deduplicated, then linearly resampled before SDK-model forward kinematics. The encoder log is about 50 Hz; the camera is 66.7 Hz. Resampling does not add encoder bandwidth. Gaps over 50 ms split the signal, with a one-second guard at each filter boundary.
- A 2–10 Hz sensitivity check gives camera/encoder TCP RMS **0.315/0.200 mm** for the matched circle and **0.237/0.170 mm** for cruise. The conclusion therefore persists away from the encoder log's upper frequency range.
- Camera measures the board center. The model TCP is 150 mm along mount −Z; the gripper mount is a third point. Rotational motion changes translational vibration with distance from the mount. A measured board-to-tool transform and camera-to-base calibration are needed for an exact comparison at the same point.
- Encoder quantization and interpolation contribute to its estimated fluctuation. Camera pose noise contributes to its result. Neither RMS alone proves mechanical vibration of that amplitude. Do not subtract these RMS values and label the difference as physical backlash or flex.
- Directed joint 2–20 Hz RMS during the matched circle is J1–J6: **0.00638°, 0.00809°, 0.01382°, 0.02010°, 0.01206°, 0.00719°**. Joint RMS values alone do not rank TCP contributions because their lever arms and correlations differ.
- Offline FK exactly reproduced the stored tracking positions at ten sampled checkpoints (zero numerical difference). The camera analysis reproduced the previously reported RMS to numerical precision. Ruff passes. Analysis was offline; no robot commands, settings or gains changed.

[Comparison plot](comparison.png) · [Numerical metrics and lag scans](metrics.json) · [Aligned circle waveforms](circle-signals.csv) · Reproduce with `scripts/compare_encoder_camera.py <capture directory>`.
