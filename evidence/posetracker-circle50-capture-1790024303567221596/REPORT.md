# Circle with complete PoseTracker detection coverage

One R60 mm YZ circle completed at 50 mm/s using the selected POS_VEL RAM gains. The arm started and finished at the existing circle endpoint; no seed, approach or rest-return move was needed. Video1 used exposure 150, native 640×480 capture and the existing calibrated 2× detector crop. Gripper target remained closed (0) throughout; measured closure was 0.03271–0.03284 and reported effort 1.023–1.038 Nm. No SDK communication, oscillation or motion fault occurred.

The 8.131 s circle had **542/542 valid camera poses (100%)**. The entire 24.36 s recording had **1,626/1,626 valid poses**, about **66.70 fps**, and zero capture-sequence skips reported by PoseTracker. This does not count any frames lost internally by the camera/USB driver. Previous completed camera run had 85.06% circle pose coverage.

## Apparent shaking

These measurements describe the ChArUco board center in camera coordinates. The board-to-TCP transform is unmeasured. Independent, unsmoothed per-frame pose fits were detrended and filtered at 2–20 Hz; continuous data cover essentially the whole circle. Camera pose noise, rolling-shutter distortion and board flex remain mixed with physical motion.

| Interval | Band vector RMS | Camera X/Y/Z peak-to-peak |
|---|---:|---|
| Stationary before | 0.083 mm | 0.170 / 0.112 / 0.485 mm |
| Full circle including ramps | 0.339 mm | 1.468 / 0.978 / 1.978 mm |
| Interior cruise, excluding ramps plus 1 s each side | 0.267 mm | 1.245 / 0.451 / 1.166 mm |
| Stationary afterward | 0.128 mm | 0.211 / 0.132 / 0.710 mm |

Interior cruise contains 4.95 s of data. Its largest spectral component is near 5.0 Hz; this short recording does not establish a mechanical resonance. Camera and robot use host timestamps, not hardware exposure synchronization. These are not calibrated tip accuracy or backlash measurements.

## Path and capture checks

Encoder/FK peak distance to the requested circle was **1.346 mm**; peak timed tracking error was **11.107 mm**. The full motion and final hold completed, with no 10 mm geometric deviation flag. Raw feed/sampling qualification still failed (47.05% peak raw speed error, one invalid velocity sample). The selected safe circle preset remains 10 mm/s; this requested 50 mm/s experiment does not change it.

Pose CSV is complete for all processed frames. The optional asynchronous lossless PNG archive saved **1,197/1,626 images (73.6%)**; its queue dropped 429 image copies because PNG storage could not keep up. No PNG write errors occurred and the writer drained successfully. This affected image archival only, not the 1,626 pose estimates. `images.jsonl` maps each saved image to its one-based processed CSV row. The archive must not be described as containing every frame.

Two preceding runner attempts are excluded: `../posetracker-circle50-capture-1790024203671507774/` stopped after preview because it read status before fresh publication, with no motion sent; `../posetracker-circle50-capture-1790024246093748946/` sent a circle but immediately commanded Hold on a logging error (seven tracking samples, max edge 0.134 mm). Both runner issues were corrected before this completed run. No gains, motion guards or camera pose acceptance thresholds were changed.

Current state: ENABLED at circle endpoint with powered closed grip, no pending/active motion, POS_VEL speed ceiling restored to 0.15 rad/s, automatic excursion bound 0.49 rad. PoseTracker stopped and video1 was released. Do not disable or open the gripper while the board depends on its grip.

Files: [overview plot](overview.png), [pose CSV](poses.csv), [board-center CSV](board-center.csv), [summary](summary.json), [filter report](analysis/report.json), [robot trace](circle60-50mms.jsonl), [image storage counts](image-storage.json), [image map](images.jsonl), and [final status](final-status.json).

Follow-up: [matched encoder-versus-camera shaking analysis](encoder-comparison/REPORT.md). The encoder-derived TCP gives 0.204 mm RMS versus camera 0.341 mm over matched trusted circle samples, with correlated lateral motion and weaker depth agreement. Measurement points differ; this comparison does not identify backlash or independently isolate mechanical vibration.
