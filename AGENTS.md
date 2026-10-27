# Auto welder

The idea is to build a welding assistant robot using a i2rt YAM robotic arm. Step 1 is to allow it to mount a laser line weld finder and recontruct pointclouds based on detected geometry.

## Robotic arm
You can find everything at https://github.com/i2rt-robotics/i2rt

This robot is being controlled by the canable pro USB to Can dongle. The robot is connected and live - but the SDK control is safe. Feel free to issue commands.

## General rules and optimizations:
- Use only a single robot model as it consumes memory.
- Only a single camera frame grabber can be active at any time. Make sure cameras are closed properly and do not leak memory.
- Multiple browser connections should not instantiation multiple modles and objects
- This is a python process which will serve Viser. Download the i2rt SDK.
- Use a rviz style visualization or gym sort of environment to visualize the robot
- Do not use ROS.
- Use Viser. Follow the example in examples/control with viser in the i2rt github repo. This example works perfectly.
- Headless testing using playwright should be supported.
- Use imports wisely. Imports increase RAM usage.
- SDK Failures where motors turn off are deadly. These are to be debugged and diagnosed and should be avoided at all costs.

Spawn subagents to handle these self contained parts of the code base:

## Robot control
This will be a python singleton class that controls the robot behind the scenes. Driven by the UI code and reused by other parts of the UI.

- **All routine arm movement must go through the running service or web frontend.** Submit service JSON requests or use the web controls; do not command the arm directly from standalone SDK scripts or another robot owner. Direct arm control is reserved for an emergency when the service cannot safely control the arm.
- **No separate path-tracking workflow is required.** Use the service's discrete joint/TCP pose commands for routine moves and calibration. Keep the service's existing motion planning and physical protections.
- **Use the configured default speed ceilings for all normal motion:** **3 m/s TCP and 2 rad/s per joint**, including saved poses, calibration, checkerboard hunting and shutdown return to rest. Do not substitute commissioning speeds or add hidden per-operation slow caps unless the user requests them. Existing acceleration, jerk, angular-speed, joint, collision, feedback, fault and thermal protections still constrain each planned move. An already-running profile completes under the limits with which it was validated.
- Use position-speed mode for Hold, IK and planned motion, with default TCP speed **3 m/s** and default per-joint speed **2 rad/s**. Keep the planner and position-speed ceiling consistent with these defaults. These are speed ceilings; acceleration, jerk, installed joint limits, collision checks, feedback, fault and thermal protections still apply. Never change an active move's limits; apply speed changes only when the arm is stationary and the service accepts them. Selecting Gravity compensation must automatically switch all six arm motors through the SDK to MIT torque mode and use the installed receiver/camera gravity model. Leaving Gravity compensation must capture the measured pose and automatically return to position-speed control. Tune gravity compensation on live free-space poses; exclude supported rest/contact poses from fitting.
- Operation modes are hold, gravity comp and IK. Should be switchable on the fly.
- Before normal motor disable, always return to the captured rest and verify it.
- **Shutdown must have a dedicated, unconditional request path:** `Controller.shutdown()` / JSON `shutdown`. Always accept it from the UI, filesystem and process signals, including during normal motion, calibration, cooldown, an old pending-disable latch or a previous shutdown failure. Bypass application busy/mode gates, prior targets and the automatic commissioning excursion limit. Cancel queued targets and calibration ownership; capture a controlled measured hold, wait for fresh quiet feedback, then plan the return to captured rest using the configured default speed ceilings. Healthy duplicate requests must not restart progress or settling. Failed shutdowns must expose a clear retryable state without stranding the controller behind a stale latch; the shutdown/retry UI must remain available. Hold must be able to pause shutdown safely.
- Catch **SIGHUP**, SIGINT and SIGTERM and enqueue the same dedicated shutdown request. If the UI or serving port is stuck, `kill -HUP <service-pid>` must request the controlled return to rest and verified motor disable before process exit. The signal handler must not perform SDK calls or abruptly close the robot.
- Shutdown still preserves physical protections: finite/fresh SDK feedback, installed joint limits, collision checks, controlled speeds, temperature monitoring and actual SDK fault handling. Never blindly clear hardware faults or remove torque away from verified rest. Disable only after fresh position is within 0.5° and measured speed is at most 0.05 rad/s for one second. If hardware or the return path prevents completion, retain powered hold and report the specific blocker; do not claim that a blind all-checks bypass can guarantee a physical return.
- **Operator-authorized recovery:** when the arm is at or near its supported rest and stationary, the service may be restarted **without external arm support or another confirmation request**. Establish the current pose from fresh SDK feedback, not a saved UI display. Prefer the normal verified-rest shutdown when it works. This exception permits recovery from a stuck service near rest; it does not establish that a distant calibration or working pose is safe to depower.
- **Emergency recovery only:** if a stuck old service blocks its own safe return to rest, an explicitly authorized **powered SDK recovery** is available in `scripts/powered_rest_recovery.py`. This is not a routine movement route. Preflight the entire slow path with the installed model, joint bounds and collision checks; verify native modes/watchdogs and fresh healthy feedback. Stop all old writer threads before the recovery writer starts, keep holding torque throughout transit, never resume a paused owner with stale targets, and retain fixed powered hold on failure. Kill the old owner and disable motors only after fresh rest and quiet feedback are sustained for one second. Evidence: `evidence/auto-calibration-live/powered-recovery-result.json`. If tracking lag reaches the 0.04 rad guard, settle in powered hold and explicitly replan more slowly; do not widen the guard.
- Use ONLY the SDK for robot communication; the service owns it during normal operation. Do not send raw CAN commands. The SDK handles CAN bus communication and threading, including during emergency recovery.
- Robot: **i2rt YAM Standard / YAM v1**, six arm joints, motor IDs **1–6 only**. The linear 4310 gripper and motor 7 have been removed and replaced by the receiver and camera mount. Use SDK `NO_GRIPPER` / `no_gripper`; never address or calibrate motor 7. White arm identified by the user.
- **J6 physical minimum and maximum must be calibrated for the installed receiver/camera assembly.** The current pose identified by the user is the **maximum**. Capture its SDK encoder position; a saved UI display is not fresh feedback. **Disable motor power before asking the user to position J6 at its minimum**, then read that endpoint through an SDK-supported method that does not enable the motor. Keep power disabled throughout manual endpoint positioning. Do not assume the stock J6 range, zero the encoder or write motor Flash to establish these limits; use the measured endpoints consistently in the model, planner and controller, with clearance from the physical stops.
- Installed J6 calibration: physical minimum **−308.83°**, maximum **−118.58°**; operating range **−307.83° to −119.58°** with 1° clearance. Source: `assets/camera_mount/joint-calibration.json` and `evidence/j6-limits/`. Preserve the fixed SDK encoder branch with zero motor offsets; do not apply startup ±360° normalization. Read-only SDK startup feedback must be inside the operating range before enabling. Captured rest must also be inside this range and must never expand J6 bounds.
- Adapter: **CANable Pro**, SocketCAN **can0**, gs_usb, **1,000,000 bit/s**.
- Power supply reported by the user: **24 V / 15 A**, no current-limit indicator or display.
- Make sure the robot never enters into a fault state where the motors fault. Old messages should trigger a warning at best.
- Use temperature readings to bring the robot to a cooldown mode if it reaches 60C. This is the rest position.
- The robot has been modified from the stock i2rt robot. The gripper motor is replaced by a receiver and a camera mount. You can get this from the 'autoweld-old' github worktree. Don't use anything else from there.


## Robot Main UI
- The robot should be able to get to different TCP poses by clicking on viser.
- There should be a motor enable, disable at rest button
- Mode selection should be exposed.
- Joint states, angles, errors and temps should be displayed nicely.
- Should be able to save pose and get to saved poses by clicking on them.
- No SDK / Robot control code should be in the UI. Communicate via JSON to the main process.

## Camera Calibration Classes.
- This is to be done in ~/Code/Weld-seam-finder
- We are using a Teledyne FLIR camera with the SDK already installed. It has a python interface.
- Make sure camera streams at max fps. auto exposure should be implemented.
- Do live checkerboard detection. Make this as fast as possible. Processing done inside python classes and Corners, origin and XY should be send via json.
- Reuse this code for intrinsics and extrinsics (hand eye) calibration
- The target is a print on paper - so a higher RMS threshold may be needed. Adjust RMS to sane levels till tests pass.

## Camera Calibration UI
This can be rendered as a floating panel inside the viser window.

- Camera should stream only when turned on through this panel.
- Do auto calibration. When this is pressed - use the current pose as baseline and hunt for a full checkerboard view before requiring a good checkerboard. Project the camera optical axis 150 mm to a fixed apex, then construct a cone toward the robot with a **15 degree full opening (7.5 degree half-angle)**. Capture at **25 unique, uniformly spaced points on its edge**, in the baseline camera image plane: radius `150 mm * tan(7.5 deg) = 19.75 mm`. Every camera optical +Z must point at the fixed apex. The baseline is for aiming and return, not an extra capture. Do not add multiple rings, varying depth/radius or extra roll. Render line thickness in screen pixels, never metres.
- Calibration motion uses a bounded envelope around its selected baseline, which may be far from supported rest; retain the normal SDK path, joint limits and collision checks. At 60°C, cancel calibration without replacing the controller's cooldown return-to-rest trajectory. Resume work only after cooldown below 55°C.
- During automatic calibration, hunt for the checkerboard at the baseline and at capture poses if it is only partially visible. Use OpenCV to identify available checkerboard corners, including partial-board corners; do not require full-board detection before computing the correction. Compute the centroid pixel of all identified corners and the image-plane vector from the image center to that centroid: `offset_px = mean(corners_px) - [image_width / 2, image_height / 2]`.
- Map this vector's direction through the camera/TCP frame orientation into a TCP translation toward the checkerboard centroid. Move the TCP in **5 mm increments**, acquire a fresh frame after each settled move, and recompute the centroid and direction until the full checkerboard is in view. Keep processing and hunting logic in the Python backend; exchange detections, offsets and hunt status with the UI via JSON. Reuse this behavior for intrinsics and hand-eye calibration.
- Hunt moves must use the existing SDK motion planner, joint limits and collision checks. Bound the search and support cancellation; if no corners are detected, the centroid direction is undefined, so use a bounded reacquisition search rather than inventing a direction or accepting the view.
- Re-use this same panel and python code to do instrinsics and extrinsics (Hand eye) calibration.
- No openCV or camera specific code should be in the UI. Communicate via JSON to the main process that has a camera singleton.

## Laser Plane Calibration UI
Separate selectable panel. This can be rendered as a floating panel inside the viser window.

- Assume the laser is on and do laser plane calibration using the calibration pose.
- Separate out the code. Weld finder should own actual laser plane calibration. Autoweld service talks to it to do it. Web panel calls autoweld API.
- Output from weld seam finder should be an XYZ pointcloud with noise removed from the calibrated camera origin. Autoweld converts it to the center of J6 using the measured hand-eye and mount transform. This cloud reporting frame does not redefine the nominal `weld_tcp` used for motion or invalidate saved poses. Register the world preview only with matching optical calibration, fresh camera acquisitions and fresh stationary SDK feedback.
- Use smart auto exposure to make the laser line more visible when its not.
- Expose auto-exposure minimum/maximum, manual exposure and camera gain in the laser line finder panel, with actual sensor readback and effective bounds capped by the maximum camera frame rate. Apply them through the existing camera owner. Preview gamma/brightness must brighten display pixels only; never tone-map native calibration, laser detection or XYZ inputs. Keep the 2D preview available when the point-cloud display is off.
- Feel free to move the arm automatically to calibration pose to do this autonomously.
- After each discrete laser calibration move, first verify measured convergence to the requested TCP endpoint (within 0.5 mm / 0.005 rad) and quiet feedback; elapsed trajectory duration alone is insufficient. Then capture one measured Hold through the service and require continuous quiet, pose-stable feedback and a fresh acquisition before measurement. Do not freeze tracking lag, repeatedly retarget Hold during settling or capture, or relax native exposure-bracket motion or plane-fit gates to accept a view.
- A partial fast preview alone must not trigger laser calibration hunting: laser glare can hide preview corners on an otherwise fully visible board. Try native full-resolution laser capture first at a settled capture pose. A native missing-board result may authorize one bounded hunt and retry at that view; glare, bracket movement, stripe or fit-quality failures must remain rejections.
