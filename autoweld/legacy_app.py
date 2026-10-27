"""Historical tracking/teach frontend utilities retained during JSON migration.

This module has no CLI or hardware connection entrypoint. Production uses app.py
and backend.py; legacy control tuning is not available there.
"""

import copy
import json
import logging
import queue
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.utils.viser_control_interface import ViserControlInterface
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from autoweld.cad import load_step
from autoweld.control import ANGLE_TOL, REST_TOL, Controller
from autoweld.diagnostics import MotionRejection, solve_pose
from autoweld.motion import MotionLimits, MotionPlanner
from autoweld.teach import LineRecorder, pose_file
from autoweld.tracking import TrackingMonitor, load_tracks, plan_tracking, tracking_sample
from autoweld.tracking_limits import POSITION_TOLERANCE_M, SELECTED_SAFE_FEED_M_S

CONTROLLER_NAMES = {"Factory PD": "PD", "Tuned PD": "PID", "Position–speed": "POS_VEL",
                    "Mixed: J4/J5 MIT + gravity": "MIXED_J45"}


ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "vendor/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def tool_model():
    """Add a display/IK site only; the robot's real gripper dynamics remain SDK defaults."""
    original = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    tree = ET.parse(original)
    body = tree.find(".//body[@name='gripper']")
    ET.SubElement(body, "site", name="weld_tcp", pos="0 0 -0.15", quat="1 0 0 0", size="0.004")
    path = Path(original).with_name(f"autoweld-{uuid.uuid4().hex}.xml")
    tree.write(path)
    return str(path)


class App(ViserControlInterface):
    def __init__(self, robot, model_path, live, camera, port, runtime, step=None, robot_factory=None,
                 gravity_calibration=False, start_disabled=False):
        super().__init__(robot, model_path, ee_site="weld_tcp", port=port)
        self.camera = camera
        self.runtime = Path(runtime)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.inbox = self.runtime / "requests"
        self.inbox.mkdir(exist_ok=True)
        (self.runtime / "responses").mkdir(exist_ok=True)
        self.ctrl = Controller(robot, self._model.jnt_range[:6], live=live, robot_factory=robot_factory)
        grip_record = self.runtime / "gripper-hold.json"
        if grip_record.exists():
            self.ctrl.gripper_hold_closed = bool(json.loads(grip_record.read_text())["closed"])
        if gravity_calibration:
            self.ctrl.ki.fill(0)
            self.ctrl.friction_enabled = False
            self.ctrl.gravity_model = "sdk"
        if start_disabled:
            self.ctrl.closed = True
            self.ctrl.cached_rest_display = True
            self.ctrl.disabled_q = self.ctrl.rest.copy()
            self.ctrl.mode = "Disabled"
        else:
            self.ctrl.start_hold()
        self.ctrl.motion_planner = MotionPlanner(model_path)
        saved_limits = self.runtime / "motion-limits.json"
        if saved_limits.exists():
            try:
                self.ctrl.motion_planner.limits = MotionLimits(**json.loads(saved_limits.read_text()))
            except (ValueError, TypeError) as exc:
                logging.warning("Invalid saved motion limits; using defaults: %s", exc)
        self.ctrl.motion_collision = lambda q: self._has_self_collision(q, len(q))
        self.exit_after_disable = False
        self.events = queue.Queue()
        self.pending_pose = None
        self.request_status = None
        self.tracks = []
        self.tracking_preview = None
        self.tracking_monitor = None
        self.tracking_result = None
        self.tracking_log = None
        self.line_recorder = LineRecorder()
        self.line_markers = {}
        self.tracking_pose_markers = {}
        self.running = True
        self.message = ("Saved rest display; motors disabled. Enable motors for fresh feedback."
                        if start_disabled else "Holding captured rest. Select a control mode.")
        self.fault = None
        self._mirror_robot()
        self.server = viser.ViserServer(host="127.0.0.1", port=port, label="AutoWeld · Stage 3")
        self.server.gui.configure_theme(dark_mode=True)
        self.server.scene.add_grid("/floor", width=2, height=2, cell_size=0.1, section_size=0.5)
        self.visual = ViserUrdf(self.server, URDF, root_node_name="/robot")
        self.mount = self.server.scene.add_frame("/torch", axes_length=0.025, axes_radius=0.001)
        torch_mesh = trimesh.creation.cylinder(radius=0.009, height=0.15)
        torch_mesh.apply_translation((0, 0, -0.075))
        self.server.scene.add_mesh_trimesh("/torch/placeholder", torch_mesh)
        self.ee_frame = self.server.scene.add_frame("/tcp", axes_length=0.06, axes_radius=0.002)
        self.target_handle = self.server.scene.add_transform_controls("/target", scale=0.12, visible=False)
        self.cad_frame = self.server.scene.add_transform_controls(
            "/cad",
            scale=0.15,
            position=(0.35, 0, 0.05),
            visible=False,
        )
        self._build_gui(live)
        self.sync_targets()
        self.reference_image = camera.rgb() if camera else None
        if camera:
            camera.save(self.runtime / "rest.jpg")
            with self.server.gui.add_folder("USB camera"):
                self.server.gui.add_image(self.reference_image, label="Captured rest")
                self.camera_image = self.server.gui.add_image(self.reference_image, label="Live camera")
        atomic_json(
            self.runtime / "rest.json",
            {
                "arm_rad": self.ctrl.rest[:6].tolist(),
                "gripper_normalized": float(self.ctrl.rest[6]),
                "gripper_limits_motor_rad": np.asarray(robot.get_robot_info()["gripper_limits"]).tolist(),
                "sdk_revision": (ROOT / "SDK_REVISION").read_text().strip(),
                "live": live,
            },
        )
        if step:
            self.import_step(step)

        @self.server.on_client_connect
        def on_connect(client):
            client.camera.position = (0.9, 0.9, 0.7)
            client.camera.look_at = (0, 0, 0.25)

        self.target_handle.on_update(self._target_changed)

    async def _target_changed(self, event):
        # Viser emits start/update/end. Only release is an operator waypoint.
        # Async with no await preserves gesture order on Viser's event loop.
        if event.phase == "end" and self.ctrl.mode == "IK" and not self.ctrl.closed:
            self.events.put({"action": "pose", "position": list(event.target.position),
                             "wxyz": list(event.target.wxyz), "_source": "gizmo release"})

    def handle_request(self, req):
        action = req["action"]
        started = time.monotonic()
        if action in ("mode", "rest", "disable", "shutdown", "enable", "gripper_close", "acknowledge_fault", "joint", "track_approach", "track_run"):
            if self.pending_pose is not None and hasattr(self, "pose_status_md"):
                self.pose_status_md.content = f"**Cancelled** · Pending TCP target cleared by {action}"
            self.pending_pose = None
        try:
            if (action == "pose" and "_id" not in req
                    and not self.ctrl.closed and not self.ctrl.disable_pending and not self.ctrl.motion_fault()
                    and (self.ctrl.motion_plan is not None or getattr(self, 'tracking_monitor', None)
                         or not self.ctrl.ready_for_waypoint())):
                self.pending_pose = (req.copy(), time.monotonic())
                result = {"ok": True, "status": "queued", "message": "Latest TCP target queued until arm stops"}
                self.message = result["message"]
            else:
                if action == "pose":
                    self.pending_pose = None
                if action == "pose" and hasattr(self, "pose_status_md"):
                    self.pose_status_md.content = "**Planning TCP waypoint…**"
                self.dispatch(req)
                result = {"ok": True, "status": "accepted"}
        except Exception as exc:
            self.message = str(exc)
            result = {"ok": False, "status": "rejected", "error": str(exc)}
            if isinstance(exc, MotionRejection):
                result["diagnostic"] = exc.diagnostic
            elif action == "pose":
                result["diagnostic"] = {"stage": "request", "code": "request_error",
                                        "exception_type": type(exc).__name__, "message": str(exc)}
            logging.warning("Request rejected: %s", exc)
        self.request_status = {"timestamp": time.time(), "action": action,
                               "processing_s": time.monotonic()-started, **result}
        if action.startswith("track_") and hasattr(self, "tracking_md"):
            self.tracking_md.content = f"**{result['status'].capitalize()}** · {result.get('error', self.message)}"
        if action.startswith("line_") and hasattr(self, "line_message_md"):
            self.line_message_md.content = f"**{result['status'].capitalize()}** · {result.get('error', self.message)}"
        if action == "pose" and hasattr(self, "pose_status_md"):
            detail = result.get("error", result.get("message", "Waypoint accepted"))
            diagnostic = result.get("diagnostic")
            category = f" · {diagnostic['stage'].replace('_', ' ').replace('ik', 'IK')}" if diagnostic else ""
            self.pose_status_md.content = f"**{result['status'].capitalize()}{category}**\n\n{detail}"
        with (self.runtime / "command-events.jsonl").open("a") as log:
            log.write(json.dumps({**self.request_status, "request": req}) + "\n")
        return result

    def process_pending_pose(self):
        if self.pending_pose is None:
            return
        req, queued_at = self.pending_pose
        if self.ctrl.closed or self.ctrl.disable_pending or self.ctrl.motion_fault():
            self.pending_pose = None
            if hasattr(self, "pose_status_md"):
                self.pose_status_md.content = "**Cancelled** · Pending TCP target cleared by disable or fault"
            return
        if time.monotonic()-queued_at > 15:
            self.pending_pose = None
            self.message = "Queued TCP target expired: arm did not stop within 15 seconds; send again"
            self.request_status = {"timestamp": time.time(), "action": "pose", "ok": False,
                                   "status": "expired", "error": self.message}
            if hasattr(self, "pose_status_md"):
                self.pose_status_md.content = f"**Expired** · {self.message}"
            return
        if self.ctrl.motion_plan is None and not getattr(self, 'tracking_monitor', None) and self.ctrl.ready_for_waypoint():
            self.pending_pose = None
            self.handle_request(req)

    def update_lifecycle_buttons(self):
        fault = bool(self.ctrl.motion_fault())
        pending = self.ctrl.disable_pending
        self.enable_button.disabled = not self.ctrl.closed
        self.disable_button.disabled = self.ctrl.closed or pending
        self.grab_button.disabled = (self.ctrl.closed or pending or fault
                                     or self.ctrl.motion_plan is not None or bool(self.tracking_monitor))

    def _build_gui(self, live):
        gui = self.server.gui
        gui.add_markdown(
            "# AutoWeld\nYAM Standard · Stage 3\n\n"
            + ("**LIVE HARDWARE**" if live else "**OFFLINE SIMULATION — ideal joint following**")
            + "\n\nSDK control · MIT supports gravity feedforward; position–speed uses onboard cascade control.\n\n"
            "Torch is a visual placeholder; TCP = mount −Z, 150 mm. CAD placement is manual."
        )
        self.status_md = gui.add_markdown("")
        self.telemetry_md = gui.add_markdown("")
        self.torque_md = gui.add_markdown("")

        def button(label, action, payload=None):
            handle = gui.add_button(label)
            handle.on_click(lambda _: self.events.put({"action": action, **(payload() if payload else {})}))
            return handle

        with gui.add_folder("Robot control"):
            self.enable_button = button("Enable motors", "enable")
            self.enable_button.disabled = not self.ctrl.closed
            self.disable_button = button("Disable motors at rest", "disable")
            self.grab_button = button("Grab something", "gripper_close")
            self.grab_button.disabled = self.ctrl.closed
            self.rest_button = button("Return to captured rest", "rest")
            gui.add_markdown("Grab closes the gripper at the current arm pose. "
                             "Disable returns to rest, then turns off all seven motors, including the gripper.")
            self.mode_select = gui.add_dropdown("Mode", ["Hold", "Gravity compensation", "Joint", "IK"])

            @self.mode_select.on_update
            def mode_changed(event):
                if event.client_id is not None:
                    self.events.put({"action": "mode", "mode": self.mode_select.value})

            self.controller_select = gui.add_dropdown(
                "Controller", list(CONTROLLER_NAMES), initial_value="Tuned PD" if self.ctrl.control_law == "PID" else "Factory PD"
            )
            gui.add_markdown("Position–speed or mixed mode: disable at rest, select, then enable. "
                "Onboard cascade control; startup joint speed ceiling 0.15 rad/s. "
                "Mixed mode uses MIT PD with gravity on J4/J5; J1–J3/J6 keep position–speed. "
                "Friction remains off.")
            self.pos_speed_input = gui.add_number("Position-speed joint ceiling (rad/s)", .15,
                                                  min=.01, max=1., step=.05)
            self.pos_speed_button = button("Apply position-speed ceiling", "position_speed_limit",
                lambda: {"limit_rad_s": self.pos_speed_input.value})

            @self.controller_select.on_update
            def controller_changed(event):
                if event.client_id is not None:
                    self.events.put({"action": "controller", "controller": CONTROLLER_NAMES[self.controller_select.value]})

            self.quantization_input = gui.add_checkbox("Smooth MIT commands", initial_value=self.ctrl.command_quantization)
            @self.quantization_input.on_update
            def quantization_changed(event):
                if event.client_id is not None:
                    self.events.put({"action": "command_quantization", "enabled": self.quantization_input.value})

            with gui.add_folder("Native PD gains"):
                gui.add_markdown(
                    "P and D run in the motor firmware. Host integral is disabled (I = 0).\n\n"
                    "P/D changes: at most 20% per application.\n\n"
                    "Native D range: (0, 5]. Applies to MIT only. Oscillation detection uses a full 1 s sustained-motion window."
                )
                self.kp_inputs = [
                    gui.add_number(f"Kp J{i + 1}", float(k), min=0.001, step=0.1)
                    for i, k in enumerate(self.ctrl.kp)
                ]
                self.kd_inputs = [
                    gui.add_number(f"Kd J{i + 1}", float(k), min=0.001, max=5.0, step=0.01)
                    for i, k in enumerate(self.ctrl.kd)
                ]
                self.ki_inputs = []
                self.ki_button = button(
                    "Apply native PD gains",
                    "controller",
                    lambda: {
                        "controller": CONTROLLER_NAMES[self.controller_select.value],
                        "ki": [0.0] * 6,
                        "kp": [h.value for h in self.kp_inputs],
                        "kd": [h.value for h in self.kd_inputs],
                    },
                )
                gui.add_markdown(
                    "Tuned PD uses the selected motor gains with I = 0. Mixed mode applies these gains to J4/J5 only. Factory PD is unchanged. "
                    "The host supplies gravity compensation and planned position/velocity references. "
                    "Offline motion does not simulate motor dynamics."
                )
            with gui.add_folder("Gravity feedforward"):
                self.gravity_model_select = gui.add_dropdown("Gravity model", ["sdk", "linear"], initial_value=self.ctrl.gravity_model)
                self.gravity_inputs = [gui.add_number(f"Gravity scale J{i + 1}", float(value), min=0.001, step=0.05)
                                       for i, value in enumerate(self.ctrl.gravity_factors)]
                self.gravity_button = button("Apply gravity feedforward", "gravity_feedforward",
                    lambda: {"factors": [h.value for h in self.gravity_inputs], "model": self.gravity_model_select.value})
                gui.add_markdown("In MIT mode, downward gravity uses the SDK link masses and lever arms. Linear mode approximates its coupled angle dependence around captured rest; beyond ±0.15 rad it uses the exact SDK model. Position–speed has no torque feedforward.")
            with gui.add_folder("Friction compensation"):
                self.friction_enabled_input = gui.add_checkbox("Enable friction compensation", self.ctrl.friction_enabled)
                self.friction_width_inputs = [gui.add_number(f"Friction transition J{i+1} (rad/s)", float(value), min=0.0, step=.01)
                                              for i, value in enumerate(self.ctrl.friction_transition)]
                self.friction_coeff_inputs = [gui.add_number(f"Friction J{i+1} (Nm)", float(value), min=0.0, step=.01)
                                              for i, value in enumerate(self.ctrl.friction_coefficients)]
                self.friction_button = button("Apply friction compensation", "friction_feedforward", lambda: {
                    "enabled": self.friction_enabled_input.value,
                    "transition_rad_s": [h.value for h in self.friction_width_inputs],
                    "coefficients": [h.value for h in self.friction_coeff_inputs]})
                gui.add_markdown("Friction compensation is off by default at the user's request. When enabled, zero transition uses SDK sign compensation; a positive transition smooths it near zero velocity.")
            self.ack_fault_button = button("Acknowledge oscillation fault", "acknowledge_fault")
            self.ack_fault_button.disabled = True
        self.hold_button = button("Stop and hold current pose", "mode", lambda: {"mode": "Hold"})
        with gui.add_folder("Record a line · A → B"):
            gui.add_markdown("In **Gravity compensation** mode, guide the tool to each endpoint and record it. "
                             "Each point stores TCP position and orientation in robot-base coordinates: mount −Z, 150 mm.")
            self.line_name = gui.add_text("Line file name", initial_value="my-line.step")
            self.line_a_button = button("Record point A", "line_record", lambda: {"point": "A"})
            self.line_b_button = button("Record point B", "line_record", lambda: {"point": "B"})
            self.line_points_md = gui.add_markdown("")
            self.line_save_button = button("Save path + poses and load", "line_save", lambda: {"name": self.line_name.value})
            self.line_clear_button = button("Clear recorded points", "line_clear")
            self.line_message_md = gui.add_markdown("Record A and B, then save the STEP line and companion pose file. "
                f"Files are saved in `{self.runtime / 'trajectories'}`. "
                "Gravity mode requires a Factory PD or Tuned PD controller.")
        self.update_line_recorder_gui()
        with gui.add_folder("Waypoint motion · Stage 2"):
            labels = ["TCP speed (m/s)", "TCP acceleration (m/s²)", "TCP jerk (m/s³)",
                      "TCP angular speed (rad/s)", "TCP angular acceleration (rad/s²)", "TCP angular jerk (rad/s³)",
                      "Joint speed (rad/s)", "Joint acceleration (rad/s²)", "Joint jerk (rad/s³)"]
            self.motion_inputs = {key: gui.add_number(label, value, min=.0001, step=.01)
                                  for (key, value), label in zip(asdict(self.ctrl.motion_planner.limits).items(), labels)}
            self.motion_button = button("Apply motion limits", "motion_limits",
                                        lambda: {"limits": {k: h.value for k, h in self.motion_inputs.items()}})
            gui.add_markdown("Release the TCP gizmo to send a waypoint. If a move is finishing, only your latest released target is queued. Hold/Return/Disable cancel it.")
            self.motion_md = gui.add_markdown("Every waypoint starts and ends with zero planned speed, acceleration and jerk. TCP poses follow a straight position path with smooth orientation. Limits apply to the planned reference; actual tracking must be measured.")

        with gui.add_folder("Joint target · radians"):
            self.joint_inputs = [
                gui.add_number(f"Joint {i + 1} (rad)", float(self.ctrl.rest[i]), step=0.001) for i in range(6)
            ]
            self.joint_button = button(
                "Send joint waypoint", "joint", lambda: {"q": [s.value for s in self.joint_inputs]}
            )
        with gui.add_folder("TCP target · robot base frame"):
            self.xyz = gui.add_vector3("Position (m)", (0.0, 0.0, 0.0), step=0.001)
            self.rpy = gui.add_vector3("Roll / pitch / yaw (rad)", (0.0, 0.0, 0.0), step=0.01)
            self.pose_button = button(
                "Send TCP pose",
                "pose",
                lambda: {"position": list(self.xyz.value), "rpy": list(self.rpy.value)},
            )
            button("Snap target to measured TCP", "sync")
            self.pose_status_md = gui.add_markdown("Drag and release the gizmo, or send a numeric TCP pose.")
        with gui.add_folder("Path tracking · Stage 3"):
            self.track_file = gui.add_text("Tracking STEP path", initial_value="assets/stage3/lines.step")
            button("Load tracking edges", "track_load", lambda: {"path": self.track_file.value})
            self.track_select = gui.add_dropdown("Edge", options=["Load a STEP first"])
            self.track_feed = gui.add_number("Feed (mm/s)", SELECTED_SAFE_FEED_M_S['line']*1000, min=.1, max=100., step=.5)
            gui.add_markdown("**Selected safe test speeds:** lines **20 mm/s**; circles/arcs **10 mm/s**. "
                             "Operator-selected from the recorded trials; each new path still requires preview and tracking checks.")
            button("Use selected safe speed", "track_safe_feed", lambda: {"edge": self.track_select.value})
            self.track_orientation = gui.add_dropdown("Tool orientation", options=["Unconstrained", "Perpendicular", "Center facing"])
            self.track_ramp = gui.add_dropdown("Acceleration placement", options=["Outside edge", "Inside edge"])
            def payload():
                return {"edge": self.track_select.value, "feed_m_s": self.track_feed.value*.001,
                        "orientation": self.track_orientation.value, "ramp_placement": self.track_ramp.value}
            self.track_preview_button = button("Preview tracking path", "track_preview", payload)
            self.track_approach_button = button("Move to path start", "track_approach", payload)
            self.track_run_button = button("Run path tracking", "track_run", payload)
            self.tracking_md = gui.add_markdown("Load STEP edges, preview, approach, then run. Hold interrupts tracking. "
                "Edge deviation over 10 mm or constrained orientation error over 5° is flagged without stopping; SDK fault checks remain active. "
                "Paths use robot-base coordinates in STEP (mm), independently of workpiece placement. "
                "Outside edge puts ramps beyond its endpoints. Inside edge keeps ramps on the edge "
                "and scores feed only during the constant-speed section. "
                "Unconstrained follows position with minimal wrist motion and does not score orientation. Tracking is measured from encoder FK; absolute tip calibration remains unverified.")
        with gui.add_folder("STEP workpiece"):
            self.cad_path = gui.add_text("Local STEP path", initial_value="")
            button("Load STEP path", "step", lambda: {"path": self.cad_path.value})
            upload = gui.add_upload_button("Upload STEP", mime_type=".step,.stp")

            @upload.on_upload
            def uploaded(_):
                if upload.value is None:
                    return
                name = Path(upload.value.name).name
                if Path(name).suffix.lower() not in (".step", ".stp"):
                    self.events.put({"action": "error", "message": "Select a .step or .stp file"})
                    return
                path = self.runtime / f"upload-{uuid.uuid4().hex}-{name}"
                path.write_bytes(upload.value.content)
                self.events.put({"action": "step", "path": str(path)})

            self.cad_xyz = gui.add_vector3("CAD position (m)", (0.35, 0.0, 0.05), step=0.001)
            self.cad_rpy = gui.add_vector3("CAD rotation (rad)", (0.0, 0.0, 0.0), step=0.01)
            button(
                "Place workpiece",
                "place_cad",
                lambda: {"position": list(self.cad_xyz.value), "rpy": list(self.cad_rpy.value)},
            )
            self.cad_md = gui.add_markdown("No workpiece loaded. STEP units are converted to metres.")

    def sync_targets(self):
        self._mirror_robot()
        T = self._ee_pose_4x4()
        self.target_handle.position = T[:3, 3]
        self.target_handle.wxyz = self._mat3_to_wxyz(T[:3, :3])
        self.xyz.value = tuple(T[:3, 3])
        self.rpy.value = tuple(Rotation.from_matrix(T[:3, :3]).as_euler("xyz"))
        for i, handle in enumerate(self.joint_inputs):
            handle.value = float(self.ctrl.positions()[i])

    def import_step(self, path):
        mesh = load_step(path)
        self.server.scene.add_mesh_simple(
            "/cad/workpiece", mesh.vertices, mesh.faces, color=(100, 170, 220), side="double"
        )
        self.cad_frame.visible = True
        self.cad_md.content = f"**{Path(path).name}** · {len(mesh.faces):,} triangles\n\nBounds (m): {mesh.extents.round(4).tolist()}"

    def update_line_recorder_gui(self):
        idle = not (self.ctrl.motion_plan is not None or self.ctrl.disable_pending or self.tracking_monitor)
        can_record = (idle and not self.ctrl.closed and not self.ctrl.motion_fault()
                      and self.ctrl.mode == "Gravity compensation")
        self.line_a_button.disabled = self.line_b_button.disabled = not can_record
        length = self.line_recorder.length
        self.line_save_button.disabled = not idle or length is None or length <= 1e-6
        self.line_clear_button.disabled = not self.line_recorder.points
        rows = []
        for label in ("A", "B"):
            point = self.line_recorder.points.get(label)
            coords = "Not recorded" if point is None else ", ".join(f"{x*1000:.3f}" for x in point["position_m"])
            rows.append(f"**{label}** · {coords}" + (" mm" if point else ""))
            if point:
                quat = point["quaternion_wxyz"]
                rows[-1] += " · orientation (w, x, y, z): " + ", ".join(f"{x:.5f}" for x in quat)
        rows.append("Record buttons are available in Gravity compensation mode." if not can_record
                    else "Guide the tool to an endpoint, then record its current position and orientation.")
        if length is not None:
            rows.insert(2, f"**A → B length:** {length*1000:.3f} mm")
        self.line_points_md.content = "\n\n".join(rows)

    def draw_recorded_points(self, points, root, markers, connect=True):
        for marker in reversed(list(markers.values())):
            marker.remove()
        markers.clear()
        for label, point in points.items():
            color = (70, 220, 130) if label == "A" else (255, 170, 60)
            name = f"{root}/{label}"
            markers[label] = self.server.scene.add_frame(name, axes_length=.025, axes_radius=.001,
                position=point["position_m"], wxyz=point["quaternion_wxyz"])
            markers[label+"_dot"] = self.server.scene.add_icosphere(name+"/point", radius=.004, color=color)
            markers[label+"_label"] = self.server.scene.add_label(name+"/label", label)
        if connect and len(points) == 2:
            positions = np.array([points[k]["position_m"] for k in ("A", "B")])
            markers["line"] = self.server.scene.add_line_segments(
                root+"/line", points=positions[None], colors=(255, 210, 80), thickness=3)

    def line_request(self, req):
        action = req["action"]
        if action == "line_clear":
            self.line_recorder = LineRecorder()
            self.draw_recorded_points({}, "/recorded-line", self.line_markers)
            self.message = "Recorded points cleared"
        elif action == "line_record":
            self.ctrl.health()
            if (self.ctrl.mode != "Gravity compensation" or self.ctrl.motion_plan is not None
                    or self.ctrl.disable_pending or self.tracking_monitor):
                raise RuntimeError("Select Gravity compensation mode before recording a point")
            obs = self.ctrl.robot.get_observations()
            q = np.asarray(obs["joint_pos"], dtype=float)[:6].copy()
            now = time.time()
            timestamp = float(obs.get("feedback_timestamp", now if not self.ctrl.live else float("nan")))
            if not np.isfinite(timestamp) or not 0 <= now-timestamp <= .1:
                raise RuntimeError("Fresh SDK feedback is required to record a point; try again")
            pose = self.ctrl.motion_planner.fk(q).copy()
            label = req["point"]
            self.line_recorder.record(label, pose, q, now, timestamp)
            self.draw_recorded_points(self.line_recorder.points, "/recorded-line", self.line_markers)
            self.message = f"Recorded point {label}: measured TCP position and orientation"
        elif action == "line_save":
            if self.ctrl.motion_plan is not None or self.ctrl.disable_pending or self.tracking_monitor:
                raise RuntimeError("Wait for the current motion to finish before saving a line")
            path, curves = self.line_recorder.save(self.runtime / "trajectories", req["name"])
            self.load_tracking_curves(curves, path)
            self.message = (f"Saved and loaded {path.name} + {pose_file(path).name} · "
                            f"A → B, {self.line_recorder.length*1000:.3f} mm, both endpoint poses")
        else:
            raise ValueError(f"Unknown line action: {action}")
        self.update_line_recorder_gui()

    def dispatch(self, req):
        action = req["action"]
        # File requests get an ID from our inbox reader. Browser events do not.
        # The automatic-test scope must never become a global UI/SDK restriction.
        automatic = "_id" in req
        if action == "error":
            raise ValueError(req["message"])
        if action.startswith("line_"):
            self.line_request(req)
            return
        if action.startswith("track_"):
            self.tracking_request(req, automatic)
            return
        if action == "mode":
            if req["mode"] == "Gravity compensation":
                self.ctrl.gravity()
            elif req["mode"] in ("Hold", "Joint", "IK"):
                self.ctrl.hold(automatic=automatic)
                self.ctrl.mode = req["mode"]
                self.sync_targets()
            else:
                raise ValueError("Unknown mode")
        elif action == "controller":
            if np.any(np.asarray(req.get("ki", np.zeros(6)), dtype=float) != 0):
                raise ValueError("Host integral is disabled; use native P/D gains with I = 0")
            self.ctrl.select_controller(
                CONTROLLER_NAMES.get(req["controller"], req["controller"]), np.zeros(6), req.get("deadband_deg"), req.get("kp"), req.get("kd")
            )
            for handle, gain in zip(self.kp_inputs, self.ctrl.kp):
                handle.value = float(gain)
            for handle, gain in zip(self.kd_inputs, self.ctrl.kd):
                handle.value = float(gain)
            for handle, gain in zip(self.ki_inputs, self.ctrl.ki):
                handle.value = float(gain)
        elif action == "command_quantization":
            if self.tracking_monitor:
                raise RuntimeError("Wait for tracking to finish before changing command encoding")
            self.ctrl.configure_command_quantization(req["enabled"])
            self.quantization_input.value = self.ctrl.command_quantization
        elif action == "gravity_feedforward":
            self.ctrl.configure_gravity(req["factors"], req.get("model", self.ctrl.gravity_model))
            for handle, factor in zip(self.gravity_inputs, self.ctrl.gravity_factors):
                handle.value = float(factor)
            self.gravity_model_select.value = self.ctrl.gravity_model
        elif action == "friction_feedforward":
            self.ctrl.configure_friction(req.get("enabled", True), req.get("transition_rad_s"), req.get("coefficients"))
            self.friction_enabled_input.value = self.ctrl.friction_enabled
            for handles, values in [(self.friction_width_inputs, self.ctrl.friction_transition),
                                    (self.friction_coeff_inputs, self.ctrl.friction_coefficients)]:
                for handle, value in zip(handles, values):
                    handle.value = float(value)
        elif action == "acknowledge_fault":
            self.ctrl.acknowledge_fault()
            for handles, gains in [(self.kp_inputs, self.ctrl.kp), (self.ki_inputs, self.ctrl.ki), (self.kd_inputs, self.ctrl.kd)]:
                for handle, gain in zip(handles, gains):
                    handle.value = float(gain)
            self.sync_targets()
        elif action == "motion_limits":
            if self.ctrl.motion_plan is not None or self.ctrl.disable_pending:
                raise RuntimeError("Wait for the current move before changing motion limits")
            limits = MotionLimits(**req["limits"])
            self.ctrl.motion_planner.limits = limits
            atomic_json(self.runtime / "motion-limits.json", asdict(limits))
            for key, value in asdict(limits).items():
                self.motion_inputs[key].value = value
        elif action == "joint":
            arm = np.asarray(req["q"], dtype=float)
            if arm.shape != (6,):
                raise ValueError("Expected six arm angles in radians")
            self.ctrl.command(np.r_[arm, self.ctrl.gripper_target], "Joint", automatic=automatic)
        elif action == "gripper_close":
            self.ctrl.close_gripper()
            self.tracking_preview = None
            atomic_json(self.runtime / "gripper-hold.json", {"closed": True})
        elif action == "position_speed_limit":
            self.ctrl.configure_position_speed_limit(float(req["limit_rad_s"]))
            self.pos_speed_input.value = self.ctrl.position_speed_limit
            self.tracking_preview = None
        elif action == "pose":
            T = np.eye(4)
            T[:3, 3] = req["position"]
            if "rpy" in req:
                T[:3, :3] = Rotation.from_euler("xyz", req["rpy"]).as_matrix()
            else:
                w, x, y, z = req["wxyz"]
                T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            if not np.isfinite(T).all():
                raise ValueError("TCP pose must be finite")
            self._mirror_robot()
            q = solve_pose(self._kin, T, "weld_tcp", self._data.qpos, stage="endpoint_ik")
            cmd = np.r_[q[:6], self.ctrl.gripper_target]
            # Same collision check as the upstream Viser example (robot only, not CAD).
            if self._has_self_collision(cmd, len(cmd)):
                raise MotionRejection("endpoint_validation", "self_collision",
                                      "Endpoint rejected: SDK robot self-collision check failed")
            self.ctrl.command(cmd, "IK", automatic=automatic, cartesian=T)
            self.target_handle.position = T[:3, 3]
            self.target_handle.wxyz = self._mat3_to_wxyz(T[:3, :3])
        elif action == "rest":
            self.ctrl.return_rest(automatic=automatic)
        elif action == "disable":
            self.ctrl.request_disable(automatic=automatic)
        elif action == "end_long_cruise_test":
            self.ctrl.configure_long_cruise_test(False)
        elif action == "enable":
            self._robot = self.ctrl.enable()
            self.sync_targets()
        elif action == "sync":
            self.sync_targets()
        elif action == "step":
            self.import_step(req["path"])
        elif action == "place_cad":
            self.cad_frame.position = req["position"]
            xyzw = Rotation.from_euler("xyz", req["rpy"]).as_quat()
            self.cad_frame.wxyz = np.r_[xyzw[3], xyzw[:3]]
        elif action == "snapshot":
            if not self.camera:
                raise RuntimeError("No camera in offline mode")
            self.camera.save(self.runtime / "camera.jpg")
        elif action == "shutdown":
            self.ctrl.request_disable(automatic=automatic)
            self.exit_after_disable = True
            if self.ctrl.closed:
                self.running = False
        else:
            raise ValueError(f"Unknown action: {action}")
        self.target_handle.visible = self.ctrl.mode == "IK" and not self.ctrl.closed
        self.message = f"{action}: accepted"
        self.fault = None

    def load_tracking_curves(self, curves, path):
        recorded = curves[0].recorded_points if len(curves) == 1 else None
        options = ["Unconstrained", "Perpendicular", "Center facing"]
        self.track_orientation.options = options + (["Recorded poses"] if recorded else [])
        if recorded:
            self.line_recorder = LineRecorder()
            self.line_recorder.points = copy.deepcopy(recorded)
            self.line_recorder.saved_path = str(Path(path).resolve())
            self.line_name.value = Path(path).name
            self.draw_recorded_points({}, "/recorded-line", self.line_markers)
            self.track_orientation.value = "Recorded poses"
            self.track_ramp.value = "Inside edge"
        elif self.track_orientation.value == "Recorded poses":
            self.track_orientation.value = "Unconstrained"
        self.draw_recorded_points(recorded or {}, "/tracking/poses", self.tracking_pose_markers, connect=False)
        self.update_line_recorder_gui()
        self.tracks = curves
        self.tracking_preview = None
        self.tracking_result = None
        self.track_file.value = str(path)
        self.track_select.options = [f'{i+1}: {c.kind} · {c.length*1000:.2f} mm' for i,c in enumerate(curves)]
        self.track_select.value = self.track_select.options[0]
        self.track_feed.value = SELECTED_SAFE_FEED_M_S[curves[0].kind]*1000
        points = []
        for curve in curves:
            p = np.array([curve.point(s) for s in np.linspace(0, 1, 101)])
            points.extend(np.stack((p[:-1], p[1:]), axis=1))
        self.server.scene.add_line_segments('/tracking/edges', points=np.array(points),
                                             colors=(60,200,255), thickness=3)
        self.message = f'Loaded {len(curves)} tracking edges from {Path(path).name}'
        if recorded:
            self.message += ' with saved A/B poses; tool orientation interpolates from A to B'
        self.tracking_md.content = self.message + '. Preview, move to path start, then run.'

    def tracking_request(self, req, automatic):
        if self.ctrl.motion_plan is not None or self.ctrl.disable_pending or self.tracking_monitor:
            raise RuntimeError('Wait for the current motion to finish before editing a tracking path')
        action = req['action']
        if action == 'track_load':
            curves = load_tracks(req['path'])
            self.load_tracking_curves(curves, req['path'])
            return
        if not self.tracks:
            raise ValueError('Load a tracking STEP first')
        selected = req.get('edge', self.track_select.value)
        index = selected if isinstance(selected, int) else int(selected.split(':')[0])-1
        if not 0 <= index < len(self.tracks):
            raise ValueError('Invalid edge index')
        if action == 'track_safe_feed':
            self.track_feed.value = SELECTED_SAFE_FEED_M_S[self.tracks[index].kind]*1000
            self.tracking_preview = None
            self.message = f'Selected safe test feed: {self.track_feed.value:g} mm/s; preview this path before running'
            return
        feed = float(req.get('feed_m_s', self.track_feed.value*.001))
        orientation = req.get('orientation', self.track_orientation.value)
        ramp_placement = req.get('ramp_placement', getattr(getattr(self, 'track_ramp', None), 'value', 'Outside edge'))
        signature = (index, feed, orientation, ramp_placement,
                     self.ctrl.position_speed_limit, asdict(self.ctrl.motion_planner.limits))
        if action in ('track_preview', 'track_approach'):
            bounds = self.ctrl.limits.copy()
            bounds[:,0] = np.minimum(bounds[:,0], self.ctrl.rest[:6])
            bounds[:,1] = np.maximum(bounds[:,1], self.ctrl.rest[:6])
            seed = self.ctrl.positions()
            seed[6] = self.ctrl.gripper_target
            plan = plan_tracking(self.ctrl.planner_for_controller(), seed, self.tracks[index],
                                 feed, orientation, bounds=bounds, collision=self.ctrl.motion_collision,
                                 ramp_placement=ramp_placement)
            # Include lead-in/out and spline interiors in automatic-test checks.
            if automatic and self.ctrl.live:
                for s in np.linspace(0,1,1001):
                    q = plan.motion.path.spline(s)[:6]
                    for reference in (self.ctrl.positions()[:6], self.ctrl.rest[:6], self.ctrl.target[:6]):
                        if np.max(abs(q-reference)) >= self.ctrl.test_excursion:
                            raise ValueError(f'Automatic tracking path including lead-in/out exceeds {self.ctrl.test_excursion:g} rad')
            self.tracking_preview = (signature, plan)
            p = np.array([plan.expected(t)[:3,3] for t in np.linspace(0,plan.motion.path.duration,301)])
            self.server.scene.add_line_segments('/tracking/preview', points=np.stack((p[:-1],p[1:]),axis=1),
                                                 colors=(255,170,40), thickness=4)
            self.message = (f'Path ready: {plan.curve.length*1000:.2f} mm at {feed*1000:g} mm/s; '
                            f'{plan.motion.path.duration:.2f} s including {plan.lead*1000:.2f} mm lead-in/out each')
            if action == 'track_approach':
                self.ctrl.command(plan.motion.path.positions[0], 'Joint', automatic=automatic)
                self.message += '. Approaching start; wait for convergence, then Run.'
        elif action == 'track_run':
            if self.tracking_preview is None or self.tracking_preview[0] != signature:
                raise ValueError('Preview or approach this edge with the current feed, orientation and motion limits first')
            plan = self.tracking_preview[1]
            self.ctrl.command_tracking(plan, automatic=automatic)
            self.tracking_monitor = TrackingMonitor(plan)
            self.tracking_result = {'running':True, 'plan':plan.motion.report}
            self.tracking_log = self.runtime / f'tracking-{time.time_ns()}.jsonl'
            self.message = 'Tracking started; measured tip error and feed are being recorded'
        else:
            raise ValueError('Unknown tracking action')

    def update_tracking(self):
        monitor = self.tracking_monitor
        if monitor is None:
            return
        plan = monitor.plan
        elapsed = time.monotonic()-self.ctrl.motion_started
        interrupted = self.ctrl.mode != 'Path tracking' or bool(self.ctrl.motion_fault()) or self.ctrl.closed
        reason = 'Interrupted by operator, motor disable or SDK fault' if interrupted else None
        if not interrupted:
            if self.ctrl.live:
                obs = self.ctrl.robot.get_observations()
                q, velocity = np.asarray(obs['joint_pos']), np.asarray(obs['joint_vel'])
            else:
                q, velocity, _, _ = plan.motion.path.sample(elapsed)
            now = time.time()
            timestamps = np.asarray(obs["joint_timestamps"]) if self.ctrl.live else np.full(6, now)
            # Score the reference at the feedback time, not at a later UI poll.
            sample_elapsed = max(0., elapsed-(now-float(np.min(timestamps))))
            elapsed = sample_elapsed
            row = tracking_sample(plan, self.ctrl.motion_planner, q, velocity, elapsed)
            monitor.add_velocity_diagnostics(row, self.ctrl.motion_planner, q, timestamps, now)
            row["sdk_timing"] = self.ctrl.robot.get_feedback_diagnostics() if self.ctrl.live else None
            monitor.rows.append(row)
            with self.tracking_log.open('a') as log:
                log.write(json.dumps(row)+'\n')
            # User-selected acquisition policy: geometric deviation is recorded
            # through completion. SDK faults and lifecycle/monitor failures still Hold.
            self.tracking_result = {'running':bool(reason is None and elapsed<plan.motion.path.duration+1.),
                                    'sample':row, **monitor.summary()}
            deviation_label = 'DEVIATED — recording; motion continues' if self.tracking_result['deviated'] else 'Tracking'
            self.tracking_md.content = (f"**{deviation_label}** · {np.clip(row['edge_fraction'],0,1):.0%}\n\n"
                f"Edge distance {row['cross_track_mm']:.3f} mm ({POSITION_TOLERANCE_M*1000:g} mm record-only threshold); "
                f"timed error {row['tracking_error_mm']:.3f} mm (recorded only); "
                f"orientation {row['orientation_error_deg']:.2f}°; "
                f"raw speed {row['speed_m_s']*1000:.2f} / {plan.feed*1000:.2f} mm/s; "
                + (f"position estimate {row['position_velocity']['along_speed_m_s']*1000:.2f} / "
                   f"{row['position_velocity']['reference_speed_m_s']*1000:.2f} mm/s along path "
                   f"at t={row['position_velocity']['elapsed_s']:.3f}s "
                   f"(120 ms window; quantization-only bound ±{row['position_velocity']['tcp_quantization_bound_m_s']*1000:.2f} mm/s)\n\n"
                   if row.get('position_velocity') else "position estimate unavailable / warming up\n\n") +
                ("Velocity sample stale, duplicated or skewed; not valid for qualification. "
                 if not row['velocity_sample_valid'] else "") +
                f"{'Final hold verification' if elapsed > plan.motion.path.duration else 'On requested edge' if row['on_edge'] else 'Lead-in/out ramp'} · encoder-based measurement")
        if reason or elapsed >= plan.motion.path.duration+1.:
            result = monitor.summary(completed=reason is None, reason=reason)
            result.update({'running':False, 'trace':str(self.tracking_log), 'plan':plan.motion.report})
            if not self.ctrl.live:
                result['measurement'] = 'Ideal simulation; no measured robot dynamics or external tip metrology'
            self.tracking_result = result
            atomic_json(self.tracking_log.with_suffix('.json'), result)
            speed = result['max_speed_error_percent']
            orientation = result['max_orientation_error_deg']
            speed_label = f'{speed:.1f}%' if speed is not None else 'edge not reached'
            orientation_label = f'{orientation:.2f}°' if orientation is not None else 'edge not reached'
            qualification = 'DEVIATED' if result['deviated'] else 'PASS (encoder model)' if result['passed'] else 'NOT QUALIFIED'
            self.tracking_md.content = (f"**{qualification}** · "
                f"{reason or 'Path complete'}\n\nMax edge distance {result['max_cross_track_mm'] or 0:.3f} mm; "
                f"max timed error {result['max_tracking_error_mm']:.3f} mm (recorded only); "
                f"max feed deviation {speed_label}; "
                f"max orientation error {orientation_label}. "
                f"Trace: {self.tracking_log.name}")
            self.tracking_monitor = None

    def tick_motion(self):
        """Reporting failures must not block SDK hold or verified-rest disable."""
        try:
            self.update_tracking()
        except Exception as exc:
            self.tracking_monitor = None
            self.tracking_result = {"running": False, "completed": False, "passed": False,
                                    "reason": f"Tracking monitor failed: {exc}"}
            if not self.ctrl.closed and self.ctrl.mode == 'Path tracking':
                self.ctrl.hold()
            raise
        finally:
            self.ctrl.tick()

    def state(self):
        q = self.ctrl.positions()
        error = np.rad2deg(self.ctrl.target[:6] - q[:6])
        T = self._ee_pose_4x4()
        return {
            "live": self.ctrl.live,
            "mode": self.ctrl.mode,
            "automatic_test_excursion_rad": self.ctrl.test_excursion,
            "controller": self.ctrl.control_law,
            "controller_description": ("J4/J5 MIT PD + gravity; J1-J3/J6 onboard position-speed" if self.ctrl.control_law == "MIXED_J45" else
                "Onboard position-speed cascade; no torque feedforward" if self.ctrl.control_law == "POS_VEL"
                else "Tuned native PD, host integral disabled" if self.ctrl.control_law == "PID" else "Factory PD"),
            "position_speed_limit_rad_s": (min(self.ctrl.position_speed_limit, self.ctrl.motion_planner.limits.joint_velocity)
                if self.ctrl.control_law in ("POS_VEL", "MIXED_J45") and self.ctrl.motion_planner else None),
            "arm_control_modes": (["POS_VEL"]*3+["MIT", "MIT", "POS_VEL"] if self.ctrl.control_law == "MIXED_J45"
                                  else ["POS_VEL" if self.ctrl.control_law == "POS_VEL" else "MIT"]*6),
            "motor_mode_survey": getattr(getattr(self.ctrl.robot, "motor_chain", None), "mode_survey", None),
            "motion_active": self.ctrl.motion_plan is not None,
            "pending_tcp_target": getattr(self, "pending_pose", None) is not None,
            "last_request": getattr(self, "request_status", None),
            "tracking": getattr(self, "tracking_result", None),
            "line_recording": self.line_recorder.snapshot() if hasattr(self, "line_recorder") else None,
            "tracking_position_tolerance_mm": POSITION_TOLERANCE_M*1000,
            "tracking_deviation_action": "record_only",
            "selected_safe_feed_mm_s": {kind: feed*1000 for kind, feed in SELECTED_SAFE_FEED_M_S.items()},
            "tracking_position_metric": "edge_distance",
            "motion_limits": asdict(self.ctrl.motion_planner.limits) if self.ctrl.motion_planner else None,
            "motion_plan": self.ctrl.last_motion,
            "feedback_diagnostics": self.ctrl.robot.get_feedback_diagnostics() if hasattr(self.ctrl.robot, "get_feedback_diagnostics") else None,
            "command_quantization_compensation": self.ctrl.command_quantization,
            "motion_reference": self.ctrl.robot.get_motion_reference() if hasattr(self.ctrl.robot, "get_motion_reference") else None,
            "ki": self.ctrl.ki.tolist(),
            "pid_kp": self.ctrl.kp.tolist(),
            "pid_kd": self.ctrl.kd.tolist(),
            "gravity_factors": self.ctrl.gravity_factors.tolist(),
            "gravity_model": self.ctrl.gravity_model,
            "friction_enabled": self.ctrl.friction_enabled,
            "friction_transition_rad_s": self.ctrl.friction_transition.tolist(),
            "friction_coefficients_nm": self.ctrl.friction_coefficients.tolist(),
            "friction": self.ctrl.robot.get_friction_info() if hasattr(self.ctrl.robot, "get_friction_info") else None,
            "gravity": self.ctrl.robot.get_gravity_info() if hasattr(self.ctrl.robot, "get_gravity_info") else None,
            "torque": self.ctrl.robot.get_torque_breakdown() if hasattr(self.ctrl.robot, "get_torque_breakdown") else None,
            "integral_deadband_deg": self.ctrl.integral_deadband_deg,
            "integral_torque_nm": (
                self.ctrl.robot.get_integral_torque()[:6].tolist()
                if hasattr(self.ctrl.robot, "get_integral_torque")
                else [0.0] * 6
            ),
            "closed": self.ctrl.closed,
            "disable_pending": self.ctrl.disable_pending,
            "feedback_live": not self.ctrl.closed,
            "feedback_source": ("saved rest record" if self.ctrl.cached_rest_display else
                                "last SDK reading" if self.ctrl.closed else "SDK feedback" if self.ctrl.live else "ideal simulation"),
            "q_rad": q[:6].tolist(),
            "target_rad": self.ctrl.target[:6].tolist(),
            "rest_rad": self.ctrl.rest[:6].tolist(),
            "gripper_hold_closed": self.ctrl.gripper_hold_closed,
            "gripper_target_normalized": self.ctrl.gripper_target,
            "gripper_position_normalized": float(q[6]),
            "rest_verification_scope": "arm only; gripper object contact excluded" if self.ctrl.gripper_hold_closed else "arm and gripper",
            "error_deg": error.tolist(),
            "max_error_deg": float(np.max(np.abs(error))),
            "accuracy_pass": None if self.ctrl.cached_rest_display else bool(np.all(np.abs(error) <= np.rad2deg(ANGLE_TOL) + 1e-12)),
            "accuracy_tolerance_deg": np.rad2deg(ANGLE_TOL).tolist(),
            "joint_accuracy_pass": (np.abs(error) <= np.rad2deg(ANGLE_TOL) + 1e-12).tolist(),
            "max_normalized_error": float(np.max(np.abs(error) / np.rad2deg(ANGLE_TOL))),
            "convergence_goal_s": 1.0,
            "rest_tolerance_deg": float(np.rad2deg(REST_TOL)),
            "rest_verified": self.ctrl.rest_verified,
            "at_rest": self.ctrl.at_rest(),
            "camera_fresh": self.camera.fresh if self.camera else False,
            "tcp_position_m": T[:3, 3].tolist(),
            "tcp_rpy_rad": Rotation.from_matrix(T[:3, :3]).as_euler("xyz").tolist(),
            "message": self.message,
            "fault": (self.ctrl.motion_fault() or {}).get("reason", self.fault),
            "motion_fault": self.ctrl.motion_fault(),
            "oscillation_guard_active": hasattr(self.ctrl.robot, "get_motion_fault"),
            "oscillation_guard": self.ctrl.robot.get_oscillation_guard_info() if hasattr(self.ctrl.robot, "get_oscillation_guard_info") else None,
            "timestamp": time.time(),
        }

    def run(self):
        last_render = 0.0
        print(f"AutoWeld ready: http://127.0.0.1:{self._port} | runtime={self.runtime}", flush=True)
        while self.running:
            started = time.monotonic()
            for path in sorted(self.inbox.glob("*.json")):
                try:
                    req = json.loads(path.read_text())
                    req["_id"] = path.stem
                    self.events.put(req)
                    path.unlink()
                except Exception as exc:
                    self.message = str(exc)
                    path.rename(path.with_suffix(".invalid"))
            # Coalesce drag targets so old UI poses cannot queue up a delayed sweep.
            pending = []
            while not self.events.empty():
                event = self.events.get_nowait()
                if (
                    pending
                    and event["action"] == "pose"
                    and pending[-1]["action"] == "pose"
                    and "_id" not in event
                    and "_id" not in pending[-1]
                ):
                    pending[-1] = event
                else:
                    pending.append(event)
            for req in pending:
                result = self.handle_request(req)
                if "_id" in req:
                    atomic_json(self.runtime / "responses" / f"{req['_id']}.json", result)
            try:
                was_closed = self.ctrl.closed
                self.tick_motion()
                self.process_pending_pose()
                if self.ctrl.closed and not was_closed:
                    self.message = "Motors disabled at rest. Click Enable motors to resume."
                    if self.camera and self.camera.fresh:
                        self.camera.save(self.runtime / "disabled-rest.jpg")
                if self.exit_after_disable and self.ctrl.closed:
                    self.running = False
                motion_fault = self.ctrl.motion_fault()
                if motion_fault and not self.ctrl.closed:
                    self.message = f"{motion_fault['reason']}: {motion_fault['response']}; inspect arm before acknowledging."
                    if self.ctrl.disable_pending:
                        self.message += (" Disable requested: verifying quiet rest." if self.ctrl.at_rest()
                                         else " Disable requested: acknowledge the fault after recovery to allow return to rest.")
                    atomic_json(self.runtime / "motion-fault.json", motion_fault)
                moving = self.ctrl.motion_plan is not None
                busy = self.ctrl.closed or moving or self.ctrl.disable_pending or bool(motion_fault) or bool(self.tracking_monitor)
                self.track_approach_button.disabled = busy
                self.track_run_button.disabled = busy
                self.track_preview_button.disabled = moving or self.ctrl.disable_pending
                self.ack_fault_button.disabled = self.ctrl.closed or not bool(motion_fault)
                if self.ctrl.mode in ("Hold", "Gravity compensation", "Joint", "IK"):
                    self.mode_select.value = self.ctrl.mode
                self.controller_select.value = next(k for k, v in CONTROLLER_NAMES.items() if v == self.ctrl.control_law)
                moving = self.ctrl.motion_plan is not None
                self.motion_button.disabled = moving or self.ctrl.disable_pending or bool(motion_fault)
                self.pos_speed_button.disabled = self.ctrl.control_law not in ("POS_VEL", "MIXED_J45") or moving or self.ctrl.disable_pending or bool(motion_fault) or bool(self.tracking_monitor)
                self.pos_speed_input.disabled = self.pos_speed_button.disabled
                for handle in self.motion_inputs.values():
                    handle.disabled = self.motion_button.disabled
                if self.ctrl.last_motion:
                    plan = self.ctrl.last_motion
                    self.motion_md.content = (
                        f"**{'Moving' if moving else 'Reference complete'}** · {plan['kind']} · {plan['duration_s']:.3f} s\n\n"
                        + f"Duration limited by **{plan.get('limiting_constraint', 'unknown').replace('_', ' ')}**.\n\n"
                        +
                        "| Planned quantity | Peak | Limit |\n|---|---:|---:|\n"
                        + "\n".join(f"| {key.replace('_', ' ')} | {plan['planned_peaks'][key]:.5f} | {value:g} |"
                                    for key, value in plan['limits'].items())
                        + "\n\nReference limits only; measured tracking may differ.")
                self.quantization_input.disabled = self.ctrl.control_law in ("POS_VEL", "MIXED_J45") or moving or self.ctrl.disable_pending or bool(motion_fault) or bool(self.tracking_monitor)
                self.controller_select.disabled = moving or self.ctrl.disable_pending or bool(motion_fault)
                self.ki_button.disabled = self.ctrl.control_law == "POS_VEL" or moving or self.ctrl.disable_pending or bool(motion_fault)
                for handle in (*self.gravity_inputs, self.gravity_model_select, self.gravity_button,
                               self.friction_enabled_input, *self.friction_width_inputs,
                               *self.friction_coeff_inputs, self.friction_button):
                    handle.disabled = self.ctrl.control_law == "POS_VEL" or moving or self.ctrl.disable_pending or bool(motion_fault)
                for handle in (*self.kp_inputs, *self.kd_inputs):
                    # Keep text edits intact until Apply; sync through dispatch below.
                    handle.disabled = self.ctrl.control_law == "POS_VEL" or moving or self.ctrl.disable_pending or bool(motion_fault)
                if self.ctrl.control_law == "MIXED_J45":
                    for handle in (self.friction_enabled_input, *self.friction_width_inputs,
                                   *self.friction_coeff_inputs, self.friction_button):
                        handle.disabled = True
                self.hold_button.disabled = self.ctrl.closed or self.ctrl.disable_pending or bool(motion_fault)
                self.update_lifecycle_buttons()
                self.update_line_recorder_gui()
                for handle in (
                    self.mode_select,
                    self.rest_button,
                    self.joint_button,
                    self.pose_button,
                ):
                    handle.disabled = busy
                self._mirror_robot()
                q = self.ctrl.positions()
                # URDF fingers run from negative (closed) to zero (open).
                opening = float(np.clip(q[6], 0, 1))
                self.visual.update_cfg(np.r_[q[:6], [-0.04695 * (1 - opening)] * 2])
                T = self._ee_pose_4x4()
                self.ee_frame.position = T[:3, 3]
                self.ee_frame.wxyz = self._mat3_to_wxyz(T[:3, :3])
                mount_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
                self.mount.position = self._data.xpos[mount_id]
                self.mount.wxyz = self._data.xquat[mount_id]
                if not self.ctrl.closed:
                    feedback = self.ctrl.robot.get_observations() if self.ctrl.live else None
                    with (self.runtime / "control-telemetry.jsonl").open("a") as log:
                        log.write(
                            json.dumps(
                                {
                                    "timestamp": time.time(),
                                    "q": feedback["joint_pos"][:6].tolist() if feedback else q[:6].tolist(),
                                    "feedback_timestamps": feedback.get("joint_timestamps", np.zeros(6)).tolist() if feedback else None,
                                    "sdk_timing": self.ctrl.robot.get_feedback_diagnostics() if self.ctrl.live else None,
                                    "target": self.ctrl.target[:6].tolist(),
                                    "gripper_position_normalized": float(q[6]),
                                    "gripper_target_normalized": self.ctrl.gripper_target,
                                    "gripper_hold_closed": self.ctrl.gripper_hold_closed,
                                    "gripper_effort_nm": float(feedback["gripper_eff"][0]) if feedback and "gripper_eff" in feedback else None,
                                    "controller": self.ctrl.control_law,
                                    "motion_reference": self.ctrl.robot.get_motion_reference() if hasattr(self.ctrl.robot, "get_motion_reference") else None,
                                    "ki": self.ctrl.ki.tolist(),
                                    "pid_kp": self.ctrl.kp.tolist(),
                                    "pid_kd": self.ctrl.kd.tolist(),
                                    "gravity_factors": self.ctrl.gravity_factors.tolist(),
                                    "gravity_model": self.ctrl.gravity_model,
                                    "friction": self.ctrl.robot.get_friction_info() if hasattr(self.ctrl.robot, "get_friction_info") else None,
                                    "torque": self.ctrl.robot.get_torque_breakdown() if hasattr(self.ctrl.robot, "get_torque_breakdown") else None,
                                    "integral_deadband_deg": self.ctrl.integral_deadband_deg,
                                    "velocity": feedback["joint_vel"][:6].tolist()
                                    if self.ctrl.live
                                    else None,
                                    "integral": self.ctrl.robot.get_integral_torque()[:6].tolist()
                                    if hasattr(self.ctrl.robot, "get_integral_torque")
                                    else [0.0] * 6,
                                }
                            )
                            + "\n"
                        )
                if started - last_render > 0.2:
                    state = self.state()
                    atomic_json(self.runtime / "status.json", state)
                    with (self.runtime / "telemetry.jsonl").open("a") as log:
                        log.write(json.dumps(state) + "\n")
                    self.status_md.content = f"**{self.ctrl.mode}** · {self.message}"
                    rows = "\n".join(
                        f"| J{i + 1} | {q[i]:.6f} | {self.ctrl.target[i]:.6f} | {state['error_deg'][i]:+.4f} |"
                        for i in range(6)
                    )
                    self.telemetry_md.content = (
                        "| Joint | Actual rad | Target rad | Error ° |\n|---|---:|---:|---:|\n"
                        + rows
                        + f"\n\nJ1–J3 ±0.2°, J4–J6 ±1°: **{'NO FEEDBACK' if state['accuracy_pass'] is None else 'PASS' if state['accuracy_pass'] else 'NOT MET'}**"
                        + f"\n\nRest verification: within {state['rest_tolerance_deg']:g}° for one second."
                        + (("\n\nClosed gripper target saved; gripper is unpowered."
                            if self.ctrl.closed else
                            "\n\nGripper held closed with SDK force limiting. Disable at rest also turns off the gripper.")
                           if self.ctrl.gripper_hold_closed else "")
                        + (
                            ("\n\nMotors disabled · saved rest display, not measured feedback."
                             if self.ctrl.cached_rest_display else "\n\nMotors disabled · showing last measured position.")
                            if self.ctrl.closed
                            else ""
                        )
                    )
                    torque = state.get("torque")
                    if torque and torque.get("controller") == "POS_VEL":
                        self.torque_md.content = ("Position–speed onboard loop terms are unavailable. "
                            "Motor-reported torque (Nm): " + ", ".join(f"{t:.3f}" for t in torque["motor_reported_nm"][:6]))
                    if torque and torque.get("controller") == "MIXED_J45":
                        self.torque_md.content = ("Mixed mode: J4/J5 MIT PD + gravity; other arm joints use onboard position–speed. "
                            "Motor-reported torque (Nm): " + ", ".join(f"{t:.3f}" for t in torque["motor_reported_nm"][:6])
                            + "\n\nGravity feedforward J4/J5 (Nm): "
                            + ", ".join(f"{torque['gravity_nm'][j]:.3f}" for j in (3, 4)))
                    if torque and "gravity_nm" in torque and torque.get("controller") != "MIXED_J45":
                        cols = ["p_estimate_nm", "i_nm", "d_estimate_nm", "gravity_nm", "friction_nm", "total_estimate_nm", "motor_reported_nm"]
                        self.torque_md.content = (
                            "Torque (Nm) · P/D and total are estimates; reported is SDK motor feedback. "
                            + ("Last active sample; motors disabled." if self.ctrl.closed else "")
                            + "\n\n| Joint | P | I | D | Gravity | Friction | Total est. | Reported |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
                            + "\n".join(f"| J{i+1} | " + " | ".join(f"{torque[c][i]:.3f}" for c in cols) + " |" for i in range(6))
                        )
                    if self.camera and self.camera.fresh:
                        self.camera_image.image = self.camera.rgb()
                    last_render = started
            except Exception as exc:
                self.fault = str(exc)
                self.message = str(exc)
                self.status_md.content = f"**Fault:** {exc}. SDK fault handling applies; inspect hardware."
                atomic_json(self.runtime / "status.json", {"fault": str(exc), "timestamp": time.time()})
            time.sleep(max(0.0, self._dt - (time.monotonic() - started)))
        self.server.stop()
        if not self.ctrl.live and not self.ctrl.closed:
            self._robot.close()
        if self.camera:
            self.camera.close()


