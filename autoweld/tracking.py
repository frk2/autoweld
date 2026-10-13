"""STEP line/arc tracking, model-based preflight and measured-FK evaluation."""

from dataclasses import asdict, dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
from i2rt.robots.waypoint_motion import FeedJointPath
from scipy.spatial.transform import Rotation

from autoweld.diagnostics import MotionRejection, solve_pose
from autoweld.motion import MotionPlan
from autoweld.teach import LineRecorder
from autoweld.tracking_limits import FEED_TOLERANCE_FRACTION, ORIENTATION_TOLERANCE_DEG, POSITION_TOLERANCE_M
from autoweld.velocity import PositionVelocity


def unit(v):
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-10:
        raise ValueError("Expected a finite nonzero 3D direction")
    return v / np.linalg.norm(v)


@dataclass
class TrackCurve:
    kind: str
    start: np.ndarray
    direction: np.ndarray
    length: float
    center: np.ndarray | None = None
    normal: np.ndarray | None = None
    radius: float = 0.0
    sweep: float = 0.0
    recorded_points: dict | None = None

    def point(self, s):
        if self.kind == "line":
            return self.start + s * self.length * self.direction
        return self.center + Rotation.from_rotvec(self.normal * s * self.sweep).apply(
            self.start - self.center
        )

    def tangent(self, s):
        if self.kind == "line":
            return self.direction
        return unit(np.cross(self.normal, self.point(s) - self.center))

    def pose(self, s, initial_rotation, orientation):
        T = np.eye(4)
        T[:3, 3] = self.point(s)
        if orientation == "Recorded poses":
            if self.kind != "line" or self.recorded_points is None:
                raise ValueError("Recorded poses requires a line saved with both endpoint poses")
            rotations = [Rotation.from_quat(np.array(self.recorded_points[k]["quaternion_wxyz"])[[1, 2, 3, 0]])
                         for k in ("A", "B")]
            # Shortest rotation interpolated by distance. Extrapolate on outside-edge
            # ramps so angular derivatives stay continuous at A/B; default is inside.
            delta = (rotations[0].inv() * rotations[1]).as_rotvec()
            T[:3, :3] = (rotations[0] * Rotation.from_rotvec(s * delta)).as_matrix()
        elif orientation == "Center facing":
            if self.kind != "arc":
                raise ValueError("Center-facing orientation requires a circular edge")
            z = unit(self.point(s) - self.center)  # Torch points along TCP -Z.
            x = self.normal
            T[:3, :3] = np.column_stack((x, np.cross(z, x), z))
        elif orientation == "Perpendicular":
            if self.kind != "line":
                raise ValueError("Perpendicular orientation requires a straight edge")
            x = self.direction
            z = initial_rotation[:, 2] - x * np.dot(initial_rotation[:, 2], x)
            if np.linalg.norm(z) < 1e-6:
                z = np.cross(x, np.eye(3)[np.argmin(abs(x))])
            z = unit(z)
            turn, _ = Rotation.align_vectors(z[None, :], initial_rotation[:, 2][None, :])
            T[:3, :3] = turn.as_matrix() @ initial_rotation
        else:
            T[:3, :3] = initial_rotation
        return T

    def distance(self, point):
        """Distance to finite requested edge (not its extension)."""
        if self.kind == "line":
            s = np.clip(np.dot(point - self.start, self.direction) / self.length, 0, 1)
            return float(np.linalg.norm(point - self.point(s)))
        delta = point - self.center
        radial = delta - self.normal * np.dot(delta, self.normal)
        initial = unit(self.start - self.center)
        angle = np.arctan2(np.dot(self.normal, np.cross(initial, radial)), np.dot(initial, radial)) % (
            2 * np.pi
        )
        if angle <= self.sweep or self.sweep >= 2 * np.pi - 1e-7:
            return float(np.hypot(np.linalg.norm(radial) - self.radius, np.dot(delta, self.normal)))
        return float(min(np.linalg.norm(point - self.start), np.linalg.norm(point - self.point(1))))


def load_tracks(path):
    if Path(path).suffix.lower() not in (".step", ".stp"):
        raise ValueError("Select a STEP file containing line or circular edges")
    shape = cq.importers.importStep(str(path))
    curves = []
    for edge in shape.edges().vals():
        adaptor = edge._geomAdaptor()
        first, last = adaptor.FirstParameter(), adaptor.LastParameter()
        a, b = (np.array(adaptor.Value(u).Coord()) * 0.001 for u in (first, last))
        if edge.geomType() == "LINE":
            length = float(np.linalg.norm(b - a))
            if length > 1e-6:
                curves.append(TrackCurve("line", a, unit(b - a), length))
        elif edge.geomType() == "CIRCLE":
            circle = adaptor.Circle()
            center = np.array(circle.Location().Coord()) * 0.001
            normal = unit(circle.Axis().Direction().Coord())
            radius = circle.Radius() * 0.001
            sweep = last - first
            curves.append(
                TrackCurve(
                    "arc",
                    a,
                    unit(np.cross(normal, a - center)),
                    radius * sweep,
                    center,
                    normal,
                    radius,
                    sweep,
                )
            )
    if not curves:
        raise ValueError("STEP has no supported straight or circular edges")
    recording = LineRecorder.load(path, curves)
    if recording is not None:
        curves[0].recorded_points = recording.points
    return curves


@dataclass
class TrackingPlan:
    motion: MotionPlan
    curve: TrackCurve
    orientation: str
    rotation: np.ndarray
    feed: float
    ramp: float
    lead: float

    def curve_fraction(self, elapsed):
        s = self.motion.path.phase(elapsed)[0]
        return (s * (self.curve.length + 2 * self.lead) - self.lead) / self.curve.length

    def expected(self, elapsed):
        return self.curve.pose(self.curve_fraction(elapsed), self.rotation, self.orientation)


def plan_tracking(
    planner, seed, curve, feed=0.005, orientation="Unconstrained", bounds=None, validate=None, collision=None,
    ramp_placement="Outside edge",
):
    if ramp_placement not in ("Outside edge", "Inside edge"):
        raise ValueError("Ramp placement must be Outside edge or Inside edge")
    if orientation not in ("Unconstrained", "Perpendicular", "Center facing", "Recorded poses"):
        raise ValueError("Unknown tracking orientation")
    if not np.isfinite(feed) or not 0 < feed <= planner.limits.linear_velocity:
        raise ValueError("Feed must be positive and no greater than the TCP speed limit")
    rotation = planner.fk(seed)[:3, :3].copy()
    orientation_required = orientation != "Unconstrained"
    # Free orientation should not force opposing J1/J5 motion to hold a wrist
    # rotation the operator did not request. Prefer proximal joints smoothly,
    # while retaining all DOFs when needed to reach the position.
    joint_motion_cost = None
    if not orientation_required:
        joint_motion_cost = np.full(planner.kin._configuration.model.nv, .1)
        joint_motion_cost[:3] = .01
    # Lead-in/out place all of the requested edge inside the constant-feed interval.
    ramp = float(max(
        0.5, feed * 2 / planner.limits.linear_acceleration, np.sqrt(feed * 7 / planner.limits.linear_jerk)
    ))
    for attempt in range(5):
        inside = ramp_placement == "Inside edge"
        if inside and curve.length <= feed * ramp:
            raise MotionRejection("tracking_limits", "insufficient_feed_length",
                "Edge is too short to reach this feed within acceleration/jerk limits")
        lead = 0. if inside else feed * ramp / 2
        length = curve.length + 2 * lead
        duration = curve.length / feed + (ramp if inside else 2 * ramp)
        knots = np.linspace(0, 1, max(41, min(801, int(np.ceil(length / 0.0005)) + 1)))
        positions = []
        q = planner._configuration(seed)
        for s in knots:
            fraction = (s * length - lead) / curve.length
            T = curve.pose(fraction, rotation, orientation)
            q = solve_pose(planner.kin, T, planner.site_name, q, stage="tracking_ik", path_fraction=s,
                           orientation_required=orientation_required, joint_motion_cost=joint_motion_cost)
            arm = np.r_[q[:6], seed[6]]
            positions.append(arm)
        path = FeedJointPath(knots, np.array(positions), duration, ramp / duration)
        peaks = np.zeros(9)
        joint_peaks = np.zeros((3, 6))
        residual = np.zeros(2)
        for t in np.linspace(0, duration, 1001):
            s, ds, dds, ddds = path.phase(t)
            q, v, a, j = path.sample(t)
            if bounds is not None and (np.any(q[:6] < bounds[:, 0]) or np.any(q[:6] > bounds[:, 1])):
                raise MotionRejection(
                    "tracking_validation", "joint_bounds", f"Tracking joint bounds exceeded at {s:.1%}"
                )
            if validate:
                validate(q)
            if collision and collision(q):
                raise MotionRejection(
                    "tracking_validation", "self_collision", f"Tracking self-collision at {s:.1%}"
                )
            pv, pa, pj = planner.path_derivatives(path, s)
            derivatives = np.array(
                [pv * ds, pa * ds**2 + pv * dds, pj * ds**3 + 3 * pa * ds * dds + pv * ddds]
            )
            raw = np.r_[
                np.linalg.norm(derivatives[:, :3], axis=1),
                np.linalg.norm(derivatives[:, 3:], axis=1),
                np.max(abs(np.array([v, a, j])[:, :6]), axis=1),
            ]
            peaks = np.maximum(peaks, raw)
            joint_peaks = np.maximum(joint_peaks, abs(np.array([v, a, j])[:, :6]))
            actual = planner.fk(q)
            desired = curve.pose((s * length - lead) / curve.length, rotation, orientation)
            err = [
                np.linalg.norm(actual[:3, 3] - desired[:3, 3]),
                np.linalg.norm(Rotation.from_matrix(actual[:3, :3] @ desired[:3, :3].T).as_rotvec()),
            ]
            residual = np.maximum(residual, err)
        if residual[0] > .0002 or (orientation_required and residual[1] > .0002):
            raise MotionRejection(
                "tracking_tolerance",
                "tcp_residual",
                f"Tracking interpolation error {residual[0] * 1000:.4g} mm, {np.rad2deg(residual[1]):.4g}° exceeds planning tolerance",
            )
        limits = np.array(list(asdict(planner.limits).values()))
        acceptance = limits * .95
        # The requested constant feed may equal the operator TCP speed cap.
        # Keep the 5% reserve on every other derivative; permit only numerical
        # spline/Jacobian noise (1 micrometre/s) at the translational speed cap.
        acceptance[0] = limits[0] + 1e-6
        if np.all(peaks <= acceptance):
            break
        ramp *= 1.5
    else:
        name = list(asdict(planner.limits))[int(np.argmax(peaks / limits))]
        raise MotionRejection(
            "tracking_limits",
            "derivative_limit",
            f"Tracking exceeds {name} at this feed; lower the requested feed speed",
        )
    report = {
        "kind": "Path tracking",
        "duration_s": duration,
        "limiting_constraint": "requested feed",
        "limits": asdict(planner.limits),
        "planned_peaks": dict(zip(asdict(planner.limits), peaks.tolist())),
        "tcp_distance_m": length,
        "edge_length_m": curve.length,
        "feed_m_s": feed,
        "ramp_s": ramp,
        "lead_in_out_m": lead,
        "ramp_placement": ramp_placement,
        "constant_feed_duration_s": duration - 2*ramp,
        "orientation": orientation,
        "orientation_policy": "pose constrained" if orientation_required else "position only, minimum wrist motion",
        "tcp_path_error_m": float(residual[0]),
        "tcp_path_error_rad": float(residual[1]),
        "joint_peak_velocity_rad_s": joint_peaks[0].tolist(),
        "joint_peak_acceleration_rad_s2": joint_peaks[1].tolist(),
        "joint_peak_jerk_rad_s3": joint_peaks[2].tolist(),
        "verification": "1001 reference samples; measured tracking evaluated separately",
    }
    return TrackingPlan(MotionPlan(path, report), curve, orientation, rotation, feed, ramp, lead)


def tracking_sample(plan, planner, q, velocity, elapsed):
    actual = planner.fk(q).copy()
    expected = plan.expected(elapsed)
    fraction = plan.curve_fraction(elapsed)
    jacobian = planner.jacobian(q)[:3]
    tcp_velocity = jacobian @ np.asarray(velocity)[:6]
    tangent = plan.curve.tangent(fraction)
    along = float(tcp_velocity @ tangent)
    speed = float(np.linalg.norm(tcp_velocity))
    error = float(np.linalg.norm(actual[:3, 3] - expected[:3, 3]))
    edge_distance = plan.curve.distance(actual[:3, 3])
    angle = float(
        np.rad2deg(np.linalg.norm(Rotation.from_matrix(actual[:3, :3] @ expected[:3, :3].T).as_rotvec()))
    )
    on_edge = plan.ramp <= elapsed <= plan.motion.path.duration - plan.ramp
    position_pass = edge_distance <= POSITION_TOLERANCE_M
    orientation_pass = plan.orientation == "Unconstrained" or angle <= ORIENTATION_TOLERANCE_DEG
    return {
        "elapsed_s": elapsed,
        "edge_fraction": float(fraction),
        "on_edge": bool(on_edge),
        "actual_m": actual[:3, 3].tolist(),
        "reference_m": expected[:3, 3].tolist(),
        "tracking_error_mm": error * 1000,
        "cross_track_mm": edge_distance * 1000,
        "orientation_error_deg": angle,
        "speed_m_s": speed,
        "raw_along_speed_m_s": along,
        "raw_cross_speed_m_s": float(np.linalg.norm(tcp_velocity-along*tangent)),
        "joint_position_rad": np.asarray(q)[:6].tolist(),
        "joint_velocity_raw_rad_s": np.asarray(velocity)[:6].tolist(),
        "requested_speed_m_s": plan.feed,
        "position_tolerance_mm": POSITION_TOLERANCE_M * 1000,
        "position_metric": "edge_distance",
        "position_pass": position_pass,
        "orientation_pass": orientation_pass,
        "deviating": not (position_pass and orientation_pass),
        "deviation_action": "record_only",
        "speed_pass": abs(speed / plan.feed - 1) <= FEED_TOLERANCE_FRACTION,
    }


class TrackingMonitor:
    def __init__(self, plan):
        self.plan = plan
        self.rows = []
        self.velocity = PositionVelocity()
        self.last_feedback_timestamps = None

    def add_velocity_diagnostics(self, row, planner, q, timestamps, now):
        timestamps = np.asarray(timestamps, float)[:6]
        row["feedback_timestamps"] = timestamps.tolist()
        row["feedback_age_s"] = float(now-min(timestamps))
        row["feedback_skew_s"] = float(np.ptp(timestamps))
        fresh = bool(np.isfinite(timestamps).all() and row["feedback_age_s"] <= .05
                     and max(timestamps) <= now+.005 and row["feedback_skew_s"] <= .01)
        unique = bool(self.last_feedback_timestamps is None
                      or np.all(timestamps > self.last_feedback_timestamps))
        row["velocity_sample_valid"] = fresh and unique
        self.last_feedback_timestamps = timestamps.copy()
        if not row["velocity_sample_valid"]:
            row["speed_pass"] = False
        estimate = self.velocity.update(timestamps, q, now)
        row["position_velocity"] = estimate
        if estimate is not None:
            # The velocity fit and Jacobian must describe the same pose/time.
            jacobian = planner.jacobian(estimate["joint_position_rad"])[:3]
            v = jacobian @ np.array(estimate["joint_velocity_rad_s"])
            elapsed = row["elapsed_s"] + (estimate["timestamp"] - float(min(timestamps)))
            fraction = self.plan.curve_fraction(elapsed)
            tangent = self.plan.curve.tangent(fraction)
            along = float(v @ tangent)
            estimate.update({"speed_m_s": float(np.linalg.norm(v)), "along_speed_m_s": along,
                             "elapsed_s": float(elapsed),
                             "on_edge": bool(self.plan.ramp <= elapsed <= self.plan.motion.path.duration-self.plan.ramp),
                             "reference_speed_m_s": float(self.plan.motion.path.phase(elapsed)[1]
                                 * (self.plan.curve.length+2*self.plan.lead)),
                             "cross_speed_m_s": float(np.linalg.norm(v-along*tangent)),
                             "tcp_quantization_bound_m_s": float(np.linalg.norm(
                                 abs(jacobian) @ np.array(estimate["joint_quantization_bound_rad_s"])))})

    def summary(self, completed=False, reason=None):
        edge = [r for r in self.rows if r["on_edge"]]
        gaps = np.diff([r["elapsed_s"] for r in self.rows])
        position = max((r["tracking_error_mm"] for r in self.rows), default=0.0)
        edge_distance = max((r["cross_track_mm"] for r in self.rows), default=0.0)
        orientation = max((r["orientation_error_deg"] for r in edge), default=0.0)
        speed_error = max((abs(r["speed_m_s"] / self.plan.feed - 1) * 100 for r in edge), default=0.0)
        feed_start = float(self.plan.curve_fraction(self.plan.ramp))
        feed_end = float(self.plan.curve_fraction(self.plan.motion.path.duration-self.plan.ramp))
        margin = .03 * (feed_end-feed_start)
        coverage = bool(edge and edge[0]["edge_fraction"] < feed_start+margin and edge[-1]["edge_fraction"] > feed_end-margin)
        sampling_ok = (len(gaps) > 0 and float(min(gaps)) > 0 and float(max(gaps)) <= 0.1
                       and all(r.get("velocity_sample_valid", True) for r in edge))
        position_speeds = [r["position_velocity"] for r in self.rows
                           if r.get("position_velocity") and r["position_velocity"].get("on_edge")]
        final_hold = bool(self.rows and self.rows[-1]["elapsed_s"] >= self.plan.motion.path.duration + 1.0)
        position_deviations = [r for r in self.rows if r['cross_track_mm'] > POSITION_TOLERANCE_M * 1000]
        orientation_deviations = [r for r in self.rows if self.plan.orientation != 'Unconstrained'
                                  and r['orientation_error_deg'] > ORIENTATION_TOLERANCE_DEG]
        deviations = [r for r in self.rows if r['cross_track_mm'] > POSITION_TOLERANCE_M * 1000
                      or (self.plan.orientation != 'Unconstrained'
                          and r['orientation_error_deg'] > ORIENTATION_TOLERANCE_DEG)]
        passed = bool(
            completed
            and final_hold
            and coverage
            and sampling_ok
            and not deviations
            and edge_distance <= POSITION_TOLERANCE_M * 1000
            and all(r["speed_pass"] and r["orientation_pass"] for r in edge)
        )
        return {
            "completed": completed,
            "passed": passed,
            "reason": reason,
            "deviated": bool(deviations),
            "position_deviated": bool(position_deviations),
            "orientation_deviated": bool(orientation_deviations),
            "deviation_action": "record_only",
            "deviation_samples": len(deviations),
            "first_deviation_s": deviations[0]['elapsed_s'] if deviations else None,
            "samples": len(self.rows),
            "edge_samples": len(edge),
            "edge_coverage": coverage,
            "feed_interval_edge_fraction": [feed_start, feed_end],
            "sampling_ok": sampling_ok,
            "invalid_velocity_samples": sum(not r.get("velocity_sample_valid", True) for r in self.rows),
            "final_hold_verified": final_hold,
            "position_tolerance_mm": POSITION_TOLERANCE_M * 1000,
            "position_metric": "edge_distance",
            "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
            "feed_tolerance_percent": FEED_TOLERANCE_FRACTION * 100,
            "max_tracking_error_mm": position,
            "max_cross_track_mm": edge_distance if self.rows else None,
            "max_on_edge_cross_track_mm": max((r["cross_track_mm"] for r in edge), default=None),
            "max_orientation_error_deg": orientation if edge else None,
            "max_orientation_error_all_deg": max((r['orientation_error_deg'] for r in self.rows), default=None),
            "max_speed_error_percent": speed_error if edge else None,
            "measurement": "Raw encoder Jacobian velocity determines qualification; position derivatives use a common fit time/pose and their own reference time. Encoder estimates, not external tip metrology.",
            "velocity_estimator_window_s": self.velocity.window,
            "max_position_slope_speed_m_s": max((r["speed_m_s"] for r in position_speeds), default=None),
        }
