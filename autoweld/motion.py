"""Cartesian waypoint planning and sampled kinematic limits, without hardware I/O.

T_world_tcp maps the configured TCP to robot base/world coordinates. Angular rates
are world-frame angular velocity derivatives, not Euler-angle derivatives.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.waypoint_motion import JointPath, time_scale
from scipy.spatial.transform import Rotation

from autoweld.diagnostics import MotionRejection, solve_pose


@dataclass(frozen=True)
class MotionLimits:
    linear_velocity: float = 0.03  # m/s
    linear_acceleration: float = 0.15  # m/s²
    linear_jerk: float = 1.0  # m/s³
    angular_velocity: float = 0.25  # rad/s
    angular_acceleration: float = 1.0  # rad/s²
    angular_jerk: float = 5.0  # rad/s³
    joint_velocity: float = 0.35  # rad/s, each joint
    joint_acceleration: float = 1.0  # rad/s², each joint
    joint_jerk: float = 5.0  # rad/s³, each joint

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Motion limits must be finite and positive")


@dataclass
class MotionPlan:
    path: JointPath
    report: dict


class MotionPlanner:
    def __init__(self, model_path, site="weld_tcp"):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.site < 0:
            raise ValueError("TCP site not found")
        self.site_name = site
        self.kin = Kinematics(str(model_path), site)
        self.limits = MotionLimits()

    def _configuration(self, q):
        result = np.zeros(self.model.nq)
        result[:6] = q[:6]
        return result

    def fk(self, q):
        self.data.qpos[:] = self._configuration(q)
        mujoco.mj_forward(self.model, self.data)
        T = np.eye(4)
        T[:3, 3] = self.data.site_xpos[self.site]
        T[:3, :3] = self.data.site_xmat[self.site].reshape(3, 3)
        return T

    def jacobian(self, q):
        self.fk(q)
        p, r = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, p, r, self.site)
        return np.vstack((p[:, :6], r[:, :6]))

    def path_derivatives(self, path, s):
        q, qs, qss, qsss = (path.spline(s, n)[:6] for n in range(4))
        h = 1e-4
        J = self.jacobian(q)
        Jp = self.jacobian(path.spline(s + h))
        Jm = self.jacobian(path.spline(s - h))
        Js = (Jp - Jm) / (2 * h)
        Jss = (Jp - 2 * J + Jm) / h**2
        return J @ qs, J @ qss + Js @ qs, J @ qsss + 2 * Js @ qss + Jss @ qs

    def derivatives(self, path, elapsed):
        s, ds, dds, ddds = time_scale(elapsed / path.duration)
        v, a, j = self.path_derivatives(path, float(s))
        return (
            v * ds / path.duration,
            (a * ds**2 + v * dds) / path.duration**2,
            (j * ds**3 + 3 * a * ds * dds + v * ddds) / path.duration**3,
        )

    def plan(self, start, goal, *, cartesian=None, bounds=None, validate=None, collision=None):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        if start.shape != (7,) or goal.shape != (7,) or not np.isfinite([start, goal]).all():
            raise ValueError("Expected finite seven-joint start and goal")
        if not np.isclose(start[6], goal[6], atol=1e-12):
            raise ValueError("Gripper must stay fixed through a waypoint")
        T0 = self.fk(start)
        if cartesian is None:
            knots, positions = np.array([0.0, 1.0]), np.array([start, goal])
        else:
            T1 = np.asarray(cartesian, float)
            if T1.shape != (4, 4) or not np.isfinite(T1).all():
                raise ValueError("Expected finite TCP transform")
            rotvec = Rotation.from_matrix(T1[:3, :3] @ T0[:3, :3].T).as_rotvec()
            knots = np.linspace(0, 1, 41)
            positions = [start.copy()]
            seed = self._configuration(start)
            for s in knots[1:]:
                T = np.eye(4)
                T[:3, 3] = T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3])
                T[:3, :3] = Rotation.from_rotvec(s * rotvec).as_matrix() @ T0[:3, :3]
                seed = solve_pose(self.kin, T, self.site_name, seed, stage="path_ik", path_fraction=s)
                q = np.r_[seed[:6], start[6]]
                steps = np.abs(q[:6] - positions[-1][:6])
                if np.max(steps) > 0.05:
                    joint = int(np.argmax(steps))
                    raise MotionRejection("path_continuity", "joint_step",
                        f"Path rejection at {s:.1%}: J{joint+1} changes {steps[joint]:.6g} rad "
                        f"between samples (limit 0.05 rad). Possible IK branch change, singularity, "
                        "or insufficient path sampling; cause not established.",
                        path_fraction=float(s), prior_path_fraction=float(s - 1/40),
                        joint=joint+1, step_rad=float(steps[joint]), limit_rad=0.05)
                positions.append(q)
            positions = np.asarray(positions)
        path = JointPath(knots, positions, 1.0)
        # Dense path validation includes spline interiors, not just IK knots.
        cart_error = np.zeros(2)
        cart_error_at = np.zeros(2)
        for s in np.linspace(0, 1, 401):
            q = path.spline(s)
            if bounds is not None and (
                np.any(q[:6] < bounds[:, 0] - 1e-10) or np.any(q[:6] > bounds[:, 1] + 1e-10)
            ):
                bad = np.flatnonzero((q[:6] < bounds[:, 0] - 1e-10) | (q[:6] > bounds[:, 1] + 1e-10))
                joint = int(bad[0])
                raise MotionRejection("path_validation", "joint_bounds",
                    f"Motion path exceeds joint bounds at {s:.1%}: J{joint+1} = {q[joint]:.6g} rad, "
                    f"allowed [{bounds[joint, 0]:.6g}, {bounds[joint, 1]:.6g}] rad.",
                    path_fraction=float(s), joint=joint+1, value_rad=float(q[joint]),
                    bounds_rad=bounds[joint].tolist())
            if validate:
                try:
                    validate(q)
                except (ValueError, RuntimeError) as exc:
                    raise MotionRejection("path_validation", "command_constraint",
                        f"Path validation at {s:.1%}: {exc}", path_fraction=float(s),
                        cause=exc.diagnostic if isinstance(exc, MotionRejection) else None) from exc
            if collision and collision(q):
                raise MotionRejection("path_validation", "self_collision",
                    f"Robot self-collision along motion path at {s:.1%}", path_fraction=float(s))
            if cartesian is not None:
                T = self.fk(q)
                desired = Rotation.from_rotvec(s * rotvec).as_matrix() @ T0[:3, :3]
                residual = np.array([
                        np.linalg.norm(T[:3, 3] - (T0[:3, 3] + s * (T1[:3, 3] - T0[:3, 3]))),
                        np.linalg.norm(Rotation.from_matrix(T[:3, :3] @ desired.T).as_rotvec()),
                    ])
                cart_error_at = np.where(residual > cart_error, s, cart_error_at)
                cart_error = np.maximum(cart_error, residual)
        if np.any(cart_error > [2e-4, 2e-4]):
            raise MotionRejection("path_tolerance", "tcp_residual",
                f"Interpolated IK path exceeds TCP pose residual tolerance. "
                f"Peak position error {cart_error[0]*1000:.6g} mm at {cart_error_at[0]:.1%} "
                f"(limit 0.2 mm, {'FAIL' if cart_error[0] > 2e-4 else 'PASS'}); "
                f"peak orientation error {np.rad2deg(cart_error[1]):.6g}° at {cart_error_at[1]:.1%} "
                f"(limit {np.rad2deg(2e-4):.6g}°, {'FAIL' if cart_error[1] > 2e-4 else 'PASS'}).",
                checks={name: {"error": float(cart_error[i]), "limit": 2e-4, "unit": unit,
                               "passed": bool(cart_error[i] <= 2e-4), "path_fraction": float(cart_error_at[i])}
                        for i, (name, unit) in enumerate((("position", "m"), ("orientation", "rad")))})
        # Analytic chain rule; J derivatives are central differences along the path.
        tcp = np.array([self.derivatives(path, u) for u in np.linspace(0, 1, 801)])
        joints = np.array([path.sample(u)[1:] for u in np.linspace(0, 1, 801)])[:, :, :6]
        raw = np.r_[
            np.max(np.linalg.norm(tcp[:, :, :3], axis=2), axis=0),
            np.max(np.linalg.norm(tcp[:, :, 3:], axis=2), axis=0),
            np.max(np.abs(joints), axis=(0, 2)),
        ]
        maxima = np.array(list(asdict(self.limits).values()))
        orders = np.tile([1, 2, 3], 3)
        required_times = 1.10 * (raw / maxima) ** (1 / orders)
        duration = max(0.2, float(np.max(required_times)))
        path.duration = duration
        report = {
            "duration_s": duration,
            "limiting_constraint": list(asdict(self.limits))[int(np.argmax(required_times))] if duration > .2 else "minimum duration",
            "duration_by_constraint_s": dict(zip(asdict(self.limits), required_times.tolist())),
            "tcp_distance_m": float(np.linalg.norm(self.fk(path.positions[-1])[:3, 3]-T0[:3, 3])),
            "kind": "Cartesian IK" if cartesian is not None else "Joint",
            "limits": asdict(self.limits),
            "planned_peaks": dict(zip(asdict(self.limits), (raw / duration**orders).tolist())),
            "joint_peak_velocity_rad_s": (np.max(np.abs(joints[:, 0]), axis=0) / duration).tolist(),
            "joint_peak_acceleration_rad_s2": (np.max(np.abs(joints[:, 1]), axis=0) / duration**2).tolist(),
            "joint_peak_jerk_rad_s3": (np.max(np.abs(joints[:, 2]), axis=0) / duration**3).tolist(),
            "tcp_path_error_m": float(cart_error[0]),
            "tcp_path_error_rad": float(cart_error[1]),
            "verification": "801 samples, 10% duration margin; reference limits, not measured-motion guarantees",
        }
        return MotionPlan(path, report)
