"""Structured rejection reasons; diagnostic work never sends robot commands."""

import mink
import numpy as np

IK_POSITION_TOL = 1e-8
IK_ORIENTATION_TOL = 1e-8
IK_MAX_ITERS = 300


class MotionRejection(ValueError):
    def __init__(self, stage, code, message, **details):
        super().__init__(message)
        self.diagnostic = {"stage": stage, "code": code, "message": message, **details}


def solve_pose(kin, target, site, seed, *, stage, path_fraction=None, orientation_required=True, joint_motion_cost=None):
    """Preserve the SDK solve and thresholds; report its actual stopping residual."""
    location = {} if path_fraction is None else {"path_fraction": float(path_fraction)}
    where = "Endpoint IK" if path_fraction is None else f"Path IK at {path_fraction:.1%}"
    options = {}
    if not orientation_required:
        options["orientation_cost"] = 0.0
    if joint_motion_cost is not None:
        options["joint_motion_cost"] = joint_motion_cost
    try:
        ok, q = kin.ik(target, site, init_q=seed.copy(),
                       pos_threshold=IK_POSITION_TOL, ori_threshold=IK_ORIENTATION_TOL,
                       max_iters=IK_MAX_ITERS, **options)
    except Exception as exc:
        raise MotionRejection(stage, "ik_solver_error",
                              f"{where}: solver raised {type(exc).__name__}: {exc}",
                              **location, exception_type=type(exc).__name__) from exc
    if ok:
        return q.copy()
    if not np.isfinite(q).all():
        raise MotionRejection(stage, "ik_nonfinite", f"{where}: solver returned non-finite joint values",
                              **location)
    actual = kin.fk(q.copy(), site)
    # Match Mink FrameTask.compute_error, including its SE(3) translation-log metric.
    error = mink.SE3.from_matrix(target).minus(mink.SE3.from_matrix(actual))
    pos, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
    checks = {
        "position": {"error": pos, "limit": IK_POSITION_TOL, "unit": "m", "passed": pos <= IK_POSITION_TOL},
        "orientation": {"error": angle, "limit": IK_ORIENTATION_TOL, "unit": "rad", "required": orientation_required, "passed": not orientation_required or angle <= IK_ORIENTATION_TOL},
    }
    failed = ", ".join(name for name, check in checks.items() if not check["passed"])
    raise MotionRejection(
        stage, "ik_nonconvergence",
        f"{where}: IK did not converge after {IK_MAX_ITERS} iterations. "
        f"Failed tolerance: {failed or 'SDK success flag'}. "
        f"Position residual {pos * 1000:.6g} mm (limit {IK_POSITION_TOL * 1000:g} mm); "
        f"orientation {np.rad2deg(angle):.6g}° (limit {np.rad2deg(IK_ORIENTATION_TOL):.6g}°). "
        "This search failure does not prove the pose is unreachable.",
        **location, iterations=IK_MAX_ITERS, checks=checks,
        cartesian_position_error_m=float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
        candidate_joint_rad=q[:6].tolist(),
    )
