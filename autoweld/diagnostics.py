"""Structured rejection reasons; diagnostic work never sends robot commands."""

import time

import mink
import numpy as np

IK_POSITION_TOL = 1e-8
IK_ORIENTATION_TOL = 1e-8
IK_MAX_ITERS = 300
IK_CONTINUATION_CHUNK = 100
IK_TOTAL_MAX_ITERS = 900
IK_TOTAL_TIME_BUDGET_S = .30
IK_CONTINUATION_POSITION_M = .001
IK_CONTINUATION_ORIENTATION_RAD = .001


class MotionRejection(ValueError):
    def __init__(self, stage, code, message, **details):
        super().__init__(message)
        self.diagnostic = {"stage": stage, "code": code, "message": message, **details}


def solve_pose(kin, target, site, seed, *, stage, path_fraction=None, orientation_required=True, joint_motion_cost=None):
    """Strict SDK solve, with bounded same-candidate continuation near a solution.

    Continuation never accepts a failed SDK flag or widens its1e-8 tolerances.
    A finite candidate within1mm/1mrad may need more differential-IK iterations;
    restart from that exact configuration with identical task/damping options.
    The installed planner still validates branch continuity, bounds, collision,
    interpolation residual and all derivative limits before any motor command.
    """
    location = {} if path_fraction is None else {"path_fraction": float(path_fraction)}
    where = "Endpoint IK" if path_fraction is None else f"Path IK at {path_fraction:.1%}"
    options = {}
    if not orientation_required:
        options["orientation_cost"] = 0.0
    if joint_motion_cost is not None:
        options["joint_motion_cost"] = joint_motion_cost
    started = time.monotonic()
    attempts, iterations = [], 0
    candidate = np.array(seed, float, copy=True)
    count = IK_MAX_ITERS
    budget_exhausted = False
    while True:
        try:
            ok, q = kin.ik(target, site, init_q=candidate.copy(),
                           pos_threshold=IK_POSITION_TOL, ori_threshold=IK_ORIENTATION_TOL,
                           max_iters=count, **options)
        except Exception as exc:
            raise MotionRejection(stage, "ik_solver_error",
                                  f"{where}: solver raised {type(exc).__name__}: {exc}",
                                  **location, exception_type=type(exc).__name__,
                                  attempt_count=len(attempts) + 1, iterations=iterations,
                                  residual_history=attempts) from exc
        iterations += count
        q = np.array(q, float, copy=True)
        if q.shape != candidate.shape or not np.isfinite(q).all():
            raise MotionRejection(stage, "ik_nonfinite", f"{where}: solver returned non-finite or malformed joint values",
                                  **location, attempt_count=len(attempts) + 1,
                                  iterations=iterations, residual_history=attempts)
        if ok:
            return q
        actual = kin.fk(q.copy(), site)
        # Match Mink FrameTask.compute_error, including SE(3) translation-log.
        error = mink.SE3.from_matrix(target).minus(mink.SE3.from_matrix(actual))
        pos, angle = float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:]))
        elapsed = time.monotonic() - started
        attempts.append(dict(attempt=len(attempts) + 1, iteration_budget=count,
                             position_error_m=pos, orientation_error_rad=angle, elapsed_s=elapsed))
        near = (np.isfinite([pos, angle]).all() and pos <= IK_CONTINUATION_POSITION_M
                and (not orientation_required or angle <= IK_CONTINUATION_ORIENTATION_RAD))
        budget_exhausted = elapsed >= IK_TOTAL_TIME_BUDGET_S
        if not near or iterations >= IK_TOTAL_MAX_ITERS or budget_exhausted:
            break
        candidate = q
        count = min(IK_CONTINUATION_CHUNK, IK_TOTAL_MAX_ITERS - iterations)
    checks = {
        "position": {"error": pos, "limit": IK_POSITION_TOL, "unit": "m", "passed": pos <= IK_POSITION_TOL},
        "orientation": {"error": angle, "limit": IK_ORIENTATION_TOL, "unit": "rad", "required": orientation_required, "passed": not orientation_required or angle <= IK_ORIENTATION_TOL},
    }
    failed = ", ".join(name for name, check in checks.items() if not check["passed"])
    raise MotionRejection(
        stage, "ik_nonconvergence",
        f"{where}: IK did not converge after {iterations} bounded iterations. "
        f"Failed tolerance: {failed or 'SDK success flag'}. "
        f"Position residual {pos * 1000:.6g} mm (limit {IK_POSITION_TOL * 1000:g} mm); "
        f"orientation {np.rad2deg(angle):.6g}° (limit {np.rad2deg(IK_ORIENTATION_TOL):.6g}°). "
        "This search failure does not prove the pose is unreachable.",
        **location, iterations=iterations, checks=checks,
        attempt_count=len(attempts), residual_history=attempts,
        solver_elapsed_s=time.monotonic() - started, continuation_budget_exhausted=budget_exhausted,
        total_iteration_budget=IK_TOTAL_MAX_ITERS, total_solver_time_budget_s=IK_TOTAL_TIME_BUDGET_S,
        cartesian_position_error_m=float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
        candidate_joint_rad=q[:6].tolist(),
    )
