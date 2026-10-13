"""One authorized bounded line test. All robot commands use the running SDK app."""

import csv
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import tyro

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner
from autoweld.tracking_limits import POSITION_TOLERANCE_M
from scripts.tune_pid import request


def main(
    index: int = 0,
    feed: float = 0.005,
    name: str | None = None,
    suite: str = "lines",
    evidence_dir: str = "evidence/stage3-edge",
    step_path: str | None = None,
    ramp_placement: str = "Outside edge",
    seed_q: tuple[float, ...] | None = None,
    settle_hold_s: float = 0.7,
    pose_recording: str | None = None,
):
    configurations = {
        "lines": ("lines", "Unconstrained"),
        "circles": ("circles", "Unconstrained"),
        "perpendicular-lines": ("perpendicular-lines", "Perpendicular"),
        "center-arcs": ("center-arcs", "Center facing"),
    }
    if suite not in configurations:
        raise ValueError("Unknown Stage 3 suite")
    fixture, orientation = configurations[suite]
    root = Path("runtime/live")
    dest = Path(evidence_dir)
    dest.mkdir(exist_ok=True)
    telemetry = root / "control-telemetry.jsonl"
    telemetry_start = telemetry.stat().st_size if telemetry.exists() else 0
    name = name or f"{suite}-{index + 1:02}-{feed * 1000:g}mms"
    model = tool_model()
    planner = MotionPlanner(model)

    def state():
        s = json.loads((root / "status.json").read_text())
        if time.time() - s["timestamp"] > 2 or s.get("fault") or s.get("motion_fault"):
            raise RuntimeError(f"Controller not healthy: {s.get('fault') or s.get('motion_fault')}")
        if not s.get("oscillation_guard_active") or s.get("friction_enabled"):
            raise RuntimeError("Need active guard and friction disabled")
        return s

    def send(payload):
        before = time.time()
        # IK/path planning can exceed the ordinary inbox reply timeout while
        # the SDK holds the previous target. This is not a motion watchdog.
        timeout = 30. if payload['action'] in ('track_approach', 'track_run') else 5.
        request(root, payload, timeout_s=timeout)
        deadline = time.monotonic() + 3
        while True:
            s = json.loads((root / "status.json").read_text())
            if (s.get("last_request") or {}).get("timestamp", 0) >= before:
                return state()
            if time.monotonic() > deadline:
                raise RuntimeError("No post-request status")
            time.sleep(0.03)

    def settled(start_tolerance=POSITION_TOLERANCE_M * 0.8, timeout=25):
        deadline = time.monotonic() + timeout
        since = None
        while time.monotonic() < deadline:
            s = state()
            q = np.r_[s["q_rad"], 0]
            target = np.r_[s["target_rad"], 0]
            error = np.linalg.norm(planner.fk(q)[:3, 3] - planner.fk(target)[:3, 3])
            ready = (
                not s["motion_active"] and error < start_tolerance and np.max(abs(q[:6] - target[:6])) < 0.005
            )
            if ready:
                since = since or time.monotonic()
                if time.monotonic() - since > max(0.7, settle_hold_s):
                    return s
            else:
                since = None
            time.sleep(0.05)
        raise RuntimeError(f"Start did not settle: TCP error {error * 1000:.3f} mm")

    with (root / "tuning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            before = state()
            if before["closed"] or before["motion_active"]:
                raise RuntimeError("Need healthy enabled idle arm")
            seed = np.array(seed_q if seed_q is not None else
                json.loads(Path("assets/stage3/manifest.json").read_text())[index]["seed_q"][:6], dtype=float)
            if seed.shape != (6,) or not np.isfinite(seed).all():
                raise ValueError("Expected six finite seed angles")
            excursion = before.get("automatic_test_excursion_rad", .49)
            for ref in (before["q_rad"], before["target_rad"], before["rest_rad"]):
                if np.max(abs(seed - ref)) >= excursion:
                    raise ValueError(f"Standby test delta exceeds {excursion:g} rad")
            send({"action": "joint", "q": seed.tolist()})
            settled(start_tolerance=POSITION_TOLERANCE_M * 0.8)
            send({"action": "track_load", "path": step_path or f"assets/stage3/{fixture}.step"})
            payload = {"edge": index, "feed_m_s": feed, "orientation": orientation,
                       "ramp_placement": ramp_placement}
            send({"action": "track_approach", **payload})
            start = settled()
            if pose_recording is not None:
                with Path(pose_recording).open() as f:
                    poses = list(csv.DictReader(f))
                latest = int(poses[-1]['host_monotonic_ns']) / 1e9
                recent = [r for r in poses if int(r['host_monotonic_ns']) / 1e9 >= latest - 2]
                valid = np.mean([r['valid'] == '1' for r in recent])
                if time.monotonic() - latest > 2 or len(recent) < 30 or valid < .8:
                    raise RuntimeError(f'Camera coverage at circle start is insufficient ({valid:.1%}); no path run sent')
            running = send({"action": "track_run", **payload})
            deadline = time.monotonic() + max(0.1 / feed + 20, running["motion_plan"]["duration_s"]+10)
            while time.monotonic() < deadline:
                s = state()
                result = s.get("tracking")
                if result and not result.get("running"):
                    trace = Path(result["trace"])
                    (dest / f"{name}.jsonl").write_bytes(trace.read_bytes())
                    record = {
                        "suite": suite,
                        "index": index,
                        "before": before,
                        "start": start,
                        "after": s,
                        "result": result,
                    }
                    (dest / f"{name}.json").write_text(json.dumps(record, indent=2))
                    print(json.dumps({k: v for k, v in result.items() if k != "plan"}, indent=2), flush=True)
                    return
                time.sleep(0.05)
            raise RuntimeError("Tracking did not finish")
        except Exception as exc:
            (dest / f"{name}-error.json").write_text(
                json.dumps(
                    {"error": str(exc), "status": json.loads((root / "status.json").read_text())}, indent=2
                )
            )
            print(str(exc), flush=True)
            # Interrupt an in-progress test on unexpected runner errors.
            s = json.loads((root / "status.json").read_text())
            if not s.get("closed") and not s.get("fault") and s.get("motion_active"):
                request(root, {"action": "mode", "mode": "Hold"})
            raise
        finally:
            with telemetry.open("rb") as source:
                source.seek(telemetry_start)
                (dest / f"{name}-control.jsonl").write_bytes(source.read())
            Path(model).unlink()


if __name__ == "__main__":
    tyro.cli(main)
