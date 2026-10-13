"""Create STEP wire-edge trials in base coordinates (STEP stores millimetres). No robot I/O."""

import json
from pathlib import Path

import cadquery as cq
import numpy as np

from autoweld.app import tool_model
from autoweld.motion import MotionPlanner


def main():
    dest = Path("assets/stage3")
    dest.mkdir(parents=True, exist_ok=True)
    state = json.loads(Path("runtime/live/status.json").read_text())
    rest = np.array(state["rest_rad"])
    model = tool_model()
    try:
        planner = MotionPlanner(model)
        lines, circles, arcs, manifest = [], [], [], []
        for i, offset in enumerate(
            [
                [0.0, 0.12, 0.23, 0, 0, 0],
                [0.05, 0.14, 0.25, 0, 0, 0],
                [-0.05, 0.16, 0.28, 0, 0, 0],
                [0.03, 0.18, 0.32, 0, 0, 0],
                [-0.03, 0.20, 0.35, 0, 0, 0],
            ]
        ):
            q = np.r_[rest + offset, 0.5]
            T = planner.fk(q)
            center = T[:3, 3]
            direction = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]][i], float)
            direction /= np.linalg.norm(direction)
            a, b = center - 0.01 * direction, center + 0.01 * direction
            line = cq.Edge.makeLine(cq.Vector(*(a * 1000)), cq.Vector(*(b * 1000)))
            circle = cq.Edge.makeCircle(12.7, cq.Vector(*(center * 1000)), cq.Vector(0, 0, 1))
            arc = cq.Edge.makeCircle(12.7, cq.Vector(*(center * 1000)), cq.Vector(0, 0, 1), 0, 90)
            for name, shape in [("line", line), ("circle", circle), ("quarter-arc", arc)]:
                cq.exporters.export(shape, str(dest / f"{name}-{i + 1:02}.step"))
            lines.append(line)
            circles.append(circle)
            arcs.append(arc)
            manifest.append(
                {
                    "index": i,
                    "seed_q": q.tolist(),
                    "center_m": center.tolist(),
                    "line_direction": direction.tolist(),
                    "length_m": 0.02,
                    "radius_m": 0.0127,
                }
            )
        for name, edges in [("lines", lines), ("circles", circles), ("quarter-arcs", arcs)]:
            cq.exporters.export(cq.Compound.makeCompound(edges), str(dest / f"{name}.step"))
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    finally:
        Path(model).unlink()


if __name__ == "__main__":
    main()
